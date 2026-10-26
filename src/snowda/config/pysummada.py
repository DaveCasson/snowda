from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from snowda.config.loader import load_config
from snowda.utils.logging import get_logger

log = get_logger(__name__)


def _normalize_path(raw: str, base_dir: Path, *, allow_relative: bool) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        if not allow_relative:
            raise ValueError(f"Path must be absolute (got: {raw})")
        p = (base_dir / p).resolve()
    return p


def _require(cfg: Dict[str, Any], key: str) -> Any:
    val = cfg.get(key)
    if val in (None, ""):
        raise ValueError(f"Config must define `{key}`.")
    return val


def default_output_dir(case_name: str) -> Path:
    """``<repo>/outputs/<case_name>``; timestamped run folders are created under this."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "outputs" / case_name


@dataclass(frozen=True)
class StreamflowEvaluationConfig:
    obs_csv: Path
    variable: str = "IRFroutedRunoff"
    resample: Optional[str] = "1D"
    seg: int = 0
    reach_id: Any | None = None
    date_col: Optional[str] = None
    flow_col: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass(frozen=True)
class MizuRouteEspConfig:
    executable: Path
    control_file: Path
    ancil_dir: Path
    case_name: str
    restart_write: str
    param_nml: Path | None = None
    topology_nc: Path | None = None
    log_output: bool = False


@dataclass(frozen=True)
class DeterministicEspState:
    path: Path
    date: datetime
    member: Optional[str]


@dataclass(frozen=True)
class HruStation:
    """One SUMMA ``hruId`` paired with one SWE observation station."""

    hru_id: int
    station_id: str


@dataclass(frozen=True)
class AssimilationConfig:
    """Dated SWE assimilation schedule. Dates are explicit; frequency is optional."""

    dates: tuple[datetime, ...]
    frequency_days: Optional[int]
    months: tuple[int, ...]
    obs_window_days: int
    swe_nc: Path
    swe_var: str
    summa_swe_var: str
    z_var: float
    hru_stations: tuple[HruStation, ...]

    def station_hru_map(self) -> dict[str, int]:
        """Observation ``station_id`` to SUMMA ``hruId``."""
        return {item.station_id: item.hru_id for item in self.hru_stations}


@dataclass(frozen=True)
class EspConfig:
    enabled: bool
    deterministic_state: Optional[DeterministicEspState]
    deterministic: bool
    start_year: int
    end_year: int
    forecast_length_days: Optional[int]
    forecast_end: Optional[tuple[int, int]]
    frequency_days: Optional[int]
    dates: tuple[datetime, ...]
    months: tuple[int, ...]
    restart_file_filter: str
    state_members: tuple[str, ...]
    run_mizuroute: bool
    mizuroute: Optional[MizuRouteEspConfig]
    initialization: str = "incremental"
    n_members: Optional[int] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class PysummadaConfig:
    case_name: str
    output_dir: Path
    model_run_id: str
    base_file_manager: Path
    input_swe_hru_nc: Optional[Path]
    initial_state: Optional[str]
    forcing_file_base: str
    ensemble_forcing_mode: str
    ensemble_forcing_dir: Optional[Path]
    ensemble_members: Optional[tuple[str, ...]]
    job_list_filename: Optional[str]
    num_ens: int
    start_time: datetime
    end_time: datetime
    time_delta: timedelta
    summa_exe: Path
    summa_log: bool
    pf_scheme: str
    n_eff_threshold: float
    summa_args: Dict[str, str]
    file_manager_dir: Path
    file_list_dir: Path
    fsca_obs_nc: Optional[Path]
    fsca_obs_var: Optional[str]
    summa_output_dir: Optional[Path]
    summa_output_suffix: Optional[str]
    fsca_output_dir: Optional[Path]
    fsca_model: Optional[str]
    fsca_cv: Optional[float]
    helbig_settings: Optional[Dict[str, Any]]
    snow_depth_var: Optional[str]
    assimilation_mode: Optional[str]
    assimilation: Optional[AssimilationConfig]
    swe_weight: Optional[float]
    fsca_weight: Optional[float]
    esp: Optional[EspConfig]
    mizuroute: Optional[MizuRouteEspConfig]
    evaluation: Optional[StreamflowEvaluationConfig]
    num_workers: int = 5

    @property
    def model_result_output_dir(self) -> Path:
        return self.output_dir / self.model_run_id

    @property
    def ens_label_array(self) -> list[str]:
        if self.ensemble_members:
            return list(self.ensemble_members)
        return [f"{idx + 1:03d}" for idx in range(self.num_ens)]


def load_pysummada_config(config_path: str | Path) -> PysummadaConfig:
    cfg: Dict[str, Any] = load_config(config_path)
    base_dir = Path(cfg.get("_meta", {}).get("config_dir", Path(config_path).resolve().parent))
    _warn_ignored_legacy_keys(cfg)

    case_name = str(_require(cfg, "case_name"))
    output_dir_raw = cfg.get("output_dir")
    if output_dir_raw:
        output_dir = _normalize_path(str(output_dir_raw), base_dir, allow_relative=True)
    else:
        output_dir = default_output_dir(case_name)
        log.info("output_dir not set; defaulting to %s", output_dir)
    model_run_id = str(_require(cfg, "model_run_id"))
    base_file_manager = _normalize_path(
        str(_require(cfg, "base_file_manager")), base_dir, allow_relative=False
    )
    input_swe_hru_nc = cfg.get("input_swe_hru_nc")
    input_swe_hru_nc_path = (
        _normalize_path(str(input_swe_hru_nc), base_dir, allow_relative=False)
        if input_swe_hru_nc
        else None
    )
    initial_state = cfg.get("initial_state")
    ensemble_forcing_mode = str(cfg.get("ensemble_forcing_mode", "monthly"))
    if ensemble_forcing_mode not in {"monthly", "historic"}:
        raise ValueError(
            f"Unsupported ensemble_forcing_mode '{ensemble_forcing_mode}'. "
            "Use 'monthly' or 'historic'."
        )
    ensemble_forcing_dir_raw = cfg.get("ensemble_forcing_dir")
    if ensemble_forcing_mode == "historic":
        if not ensemble_forcing_dir_raw:
            raise ValueError(
                "`ensemble_forcing_dir` is required when `ensemble_forcing_mode` is historic."
            )
        ensemble_forcing_dir = _normalize_path(
            str(ensemble_forcing_dir_raw), base_dir, allow_relative=False
        )
        forcing_file_base = str(cfg.get("forcing_file_base") or "")
    else:
        ensemble_forcing_dir = (
            _normalize_path(str(ensemble_forcing_dir_raw), base_dir, allow_relative=False)
            if ensemble_forcing_dir_raw
            else None
        )
        forcing_file_base = str(_require(cfg, "forcing_file_base"))
    ensemble_members_raw = cfg.get("ensemble_members")
    ensemble_members = (
        tuple(str(member) for member in ensemble_members_raw)
        if ensemble_members_raw
        else None
    )
    job_list_filename = cfg.get("job_list_filename")
    num_ens = int(_require(cfg, "num_ens"))
    num_workers = int(cfg.get("num_workers", 5))
    if num_workers < 1:
        raise ValueError("`num_workers` must be >= 1.")
    if ensemble_members is not None and len(ensemble_members) != num_ens:
        raise ValueError(
            f"`ensemble_members` has {len(ensemble_members)} ids but `num_ens` is {num_ens}."
        )

    time_format = "%Y-%m-%d %H:%M"
    start_time = datetime.strptime(str(_require(cfg, "start_time")), time_format)
    end_time = datetime.strptime(str(_require(cfg, "end_time")), time_format)
    time_delta_days = int(_require(cfg, "time_delta_days"))
    time_delta = timedelta(days=time_delta_days)

    summa_exe = _normalize_path(str(_require(cfg, "summa_exe")), base_dir, allow_relative=False)
    summa_log = bool(cfg.get("summa_log", False))
    pf_scheme = _parse_pf_scheme(cfg)
    n_eff_threshold = float(cfg.get("n_eff_threshold", 0.0))

    summa_args = cfg.get("summa_args", {}) or {}
    summa_args = {str(k): str(v) for k, v in dict(summa_args).items()}

    file_manager_dir = cfg.get("file_manager_dir")
    file_list_dir = cfg.get("file_list_dir")
    if file_manager_dir:
        file_manager_dir = _normalize_path(str(file_manager_dir), base_dir, allow_relative=True)
    else:
        file_manager_dir = base_file_manager.parent
    if file_list_dir:
        file_list_dir = _normalize_path(str(file_list_dir), base_dir, allow_relative=True)
    else:
        file_list_dir = file_manager_dir

    fsca_obs_nc = cfg.get("fsca_obs_nc")
    fsca_obs_nc_path = (
        _normalize_path(str(fsca_obs_nc), base_dir, allow_relative=False)
        if fsca_obs_nc
        else None
    )
    fsca_obs_var = cfg.get("fsca_obs_var")
    summa_output_dir = cfg.get("summa_output_dir")
    summa_output_dir_path = (
        _normalize_path(str(summa_output_dir), base_dir, allow_relative=False)
        if summa_output_dir
        else None
    )
    summa_output_suffix = cfg.get("summa_output_suffix")
    fsca_output_dir = cfg.get("fsca_output_dir")
    fsca_output_dir_path = (
        _normalize_path(str(fsca_output_dir), base_dir, allow_relative=True)
        if fsca_output_dir
        else None
    )
    fsca_model = cfg.get("fsca_model")
    fsca_cv = cfg.get("fsca_cv")
    helbig_settings = cfg.get("helbig_settings")
    snow_depth_var = cfg.get("snow_depth_var")
    assimilation_mode = cfg.get("assimilation_mode")
    assimilation = _load_assimilation_config(cfg, base_dir)
    swe_weight = cfg.get("swe_weight")
    fsca_weight = cfg.get("fsca_weight")
    esp = _load_esp_config(cfg, base_dir)
    mizuroute = _resolve_mizuroute(cfg, esp, base_dir)
    evaluation = _load_evaluation_config(cfg, base_dir)

    return PysummadaConfig(
        case_name=case_name,
        output_dir=output_dir,
        model_run_id=model_run_id,
        base_file_manager=base_file_manager,
        input_swe_hru_nc=input_swe_hru_nc_path,
        initial_state=str(initial_state) if initial_state else None,
        forcing_file_base=forcing_file_base,
        ensemble_forcing_mode=ensemble_forcing_mode,
        ensemble_forcing_dir=ensemble_forcing_dir,
        ensemble_members=ensemble_members,
        job_list_filename=str(job_list_filename) if job_list_filename else None,
        num_ens=num_ens,
        start_time=start_time,
        end_time=end_time,
        time_delta=time_delta,
        summa_exe=summa_exe,
        summa_log=summa_log,
        pf_scheme=pf_scheme,
        n_eff_threshold=n_eff_threshold,
        summa_args=summa_args,
        file_manager_dir=Path(file_manager_dir),
        file_list_dir=Path(file_list_dir),
        fsca_obs_nc=fsca_obs_nc_path,
        fsca_obs_var=str(fsca_obs_var) if fsca_obs_var else None,
        summa_output_dir=Path(summa_output_dir_path) if summa_output_dir_path else None,
        summa_output_suffix=str(summa_output_suffix) if summa_output_suffix else None,
        fsca_output_dir=Path(fsca_output_dir_path) if fsca_output_dir_path else None,
        fsca_model=str(fsca_model) if fsca_model else None,
        fsca_cv=float(fsca_cv) if fsca_cv is not None else None,
        helbig_settings=dict(helbig_settings) if helbig_settings else None,
        snow_depth_var=str(snow_depth_var) if snow_depth_var else None,
        assimilation_mode=str(assimilation_mode) if assimilation_mode else None,
        assimilation=assimilation,
        swe_weight=float(swe_weight) if swe_weight is not None else None,
        fsca_weight=float(fsca_weight) if fsca_weight is not None else None,
        esp=esp,
        mizuroute=mizuroute,
        evaluation=evaluation,
        num_workers=num_workers,
    )


def validate_deterministic_config(cfg: PysummadaConfig) -> None:
    if cfg.esp and cfg.esp.enabled:
        raise ValueError(
            "Deterministic SUMMA does not run ESP traces. "
            "Use `snowda.workflows.run_summa_esp` or `snowda.workflows.run_summa_da`."
        )
    if cfg.evaluation is not None and cfg.mizuroute is None:
        raise ValueError(
            "`evaluation` requires top-level `mizuroute` so routed streamflow exists."
        )


def validate_esp_config(cfg: PysummadaConfig) -> None:
    if cfg.esp is None or not cfg.esp.enabled:
        raise ValueError(
            "ESP workflow requires an `esp` section with forecast years/dates."
        )
    if cfg.fsca_obs_nc:
        raise ValueError(
            "ESP workflow does not run data assimilation. "
            "Use `snowda.workflows.run_summa_da`."
        )
    if cfg.esp.n_members is None:
        raise ValueError(
            "`esp.n_members` is required for open-loop ESP. "
            "Each forecast uses one historic member and one weather year."
        )
    if cfg.esp.seed is None:
        raise ValueError("`esp.seed` is required for open-loop ESP.")


def validate_da_config(cfg: PysummadaConfig) -> None:
    if cfg.esp is None or not cfg.esp.enabled:
        raise ValueError(
            "SUMMA DA requires an `esp` section; DA always runs ESP after assimilation."
        )
    if cfg.esp.deterministic or cfg.esp.deterministic_state:
        raise ValueError(
            "SUMMA DA does not allow `esp.deterministic` or `esp.deterministic_state`."
        )
    if cfg.esp.initialization == "balanced_independent":
        if cfg.esp.n_members is None:
            raise ValueError(
                "`esp.n_members` is required when `esp.initialization` is "
                "balanced_independent."
            )
        if cfg.esp.seed is None:
            raise ValueError(
                "`esp.seed` is required when `esp.initialization` is "
                "balanced_independent."
            )
    _validate_assimilation_config(cfg)


def validate_historic_da_config(cfg: PysummadaConfig) -> None:
    if cfg.esp is not None:
        raise ValueError(
            "Historic SUMMA DA does not run ESP. Remove the `esp` section, "
            "or use `snowda.workflows.run_summa_da`."
        )
    if cfg.mizuroute is None:
        raise ValueError(
            "Historic SUMMA DA requires top-level `mizuroute` so each SUMMA "
            "segment is routed."
        )
    _validate_assimilation_config(cfg)


def validate_historic_config(cfg: PysummadaConfig) -> None:
    if cfg.ensemble_forcing_mode != "historic":
        raise ValueError(
            "Historic SUMMA requires `ensemble_forcing_mode: historic`. "
            "Use `snowda.workflows.run_summa` for a single forcing file."
        )
    _validate_historic_members(cfg)
    if cfg.esp is not None:
        raise ValueError(
            "Historic SUMMA does not run ESP. Remove the `esp` section, "
            "or use `snowda.workflows.run_summa_esp` or "
            "`snowda.workflows.run_summa_da`."
        )
    if cfg.assimilation is not None or cfg.assimilation_mode:
        raise ValueError(
            "Historic SUMMA does not assimilate. Remove `assimilation` and "
            "`assimilation_mode`, or use `snowda.workflows.run_summa_historic_da` "
            "or `snowda.workflows.run_summa_da`."
        )
    if cfg.evaluation is not None:
        raise ValueError(
            "Historic SUMMA does not evaluate a single routed series. "
            "Use `snowda.workflows.run_summa`."
        )


def _validate_assimilation_config(cfg: PysummadaConfig) -> None:
    assimilation_mode = cfg.assimilation_mode or "fsca_only"
    if assimilation_mode not in {"fsca_only", "swe"}:
        raise ValueError(
            f"Unsupported assimilation_mode '{assimilation_mode}'. "
            "Use 'swe' or 'fsca_only'."
        )
    if cfg.ensemble_forcing_mode == "historic":
        _validate_historic_members(cfg)
    if assimilation_mode == "fsca_only":
        missing = [
            name
            for name, value in (
                ("fsca_obs_nc", cfg.fsca_obs_nc),
                ("summa_output_dir", cfg.summa_output_dir),
                ("fsca_output_dir", cfg.fsca_output_dir),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "SUMMA assimilation in fsca_only mode requires "
                + ", ".join(f"`{name}`" for name in missing)
                + "."
            )
        return
    if cfg.assimilation is None:
        raise ValueError("SUMMA assimilation in swe mode requires an `assimilation` section.")
    from snowda.core.assimilation import resolve_assimilation_dates

    resolve_assimilation_dates(
        cfg.start_time,
        cfg.end_time,
        cfg.assimilation.dates,
        cfg.assimilation.frequency_days,
        cfg.assimilation.months,
    )


def _parse_pf_scheme(cfg: Dict[str, Any]) -> str:
    if "pf_resample" in cfg:
        raise ValueError(
            "`pf_resample` has been replaced by `pf_scheme`. Use 'sis' or 'sir'."
        )
    raw = cfg.get("pf_scheme", "sis")
    scheme = str(raw).strip().lower()
    if scheme not in {"sis", "sir"}:
        raise ValueError(f"Unsupported pf_scheme {raw!r}. Use 'sis' or 'sir'.")
    return scheme


def _warn_ignored_legacy_keys(cfg: Dict[str, Any]) -> None:
    if "run_mode" in cfg:
        log.warning(
            "`run_mode` is ignored; the workflow entrypoint selects the SUMMA variant."
        )
    if "run_da" in cfg:
        log.warning(
            "`run_da` is ignored; use run_summa_historic for a historic ensemble "
            "without assimilation, run_summa_historic_da for historic assimilation, "
            "run_summa_da for assimilation then ESP, or run_summa_esp for ESP without DA."
        )


def _parse_esp_initialization(
    esp_cfg_raw: Dict[str, Any],
) -> tuple[str, Optional[int], Optional[int]]:
    initialization = str(esp_cfg_raw.get("initialization", "incremental")).strip().lower()
    if initialization not in {"incremental", "balanced_independent"}:
        raise ValueError(
            f"Unsupported esp.initialization {esp_cfg_raw.get('initialization')!r}. "
            "Use 'incremental' or 'balanced_independent'."
        )
    n_members_raw = esp_cfg_raw.get("n_members")
    n_members = None if n_members_raw is None else int(n_members_raw)
    if n_members is not None and n_members < 1:
        raise ValueError("`esp.n_members` must be >= 1.")
    seed_raw = esp_cfg_raw.get("seed")
    seed = None if seed_raw is None else int(seed_raw)
    if seed is not None and seed < 0:
        raise ValueError("`esp.seed` must be >= 0.")
    return initialization, n_members, seed


def _esp_section_is_enabled(esp_cfg_raw: Dict[str, Any]) -> bool:
    if "enabled" in esp_cfg_raw:
        return bool(esp_cfg_raw["enabled"])
    return any(
        esp_cfg_raw.get(key) not in (None, "", [], {})
        for key in (
            "start_year",
            "end_year",
            "forecast_length_days",
            "forecast_end",
            "dates",
            "frequency_days",
        )
    )


def _parse_esp_dates(dates: Sequence[str]) -> tuple[datetime, ...]:
    out: list[datetime] = []
    for raw in dates:
        out.append(datetime.strptime(str(raw), "%Y-%m-%d %H:%M"))
    return tuple(out)


def _has_esp_horizon_value(value: Any) -> bool:
    return value not in (None, "")


def _parse_forecast_end(raw: Any) -> tuple[int, int]:
    text = str(raw).strip()
    try:
        parsed = datetime.strptime(f"2000-{text}", "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f'`esp.forecast_end` must be "MM-DD" (got: {raw!r}).') from exc
    if text != parsed.strftime("%m-%d"):
        raise ValueError(f'`esp.forecast_end` must be "MM-DD" (got: {raw!r}).')
    return parsed.month, parsed.day


def _parse_esp_horizon(
    esp_cfg_raw: Dict[str, Any],
) -> tuple[Optional[int], Optional[tuple[int, int]]]:
    has_length = _has_esp_horizon_value(esp_cfg_raw.get("forecast_length_days"))
    has_end = _has_esp_horizon_value(esp_cfg_raw.get("forecast_end"))
    if has_length and has_end:
        raise ValueError(
            "ESP requires exactly one of `esp.forecast_length_days` or `esp.forecast_end`."
        )
    if not has_length and not has_end:
        raise ValueError(
            "ESP requires either `esp.forecast_length_days` or `esp.forecast_end`."
        )
    if has_length:
        forecast_length_days = int(esp_cfg_raw["forecast_length_days"])
        if forecast_length_days <= 0:
            raise ValueError("`esp.forecast_length_days` must be > 0.")
        return forecast_length_days, None
    return None, _parse_forecast_end(esp_cfg_raw["forecast_end"])


def _load_mizuroute_from_raw(mizu_raw: Dict[str, Any], base_dir: Path) -> MizuRouteEspConfig:
    if not isinstance(mizu_raw, dict):
        raise TypeError("`mizuroute` must be a mapping.")
    executable = _normalize_path(str(_require(mizu_raw, "executable")), base_dir, allow_relative=False)
    control_file = _normalize_path(str(_require(mizu_raw, "control_file")), base_dir, allow_relative=False)
    ancil_dir_raw = mizu_raw.get("ancil_dir") or str(Path(control_file).parent)
    ancil_dir = _normalize_path(str(ancil_dir_raw), base_dir, allow_relative=False)
    if mizu_raw.get("output_dir") or mizu_raw.get("restart_dir"):
        log.warning(
            "`mizuroute.output_dir` and `mizuroute.restart_dir` are ignored; "
            "mizuRoute writes under the timestamped run directory."
        )
    case_name = str(_require(mizu_raw, "case_name"))
    restart_write = str(mizu_raw.get("restart_write", "monthly"))
    param_nml_raw = mizu_raw.get("param_nml")
    topology_nc_raw = mizu_raw.get("topology_nc")
    param_nml = (
        _normalize_path(str(param_nml_raw), base_dir, allow_relative=False)
        if param_nml_raw
        else None
    )
    topology_nc = (
        _normalize_path(str(topology_nc_raw), base_dir, allow_relative=False)
        if topology_nc_raw
        else None
    )
    log_output = bool(mizu_raw.get("log_output", False))
    return MizuRouteEspConfig(
        executable=Path(executable),
        control_file=Path(control_file),
        ancil_dir=Path(ancil_dir),
        case_name=case_name,
        restart_write=restart_write,
        param_nml=Path(param_nml) if param_nml else None,
        topology_nc=Path(topology_nc) if topology_nc else None,
        log_output=log_output,
    )


def _load_esp_mizuroute(esp_cfg_raw: Dict[str, Any], base_dir: Path) -> Optional[MizuRouteEspConfig]:
    run_mizuroute = bool(esp_cfg_raw.get("run_mizuroute", False))
    if not run_mizuroute:
        return None
    mizu_raw = esp_cfg_raw.get("mizuroute")
    if not isinstance(mizu_raw, dict):
        raise ValueError("`esp.mizuroute` must be provided when `esp.run_mizuroute` is true.")
    return _load_mizuroute_from_raw(mizu_raw, base_dir)


def _validate_historic_members(cfg: PysummadaConfig) -> None:
    if cfg.ensemble_forcing_dir is None or not cfg.ensemble_forcing_dir.is_dir():
        raise ValueError(
            f"Historic ensemble forcing directory not found: {cfg.ensemble_forcing_dir}"
        )
    missing = [
        label
        for label in cfg.ens_label_array
        if not (cfg.ensemble_forcing_dir / label).is_dir()
    ]
    if missing:
        raise ValueError(
            "Historic ensemble member directories not found: "
            + ", ".join(missing)
        )


def _load_assimilation_config(
    cfg: Dict[str, Any], base_dir: Path
) -> Optional[AssimilationConfig]:
    raw = cfg.get("assimilation")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise TypeError("`assimilation` must be a mapping.")
    explicit_dates = _parse_esp_dates([str(value) for value in (raw.get("dates") or [])])
    frequency_raw = raw.get("frequency_days")
    frequency_days = int(frequency_raw) if frequency_raw is not None else None
    if frequency_days is not None and frequency_days <= 0:
        raise ValueError("`assimilation.frequency_days` must be > 0 when provided.")
    months = tuple(int(month) for month in (raw.get("months") or []))
    for month in months:
        if month < 1 or month > 12:
            raise ValueError("`assimilation.months` entries must be between 1 and 12.")
    if "obs_window_days" not in raw:
        raise ValueError("`assimilation.obs_window_days` is required.")
    obs_window_days = int(raw["obs_window_days"])
    if obs_window_days < 0:
        raise ValueError("`assimilation.obs_window_days` must be >= 0.")
    swe_nc = _normalize_path(str(_require(raw, "swe_nc")), base_dir, allow_relative=False)
    if not swe_nc.is_file():
        raise ValueError(f"Assimilation SWE file not found: {swe_nc}")
    if "z_var" not in raw:
        raise ValueError(
            "`assimilation.z_var` is required for SWE updates. "
            "The weight is exp(-0.5 * mean(abs(error)) * z_var), not a Gaussian variance."
        )
    z_var = float(raw["z_var"])
    if z_var <= 0:
        raise ValueError("`assimilation.z_var` must be > 0.")
    hru_stations = _parse_hru_stations(raw.get("hru_stations"))
    return AssimilationConfig(
        dates=explicit_dates,
        frequency_days=frequency_days,
        months=months,
        obs_window_days=obs_window_days,
        swe_nc=swe_nc,
        swe_var=str(raw.get("swe_var", "swe_model_equivalent")),
        summa_swe_var=str(raw.get("summa_swe_var", "scalarSWE")),
        z_var=z_var,
        hru_stations=hru_stations,
    )


def _parse_hru_stations(raw: Any) -> tuple[HruStation, ...]:
    """Read the manual HRU-to-station list.

    Each entry names one SUMMA ``hruId`` and one observation ``station_id``.
    The same HRU may be listed more than once. A station may not.
    """
    if not raw:
        raise ValueError(
            "`assimilation.hru_stations` is required. List each SUMMA hruId "
            "with the SWE station_id that should update it."
        )
    if not isinstance(raw, list):
        raise TypeError("`assimilation.hru_stations` must be a list.")
    pairs: list[HruStation] = []
    seen_stations: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(
                f"`assimilation.hru_stations[{index}]` must be a mapping "
                "with `hru_id` and `station_id`."
            )
        if "hru_id" not in item or "station_id" not in item:
            raise ValueError(
                f"`assimilation.hru_stations[{index}]` requires `hru_id` and `station_id`."
            )
        station_id = str(item["station_id"]).strip()
        if not station_id:
            raise ValueError(
                f"`assimilation.hru_stations[{index}].station_id` is empty."
            )
        if station_id in seen_stations:
            raise ValueError(
                f"Station {station_id!r} is listed more than once in "
                "`assimilation.hru_stations`."
            )
        seen_stations.add(station_id)
        pairs.append(HruStation(hru_id=int(item["hru_id"]), station_id=station_id))
    return tuple(pairs)


def _load_esp_config(cfg: Dict[str, Any], base_dir: Path) -> Optional[EspConfig]:
    esp_cfg_raw = cfg.get("esp")
    if not esp_cfg_raw:
        return None
    if not isinstance(esp_cfg_raw, dict):
        raise TypeError("`esp` must be a mapping.")

    enabled = _esp_section_is_enabled(esp_cfg_raw)
    if not enabled:
        return EspConfig(
            enabled=False,
            deterministic_state=None,
            deterministic=False,
            start_year=0,
            end_year=0,
            forecast_length_days=None,
            forecast_end=None,
            frequency_days=None,
            dates=tuple(),
            months=tuple(),
            restart_file_filter="restart",
            state_members=tuple(),
            run_mizuroute=bool(esp_cfg_raw.get("run_mizuroute", False)),
            mizuroute=_load_esp_mizuroute(esp_cfg_raw, base_dir),
        )

    start_year = int(_require(esp_cfg_raw, "start_year"))
    end_year = int(_require(esp_cfg_raw, "end_year"))
    if end_year < start_year:
        raise ValueError("`esp.end_year` must be >= `esp.start_year`.")

    forecast_length_days, forecast_end = _parse_esp_horizon(esp_cfg_raw)

    frequency_days_raw = esp_cfg_raw.get("frequency_days")
    frequency_days = int(frequency_days_raw) if frequency_days_raw is not None else None
    if frequency_days is not None and frequency_days <= 0:
        raise ValueError("`esp.frequency_days` must be > 0 when provided.")

    explicit_dates_raw = esp_cfg_raw.get("dates", []) or []
    explicit_dates = _parse_esp_dates([str(v) for v in explicit_dates_raw])

    months_raw = esp_cfg_raw.get("months", []) or []
    months = tuple(int(m) for m in months_raw)
    for month in months:
        if month < 1 or month > 12:
            raise ValueError("`esp.months` entries must be between 1 and 12.")

    if frequency_days is None and not explicit_dates:
        raise ValueError("ESP requires either `esp.frequency_days` or `esp.dates`.")

    restart_file_filter = str(esp_cfg_raw.get("restart_file_filter", "restart"))
    state_members_raw = esp_cfg_raw.get("state_members", []) or []
    state_members = tuple(str(m) for m in state_members_raw)
    initialization, n_members, seed = _parse_esp_initialization(esp_cfg_raw)
    deterministic = bool(esp_cfg_raw.get("deterministic", False))

    deterministic_state = None
    deterministic_state_raw = esp_cfg_raw.get("deterministic_state")
    if deterministic_state_raw:
        if not isinstance(deterministic_state_raw, dict):
            raise TypeError("`esp.deterministic_state` must be a mapping.")
        state_path = _normalize_path(
            str(_require(deterministic_state_raw, "path")), base_dir, allow_relative=False
        )
        state_date = datetime.strptime(
            str(_require(deterministic_state_raw, "date")), "%Y-%m-%d %H:%M"
        )
        state_member = deterministic_state_raw.get("member")
        deterministic_state = DeterministicEspState(
            path=Path(state_path),
            date=state_date,
            member=str(state_member) if state_member else None,
        )

    return EspConfig(
        enabled=True,
        deterministic_state=deterministic_state,
        deterministic=deterministic,
        start_year=start_year,
        end_year=end_year,
        forecast_length_days=forecast_length_days,
        forecast_end=forecast_end,
        frequency_days=frequency_days,
        dates=explicit_dates,
        months=months,
        restart_file_filter=restart_file_filter,
        state_members=state_members,
        run_mizuroute=bool(esp_cfg_raw.get("run_mizuroute", False)),
        mizuroute=_load_esp_mizuroute(esp_cfg_raw, base_dir),
        initialization=initialization,
        n_members=n_members,
        seed=seed,
    )


def _resolve_mizuroute(
    cfg: Dict[str, Any], esp: Optional[EspConfig], base_dir: Path
) -> Optional[MizuRouteEspConfig]:
    mizu_raw = cfg.get("mizuroute")
    if mizu_raw:
        return _load_mizuroute_from_raw(mizu_raw, base_dir)
    if esp is not None and not esp.enabled and esp.mizuroute:
        return esp.mizuroute
    return None


def _load_evaluation_config(
    cfg: Dict[str, Any], base_dir: Path
) -> Optional[StreamflowEvaluationConfig]:
    raw = cfg.get("evaluation")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise TypeError("`evaluation` must be a mapping.")
    obs_csv = _normalize_path(str(_require(raw, "obs_csv")), base_dir, allow_relative=False)
    resample = raw.get("resample", "1D")
    if resample in (None, "null", ""):
        resample = None
    start_time = raw.get("start_time")
    end_time = raw.get("end_time")
    date_col = raw.get("date_col") or raw.get("obs_date_col")
    flow_col = raw.get("flow_col") or raw.get("obs_flow_col")
    reach_id = raw.get("reach_id")
    return StreamflowEvaluationConfig(
        obs_csv=Path(obs_csv),
        variable=str(raw.get("variable", "IRFroutedRunoff")),
        resample=None if resample is None else str(resample),
        seg=int(raw.get("seg", 0)),
        reach_id=reach_id,
        date_col=str(date_col) if date_col else None,
        flow_col=str(flow_col) if flow_col else None,
        start_time=str(start_time) if start_time else None,
        end_time=str(end_time) if end_time else None,
    )
