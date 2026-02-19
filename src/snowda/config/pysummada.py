from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from snowda.config.loader import load_config


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


@dataclass(frozen=True)
class MizuRouteEspConfig:
    executable: Path
    control_file: Path
    ancil_dir: Path
    output_dir: Path
    restart_dir: Path
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
class EspConfig:
    enabled: bool
    deterministic_state: Optional[DeterministicEspState]
    deterministic: bool
    start_year: int
    end_year: int
    forecast_length_days: int
    frequency_days: Optional[int]
    dates: tuple[datetime, ...]
    months: tuple[int, ...]
    restart_file_filter: str
    state_members: tuple[str, ...]
    run_mizuroute: bool
    mizuroute: Optional[MizuRouteEspConfig]


@dataclass(frozen=True)
class PysummadaConfig:
    case_name: str
    output_dir: Path
    model_run_id: str
    base_file_manager: Path
    input_swe_hru_nc: Optional[Path]
    initial_state: Optional[str]
    forcing_file_base: str
    job_list_filename: Optional[str]
    num_ens: int
    start_time: datetime
    end_time: datetime
    time_delta: timedelta
    summa_exe: Path
    summa_log: bool
    pf_resample: bool
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
    swe_weight: Optional[float]
    fsca_weight: Optional[float]
    esp: Optional[EspConfig]
    run_mode: str
    run_da: bool

    @property
    def model_result_output_dir(self) -> Path:
        return self.output_dir / self.model_run_id

    @property
    def ens_label_array(self) -> list[str]:
        return [f"{idx + 1:03d}" for idx in range(self.num_ens)]


def load_pysummada_config(config_path: str | Path) -> PysummadaConfig:
    cfg: Dict[str, Any] = load_config(config_path)
    base_dir = Path(cfg.get("_meta", {}).get("config_dir", Path(config_path).resolve().parent))

    run_mode, explicit_run_mode = _resolve_run_mode(cfg)
    esp_cfg_raw = cfg.get("esp")
    run_da = bool(cfg.get("run_da", True))
    esp_cfg_raw = _apply_run_mode_constraints(
        run_mode, esp_cfg_raw, explicit_run_mode=explicit_run_mode
    )
    run_da = _run_da_for_mode(run_mode) if explicit_run_mode else run_da
    cfg["run_da"] = run_da
    cfg["esp"] = esp_cfg_raw

    case_name = str(_require(cfg, "case_name"))
    output_dir = _normalize_path(str(_require(cfg, "output_dir")), base_dir, allow_relative=True)
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
    forcing_file_base = str(_require(cfg, "forcing_file_base"))
    job_list_filename = cfg.get("job_list_filename")
    num_ens = int(_require(cfg, "num_ens"))

    time_format = "%Y-%m-%d %H:%M"
    start_time = datetime.strptime(str(_require(cfg, "start_time")), time_format)
    end_time = datetime.strptime(str(_require(cfg, "end_time")), time_format)
    time_delta_days = int(_require(cfg, "time_delta_days"))
    time_delta = timedelta(days=time_delta_days)

    summa_exe = _normalize_path(str(_require(cfg, "summa_exe")), base_dir, allow_relative=False)
    summa_log = bool(cfg.get("summa_log", False))
    pf_resample = bool(cfg.get("pf_resample", False))
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
    swe_weight = cfg.get("swe_weight")
    fsca_weight = cfg.get("fsca_weight")
    esp = _load_esp_config(cfg, base_dir)

    return PysummadaConfig(
        case_name=case_name,
        output_dir=output_dir,
        model_run_id=model_run_id,
        base_file_manager=base_file_manager,
        input_swe_hru_nc=input_swe_hru_nc_path,
        initial_state=str(initial_state) if initial_state else None,
        forcing_file_base=forcing_file_base,
        job_list_filename=str(job_list_filename) if job_list_filename else None,
        num_ens=num_ens,
        start_time=start_time,
        end_time=end_time,
        time_delta=time_delta,
        summa_exe=summa_exe,
        summa_log=summa_log,
        pf_resample=pf_resample,
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
        swe_weight=float(swe_weight) if swe_weight is not None else None,
        fsca_weight=float(fsca_weight) if fsca_weight is not None else None,
        run_mode=run_mode,
        run_da=bool(run_da),
        esp=esp,
    )


def _run_da_for_mode(run_mode: str) -> bool:
    return run_mode in {"da", "da_esp"}


def _resolve_run_mode(cfg: Dict[str, Any]) -> tuple[str, bool]:
    allowed = {"deterministic", "deterministic_esp", "da", "da_esp"}
    raw = cfg.get("run_mode")
    if raw is None:
        return _infer_run_mode(cfg), False
    run_mode = str(raw).strip()
    if run_mode not in allowed:
        raise ValueError(
            f"`run_mode` must be one of {sorted(allowed)} (got: {run_mode})."
        )
    return run_mode, True


def _infer_run_mode(cfg: Dict[str, Any]) -> str:
    esp_cfg = cfg.get("esp")
    esp_enabled = isinstance(esp_cfg, dict) and bool(esp_cfg.get("enabled", False))
    deterministic_esp = (
        isinstance(esp_cfg, dict)
        and bool(esp_cfg.get("deterministic") or esp_cfg.get("deterministic_state"))
    )
    run_da = bool(cfg.get("run_da", True))
    if deterministic_esp:
        return "deterministic_esp"
    if run_da and esp_enabled:
        return "da_esp"
    if run_da:
        return "da"
    if esp_enabled:
        return "deterministic_esp"
    return "deterministic"


def _apply_run_mode_constraints(
    run_mode: str, esp_cfg_raw: Any, *, explicit_run_mode: bool
) -> Any:
    if not explicit_run_mode:
        return esp_cfg_raw

    if run_mode in {"deterministic", "da"}:
        if isinstance(esp_cfg_raw, dict):
            if esp_cfg_raw.get("enabled", False):
                raise ValueError(f"`run_mode: {run_mode}` does not allow ESP.")
            if esp_cfg_raw.get("deterministic") or esp_cfg_raw.get("deterministic_state"):
                raise ValueError(
                    f"`run_mode: {run_mode}` does not allow deterministic ESP settings."
                )
        return esp_cfg_raw

    if not isinstance(esp_cfg_raw, dict):
        raise ValueError(f"`run_mode: {run_mode}` requires an `esp` section.")

    esp_cfg = dict(esp_cfg_raw)
    esp_cfg["enabled"] = True
    if run_mode == "deterministic_esp":
        if not (esp_cfg.get("deterministic") or esp_cfg.get("deterministic_state")):
            raise ValueError(
                "`run_mode: deterministic_esp` requires `esp.deterministic` or "
                "`esp.deterministic_state`."
            )
    if run_mode == "da_esp":
        if esp_cfg.get("deterministic") or esp_cfg.get("deterministic_state"):
            raise ValueError(
                "`run_mode: da_esp` does not allow deterministic ESP state settings."
            )
    return esp_cfg


def _parse_esp_dates(dates: Sequence[str]) -> tuple[datetime, ...]:
    out: list[datetime] = []
    for raw in dates:
        out.append(datetime.strptime(str(raw), "%Y-%m-%d %H:%M"))
    return tuple(out)


def _load_esp_config(cfg: Dict[str, Any], base_dir: Path) -> Optional[EspConfig]:
    esp_cfg_raw = cfg.get("esp")
    if not esp_cfg_raw:
        return None
    if not isinstance(esp_cfg_raw, dict):
        raise TypeError("`esp` must be a mapping.")

    enabled = bool(esp_cfg_raw.get("enabled", False))
    if not enabled:
        run_mizuroute = bool(esp_cfg_raw.get("run_mizuroute", False))
        mizuroute_cfg = (
            _load_mizuroute_config(esp_cfg_raw, base_dir) if run_mizuroute else None
        )
        return EspConfig(
            enabled=False,
            deterministic_state=None,
            deterministic=False,
            start_year=0,
            end_year=0,
            forecast_length_days=0,
            frequency_days=None,
            dates=tuple(),
            months=tuple(),
            restart_file_filter="restart",
            state_members=tuple(),
            run_mizuroute=run_mizuroute,
            mizuroute=mizuroute_cfg,
        )

    start_year = int(_require(esp_cfg_raw, "start_year"))
    end_year = int(_require(esp_cfg_raw, "end_year"))
    if end_year < start_year:
        raise ValueError("`esp.end_year` must be >= `esp.start_year`.")

    forecast_length_days = int(_require(esp_cfg_raw, "forecast_length_days"))
    if forecast_length_days <= 0:
        raise ValueError("`esp.forecast_length_days` must be > 0.")

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

    run_mizuroute = bool(esp_cfg_raw.get("run_mizuroute", False))
    mizuroute_cfg = (
        _load_mizuroute_config(esp_cfg_raw, base_dir) if run_mizuroute else None
    )

    return EspConfig(
        enabled=True,
        deterministic_state=deterministic_state,
        deterministic=deterministic,
        start_year=start_year,
        end_year=end_year,
        forecast_length_days=forecast_length_days,
        frequency_days=frequency_days,
        dates=explicit_dates,
        months=months,
        restart_file_filter=restart_file_filter,
        state_members=state_members,
        run_mizuroute=run_mizuroute,
        mizuroute=mizuroute_cfg,
    )


def _load_mizuroute_config(esp_cfg_raw: Dict[str, Any], base_dir: Path) -> MizuRouteEspConfig:
    mizu_raw = esp_cfg_raw.get("mizuroute")
    if not isinstance(mizu_raw, dict):
        raise ValueError("`esp.mizuroute` must be provided when `esp.run_mizuroute` is true.")
    executable = _normalize_path(str(_require(mizu_raw, "executable")), base_dir, allow_relative=False)
    control_file = _normalize_path(str(_require(mizu_raw, "control_file")), base_dir, allow_relative=False)
    ancil_dir_raw = mizu_raw.get("ancil_dir") or str(Path(control_file).parent)
    ancil_dir = _normalize_path(str(ancil_dir_raw), base_dir, allow_relative=False)
    output_dir = _normalize_path(str(_require(mizu_raw, "output_dir")), base_dir, allow_relative=True)
    restart_dir = _normalize_path(str(_require(mizu_raw, "restart_dir")), base_dir, allow_relative=True)
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
        output_dir=Path(output_dir),
        restart_dir=Path(restart_dir),
        case_name=case_name,
        restart_write=restart_write,
        param_nml=Path(param_nml) if param_nml else None,
        topology_nc=Path(topology_nc) if topology_nc else None,
        log_output=log_output,
    )
