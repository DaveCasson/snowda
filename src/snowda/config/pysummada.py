from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

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

    @property
    def model_result_output_dir(self) -> Path:
        return self.output_dir / self.model_run_id

    @property
    def ens_label_array(self) -> list[str]:
        return [f"{idx + 1:03d}" for idx in range(self.num_ens)]


def load_pysummada_config(config_path: str | Path) -> PysummadaConfig:
    cfg: Dict[str, Any] = load_config(config_path)
    base_dir = Path(cfg.get("_meta", {}).get("config_dir", Path(config_path).resolve().parent))

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
    )
