from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from snowda.adapters.mizuroute import run_mizuroute, update_control_file
from snowda.adapters.pysumma import get_file_manager_value, run_simulation
from snowda.config.pysummada import EspConfig, PysummadaConfig
from snowda.utils.logging import get_logger

log = get_logger(__name__)

SUMMA_TIME_FORMAT = "%Y-%m-%d %H:%M"
_RESTART_DATE_RE = re.compile(r"_(\d{10})")


@dataclass(frozen=True)
class EspRunRecord:
    state_member: str
    state_date: str
    trace_year: int
    summa_output: str
    mizuroute_output: str | None


def _replace_year(dt: datetime, year: int) -> datetime:
    try:
        return dt.replace(year=year)
    except ValueError:
        # Leap-day rollover for non-leap years.
        max_day = pd.Timestamp(f"{year}-{dt.month:02d}-01").days_in_month
        return dt.replace(year=year, day=min(dt.day, max_day))


def _replace_year_timestamp(ts: pd.Timestamp, year: int) -> pd.Timestamp:
    try:
        return ts.replace(year=year)
    except ValueError:
        max_day = pd.Timestamp(f"{year}-{ts.month:02d}-01").days_in_month
        return ts.replace(year=year, day=min(ts.day, max_day))


def _to_pandas_timestamp(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if hasattr(value, "strftime"):
        return pd.Timestamp(value.strftime("%Y-%m-%d %H:%M:%S"))
    return pd.Timestamp(str(value))


def _apply_month_filter(dates: Iterable[datetime], months: Sequence[int] | None) -> list[datetime]:
    if not months:
        return sorted(set(dates))
    month_set = set(months)
    return sorted({dt for dt in dates if dt.month in month_set})


def resolve_esp_trigger_dates(
    start_time: datetime,
    end_time: datetime,
    explicit_dates: Sequence[datetime],
    frequency_days: int | None,
    months: Sequence[int] | None = None,
) -> list[datetime]:
    """
    Resolve ESP trigger dates from explicit dates and/or a fixed day frequency.
    """
    dates: set[datetime] = set(explicit_dates)
    if frequency_days:
        current = start_time
        while current <= end_time:
            dates.add(current)
            current += timedelta(days=frequency_days)
    return _apply_month_filter(dates, months)


def discover_restart_dates(
    state_dir: str | Path,
    restart_file_filter: str,
    state_member: str,
) -> dict[str, Path]:
    """
    Discover restart files for a DA member and group by `YYYYMMDDHH`.
    """
    root = Path(state_dir)
    if not root.exists():
        raise FileNotFoundError(f"SUMMA statePath does not exist: {root}")

    by_date: dict[str, Path] = {}
    for candidate in sorted(root.glob("*.nc")):
        name = candidate.name
        if restart_file_filter not in name:
            continue
        if state_member and not name.startswith(f"{state_member}_"):
            continue
        match = _RESTART_DATE_RE.search(name)
        if not match:
            continue
        by_date[match.group(1)] = candidate
    return by_date


def _find_summa_timestep_output(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_timestep.nc"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No SUMMA timestep output found in {output_dir}")
    return candidates[-1]


def _shift_time_to_state_year(input_file: Path, state_year: int, output_file: Path) -> Path:
    with xr.open_dataset(input_file) as ds:
        ds = ds.load()
        if "time" not in ds.variables:
            raise ValueError(f"SUMMA output missing time variable: {input_file}")
        original = pd.to_datetime(ds["time"].values)
        shifted = pd.DatetimeIndex(
            [_replace_year_timestamp(_to_pandas_timestamp(ts), state_year) for ts in original]
        )
        if shifted.has_duplicates:
            _, unique_positions = np.unique(shifted.values, return_index=True)
            unique_positions = np.sort(unique_positions)
            ds = ds.isel(time=unique_positions)
            shifted = shifted[unique_positions]
        ds["time"] = shifted
        ds["time"].encoding = {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
            "dtype": "float64",
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_file)
    return output_file


def _find_mizuroute_output(output_dir: Path, case_name: str, sim_start: datetime) -> Path:
    # mizuRoute output naming varies across setups; prefer canonical path, then fallback.
    canonical = output_dir / f"{case_name}.h.{sim_start.strftime('%Y-%m-%d')}-00000.nc"
    if canonical.exists():
        return canonical
    globbed = sorted(output_dir.glob(f"{case_name}.h.*.nc"), key=lambda p: p.stat().st_mtime)
    if globbed:
        return globbed[-1]
    raise FileNotFoundError(f"Could not find mizuRoute output in {output_dir}")


def _trace_output_dir(run_dir: Path, member: str, state_date_key: str, trace_year: int) -> Path:
    return run_dir / "esp" / member / state_date_key / str(trace_year)


def _locate_resource(
    explicit: Path | None,
    fallback_dirs: Sequence[Path],
    filename: str,
) -> Path | None:
    if explicit:
        return explicit
    for folder in fallback_dirs:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None


def _copy_optional_resource(
    source: Path | None,
    fallback_dirs: Sequence[Path],
    destination: Path,
    label: str,
    required: bool = False,
) -> None:
    resource = _locate_resource(source, fallback_dirs, destination.name)
    if not resource:
        if required:
            raise FileNotFoundError(
                f"Could not locate required {label} (looked in {fallback_dirs})."
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resource, destination)


def _strip_quotes(value: str) -> str:
    return str(value).strip().strip("'\"")


def _ensure_summa_restart(
    cfg: PysummadaConfig,
    run_dir: Path,
    member: str,
    trigger: datetime,
    *,
    state_path: Path,
    restart_file_filter: str,
) -> Path:
    trigger_key = trigger.strftime("%Y%m%d%H")
    restarts = discover_restart_dates(state_path, restart_file_filter, member)
    existing = restarts.get(trigger_key)
    if existing:
        return existing
    log.info("SUMMA restart missing for %s; running pre-ESP SUMMA to %s.", member, trigger)
    init_cond = Path(
        _strip_quotes(get_file_manager_value(str(cfg.base_file_manager), "initConditionFile"))
    )
    pre_output = run_dir / "esp_pre" / "summa" / member / trigger.strftime("%Y%m%d%H")
    pre_output.mkdir(parents=True, exist_ok=True)
    sim_config = {
        "manager": {
            "simStartTime": cfg.start_time.strftime(SUMMA_TIME_FORMAT),
            "simEndTime": trigger.strftime(SUMMA_TIME_FORMAT),
            "statePath": str(state_path) + "/",
            "initConditionFile": init_cond.name,
            "outputPath": str(pre_output) + "/",
        }
    }
    run_simulation(
        str(cfg.summa_exe),
        str(cfg.base_file_manager),
        run_suffix=f"preesp_{member}_{trigger.strftime('%Y%m%d%H')}",
        write_config=True,
        update_manager=False,
        log_stdout=cfg.summa_log,
        sim_config=sim_config,
    )
    restarts = discover_restart_dates(state_path, restart_file_filter, member)
    restart_path = restarts.get(trigger_key)
    if not restart_path:
        raise FileNotFoundError(
            f"Expected SUMMA restart for {trigger_key} after pre-ESP run."
        )
    return restart_path


def _mizuroute_restart_name(case_name: str, trigger: datetime) -> str:
    return f"{case_name}.r.{trigger.strftime('%Y-%m-%d')}-00000.nc"


def _ensure_mizuroute_restart(
    cfg: PysummadaConfig,
    mizu,
    run_dir: Path,
    trigger: datetime,
    *,
    summa_output: Path,
) -> Path:
    restart_path = Path(mizu.restart_dir) / _mizuroute_restart_name(mizu.case_name, trigger)
    if restart_path.exists():
        return restart_path
    log.info("mizuRoute restart missing for %s; running pre-ESP mizuRoute.", trigger)
    control_parent = Path(mizu.control_file).parent
    pre_output = run_dir / "esp_pre" / "mizuroute" / trigger.strftime("%Y%m%d%H")
    pre_output.mkdir(parents=True, exist_ok=True)
    _copy_optional_resource(
        mizu.param_nml,
        [mizu.ancil_dir, control_parent],
        pre_output / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    _copy_optional_resource(
        mizu.topology_nc,
        [mizu.ancil_dir, control_parent],
        pre_output / "topology.nc",
        "mizuRoute topology.nc",
    )
    control_file = pre_output / "mizuroute_pre_esp.control"
    control_settings = {
        "fname_qsim": summa_output.name,
        "case_name": mizu.case_name,
        "sim_start": cfg.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_end": (trigger - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "ancil_dir": str(mizu.ancil_dir) + "/",
        "input_dir": str(summa_output.parent) + "/",
        "output_dir": str(pre_output) + "/",
        "restart_dir": str(mizu.restart_dir) + "/",
        "restart_write": mizu.restart_write,
    }
    update_control_file(mizu.control_file, control_file, control_settings)
    run_mizuroute(mizu.executable, control_file, log_stdout=mizu.log_output)
    if not restart_path.exists():
        raise FileNotFoundError(
            f"Expected mizuRoute restart at {restart_path} after pre-ESP run."
        )
    return restart_path


def run_summa_esp(cfg: PysummadaConfig, esp: EspConfig, run_dir: Path) -> list[EspRunRecord]:
    state_path = Path(
        get_file_manager_value(str(cfg.base_file_manager), "statePath").replace("'", "")
    )
    trigger_dates = resolve_esp_trigger_dates(
        cfg.start_time,
        cfg.end_time,
        list(esp.dates),
        esp.frequency_days,
        esp.months,
    )

    def _resolve_member() -> str:
        if esp.state_members:
            return esp.state_members[0]
        return cfg.ens_label_array[0]

    state_entries: list[tuple[str, str, Path, datetime]] = []

    if esp.deterministic_state:
        member = esp.deterministic_state.member or _resolve_member()
        state_dt = esp.deterministic_state.date
        state_entries.append(
            (
                member,
                state_dt.strftime("%Y%m%d%H"),
                esp.deterministic_state.path,
                state_dt,
            )
        )
    elif esp.deterministic:
        member = _resolve_member()
        if not trigger_dates:
            log.warning("ESP deterministic run requires trigger dates.")
        summa_output_dir = Path(
            get_file_manager_value(str(cfg.base_file_manager), "outputPath").replace("'", "")
        )
        summa_timestep = _find_summa_timestep_output(summa_output_dir)
        for trigger in trigger_dates:
            state_file = _ensure_summa_restart(
                cfg,
                run_dir,
                member,
                trigger,
                state_path=state_path,
                restart_file_filter=esp.restart_file_filter,
            )
            if esp.run_mizuroute and esp.mizuroute:
                _ensure_mizuroute_restart(
                    cfg, esp.mizuroute, run_dir, trigger, summa_output=summa_timestep
                )
            state_entries.append(
                (
                    member,
                    trigger.strftime("%Y%m%d%H"),
                    state_file,
                    trigger,
                )
            )
    else:
        state_members = list(esp.state_members) if esp.state_members else [cfg.ens_label_array[0]]
        for member in state_members:
            restarts = discover_restart_dates(
                state_path, esp.restart_file_filter, member
            )
            if not restarts:
                log.warning(
                    "No restart files found for ESP member %s in %s", member, state_path
                )
                continue
            for state_date_key, state_file in restarts.items():
                state_dt = pd.to_datetime(state_date_key, format="%Y%m%d%H")
                state_entries.append((member, state_date_key, state_file, state_dt))

    if not state_entries:
        log.warning("No ESP state entries available.")
        return []

    if not trigger_dates and esp.deterministic:
        trigger_dates = list({entry[3] for entry in state_entries})
    if not trigger_dates:
        log.warning("ESP enabled but no trigger dates resolved.")
        return []

    records: list[EspRunRecord] = []
    for member, state_date_key, state_file, _ in state_entries:
        for trigger in trigger_dates:
            if esp.deterministic and trigger.strftime("%Y%m%d%H") != state_date_key:
                continue
            if not esp.deterministic and trigger.strftime("%Y%m%d%H") != state_date_key:
                continue

            for trace_year in range(esp.start_year, esp.end_year + 1):
                trace_start = _replace_year(trigger, trace_year)
                trace_end = trace_start + timedelta(days=esp.forecast_length_days)
                base_start = trigger
                base_end = base_start + timedelta(days=esp.forecast_length_days)
                trace_output_dir = _trace_output_dir(run_dir, member, state_date_key, trace_year)
                trace_output_dir.mkdir(parents=True, exist_ok=True)

                run_suffix = f"esp_state{state_date_key}_year{trace_year}"
                state_path_for_sim = state_file.parent if esp.deterministic_state else state_path
                sim_config = {
                    "manager": {
                        "simStartTime": trace_start.strftime(SUMMA_TIME_FORMAT),
                        "simEndTime": trace_end.strftime(SUMMA_TIME_FORMAT),
                        "statePath": str(state_path_for_sim) + "/",
                        "initConditionFile": state_file.name,
                        "outputPath": str(trace_output_dir) + "/",
                    }
                }
                run_simulation(
                    str(cfg.summa_exe),
                    str(cfg.base_file_manager),
                    run_suffix=run_suffix,
                    write_config=True,
                    update_manager=False,
                    log_stdout=cfg.summa_log,
                    sim_config=sim_config,
                )

                raw_summa_output = _find_summa_timestep_output(trace_output_dir)
                mizu_output_dir = trace_output_dir / "mizuroute"
                mizu_output_dir.mkdir(parents=True, exist_ok=True)
                shifted_summa_output = mizu_output_dir / f"summa_{trace_year}.nc"
                _shift_time_to_state_year(raw_summa_output, trigger.year, shifted_summa_output)

                mizuroute_output: Path | None = None
                if esp.run_mizuroute and esp.mizuroute:
                    mizu = esp.mizuroute
                    control_parent = Path(mizu.control_file).parent
                    _copy_optional_resource(
                        mizu.param_nml,
                        [mizu.ancil_dir, control_parent],
                        mizu_output_dir / "param.nml.default",
                        "mizuRoute param.nml.default",
                        required=True,
                    )
                    _copy_optional_resource(
                        mizu.topology_nc,
                        [mizu.ancil_dir, control_parent],
                        mizu_output_dir / "topology.nc",
                        "mizuRoute topology.nc",
                    )
                    control_file = mizu_output_dir / f"mizuroute_{trace_year}.control"
                    control_settings = {
                        "fname_qsim": shifted_summa_output.name,
                        "fname_state_in": f"{mizu.case_name}.r.{base_start.strftime('%Y-%m-%d')}-00000.nc",
                        "case_name": mizu.case_name,
                        "sim_start": base_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "sim_end": (base_end - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                        "ancil_dir": str(mizu.ancil_dir) + "/",
                        "input_dir": str(shifted_summa_output.parent) + "/",
                        "output_dir": str(mizu_output_dir) + "/",
                        "restart_dir": str(mizu.restart_dir) + "/",
                        "restart_write": mizu.restart_write,
                    }
                    update_control_file(mizu.control_file, control_file, control_settings)
                    run_mizuroute(mizu.executable, control_file, log_stdout=mizu.log_output)
                    routed_generated = _find_mizuroute_output(mizu_output_dir, mizu.case_name, trace_start)
                    mizuroute_output = mizu_output_dir / f"mizuroute_{trace_year}.nc"
                    routed_generated.rename(mizuroute_output)

                records.append(
                    EspRunRecord(
                        state_member=member,
                        state_date=state_date_key,
                        trace_year=trace_year,
                        summa_output=str(shifted_summa_output),
                        mizuroute_output=str(mizuroute_output) if mizuroute_output else None,
                    )
                )
    return records
