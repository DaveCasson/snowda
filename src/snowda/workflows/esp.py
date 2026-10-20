from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import calendar
import numpy as np
import pandas as pd
import xarray as xr

from snowda.adapters.mizuroute import run_mizuroute, update_control_file
from snowda.adapters.pysumma import get_file_manager_value, run_simulation
from snowda.config.pysummada import EspConfig, PysummadaConfig
from snowda.utils.logging import get_logger
from snowda.workflows.summa_common import (
    copy_optional_resource,
    find_mizuroute_output,
    find_summa_timestep_output,
)

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


def forecast_end_datetime(start: datetime, month: int, day: int) -> datetime:
    """Next occurrence of ``MM-DD`` strictly after ``start``, keeping clock time."""
    leap_anchor = datetime(
        2000,
        month,
        day,
        start.hour,
        start.minute,
        start.second,
        start.microsecond,
    )
    candidate = _replace_year(leap_anchor, start.year)
    if candidate <= start:
        candidate = _replace_year(leap_anchor, start.year + 1)
    return candidate


def resolve_forecast_window_end(
    start: datetime,
    forecast_length_days: int | None,
    forecast_end: tuple[int, int] | None,
) -> datetime:
    """Resolve the ESP trace end from a duration or a calendar ``MM-DD``."""
    if forecast_end is not None:
        return forecast_end_datetime(start, forecast_end[0], forecast_end[1])
    if forecast_length_days is None:
        raise ValueError(
            "ESP config is missing both `forecast_end` and `forecast_length_days`."
        )
    return start + timedelta(days=forecast_length_days)


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


def _trace_output_dir(run_dir: Path, member: str, state_date_key: str, trace_year: int) -> Path:
    return run_dir / "esp" / member / state_date_key / str(trace_year)


def _member_label(member: str | None) -> str:
    return member if member else "default"


def _esp_state_root(run_dir: Path) -> Path:
    return run_dir / "esp_states"


def _esp_summa_state_dir(run_dir: Path, member: str | None) -> Path:
    return _esp_state_root(run_dir) / "summa" / _member_label(member)


def _esp_mizuroute_state_dir(run_dir: Path, member: str | None) -> Path:
    return _esp_state_root(run_dir) / "mizuroute" / _member_label(member)


def _esp_interval_dir(run_dir: Path, member: str | None, trigger: datetime) -> Path:
    return run_dir / "esp_intervals" / _member_label(member) / trigger.strftime("%Y%m%d%H")


def _strip_quotes(value: str) -> str:
    return str(value).strip().strip("'\"")


def _read_control_value(control_file: Path, key: str) -> str | None:
    token = f"<{key}>"
    try:
        lines = control_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(token):
            continue
        tail = stripped[len(token) :].strip()
        if not tail:
            return None
        return tail.split()[0].strip()
    return None


def _seed_state_file(destination_dir: Path, source: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest = destination_dir / source.name
    if not dest.exists():
        if not source.exists():
            raise FileNotFoundError(f"Missing SUMMA init state file: {source}")
        shutil.copy2(source, dest)
    return dest


def _find_restart_for_trigger(
    state_dir: Path,
    restart_file_filter: str,
    trigger_key: str,
    member: str | None,
) -> Path:
    matches: list[tuple[int, Path]] = []
    for candidate in sorted(state_dir.glob("*.nc")):
        name = candidate.name
        if restart_file_filter not in name:
            continue
        match = _RESTART_DATE_RE.search(name)
        if not match or match.group(1) != trigger_key:
            continue
        if member and name.startswith(f"{member}_"):
            matches.append((1, candidate))
        else:
            matches.append((0, candidate))
    if not matches:
        raise FileNotFoundError(
            f"Expected SUMMA restart for {trigger_key} in {state_dir} (filter={restart_file_filter})."
        )
    matches.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    return matches[0][1]


def _ensure_summa_restart(
    cfg: PysummadaConfig,
    member: str,
    trigger: datetime,
    *,
    sim_start: datetime,
    state_path: Path,
    init_condition_file: str,
    base_state_path: Path,
    output_dir: Path,
    restart_file_filter: str,
) -> Path:
    trigger_key = trigger.strftime("%Y%m%d%H")
    try:
        existing = _find_restart_for_trigger(state_path, restart_file_filter, trigger_key, member)
    except FileNotFoundError:
        existing = None
    if existing:
        log.info("Found SUMMA restart for %s at %s", trigger_key, existing)
        return existing
    init_source = base_state_path / init_condition_file
    _seed_state_file(state_path, init_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    freq_restart = None
    if cfg.summa_args:
        freq_restart = cfg.summa_args.get("-r") or cfg.summa_args.get("freq_restart")
    if not freq_restart:
        log.warning("SUMMA freq_restart not set; defaulting to daily restarts.")
        freq_restart = "d"
    sim_config = {
        "manager": {
            "simStartTime": sim_start.strftime(SUMMA_TIME_FORMAT),
            "simEndTime": trigger.strftime(SUMMA_TIME_FORMAT),
            "statePath": str(state_path) + "/",
            "initConditionFile": init_condition_file,
            "outputPath": str(output_dir) + "/",
        }
    }
    run_simulation(
        str(cfg.summa_exe),
        str(cfg.base_file_manager),
        run_suffix=f"esp_interval_{member}_{trigger.strftime('%Y%m%d%H')}",
        freq_restart=freq_restart,
        write_config=True,
        update_manager=False,
        log_stdout=cfg.summa_log,
        sim_config=sim_config,
    )
    restart_path = _find_restart_for_trigger(
        state_path, restart_file_filter, trigger_key, member
    )
    log.info("Wrote SUMMA restart for %s at %s", trigger_key, restart_path)
    return restart_path


def _mizuroute_restart_name(case_name: str, trigger: datetime) -> str:
    return f"{case_name}.r.{trigger.strftime('%Y-%m-%d')}-00000.nc"


def _mizuroute_sim_end(
    base_end: datetime,
    trace_year: int | None = None,
    *,
    hour_offset: int = 1,
    leap_day_adjust: bool = True,
) -> datetime:
    sim_end = base_end - timedelta(hours=hour_offset)
    if leap_day_adjust and trace_year is not None and calendar.isleap(trace_year):
        sim_end -= timedelta(days=1)
    return sim_end


def _ensure_mizuroute_restart(
    cfg: PysummadaConfig,
    mizu,
    trigger: datetime,
    *,
    summa_output: Path,
    sim_start: datetime,
    output_dir: Path,
    restart_dir: Path,
    restart_in: Path | None = None,
) -> Path:
    restart_path = Path(restart_dir) / _mizuroute_restart_name(mizu.case_name, trigger)
    restart_dir = Path(restart_dir)
    restart_dir.mkdir(parents=True, exist_ok=True)
    if restart_path.exists():
        log.info("Found mizuRoute restart for %s at %s", trigger, restart_path)
        return restart_path
    log.info("mizuRoute restart missing for %s; running interval mizuRoute.", trigger)
    control_parent = Path(mizu.control_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_optional_resource(
        mizu.param_nml,
        [mizu.ancil_dir, control_parent],
        output_dir / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    copy_optional_resource(
        mizu.param_nml,
        [mizu.ancil_dir, control_parent],
        summa_output.parent / "param.nml.default",
        "mizuRoute param.nml.default",
        required=True,
    )
    copy_optional_resource(
        mizu.topology_nc,
        [mizu.ancil_dir, control_parent],
        output_dir / "topology.nc",
        "mizuRoute topology.nc",
    )
    if restart_in is None:
        init_state_name = _read_control_value(Path(mizu.control_file), "fname_state_in")
        if init_state_name:
            copy_optional_resource(
                None,
                [mizu.ancil_dir, control_parent],
                restart_dir / init_state_name,
                "mizuRoute initial state file",
                required=True,
            )
        else:
            log.warning(
                "mizuRoute initial state name not found in %s; restart_dir=%s",
                mizu.control_file,
                restart_dir,
            )
    control_file = output_dir / "mizuroute_interval.control"
    control_settings = {
        "fname_qsim": summa_output.name,
        "case_name": mizu.case_name,
        "sim_start": sim_start.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_end": _mizuroute_sim_end(trigger).strftime("%Y-%m-%d %H:%M:%S"),
        "ancil_dir": str(mizu.ancil_dir) + "/",
        "input_dir": str(summa_output.parent) + "/",
        "output_dir": str(output_dir) + "/",
        "restart_dir": str(restart_dir) + "/",
        "restart_write": mizu.restart_write,
    }
    if restart_in:
        control_settings["fname_state_in"] = restart_in.name
    update_control_file(mizu.control_file, control_file, control_settings)
    run_mizuroute(mizu.executable, control_file, log_stdout=mizu.log_output)
    if not restart_path.exists():
        raise FileNotFoundError(
            f"Expected mizuRoute restart at {restart_path} after interval run."
        )
    log.info("Wrote mizuRoute restart for %s at %s", trigger, restart_path)
    return restart_path


def run_esp_traces(cfg: PysummadaConfig, esp: EspConfig, run_dir: Path) -> list[EspRunRecord]:
    state_path = Path(
        get_file_manager_value(str(cfg.base_file_manager), "statePath").replace("'", "")
    )
    init_condition_file = _strip_quotes(
        get_file_manager_value(str(cfg.base_file_manager), "initConditionFile")
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
        try:
            return _strip_quotes(
                get_file_manager_value(str(cfg.base_file_manager), "outFilePrefix")
            )
        except Exception:
            return cfg.ens_label_array[0]

    trigger_dates = sorted(trigger_dates)
    if not trigger_dates:
        log.warning("ESP enabled but no trigger dates resolved.")
        return []

    records: list[EspRunRecord] = []
    state_members = list(esp.state_members) if esp.state_members else [_resolve_member()]
    for member in state_members:
        if esp.deterministic_state:
            log.warning(
                "ESP deterministic_state provided; incremental ESP sequencing ignores external states."
            )
        summa_state_dir = _esp_summa_state_dir(run_dir, member)
        mizu_state_dir = _esp_mizuroute_state_dir(run_dir, member)
        current_start = cfg.start_time
        init_state_path = _seed_state_file(summa_state_dir, state_path / init_condition_file)
        init_condition_name = init_state_path.name
        last_mizu_restart: Path | None = None

        for trigger in trigger_dates:
            if trigger < current_start:
                log.warning("Skipping ESP trigger %s before %s.", trigger, current_start)
                continue
            interval_dir = _esp_interval_dir(run_dir, member, trigger)
            summa_interval_output = interval_dir / "summa"
            state_file = _ensure_summa_restart(
                cfg,
                member,
                trigger,
                sim_start=current_start,
                state_path=summa_state_dir,
                init_condition_file=init_condition_name,
                base_state_path=state_path,
                output_dir=summa_interval_output,
                restart_file_filter=esp.restart_file_filter,
            )
            if esp.run_mizuroute and esp.mizuroute:
                summa_timestep = find_summa_timestep_output(summa_interval_output)
                mizu_interval_output = interval_dir / "mizuroute"
                last_mizu_restart = _ensure_mizuroute_restart(
                    cfg,
                    esp.mizuroute,
                    trigger,
                    summa_output=summa_timestep,
                    sim_start=current_start,
                    output_dir=mizu_interval_output,
                    restart_dir=mizu_state_dir,
                    restart_in=last_mizu_restart,
                )

            state_date_key = trigger.strftime("%Y%m%d%H")
            for trace_year in range(esp.start_year, esp.end_year + 1):
                trace_start = _replace_year(trigger, trace_year)
                trace_end = resolve_forecast_window_end(
                    trace_start, esp.forecast_length_days, esp.forecast_end
                )
                base_start = trigger
                base_end = resolve_forecast_window_end(
                    base_start, esp.forecast_length_days, esp.forecast_end
                )
                trace_output_dir = _trace_output_dir(run_dir, member, state_date_key, trace_year)
                trace_output_dir.mkdir(parents=True, exist_ok=True)

                run_suffix = f"esp_state{state_date_key}_year{trace_year}"
                sim_config = {
                    "manager": {
                        "simStartTime": trace_start.strftime(SUMMA_TIME_FORMAT),
                        "simEndTime": trace_end.strftime(SUMMA_TIME_FORMAT),
                        "statePath": str(summa_state_dir) + "/",
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

                raw_summa_output = find_summa_timestep_output(trace_output_dir)
                mizu_output_dir = trace_output_dir / "mizuroute"
                mizu_output_dir.mkdir(parents=True, exist_ok=True)
                shifted_summa_output = mizu_output_dir / f"summa_{trace_year}.nc"
                _shift_time_to_state_year(raw_summa_output, trigger.year, shifted_summa_output)

                mizuroute_output: Path | None = None
                if esp.run_mizuroute and esp.mizuroute:
                    mizu = esp.mizuroute
                    control_parent = Path(mizu.control_file).parent
                    copy_optional_resource(
                        mizu.param_nml,
                        [mizu.ancil_dir, control_parent],
                        mizu_output_dir / "param.nml.default",
                        "mizuRoute param.nml.default",
                        required=True,
                    )
                    copy_optional_resource(
                        mizu.topology_nc,
                        [mizu.ancil_dir, control_parent],
                        mizu_output_dir / "topology.nc",
                        "mizuRoute topology.nc",
                    )
                    control_file = mizu_output_dir / f"mizuroute_{trace_year}.control"
                    control_settings = {
                        "fname_qsim": shifted_summa_output.name,
                        "fname_state_in": _mizuroute_restart_name(mizu.case_name, base_start),
                        "case_name": mizu.case_name,
                        "sim_start": base_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "sim_end": _mizuroute_sim_end(
                            base_end,
                            trace_year,
                            leap_day_adjust=esp.forecast_end is None,
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "ancil_dir": str(mizu.ancil_dir) + "/",
                        "input_dir": str(shifted_summa_output.parent) + "/",
                        "output_dir": str(mizu_output_dir) + "/",
                        "restart_dir": str(mizu_state_dir) + "/",
                        "restart_write": mizu.restart_write,
                    }
                    update_control_file(mizu.control_file, control_file, control_settings)
                    run_mizuroute(mizu.executable, control_file, log_stdout=mizu.log_output)
                    routed_generated = find_mizuroute_output(
                        mizu_output_dir, mizu.case_name, trace_start
                    )
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
            current_start = trigger
            init_condition_name = state_file.name
    return records
