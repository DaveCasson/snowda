from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence

import logging
import numpy as np
import pandas as pd
import xarray as xr

from snowda.config.loader import load_config
from snowda.core.assimilation import DataAssimilationCore
from snowda.core.fsca_models import FscaState, initialize_model_arrays
from snowda.core.particle_filter import ParticleFilterCore
from snowda.core.snow5 import Snow5Parameters, run_snow5_simulation
from snowda.io.snow5 import convert_arrays_to_dataset, read_snow5_forcing, write_results
from snowda.schaake_shuffle.archive import Snow5ArchiveAdapter
from snowda.schaake_shuffle.config import SchaakeShuffleConfig
from snowda.schaake_shuffle.core import apply_permutation, schaake_shuffle
from snowda.utils.logging import configure_logging, get_logger
from snowda.workflows.run_snow5 import _load_cv_array

log = get_logger(__name__)


def _parse_time(value: str | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD (got: {value})") from exc


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_helbig_params_nc(path: str | Path) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if "hru" not in ds.dims:
            raise ValueError("Helbig params dataset must have an 'hru' dimension.")
        return {"mu": ds["mu"].values.astype("float64"), "xi": ds["xi"].values.astype("float64")}


def _load_observations(
    path: str | Path, variable: str, start: datetime, end: datetime
) -> xr.DataArray:
    obs_ds = xr.open_dataset(path)
    if variable not in obs_ds:
        raise ValueError(f"Observation variable {variable} not found in {path}")
    obs_da = obs_ds[variable].sel(time=slice(start, end))
    if obs_da.sizes.get("time", 0) == 0:
        raise ValueError("No observation data falls within the requested time window.")
    return obs_da


def _serialize_indexed_dict(values: dict) -> dict:
    serialized = {}
    for key, value in values.items():
        if isinstance(key, tuple):
            key_repr = f"{key[0]}_{key[1]}"
        else:
            key_repr = str(key)
        serialized[key_repr] = float(value)
    return serialized


def _build_assimilation_times(
    start_time: datetime, end_time: datetime, interval_days: int
) -> list[datetime]:
    if interval_days <= 0:
        raise ValueError("assimilation_interval_days must be > 0")
    window_start = start_time
    interval = timedelta(days=interval_days)
    times: list[datetime] = []
    while window_start <= end_time:
        window_end = min(window_start + interval - timedelta(days=1), end_time)
        times.append(window_end)
        window_start = window_end + timedelta(days=1)
    return times


@dataclass(frozen=True)
class Snow5DAConfig:
    start_time: datetime
    end_time: datetime
    time_delta: timedelta
    ens_label_array: list[str]


@dataclass
class Snow5Particle:
    label: str
    forcing_path: Path
    forcing_ds: xr.Dataset
    model_arrays: dict[str, np.ndarray]
    fsca_states: list[FscaState]
    outputs: list[xr.Dataset] = field(default_factory=list)

    @property
    def hru_index(self) -> np.ndarray:
        if "hruId" in self.forcing_ds.coords:
            return self.forcing_ds["hruId"].values
        return self.forcing_ds["hru"].values


@dataclass
class Snow5ParticleState:
    label: str
    model_arrays: dict[str, np.ndarray]
    fsca_states: list[FscaState]


class Snow5DAEngine:
    def __init__(
        self,
        particles: list[Snow5Particle],
        ens_labels: list[str],
        particle_filter: ParticleFilterCore,
        assimilation_times: list[datetime],
        assimilation_var: str,
        modeled_var: str,
        observation_da: xr.DataArray,
        z_var: float,
        n_eff_threshold: float,
        fsca_model: str,
        helbig_params: dict[str, np.ndarray] | None,
        helbig_settings: dict[str, Any] | None,
        snow_density: float,
        pf_resample: bool,
        state_mean_name: str,
        state_var_name: str,
        obs_var_name: str,
        schaake_shuffle_config: SchaakeShuffleConfig | None = None,
    ):
        self.particles = particles
        self._forcing_lookup = {p.label: p.forcing_ds for p in particles}
        self.ens_labels = ens_labels
        self.pf = particle_filter
        self.assimilation_times = assimilation_times
        self.assimilation_var = assimilation_var
        self.observation = observation_da
        self.z_var = z_var
        self.n_eff_threshold = n_eff_threshold
        self.fsca_model = fsca_model
        self.helbig_params = helbig_params
        self.helbig_settings = helbig_settings
        self.snow_density = snow_density
        self.modeled_var = modeled_var
        self.pf_resample = pf_resample
        self.state_mean_name = state_mean_name
        self.state_var_name = state_var_name
        self.obs_var_name = obs_var_name
        self.schaake_shuffle_config = schaake_shuffle_config
        self._schaake_archive = self._initialize_schaake_archive(schaake_shuffle_config)
        self._schaake_rng = np.random.default_rng(
            schaake_shuffle_config.seed if schaake_shuffle_config else None
        )
        self.state_snapshots: dict[datetime, dict[str, Snow5ParticleState]] = {}
        first_hrus = particles[0].hru_index
        self.hru_index_df = pd.DataFrame({"hruId": first_hrus})

    def run(
        self, params: Snow5Parameters, cv_array: np.ndarray, start_time: datetime
    ) -> list[Dict[str, Any]]:
        assimilation_records: list[Dict[str, Any]] = []
        current_start = start_time
        for step_idx, assim_end in enumerate(self.assimilation_times):
            interval_start = current_start
            interval_end = assim_end
            step_records = self._step_interval(
                params,
                cv_array,
                interval_start,
                interval_end,
            )
            assimilation_records.append(step_records)
            current_start = interval_end + timedelta(days=1)
        return assimilation_records

    def _step_interval(
        self,
        params: Snow5Parameters,
        cv_array: np.ndarray,
        interval_start: datetime,
        interval_end: datetime,
    ) -> Dict[str, Any]:
        for particle in self.particles:
            subset = particle.forcing_ds.sel(time=slice(interval_start, interval_end))
            if subset.sizes.get("time", 0) == 0:
                raise ValueError(f"No forcing data between {interval_start} and {interval_end} for {particle.label}")
            output_arrays, model_arrays, fsca_states = run_snow5_simulation(
                subset,
                params,
                cv_array,
                fsca_model=self.fsca_model,
                helbig_settings=self.helbig_settings,
                helbig_params=self.helbig_params,
                snow_density=self.snow_density,
                initial_model_arrays=particle.model_arrays,
                initial_states=particle.fsca_states,
            )
            result_ds = convert_arrays_to_dataset(output_arrays, subset)
            particle.outputs.append(result_ds)
            particle.model_arrays = model_arrays
            particle.fsca_states = fsca_states
        particles_df = self._build_particles_df(interval_end)
        obs_df = self._extract_obs(interval_end)
        weights = self._update_weights(particles_df, obs_df, interval_end)
        log.info(
            "Particle weights (%s): %s",
            interval_end.strftime("%Y-%m-%d"),
            weights.iloc[0].to_dict(),
        )
        mean_df, var_df = self.pf.state_estimate(
            particles_df,
            weights,
            interval_end,
            self.hru_index_df,
            mean_name=self.state_mean_name,
            var_name=self.state_var_name,
        )
        log.info(
            "State estimate (%s): %s range %s - %s",
            interval_end.strftime("%Y-%m-%d"),
            self.state_mean_name,
            float(mean_df[self.state_mean_name].min()),
            float(mean_df[self.state_mean_name].max()),
        )
        resample = self.pf.resample_indices(weights, len(self.particles), self.n_eff_threshold)
        if not self.pf_resample:
            resample = None
        elif resample is not None:
            self._apply_resample(resample)
            log.info("Resampled particles at %s -> %s", interval_end, resample.tolist())
        schaake_applied = self._apply_schaake_shuffle(interval_end)
        self._capture_state_snapshot(interval_end)
        return {
            "time": interval_end,
            "weights": weights,
            "resample": resample,
            self.state_mean_name: mean_df[self.state_mean_name].to_dict(),
            self.state_var_name: var_df[self.state_var_name].to_dict(),
            "schaake_shuffle_applied": schaake_applied,
        }

    def _capture_state_snapshot(self, time: datetime) -> None:
        self.state_snapshots[time] = {
            particle.label: Snow5ParticleState(
                label=particle.label,
                model_arrays={
                    key: np.asarray(value, dtype="float64").copy()
                    for key, value in particle.model_arrays.items()
                },
                fsca_states=[copy.deepcopy(state) for state in particle.fsca_states],
            )
            for particle in self.particles
        }

    def _build_particles_df(self, time: datetime) -> pd.DataFrame:
        index = pd.MultiIndex.from_arrays(
            [self.hru_index_df["hruId"].values, np.repeat(time, len(self.hru_index_df))],
            names=["hru", "time"],
        )
        df = pd.DataFrame(index=index)
        for label, particle in zip(self.ens_labels, self.particles):
            modeled_vals = particle.outputs[-1][self.modeled_var].sel(time=time).values
            df[label] = modeled_vals
        return df

    def _extract_obs(self, time: datetime) -> pd.DataFrame:
        obs_slice = self.observation.sel(time=time)
        df = obs_slice.to_dataframe().reset_index()
        if self.obs_var_name not in df.columns:
            df.rename(columns={obs_slice.name: self.obs_var_name}, inplace=True)
        df.set_index(["hru", "time"], inplace=True)
        return df

    def _update_weights(self, particles_df: pd.DataFrame, obs_df: pd.DataFrame, time: datetime) -> pd.DataFrame:
        if self.assimilation_var == "fsca":
            return self.pf.update_particle_weights_fsca(
                particles_df,
                obs_df,
                self.z_var,
                time,
                self.hru_index_df,
                self.ens_labels,
                obs_column=self.obs_var_name,
            )
        return self.pf.update_particle_weights(
            particles_df,
            obs_df,
            self.z_var,
            time,
            self.hru_index_df,
            self.ens_labels,
            obs_column=self.obs_var_name,
        )

    def _apply_resample(self, indices: np.ndarray) -> None:
        old_particles = list(self.particles)
        new_particles: list[Snow5Particle] = []
        for dest_idx, src_idx in enumerate(indices):
            base = old_particles[src_idx]
            dest_old = old_particles[dest_idx]
            particle_copy = Snow5Particle(
                label=self.ens_labels[dest_idx],
                forcing_path=dest_old.forcing_path,
                forcing_ds=self._forcing_lookup[self.ens_labels[dest_idx]],
                model_arrays={key: value.copy() for key, value in base.model_arrays.items()},
                fsca_states=[copy.deepcopy(state) for state in base.fsca_states],
                outputs=list(dest_old.outputs),
            )
            new_particles.append(particle_copy)
        self.particles = new_particles

    def _initialize_schaake_archive(
        self, config: SchaakeShuffleConfig | None
    ) -> Snow5ArchiveAdapter | None:
        if not config or not config.enabled:
            return None
        return Snow5ArchiveAdapter(config.archive_source, config.archive_variable)

    def _collect_variable_matrix(
        self, variable: str, time: datetime
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        values = []
        value_shape: tuple[int, ...] | None = None
        for particle in self.particles:
            ds = particle.outputs[-1]
            if variable not in ds:
                raise KeyError(f"Variable {variable} missing from particle outputs.")
            selection = ds[variable].sel(time=time)
            array = np.asarray(selection.values, dtype="float64")
            if value_shape is None:
                value_shape = array.shape
            elif array.shape != value_shape:
                raise ValueError(
                    f"Variable {variable} has inconsistent shapes across particles."
                )
            values.append(array)
        matrix = np.stack(values, axis=0)
        assert value_shape is not None  # for mypy
        if matrix.shape[0] != len(self.particles):
            raise ValueError("Mismatch between particles and matrix rows.")
        return matrix, value_shape

    def _write_variable_matrix(
        self,
        variable: str,
        matrix: np.ndarray,
        value_shape: tuple[int, ...],
        time: datetime,
    ) -> None:
        for member_idx, particle in enumerate(self.particles):
            reshaped = matrix[member_idx].reshape(value_shape)
            ds = particle.outputs[-1]
            ds[variable].loc[{"time": time}] = reshaped
            if variable in particle.model_arrays:
                particle.model_arrays[variable] = reshaped.reshape(-1).copy()

    def _apply_schaake_shuffle(self, time: datetime) -> bool:
        config = self.schaake_shuffle_config
        archive = self._schaake_archive
        if not config or not config.enabled or archive is None:
            return False
        primary_var = config.target_variables[0]
        matrix, value_shape = self._collect_variable_matrix(primary_var, time)
        try:
            reference = archive.sample_Y(
                time,
                len(self.particles),
                config.window_days,
                rng=self._schaake_rng,
            )
        except ValueError:
            log.warning(
                "Schaake shuffle skipped: no archive candidates for %s", time
            )
            return False
        shuffled, permutation = schaake_shuffle(matrix, reference)
        self._write_variable_matrix(primary_var, shuffled, value_shape, time)
        for var in config.apply_same_permutation_to:
            if var == primary_var:
                continue
            try:
                target_matrix, target_shape = self._collect_variable_matrix(var, time)
            except (KeyError, ValueError):
                log.debug("Skipping Schaake permutation for %s (missing or invalid).", var)
                continue
            if target_matrix.shape != permutation.shape:
                log.warning(
                    "Skipping Schaake permutation for %s due to shape mismatch.", var
                )
                continue
            permuted = apply_permutation(target_matrix, permutation)
            self._write_variable_matrix(var, permuted, target_shape, time)
        log.info(
            "Applied Schaake shuffle for %s on variables %s",
            time.strftime("%Y-%m-%d"),
            config.target_variables,
        )
        return True

    def finalize_outputs(self, run_dir: Path) -> list[Path]:
        output_paths = []
        for particle in self.particles:
            combined = xr.concat(particle.outputs, dim="time")
            output_path = run_dir / f"{particle.label}_snow5_output.nc"
            write_results(combined, output_path)
            output_paths.append(output_path)
        return output_paths


def _replace_year(dt: datetime, year: int) -> datetime:
    try:
        return dt.replace(year=year)
    except ValueError:
        max_day = pd.Timestamp(f"{year}-{dt.month:02d}-01").days_in_month
        return dt.replace(year=year, day=min(dt.day, max_day))


def _parse_esp_dates(values: Sequence[str]) -> list[datetime]:
    return [pd.Timestamp(v).to_pydatetime() for v in values]


def _resolve_esp_trigger_dates(
    start: datetime,
    end: datetime,
    explicit_dates: Sequence[datetime],
    frequency_days: int | None,
    months: Sequence[int],
) -> list[datetime]:
    dates = set(explicit_dates)
    if frequency_days:
        current = start
        while current <= end:
            dates.add(current)
            current += timedelta(days=frequency_days)
    if months:
        month_set = set(months)
        dates = {dt for dt in dates if dt.month in month_set}
    return sorted(dates)


def _shift_dataset_time_to_year(ds: xr.Dataset, target_year: int) -> xr.Dataset:
    shifted_times = [
        _replace_year(pd.Timestamp(ts).to_pydatetime(), target_year)
        for ts in pd.to_datetime(ds["time"].values)
    ]
    out = ds.copy(deep=True)
    out["time"] = pd.DatetimeIndex(shifted_times)
    return out


def _run_snow5_esp_from_da(
    cfg: Dict[str, Any],
    output_dir: Path,
    da_engine: Snow5DAEngine,
    params: Snow5Parameters,
    cv_array: np.ndarray,
    ens_labels: list[str],
    assimilation_times: list[datetime],
    forcing_lookup: dict[str, xr.Dataset],
    fsca_model: str,
    helbig_params: dict[str, np.ndarray] | None,
    helbig_settings: dict[str, Any] | None,
    snow_density: float,
) -> list[dict[str, Any]]:
    esp_cfg = cfg.get("esp")
    if not isinstance(esp_cfg, dict) or not bool(esp_cfg.get("enabled", False)):
        return []

    start_year = int(esp_cfg["start_year"])
    end_year = int(esp_cfg["end_year"])
    forecast_length_days = int(esp_cfg["forecast_length_days"])
    if end_year < start_year:
        raise ValueError("`esp.end_year` must be >= `esp.start_year`.")
    if forecast_length_days <= 0:
        raise ValueError("`esp.forecast_length_days` must be > 0.")

    frequency_days_raw = esp_cfg.get("frequency_days")
    frequency_days = int(frequency_days_raw) if frequency_days_raw is not None else None
    explicit_dates = _parse_esp_dates([str(v) for v in (esp_cfg.get("dates", []) or [])])
    months = [int(m) for m in (esp_cfg.get("months", []) or [])]
    if frequency_days is None and not explicit_dates:
        raise ValueError("ESP requires either `esp.frequency_days` or `esp.dates`.")

    trigger_dates = _resolve_esp_trigger_dates(
        assimilation_times[0],
        assimilation_times[-1],
        explicit_dates,
        frequency_days,
        months,
    )
    if not trigger_dates:
        log.warning("ESP enabled but no trigger dates resolved for Snow5 DA.")
        return []

    state_members_cfg = esp_cfg.get("state_members", []) or []
    state_members = [str(m) for m in state_members_cfg] if state_members_cfg else [ens_labels[0]]
    available_times = set(da_engine.state_snapshots.keys())

    records: list[dict[str, Any]] = []
    for trigger in trigger_dates:
        if trigger not in available_times:
            log.info("Skipping ESP trigger %s (not an assimilation/state date).", trigger)
            continue
        state_snapshot = da_engine.state_snapshots[trigger]
        for state_member in state_members:
            particle_state = state_snapshot.get(state_member)
            if particle_state is None:
                log.info(
                    "Skipping ESP state member %s at %s (snapshot missing).",
                    state_member,
                    trigger,
                )
                continue
            forcing_ds = forcing_lookup[state_member]
            for trace_year in range(start_year, end_year + 1):
                trace_start = _replace_year(trigger, trace_year)
                trace_end = trace_start + timedelta(days=forecast_length_days)
                forcing_subset = forcing_ds.sel(time=slice(trace_start, trace_end))
                if forcing_subset.sizes.get("time", 0) == 0:
                    log.info(
                        "Skipping Snow5 ESP trace year %s for %s at %s (no forcing window).",
                        trace_year,
                        state_member,
                        trigger,
                    )
                    continue
                output_arrays, _, _ = run_snow5_simulation(
                    forcing_subset,
                    params,
                    cv_array,
                    fsca_model=fsca_model,
                    helbig_settings=helbig_settings,
                    helbig_params=helbig_params,
                    snow_density=snow_density,
                    initial_model_arrays={
                        k: v.copy() for k, v in particle_state.model_arrays.items()
                    },
                    initial_states=[copy.deepcopy(state) for state in particle_state.fsca_states],
                )
                trace_ds = convert_arrays_to_dataset(output_arrays, forcing_subset)
                trace_ds = _shift_dataset_time_to_year(trace_ds, trigger.year)
                trace_dir = output_dir / "esp" / state_member / trigger.strftime("%Y%m%d")
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"snow5_{trace_year}.nc"
                write_results(trace_ds, trace_path)
                records.append(
                    {
                        "state_member": state_member,
                        "state_date": trigger.isoformat(),
                        "trace_year": trace_year,
                        "output": str(trace_path),
                    }
                )
    return records


def run_snow5_da(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    base_dir = Path(cfg.get("_meta", {}).get("config_dir", Path(config_path).resolve().parent))
    output_dir = _ensure_dir(Path(cfg.get("output_dir") or ".") / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))

    log_file = cfg.get("log_file")
    if log_file:
        log_path = output_dir / Path(log_file).name
        configure_logging(level=logging.INFO, log_file=log_path)
    else:
        configure_logging(level=logging.INFO)

    forcing_dir = cfg.get("ensemble_forcing_dir")
    if not forcing_dir:
        raise ValueError("Config must define `ensemble_forcing_dir`.")
    forcing_dir_path = Path(forcing_dir)
    if not forcing_dir_path.is_absolute():
        forcing_dir_path = (base_dir / forcing_dir_path).resolve()
    if not forcing_dir_path.exists():
        raise FileNotFoundError(f"Forcing directory not found: {forcing_dir_path}")
    pattern = cfg.get("ensemble_forcing_pattern", "*.nc")
    forcing_paths = sorted(forcing_dir_path.glob(pattern))
    if not forcing_paths:
        raise FileNotFoundError(f"No forcing files matching {pattern} found in {forcing_dir_path}")

    start_time = _parse_time(cfg.get("start_time"), "start_time")
    end_time = _parse_time(cfg.get("end_time"), "end_time")
    if not start_time or not end_time:
        raise ValueError("start_time and end_time are required.")
    assimilation_interval_days = int(cfg.get("assimilation_interval_days", 7))
    assimilation_times = _build_assimilation_times(start_time, end_time, assimilation_interval_days)

    obs_nc = cfg.get("assimilation_nc")
    if not obs_nc:
        raise ValueError("assimilation_nc is required for DA runs.")
    obs_var = cfg.get("assimilation_var", "swe")
    obs_var_name = cfg.get("obs_var_name", obs_var)
    modeled_var = cfg.get("modeled_var")
    if not modeled_var:
        modeled_var = "swe" if obs_var == "swe" else obs_var
    modeled_var_name = cfg.get("modeled_var_name", modeled_var)
    state_mean_name = cfg.get("state_mean_name", "mean_state")
    state_var_name = cfg.get("state_var_name", "var_state")
    observation = _load_observations(obs_nc, obs_var_name, start_time, end_time)
    obs_times = {pd.Timestamp(t).to_pydatetime() for t in observation["time"].values}
    missing_times = [t for t in assimilation_times if t not in obs_times]
    if missing_times:
        raise ValueError(f"Observations missing for assimilation times: {missing_times}")

    model_params_cfg = cfg.get("model_parameters")
    if not isinstance(model_params_cfg, dict):
        raise ValueError("`model_parameters` must be a mapping.")
    params = Snow5Parameters.from_dict(model_params_cfg)

    cv_array = _load_cv_array(cfg, read_snow5_forcing(forcing_paths[0]))
    snow_density = float(cfg.get("snow_density", 300.0))
    fsca_model = cfg.get("fsca_model", "ssnowd")
    helbig_params = (
        _load_helbig_params_nc(cfg["helbig_params_nc"])
        if fsca_model == "helbig"
        else None
    )
    helbig_settings = cfg.get("helbig_settings")
    if fsca_model == "helbig" and not helbig_settings:
        raise ValueError("helbig_settings are required when fsca_model is 'helbig'.")
    pf_resample = bool(cfg.get("pf_resample", True))

    particles = []
    forcing_lookup: dict[str, xr.Dataset] = {}
    ens_labels = [f"{idx+1:03d}" for idx in range(len(forcing_paths))]
    for label, fp in zip(ens_labels, forcing_paths):
        ds = read_snow5_forcing(fp)
        forcing_lookup[label] = ds
        particles.append(
            Snow5Particle(
                label=label,
                forcing_path=fp,
                forcing_ds=ds,
                model_arrays=initialize_model_arrays(np.zeros(len(ds["hru"]))),
                fsca_states=[FscaState() for _ in range(len(ds["hru"]))],
            )
        )

    pf_core = ParticleFilterCore()
    da_config = Snow5DAConfig(
        start_time=start_time,
        end_time=end_time,
        time_delta=timedelta(days=assimilation_interval_days),
        ens_label_array=ens_labels,
    )
    da_core = DataAssimilationCore(da_config)
    da_core.set_run_period()
    schaake_shuffle_config = SchaakeShuffleConfig.from_config(
        cfg.get("schaake_shuffle"),
        Path(cfg["_meta"]["config_dir"]),
        modeled_var,
    )
    da_engine = Snow5DAEngine(
        particles=particles,
        ens_labels=ens_labels,
        particle_filter=pf_core,
        assimilation_times=assimilation_times,
        assimilation_var=obs_var,
        modeled_var=modeled_var,
        observation_da=observation,
        z_var=float(cfg.get("z_var", 1.0)),
        n_eff_threshold=float(cfg.get("n_eff_threshold", 0.5)),
        fsca_model=fsca_model,
        helbig_params=helbig_params,
        helbig_settings=helbig_settings,
        snow_density=snow_density,
        pf_resample=pf_resample,
        state_mean_name=state_mean_name,
        state_var_name=state_var_name,
        obs_var_name=obs_var_name,
        schaake_shuffle_config=schaake_shuffle_config,
    )

    records = da_engine.run(params, cv_array, start_time)
    outputs = da_engine.finalize_outputs(output_dir)
    esp_records = _run_snow5_esp_from_da(
        cfg,
        output_dir,
        da_engine,
        params,
        cv_array,
        ens_labels,
        assimilation_times,
        forcing_lookup,
        fsca_model,
        helbig_params,
        helbig_settings,
        snow_density,
    )

    prov = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "ensemble_forcing_files": [str(p) for p in forcing_paths],
        "assimilation_times": [t.isoformat() for t in assimilation_times],
        "outputs": [str(p) for p in outputs],
        "pf_resample": pf_resample,
        "obs_var_name": obs_var_name,
        "modeled_var_name": modeled_var_name,
        "state_mean_name": state_mean_name,
        "state_var_name": state_var_name,
        "esp_enabled": bool(isinstance(cfg.get("esp"), dict) and cfg["esp"].get("enabled")),
        "esp_trace_count": len(esp_records),
        "esp_traces": esp_records,
        "records": [
            {
                "time": record["time"].isoformat(),
                "resample": record["resample"].tolist()
                if record["resample"] is not None
                else None,
                state_mean_name: _serialize_indexed_dict(record[state_mean_name]),
                state_var_name: _serialize_indexed_dict(record[state_var_name]),
            }
            for record in records
        ],
    }
    prov_out = output_dir / "provenance.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("Wrote provenance: %s", prov_out)
    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run snow5 data assimilation ensemble.")
    parser.add_argument("config", type=str, help="Path to snow5 DA config")
    args = parser.parse_args()
    run_snow5_da(args.config)


if __name__ == "__main__":
    main()
