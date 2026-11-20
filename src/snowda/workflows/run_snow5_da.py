from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from snowda.config.loader import load_config
from snowda.config.pysummada import _parse_esp_horizon
from snowda.config.run_dir import make_run_dir, run_name_from_config, utc_timestamp
from snowda.core.fsca_models import FscaState, initialize_model_arrays
from snowda.core.particle_filter import (
    DEFAULT_N_EFF_THRESHOLD,
    TEMPERING_BETA,
    ParticleFilterCore,
    resampling_rng,
)
from snowda.core.snow5 import Snow5Parameters, run_snow5_simulation
from snowda.io.snow5 import convert_arrays_to_dataset, read_snow5_forcing, write_results
from snowda.schaake_shuffle.archive import Snow5ArchiveAdapter
from snowda.schaake_shuffle.config import SchaakeShuffleConfig
from snowda.schaake_shuffle.core import (
    apply_permutation,
    schaake_shuffle,
    shuffle_source,
)
from snowda.utils.dates import replace_year, scheduled_dates
from snowda.utils.logging import configure_logging, get_logger
from snowda.workflows.esp import resolve_forecast_window_end
from snowda.workflows.run_snow5 import _load_cv_array

log = get_logger(__name__)


def _positive_swe_setting(cfg: Dict[str, Any], key: str) -> float:
    """Read one required SWE observation-uncertainty setting.

    ``swe_relative_uncertainty`` is one standard deviation as a fraction of
    absolute observed SWE. ``swe_sigma_floor`` is the absolute floor in kg m-2.
    """
    if key not in cfg:
        raise ValueError(
            f"`{key}` is required for SWE updates. "
            "`swe_relative_uncertainty` is one standard deviation as a fraction "
            "of |observed SWE|. `swe_sigma_floor` is the absolute "
            "standard-deviation floor in kg m-2."
        )
    value = float(cfg[key])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"`{key}` must be finite and > 0.")
    return value


def _parse_time(value: str | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD (got: {value})") from exc


def _load_helbig_params_nc(path: str | Path) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if "hru" not in ds.dims:
            raise ValueError("Helbig params dataset must have an 'hru' dimension.")
        return {"mu": ds["mu"].values.astype("float64"), "xi": ds["xi"].values.astype("float64")}


def _station_obs_to_hru(obs_da: xr.DataArray, hru_stations: Sequence[dict]) -> xr.DataArray:
    """Map ``(time, station_id)`` observations onto ``(time, hru)``.

    ``hru_stations`` is a list of ``{hru_id, station_id}`` pairs. Stations that
    share an HRU are averaged over their finite values on each date. Stations
    not listed are ignored.
    """
    known = {str(s) for s in obs_da["station_id"].values}
    by_hru: dict[int, list[str]] = {}
    for item in hru_stations:
        sid = str(item["station_id"])
        if sid not in known:
            raise ValueError(f"hru_stations station_id {sid!r} not in observations")
        by_hru.setdefault(int(item["hru_id"]), []).append(sid)
    columns = []
    for hru_id, stations in sorted(by_hru.items()):
        values = obs_da.sel(station_id=stations).values
        with np.errstate(all="ignore"):
            columns.append(np.nanmean(values, axis=1) if values.ndim == 2 else values)
    out = xr.DataArray(
        np.column_stack(columns),
        dims=("time", "hru"),
        coords={"time": obs_da["time"].values, "hru": sorted(by_hru)},
        name=obs_da.name,
        attrs=obs_da.attrs,
    )
    return out


def _load_observations(
    path: str | Path,
    variable: str,
    start: datetime,
    end: datetime,
    hru_stations: Sequence[dict] | None = None,
) -> xr.DataArray:
    obs_ds = xr.open_dataset(path)
    if variable not in obs_ds:
        raise ValueError(f"Observation variable {variable} not found in {path}")
    obs_da = obs_ds[variable].sel(time=slice(start, end))
    if "station_id" in obs_da.dims:
        if not hru_stations:
            raise ValueError(
                "Observations are indexed by station_id; set `hru_stations` "
                "(list of {hru_id, station_id}) in the config."
            )
        obs_da = _station_obs_to_hru(obs_da.transpose("time", "station_id"), hru_stations)
    if obs_da.sizes.get("time", 0) == 0:
        raise ValueError("No observation data falls within the requested time window.")
    return obs_da


def _observation_times(obs_da: xr.DataArray, end_time: datetime) -> list[datetime]:
    """Update on each date with a finite observation, then run on to ``end_time``.

    ``end_time`` is added when it is after the last observation; that segment
    only propagates the particles (no observation, no update).
    """
    finite = np.isfinite(obs_da.values).reshape(obs_da.sizes["time"], -1).any(axis=1)
    times = sorted(pd.Timestamp(t).to_pydatetime() for t in obs_da["time"].values[finite])
    if not times:
        raise ValueError("No finite observations in the requested time window.")
    if times[-1] < end_time:
        times.append(end_time)
    return times


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
        z_var: float | None,
        n_eff_threshold: float,
        fsca_model: str,
        helbig_params: dict[str, np.ndarray] | None,
        helbig_settings: dict[str, Any] | None,
        snow_density: float,
        pf_resample: bool,
        state_mean_name: str,
        state_var_name: str,
        obs_var_name: str,
        swe_relative_uncertainty: float | None = None,
        swe_sigma_floor: float | None = None,
        schaake_shuffle_config: SchaakeShuffleConfig | None = None,
        pf_seed: int | None = None,
        tempering_target: float | None = None,
        season_start_month: int | None = None,
    ):
        self.particles = particles
        self._forcing_lookup = {p.label: p.forcing_ds for p in particles}
        self.ens_labels = ens_labels
        self.pf = particle_filter
        self.assimilation_times = assimilation_times
        self.assimilation_var = assimilation_var
        self.observation = observation_da
        self.z_var = z_var
        self.swe_relative_uncertainty = swe_relative_uncertainty
        self.swe_sigma_floor = swe_sigma_floor
        self.n_eff_threshold = n_eff_threshold
        self.pf_seed = pf_seed
        self.tempering_target = tempering_target
        self.season_start_month = season_start_month
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
        self.prior_weights: pd.DataFrame | None = None
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
        self.prior_weights = self.pf.season_prior(
            self.prior_weights, interval_end, self.season_start_month
        )
        prior = self._prior_frame(interval_end)
        if obs_df is None or not np.isfinite(obs_df[self.obs_var_name].values).any():
            log.info("No observation at %s: propagate only", interval_end.strftime("%Y-%m-%d"))
            self._capture_state_snapshot(interval_end)
            return {
                "time": interval_end,
                "weights": prior,
                "prior_weights": prior,
                "n_eff": self.pf.effective_sample_size(prior),
                "tempering_beta": float("nan"),
                "resample": None,
                "updated": False,
                self.state_mean_name: {},
                self.state_var_name: {},
                "schaake_shuffle_applied": False,
            }
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
        n_eff = self.pf.effective_sample_size(weights)
        resample = self.pf.resample_indices(
            weights,
            len(self.particles),
            self.n_eff_threshold,
            rng=resampling_rng(self.pf_seed, interval_end),
        )
        if not self.pf_resample:
            resample = None
        elif resample is not None:
            self._apply_resample(resample)
            log.info("Resampled particles at %s -> %s", interval_end, resample.tolist())
        beta = weights.attrs[TEMPERING_BETA]
        log.info(
            "PF update %s: beta %.4g, N_eff %.1f of %d, resampled %s",
            interval_end.strftime("%Y-%m-%d"),
            beta,
            n_eff,
            len(self.particles),
            resample is not None,
        )
        self.prior_weights = self.pf.next_prior(weights, resample is not None)
        schaake_applied = self._apply_schaake_shuffle(interval_end)
        self._capture_state_snapshot(interval_end)
        return {
            "time": interval_end,
            "weights": weights,
            "prior_weights": prior,
            "updated": True,
            "n_eff": n_eff,
            "tempering_beta": beta,
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

    def _prior_frame(self, time: datetime) -> pd.DataFrame:
        """Prior weights for an update at ``time`` (equal when there is none)."""
        n = len(self.ens_labels)
        if self.prior_weights is None:
            values = np.full(n, 1.0 / n)
        else:
            values = self.prior_weights.iloc[0].to_numpy(dtype=float)
            values = values / values.sum()
        return pd.DataFrame([values], index=[time], columns=self.ens_labels)

    def _extract_obs(self, time: datetime) -> pd.DataFrame | None:
        if np.datetime64(time) not in self.observation["time"].values:
            return None
        obs_slice = self.observation.sel(time=time)
        df = obs_slice.to_dataframe().reset_index()
        if self.obs_var_name not in df.columns:
            df.rename(columns={obs_slice.name: self.obs_var_name}, inplace=True)
        df.set_index(["hru", "time"], inplace=True)
        return df

    def _update_weights(self, particles_df: pd.DataFrame, obs_df: pd.DataFrame, time: datetime) -> pd.DataFrame:
        if self.assimilation_var == "fsca":
            if self.z_var is None:
                raise ValueError("fSCA updates require z_var")
            return self.pf.update_particle_weights_fsca(
                particles_df,
                obs_df,
                self.z_var,
                time,
                self.hru_index_df,
                self.ens_labels,
                obs_column=self.obs_var_name,
                prior_weights=self.prior_weights,
                tempering_target=self.tempering_target,
            )
        if self.swe_relative_uncertainty is None or self.swe_sigma_floor is None:
            raise ValueError(
                "SWE updates require swe_relative_uncertainty and swe_sigma_floor"
            )
        return self.pf.update_particle_weights(
            particles_df,
            obs_df,
            time,
            self.hru_index_df,
            self.ens_labels,
            obs_column=self.obs_var_name,
            relative_uncertainty=self.swe_relative_uncertainty,
            sigma_floor=self.swe_sigma_floor,
            prior_weights=self.prior_weights,
            tempering_target=self.tempering_target,
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
        shuffled, rank_index = schaake_shuffle(matrix, reference)
        # Other variables follow the member their SWE value came from.
        permutation = shuffle_source(matrix, rank_index)
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


def _parse_esp_dates(values: Sequence[str]) -> list[datetime]:
    return [pd.Timestamp(v).to_pydatetime() for v in values]


def _shift_dataset_time_by_years(ds: xr.Dataset, year_offset: int) -> xr.Dataset:
    """Move every time stamp by whole years, so a trace can cross 31 December."""
    shifted_times = [
        replace_year(stamp, stamp.year + year_offset)
        for stamp in (pd.Timestamp(ts).to_pydatetime() for ts in ds["time"].values)
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
    if end_year < start_year:
        raise ValueError("`esp.end_year` must be >= `esp.start_year`.")
    forecast_length_days, forecast_end = _parse_esp_horizon(esp_cfg)
    # The issue year's weather is the verifying weather, so it is not a trace.
    exclude_issue_year = bool(esp_cfg.get("exclude_issue_year", True))

    frequency_days_raw = esp_cfg.get("frequency_days")
    frequency_days = int(frequency_days_raw) if frequency_days_raw is not None else None
    explicit_dates = _parse_esp_dates([str(v) for v in (esp_cfg.get("dates", []) or [])])
    months = [int(m) for m in (esp_cfg.get("months", []) or [])]
    if frequency_days is None and not explicit_dates:
        raise ValueError("ESP requires either `esp.frequency_days` or `esp.dates`.")

    trigger_dates = scheduled_dates(
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
                if exclude_issue_year and trace_year == trigger.year:
                    continue
                trace_start = replace_year(trigger, trace_year)
                trace_end = resolve_forecast_window_end(
                    trace_start, forecast_length_days, forecast_end
                )
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
                trace_ds = _shift_dataset_time_by_years(
                    trace_ds, trigger.year - trace_year
                )
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
    output_dir = make_run_dir(cfg.get("output_dir") or ".", run_name_from_config(cfg))

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
    # "interval": update every assimilation_interval_days (obs required on each date).
    # "observations": update on each date with a finite observation.
    schedule = str(cfg.get("assimilation_schedule", "interval"))
    if schedule not in ("interval", "observations"):
        raise ValueError("`assimilation_schedule` must be 'interval' or 'observations'.")

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
    observation = _load_observations(
        obs_nc, obs_var_name, start_time, end_time, cfg.get("hru_stations")
    )
    if schedule == "observations":
        assimilation_times = _observation_times(observation, end_time)
        log.info("Assimilation on %d observation dates", len(assimilation_times))
    else:
        assimilation_times = _build_assimilation_times(
            start_time, end_time, assimilation_interval_days
        )
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
    n_eff_threshold = float(cfg.get("n_eff_threshold", DEFAULT_N_EFF_THRESHOLD))
    pf_seed = None if cfg.get("pf_seed") is None else int(cfg["pf_seed"])
    if pf_seed is not None and pf_seed < 0:
        raise ValueError("`pf_seed` must be >= 0.")
    tempering_target = (
        None if cfg.get("tempering_target") is None else float(cfg["tempering_target"])
    )
    if tempering_target is not None and not 0.0 < tempering_target <= 1.0:
        raise ValueError("`tempering_target` must be a fraction in (0, 1].")
    season_start_month = (
        None if cfg.get("season_start_month") is None else int(cfg["season_start_month"])
    )
    if season_start_month is not None and not 1 <= season_start_month <= 12:
        raise ValueError("`season_start_month` must be a month between 1 and 12.")

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
    schaake_shuffle_config = SchaakeShuffleConfig.from_config(
        cfg.get("schaake_shuffle"),
        Path(cfg["_meta"]["config_dir"]),
        modeled_var,
    )
    if obs_var == "fsca":
        z_var: float | None = float(cfg.get("z_var", 1.0))
        swe_relative_uncertainty = None
        swe_sigma_floor = None
    else:
        z_var = None
        swe_relative_uncertainty = _positive_swe_setting(cfg, "swe_relative_uncertainty")
        swe_sigma_floor = _positive_swe_setting(cfg, "swe_sigma_floor")
    da_engine = Snow5DAEngine(
        particles=particles,
        ens_labels=ens_labels,
        particle_filter=pf_core,
        assimilation_times=assimilation_times,
        assimilation_var=obs_var,
        modeled_var=modeled_var,
        observation_da=observation,
        z_var=z_var,
        n_eff_threshold=n_eff_threshold,
        swe_relative_uncertainty=swe_relative_uncertainty,
        swe_sigma_floor=swe_sigma_floor,
        fsca_model=fsca_model,
        helbig_params=helbig_params,
        helbig_settings=helbig_settings,
        snow_density=snow_density,
        pf_resample=pf_resample,
        state_mean_name=state_mean_name,
        state_var_name=state_var_name,
        obs_var_name=obs_var_name,
        schaake_shuffle_config=schaake_shuffle_config,
        pf_seed=pf_seed,
        tempering_target=tempering_target,
        season_start_month=season_start_month,
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
        "timestamp_utc": utc_timestamp(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "ensemble_forcing_files": [str(p) for p in forcing_paths],
        "assimilation_times": [t.isoformat() for t in assimilation_times],
        "outputs": [str(p) for p in outputs],
        "pf_resample": pf_resample,
        "n_eff_threshold": n_eff_threshold,
        "pf_seed": pf_seed,
        "tempering_target": tempering_target,
        "season_start_month": season_start_month,
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
                "n_eff": record["n_eff"],
                "tempering_beta": record["tempering_beta"],
                "updated": record.get("updated", True),
                "weights": {
                    str(k): float(v) for k, v in record["weights"].iloc[0].items()
                },
                "prior_weights": {
                    str(k): float(v) for k, v in record["prior_weights"].iloc[0].items()
                },
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
