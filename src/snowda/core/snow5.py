from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from snowda.core.fsca_models import (
    FscaState,
    compute_fsca_hru,
    initialize_model_arrays,
    initialize_output_arrays,
)


@dataclass
class Snow5Parameters:
    """Configuration needed to drive the 5-parameter snow model."""

    guc: float
    src: float
    f_max: float
    f_min: float
    t_mlt: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, float]) -> "Snow5Parameters":
        """Build parameters from a (possibly YAML-derived) mapping."""
        required = ("guc", "src", "f_max", "f_min", "t_mlt")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Missing snow_5 parameters: {missing}")
        return cls(
            guc=float(raw["guc"]),
            src=float(raw["src"]),
            f_max=float(raw["f_max"]),
            f_min=float(raw["f_min"]),
            t_mlt=float(raw["t_mlt"]),
        )


def _to_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype="float64")


def generate_snowfall(precip: Iterable[float], airtemp: Iterable[float], guc: float, src: float) -> np.ndarray:
    """Convert precipitation into snowfall when temperature is below the rain/snow transition."""
    precip_arr = _to_float_array(precip)
    airtemp_arr = _to_float_array(airtemp)
    snowfall = np.where(airtemp_arr < src, precip_arr * guc, 0.0)
    return snowfall


def phase_shift_day_of_year(doy: Iterable[float]) -> np.ndarray:
    """Apply the phase shift described in Slater et al. (2013)."""
    doy_arr = _to_float_array(doy)
    return doy_arr - (366.0 / 4.0) + 10.0


def compute_melt_factor(mdoy: Iterable[float], f_max: float, f_min: float) -> np.ndarray:
    """Seasonal melt factor derived from simple sinusoidal modulation."""
    mdoy_arr = _to_float_array(mdoy)
    mean_factor = (f_max + f_min) / 2.0
    diff_factor = (f_max - f_min) / 2.0
    sin_term = np.sin((2.0 * np.pi * mdoy_arr) / 366.0)
    melt_rate_factor = mean_factor * sin_term * diff_factor
    melt_rate_factor[melt_rate_factor < 0] = 0.0
    return melt_rate_factor


def compute_potential_melt(airtemp: Iterable[float], melt_factor: Iterable[float], t_mlt: float) -> np.ndarray:
    """Compute melt potential when air temperature exceeds the melt threshold."""
    airtemp_arr = _to_float_array(airtemp)
    melt_factor_arr = _to_float_array(melt_factor)
    temperature_excess = np.maximum(airtemp_arr - t_mlt, 0.0)
    return melt_factor_arr * temperature_excess


def melt_swe(swe: Iterable[float], melt_potential: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Melt SWE without allowing negative storage."""
    swe_arr = _to_float_array(swe)
    melt_potential_arr = _to_float_array(melt_potential)
    melt = np.minimum(swe_arr, melt_potential_arr)
    swe_after = swe_arr - melt
    return swe_after, melt


def run_snow5_timestep(
    pptrate: Iterable[float],
    airtemp: Iterable[float],
    doy: Iterable[float],
    swe: Iterable[float],
    params: Snow5Parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step the 5-parameter snow model for a single timestep."""
    snowfall = generate_snowfall(pptrate, airtemp, params.guc, params.src)
    swe_with_snow = _to_float_array(swe) + snowfall
    mdoy = phase_shift_day_of_year(doy)
    melt_factor = compute_melt_factor(mdoy, params.f_max, params.f_min)
    melt_potential = compute_potential_melt(airtemp, melt_factor, params.t_mlt)
    swe_after, melt = melt_swe(swe_with_snow, melt_potential)
    return snowfall, melt, melt_potential, swe_after


def _copy_model_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype="float64").copy() for key, value in arrays.items()}


def _copy_fsca_states(states: list[FscaState]) -> list[FscaState]:
    return [FscaState(**state.__dict__) for state in states]


def run_snow5_simulation(
    forcing_ds,
    params: Snow5Parameters,
    cv_array: np.ndarray,
    *,
    fsca_model: str = "ssnowd",
    helbig_settings: dict | None = None,
    helbig_params: dict[str, np.ndarray] | None = None,
    snow_density: float = 300.0,
    initial_model_arrays: dict[str, np.ndarray] | None = None,
    initial_states: list[FscaState] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[FscaState]]:
    """Simulate the snowpack for every HRU and timestep, updating fSCA each step."""
    timesteps = forcing_ds["time"].values
    hru_coord = forcing_ds["hru"].values
    n_hrus = len(hru_coord)
    if cv_array.shape != (n_hrus,):
        raise ValueError("cv_array must match number of HRUs in forcing dataset.")

    zeros_array = np.zeros_like(hru_coord, dtype="float64")
    base_model_arrays = initialize_model_arrays(zeros_array)
    if initial_model_arrays:
        model_arrays = _copy_model_arrays(initial_model_arrays)
        for key, value in base_model_arrays.items():
            model_arrays.setdefault(key, value)
    else:
        model_arrays = base_model_arrays
    output_arrays = initialize_output_arrays(timesteps, zeros_array)
    if initial_states:
        fsca_states = _copy_fsca_states(initial_states)
    else:
        fsca_states = [FscaState() for _ in range(n_hrus)]

    for ti in range(len(timesteps)):
        pptrate = forcing_ds["pptrate"].isel(time=ti).values
        airtemp = forcing_ds["airtemp"].isel(time=ti).values
        doy = forcing_ds["doy"].isel(time=ti).values
        (
            snowfall,
            melt_rate,
            _,
            swe_after,
        ) = run_snow5_timestep(pptrate, airtemp, doy, model_arrays["swe"], params)
        model_arrays["swe"] = swe_after
        model_arrays["snowfall"] = snowfall
        model_arrays["melt"] = melt_rate
        model_arrays["melt_flux"] = melt_rate
        model_arrays["snow_depth"] = model_arrays["swe"] / float(snow_density)

        # compute fSCA & update running totals per HRU
        for hru_idx in range(n_hrus):
            state = fsca_states[hru_idx]
            hru_inputs = {
                "swe": model_arrays["swe"][hru_idx],
                "snow_depth": model_arrays["snow_depth"][hru_idx],
                "snowfall": snowfall[hru_idx],
                "melt": melt_rate[hru_idx],
                "snowfall_total": state.snowfall_total,
                "melt_total": state.melt_total,
                "residual_new_snow": state.residual_new_snow,
            }
            helbig_cfg = dict(helbig_settings) if helbig_settings else {}
            if helbig_params and fsca_model == "helbig":
                helbig_cfg["slope_mu"] = float(helbig_params["mu"][hru_idx])
                helbig_cfg["xi"] = float(helbig_params["xi"][hru_idx])
            hru_values = compute_fsca_hru(
                hru_inputs,
                cv_array[hru_idx],
                fsca_model=fsca_model,
                state=state,
                helbig_settings=helbig_cfg if fsca_model == "helbig" else helbig_settings,
            )
            for label, value in hru_values.items():
                if label in model_arrays:
                    model_arrays[label][hru_idx] = value
                if label in output_arrays:
                    output_arrays[label][ti, hru_idx] = value
            output_arrays["snowfall"][ti, hru_idx] = snowfall[hru_idx]
            output_arrays["melt"][ti, hru_idx] = melt_rate[hru_idx]
            output_arrays["melt_flux"][ti, hru_idx] = melt_rate[hru_idx]
            output_arrays["swe"][ti, hru_idx] = model_arrays["swe"][hru_idx]
            output_arrays["snow_depth"][ti, hru_idx] = model_arrays["snow_depth"][hru_idx]

    return output_arrays, model_arrays, fsca_states
