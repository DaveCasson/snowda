from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Optional

from snowda.core.debeer import debeer
from snowda.core.helbig import snow_cover_fraction
from snowda.core.ssnowd import ssnowd


def initialize_output_arrays(timesteps, zeros_array):
    n_timesteps = len(timesteps)
    n_hrus = len(zeros_array)
    return {
        "swe": np.empty((n_timesteps, n_hrus)),
        "melt": np.empty((n_timesteps, n_hrus)),
        "snowfall": np.empty((n_timesteps, n_hrus)),
        "snowfall_total": np.empty((n_timesteps, n_hrus)),
        "melt_total": np.empty((n_timesteps, n_hrus)),
        "melt_flux": np.empty((n_timesteps, n_hrus)),
        "fsca": np.empty((n_timesteps, n_hrus)),
        "residual_new_snow": np.empty((n_timesteps, n_hrus)),
        "snow_depth": np.empty((n_timesteps, n_hrus)),
    }


def initialize_model_arrays(zeros_array):
    n_hrus = len(zeros_array)
    return {
        "swe": np.zeros(n_hrus),
        "melt": np.zeros(n_hrus),
        "snowfall": np.zeros(n_hrus),
        "snowfall_total": np.zeros(n_hrus),
        "melt_total": np.zeros(n_hrus),
        "melt_flux": np.zeros(n_hrus),
        "fsca": np.zeros(n_hrus),
        "residual_new_snow": np.zeros(n_hrus),
        "snow_depth": np.zeros(n_hrus),
    }


@dataclass
class FscaState:
    snowfall_total: float = 0.0
    melt_total: float = 0.0
    fsca: float = 0.0
    residual_new_snow: float = 0.0
    swe_hist: Optional[list] = field(default=None)
    snow_depth_hist: Optional[list] = field(default=None)
    swe_min: float = 0.0
    swe_max: float = 0.0
    snow_depth_min: float = 0.0
    snow_depth_max: float = 0.0


def compute_fsca_hru(
    model_array,
    cv,
    *,
    fsca_model: str = "ssnowd",
    state: Optional[FscaState] = None,
    helbig_settings: Optional[dict] = None,
):
    cv = float(cv)
    state = state or FscaState()

    if model_array["swe"] <= 0.01:
        swe_output = 0
        state.snowfall_total = 0
        state.melt_total = 0
        state.fsca = 0
        state.residual_new_snow = 0
        return {
            "swe": swe_output,
            "snowfall": model_array["snowfall"],
            "melt": model_array["melt"],
            "snowfall_total": state.snowfall_total,
            "melt_total": state.melt_total,
            "fsca": state.fsca,
            "residual_new_snow": state.residual_new_snow,
        }

    if fsca_model == "ssnowd":
        fsca_output, swe_output, snowfall_total_output, melt_total_output, _, _ = ssnowd(
            model_array["snowfall"],
            model_array["melt"],
            state.snowfall_total,
            state.melt_total,
            model_array["swe"],
            state.residual_new_snow,
            cv,
        )
        state.snowfall_total = snowfall_total_output
        state.melt_total = melt_total_output
        state.fsca = fsca_output
        state.residual_new_snow = 0
        return {
            "swe": swe_output,
            "snowfall": model_array["snowfall"],
            "melt": model_array["melt"],
            "snowfall_total": state.snowfall_total,
            "melt_total": state.melt_total,
            "fsca": state.fsca,
            "residual_new_snow": state.residual_new_snow,
        }

    if fsca_model == "debeer":
        fsca_output, snowfall_total_output, melt_total_output, residual_new_snow = debeer(
            model_array["snowfall"],
            model_array["melt"],
            state.snowfall_total,
            state.melt_total,
            state.fsca,
            state.residual_new_snow,
            cv,
        )
        state.snowfall_total = snowfall_total_output
        state.melt_total = melt_total_output
        state.fsca = fsca_output
        state.residual_new_snow = residual_new_snow
        return {
            "snowfall_total": state.snowfall_total,
            "melt_total": state.melt_total,
            "fsca": state.fsca,
            "residual_new_snow": state.residual_new_snow,
        }

    if fsca_model == "helbig":
        if not helbig_settings:
            raise ValueError("helbig_settings are required for helbig fSCA model.")
        snow_depth = model_array.get("snow_depth")
        if snow_depth is None:
            raise ValueError("snow_depth is required for helbig fSCA model.")
        if state.swe_hist is None:
            state.swe_hist = [0.0] * (helbig_settings["buffer_length"] - 1)
        if state.snow_depth_hist is None:
            state.snow_depth_hist = [0.0] * (helbig_settings["buffer_length"] - 1)
        (
            fsca_output,
            state.snow_depth_min,
            state.snow_depth_max,
            state.snow_depth_hist,
            state.swe_min,
            state.swe_max,
            state.swe_hist,
            _coeff_vari,
        ) = snow_cover_fraction(
            snow_depth,
            model_array["swe"],
            state.swe_hist,
            state.swe_min,
            state.swe_max,
            state.snow_depth_hist,
            state.snow_depth_min,
            state.snow_depth_max,
            helbig_settings["slope_mu"],
            helbig_settings["xi"],
            helbig_settings["Ld"],
            helbig_settings["snow_depth_threshold"],
            helbig_settings["buffer_length"],
        )
        state.fsca = fsca_output
        return {
            "fsca": state.fsca,
        }

    raise ValueError(f"Unsupported fSCA model: {fsca_model}")
