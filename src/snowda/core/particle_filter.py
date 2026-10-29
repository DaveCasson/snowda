from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from snowda.utils.logging import get_logger
from snowda.utils.timing import timer

log = get_logger(__name__)


def _series_on_hrus(series: pd.Series, hru_ids) -> np.ndarray:
    """One value per HRU, in ``hru_ids`` order.

    Particle and observation frames are indexed by ``(hru, time)``. Selecting
    the ``hru`` level leaves a plain index that lines up with ``hruId``.
    """
    aligned = series
    if isinstance(aligned.index, pd.MultiIndex):
        if "hru" in aligned.index.names:
            level = "hru"
        elif "hruId" in aligned.index.names:
            level = "hruId"
        else:
            raise ValueError("index must contain an 'hru' or 'hruId' level")
        extra = [name for name in aligned.index.names if name != level]
        if extra:
            aligned = aligned.droplevel(extra)
    if aligned.index.has_duplicates:
        raise ValueError("Multiple values found for an HRU at this run date")
    return aligned.reindex(pd.Index(np.asarray(hru_ids))).to_numpy(dtype=float)


def _observations_on_hrus(frame: pd.DataFrame, obs_column: str, hru_ids) -> np.ndarray:
    """Observation values for ``hru_ids``.

    SUMMA SWE frames carry an ``hruId`` column. Snow5 and fSCA frames are
    indexed by ``hru`` instead.
    """
    if "hruId" in frame.columns:
        if frame["hruId"].duplicated().any():
            raise ValueError("Multiple observations found for an HRU at this run date")
        observed = frame.set_index("hruId")[obs_column]
        return observed.reindex(pd.Index(np.asarray(hru_ids))).to_numpy(dtype=float)
    return _series_on_hrus(frame[obs_column], hru_ids)


def _gaussian_weights(
    particles: pd.DataFrame,
    observed: np.ndarray,
    hru_ids,
    sigma: np.ndarray,
) -> np.ndarray:
    """Normalised independent-Gaussian weights.

    ``sigma`` is one standard deviation per HRU, in ``hru_ids`` order.
    ``log L_i = -0.5 * sum(((x_i - z) / sigma)^2)`` over HRUs with a finite
    observation. A particle with a non-finite prediction on one of those HRUs
    gets weight 0. Arulampalam et al. (2002).
    """
    valid_obs = np.isfinite(observed)
    if not valid_obs.any():
        raise ValueError("No finite observations at this run date")
    if np.any(~np.isfinite(sigma[valid_obs])) or np.any(sigma[valid_obs] <= 0):
        raise ValueError("sigma must be finite and positive on every observed HRU")

    log_likelihoods = np.full(particles.shape[1], -np.inf, dtype=float)
    for i in range(particles.shape[1]):
        predicted = _series_on_hrus(particles.iloc[:, i], hru_ids)
        residuals = predicted[valid_obs] - observed[valid_obs]
        if np.all(np.isfinite(residuals)):
            log_likelihoods[i] = -0.5 * np.sum((residuals / sigma[valid_obs]) ** 2)

    finite = np.isfinite(log_likelihoods)
    if not finite.any():
        raise ValueError("All particles have invalid predictions for observed HRUs")

    # Shift before exponentiating to avoid underflow.
    shifted = np.zeros_like(log_likelihoods)
    shifted[finite] = np.exp(log_likelihoods[finite] - np.max(log_likelihoods[finite]))
    return shifted / shifted.sum()


def _swe_observation_sigma(
    observed: np.ndarray,
    relative_uncertainty: float,
    sigma_floor: float,
) -> np.ndarray:
    """One standard deviation per HRU.

    ``relative_uncertainty`` is a fraction of absolute observed SWE.
    ``sigma_floor`` is an absolute floor in the same units as ``observed``.
    """
    return np.maximum(relative_uncertainty * np.abs(observed), sigma_floor)


@dataclass
class ParticleFilterCore:
    """Core particle filter utilities (no I/O)."""

    @timer
    def update_particle_weights(
        self,
        particles: pd.DataFrame,
        swe_measurements: pd.DataFrame,
        end_date_current_run,
        hru_index: pd.DataFrame,
        ens_label_array: list[str],
        obs_column: str,
        *,
        relative_uncertainty: float,
        sigma_floor: float,
    ) -> pd.DataFrame:
        """Weight particles by a relative Gaussian SWE likelihood.

        ``relative_uncertainty`` is one standard deviation as a fraction of
        absolute observed SWE. ``sigma_floor`` is the absolute standard
        deviation in the same units as the observations (kg m-2). On each
        observed HRU, ``sigma = max(relative_uncertainty * |SWE|, sigma_floor)``.
        """
        if not np.isfinite(relative_uncertainty) or relative_uncertainty <= 0:
            raise ValueError(
                "relative_uncertainty must be a finite fraction greater than 0 "
                "(one standard deviation, as a fraction of |observed SWE|)"
            )
        if not np.isfinite(sigma_floor) or sigma_floor <= 0:
            raise ValueError(
                "sigma_floor must be a finite absolute SWE standard deviation greater than 0"
            )

        if particles.shape[1] == 0:
            raise ValueError("particles must contain at least one ensemble member")

        if len(ens_label_array) != particles.shape[1]:
            raise ValueError(
                "ens_label_array length must match the number of particle columns"
            )

        z_df = swe_measurements.loc[
            swe_measurements.index.get_level_values("time") == end_date_current_run
        ]

        hru_ids = hru_index["hruId"]
        observed = _observations_on_hrus(z_df, obs_column, hru_ids)
        if not np.isfinite(observed).any():
            raise ValueError(f"No valid SWE observations for {end_date_current_run}")
        sigma = _swe_observation_sigma(observed, relative_uncertainty, sigma_floor)
        weights = _gaussian_weights(particles, observed, hru_ids, sigma)

        weights_df = pd.DataFrame(
            [weights],
            index=[end_date_current_run],
            columns=ens_label_array,
        )

        log.info("Particle weights calculated for %s", end_date_current_run)
        return weights_df

    @timer
    def update_particle_weights_fsca(
        self,
        particles: pd.DataFrame,
        fsca_measurements: pd.DataFrame,
        z_var: float,
        end_date_current_run,
        hru_index: pd.DataFrame,
        ens_label_array: list[str],
        obs_column: str,
    ) -> pd.DataFrame:
        if not np.isfinite(z_var) or z_var <= 0:
            raise ValueError("z_var must be a finite, positive observation-error variance")

        if particles.shape[1] == 0:
            raise ValueError("particles must contain at least one ensemble member")

        if len(ens_label_array) != particles.shape[1]:
            raise ValueError(
                "ens_label_array length must match the number of particle columns"
            )

        z_df = fsca_measurements.loc[
            fsca_measurements.index.get_level_values("time") == end_date_current_run
        ]

        hru_ids = hru_index["hruId"]
        observed = _observations_on_hrus(z_df, obs_column, hru_ids)
        if not np.isfinite(observed).any():
            raise ValueError(f"No valid fSCA observations for {end_date_current_run}")
        sigma = np.full(observed.shape, np.sqrt(z_var), dtype=float)
        weights = _gaussian_weights(particles, observed, hru_ids, sigma)

        weights_df = pd.DataFrame(
            [weights],
            index=[end_date_current_run],
            columns=ens_label_array,
        )

        log.info("Particle weights calculated for %s", end_date_current_run)
        return weights_df

    @timer
    def state_estimate(
        self,
        particles: pd.DataFrame,
        weights: pd.DataFrame,
        end_date_current_run,
        hru_index: pd.DataFrame,
        mean_name: str = "mean_state",
        var_name: str = "var_state",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mean = {}
        var = {}
        for hru_id in hru_index["hruId"]:
            mean[hru_id] = np.average(
                particles.loc[hru_id].values[0], weights=weights.values[0], axis=0
            )
            var[hru_id] = np.average(
                (particles.loc[hru_id].values[0] - mean[hru_id]) ** 2,
                weights=weights.values[0],
                axis=0,
            )

        mean_df = pd.DataFrame.from_dict(mean, orient="index")
        var_df = pd.DataFrame.from_dict(var, orient="index")

        mean_df["time"] = end_date_current_run
        mean_df.reset_index(inplace=True)
        mean_df.set_index(["index", "time"], inplace=True)
        mean_df.index.names = ["hru", "time"]
        mean_df.columns = [mean_name]

        var_df["time"] = end_date_current_run
        var_df.reset_index(inplace=True)
        var_df.set_index(["index", "time"], inplace=True)
        var_df.index.names = ["hru", "time"]
        var_df.columns = [var_name]

        return mean_df, var_df

    def efficient_weights(self, weights: pd.DataFrame, num_ens: int) -> float:
        weights_np = weights.iloc[0].to_numpy()
        return np.around(1.0 / (num_ens * np.mean([w**2 for w in weights_np])))

    def resample_indices(
        self, weights: pd.DataFrame, num_ens: int, n_eff_threshold: float
    ) -> Optional[np.ndarray]:
        weights_np = weights.iloc[0].to_numpy()
        effweights = self.efficient_weights(weights, num_ens)
        # Arulampalam et al. (2002): resample when N_eff falls to the threshold
        # or below. A value in (0, 1] is a fraction of the ensemble, so 0.5
        # means N/2. A value above 1 is a count. Zero never resamples.
        threshold = float(n_eff_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            return None
        cutoff = threshold * num_ens if threshold <= 1.0 else threshold
        if effweights > cutoff:
            return None

        resample = np.empty((num_ens), dtype=int)
        zweightcumul = np.cumsum(weights_np)
        zrand = np.random.random() / num_ens

        for ires in range(0, num_ens):
            if zrand <= zweightcumul[0]:
                resample[ires] = 1
            for rk in range(1, num_ens):
                if (zweightcumul[rk - 1] < zrand) and (zrand <= zweightcumul[rk]):
                    resample[ires] = rk + 1
            zrand += 1.0 / num_ens

        return resample - 1
