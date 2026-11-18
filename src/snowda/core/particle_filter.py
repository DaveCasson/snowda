from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from snowda.utils.logging import get_logger
from snowda.utils.timing import timer

log = get_logger(__name__)

# Resample when N_eff is at or below half the ensemble.
DEFAULT_N_EFF_THRESHOLD = 0.5
# Key in ``weights.attrs`` for the likelihood exponent used on that date.
TEMPERING_BETA = "tempering_beta"
# Bisection stops when the bracket on the tempering exponent is this narrow.
_TEMPERING_TOLERANCE = 1e-6


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


def _gaussian_log_likelihoods(
    particles: pd.DataFrame,
    observed: np.ndarray,
    hru_ids,
    sigma: np.ndarray,
) -> np.ndarray:
    """Independent-Gaussian log-likelihood of each particle, up to a constant.

    ``sigma`` is one standard deviation per HRU, in ``hru_ids`` order.
    ``log L_i = -0.5 * sum(((x_i - z) / sigma)^2)`` over HRUs with a finite
    observation. A particle with a non-finite prediction on one of those HRUs
    gets ``-inf``. Arulampalam et al. (2002).
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

    if not np.isfinite(log_likelihoods).any():
        raise ValueError("All particles have invalid predictions for observed HRUs")
    return log_likelihoods


def _prior_log_weights(
    prior_weights: Optional[pd.DataFrame], labels: list[str]
) -> np.ndarray:
    """Log of the previous normalised weights, in ``labels`` order.

    ``None`` is an equally weighted ensemble. ``prior_weights`` is the
    one-row frame returned by the previous update, or by
    ``ParticleFilterCore.next_prior``.
    """
    if prior_weights is None:
        return np.zeros(len(labels), dtype=float)
    row = prior_weights.iloc[0]
    missing = [label for label in labels if label not in row.index]
    if missing:
        raise ValueError(
            f"Prior weights have no column for member(s) {', '.join(missing)}"
        )
    values = row.reindex(labels).to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Prior weights must be finite, non-negative and not all zero")
    with np.errstate(divide="ignore"):
        return np.log(values / values.sum())


def _normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_weights)
    if not finite.any():
        raise ValueError("Every particle with a valid prediction has zero prior weight")
    weights = np.zeros_like(log_weights)
    weights[finite] = np.exp(log_weights[finite] - np.max(log_weights[finite]))
    return weights / weights.sum()


def _n_eff(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def _posterior_weights(
    prior_log_weights: np.ndarray,
    log_likelihoods: np.ndarray,
    tempering_target: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """``w_k ∝ w_{k-1} * L_k^beta``, normalised, and ``beta``.

    Sequential importance sampling with the model as the proposal
    (Arulampalam et al. 2002, Eq. 63; Doucet and Johansen 2011).
    Computed in log space and shifted by the maximum to avoid underflow.

    Without ``tempering_target``, ``beta = 1``. With it, ``beta`` is the
    largest value in [0, 1] whose posterior keeps
    ``N_eff >= tempering_target * N_eff(prior)``, found by bisection.
    Particles with a non-finite likelihood get zero weight for every
    ``beta > 0``, so the prior N_eff is taken over valid particles only.
    ``L^beta`` for a Gaussian is the same as inflating sigma by
    ``1 / sqrt(beta)``. This is the iterative observation-error inflation
    of Cluzet et al. (2021), done in one step.
    """
    valid = np.isfinite(log_likelihoods)

    def tempered(beta: float) -> np.ndarray:
        scaled = np.where(valid, beta * np.where(valid, log_likelihoods, 0.0), -np.inf)
        return _normalise_log_weights(prior_log_weights + scaled)

    full = tempered(1.0)
    if tempering_target is None:
        return full, 1.0
    if not np.isfinite(tempering_target) or not 0.0 < tempering_target <= 1.0:
        raise ValueError("tempering_target must be a fraction in (0, 1]")
    floor = tempering_target * _n_eff(tempered(0.0))
    if _n_eff(full) >= floor:
        return full, 1.0
    low, high = 0.0, 1.0
    while high - low > _TEMPERING_TOLERANCE:
        mid = 0.5 * (low + high)
        if _n_eff(tempered(mid)) >= floor:
            low = mid
        else:
            high = mid
    return tempered(low), low


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
    """Core particle filter utilities (no I/O).

    Weights are sequential: ``w_k ∝ w_{k-1} * L_k``. Pass the previous
    update's weights as ``prior_weights``, after ``next_prior``, which resets
    them to equal when the ensemble was resampled.
    """

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
        prior_weights: Optional[pd.DataFrame] = None,
        tempering_target: Optional[float] = None,
    ) -> pd.DataFrame:
        """Weight particles by a relative Gaussian SWE likelihood.

        ``relative_uncertainty`` is one standard deviation as a fraction of
        absolute observed SWE. ``sigma_floor`` is the absolute standard
        deviation in the same units as the observations (kg m-2). On each
        observed HRU, ``sigma = max(relative_uncertainty * |SWE|, sigma_floor)``.
        ``prior_weights`` are the weights before this observation; ``None``
        means equal weights. ``tempering_target`` is described in
        ``_posterior_weights``; the exponent used is in
        ``weights.attrs[TEMPERING_BETA]``.
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
        return self._update_weights(
            particles,
            swe_measurements,
            end_date_current_run,
            hru_index,
            ens_label_array,
            obs_column,
            lambda observed: _swe_observation_sigma(
                observed, relative_uncertainty, sigma_floor
            ),
            prior_weights,
            kind="SWE",
            tempering_target=tempering_target,
        )

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
        *,
        prior_weights: Optional[pd.DataFrame] = None,
        tempering_target: Optional[float] = None,
    ) -> pd.DataFrame:
        """Weight particles by a Gaussian fSCA likelihood with variance ``z_var``."""
        if not np.isfinite(z_var) or z_var <= 0:
            raise ValueError(
                "z_var must be a finite, positive observation-error variance"
            )
        return self._update_weights(
            particles,
            fsca_measurements,
            end_date_current_run,
            hru_index,
            ens_label_array,
            obs_column,
            lambda observed: np.full(observed.shape, np.sqrt(z_var), dtype=float),
            prior_weights,
            kind="fSCA",
            tempering_target=tempering_target,
        )

    @staticmethod
    def next_prior(weights: pd.DataFrame, resampled: bool) -> pd.DataFrame:
        """Prior weights for the next update.

        Resampling leaves an equally weighted ensemble, so the weights reset
        to 1/N (Arulampalam et al. 2002). Without it, the posterior carries
        forward.
        """
        if not resampled:
            return weights
        equal = np.full(weights.shape[1], 1.0 / weights.shape[1])
        return pd.DataFrame([equal], index=weights.index, columns=weights.columns)

    @staticmethod
    def _update_weights(
        particles: pd.DataFrame,
        measurements: pd.DataFrame,
        end_date_current_run,
        hru_index: pd.DataFrame,
        ens_label_array: list[str],
        obs_column: str,
        sigma_for,
        prior_weights: Optional[pd.DataFrame],
        *,
        kind: str,
        tempering_target: Optional[float],
    ) -> pd.DataFrame:
        if particles.shape[1] == 0:
            raise ValueError("particles must contain at least one ensemble member")
        if len(ens_label_array) != particles.shape[1]:
            raise ValueError(
                "ens_label_array length must match the number of particle columns"
            )

        z_df = measurements.loc[
            measurements.index.get_level_values("time") == end_date_current_run
        ]
        hru_ids = hru_index["hruId"]
        observed = _observations_on_hrus(z_df, obs_column, hru_ids)
        if not np.isfinite(observed).any():
            raise ValueError(f"No valid {kind} observations for {end_date_current_run}")

        log_likelihoods = _gaussian_log_likelihoods(
            particles, observed, hru_ids, sigma_for(observed)
        )
        weights, beta = _posterior_weights(
            _prior_log_weights(prior_weights, list(ens_label_array)),
            log_likelihoods,
            tempering_target,
        )
        log.info(
            "Particle weights calculated for %s (tempering beta %.4g)",
            end_date_current_run,
            beta,
        )
        frame = pd.DataFrame(
            [weights], index=[end_date_current_run], columns=ens_label_array
        )
        frame.attrs[TEMPERING_BETA] = beta
        return frame

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

    @staticmethod
    def effective_sample_size(weights: pd.DataFrame) -> float:
        """``N_eff = 1 / sum(w_i^2)`` for normalised weights (Arulampalam et al. 2002)."""
        values = weights.iloc[0].to_numpy(dtype=float)
        return float(1.0 / np.sum(np.square(values / values.sum())))

    def resample_indices(
        self,
        weights: pd.DataFrame,
        num_ens: int,
        n_eff_threshold: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[np.ndarray]:
        """Donor index for each member, or ``None`` when N_eff is above the cutoff.

        Arulampalam et al. (2002): resample when N_eff is at or below the
        threshold. A value in (0, 1] is a fraction of the ensemble, so 0.5
        means N/2. A value above 1 is a count. Zero never resamples.
        ``rng`` drives systematic resampling; see ``resampling_rng``.
        """
        threshold = float(n_eff_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            return None
        cutoff = threshold * num_ens if threshold <= 1.0 else threshold
        if self.effective_sample_size(weights) > cutoff:
            return None
        generator = rng if rng is not None else np.random.default_rng()
        return systematic_resample(
            weights.iloc[0].to_numpy(dtype=float), num_ens, generator
        )


def systematic_resample(
    weights: np.ndarray, n_draw: int, rng: np.random.Generator
) -> np.ndarray:
    """Kitagawa (1996) systematic resampling: one offset, then strides of 1/n.

    Returns ``n_draw`` donor indices in ascending order, so particle ``i`` is
    copied ``floor(n * w_i)`` or ``ceil(n * w_i)`` times.
    """
    if n_draw < 1:
        raise ValueError("n_draw must be >= 1")
    cdf = np.cumsum(np.asarray(weights, dtype=float))
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    positions = (float(rng.random()) + np.arange(n_draw, dtype=float)) / n_draw
    indices = np.searchsorted(cdf, positions, side="left")
    return np.clip(indices, 0, cdf.size - 1).astype(int)


def weighted_ensemble_summary(
    values: pd.DataFrame,
    weights: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Weighted mean and quantiles across members, one row per time.

    ``values`` has one column per member, in ``weights`` order. A quantile
    is the smallest member value whose cumulative weight reaches ``q``.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    data = values.to_numpy(dtype=float)
    order = np.argsort(data, axis=1)
    sorted_values = np.take_along_axis(data, order, axis=1)
    cdf = np.cumsum(w[order], axis=1)
    rows = np.arange(data.shape[0])
    summary = {"mean": data @ w}
    for q in quantiles:
        index = np.minimum((cdf < q).sum(axis=1), data.shape[1] - 1)
        summary[f"q{round(q * 100):02d}"] = sorted_values[rows, index]
    return pd.DataFrame(summary, index=values.index)


def resampling_rng(seed: Optional[int], when: datetime) -> np.random.Generator:
    """Resampling stream for one assimilation date.

    The same ``seed`` and date always give the same donors, independent of
    which other dates were assimilated. ``None`` draws fresh entropy.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(when.strftime("%Y%m%d%H"))])
    )
