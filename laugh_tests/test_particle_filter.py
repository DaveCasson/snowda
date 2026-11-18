"""Laugh tests for the particle filter.

References
----------
Gordon, Salmond, and Smith (1993), IEE Proc. F 140(2):107-113.
    doi:10.1049/ip-f-2.1993.0015
    Weights are normalised likelihoods. Resampling replaces the ensemble.

Arulampalam, Maskell, Gordon, and Clapp (2002), IEEE Trans. Signal Process.
    50(2):174-188. doi:10.1109/78.978374
    N_eff = 1 / sum(w_i^2). Resample when N_eff falls below a threshold.
    The usual likelihood is Gaussian, exp(-0.5 * (error / sigma)^2).
    For SWE, sigma = max(relative_uncertainty * |observation|, sigma_floor).
    With the model as the proposal, w_k ∝ w_{k-1} * p(z_k | x_k) (Eq. 63);
    resampled weights are reset to 1/N.

Doucet and Johansen (2011), A tutorial on particle filtering and smoothing:
    fifteen years later. In The Oxford Handbook of Nonlinear Filtering.
    The same recursion for sequential importance sampling.

Margulis, Girotto, Cortés, and Durand (2015), J. Hydrometeorol.
    16(4):1752-1772. doi:10.1175/JHM-D-14-0177.1
    Particle batch smoother: one weight per particle from all observations.

Kitagawa (1996), J. Comput. Graph. Stat. 5(1):1-25.
    doi:10.1080/10618600.1996.10474692
    Systematic resampling: one random offset, then strides of 1/N.
"""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laugh_tests._figures import save_figure
from snowda.core.particle_filter import ParticleFilterCore, resampling_rng

WHEN = datetime(2000, 3, 1)
_GAUSSIAN = (
    "Arulampalam et al. (2002): Gaussian likelihood "
    "exp(-0.5 * (error / sigma)^2). SWE sigma is "
    "max(relative_uncertainty * |observation|, sigma_floor)"
)
# Documented SWE settings: 10% is one standard deviation, and 5 kg m-2 is the floor.
RELATIVE = 0.10
SIGMA_FLOOR = 5.0
_NEFF = "Arulampalam et al. (2002): N_eff = 1 / sum(w_i^2)"
_KITAGAWA = (
    "Kitagawa (1996): systematic resampling copies particle i "
    "floor(N*w_i) or ceil(N*w_i) times"
)
_SIS = (
    "Arulampalam et al. (2002), Eq. 63; Doucet and Johansen (2011): "
    "w_k ∝ w_{k-1} * p(z_k | x_k) with the model as the proposal"
)
LABELS3 = ["001", "002", "003"]


def observation_sigma(
    observations: np.ndarray,
    relative_uncertainty: float,
    sigma_floor: float,
) -> np.ndarray:
    """``max(relative_uncertainty * |SWE|, sigma_floor)`` per observation."""
    observed = np.atleast_1d(np.asarray(observations, dtype=float))
    return np.maximum(relative_uncertainty * np.abs(observed), sigma_floor)


def gaussian_weights(error: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    """Independent Gaussian likelihood, normalised.

    ``error`` has shape (n_members,) or (n_members, n_obs). ``sigma`` is one
    standard deviation, either a scalar or one value per observation.
    The likelihood is exp(-0.5 * sum((error / sigma)^2)).
    This is the reference equation, not a copy of ``ParticleFilterCore``.
    """
    residuals = np.asarray(error, dtype=float)
    standardized = residuals / np.asarray(sigma, dtype=float)
    if standardized.ndim == 1:
        sse = np.square(standardized)
    else:
        sse = np.sum(np.square(standardized), axis=1)
    likelihood = np.exp(-0.5 * sse)
    return likelihood / likelihood.sum()


def particle_weights(
    member_values: np.ndarray,
    observations: np.ndarray,
    *,
    relative_uncertainty: float,
    sigma_floor: float,
    prior_weights: pd.DataFrame | None = None,
) -> np.ndarray:
    """Run ``update_particle_weights`` on one date.

    ``member_values`` has shape (n_members,) or (n_members, n_hru).
    ``observations`` is one value per HRU.
    """
    values = np.asarray(member_values, dtype=float)
    obs = np.atleast_1d(np.asarray(observations, dtype=float))
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    n_members, n_hru = values.shape
    if obs.shape != (n_hru,):
        raise ValueError("observations must have one value per HRU")
    labels = [f"{i + 1:03d}" for i in range(n_members)]
    hru_ids = [i + 1 for i in range(n_hru)]
    index = pd.MultiIndex.from_product([hru_ids, [WHEN]], names=["hru", "time"])
    particles = pd.DataFrame(
        {label: [values[m, h] for h in range(n_hru)] for m, label in enumerate(labels)},
        index=index,
    )
    measurements = pd.DataFrame({"hruId": hru_ids, "swe": obs}, index=index)
    hru_index = pd.DataFrame({"hruId": hru_ids})
    weights = ParticleFilterCore().update_particle_weights(
        particles,
        measurements,
        WHEN,
        hru_index,
        labels,
        "swe",
        relative_uncertainty=relative_uncertainty,
        sigma_floor=sigma_floor,
        prior_weights=prior_weights,
    )
    return weights.iloc[0].to_numpy(dtype=float)


def test_weights_match_the_gaussian_likelihood(laugh) -> None:
    """One HRU at 100 kg m-2. 10% of that is 10, above the floor of 5."""
    swe = np.array([100.0, 110.0, 120.0, 140.0])
    observation = 100.0
    sigma = observation_sigma(observation, RELATIVE, SIGMA_FLOOR)
    expected = gaussian_weights(swe - observation, sigma)
    computed = particle_weights(
        swe,
        observation,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
    )
    laugh.check(
        "normalised weights",
        expected,
        computed,
        rtol=1e-12,
        reference=(
            "Arulampalam et al. (2002): L = exp(-0.5 * (error / sigma)^2), "
            "then L/sum(L). sigma = 0.10 * |SWE| when that exceeds the floor. "
            "Gordon et al. (1993) normalise the likelihood"
        ),
    )
    laugh.check(
        "weights are finite",
        True,
        bool(np.all(np.isfinite(computed))),
        reference="A positive sigma keeps the Gaussian likelihood finite",
    )
    laugh.check(
        "weights sum to 1",
        1.0,
        float(computed.sum()),
        atol=1e-12,
        reference="Gordon, Salmond, and Smith (1993)",
    )
    laugh.check(
        "exact match has the largest weight",
        True,
        bool(np.argmax(computed) == 0),
        reference="A particle on the observation is the most likely",
    )


def test_two_hrus_multiply_independent_gaussians(laugh) -> None:
    """Each HRU keeps its own sigma. Both observations sit above the floor."""
    values = np.array(
        [
            [200.0, 100.0],
            [220.0, 90.0],
            [180.0, 110.0],
        ]
    )
    obs = np.array([200.0, 100.0])
    sigma = observation_sigma(obs, RELATIVE, SIGMA_FLOOR)
    expected = gaussian_weights(values - obs, sigma)
    computed = particle_weights(
        values,
        obs,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
    )
    laugh.check(
        "weights from each HRU's own sigma",
        expected,
        computed,
        rtol=1e-12,
        reference=(
            "Arulampalam et al. (2002): independent observations multiply, "
            "so the exponent sums ((x - z) / sigma)^2. "
            "sigma is 20 and 10 kg m-2 here, both above the floor of 5"
        ),
    )
    laugh.check(
        "weights are finite and sum to 1",
        True,
        bool(np.all(np.isfinite(computed)) and np.isclose(computed.sum(), 1.0)),
        reference="Gordon, Salmond, and Smith (1993)",
    )
    laugh.check(
        "closest particle has the largest weight",
        True,
        bool(np.argmax(computed) == 0),
        reference="A particle on both observations is the most likely",
    )


def test_zero_swe_uses_the_absolute_floor(laugh) -> None:
    """10% of zero is zero, so sigma is the floor and the weights stay finite."""
    swe = np.array([0.0, 5.0, 10.0])
    observation = 0.0
    expected = gaussian_weights(swe - observation, SIGMA_FLOOR)
    computed = particle_weights(
        swe,
        observation,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
    )
    laugh.check(
        "weights at observed SWE of 0",
        expected,
        computed,
        rtol=1e-12,
        reference=(
            "sigma = max(0.10 * |SWE|, swe_sigma_floor). "
            "At SWE = 0 the floor of 5 kg m-2 is the standard deviation"
        ),
    )
    laugh.check(
        "weights are finite",
        True,
        bool(np.all(np.isfinite(computed))),
        reference="The floor keeps sigma positive when observed SWE is zero",
    )
    laugh.check(
        "weights sum to 1",
        1.0,
        float(computed.sum()),
        atol=1e-12,
        reference="Gordon, Salmond, and Smith (1993)",
    )
    laugh.check(
        "particle on the observation has the largest weight",
        True,
        bool(np.argmax(computed) == 0),
        reference="A particle on the observation is the most likely",
    )


def test_floor_binds_only_below_the_relative_term(laugh) -> None:
    """20 kg m-2 gives 2 from the 10% term, so the floor of 5 is used."""
    low = 20.0
    particles = np.array([20.0, 25.0, 30.0])
    computed = particle_weights(
        particles,
        low,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
    )
    laugh.check(
        "weights when 0.10 * |SWE| is below the floor",
        gaussian_weights(particles - low, SIGMA_FLOOR),
        computed,
        rtol=1e-12,
        reference="sigma = max(0.10 * 20, 5) = 5 kg m-2",
    )
    unfloored = gaussian_weights(particles - low, RELATIVE * abs(low))
    laugh.check(
        "those weights differ from sigma = 0.10 * |SWE|",
        True,
        bool(np.max(np.abs(computed - unfloored)) > 0.1),
        reference="The floor replaces the relative term when the relative term is smaller",
    )

    high = 100.0
    above = np.array([100.0, 110.0, 120.0])
    computed_above = particle_weights(
        above,
        high,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
    )
    laugh.check(
        "weights when 0.10 * |SWE| is above the floor",
        gaussian_weights(above - high, RELATIVE * abs(high)),
        computed_above,
        rtol=1e-12,
        reference="sigma = max(0.10 * 100, 5) = 10 kg m-2",
    )


def test_weights_follow_hru_id_not_row_order(laugh) -> None:
    """Particle rows are HRU 2 then 1. Observation rows are HRU 1 then 2."""
    particle_index = pd.MultiIndex.from_tuples(
        [(2, WHEN), (1, WHEN)],
        names=["hru", "time"],
    )
    particles = pd.DataFrame(
        {"001": [10.0, 100.0], "002": [30.0, 100.0]},
        index=particle_index,
    )
    meas_index = pd.MultiIndex.from_tuples(
        [(1, WHEN), (2, WHEN)],
        names=["hru", "time"],
    )
    measurements = pd.DataFrame(
        {"hruId": [1, 2], "swe": [100.0, 10.0]},
        index=meas_index,
    )
    weights = ParticleFilterCore().update_particle_weights(
        particles,
        measurements,
        WHEN,
        pd.DataFrame({"hruId": [1, 2]}),
        ["001", "002"],
        "swe",
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
    )
    computed = weights.iloc[0].to_numpy(dtype=float)
    observed = np.array([100.0, 10.0])
    errors = np.array(
        [
            [0.0, 0.0],
            [0.0, 20.0],
        ]
    )
    expected = gaussian_weights(
        errors, observation_sigma(observed, RELATIVE, SIGMA_FLOOR)
    )
    laugh.check(
        "weights aligned by hruId",
        expected,
        computed,
        rtol=1e-12,
        reference="Observations and predictions are paired on hruId, not row position",
    )
    laugh.check(
        "matching particle has the larger weight",
        True,
        bool(computed[0] > computed[1]),
        reference="Member 001 matches both HRUs; member 002 misses HRU 2 by 20 kg m-2",
    )


def test_identical_particles_share_weight_equally(laugh) -> None:
    computed = particle_weights(
        np.array([5.0, 5.0, 5.0]),
        7.0,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
    )
    laugh.check(
        "equal weights",
        np.full(3, 1.0 / 3.0),
        computed,
        atol=1e-12,
        reference="Equal errors produce equal likelihoods",
    )


def test_weights_carry_forward_between_dates(laugh) -> None:
    """Two dates without resampling: the posterior is proportional to L1 * L2.

    Date 1 at 100 kg m-2 favours member 1. Date 2 at 200 kg m-2 favours
    member 3. Keeping only the date-2 likelihood would forget date 1.
    """
    pf = ParticleFilterCore()
    first = np.array([100.0, 110.0, 120.0])
    second = np.array([180.0, 190.0, 200.0])
    w1 = particle_weights(
        first, 100.0, relative_uncertainty=RELATIVE, sigma_floor=SIGMA_FLOOR
    )
    prior = pf.next_prior(pd.DataFrame([w1], columns=LABELS3), resampled=False)
    w2 = particle_weights(
        second,
        200.0,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
        prior_weights=prior,
    )
    sigma1 = observation_sigma(100.0, RELATIVE, SIGMA_FLOOR)
    sigma2 = observation_sigma(200.0, RELATIVE, SIGMA_FLOOR)
    l1 = np.exp(-0.5 * ((first - 100.0) / sigma1) ** 2)
    l2 = np.exp(-0.5 * ((second - 200.0) / sigma2) ** 2)
    expected = l1 * l2 / np.sum(l1 * l2)
    laugh.check(
        "weights after two dates",
        expected,
        w2,
        rtol=1e-12,
        reference=_SIS,
    )
    laugh.check(
        "those weights differ from the date-2 likelihood alone",
        True,
        bool(np.max(np.abs(w2 - l2 / l2.sum())) > 0.1),
        reference=_SIS,
    )


def test_sequential_weights_match_the_batch_smoother(laugh) -> None:
    """Three dates in sequence equal one weighting by all three observations.

    Margulis et al. (2015) weight each particle once by the joint likelihood
    of every observation in the window. With independent errors that is the
    product of the dated likelihoods, which sequential weighting reproduces.
    """
    obs = np.array([80.0, 150.0, 60.0])
    values = np.array(
        [
            [70.0, 140.0, 70.0],
            [85.0, 160.0, 55.0],
            [95.0, 150.0, 40.0],
            [80.0, 120.0, 60.0],
        ]
    )
    labels = [f"{i + 1:03d}" for i in range(values.shape[0])]
    prior = None
    for date in range(obs.size):
        weights = particle_weights(
            values[:, date],
            obs[date],
            relative_uncertainty=RELATIVE,
            sigma_floor=SIGMA_FLOOR,
            prior_weights=prior,
        )
        prior = ParticleFilterCore.next_prior(
            pd.DataFrame([weights], columns=labels), resampled=False
        )
    batch = gaussian_weights(
        values - obs, observation_sigma(obs, RELATIVE, SIGMA_FLOOR)
    )
    laugh.check(
        "sequential weights after three dates",
        batch,
        weights,
        rtol=1e-10,
        reference=(
            "Margulis et al. (2015) particle batch smoother: w ∝ p(y_1:K | x). "
            "Independent errors make this the product of dated likelihoods"
        ),
    )


def test_resampling_resets_the_prior(laugh) -> None:
    """After resampling, every particle starts the next date at 1/N."""
    weights = pd.DataFrame([[0.7, 0.2, 0.1]], columns=LABELS3)
    reset = ParticleFilterCore.next_prior(weights, resampled=True)
    kept = ParticleFilterCore.next_prior(weights, resampled=False)
    laugh.check(
        "prior after resampling",
        np.full(3, 1.0 / 3.0),
        reset.iloc[0].to_numpy(dtype=float),
        atol=1e-15,
        reference="Arulampalam et al. (2002): resampled weights are reset to 1/N",
    )
    laugh.check(
        "prior without resampling",
        np.array([0.7, 0.2, 0.1]),
        kept.iloc[0].to_numpy(dtype=float),
        exact=True,
        reference=_SIS,
    )
    swe = np.array([100.0, 110.0, 120.0])
    after_reset = particle_weights(
        swe,
        100.0,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
        prior_weights=reset,
    )
    laugh.check(
        "weights from an equal prior",
        particle_weights(
            swe, 100.0, relative_uncertainty=RELATIVE, sigma_floor=SIGMA_FLOOR
        ),
        after_reset,
        rtol=1e-12,
        reference="An equal prior leaves the normalised likelihood (SIR)",
    )


def test_zero_prior_weight_stays_zero(laugh) -> None:
    """A particle with no prior weight cannot regain weight."""
    prior = pd.DataFrame([[0.0, 0.5, 0.5]], columns=LABELS3)
    weights = particle_weights(
        np.array([100.0, 110.0, 120.0]),
        100.0,
        relative_uncertainty=RELATIVE,
        sigma_floor=SIGMA_FLOOR,
        prior_weights=prior,
    )
    laugh.check(
        "weight of the particle with prior 0",
        0.0,
        float(weights[0]),
        exact=True,
        reference=_SIS,
    )


def test_state_estimate_matches_weighted_moments(laugh) -> None:
    """Hand-computed weighted mean and population variance."""
    hru_id = 1
    index = pd.MultiIndex.from_tuples([(hru_id, WHEN)], names=["hru", "time"])
    particles = pd.DataFrame(
        {"a": [1.0], "b": [2.0], "c": [3.0]},
        index=index,
    )
    weights = pd.DataFrame([[0.5, 0.25, 0.25]], columns=["a", "b", "c"])
    mean_df, var_df = ParticleFilterCore().state_estimate(
        particles,
        weights,
        WHEN,
        pd.DataFrame({"hruId": [hru_id]}),
    )
    # mean = 1*0.5 + 2*0.25 + 3*0.25 = 1.75
    # var  = 0.5625*0.5 + 0.0625*0.25 + 1.5625*0.25 = 0.6875
    laugh.check(
        "weighted mean",
        1.75,
        float(mean_df.iloc[0, 0]),
        atol=1e-12,
        reference="Hand calculation of sum(w * x)",
    )
    laugh.check(
        "weighted population variance",
        0.6875,
        float(var_df.iloc[0, 0]),
        atol=1e-12,
        reference="Hand calculation of sum(w * (x - mean)^2), divisor 1",
    )

    equal = pd.DataFrame([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], columns=["a", "b", "c"])
    mean_eq, var_eq = ParticleFilterCore().state_estimate(
        particles,
        equal,
        WHEN,
        pd.DataFrame({"hruId": [hru_id]}),
    )
    laugh.check(
        "equal-weight mean",
        2.0,
        float(mean_eq.iloc[0, 0]),
        atol=1e-12,
        reference="Arithmetic mean of 1, 2, 3",
    )
    laugh.check(
        "equal-weight population variance",
        2.0 / 3.0,
        float(var_eq.iloc[0, 0]),
        rtol=1e-12,
        reference="Population variance of 1, 2, 3 is 2/3",
    )


def test_effective_sample_size(laugh) -> None:
    pf = ParticleFilterCore()
    cases = (
        ("equal weights", [0.25, 0.25, 0.25, 0.25], 4.0),
        ("one particle certain", [1.0, 0.0, 0.0, 0.0], 1.0),
        ("two particles share the weight", [0.5, 0.5, 0.0, 0.0], 2.0),
    )
    for label, weights, expected in cases:
        computed = pf.effective_sample_size(pd.DataFrame([weights]))
        laugh.check(label, expected, float(computed), atol=1e-12, reference=_NEFF)


def test_systematic_resample_counts(laugh) -> None:
    """N*w is an integer, so every offset copies each particle exactly N*w times."""
    weights = np.array([0.5, 0.25, 0.25])
    expected = np.array([2, 1, 1])
    pf = ParticleFilterCore()
    mismatches = 0
    last = None
    for seed in range(10):
        last = pf.resample_indices(
            pd.DataFrame([weights]),
            num_ens=4,
            n_eff_threshold=1.0,
            rng=np.random.default_rng(seed),
        )
        if last is None or not np.array_equal(np.bincount(last, minlength=3), expected):
            mismatches += 1
    laugh.check(
        "copy counts for weights 0.5, 0.25, 0.25",
        expected,
        np.bincount(last, minlength=3),
        exact=True,
        reference=_KITAGAWA,
    )
    laugh.check(
        "seeds that missed those counts",
        0,
        mismatches,
        exact=True,
        reference=_KITAGAWA,
    )


def test_certain_particle_is_copied_everywhere(laugh) -> None:
    indices = ParticleFilterCore().resample_indices(
        pd.DataFrame([[1.0, 0.0, 0.0, 0.0]]),
        num_ens=4,
        n_eff_threshold=1.0,
        rng=np.random.default_rng(0),
    )
    laugh.check(
        "indices when the first weight is 1",
        np.zeros(4, dtype=int),
        indices,
        exact=True,
        reference="A weight of 1 puts every copy on that particle",
    )


def test_relative_uncertainty_flattens_the_weights(laugh) -> None:
    """A larger fractional standard deviation spreads weight toward the outliers."""
    observation = 200.0
    swe = observation + np.array([0.0, 20.0, 40.0])
    tight = particle_weights(
        swe,
        observation,
        relative_uncertainty=0.10,
        sigma_floor=SIGMA_FLOOR,
    )
    wide = particle_weights(
        swe,
        observation,
        relative_uncertainty=0.50,
        sigma_floor=SIGMA_FLOOR,
    )
    laugh.check(
        "weights at 10% of 200 kg m-2",
        gaussian_weights(swe - observation, 0.10 * observation),
        tight,
        atol=1e-12,
        reference=_GAUSSIAN,
    )
    laugh.check(
        "weights at 50% of 200 kg m-2",
        gaussian_weights(swe - observation, 0.50 * observation),
        wide,
        atol=1e-12,
        reference=_GAUSSIAN,
    )
    laugh.check(
        "the exact particle keeps more weight when the uncertainty is smaller",
        True,
        bool(tight[0] > wide[0]),
        reference=_GAUSSIAN,
    )


def test_resample_when_effective_sample_size_is_low(laugh) -> None:
    indices = ParticleFilterCore().resample_indices(
        pd.DataFrame([[1.0, 0.0, 0.0, 0.0]]),
        num_ens=4,
        n_eff_threshold=2.0,
        rng=np.random.default_rng(0),
    )
    laugh.check(
        "indices when N_eff = 1 and the threshold is 2",
        np.zeros(4, dtype=int),
        indices,
        exact=True,
        reference="Arulampalam et al. (2002): resample when N_eff is at or below N_thr",
    )
    uniform = ParticleFilterCore().resample_indices(
        pd.DataFrame([[0.25, 0.25, 0.25, 0.25]]),
        num_ens=4,
        n_eff_threshold=2.0,
    )
    laugh.check(
        "indices when N_eff = 4 and the threshold is 2",
        None,
        uniform,
        exact=True,
        reference="A uniform ensemble is above the threshold, so it is kept",
    )


def test_effective_sample_size_is_not_rounded(laugh) -> None:
    """N_eff = 1 / (0.36 + 0.04 + 0.01 + 0.01) = 2.381, above a threshold of 2."""
    weights = pd.DataFrame([[0.6, 0.2, 0.1, 0.1]])
    pf = ParticleFilterCore()
    laugh.check(
        "N_eff for weights 0.6, 0.2, 0.1, 0.1",
        1.0 / 0.42,
        pf.effective_sample_size(weights),
        rtol=1e-12,
        reference=_NEFF,
    )
    laugh.check(
        "indices when N_eff = 2.38 and the threshold is 2",
        None,
        pf.resample_indices(
            weights, num_ens=4, n_eff_threshold=2.0, rng=np.random.default_rng(0)
        ),
        exact=True,
        reference="Rounding 2.38 down to 2 would resample; the unrounded value does not",
    )


def test_seeded_resampling_is_reproducible(laugh) -> None:
    """One stream per seed and date, so a rerun copies the same donors."""
    weights = pd.DataFrame([[0.4, 0.3, 0.2, 0.05, 0.05]])
    pf = ParticleFilterCore()

    def donors(seed: int) -> np.ndarray:
        return pf.resample_indices(
            weights, num_ens=5, n_eff_threshold=1.0, rng=resampling_rng(seed, WHEN)
        )

    first = donors(7)
    np.random.seed(123)  # the global NumPy state must not matter
    laugh.check(
        "donors from the same seed and date",
        first,
        donors(7),
        exact=True,
        reference="numpy.random.SeedSequence([seed, YYYYMMDDHH])",
    )
    laugh.check(
        "donors are sorted",
        np.sort(first),
        first,
        exact=True,
        reference=_KITAGAWA,
    )


def test_figure_resampling(laugh) -> None:
    """N_eff against a threshold of 2, and Kitagawa copy counts."""
    pf = ParticleFilterCore()
    cases = (
        ("equal weights", [0.25, 0.25, 0.25, 0.25], 4.0),
        ("two share", [0.5, 0.5, 0.0, 0.0], 2.0),
        ("one certain", [1.0, 0.0, 0.0, 0.0], 1.0),
    )
    labels = []
    computed_neff = []
    for label, weights, _expected in cases:
        labels.append(label)
        computed_neff.append(float(pf.effective_sample_size(pd.DataFrame([weights]))))
    weights = np.array([0.5, 0.25, 0.25])
    indices = pf.resample_indices(
        pd.DataFrame([weights]),
        num_ens=4,
        n_eff_threshold=1.0,
        rng=np.random.default_rng(0),
    )
    counts = np.bincount(indices, minlength=3)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    colors = ["C0" if value > 2.0 else "C3" for value in computed_neff]
    axes[0].bar(labels, computed_neff, color=colors, width=0.65)
    axes[0].axhline(2.0, color="0.25", ls="--", lw=1.0, label="threshold = 2")
    axes[0].set_ylabel("effective sample size")
    axes[0].set_ylim(0.0, 4.8)
    axes[0].set_title("Resample at or below the line")
    axes[0].legend(
        handles=[
            axes[0].get_lines()[0],
            plt.Rectangle((0, 0), 1, 1, color="C0", label="kept"),
            plt.Rectangle((0, 0), 1, 1, color="C3", label="resampled"),
        ],
        fontsize=8,
    )

    particles = np.arange(3)
    axes[1].bar(particles - 0.16, [2, 1, 1], width=0.32, label="4 × weight")
    axes[1].bar(particles + 0.16, counts, width=0.32, label="copies from the code")
    axes[1].set_xticks(particles, ["particle 1", "particle 2", "particle 3"])
    axes[1].set_xlabel("weights 1/2, 1/4, 1/4")
    axes[1].set_ylabel("copies in the new ensemble")
    axes[1].set_ylim(0.0, 3.2)
    axes[1].set_title("Systematic resampling")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "particle_resample.png")

    laugh.check(
        "effective sample sizes",
        np.array([case[2] for case in cases]),
        np.asarray(computed_neff),
        atol=1e-12,
        reference=_NEFF,
    )
    laugh.check(
        "copy counts for weights 0.5, 0.25, 0.25",
        np.array([2, 1, 1]),
        counts,
        exact=True,
        reference=_KITAGAWA,
    )


def test_figure_weights_versus_distance(laugh) -> None:
    """Markers are ParticleFilterCore. Solid curves are the relative Gaussian.

    The observation is 100 kg m-2, so 10%, 25%, and 50% are all above the
    floor of 5 kg m-2. Likelihoods are divided by the perfect particle.
    """
    observation = 100.0
    distances = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    relatives = (0.10, 0.25, 0.50)
    grid = np.linspace(0.0, 30.0, 400)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    largest_gap = 0.0
    for relative, color in zip(relatives, ("C0", "C1", "C2")):
        sigma = float(observation_sigma(observation, relative, SIGMA_FLOOR)[0])
        weights = particle_weights(
            observation + distances,
            observations=observation,
            relative_uncertainty=relative,
            sigma_floor=SIGMA_FLOOR,
        )
        scaled = weights / weights[0]
        curve = np.exp(-0.5 * np.square(distances / sigma))
        largest_gap = max(largest_gap, float(np.max(np.abs(scaled - curve))))
        ax.plot(
            grid,
            np.exp(-0.5 * np.square(grid / sigma)),
            color=color,
            label=f"{relative:.0%} of |SWE|, sigma={sigma:g}",
        )
        ax.scatter(distances, scaled, color=color, zorder=3, s=28)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel("|particle − observation| (kg m-2)")
    ax.set_ylabel("likelihood relative to a perfect particle")
    ax.set_title("SWE weights at an observation of 100 kg m-2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "particle_weights.png")
    laugh.check(
        "largest gap between the code and exp(-0.5 (error / sigma)^2)",
        0.0,
        largest_gap,
        atol=1e-12,
        reference=(
            "Arulampalam et al. (2002): Gaussian likelihood. "
            "sigma = max(relative_uncertainty * |observed SWE|, 5 kg m-2)"
        ),
    )
