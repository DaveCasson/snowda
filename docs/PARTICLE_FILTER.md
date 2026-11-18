# Particle filter

The particle filter in `snowda/core/particle_filter.py` updates an ensemble of model states with
dated snow observations: model-equivalent SWE for SUMMA (`assimilation_mode: swe`), and SWE or
fSCA for Snow5. A particle is one ensemble member's state. For SUMMA it is the SUMMA restart and
the mizuRoute restart from that same member, stored and copied together. Each particle keeps its
own forcing. All maths is in `core/` and has no file I/O. The workflows
(`run_summa_da`, `run_summa_historic_da`, `run_snow5_da`) read observations, call the core, and
copy restarts.

This note explains how weights are computed and carried, when and how particles are resampled,
how tempering limits collapse, and which settings and diagnostics control each step.

## 1. Weights

### Likelihood

On each assimilation date, every particle `i` is scored against the observations `z` on the
observed HRUs. The likelihood is an independent Gaussian:

```
log L_i = -0.5 * sum_over_HRUs ((x_i - z) / sigma)^2
```

- SWE: `sigma = max(swe_relative_uncertainty * |z|, swe_sigma_floor)` on each HRU, in kg m-2.
  The documented settings are 10 % and 5 kg m-2.
- fSCA (SUMMA): `sigma = fsca_sigma`, one standard deviation in 0–1 fSCA units. It is required in
  `fsca_only` mode. Snow5 uses `z_var`, a variance.

A particle with a non-finite prediction on an observed HRU gets `log L = -inf`, so its weight is 0.

### Weights carry forward between dates

With the model itself as the proposal (the bootstrap filter), the weight recursion is

```
w_k  ∝  w_{k-1} * L_k
```

so the weight after date `k` holds the information from every observation so far, not just the
latest one (Arulampalam et al. 2002, Eq. 63; Doucet and Johansen 2011). After resampling, the
ensemble is equally weighted, so the weights reset to `1/N` before the next date.
`ParticleFilterCore.next_prior(weights, resampled)` applies that rule. A date with no usable
observation leaves the weights unchanged.

For a single forecast issue date, sequential weighting without resampling gives the same weights
as weighting each particle once by all observations up to that date. That one-shot weighting is
the particle batch smoother of Margulis et al. (2015). A laugh test checks this equivalence.

Weights are computed in log space and shifted by their maximum before exponentiating, so very
small weights do not underflow to zero.

| Case | Weight after date k |
| --- | --- |
| `pf_scheme: sis` | `w_{k-1} * L_k`, normalised |
| `pf_scheme: sir`, no resample on the previous date | `w_{k-1} * L_k`, normalised |
| `pf_scheme: sir`, resampled on the previous date | `(1/N) * L_k`, normalised |

The assumption behind multiplying likelihoods is that observation errors are independent between
dates. `_assign_samples` gives each station sample to exactly one date, so no measurement is
counted twice. A station that is wrong in the same way every month (a representativeness error)
breaks the assumption. The remedy is a larger `sigma` or tempering, not dropping earlier weights.

## 2. Effective sample size and resampling

```
N_eff = 1 / sum(w_i^2)
```

`N_eff` equals `N` for equal weights and 1 when one particle holds all the weight. It is not
rounded.

Under `pf_scheme: sir`, the ensemble is resampled when `N_eff <= n_eff_threshold`. A threshold in
`(0, 1]` is a fraction of the ensemble, a value above 1 is a count, and 0 never resamples. The
default is 0.5 for SUMMA and Snow5.

Resampling is systematic (Kitagawa 1996): one random offset, then strides of `1/N` through the
cumulative weights. Particle `i` is copied `floor(N w_i)` or `ceil(N w_i)` times. The donor indices
come back in ascending order, and member `m` takes donor `m`'s SUMMA and mizuRoute restarts.
ESP sampling uses the same function (`systematic_resample`) and then shuffles.

The random offset comes from `pf_seed` and the assimilation date
(`numpy.random.SeedSequence([pf_seed, YYYYMMDDHH])`). A rerun with the same seed copies the same
donors, whichever other dates were assimilated. With `pf_seed` unset, each run draws fresh.

## 3. Tempering

### Why

With one point station standing in for a lumped basin, a tight `sigma` and 50 members, a single
survey can put nearly all the weight on a few particles. Resampling then fills the ensemble with
copies of those few, and the spread that ESP needs is gone. Cluzet et al. (2021) met the same
problem in CrocO, the snowpack particle filter at Météo-France, and inflated the observation
errors iteratively until enough particles survived. Aalstad, Alonso-González et al. (2026) take a
different route in AdaPBS (in MuSA): they rerun the model with new perturbation-parameter proposals
and, between iterations, cap the largest weights so that a set fraction of particles survives.
That needs extra model runs, so it is not what snowda does here.

### What the code does

With `tempering_target` set to a fraction `t` in `(0, 1]`, each update uses

```
w_k  ∝  w_{k-1} * L_k^beta
```

where `beta` is the largest value in `[0, 1]` that keeps

```
N_eff(posterior)  >=  t * N_eff(prior)
```

If the full update already meets the target, `beta = 1` and nothing changes. Otherwise `beta` is
found by bisection to within `1e-6`. The target is relative to the prior `N_eff`, so it is always
reachable. `beta -> 0` returns the prior. Particles with a non-finite prediction are left out of
the prior `N_eff` and keep zero weight for every `beta > 0`.

For a Gaussian likelihood,

```
L^beta = exp(-0.5 * r^2 / sigma^2)^beta = exp(-0.5 * r^2 / (sigma / sqrt(beta))^2)
```

so tempering is the same as inflating `sigma` by `1 / sqrt(beta)` on that date. `sigma / sqrt(beta)`
is the observation error the filter would have needed to keep `t` of its effective members.

### Example

Fifty synthetic particles drawn from 150 ± 40 kg m-2, one observation of 110 kg m-2,
`sigma = 11`:

| `tempering_target` | `beta` | `sigma / sqrt(beta)` | `N_eff` | Weighted mean |
| --- | --- | --- | --- | --- |
| unset | 1.00 | 11.0 | 9.7 / 50 | 114.3 |
| 0.3 | 0.50 | 15.5 | 15.0 / 50 | 118.4 |
| 0.5 | 0.22 | 23.6 | 25.0 / 50 | 125.5 |

A higher target keeps more particles and pulls less toward the observation.

### Choosing a value

- Start below `n_eff_threshold`, for example `tempering_target: 0.3` with `n_eff_threshold: 0.5`.
  Tempering then stops a single date from collapsing the ensemble, and SIR still resamples.
  With `tempering_target >= n_eff_threshold`, a date that starts from equal weights will seldom,
  if ever, resample.
- Read `tempering_beta` in the log and provenance. A `beta` close to 1 means `sigma` was
  realistic. A `beta` that stays well below 1 means the observation error model is too tight,
  most likely because it leaves out representativeness and timing error. Fix the error model
  rather than relying on tempering.
- Tempering is a calibration device, not an exact Bayesian update. `beta < 1` discards part of
  each observation. Report the target and the `beta` values with results.

## 4. Configuration

Top-level keys in the SUMMA pysummada YAML (Snow5 accepts `n_eff_threshold`, `pf_seed`,
`tempering_target` and `z_var` at its top level):

```yaml
pf_scheme: sir           # sis: weights only; sir: copy donor restarts when N_eff is low
n_eff_threshold: 0.5     # fraction of N in (0, 1], or a count above 1; 0 never resamples
pf_seed: 42              # optional; reproducible resampling per assimilation date
tempering_target: 0.3    # optional; fraction of prior N_eff to keep on each date
fsca_sigma: 0.1          # required for assimilation_mode: fsca_only

assimilation:
  swe_relative_uncertainty: 0.10   # one standard deviation, fraction of |observed SWE|
  swe_sigma_floor: 5               # kg m-2
```

## 5. Diagnostics

Each SUMMA update logs one line, for example (illustrative values):

```
PF update 2014-03-01: beta 0.504, N_eff 15.0 of 50, resampled True, 21 unique donors
```

- `swe_updates` in `provenance.json` records `n_eff`, `tempering_beta` and `resampled` for each
  SWE date. Run provenance records `pf_scheme`, `n_eff_threshold`, `pf_seed` and
  `tempering_target`.
- `assimilation/<date>_swe_weights.csv` holds the posterior weights after carry-forward and
  tempering.
- `assimilation/sis_weighted_streamflow.csv` (`pf_scheme: sis` with ensemble mizuRoute, SWE
  mode) holds the filtered streamflow: the weighted mean and 5/50/95 % weighted quantiles of the
  members' routed flow. Each segment uses the weights known at its start (equal before the first
  update). SIS never changes member traces, so this is the SIS streamflow estimate. The reach and
  variable come from `evaluation` (`seg`, `reach_id`, `variable`) when set.
- `assimilation/post_da_<YYYYMMDDHH>.json` binds each member to its donor's restarts. The number
  of distinct donors there shows how many particles survived resampling.
- Snow5 logs `beta` and `N_eff` and records both for each assimilation time.

## 6. Tests

`laugh_tests/test_particle_filter.py` checks the filter against results known in advance:

- Gaussian weights for one and two HRUs, the `sigma` floor, and alignment by `hruId`.
- Carry-forward: two dates give `w ∝ L1 * L2`; three sequential dates equal one batch-smoother
  weighting; resampling resets to `1/N`; a zero prior stays zero.
- `N_eff` for equal, one-hot and split weights, and no rounding (2.38 stays above 2).
- Kitagawa copy counts, sorted donors, and identical donors from the same seed and date.
- Tempering: two equal-prior particles with `log L = 0` and `-2` and a target `N_eff` of 1.8 give
  `beta = ln 2 / 2` and weights 2/3 and 1/3. Tempered weights equal the Gaussian with
  `sigma / sqrt(beta)` and reach the target `N_eff`. An easy update keeps `beta = 1`, and an
  invalid particle stays at zero.

Run them with `python -m laugh_tests`.

## 7. Known limits

- Snow5's Schaake shuffle reorders state values across members after weighting. With unequal
  carried weights, a member's weight no longer describes its state. Use the shuffle only with an
  equally weighted (resampled) ensemble until this is resolved.
- Resampling copies states exactly. Diversity after resampling comes only from each member's own
  forcing. Perturbing precipitation per particle (Magnusson et al. 2017) with Liu–West jitter of the
  multiplier (Liu and West 2001) is the next step if impoverishment persists.
- Snow5 fSCA still defaults `z_var` to 1.0, which is too wide for a quantity in [0, 1].

## References

- Aalstad, K., Alonso-González, E., Pirk, N., Westermann, S., Willmes, C., and Yang, R. (2026).
  Evolving beyond collapse: an adaptive particle batch smoother for cryospheric data assimilation.
  *Geosci. Model Dev.* 19:8565.
- Arulampalam, M. S., Maskell, S., Gordon, N., and Clapp, T. (2002). A tutorial on particle filters
  for online nonlinear/non-Gaussian Bayesian tracking. *IEEE Trans. Signal Process.* 50(2):174–188.
  doi:10.1109/78.978374
- Cluzet, B., Lafaysse, M., Cosme, E., Albergel, C., Meunier, L.-F., and Dumont, M. (2021).
  CrocO_v1.0: a particle filter to assimilate snowpack observations in a spatialised framework.
  *Geosci. Model Dev.* 14:1595–1614. doi:10.5194/gmd-14-1595-2021
- Doucet, A., and Johansen, A. M. (2011). A tutorial on particle filtering and smoothing: fifteen
  years later. In *The Oxford Handbook of Nonlinear Filtering*. Oxford University Press.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian nonlinear state space
  models. *J. Comput. Graph. Stat.* 5(1):1–25. doi:10.1080/10618600.1996.10474692
- Liu, J., and West, M. (2001). Combined parameter and state estimation in simulation-based
  filtering. In *Sequential Monte Carlo Methods in Practice*. Springer.
  doi:10.1007/978-1-4757-3437-9_10
- Magnusson, J., Winstral, A., Stordal, A. S., Essery, R., and Jonas, T. (2017). Improving
  physically based snow simulations by assimilating snow depths using the particle filter.
  *Water Resour. Res.* 53(2):1125–1143. doi:10.1002/2016WR019092
- Margulis, S. A., Girotto, M., Cortés, G., and Durand, M. (2015). A particle batch smoother
  approach to snow water equivalent estimation. *J. Hydrometeorol.* 16(4):1752–1772.
  doi:10.1175/JHM-D-14-0177.1
