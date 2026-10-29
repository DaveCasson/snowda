# snowda scientific review — subsystem map

Read this after the skill instructions when reviewing a named subsystem
or when you need equation-to-file mapping.

## 1. Snow5 (SWE evolution)

**Files:** `src/snowda/core/snow5.py`, `src/snowda/core/fsca_models.py`,
`src/snowda/workflows/run_snow5.py`, `src/snowda/workflows/run_snow5_ensemble.py`

**Reference port:** `legacy_code/snow_dist/snow_5_param_model.py`
(Slater et al. 2013)

**Algorithm steps to map:**

| Step | Function | Check |
|------|----------|--------|
| Rain/snow partition | `generate_snowfall` | `airtemp < src` → `precip * guc`, else 0 |
| DOY phase shift | `phase_shift_day_of_year` | `doy - 366/4 + 10` vs Slater |
| Melt factor | `compute_melt_factor` | sinusoid in `f_max`, `f_min`; negative clipped to 0 |
| Potential melt | `compute_potential_melt` | `(T - t_mlt)+ * melt_factor` |
| SWE update | `melt_swe` / `run_snow5_timestep` | SWE ≥ 0; snowfall added before melt |
| Depth | `swe / snow_density` | density default 300; units of SWE vs depth |

**Suggested independent checks:** T ≪ src and T ≫ src (all precip as
snow vs rain); T < t_mlt (zero melt); SWE=0 (no negative storage);
compare one-step melt factor to Slater Eq. form; conservation
`ΔSWE = snowfall − melt` per HRU.

**Existing tests:** none for Snow5 core.

## 2. fSCA operators

**Dispatcher:** `src/snowda/core/fsca_models.py` (`compute_fsca_hru`)

### ssnowd

**Files:** `src/snowda/core/ssnowd.py`

Lognormal SWE distribution; Brent inversion for cumulative melt depth
`dm` given target mean SWE. Guards: `cv ≥ 0.001`, `mu > 0`,
`swe_mean` clipped to `[0, mu]`.

**Checks:** `calc_lambda` / `calc_zeta` vs standard lognormal
`λ = ln μ − ζ²/2`, `ζ = √ln(1+CV²)`; residual sign and bracket;
fSCA ∈ [0, 1]; SWE=0 and SWE=μ limits; solver success (Brent).

**Existing tests:** none.

### debeer

**Files:** `src/snowda/core/debeer.py`

**Reference port:** `legacy_code/snow_dist/debeer.py` (DeBeer and
Pomeroy 2009; DeBeer 2012 eq. 3.9 cited in legacy)

Lognormal exceedance + linear reversion after new snowfall.
`new_snowfall_threshold` default 2 (units must match SWE/precip).

**Checks:** `current_swe = premelt − melt`; `k_min`, `k_y`, `erfc`
exceedance; accumulation totals; threshold units; fSCA bounds.

**Existing tests:** none.

### helbig

**Files:** `src/snowda/core/helbig.py`, `src/snowda/io/helbig_preprocess.py`

**Reference port:** `legacy_code/snow_dist/helbig_fsca.py` (Helbig et
al. 2021; comments also cite Helbig 2015, Egli and Jonas 2009)

Seasonal fSCA from snow-depth history, slope `μ`, correlation length
`ξ`, length scale `Ld`. Empirical exponents (0.3193, 0.1034, …).

**Checks:** buffer/history indexing (off-by-one in `diff_swe_buffer`
loop); SWE=0 reset; flat-slope branch `sd ∝ h_max^0.84`; DEM
preprocessing of `μ`, `σz`, `ξ` vs Helbig 2021; units of depth vs SWE.

**Existing tests:** none in `tests/`. Legacy has
`legacy_code/snow_dist/helbig_fsca_test.py` (oracle, not CI).

## 3. Particle filter

**Files:** `src/snowda/core/particle_filter.py`,
`src/snowda/core/assimilation.py`,
`src/snowda/workflows/run_snow5_da.py`,
`src/snowda/workflows/run_summa_da.py`

**Likelihood (implemented):** independent Gaussian product, Arulampalam
et al. (2002). SWE uses a per-HRU standard deviation. fSCA still uses
scalar variance `z_var`, passed in as `sigma = sqrt(z_var)`.

```
sigma_hru = max(swe_relative_uncertainty * |z_hru|, swe_sigma_floor)
log L_i = −0.5 * sum_hru ((x_i,hru − z_hru) / sigma_hru)^2
w = softmax(log L)
```

`swe_relative_uncertainty` is one standard deviation as a fraction of
absolute observed SWE. The documented config value is 0.10.
`swe_sigma_floor` is the absolute floor in kg m-2. The documented
config value is 5. Both are required config settings; the likelihood
does not hardcode them. The floor binds whenever the relative term is
smaller, including observed SWE of 0.

The sum is over HRUs with a finite observation. A particle with a
non-finite prediction on one of those HRUs gets weight 0.

**State estimate:** weighted mean and variance per HRU.

**Effective sample size:**
`round(1 / (N * mean(w²)))`, equal to `N_eff = 1 / ∑ w_i²` when the
weight row has length N.

**Resample threshold:** resample when `N_eff` is at or below the cutoff.
A threshold in `(0, 1]` is a fraction of the ensemble (`cutoff = threshold * N`).
A threshold above 1 is a count. Zero does not resample. `resample_indices`
returns `None` when `N_eff` is above the cutoff.

**Resampling:** residual/systematic style with
`zrand = U(0,1)/N` and 1-based then `− 1`. Check empty-slot
initialization and whether all particles are assigned.

**Suggested checks:** identical particles + identical obs → equal
weights; one exact match vs outliers; a larger relative uncertainty
flattens the weights; the floor binds when
`relative * |SWE|` is smaller and is the whole sigma at SWE = 0;
weight normalization ∑w=1; two HRUs each use their own sigma;
resample when `N_eff` is at or below the cutoff and keep a
uniform ensemble above it.

**Existing tests:** `laugh_tests/test_particle_filter.py` checks the
Gaussian weights, `N_eff`, and the resample threshold. Schaake DA
integration uses `ParticleFilterCore` as a collaborator.

## 4. Schaake Shuffle

**Files:** `src/snowda/schaake_shuffle/core.py`,
`archive.py`, `config.py`; applied in
`src/snowda/workflows/run_snow5_da.py`

**Docs:** `docs/SCHAKE_SHUFFLE.md`

**Algorithm:** sample `Y` (n_members × sites) from archive within
DOY ± `window_days` (wrap 366); jitter ties;
`B = argsort(Y)` per site; sort `X` and scatter into rank order of
`Y`; optionally `apply_permutation` to other states.

**Invariants:** per-site sorted `X` unchanged (marginals);
`B` is a permutation of `0..n-1` per column; same `B` on paired
states; reproducible given seed and tied `Y`.

**Checks:** archive units vs ensemble units; `replace=False` sampling;
DOY wraparound; NaN handling; jitter scale
`max(1, nanstd(Y)) * eps * (member + 1e-3*site)`.

**Existing tests:** `tests/test_schaake_shuffle.py` (marginals,
permutation validity, tie reproducibility, DA integration). These do
not prove archive climatology or scientific appropriateness of the
template variable.

## 5. ESP (SUMMA + mizuRoute)

**Files:** `src/snowda/workflows/esp.py`,
`src/snowda/adapters/pysumma.py`,
`src/snowda/adapters/mizuroute.py`,
`src/snowda/workflows/run_summa_esp.py`,
`src/snowda/workflows/run_summa_da.py`,
`src/snowda/workflows/run_snow5_da.py` (Snow5 ESP branch)

**Evaluation (not a score):** `scripts/evaluate_esp_streamflow.py`
(p10–p90 / p25–p75 envelopes, climatology overlay). No CRPS/reliability
implementation in-repo as of this map.

**Checks:** trigger-date resolution vs `start_time`/`end_time`;
leap-day `_replace_year`; trace-year forcing window alignment;
state file date parsing; mizuRoute `input_dir` = SUMMA trace dir;
no leakage of forecast-year forcing into traces; ensemble
independence of traces.

**Existing tests:** `tests/test_esp.py`, `tests/test_snow5_esp_helpers.py`,
`tests/test_pysummada_config.py` — config and date helpers, not
hydrologic skill.

## 6. Units, CF, adapters

**Files:** `DATA_CONVENTIONS.md`, `src/snowda/io/units.py`,
`src/snowda/io/snow5.py`, `src/snowda/io/summa_fsca.py`,
`src/snowda/io/fsca_obs.py`, `src/snowda/adapters/pysumma.py`,
`src/snowda/adapters/mizuroute.py`

**Documented vs code:** docs say SWE `kg m-2`; `INTERNAL_TARGETS_DEFAULT`
says SWE `millimeter`. Hard-coded factors in `core/` are a convention
violation and a likely unit bug.

**Checks:** Pint conversions logged; provenance `unit_conversions`;
SUMMA `scalarSWE` / `scalarSnowDepth` / `pptrate` units; fSCA
dimensionless; observation time alignment with model `end_date`.

## Existing automated tests (coverage, not proof)

| File | What it actually checks |
|------|-------------------------|
| `tests/test_schaake_shuffle.py` | Shuffle invariants + Snow5 DA wiring |
| `tests/test_esp.py` | ESP date/config helpers |
| `tests/test_snow5_esp_helpers.py` | Snow5 ESP helper behavior |
| `tests/test_pysummada_config.py` | Per-workflow ESP / DA / deterministic constraints |

No CI tests for Snow5 melt/SWE, ssnowd, DeBeer, Helbig, or PF
likelihood/resampling mathematics.

## Validation design notes

Derive expected values from the cited equation or from an independent
implementation (legacy port or closed form), not by copying the
function under test.

Meaningful tolerances (starting points, justify if you change them):

- Exact discrete invariants (permutations, SWE ≥ 0, ∑w = 1):
  `atol=0` or `np.array_equal` / `np.allclose(..., atol=1e-12)`.
- Lognormal / `erfc` identities: `rtol=1e-12` (special functions).
- Brent `dm` inversion: residual `|target − implied_mean|` vs solver
  `xtol`, not an arbitrary SWE percent.
- Ensemble rank shuffle: exact marginal sort match.
- Do not use plot similarity as a scientific pass.

Keep synthetics tiny: 1–3 HRUs, 2–10 members, a few daily steps.
Do not run full SUMMA/mizuRoute cases unless requested.
