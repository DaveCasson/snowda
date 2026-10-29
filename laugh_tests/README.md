# Laugh tests

A laugh test is a case small enough that the answer is known before the code runs: a closed form, a limit, or a worked example printed in a paper. Each check records the reference, the expected value, the computed value, and the tolerance.

These tests sit beside the package. They do not replace `tests/`, and they do not change `src/snowda/`. Where the code disagrees with the textbook result, the test still expects the textbook result and is marked `xfail(strict=True)`. The report calls that row **KNOWN DEVIATION**. If the code is later changed so the textbook result holds, that row fails the suite until the deviation note is removed.

## Run

From the repository root:

```bash
python -m laugh_tests
```

`pytest laugh_tests` does the same thing. Plain `pytest` does not collect this directory (`testpaths` in `pyproject.toml` is `tests` only).

The run prints an expected-versus-computed list and writes:

- `laugh_tests/output/report.md`
- `laugh_tests/output/*.png`

That output directory is gitignored.

## How to read a row

| Status | Meaning |
| --- | --- |
| PASS | The code matches the analytical result within the stated tolerance. |
| KNOWN DEVIATION | The code does not match the textbook result. The numbers in the row are the comparison. The note says why. |
| FAIL | A result that should have matched did not. This is a new problem. |

## References

Checked against the publisher record while this suite was written, unless the line says otherwise.

**Particle filter**

- Gordon, Salmond, and Smith (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proc. F* 140(2):107–113. doi:10.1049/ip-f-2.1993.0015. Weights are normalised likelihoods.
- Arulampalam, Maskell, Gordon, and Clapp (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. *IEEE Trans. Signal Process.* 50(2):174–188. doi:10.1109/78.978374. Defines N_eff = 1 / sum(w_i^2), and resamples when N_eff falls below a threshold. The likelihood in that tutorial is Gaussian.
- Kitagawa (1996). Monte Carlo filter and smoother for non-Gaussian nonlinear state space models. *J. Comput. Graph. Stat.* 5(1):1–25. doi:10.1080/10618600.1996.10474692. Systematic resampling uses one random offset and then strides of 1/N.
- Kong, Liu, and Wong (1994). Sequential imputations and Bayesian missing data problems. *J. Amer. Statist. Assoc.* 89:278–288. The usual source of the effective-sample-size formula. Not re-checked against the journal page for this suite.
- Snow data-assimilation applications, for context only, not as the specification of this code: Leisenring and Moradkhani (2011), *Stoch. Environ. Res. Risk Assess.* 25:253–270; Moradkhani, Hsu, Gupta, and Sorooshian (2005), *Water Resour. Res.* 41:W05012. Not re-checked for this suite. The SWE likelihood in `particle_filter.py` is the Gaussian in Arulampalam et al. (2002), with `sigma = max(swe_relative_uncertainty * |observed SWE|, swe_sigma_floor)`. The documented settings are a 10% standard deviation and a floor of 5 kg m-2. fSCA still uses a scalar variance.

**Schaake Shuffle**

- Clark, Gangopadhyay, Hay, Rajagopalan, and Wilby (2004). The Schaake Shuffle: a method for reconstructing space–time variability in forecasted precipitation and temperature fields. *J. Hydrometeorol.* 5(1):243–262. doi:10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2. Section 3b, equations 5–12, prints a full 10-member example.
- Schefzik, Thorarinsdottir, and Gneiting (2013). Uncertainty quantification in complex simulation models using ensemble copula coupling. *Statist. Sci.* 28(4):616–640. Rank-dependence framing. Not re-checked for this suite. `docs/SCHAKE_SHUFFLE.md` does not cite a paper.

**Empirical quantile mapping**

- Panofsky and Brier (1968). *Some Applications of Statistics to Meteorology*. Pennsylvania State University Press. The empirical quantile–quantile correction. A book; no DOI was fetched.
- Boé, Terray, Habets, and Martin (2007). Statistical and dynamical downscaling of the Seine basin climate for hydro-meteorological studies. *Int. J. Climatol.* 27:1643–1655. doi:10.1002/joc.1602. Empirical percentiles with linear interpolation.
- Gudmundsson, Bremnes, Haugen, and Engen-Skaugen (2012). Technical Note: Downscaling RCM precipitation to the station scale using statistical transformations — a comparison of methods. *Hydrol. Earth Syst. Sci.* 16:3383–3390. doi:10.5194/hess-16-3383-2012. The QUANT method: empirical CDFs, linear interpolation, no extrapolation past the calibration sample.
- Hazen (1914), *Trans. ASCE* 77, and Cunnane (1978), *J. Hydrol.* 37:205–222, for the plotting position p = (i − 0.5) / n. Not re-checked for this suite. The formula itself is written in `src/snowda/obs_qm/mapping.py`.

**Depletion curve and snow model**

- Liston (2004). Representing subgrid snow cover heterogeneities in regional and global models. *J. Climate* 17(6):1381–1397. doi:10.1175/1520-0442(2004)017<1381:RSSCHI>2.0.CO;2.
- Slater et al. (2013). *Adv. Water Resour.* 55:165–177. doi:10.1016/j.advwatres.2012.07.006. Cited from `src/snowda/core/snow5.py`.
- The additive SNOW-17 melt factor, (f_max + f_min) / 2 + (f_max − f_min) / 2 · sin(phase), is the standard temperature-index form. The NOAA document equation number was not re-checked for this suite.

## What each file checks

`test_particle_filter.py` — SWE Gaussian weights `exp(-0.5 * sum((error / sigma)^2))` from `update_particle_weights`, with `sigma = max(0.10 * |observed SWE|, 5 kg m-2)` on each HRU. Checks include two HRUs with their own sigmas, a zero-SWE case that uses the floor, the floor binding only when the relative term is smaller, alignment by `hruId`, and equal particles. Also a hand-computed weighted mean and variance (1, 2, 3 with weights 0.5, 0.25, 0.25 gives mean 1.75 and variance 0.6875); N_eff for equal, one-hot, and two-way split weights; Kitagawa copy counts; resampling when N_eff is at or below the threshold, and no resampling of a uniform ensemble above it. A threshold in (0, 1] is a fraction of the ensemble. Figures: `particle_weights.png`, `particle_resample.png`.

`test_schaake_shuffle.py` — Clark et al. (2004) vectors X, Y, B, and X_ss exactly; marginals unchanged; ranks copied from the template; Spearman correlation copied, including +1 and −1 templates; shuffling a field by itself returns it; ties break in member order. Figures: `schaake_clark_example.png`, `schaake_rank_dependence.png`.

`test_quantile_mapping.py` — Hazen positions for n = 4; the middle of five points sits at probability 0.5; an identity map and a positive affine map are exact, including between knots; Exp(scale=1) quantiles map onto Exp(scale=2) by y* = 2y; with independent draws the error falls from n = 10 to 100 to 1000; the map is non-decreasing; values outside the sample clip to the sample ends. Figures: `quantile_mapping_transfer.png`, `quantile_mapping_convergence.png`.

`test_ssnowd.py` — the lognormal parameterised by `calc_lambda` and `calc_zeta` has mean mu; fSCA matches the scipy survival function; mean SWE matches a numerical integral of (x − dm) f(x); no melt leaves fSCA = 1 and mean = mu; fSCA(z) + fSCA(−z) = 1; CV = 0.001 behaves as a step at mu; Brent recovers a chosen melt depth. Figures: `ssnowd_depletion.png`, `ssnowd_step_and_inversion.png`.

`test_snow5.py` — snow only when T < src; no melt at or below t_mlt; melt cannot exceed SWE; a cold day stores all snowfall (20 + 5 = 25); one step conserves SWE. One strict deviation: equal melt bounds.

## Known deviations

These were confirmed by running the code. Production code was left as it is.

1. **Equal melt bounds produce no melt.** SNOW-17 uses Mf = (f_max + f_min) / 2 + (f_max − f_min) / 2 · sin(phase), which is the constant f when f_max = f_min, and which stays at or above f_min. `compute_melt_factor` in `src/snowda/core/snow5.py` multiplies those pieces (`mean * sin * diff`) and clips negatives to zero. With f_max = f_min = 2 the factor is 0 on every day, so the model cannot melt. This matches the legacy port in `legacy_code/snow_dist/snow_5_param_model.py`; it is not a transcription slip in the new module. A cold day (T below both the rain/snow threshold and the melt threshold) does not use the factor, and that case passes.

## Other opportunities

Not built in this suite:

- DeBeer depletion (`src/snowda/core/debeer.py`): no melt should leave fSCA = 1, the exceedance should match a lognormal survival, and fSCA should fall as melt rises.
- Helbig fSCA (`src/snowda/core/helbig.py`): on flat terrain, fSCA = tanh(1.3 · HS_min / HS_max^0.84); HS_min = 0 gives 0, and a deep snowpack gives 1.
- ESP sampling (`src/snowda/core/esp_sampling.py`): balanced draws hit floor(N) or ceil(N) each, and systematic draws sit within one count of N · w. `tests/test_esp_sampling.py` already covers part of this.
- Verification scores (`src/snowda/adapters/scores.py`): CRPS of a large Gaussian ensemble against the Gneiting closed form, and NSE = KGE = 1 for a perfect forecast. `tests/test_evaluate_streamflow.py` already checks the perfect-forecast scores.
