# Ross snow5 historic modes vs snow surveys

Runs snow5 for WY1982–WY2024 on the Ross lumped basin and scores it against the
Yukon snow surveys at Twin Creeks (09BA-SC02) in
`data/obs_qm/ross_yukon_swe_model_equivalent.nc`.

| Config | Mode |
|---|---|
| `ross_snow5_calibration.yaml` | split-sample parameter calibration (run first) |
| `ross_snow5_obs_historic_mean.yaml` | deterministic, SUMMA 50-member mean forcing |
| `ross_snow5_obs_historic_ensemble.yaml` | 50-member open loop |
| `ross_snow5_obs_sir.yaml` | SIR, updates on each survey date |
| `ross_snow5_obs_sis.yaml` | SIS, updates on each survey date, weights reset each Oct |

All modes use parameters calibrated to raw station SWE (`swe_obs`). The DA
runs assimilate `swe_obs` with `assimilation_schedule: observations`, which
updates on every date with a finite observation and then runs on to `end_time`. The
station→HRU pairing is set by `hru_stations`. Obs error is 15% of SWE, with a
floor of 35 kg m-2.

Outputs go to `outputs/snow5/ross_obs/<mode>/<timestamp>/`. Figures and
`scores.csv` come from:

```bash
python scripts/snow5/plot_ross_obs.py
```

## Reading the scores

- **prior**: the DA estimate on a survey date *before* that survey is
  assimilated, using only earlier surveys from the same season. This is the
  independent skill measure.
- **analysis**: the estimate after assimilating that survey. It is a fit to
  the assimilated data, not skill.
- Scores are computed against both raw station `swe_obs` (assimilated) and
  `swe_model_equivalent` (reference only). `scores.csv` splits them into all,
  calibration (WY1982–2003) and validation (WY2004–2024) periods.

## Calibration

```bash
python -m snowda.workflows.calibrate_snow5 runs/snow5/ross_obs/ross_snow5_calibration.yaml
```

- **Setup:** differential evolution minimising RMSE against `swe_obs` (zero-SWE
  surveys included). Forcing is the SUMMA ensemble mean. Calibration covers
  WY1982–2003 and validation WY2004–2024.
- **Melt-factor parameters:** with the multiplicative melt factor, only
  f_max² − f_min² matters, so `f_min` is fixed at 0.1.
- **Result:** guc 0.713, src 2.51, f_max 3.80, t_mlt −1.17.

| Parameters | Calibration RMSE / NSE | Validation RMSE / NSE |
|---|---|---|
| Initial (Chena set) | 111 / −1.51 | 79 / −0.79 |
| Calibrated | 25 / 0.87 | 65 / −0.22 |

Validation is poor because the forcing loses year-to-year skill at this station.
The correlation between Oct–Mar forcing precipitation and Mar–Apr station SWE
falls from 0.79 (1982–2003) to 0.21 (2004–2024). No parameter set fixes that;
DA does part of it (see `scores.csv`, period = validation).

The pre-calibration figures (scored against model-equivalent SWE) are kept in
`outputs/snow5/ross_obs/figures_uncalibrated_model_equivalent/`.

## Caveats

- `swe_model_equivalent` is quantile-mapped to **SUMMA's** SWE climatology, not
  snow5's, so it is no longer assimilated. It is still scored for reference.
- The melt factor is the multiplicative form (see `laugh_tests/test_snow5.py`).
