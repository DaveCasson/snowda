# Ross snow5 synthetic experiment

Twin (perfect-model) experiment for snow5 on the Ross lumped basin, driven by
the SUMMA forcing ensemble. It checks historic simulation and DA modes only (no
forecasting).

| Piece | Where |
|---|---|
| Configs | `runs/snow5/ross_synthetic/` |
| Truth, synthetic obs, held-out mean forcing | `data/snow5/ross_synthetic/` |
| Run outputs (one timestamped dir per run) | `outputs/snow5/ross_synthetic/<mode>/` |
| Figures + `scores.csv` | `outputs/snow5/ross_synthetic/figures/` |
| Setup workflow | `src/snowda/workflows/snow5_synthetic_setup.py` |
| Plotting / scores | `src/snowda/viz/snow5_synthetic.py`, `scripts/snow5/plot_ross_synthetic.py` |
| Run everything | `shell_scripts/snow5/run_ross_synthetic.sh <python>` |

## Design

- **Truth**: SUMMA ensemble member 050, run through snow5 with the same
  parameters as every mode (perfect model; forcing is the only uncertainty).
- **Synthetic obs**: daily, `truth + N(0, sigma)`, clipped at 0, with
  `sigma = max(0.10 * truth, 5 kg m-2)`. That matches the filter's error model.
- **Ensemble**: members 001–049. The truth member is never in the ensemble or
  in the mean forcing.
- Period: WY2011–WY2016 (2010-10-01 to 2016-09-30). SWE starts at 0.

| Config | Mode | Workflow |
|---|---|---|
| `ross_snow5_synthetic_setup.yaml` | build truth, obs, mean forcing (run first) | `snowda.workflows.snow5_synthetic_setup` |
| `ross_snow5_historic_mean.yaml` | deterministic run on the 49-member mean forcing | `snowda.workflows.run_snow5` |
| `ross_snow5_historic_ensemble.yaml` | 49-member open loop | `snowda.workflows.run_snow5_ensemble` |
| `ross_snow5_sir.yaml` | particle filter, resample when N_eff ≤ 0.5 N | `snowda.workflows.run_snow5_da` |
| `ross_snow5_sis.yaml` | particle filter, no resampling, weights reset each Oct | `snowda.workflows.run_snow5_da` |

DA runs update every 14 days. Change `assimilation_interval_days`, the obs error
settings, or `tempering_target` in the SIR/SIS configs. If you change obs error,
change it in the setup config too and re-run setup.

## Running

```bash
shell_scripts/snow5/run_ross_synthetic.sh /Users/dcasson/.pyenv/versions/snowda/bin/python
```

The ensemble, SIR and SIS runs each read all 49 SUMMA forcing files (about 6 s
each), so they take about 5 minutes. The script runs the four modes in parallel.

## Caveat: melt factor

`compute_melt_factor` multiplies `mean * sin * diff` rather than using the
additive SNOW-17 form (see the strict xfail in `laugh_tests/test_snow5.py`).
The truth and every mode use the same code, so the twin experiment is still
self-consistent. Absolute melt rates differ from SNOW-17, though.
