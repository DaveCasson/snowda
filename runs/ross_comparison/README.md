# Ross River comparison set

Six configs for the same basin, period, ensemble and scoring settings, one per
run mode. They exist so the runs can be compared statistically: historic runs
with each other, forecast runs with each other.

| File | What it runs | Scored as |
|---|---|---|
| `deterministic.yaml` | One run on ensemble-mean forcing | historic |
| `historic_ensemble.yaml` | Open-loop 50-member ensemble | historic |
| `historic_sis.yaml` | Ensemble with SWE assimilation by SIS | historic |
| `esp_open_loop.yaml` | Open-loop ensemble, then ESP forecasts | historic and forecast |
| `esp_sis.yaml` | SIS assimilation, then ESP forecasts | historic and forecast |
| `esp_sir.yaml` | SIR assimilation, then ESP forecasts | historic and forecast |

`esp_open_loop` is the reference for the two assimilating forecast runs.

## Run

```bash
python -m snowda.workflows.run_case runs/ross_comparison/historic_sis.yaml
```

`run_mode` in the file picks the workflow and, for the assimilating modes, the
particle-filter scheme. Evaluation runs when the model run finishes and writes
`<run_dir>/evaluation/`. To score an existing run again:

```bash
python -m snowda.workflows.evaluate_run <run_dir>
```

## What must stay identical

`tests/test_comparison_configs.py` checks these.

- **All files:** `start_time`, `end_time` and the whole `evaluation` block.
- **Ensemble files:** `num_ens` and the forcing paths.
- **`esp_*` files:** the whole `esp` block. With the same `seed`, issue dates
  and `n_members`, forecast member `m007` uses the same weather year in all
  three runs, so a difference at one issue date comes from the initial states.
- **Assimilating files:** the `assimilation` block and the filter settings.
  `esp_sis.yaml` and `esp_sir.yaml` differ only in `run_mode` and names.

## What evaluation writes

- `historic/scores_by_water_year.csv`: one row per water year, window
  (`water_year` or `season`) and estimate. `ensemble` is the equally weighted
  ensemble. `weighted` applies the filter weights to the same members.
- `historic/summary.json`: the same scores pooled over the scored years.
- `historic/series.nc`: daily observation, means, 10/25/50/75/90 % bands and
  weights.
- `historic/figures/wy_<year>.png`: one figure per water year.
- `forecast/scores_by_issue.csv`: one row per forecast issue date.
- `forecast/summary.json`, `forecast/metrics.nc`: scores per issue month.
- `da/updates.csv`: `N_eff`, tempering, resampling and SWE fit per update.

Compare runs by joining the per-year or per-issue tables on `water_year` or
`init_time` and keeping the rows both runs have.

## Notes

- Paths are for the local machine. Copy the folder and change them for Anvil.
- The run ends on 2024-08-01, so water year 2024 has less than 90 % of its
  melt season and is not scored.
- No `thresholds_csv` is set, so flood thresholds are empirical quantiles of
  the observed annual maxima. With about 43 years, the 10- and 20-year event
  scores rest on a few events.
