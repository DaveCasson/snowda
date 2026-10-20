# Implementation To-Do

## Implementation
- Fill in absolute paths in `runs/examples/ross_summa_da.yaml` and `runs/examples/chena_summa_fsca.yaml`.
- Verify SUMMA outputs include `scalarSnowDepth` (or set `snow_depth_var` accordingly for Helbig).
- Ensure dependencies are installed: `pint`, `pint-xarray`, `pysumma`, `netCDF4`, `scipy`.
- Confirm fSCA observation netCDF schema: `time`, `hru`, `fsca` variable.

## Integration Checks
- Run `python -m snowda.workflows.run_summa_fsca runs/examples/chena_summa_fsca.yaml`.
- Run `python -m snowda.workflows.run_summa_da runs/examples/ross_summa_da.yaml`.
- Verify per-timestep outputs:
  - fSCA inputs/outputs per ensemble are generated.
  - `fsca_weights.csv` is created.
  - `initConditionFile` updates for resampling are applied.

## Units and Conventions
- Validate Pint conversions in `snowda/io/summa_fsca.py` against known units.
- Confirm `provenance.json` contains `unit_conversions`.

## Tests
- Add smoke tests:
  - fSCA prep + compute (single HRU, few timesteps).
  - PF resampling updates file manager entries.
- Add a minimal integration test using small netCDF fixtures for SUMMA outputs + fSCA obs.

## Future Extensions
- Add `assimilation_mode: combined` (SWE + fSCA) once SWE obs ingestion is ready.
- Add a small synthetic SWE + fSCA test case to validate combined likelihood.
