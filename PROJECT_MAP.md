# snowda project map

## Entry points
- Run a case from YAML:
  - `python -m snowda.workflows.run_case runs/examples/minimal.yaml`

## Key modules
- `snowda/config/loader.py` — YAML loading + path policy
- `snowda/workflows/run_case.py` — orchestration (config -> run dir -> pipeline)
- `snowda/io/` — netCDF + observation I/O
- `snowda/core/` — model + assimilation + metrics
- `snowda/viz/` — figure generation

## Where to put new code
- New DA method: `snowda/core/assimilation.py` (or a new file under core/)
- New netCDF convention or reader: `snowda/io/`
- New external repo integration: `snowda/adapters/<repo_name>.py`
- New figure: `snowda/viz/figures.py` + config under `runs/paper/`

## Repo conventions
- Notebooks should call code in `src/snowda/` rather than re-implement logic.
- Avoid global state; pass data/config explicitly.