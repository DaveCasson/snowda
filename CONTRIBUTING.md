# Contributing to snowda

## Development setup
- Create/activate environment
- Install in editable mode:
  - `pip install -e '.[dev]'`
- Register the environment as a Jupyter kernel:
  - `python -m ipykernel install --user --name snowda --display-name "Python (snowda)"`

## Style
- Format with black
- Lint with ruff
- Keep functions small and explicit

## Tests
- Run: `pytest`
- Add at least a smoke test when integrating new external code.

## Adding a new external codebase
1. Add the external code as a submodule/subtree or vendor it in a clearly named folder (if needed).
2. Create `src/snowda/adapters/<name>.py`
3. Expose a small API that returns xarray objects.
4. Add `tests/test_adapter_<name>_smoke.py`

## Adding a new run / figure
- Add config under `runs/paper/figX.yaml`
- Add plotting function in `snowda/viz/figures.py`
- Ensure outputs and provenance are written under `outputs/`