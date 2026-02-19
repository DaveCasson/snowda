# snowda architecture

`snowda` is a Python codebase for hydrological model simulation with a focus on snow data assimilation and paper-figure pipelines.

## Core principle
Keep the repo understandable by enforcing clear boundaries:
- **core/**: scientific logic (model stepping, DA algorithms, metrics). No file I/O, no plotting.
- **io/**: netCDF / observational data read/write and standardization.
- **adapters/**: the only place that calls code imported from other repositories.
- **viz/**: plotting only. No raw file I/O.
- **workflows/**: orchestration entry points. Loads config, calls io/adapters/core/viz, writes outputs.
- **notebooks/**: demonstrations only; notebooks call workflows/core and should not become the source of truth.

## Import rules (hard rules)
- `core/` MUST NOT import `io/`, `viz/`, or `adapters/`.
- `viz/` MUST NOT read files directly. It receives in-memory data (xarray / numpy / pandas).
- Only `workflows/` may import across all layers.

## External repositories
When bringing in external code:
- Do not scatter imports across the repo.
- Create one adapter module per external repo in `adapters/`.
- The adapter is responsible for translating:
  - units
  - variable naming
  - dimension order / coordinates
  - masks / fill values
  - time conventions

## Configuration
- All runs are defined by YAML configs under `runs/`.
- Relative paths resolve relative to the YAML file.
- Input data sources must be absolute paths.
- Outputs/logs may be relative.
- Pysummada runs use `run_mode` to select one of four modes:
  - `deterministic`: no DA, no ESP
  - `deterministic_esp`: no DA, ESP using deterministic state
  - `da`: DA only, no ESP
  - `da_esp`: DA followed by ESP using DA states

## Outputs
Each run writes to a unique run directory (timestamped), containing:
- resolved config snapshot
- provenance.json (git hash, time, config path)
- logs and produced artifacts (netCDF, figures)

- ESP traces (SUMMA + mizuRoute) live under `<run_dir>/esp/<member>/<state>/<trace_year>/`, where SUMMA outputs `summa_{trace_year}.nc` and this same directory serves as the mizuRoute `input_dir`. Optional files such as `param.nml.default` and `topology.nc` can be declared in the ESP config and are copied into the trace directory before the mizuRoute run so the paths and filenames stay aligned for distributed workloads.

## Testing
Minimum expectation:
- Adapter smoke tests using tiny synthetic data
- I/O roundtrip tests for netCDF conventions
- One workflow smoke test that runs fast (<2s) and writes expected metadata files