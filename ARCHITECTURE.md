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
- SUMMA variants are selected by the workflow entrypoint, not by YAML:
  - `python -m snowda.workflows.run_summa <yaml>`: single SUMMA run, optional mizuRoute
  - `python -m snowda.workflows.run_summa_esp <yaml>`: ESP traces, no assimilation
  - `python -m snowda.workflows.run_summa_historic_da <yaml>`: particle filter through the historic period, no ESP. Each member keeps its own forcing. After each SUMMA segment, mizuRoute routes that member's SUMMA output. A particle is the SUMMA restart and the mizuRoute restart from that same run.
  - `python -m snowda.workflows.run_summa_da <yaml>`: the same assimilation, then ESP. Assimilation dates are configured like ESP dates (`assimilation.dates`, optional `frequency_days` and `months`). `assimilation_mode: swe` reads model-equivalent SWE; `fsca_only` uses fractional snow-covered area.
- Both assimilation entry points use `pf_scheme`. `sis` updates weights and continues each particle from its own SUMMA and mizuRoute restarts. `sir` copies both donor restarts when the effective sample size is above `n_eff_threshold`; the member still keeps its own forcing.
- Shared SUMMA config fields live in pysummada YAML (`base_file_manager`, ensemble size, times, `summa_exe`). Deterministic routing uses optional top-level `mizuroute`. ESP and DA-then-ESP require an `esp` section. Historic DA rejects an `esp` section and requires top-level `mizuroute`. mizuRoute writes output and restarts under the timestamped run directory (`<run_dir>/mizuroute/` for deterministic and historic segments; ESP traces keep per-trace `mizuroute/` folders). Historic segment discharge is `<run_dir>/mizuroute/<member>_<YYYY_MM_DD>_mizuroute.nc`.

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