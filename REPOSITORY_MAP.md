# Repository Map: snowda

A comprehensive map of entry points, core modules, and where units and CF conventions are enforced.

## Entry Points

### Primary Entry Point
- **`snowda/workflows/run_case.py`**
  - Function: `run_case(config_path: str | Path) -> Path`
  - CLI: `python -m snowda.workflows.run_case <config.yaml>`
  - Purpose: Main orchestration entry point that:
    - Loads YAML configuration
    - Creates timestamped run directory
    - Configures logging
    - Saves resolved config snapshot and provenance
    - Orchestrates pipeline (currently stubbed with TODOs)

### Configuration Loading
- **`snowda/config/loader.py`**
  - Function: `load_config(config_path: str | Path) -> Dict[str, Any]`
  - Purpose: Loads and validates YAML configs with path policy enforcement
  - Rules:
    - Relative paths resolve relative to config file directory
    - Input paths (non-output/cache/log) must be absolute
    - Output/cache/log paths may be relative

## Core Modules

### Scientific Logic (Currently Empty)
- **`snowda/core/`**
  - Status: Empty directory (only `__init__.py` exists)
  - Intended purpose: Model stepping, data assimilation algorithms, metrics
  - Constraints: No file I/O, no plotting, operates on SI-unit data only

### I/O Layer
- **`snowda/io/`**
  - **`snowda/io/units.py`**: Unit handling and conversion utilities
  - Status: Unit utilities exist; netCDF read/write functions not yet implemented
  - Intended purpose: netCDF I/O, observational data I/O, CF convention enforcement

### Visualization
- **`snowda/viz/`**
  - Status: Empty directory (only `__init__.py` exists)
  - Intended purpose: Figure generation
  - Constraints: No direct file I/O; receives in-memory data only

### Utilities
- **`snowda/utils/logging.py`**
  - Functions: `get_logger()`, `configure_logging()`
  - Purpose: Centralized logging configuration

## Units Enforcement

### Location: `snowda/io/units.py`

**Key Functions:**

1. **`quantify(ds, unit_attr="units")`**
   - Attaches Pint units to xarray Dataset variables based on CF `units` attributes
   - Uses `pint-xarray` integration
   - Variables without units are left unchanged

2. **`dequantify(ds, unit_attr="units")`**
   - Removes Pint units, writes unit strings back to `attrs`
   - Used before writing to netCDF

3. **`convert_vars(ds, targets, quantify_first=True, strict=True)`**
   - Converts selected variables to target units
   - Logs each conversion: "var: from -> to"
   - Raises `UnitsError` on failure if `strict=True`

4. **`to_internal_units(ds, targets=None, quantify_first=True, strict=False)`**
   - Converts dataset to snowda's canonical internal units
   - Uses `INTERNAL_TARGETS_DEFAULT` by default

**Canonical Internal Units** (`INTERNAL_TARGETS_DEFAULT`):
- `SWE`: `"millimeter"` (water-equivalent depth)
- `snow_depth`: `"meter"`
- `tair`: `"kelvin"`

**Unit Policy:**
- All internal computations should use SI units
- Conversions happen at I/O boundaries (`snowda/io/`, `snowda/adapters/`)
- `snowda/core/` assumes inputs already use SI units
- Unit conversions are logged and recorded in provenance

**Dependencies:**
- Requires `pint` and `pint-xarray` (optional dependency)
- Raises `UnitsError` with clear message if not available

## CF Conventions Enforcement

### Policy Location: `DATA_CONVENTIONS.md`

**Current Status:** Policy is documented but implementation is not yet complete.

**Intended Enforcement Points:**

1. **`snowda/io/`** (Primary responsibility)
   - Responsible for CF-compliant metadata on netCDF write
   - Should enforce CF metadata requirements:
     - Coordinate variables: `time`, `lat`, `lon` with proper `units` and `standard_name`
     - Data variables: `standard_name`, `long_name`, `units`, `_FillValue`
     - Projected grids: `grid_mapping` variable for CRS/projection
   - Converts external units to SI where appropriate

2. **`snowda/adapters/`** (Future)
   - Does not exist yet (mentioned in architecture docs)
   - Should translate variable names, dimensions, CRS metadata, and units
   - Should convert all quantities into snowda's SI conventions
   - Should handle time conventions

**CF Metadata Requirements** (from `DATA_CONVENTIONS.md`):

**Coordinate Variables:**
- `time`: CF-compliant time coordinate with valid `units` (e.g., `"seconds since 1970-01-01 00:00:00"`)
- `lat`, `lon`: `standard_name` (`"latitude"`, `"longitude"`) and `units` (`"degrees_north"`, `"degrees_east"`)
- Projected grids (`x`, `y`): CF `grid_mapping` variable describing CRS/projection

**Data Variables:**
- `standard_name`: CF standard name when suitable one exists
- `long_name`: Human-readable description
- `units`: String units compatible with Pint
- `_FillValue` and/or `missing_value` on disk
- `coordinates` attribute where auxiliary coordinates are required

**Missing Values:**
- In memory: Prefer NaNs
- On disk: Consistent `_FillValue` appropriate for dtype
- Avoid implicit scaling or unit conversion when writing netCDF

## Architecture Boundaries

### Import Rules (Hard Constraints)
- `core/` MUST NOT import `io/`, `viz/`, or `adapters/`
- `viz/` MUST NOT read files directly
- Only `workflows/` may import across all layers

### Data Flow
1. **Entry**: `workflows/run_case.py` loads config
2. **I/O**: `io/` reads data, applies unit conversions, enforces CF conventions
3. **Core**: `core/` operates on SI-unit data (no I/O)
4. **Output**: `io/` writes CF-compliant netCDF
5. **Viz**: `viz/` generates figures from in-memory data

## Module Status Summary

| Module | Status | Purpose |
|--------|--------|---------|
| `workflows/run_case.py` | ✅ Implemented | Main entry point, orchestration |
| `config/loader.py` | ✅ Implemented | YAML config loading with path policy |
| `io/units.py` | ✅ Implemented | Unit conversion utilities |
| `io/` (netCDF I/O) | ⚠️ Not implemented | Should handle CF conventions on read/write |
| `core/` | ⚠️ Empty | Model, DA algorithms, metrics |
| `viz/` | ⚠️ Empty | Figure generation |
| `adapters/` | ❌ Does not exist | External repo integration, unit/CF translation |

## Key Documentation Files

- **`ARCHITECTURE.md`**: Overall architecture and import rules
- **`DATA_CONVENTIONS.md`**: CF conventions and units policy
- **`PROJECT_MAP.md`**: Quick reference for entry points and key modules
- **`CONTRIBUTING.md`**: Contribution guidelines

## Notes

- The repository is in early development; many modules are stubbed or empty
- CF convention enforcement is documented but not yet implemented in code
- Unit handling infrastructure exists (`io/units.py`) but needs integration with netCDF I/O
- The architecture enforces clear boundaries to keep the codebase maintainable

