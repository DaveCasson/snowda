# Data conventions (snowda)

snowda aims to enforce **netCDF CF conventions** wherever possible and uses
**SI units preferentially** for all internal computations.

The primary in-memory data structure is `xarray.Dataset` / `xarray.DataArray`.
Unit handling and conversions should be performed using **Pint**
(preferably via `pint-xarray`).

---

## CF conventions (netCDF)

### Goal
All netCDF written by snowda should be CF-compliant to the extent feasible for
snow and hydrological applications.

When strict CF compliance is not possible, we:
1) document the deviation,
2) preserve metadata consistency,
3) provide adapters to translate to/from external conventions.

### Required CF-like metadata (whenever applicable)

For coordinate variables:
- `time`: CF-compliant time coordinate with valid `units`
  (e.g. `seconds since 1970-01-01 00:00:00`) and calendar if needed.
- `lat`, `lon`: include `standard_name` (`latitude`, `longitude`) and
  `units` (`degrees_north`, `degrees_east`).
- Projected grids (`x`, `y`): provide a CF `grid_mapping` variable describing
  the CRS/projection.

For data variables:
- `standard_name` (CF standard name) when a suitable one exists.
- `long_name` (human-readable description).
- `units` (string units compatible with Pint).
- `_FillValue` and/or `missing_value` on disk.
- `coordinates` attribute where auxiliary coordinates are required.

---

## Missing values and encoding

- In memory: prefer NaNs for missing values.
- On disk: use a consistent `_FillValue` appropriate for dtype.
- Avoid implicit scaling or unit conversion when writing netCDF; all conversions
  must be explicit and recorded.

---

## Units policy (SI-first, Pint-enforced)

### Core principle
**All internal computations in snowda should use SI units.**

Non-SI units (e.g. `mm`, `mm/day`) are allowed:
- at I/O boundaries,
- in adapters for external datasets,
- in plotting/figure output if scientifically conventional.

All such cases must be converted to SI before entering `snowda.core`.

### Preferred internal (SI) units

| Quantity               | Preferred unit |
|------------------------|----------------|
| Length / depth         | m              |
| SWE                    | kg m-2         |
| Temperature            | K              |
| Time                   | s              |
| Precipitation flux     | kg m-2 s-1     |
| Density                | kg m-3         |
| Energy                 | J              |
| Power / flux           | W m-2          |
| Streamflow / discharge | m3 s-1         |

Notes:
- SWE is treated as **mass per unit area** (`kg m-2`) internally, consistent
  with SI and CF conventions.
- If SWE is read as depth (e.g. `mm`), it must be converted in I/O or adapters.
- Streamflow uses SI cubic metres per second (`m3 s-1`), also written cms.

### Pint usage rules

- All data variables should carry a `units` attribute compatible with Pint.
- Unit conversions must use Pint (preferably via `pint-xarray`).
- Avoid hard-coded conversion factors (`* 1000`, `/ 3600`) in scientific code.
- Conversions should happen at clear boundaries:
  - `snowda/io/`
  - `snowda/adapters/`
- `snowda/core/` assumes inputs already use SI units.

---

## Implementation guidance

- **`snowda/io/`**
  - Responsible for CF-ish metadata on write.
  - Converts external units to SI where appropriate.
- **`snowda/adapters/`**
  - Translate variable names, dimensions, CRS metadata, and units.
  - Convert all quantities into snowda’s SI conventions.
- **`snowda/core/`**
  - Operates exclusively on SI-unit data.
  - No file I/O and no unit guessing.
- **`snowda/viz/`**
  - May convert units for presentation (e.g. `kg m-2` → `mm SWE`), but must do
    so explicitly and locally.

---

## Provenance and logging

Whenever unit conversions or standardizations occur:
- Log the conversion (source units → target units).
- Record the conversion in `provenance.json` when relevant.
- Preserve original units in metadata where possible.