from __future__ import annotations

from typing import Mapping

import xarray as xr

from snowda.utils.logging import get_logger

log = get_logger(__name__)


class UnitsError(RuntimeError):
    """Raised when unit handling or conversion fails."""


def _require_pint_xarray() -> None:
    """
    Ensure pint-xarray is importable.

    We keep this as a runtime check so snowda can still import on systems
    without pint-xarray, while unit conversion utilities fail with a clear message.
    """
    try:
        import pint_xarray  # noqa: F401
    except Exception as e:
        raise UnitsError(
            "Unit operations require `pint-xarray` (and `pint`). "
            "Install with: pip install pint pint-xarray"
        ) from e


def quantify(ds: xr.Dataset, *, unit_attr: str = "units") -> xr.Dataset:
    """
    Attach Pint units to variables based on ds[var].attrs[unit_attr].

    Requires pint-xarray. Returns a new Dataset with quantified variables.

    Notes:
      - Variables without a units attribute are left unchanged.
      - Coordinates are left as-is (time, lat/lon, etc.).
    """
    _require_pint_xarray()

    # pint-xarray uses .pint.quantify() to interpret attrs['units']
    # By default it uses the 'units' attribute.
    try:
        q = ds.pint.quantify(unit_registry=None, units=None)  # type: ignore[attr-defined]
    except TypeError:
        # Older pint-xarray signatures differ; fall back to simplest call.
        q = ds.pint.quantify()  # type: ignore[attr-defined]

    return q


def dequantify(ds: xr.Dataset, *, unit_attr: str = "units") -> xr.Dataset:
    """
    Drop Pint units, leaving plain arrays, and write unit strings back to attrs.

    Requires pint-xarray. Useful before writing to netCDF.
    """
    _require_pint_xarray()
    return ds.pint.dequantify()  # type: ignore[attr-defined]


def convert_vars(
    ds: xr.Dataset,
    targets: Mapping[str, str],
    *,
    quantify_first: bool = True,
    strict: bool = True,
) -> xr.Dataset:
    """
    Convert selected variables to target units.

    Args:
        ds: input dataset
        targets: mapping of variable name -> target unit string (Pint-compatible)
        quantify_first: if True, call quantify(ds) before conversion
        strict: if True, raise if a target variable is missing or lacks units.
                if False, skip missing/ununit-ed variables with a warning.

    Returns:
        A new dataset with converted variables (quantified unless you dequantify()).

    Logging:
        Logs each successful conversion as "var: from -> to".
    """
    _require_pint_xarray()

    out = quantify(ds) if quantify_first else ds

    for var, to_unit in targets.items():
        if var not in out.data_vars:
            msg = f"Variable '{var}' not found for unit conversion."
            if strict:
                raise UnitsError(msg)
            log.warning(msg)
            continue

        da = out[var]
        from_unit = da.attrs.get("units", None)

        try:
            converted = da.pint.to(to_unit)  # type: ignore[attr-defined]
        except Exception as e:
            msg = (
                f"Failed converting '{var}' to '{to_unit}'. "
                f"Current units attribute is '{from_unit}'."
            )
            if strict:
                raise UnitsError(msg) from e
            log.warning("%s (%s)", msg, e)
            continue

        out[var] = converted
        log.info("Units: %s: %s -> %s", var, from_unit, to_unit)

    return out
