from __future__ import annotations

from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import rasterio.features
import rioxarray
import xarray as xr
from scipy import ndimage

from snowda.utils.logging import get_logger

log = get_logger(__name__)


def _read_dem(dem_path: Path | str) -> xr.DataArray:
    da = rioxarray.open_rasterio(dem_path, masked=True).squeeze("band", drop=True)
    if not da.rio.crs:
        raise ValueError("DEM must contain CRS metadata.")
    return da


def _rasterize_hrus(gdf: gpd.GeoDataFrame, template: xr.DataArray) -> xr.DataArray:
    transform = template.rio.transform()
    shapes = ((geom, idx + 1) for idx, geom in enumerate(gdf.geometry))
    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=template.shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return xr.DataArray(
        mask,
        dims=template.dims,
        coords={dim: template.coords[dim] for dim in template.dims},
        name="hru_mask",
    )


def _polygon_stats(
    dem: xr.DataArray,
    mask: xr.DataArray,
    idx: int,
    length_scale: float,
    mu_map: np.ndarray,
    sigma_map: np.ndarray,
) -> dict[str, float]:
    sel = mask.values == idx
    if not np.any(sel):
        return {}

    z = dem.values[sel]
    if z.size < 3:
        return {}

    mu = float(np.nanmean(mu_map[sel]))

    sigma_z = float(np.nanmean(sigma_map[sel]))

    xi = float(np.sqrt(2 * np.pi) * sigma_z / length_scale)

    c = 0.5330 * length_scale**0.0389
    d = 0.3193 * length_scale**0.1034

    return {
        "mu": float(mu),
        "sigma_z": sigma_z,
        "xi": xi,
        "c": c,
        "d": d,
    }


def preprocess_helbig(
    dem_path: Path | str,
    gpkg_path: Path | str,
    output_path: Path | str,
    *,
    hru_id_column: str = "hru",
    length_scale: float | None = None,
    same_resolution_output: Path | str | None = None,
) -> Path:
    """
    Prepare Helbig parameters for each HRU polygon from a DEM tile.

    The routine follows Helbig et al. (2021a/b) and returns μ, σz, ξ, and
    the scale-dependent constants c and d aggregated to the provided polygons.
    Optionally also writes per-pixel μ/σz/ξ fields at the original DEM resolution.
    """

    dem = _read_dem(Path(dem_path))
    hrus = gpd.read_file(gpkg_path)
    hrus = hrus.to_crs(dem.rio.crs)
    if hru_id_column not in hrus.columns:
        raise ValueError(f"HRU attribute '{hru_id_column}' not found.")

    length_scale = float(length_scale or abs(dem.rio.resolution()[0]))
    mask = _rasterize_hrus(hrus, dem)

    grad = np.gradient(
        dem.values,
        dem.rio.resolution()[0],
        dem.rio.resolution()[1],
        edge_order=2,
    )
    grad_y, grad_x = grad
    mu_map = 0.5 * np.sqrt(grad_x**2 + grad_y**2)

    def _nanstd_window(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype="float64")
        arr = arr[np.isfinite(arr)]
        return float(np.nanstd(arr)) if arr.size else np.nan

    sigma_map = ndimage.generic_filter(
        dem.values, function=_nanstd_window, size=3, mode="mirror"
    )
    xi_map = np.sqrt(2 * np.pi) * sigma_map / length_scale

    stats = []
    for idx in range(1, len(hrus) + 1):
        entry = _polygon_stats(dem, mask, idx, length_scale, mu_map, sigma_map)
        if not entry:
            continue
        stats.append(entry)

    if not stats:
        raise RuntimeError("No HRU statistics could be computed (check geometry coverage).")

    ds = xr.Dataset(coords={"hru": hrus[hru_id_column].values[: len(stats)]})
    for key in ["mu", "sigma_z", "xi", "c", "d"]:
        ds[key] = ("hru", [stat[key] for stat in stats])

    ds["length_scale"] = length_scale

    if same_resolution_output:
        res_path = Path(same_resolution_output)
        res_ds = xr.Dataset(
            data_vars={
                "mu": (dem.dims, mu_map),
                "sigma_z": (dem.dims, sigma_map),
                "xi": (dem.dims, xi_map),
            },
            coords={dim: dem.coords[dim] for dim in dem.dims},
        )
        res_ds.to_netcdf(res_path)
        log.info("Wrote per-pixel Helbig stats: %s", res_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    log.info("Wrote Helbig preprocessing data: %s", output_path)
    return Path(output_path)

if __name__ == "__main__":
    from pathlib import Path


    DEM_PATH = Path("/path/to/dem.tif")
    GPKG_PATH = Path("/path/to/hrus.gpkg")
    OUTPUT_PATH = Path("/path/to/output/helbig_params.nc")

    preprocess_helbig(
        dem_path="/Users/dcasson/Data/summa_snakemake/chena_tdx/watershed_tools/gis/dem.tif",
        gpkg_path="/Users/dcasson/Data/summa_snakemake/chena_tdx/watershed_tools/final/chena_tdx_WGS84.gpkg",
        output_path="/Users/dcasson/Data/snowda/helbig/chena_tdx_helbig_params.nc",
        hru_id_column="HRU_ID",   # change if your gpkg uses a different ID field
        length_scale=50.0,     # override if you aggregate to a coarser grid
        same_resolution_output="/Users/dcasson/Data/snowda/helbig/chena_tdx_helbig_params_same_resolution.nc",
)