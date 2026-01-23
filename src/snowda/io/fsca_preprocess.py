from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import rasterio.features

from snowda.config.loader import load_config
from snowda.utils.logging import get_logger

log = get_logger(__name__)


def preprocess_fsca(config_path: str | Path) -> Path:
    cfg = load_config(config_path)

    files = sorted(Path(d).rglob("*.nc") for d in cfg["data_dirs"])
    # flatten generator-of-generators
    files = [p for sub in files for p in sub]
    if not files:
        raise FileNotFoundError("No netCDF files found.")

    keep_vars = cfg.get("keep_vars")

    # Notebook-stable open: avoid parallel=True; chunk x/y too
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        chunks={"time": 1, "y": 600, "x": 600},  # good start for 2400x2400
    )

    # Ensure rio sees CRS (your files have a `crs` grid_mapping variable)
    if not ds.rio.crs:
        # rioxarray often picks this up automatically, but if not:
        # try to pull WKT/proj4 from the "crs" variable attrs
        crs_var = ds.get("crs")
        if crs_var is None:
            raise ValueError("Dataset has no rio CRS and no 'crs' variable.")
        # Prefer WKT if present
        wkt = crs_var.attrs.get("crs_wkt") or crs_var.attrs.get("spatial_ref")
        proj4 = crs_var.attrs.get("proj4")
        ds = ds.rio.write_crs(wkt or proj4)

    raster_crs = ds.rio.crs

    if keep_vars is None:
        keep_vars = list(ds.data_vars)
    ds = ds[keep_vars]

    # --- Vector: reproject to raster CRS (THIS is the missing step) ---
    gdf = gpd.read_file(cfg["gpkg_path"])
    if gdf.empty:
        raise ValueError("Vector file contains no geometries.")

    if gdf.crs is None:
        raise ValueError("Vector file has no CRS defined.")

    gdf = gdf.to_crs(raster_crs)  # <- key fix

    # Optional bbox clip in *raster CRS units* (meters)
    if cfg.get("use_bbox", True):
        pad = float(cfg.get("bbox_padding", 0.0))
        minx, miny, maxx, maxy = gdf.total_bounds
        ds = ds.rio.clip_box(minx=minx - pad, miny=miny - pad, maxx=maxx + pad, maxy=maxy + pad)

    # Build one zone raster: zones(y,x) with polygon IDs, 0 = background
    # (1..N are polygon IDs)
    transform = ds.rio.transform()
    ny, nx = int(ds.sizes["y"]), int(ds.sizes["x"])

    shapes = [(geom, i + 1) for i, geom in enumerate(gdf.geometry)]
    zones = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=True,
    )

    zones_da = xr.DataArray(zones, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]}, name="zone")

    # Reduce each variable by zone without creating (polygon,y,x) or expanding
    out: Dict[str, xr.DataArray] = {}
    zone_ids = np.arange(1, len(gdf) + 1, dtype=np.int32)

    for v in keep_vars:
        da = ds[v]
        # Make sure fill values become NaN so they don't skew means
        fill = da.attrs.get("_FillValue", None)
        if fill is not None:
            da = da.where(da != fill)

        means = []
        for zid in zone_ids:
            m = da.where(zones_da == zid).mean(dim=("y", "x"), skipna=True)
            means.append(m)

        out[v] = xr.concat(means, dim="polygon").assign_coords(
            polygon=[f"poly_{i}" for i in range(len(gdf))]
        )

    combined = xr.Dataset(out)

    # Attach polygon attributes as polygon coords
    attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]
    for c in attr_cols:
        combined = combined.assign_coords({c: ("polygon", gdf[c].astype(str).tolist())})

    combined.attrs["processed_polygons"] = int(len(gdf))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / cfg.get("output_file", "fsca_preprocessed.nc")

    # No persist() — stream compute into the file
    combined.to_netcdf(out_path)

    log.info("Wrote preprocessed fSCA netCDF: %s", out_path)
    return out_path
