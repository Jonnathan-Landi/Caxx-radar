# src/georef/gridding.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


def polar_to_grid_2d(ds_lonlat: xr.Dataset,
                     epsg: int,
                     dx: float,
                     dy: float,
                     buffer_m: float,
                     nodata: float = -9999.0) -> xr.Dataset:
    """
    Proyecta lon/lat a (X,Y) en EPSG dado y rasteriza el campo en una grilla regular
    mediante vecino más cercano (KNN=1).

    Retorna Dataset con dims ('y','x') y variable principal con mismo nombre (ej. 'dBZ').
    """
    varname = list(ds_lonlat.data_vars)[0]  # asumimos un solo campo (dBZ)
    v = ds_lonlat[varname].values
    lon = ds_lonlat["lon"].values
    lat = ds_lonlat["lat"].values

    # Proyección a X/Y
    crs = CRS.from_epsg(epsg)
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    X, Y = t.transform(lon, lat)

    # Extensión
    x0 = np.nanmin(X) - buffer_m
    x1 = np.nanmax(X) + buffer_m
    y0 = np.nanmin(Y) - buffer_m
    y1 = np.nanmax(Y) + buffer_m

    xs = np.arange(x0, x1 + dx, dx)
    ys = np.arange(y0, y1 + dy, dy)

    # KNN 1 vecino
    mask = np.isfinite(v) & np.isfinite(X) & np.isfinite(Y)
    pts = np.column_stack([X[mask].ravel(), Y[mask].ravel()])
    vals = v[mask].ravel()

    if pts.size == 0:
        # vacío: devolver grilla nodata
        out = np.full((ys.size, xs.size), nodata, dtype=np.float32)
    else:
        tree = cKDTree(pts)
        XX, YY = np.meshgrid(xs, ys)
        q = np.column_stack([XX.ravel(), YY.ravel()])
        _, idx = tree.query(q, k=1)
        out = vals[idx].reshape(YY.shape).astype(np.float32)
        out[~np.isfinite(out)] = nodata

    da = xr.DataArray(
        out,
        dims=("y", "x"),
        coords={"x": xs, "y": ys},
        name=varname,
        attrs={"units": ds_lonlat[varname].attrs.get("units", "")}
    )

    ds_out = da.to_dataset()
    ds_out.attrs.update({
        "crs": f"EPSG:{epsg}",
        "method": "nearest (K=1)",
        "nodata": float(nodata),
    })
    return ds_out
