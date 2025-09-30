# scripts/parse_azi.py
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np
import xarray as xr

from src.io.read_azi import read_azi
from src.georef.polar import polar_dataset_to_lonlat
from src.georef.gridding import polar_to_grid_2d


def save_netcdf(ds: xr.Dataset, out_nc: Path, nodata: float | None = None) -> None:
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    # Preparar encoding (dtype/FillValue) para variables de datos
    encoding = {}
    for v in ds.data_vars:
        enc = {}
        # Tipo por defecto float32 + _FillValue si se indica
        enc["dtype"] = "float32"
        if nodata is not None:
            enc["_FillValue"] = np.float32(nodata)
        encoding[v] = enc

    # Asegurar codificación simple en coords (evitar FillValue en coords)
    for c in ds.coords:
        encoding[c] = {"dtype": "float32"} if np.issubdtype(ds[c].dtype, np.floating) else {}

    # Metadatos globales mínimos
    gattrs = ds.attrs.copy()
    gattrs.setdefault("Conventions", "CF-1.8")
    gattrs.setdefault("title", "CAXX radar product")
    gattrs.setdefault("institution", "ETAPA EP - Departamento de Investigación y Monitoreo")
    gattrs.setdefault("source", gattrs.get("source", "Rainbow .azi via wradlib"))
    gattrs.setdefault("history", "Created by caxx-radar/scripts/parse_azi.py")
    ds = ds.assign_attrs(gattrs)

    ds.to_netcdf(out_nc, encoding=encoding)
    print(f"[OK] NetCDF guardado: {out_nc}")


def main():
    ap = argparse.ArgumentParser(description="Parsea .azi (Rainbow) del radar CAXX y exporta NetCDF.")
    ap.add_argument("--file", required=True, help="Ruta al archivo .azi")
    ap.add_argument("--config", default="config/params.yaml", help="YAML de parámetros")
    ap.add_argument("--out-prefix", default="caxx", help="Prefijo para archivos de salida")
    ap.add_argument("--nc-polar", action="store_true", help="Guardar NetCDF en coordenadas polares")
    ap.add_argument("--nc-grid", action="store_true", help="Guardar NetCDF en grilla cartesiana (EPSG del YAML)")
    args = ap.parse_args()

    azi_path = Path(args.file)
    cfg_path = Path(args.config)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1) Leer polar
    ds_polar = read_azi(azi_path)
    print("[INFO] Dataset polar")
    print(ds_polar)

    # Guardar polar si se solicita
    if args.nc_polar:
        out_nc_polar = Path("data/processed") / f"{args.out_prefix}_polar.nc"
        save_netcdf(ds_polar, out_nc_polar)

    # 2) Georreferenciar a lon/lat y rasterizar a grilla UTM, luego guardar NetCDF
    if args.nc_grid:
        radar_lon = float(ds_polar.attrs.get("radar_lon", cfg["radar"]["lon"]))
        radar_lat = float(ds_polar.attrs.get("radar_lat", cfg["radar"]["lat"]))
        radar_alt = float(ds_polar.attrs.get("radar_alt_m", cfg["radar"]["alt_m"]))

        ds_ll = polar_dataset_to_lonlat(ds_polar, radar_lon, radar_lat, radar_alt)

        epsg = int(cfg["projection"]["epsg"])
        dx = float(cfg["grid"]["dx"])
        dy = float(cfg["grid"]["dy"])
        buffer_m = float(cfg["grid"]["buffer_m"])
        nodata = float(cfg["io"]["nodata"])

        ds_grid = polar_to_grid_2d(
            ds_ll, epsg=epsg, dx=dx, dy=dy, buffer_m=buffer_m, nodata=nodata
        )

        # Añadir metadatos útiles
        ds_grid.attrs.update({
            "grid_mapping": f"EPSG:{epsg}",
            "dx": dx, "dy": dy,
            "method": ds_grid.attrs.get("method", "nearest (K=1)")
        })

        out_nc_grid = Path("data/processed") / f"{args.out_prefix}_grid.nc"
        save_netcdf(ds_grid, out_nc_grid, nodata=nodata)


if __name__ == "__main__":
    main()
