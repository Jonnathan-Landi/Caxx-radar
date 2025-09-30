# src/io/read_azi.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import xarray as xr
import wradlib as wrl


def detect_format(path: str | Path) -> str:
    """
    Detecta formato del archivo .azi. Por ahora asumimos Rainbow.
    Futuro: distinguir ASCII/HDF5/custom si aparecen.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".azi":
        return "rainbow"
    # fallback
    return "unknown"


def _extract_slice(tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Navega el dict devuelto por read_rainbow y devuelve el 'slice' principal.
    Maneja casos comunes de PPI.
    """
    vol = tree.get("volume")
    if vol is None:
        raise ValueError("Estructura Rainbow inesperada: falta 'volume'.")

    scan = vol.get("scan")
    if scan is None:
        raise ValueError("Estructura Rainbow inesperada: falta 'scan'.")

    slc = scan.get("slice")
    if slc is None:
        # algunos archivos pueden tener lista de slices o key distinta
        raise ValueError("Estructura Rainbow inesperada: falta 'slice'.")

    return slc


def _get_data_and_axes(slc: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extrae campo (dBZ), vector de azimutes (deg) y rangos (m) del slice Rainbow.
    Devuelve: data, az_deg, ranges_m, meta
    """
    if "data" not in slc:
        raise ValueError("Slice Rainbow sin 'data' decodificado.")

    data = np.asarray(slc["data"])

    # Elevación (grados)
    elev_deg = float(slc.get("@elevation", 0.0))

    # Rango inicial y tamaño de bin (Rainbow suele usar @rstart/@rscale en metros)
    r0 = float(slc.get("@rstart", 0.0))
    dr = float(slc.get("@rscale", 1000.0))

    # Número de bins
    if "rays" in slc and "ray" in slc["rays"]:
        # Cada rayo puede declarar @bins, tomamos el del primero
        n_rng = int(slc["rays"]["ray"][0].get("@bins", data.shape[-1]))
    else:
        n_rng = data.shape[-1]
    ranges_m = r0 + dr * np.arange(n_rng, dtype=float)

    # Azimutes:
    if "angles" in slc and "a" in slc["angles"]:
        az_list = slc["angles"]["a"]
        az_deg = np.array([float(a) for a in az_list], dtype=float)
    else:
        # si no vienen listados, asumimos cubrimiento uniforme [0,360)
        n_az = data.shape[0]
        az_deg = np.linspace(0.0, 360.0, n_az, endpoint=False, dtype=float)

    meta = {
        "elevation_deg": elev_deg,
        "r0_m": r0,
        "dr_m": dr,
    }
    return data, az_deg, ranges_m, meta


def read_azi(path: str | Path) -> xr.Dataset:
    """
    Lee un archivo .azi (Rainbow) y retorna un Dataset polar:
      dims: ('azimuth', 'range')
      vars: dBZ
      coords: azimuth[deg], range[m]
      attrs: metadatos útiles
    """
    fmt = detect_format(path)
    if fmt != "rainbow":
        raise ValueError(f"Formato no soportado aún: {fmt}")

    tree = wrl.io.read_rainbow(str(path))
    slc = _extract_slice(tree)
    data, az_deg, ranges_m, meta = _get_data_and_axes(slc)

    # Param del campo (nombre, unidades)
    param = slc.get("param", {})
    units = param.get("@units", "dBZ")
    pname = param.get("@name", "dBZ")

    da = xr.DataArray(
        data,
        dims=("azimuth", "range"),
        coords={
            "azimuth": ("azimuth", az_deg, {"units": "degree"}),
            "range": ("range", ranges_m, {"units": "m"}),
        },
        name=pname,
        attrs={"units": units}
    )

    # Metadatos del volumen/scan
    vol = tree.get("volume", {})
    radar_info = vol.get("sensorinfo", {})
    radar_lon = _safe_float(radar_info.get("@lon", np.nan))
    radar_lat = _safe_float(radar_info.get("@lat", np.nan))
    radar_alt = _safe_float(radar_info.get("@alt", np.nan))

    ds = da.to_dataset()
    ds.attrs.update({
        "source": "Rainbow .azi",
        "elevation_deg": meta["elevation_deg"],
        "r0_m": meta["r0_m"],
        "dr_m": meta["dr_m"],
        "radar_lon": radar_lon,
        "radar_lat": radar_lat,
        "radar_alt_m": radar_alt,
    })
    return ds


def _safe_float(x: Optional[str]) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")
