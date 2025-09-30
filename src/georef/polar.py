from __future__ import annotations
import numpy as np
import xarray as xr
import wradlib.georef as georef

def polar_to_xyz(ranges_m: np.ndarray,
                 azimuth_deg: np.ndarray,
                 elev_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte malla polar (range, azimuth, elev) a XYZ cartesianas (m) relativas al radar.
    """
    rr, aa = np.meshgrid(ranges_m, np.deg2rad(azimuth_deg))
    el = np.deg2rad(elev_deg)
    xyz = georef.spherical_to_xyz(rr, aa, el)  # (3, n_az, n_rng)
    return xyz[0], xyz[1], xyz[2]


def xyz_to_lonlat(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  radar_lon: float, radar_lat: float, radar_alt_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte XYZ relativos al radar a lon/lat/alt absolutos.
    """
    lon, lat, alt = georef.xyz_to_lonlatalt(x, y, z, radar_lon, radar_lat, radar_alt_m)
    return lon, lat, alt


def polar_dataset_to_lonlat(ds: xr.Dataset,
                            radar_lon: float,
                            radar_lat: float,
                            radar_alt_m: float) -> xr.Dataset:
    """
    Anexa coordenadas lon/lat/alt al Dataset polar (dims: azimuth, range).
    Requiere ds.attrs['elevation_deg'].
    """
    elev = float(ds.attrs.get("elevation_deg", 0.0))
    ranges = ds["range"].values
    az = ds["azimuth"].values

    x, y, z = polar_to_xyz(ranges, az, elev)
    lon, lat, alt = xyz_to_lonlat(x, y, z, radar_lon, radar_lat, radar_alt_m)

    # Añadir coords auxiliares
    for name, arr in (("lon", lon), ("lat", lat), ("alt", alt)):
        ds = ds.assign_coords({name: (("azimuth", "range"), arr)})

    return ds
