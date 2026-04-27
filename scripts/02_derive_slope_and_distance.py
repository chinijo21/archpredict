"""
Slope (Horn 1981) and Euclidean distance to a "river" mask, derived
from the synthetic DEM.

River mask = lowest 2% of elevations. Distance transform uses the
pixel size as sampling so the output is in metres directly.

Outputs:
    data/synthetic_slope.tif
    data/synthetic_dist_water.tif
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import convolve, distance_transform_edt


PROJECT_ROOT = Path(os.environ.get("ARCHPREDICT_ROOT", os.getcwd())).resolve()
DATA = PROJECT_ROOT / "data"

DEM_PATH = DATA / "synthetic_dem.tif"
SLOPE_PATH = DATA / "synthetic_slope.tif"
DIST_PATH = DATA / "synthetic_dist_water.tif"

RIVER_PCT = 2.0  # bottom percentile defining the river mask


if not DEM_PATH.is_file():
    raise FileNotFoundError(f"DEM not found: {DEM_PATH}. Run script 01 first.")

with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(np.float64)
    profile = src.profile.copy()
    cx = src.transform.a
    cy = -src.transform.e

print(f"loaded DEM: shape={dem.shape}, pixel={cx} m")


# --- slope (Horn 1981) ----------------------------------------------
# Sobel-like 3x3 kernels; dz/dy is sign-flipped because raster row
# index grows southward but we want positive y to be north.
kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float64) / (8.0 * cx)

ky = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float64) / (8.0 * cy)
ky = -ky

dzdx = convolve(dem, kx, mode="reflect")
dzdy = convolve(dem, ky, mode="reflect")

slope_deg = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))).astype(np.float32)


# --- distance to river ---------------------------------------------
threshold = np.percentile(dem, RIVER_PCT)
river_mask = dem <= threshold
n_river = int(river_mask.sum())
print(f"river threshold: {threshold:.2f} m ({RIVER_PCT}% percentile), "
      f"{n_river} px ({100.0 * n_river / river_mask.size:.2f}%)")

dist_m = distance_transform_edt(~river_mask, sampling=(cy, cx)).astype(np.float32)


# --- write ----------------------------------------------------------
out_profile = profile.copy()
out_profile.update(dtype="float32", nodata=-9999.0, compress="deflate", tiled=True)

with rasterio.open(SLOPE_PATH, "w", **out_profile) as dst:
    dst.write(slope_deg, 1)

with rasterio.open(DIST_PATH, "w", **out_profile) as dst:
    dst.write(dist_m, 1)

print(f"wrote {SLOPE_PATH}")
print(f"  slope: {slope_deg.min():.2f}..{slope_deg.max():.2f} deg, mean {slope_deg.mean():.2f}")
print(f"wrote {DIST_PATH}")
print(f"  dist: {dist_m.min():.1f}..{dist_m.max():.1f} m, mean {dist_m.mean():.1f}")
