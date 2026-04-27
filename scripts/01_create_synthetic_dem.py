"""
Synthetic DEM for the prediction pipeline.

500x500 px, 30 m, EPSG:25830 (UTM 30N). Terrain = NW-SE gradient + a
sinuous low-elevation corridor ("river") + multi-scale random noise.
Seed 42 throughout the project.

Run from a Python with rasterio + numpy. From QGIS:

    import os
    os.chdir('/path/to/project/root')
    exec(open('scripts/01_create_synthetic_dem.py').read())

The project root is taken from $ARCHPREDICT_ROOT or the current
working directory.

Output: data/synthetic_dem.tif
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


# --- config ---------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("ARCHPREDICT_ROOT", os.getcwd())).resolve()
OUTPUT = PROJECT_ROOT / "data" / "synthetic_dem.tif"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 500, 500
PIXEL = 30.0
ORIGIN_X, ORIGIN_Y = 400000.0, 4530000.0
EPSG = 25830
SEED = 42

BASE_M = 200.0
GRADIENT_M = 400.0
RIVER_DEPTH_M = 80.0
NOISE_STD_M = 25.0


# --- terrain --------------------------------------------------------
rng = np.random.default_rng(SEED)
ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
xn = xs / (WIDTH - 1)
yn = ys / (HEIGHT - 1)

gradient = (1.0 - 0.5 * xn - 0.5 * yn) * GRADIENT_M

river_axis = WIDTH / 2.0 + 60.0 * np.sin(2.0 * np.pi * yn * 1.5)
river_carve = RIVER_DEPTH_M * np.exp(-((xs - river_axis) ** 2) / (2.0 * 25.0 ** 2))

noise = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
for scale in (8, 16, 32, 64):
    coarse = rng.standard_normal((HEIGHT // scale + 2, WIDTH // scale + 2))
    upsampled = np.kron(coarse, np.ones((scale, scale)))[:HEIGHT, :WIDTH]
    noise += upsampled / scale
noise = (noise - noise.mean()) / noise.std() * NOISE_STD_M

dem = (BASE_M + gradient - river_carve + noise).astype(np.float32)


# --- write ----------------------------------------------------------
profile = {
    "driver": "GTiff",
    "height": HEIGHT,
    "width": WIDTH,
    "count": 1,
    "dtype": "float32",
    "crs": f"EPSG:{EPSG}",
    "transform": from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL),
    "nodata": -9999.0,
    "compress": "deflate",
    "tiled": True,
}

with rasterio.open(OUTPUT, "w", **profile) as dst:
    dst.write(dem, 1)

print(f"wrote {OUTPUT}")
print(f"shape={dem.shape}  range={dem.min():.1f}..{dem.max():.1f} m  mean={dem.mean():.1f} m")
