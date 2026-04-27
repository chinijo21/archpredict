"""
Generate synthetic site presence points using a known multiplicative
settlement rule:

    p(site | x, y) = water_score * slope_score * elev_score
    water_score = exp(-dist_water / 300 m)
    slope_score = exp(-slope_deg / 8 deg)
    elev_score  = exp(-(elev - 250)^2 / (2 * 80^2))

85 % of points are sampled (without replacement) weighted by p.
The remaining 15 % are placed uniformly at random over cells not
already chosen, simulating non-environmental factors that the model
cannot recover (ritual, geopolitics, cataloguing noise).

Outputs:
    data/synthetic_site_probability.tif   the suitability surface
    data/synthetic_sites.gpkg             N point features with attrs
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from osgeo import ogr, osr


PROJECT_ROOT = Path(os.environ.get("ARCHPREDICT_ROOT", os.getcwd())).resolve()
DATA = PROJECT_ROOT / "data"

DEM_PATH = DATA / "synthetic_dem.tif"
SLOPE_PATH = DATA / "synthetic_slope.tif"
DIST_PATH = DATA / "synthetic_dist_water.tif"
PROB_PATH = DATA / "synthetic_site_probability.tif"
SITES_PATH = DATA / "synthetic_sites.gpkg"

N_SITES = 80
NOISE_FRACTION = 0.15
SEED = 42

W_DECAY = 300.0
S_DECAY = 8.0
E_OPT = 250.0
E_SIGMA = 80.0


for p in (DEM_PATH, SLOPE_PATH, DIST_PATH):
    if not p.is_file():
        raise FileNotFoundError(f"Missing input {p}; run scripts 01-02 first.")

with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(np.float64)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs

with rasterio.open(SLOPE_PATH) as src:
    slope = src.read(1).astype(np.float64)

with rasterio.open(DIST_PATH) as src:
    dist_water = src.read(1).astype(np.float64)


# --- suitability surface --------------------------------------------
water_score = np.exp(-dist_water / W_DECAY)
slope_score = np.exp(-slope / S_DECAY)
elev_score = np.exp(-((dem - E_OPT) ** 2) / (2.0 * E_SIGMA ** 2))
prob = water_score * slope_score * elev_score

prob_profile = profile.copy()
prob_profile.update(dtype="float32", nodata=-9999.0, compress="deflate", tiled=True)
with rasterio.open(PROB_PATH, "w", **prob_profile) as dst:
    dst.write(prob.astype(np.float32), 1)

print(f"wrote {PROB_PATH}")
print(f"  prob: {prob.min():.4f}..{prob.max():.4f}, mean {prob.mean():.4f}")


# --- sample ---------------------------------------------------------
rng = np.random.default_rng(SEED)
n_real = int(round(N_SITES * (1.0 - NOISE_FRACTION)))
n_noise = N_SITES - n_real

flat_p = prob.ravel()
flat_p = flat_p / flat_p.sum()

real_idx = rng.choice(prob.size, size=n_real, replace=False, p=flat_p)
remaining = np.setdiff1d(np.arange(prob.size), real_idx, assume_unique=False)
noise_idx = rng.choice(remaining, size=n_noise, replace=False)

idx = np.concatenate([real_idx, noise_idx])
is_noise = np.concatenate([np.zeros(n_real, dtype=bool),
                           np.ones(n_noise, dtype=bool)])

# shuffle so noise sites aren't all at the end
order = rng.permutation(N_SITES)
idx = idx[order]
is_noise = is_noise[order]

rows, cols = np.unravel_index(idx, prob.shape)

# intra-pixel jitter so sites aren't snapped to grid
jr = rng.uniform(-0.5, 0.5, size=N_SITES)
jc = rng.uniform(-0.5, 0.5, size=N_SITES)
xs = transform.a * (cols + 0.5 + jc) + transform.c
ys = transform.e * (rows + 0.5 + jr) + transform.f

print(f"sampled {N_SITES} sites: {n_real} rule-driven, {n_noise} noise")


# --- write GPKG -----------------------------------------------------
driver = ogr.GetDriverByName("GPKG")
sites_str = str(SITES_PATH)
if SITES_PATH.exists():
    driver.DeleteDataSource(sites_str)
ds = driver.CreateDataSource(sites_str)

srs = osr.SpatialReference()
srs.ImportFromEPSG(int(str(crs).split(":")[1]))
layer = ds.CreateLayer("sites", srs, ogr.wkbPoint)

fields = [
    ("site_id", ogr.OFTInteger),
    ("is_noise", ogr.OFTInteger),
    ("water_score", ogr.OFTReal),
    ("slope_score", ogr.OFTReal),
    ("elev_score", ogr.OFTReal),
    ("prob_score", ogr.OFTReal),
    ("elev_m", ogr.OFTReal),
    ("slope_deg", ogr.OFTReal),
    ("dist_water_m", ogr.OFTReal),
]
for name, t in fields:
    layer.CreateField(ogr.FieldDefn(name, t))

defn = layer.GetLayerDefn()
for i in range(N_SITES):
    r, c = rows[i], cols[i]
    f = ogr.Feature(defn)
    f.SetField("site_id", int(i))
    f.SetField("is_noise", int(is_noise[i]))
    f.SetField("water_score", float(water_score[r, c]))
    f.SetField("slope_score", float(slope_score[r, c]))
    f.SetField("elev_score", float(elev_score[r, c]))
    f.SetField("prob_score", float(prob[r, c]))
    f.SetField("elev_m", float(dem[r, c]))
    f.SetField("slope_deg", float(slope[r, c]))
    f.SetField("dist_water_m", float(dist_water[r, c]))
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(float(xs[i]), float(ys[i]))
    f.SetGeometry(pt)
    layer.CreateFeature(f)
    f = None
ds = None

print(f"wrote {SITES_PATH}")
