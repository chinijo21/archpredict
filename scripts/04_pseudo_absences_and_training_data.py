"""
Environmentally stratified pseudo-absences and a unified training
table for sklearn.

Strategy: bin each predictor into N_BINS quantile groups (default 4),
take the cartesian product (up to N_BINS^3 cells in environmental
space), and allocate absences per bin proportional to the bin's share
of the eligible landscape. Sampling within a bin is uniform.

Eligibility: cells more than BUFFER_M metres from any presence.

Outputs:
    data/synthetic_pseudo_absences.gpkg  N_ABSENCES point features
    data/training_dataset.csv            shuffled CSV with columns
        point_id, x, y, presence, elev_m, slope_deg, dist_water_m,
        is_noise (0 for absences and rule-driven presences; 1 for
        noisy presences inherited from the site generator).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt
from osgeo import ogr, osr


PROJECT_ROOT = Path(os.environ.get("ARCHPREDICT_ROOT", os.getcwd())).resolve()
DATA = PROJECT_ROOT / "data"

DEM_PATH = DATA / "synthetic_dem.tif"
SLOPE_PATH = DATA / "synthetic_slope.tif"
DIST_PATH = DATA / "synthetic_dist_water.tif"
SITES_PATH = DATA / "synthetic_sites.gpkg"
ABSENCES_PATH = DATA / "synthetic_pseudo_absences.gpkg"
TRAINING_CSV = DATA / "training_dataset.csv"

N_BINS = 4
ABSENCE_RATIO = 1.0   # 1:1 against presences
BUFFER_M = 150.0
SEED = 42


for p in (DEM_PATH, SLOPE_PATH, DIST_PATH, SITES_PATH):
    if not p.is_file():
        raise FileNotFoundError(f"Missing input {p}; run scripts 01-03 first.")

with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(np.float64)
    transform = src.transform
    crs = src.crs
    h, w = dem.shape
    cx, cy = transform.a, -transform.e

with rasterio.open(SLOPE_PATH) as src:
    slope = src.read(1).astype(np.float64)
with rasterio.open(DIST_PATH) as src:
    dist_water = src.read(1).astype(np.float64)


# --- presences ------------------------------------------------------
ds = ogr.Open(str(SITES_PATH))
layer = ds.GetLayer(0)
presences = []
for feat in layer:
    g = feat.GetGeometryRef()
    presences.append({
        "site_id": feat.GetField("site_id"),
        "is_noise": feat.GetField("is_noise"),
        "x": g.GetX(),
        "y": g.GetY(),
        "elev_m": feat.GetField("elev_m"),
        "slope_deg": feat.GetField("slope_deg"),
        "dist_water_m": feat.GetField("dist_water_m"),
    })
ds = None

n_pres = len(presences)
n_abs = int(round(n_pres * ABSENCE_RATIO))
print(f"presences: {n_pres}, target absences: {n_abs}")


# --- exclusion mask -------------------------------------------------
pres_mask = np.zeros_like(dem, dtype=bool)
for rec in presences:
    col = int((rec["x"] - transform.c) / transform.a)
    row = int((rec["y"] - transform.f) / transform.e)
    if 0 <= row < h and 0 <= col < w:
        pres_mask[row, col] = True

dist_to_pres = distance_transform_edt(~pres_mask, sampling=(cy, cx))
eligible = dist_to_pres > BUFFER_M
n_elig = int(eligible.sum())
print(f"eligible cells: {n_elig} ({100.0 * n_elig / dem.size:.1f}% of grid)")


# --- environmental bins --------------------------------------------
def edges(values, n):
    qs = np.linspace(0.0, 100.0, n + 1)[1:-1]
    return np.percentile(values, qs)

ee = edges(dem[eligible], N_BINS)
se = edges(slope[eligible], N_BINS)
we = edges(dist_water[eligible], N_BINS)

bin_e = np.digitize(dem, ee)
bin_s = np.digitize(slope, se)
bin_w = np.digitize(dist_water, we)
bin_id = bin_e * (N_BINS ** 2) + bin_s * N_BINS + bin_w

flat_bin = bin_id.ravel()
flat_elig = eligible.ravel()
elig_idx = np.where(flat_elig)[0]
elig_bins = flat_bin[elig_idx]

uniq, counts = np.unique(elig_bins, return_counts=True)
print(f"occupied env bins: {len(uniq)} / {N_BINS ** 3}")


# --- proportional allocation ---------------------------------------
target = n_abs * counts / counts.sum()
alloc = np.floor(target).astype(int)
leftover = n_abs - int(alloc.sum())
frac = target - alloc
order = np.argsort(-frac)
alloc[order[:leftover]] += 1
assert alloc.sum() == n_abs


# --- sample within bins --------------------------------------------
rng = np.random.default_rng(SEED)
sampled = []
for b, n in zip(uniq, alloc):
    if n == 0:
        continue
    in_bin = elig_idx[elig_bins == b]
    n = min(n, len(in_bin))
    sampled.append(rng.choice(in_bin, size=n, replace=False))

sampled = np.concatenate(sampled)
rng.shuffle(sampled)
abs_rows, abs_cols = np.unravel_index(sampled, dem.shape)

jr = rng.uniform(-0.5, 0.5, size=n_abs)
jc = rng.uniform(-0.5, 0.5, size=n_abs)
abs_xs = transform.a * (abs_cols + 0.5 + jc) + transform.c
abs_ys = transform.e * (abs_rows + 0.5 + jr) + transform.f

print(f"sampled {len(sampled)} pseudo-absences")


# --- write absences GPKG -------------------------------------------
driver = ogr.GetDriverByName("GPKG")
abs_str = str(ABSENCES_PATH)
if ABSENCES_PATH.exists():
    driver.DeleteDataSource(abs_str)
ads = driver.CreateDataSource(abs_str)

srs = osr.SpatialReference()
epsg = int(str(crs).split(":")[1])
srs.ImportFromEPSG(epsg)
alayer = ads.CreateLayer("pseudo_absences", srs, ogr.wkbPoint)

afields = [
    ("absence_id", ogr.OFTInteger),
    ("elev_m", ogr.OFTReal),
    ("slope_deg", ogr.OFTReal),
    ("dist_water_m", ogr.OFTReal),
    ("env_bin_id", ogr.OFTInteger),
]
for name, t in afields:
    alayer.CreateField(ogr.FieldDefn(name, t))

adefn = alayer.GetLayerDefn()
for i in range(n_abs):
    r, c = abs_rows[i], abs_cols[i]
    f = ogr.Feature(adefn)
    f.SetField("absence_id", int(i))
    f.SetField("elev_m", float(dem[r, c]))
    f.SetField("slope_deg", float(slope[r, c]))
    f.SetField("dist_water_m", float(dist_water[r, c]))
    f.SetField("env_bin_id", int(bin_id[r, c]))
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(float(abs_xs[i]), float(abs_ys[i]))
    f.SetGeometry(pt)
    alayer.CreateFeature(f)
    f = None
ads = None

print(f"wrote {ABSENCES_PATH}")


# --- training CSV ---------------------------------------------------
pres_rows = [{
    "point_id": f"P{i:04d}",
    "x": rec["x"],
    "y": rec["y"],
    "presence": 1,
    "elev_m": rec["elev_m"],
    "slope_deg": rec["slope_deg"],
    "dist_water_m": rec["dist_water_m"],
    "is_noise": int(rec["is_noise"]),
} for i, rec in enumerate(presences)]

abs_rows_list = [{
    "point_id": f"A{i:04d}",
    "x": float(abs_xs[i]),
    "y": float(abs_ys[i]),
    "presence": 0,
    "elev_m": float(dem[abs_rows[i], abs_cols[i]]),
    "slope_deg": float(slope[abs_rows[i], abs_cols[i]]),
    "dist_water_m": float(dist_water[abs_rows[i], abs_cols[i]]),
    "is_noise": 0,
} for i in range(n_abs)]

df = pd.DataFrame(pres_rows + abs_rows_list)
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df.to_csv(TRAINING_CSV, index=False)

print(f"wrote {TRAINING_CSV}")
print(f"rows={len(df)}, presence={int(df['presence'].sum())}, "
      f"absence={int((1 - df['presence']).sum())}")
