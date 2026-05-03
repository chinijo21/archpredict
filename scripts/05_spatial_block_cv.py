"""
Spatial block cross-validation + final Random Forest training.

Tiles the AOI into square blocks of BLOCK_SIZE_M, assigns each point
in training_dataset.csv to the block it falls into, then assigns
each block (round-robin after seeded shuffle) to one of N_FOLDS folds.
For each fold, trains an RF on the other folds' points and evaluates
AUC and max-TSS on the held-out fold. Reports per-fold and summary
metrics.

After CV, retrains a final RF on all 160 points and serialises it
together with metadata (sklearn version, hyperparameters, SHA256 of
the training CSV, seed, timestamp) for downstream prediction and for
the reproducibility report.

Outputs:
    models/rf_synthetic.joblib
    reports/cv_results_synthetic.json
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import rasterio
import sklearn
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve


PROJECT_ROOT = Path(os.environ.get("ARCHPREDICT_ROOT", os.getcwd())).resolve()
DATA = PROJECT_ROOT / "data"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA / "training_dataset.csv"
DEM_PATH = DATA / "synthetic_dem.tif"
MODEL_PATH = MODELS / "rf_synthetic.joblib"
CV_PATH = REPORTS / "cv_results_synthetic.json"

BLOCK_SIZE_M = 3000.0
N_FOLDS = 5
SEED = 42

FEATURES = ["elev_m", "slope_deg", "dist_water_m"]
TARGET = "presence"

RF_PARAMS = dict(
    n_estimators=500,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)


# --- load -----------------------------------------------------------
if not CSV_PATH.is_file():
    raise FileNotFoundError(f"{CSV_PATH} not found; run scripts 01-04 first.")
if not DEM_PATH.is_file():
    raise FileNotFoundError(f"{DEM_PATH} not found; needed for AOI extent.")

df = pd.read_csv(CSV_PATH)
n_pres = int(df[TARGET].sum())
n_abs = int((1 - df[TARGET]).sum())
print(f"loaded {len(df)} points: {n_pres} pres, {n_abs} abs")

# Quick sanity: too few sites?
if n_pres < 30:
    print(f"  WARN: only {n_pres} presences (< 30). Inflating min_samples_leaf.")
    RF_PARAMS["min_samples_leaf"] = 10


# --- mean nearest-neighbor distance among presences -----------------
pres_xy = df[df[TARGET] == 1][["x", "y"]].values
nn_dists, _ = cKDTree(pres_xy).query(pres_xy, k=2)
mean_nn = float(nn_dists[:, 1].mean())
print(f"mean NN distance (presences): {mean_nn:.1f} m  "
      f"-> block size {BLOCK_SIZE_M:.0f} m = {BLOCK_SIZE_M / mean_nn:.1f}x mean NN")


# --- block grid ----------------------------------------------------
with rasterio.open(DEM_PATH) as src:
    bounds = src.bounds
xmin, ymin = bounds.left, bounds.bottom

df["block_x"] = ((df["x"] - xmin) // BLOCK_SIZE_M).astype(int)
df["block_y"] = ((df["y"] - ymin) // BLOCK_SIZE_M).astype(int)
df["block"] = df["block_x"] * 10000 + df["block_y"]

unique_blocks = df["block"].unique()
n_blocks = len(unique_blocks)
print(f"{n_blocks} occupied blocks across the AOI")


# --- assign blocks to folds (shuffle + round-robin) ----------------
rng = np.random.default_rng(SEED)
shuffled = rng.permutation(unique_blocks)
fold_for_block = {int(b): int(i % N_FOLDS) for i, b in enumerate(shuffled)}
df["fold"] = df["block"].map(fold_for_block)

print("fold composition:")
fold_composition = []
for k in range(N_FOLDS):
    sub = df[df["fold"] == k]
    n_p = int(sub[TARGET].sum())
    n_a = int((1 - sub[TARGET]).sum())
    n_b = int(sub["block"].nunique())
    print(f"  fold {k}: {len(sub)} pts ({n_p} pres, {n_a} abs), {n_b} blocks")
    fold_composition.append({
        "fold": k, "n_points": int(len(sub)),
        "n_blocks": n_b, "n_presence": n_p, "n_absence": n_a,
    })


# --- spatial block CV ----------------------------------------------
def max_tss(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float((tpr - fpr).max())


cv_results = []
for k in range(N_FOLDS):
    test = df[df["fold"] == k]
    train = df[df["fold"] != k]

    n_test_pres = int(test[TARGET].sum())
    n_test_abs = int((1 - test[TARGET]).sum())

    if n_test_pres == 0 or n_test_abs == 0:
        print(f"fold {k}: SKIP (test has only one class)")
        cv_results.append({
            "fold": k, "skipped": True,
            "reason": "test fold has only one class",
            "n_train": int(len(train)), "n_test": int(len(test)),
            "n_test_presence": n_test_pres, "n_test_absence": n_test_abs,
        })
        continue

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(train[FEATURES].values, train[TARGET].values)
    proba = model.predict_proba(test[FEATURES].values)[:, 1]

    auc = float(roc_auc_score(test[TARGET].values, proba))
    tss = max_tss(test[TARGET].values, proba)

    print(f"fold {k}: AUC={auc:.4f}  TSS@max={tss:.4f}  "
          f"(train={len(train)}, test={len(test)})")
    cv_results.append({
        "fold": k, "skipped": False,
        "auc": auc, "tss_max": tss,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_test_presence": n_test_pres, "n_test_absence": n_test_abs,
    })

valid = [r for r in cv_results if not r["skipped"]]
aucs = [r["auc"] for r in valid]
tsss = [r["tss_max"] for r in valid]
summary = {
    "n_folds_total": N_FOLDS,
    "n_folds_evaluated": len(valid),
    "auc_mean": float(np.mean(aucs)) if aucs else None,
    "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
    "tss_mean": float(np.mean(tsss)) if tsss else None,
    "tss_std": float(np.std(tsss, ddof=1)) if len(tsss) > 1 else None,
}
if summary["auc_mean"] is not None:
    print(f"summary: AUC = {summary['auc_mean']:.3f} "
          f"(+/- {summary['auc_std']:.3f}), "
          f"TSS = {summary['tss_mean']:.3f} (+/- {summary['tss_std']:.3f})")
else:
    print("summary: no folds evaluated (all skipped)")


# --- final model on all data ---------------------------------------
print("training final model on full dataset...")
final_model = RandomForestClassifier(**RF_PARAMS)
final_model.fit(df[FEATURES].values, df[TARGET].values)

csv_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()[:16]

metadata = {
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "sklearn_version": sklearn.__version__,
    "rf_params": {**RF_PARAMS, "n_jobs": "all"},
    "features": FEATURES,
    "target": TARGET,
    "n_train_total": int(len(df)),
    "n_presence": n_pres,
    "n_absence": n_abs,
    "training_csv_path": str(CSV_PATH.relative_to(PROJECT_ROOT)),
    "training_csv_sha256_16": csv_hash,
    "block_size_m": BLOCK_SIZE_M,
    "n_folds": N_FOLDS,
    "seed": SEED,
}

joblib.dump({"model": final_model, "metadata": metadata}, MODEL_PATH)
print(f"wrote {MODEL_PATH}")


# --- save CV report ------------------------------------------------
report = {
    "metadata": metadata,
    "fold_composition": fold_composition,
    "fold_results": cv_results,
    "summary": summary,
    "feature_importances": dict(
        zip(FEATURES, [float(x) for x in final_model.feature_importances_])
    ),
}

with open(CV_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"wrote {CV_PATH}")
print(f"feature importances: {report['feature_importances']}")
