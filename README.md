# archpredict

QGIS plugin for archaeological site prediction with machine learning.

The goal is an end-to-end no-code pipeline inside QGIS — from raster
ingestion through model training, probability surface generation,
spatially aware cross-validation, and iterative refinement as new sites
are recorded. Phase 1 (synthetic data + first RF prototype) is the
current focus.

## Stack

QGIS 3.34+ (developed against 3.44 on Linux), Python 3 with
scikit-learn, numpy, pandas, rasterio, joblib. Plugin will live as a
custom `QgsProcessingProvider` so each stage is callable from the
Processing Toolbox or Graphical Modeler.

## Layout

```
archpredict/
├── ArchPredict_Technical_Report.pdf   reference architecture document
├── data/                              local data (gitignored)
├── docs/
│   └── bitacora.md                    project journal
├── scripts/                           standalone scripts: synthetic
│                                      data + ML prototyping
└── README.md
```

The plugin source tree (`archpredict/{core,data,processing,ui,io}`)
will be added once the pipeline logic is validated outside of QGIS.

## Reproducing the synthetic pipeline

From any Python with rasterio + scipy + GDAL (or the QGIS Python
Console), run scripts in order:

```python
import os
os.chdir('/path/to/archpredict')        # or set ARCHPREDICT_ROOT
exec(open('scripts/01_create_synthetic_dem.py').read())
exec(open('scripts/02_derive_slope_and_distance.py').read())
exec(open('scripts/03_generate_synthetic_sites.py').read())
exec(open('scripts/04_pseudo_absences_and_training_data.py').read())
```

All scripts use `SEED = 42`.

## Project rules

- No standard k-fold CV. Always spatial block CV.
- Always emit an uncertainty raster alongside the probability raster.
- `random_state` set on every model for reproducibility.
- Probability map is a prioritisation tool, never confirmation of site
  presence.
- Datasets with fewer than 30 sites trigger small-dataset safeguards.

## Roadmap

See `ArchPredict_Technical_Report.pdf` §7 for the full plan; current
focus is the synthetic data pipeline → first RF training run with
spatial block CV.
