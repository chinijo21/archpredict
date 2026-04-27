# data/

Local data directory. Ignored by git except this README and `.gitkeep`.

Synthetic rasters and points are regenerable from the scripts under
`scripts/` (seed 42). Real data — DEMs, hydrology, soil, site
inventories — should not be committed: stays outside the repo or is
managed externally (DVC, git-lfs, Zenodo) when sharing is needed. Site
coordinates in particular are sensitive and need explicit
anonymisation before any redistribution.
