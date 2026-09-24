# Szeged V-DEI Metamodel

Code companion to the Szeged urban-microclimate metamodeling study: a
dual-head 3D-CNN metamodel that emulates ENVI-met over two real urban sites
(street **canyon** and public **plaza**) in Szeged, Hungary, using a
**V-DEI** (Vertical Directional Exposure Index) spatial feature and per-point
sun-block series.

The raw inputs are two ENVI-met NetCDF simulations (~GB-scale per site).
Everything heavy is computed on Kaggle and versioned as the private Kaggle
datasets **`szeged-envimet-raw`** and **`szeged-vdei-processed`** (hash-sealed
with a blake2b manifest); this repository contains only code.

## Pipeline at a glance

| Stage | Scripts (`scripts/`) | What it does |
|---|---|---|
| Bootstrap / mount | 00–02 | Robust discovery of the Kaggle input mounts + integrity verify against the sealed manifest |
| Phase 2 audit | 03–05 | Raw NetCDF inspection, vertical grid + geometry masks, object-code / vegetation QC, facade-% reconciliation |
| S1: V-DEI core | 06–08 | Ray-casting prototype, azimuth-convention lock vs ENVI-met ShadowFlag, numba production batch (16 az × 9 el × 144 rays per air cell) |
| S2: dataset build | 09–22 | Sanity checks, bbox filter, targets (T, RH, wind, TKE, TMRT) + forcing + per-point sun-block series, 40 m spatial block split 70/15/15, K-window vertical index, SVF validation, facade targets, manifest sealing, packaging |
| S3: metamodel | 23–31 | `VDEIDatasetV2` loader, dual-head 3D-CNN (air + facade heads), training with AMP + deterministic seeds, full-pool test evaluation, multi-seed protocol (5 seeds × 2 sites), ensemble |
| S5: evaluation | 32–43, 83 | Block-bootstrap 95% CIs, classical ML baselines with hyperparameter tuning (CatBoost/XGBoost), config comparison, joint multi-site training |
| S6: transfer | 44 | Cross-site transfer learning (zero-shot + fine-tune) |
| Ablations | 46 | Component-removal ablation study |
| P6: paper figures | 48–82, 84–87 | Fact-check cells and Figures 6–18 (spatial maps, bootstrap CIs, temporal coherence, extreme tails, sensitivity/OAT) |

## Repository layout

```
├── scripts/                 # 87 numbered scripts, one per notebook cell (execution order)
├── notebooks/
│   └── final_script.ipynb   # full Kaggle notebook (outputs stripped; reference copy)
├── requirements.txt
├── LICENSE
└── README.md
```

Script names follow `cellNN_<stage>_<description>.py` where `NN` is the
original notebook cell index, so any script maps back 1:1 to the notebook.

## Getting started

The pipeline was written for **Kaggle notebooks** (paths `/kaggle/input`,
`/kaggle/working`, GPU T4). To reproduce:

1. Attach the private Kaggle datasets `szeged-envimet-raw` and
   `szeged-vdei-processed` as notebook inputs.
2. Run the bootstrap cell (`scripts/cell00_...py`) to verify mounts.
3. Follow the scripts in numeric order; cells are grouped by stage prefix in
   the table above. GPU stages (S3, P6 GPU variants) need the T4 accelerator
   enabled.

Locally, most CPU stages run as-is after `pip install -r requirements.txt`
(adjust the `/kaggle/...` paths, which are defined once in the bootstrap).

## Key methodological constants (locked in the pipeline)

- Ray set: 16 azimuths × 9 elevations = 144 directions per air cell, `d_max = 100 m`
- Azimuth convention: `az_grid = compass_az − ModelRotation` (locked against ENVI-met ShadowFlag)
- Split: 40 m spatial blocks, 70/15/15 train/val/test, seed 42
- Vertical context: K = 7 window with nearest-valid-air hole filling
- Targets: T, RH, wind speed, TKE, TMRT (air head); Twall, Qsens, SWabs, LWbal (facade head, X-facade system)

## Data availability

All simulation data are versioned as two **private** Kaggle datasets (not
redistributed with this repository):

| Dataset | Content | Format |
|---|---|---|
| `szeged-envimet-raw` | Raw ENVI-met NetCDF output for the two sites (`Urban_Canyon.nc`, `Main_Plaza.nc`), GB-scale each | `.nc` |
| `szeged-vdei-processed` | All 18 sealed S2 artifacts (9 per site) + manifest + diagnostics | `.npz`, `.csv`, `.png` |

Sealed artifacts per site (`01_Data/02_Processed/` in the processed dataset):

`geometry` · `vdei_raw` · `vdei_features` · `bbox` · `targets_forcing` ·
`split` · `kwindow` · `svf` · `facade_targets`

Every artifact is hash-sealed (blake2b-128) in
`01_Data/03_Metadata/s2_manifest.csv`, together with file size, key count,
array shapes and dtypes, and target NaN% / range checks. The bootstrap cells
(`scripts/cell02_...py`) re-verify all hashes on every fresh Kaggle session,
so any corrupted download fails fast before training.

Resulting sample pools (from the sealed split, `scripts/cell23_...py`):

| Site | Air points | Timesteps | Facade-adjacent | Train / Val / Test points |
|---|---|---|---|---|
| canyon | 2,610,840 | 25 | 16,753 | 1,844,220 / 352,340 / 414,280 |
| plaza | 929,799 | 49 | 9,566 | 645,898 / 145,476 / 138,425 |

Access: the datasets are private during peer review. Access requests can be
made via the corresponding author; on acceptance the processed dataset will
be made available.

## Citation

If you use this code, please cite the corresponding paper (citation to be
added on acceptance) and this repository.
