# East Asian Warm-Season Precipitation Classification

> **Objective classification of East Asian warm-season precipitation using GPM DPR and two-stage hierarchical k-means clustering**
>
> Target journal: *Atmospheric Research* (Elsevier, ISSN 0169-8095)

---

## Overview

This repository contains the code, data, and figures supporting the manuscript:

**"Objective Classification of East Asian Warm-Season Precipitation Based on GPM DPR: A Two-Stage Hierarchical K-Means Clustering Approach"**

The study identifies four major precipitation types over the East Asian monsoon region during the warm season (April–September) using over ten years (2014–2024) of GPM DPR observations. A two-stage hierarchical k-means clustering framework with 22-dimensional microphysical parameters is employed, and independent physical validation is performed using ERA5 reanalysis environmental fields.

---

## Repository Structure

```
├── code/
│   ├── clustering/                    # Two-stage hierarchical k-means clustering
│   │   ├── cluster_hierarchical_kmeans_447.py       # Main clustering script (2-stage)
│   │   ├── cluster_hierarchical_kmeans_447-2.py   # Alternative clustering variant
│   │   ├── extract_22d_featureset.py              # Extract 22-dim feature subset from matched data
│   │   ├── plot_publication_quality.py            # Generate publication-quality clustering evaluation figures
│   │   ├── plot_clean_dendrogram.py               # Clean dendrogram visualization
│   │   ├── plot_bootstrap_stability.py            # Bootstrap stability analysis plots
│   │   ├── run_sensitivity_analysis.py            # Sensitivity analysis runner
│   │   ├── run_coordinated_sensitivity.py         # Coordinated sensitivity tests
│   │   ├── check_new_featureset_quality.py        # Feature quality checker
│   │   └── README.md                              # Clustering module documentation
│   ├── match/                         # GPM-ERA5 multi-GPU data matching framework
│   │   ├── match_nearest_multi_gpu_v2.py          # Main multi-GPU matching runner
│   │   ├── run_multi_gpu.py                         # Multi-GPU task orchestrator
│   │   ├── match_core.py                            # Core matching logic
│   │   ├── match_single.py                          # Single-file matching
│   │   ├── nearest_match.py                         # Nearest-neighbor matching algorithm
│   │   ├── match_pressure.py                        # Pressure-level matching
│   │   ├── distance_calc.py                         # Distance calculation utilities
│   │   └── _int_.py                                 # Module initialization
│   ├── multi_gpu/                     # Multi-GPU parallel processing utilities
│   │   ├── gpu_worker.py                            # GPU worker process
│   │   ├── gpu_worker_spatial.py                    # Spatial GPU worker
│   │   ├── task_distribute.py                       # Task distribution logic
│   │   ├── task_distribute_spatial.py              # Spatial task distribution
│   │   ├── memory_control.py                        # GPU memory management
│   │   └── _int_.py                                 # Module initialization
│   └── figure_generation/             # Figure generation scripts
│       ├── generate_supplementary_figures.py      # Generates S1, S2, S4, S8, S9
│       ├── generate_supplementary_figures_v2.py   # Generates S1, S2, S8, S9, S11
│       └── (generate_all_figures.py — main figures 1–6, to be added)
├── data/
│   ├── cluster4_microphysics_stats.csv            # Full microphysical statistics
│   ├── cluster4_environment_stats.csv             # Full environmental statistics
│   ├── cluster4_significance_tests.csv            # Complete 90-pair test results
│   ├── cluster4_physical_naming_candidates.csv  # Physical naming candidates
│   ├── Z_matrix.npy                               # Hierarchical linkage matrix (~32 KB)
│   ├── stability_bootstrap_results.csv            # Bootstrap ARI results (100 iterations)
│   ├── sample_classification_labels_1percent.csv  # 1% random sample (~96k rows)
│   ├── sample_classification_labels_0.1percent.csv # 0.1% random sample (~9.6k rows)
│   └── README.md                                  # Data directory documentation
├── figures/
│   ├── main/                                      # Figure 1–6 (PNG + PDF)
│   └── supplementary/                             # Figure S1–S11 (PNG + PDF)
├── manuscript/
│   ├── manuscript.md                              # Full manuscript (Markdown)
│   ├── manuscript.docx                            # Full manuscript (Word)
│   └── figure_table_inventory.md                  # Figure/table inventory & submission guide
├── README.md                                      # This file
├── LICENSE                                        # MIT License
├── .gitignore                                     # Git ignore rules
└── PUSH_GUIDE.md                                  # GitHub push instructions
```

---

## Data Availability

### Large datasets (not included in this repository)

| Dataset | Size | Source | Access |
|---------|------|--------|--------|
| GPM DPR Ku/Ka L2 V07 | ~TB | NASA GES DISC | https://disc.gsfc.nasa.gov/ |
| ERA5 Reanalysis | ~GB | ECMWF CDS | https://cds.climate.copernicus.eu/ |
| `cluster4_story_dataset.npz` | ~874 MB | Derived | Not included; can be reproduced from raw data using the provided code |
| `cluster4_story_dataset_plus_dsd25.npz` | ~905 MB | Derived | Not included; can be reproduced from raw data using the provided code |
| `matched_nearest_YYYY_MM.npz` | ~GB | Derived | Not included |

### Small datasets (included in this repository)

| File | Description | Rows | Size |
|------|-------------|------|------|
| `cluster4_microphysics_stats.csv` | Full microphysical statistics (mean, median, std, quartiles, z-score) | 4 clusters × 22 parameters | ~20 KB |
| `cluster4_environment_stats.csv` | Full environmental statistics | 4 clusters × 14 parameters | ~15 KB |
| `cluster4_significance_tests.csv` | All 90 pairwise test results (Mann-Whitney U, KS, Cliff's Delta) | 90 pairs | ~30 KB |
| `cluster4_physical_naming_candidates.csv` | Candidate physical names for each cluster | 4 clusters | ~5 KB |
| `Z_matrix.npy` | Hierarchical linkage matrix (Ward) | 150 × 4 | ~32 KB |
| `stability_bootstrap_results.csv` | Bootstrap ARI results (100 iterations, 200k subsamples) | K=3–8 | ~2 KB |
| `sample_classification_labels_1percent.csv` | 1% random sample of cluster labels (with lat/lon) | ~96,000 | ~3 MB |
| `sample_classification_labels_0.1percent.csv` | 0.1% random sample for quick preview | ~9,600 | ~300 KB |

### Sample classification labels

A **1% random sample** (96,188 rows, ~3 MB) of the 9.6 million classification labels is provided to enable readers to verify the clustering results without downloading the full dataset. The sample preserves the cluster distribution ratio (C0:24.0%, C1:29.1%, C2:11.8%, C3:35.1%). A smaller **0.1% sample** (9,618 rows) is also provided for quick preview. Both files contain sample index, latitude, longitude, and cluster label. Contact the corresponding author for the full label set.

---

## Code Modules

### 1. Clustering (`code/clustering/`)

The core two-stage hierarchical k-means clustering pipeline:

| Script | Purpose |
|--------|---------|
| `cluster_hierarchical_kmeans_447.py` | Main clustering script: k-means over-clustering → Ward hierarchical clustering → dendrogram evaluation |
| `extract_22d_featureset.py` | Extract 22-dimensional feature subset from matched GPM-ERA5 data |
| `plot_publication_quality.py` | Generate publication-quality clustering evaluation curves (DB, Silhouette, CH, ARI) |
| `plot_bootstrap_stability.py` | Bootstrap stability analysis visualization |
| `run_sensitivity_analysis.py` | Sensitivity analysis runner (feature subsets, k_over optimization) |
| `run_coordinated_sensitivity.py` | Coordinated sensitivity tests across multiple parameters |

### 2. Data Matching (`code/match/` + `code/multi_gpu/`)

The multi-GPU parallel GPM-ERA5 data matching framework:

| Script | Purpose |
|--------|---------|
| `match_nearest_multi_gpu_v2.py` | Main multi-GPU matching runner: orchestrates monthly GPM-ERA5 matching |
| `run_multi_gpu.py` | Multi-GPU task orchestrator and worker management |
| `match_core.py` | Core matching logic (nearest-neighbor spatial + temporal matching) |
| `nearest_match.py` | Nearest-neighbor matching algorithm implementation |
| `match_pressure.py` | Pressure-level variable matching (37 layers) |
| `distance_calc.py` | Distance calculation utilities |
| `gpu_worker.py` | GPU worker process for parallel matching |
| `task_distribute.py` | Task distribution and load balancing across GPUs |
| `memory_control.py` | GPU memory management and overflow prevention |

> **Note:** The data matching framework processes ~9.6 million precipitation samples across 66 months (2014–2024). Using 4 GPUs, the matching time is reduced from ~120 hours (single GPU) to ~35 hours (acceleration ratio ~3.4×).

### 3. Figure Generation (`code/figure_generation/`)

| Script | Generates |
|--------|-----------|
| `generate_supplementary_figures.py` | S1, S2, S4, S8, S9 |
| `generate_supplementary_figures_v2.py` | S1, S2, S8, S9, S11 |

> **Note:** `generate_all_figures.py` (main figures 1–6) is not yet included in this repository. Contact the corresponding author for the full figure generation pipeline.

---

## Reproducing the Analysis

### Requirements

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### Step 1: Data Matching (if starting from raw data)

```bash
cd code/match
python match_nearest_multi_gpu_v2.py --input-dir [GPM_DPR_DIR] --era5-dir [ERA5_DIR] --output-dir [MATCHED_DIR]
```

### Step 2: Feature Extraction

```bash
cd code/clustering
python extract_22d_featureset.py --input-dir [MATCHED_DIR] --output [FEATURESET_NPZ]
```

### Step 3: Clustering

```bash
python cluster_hierarchical_kmeans_447.py --feature-preset S2_radar_microphysics_full --n-clusters 3 4 5 6 7 8
```

### Step 4: Generate Publication Figures

```bash
python plot_publication_quality.py --npz [FEATURESET_NPZ] --output-dir [FIGURE_DIR]
```

### Step 5: Generate Supplementary Figures

```bash
cd code/figure_generation
python generate_supplementary_figures_v2.py
```

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{author2026classification,
  title={Objective Classification of East Asian Warm-Season Precipitation Based on GPM DPR: A Two-Stage Hierarchical K-Means Clustering Approach},
  journal={Atmospheric Research},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

The GPM DPR data are provided by NASA and JAXA. The ERA5 reanalysis data are provided by ECMWF. Both are subject to their respective data use policies.

---

## Contact

For questions about the code or data, please contact the corresponding author.

For the manuscript submission status, see the [manuscript](manuscript/) directory.
