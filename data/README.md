# Data Directory

This directory contains the derived statistical data files used in the manuscript.

## Files Included

| File | Description | Size | Notes |
|------|-------------|------|-------|
| `cluster4_microphysics_stats.csv` | Full microphysical parameter statistics for all 4 clusters | ~20 KB | Mean, median, std, Q1, Q3, z-score for each of 22 parameters |
| `cluster4_environment_stats.csv` | Full environmental parameter statistics for all 4 clusters | ~15 KB | Mean, median, std, Q1, Q3, z-score for each of 14 parameters |
| `cluster4_significance_tests.csv` | Complete pairwise significance test results | ~30 KB | All 90 pair comparisons (6 pairs × 15 parameters); Mann-Whitney U, KS, Cliff's Delta |
| `cluster4_physical_naming_candidates.csv` | Physical naming candidates | ~5 KB | Candidate names and rationale for each cluster |
| `Z_matrix.npy` | Hierarchical clustering linkage matrix | ~32 KB | Ward linkage, 150×4 matrix for dendrogram generation |
| `stability_bootstrap_results.csv` | Bootstrap stability results | ~2 KB | 100 iterations, 200k subsamples, ARI for K=3–8 |

## Files NOT Included (Too Large)

| File | Approximate Size | How to Obtain |
|------|-----------------|---------------|
| `cluster4_story_dataset.npz` | ~874 MB | Derived from GPM DPR + ERA5 matching; can be reproduced using the data matching framework |
| `cluster4_story_dataset_plus_dsd25.npz` | ~905 MB | Extended version with DSD 25-bin profiles; can be reproduced using the data matching framework |
| `matched_nearest_YYYY_MM.npz` | ~GB total | Monthly matched GPM-ERA5 data; can be reproduced using the data matching framework |

## Raw Data Sources

- **GPM DPR**: https://disc.gsfc.nasa.gov/ (NASA GES DISC)
- **ERA5 Reanalysis**: https://cds.climate.copernicus.eu/ (ECMWF Copernicus CDS)

## Data Matching Framework

The multi-GPU parallel data matching framework that produces the `matched_nearest_*.npz` files is described in the manuscript (Section 2.4). Contact the corresponding author for the code.
