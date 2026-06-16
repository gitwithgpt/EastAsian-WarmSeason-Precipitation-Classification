"""
GPM新特征集数据质量检查
特征集: 索引 0-17, 32, 33, 36, 37 (22维)

参数列表:
  0  precipRateNearSurface    近地面降水率          mm/hr
  1  heightStormTop           回波顶高度             m
  2  heightZeroDeg            0℃等温层高度           m
  3  free_bottom_height       自由底层高度            m
  4  zku_ns                   近地面Ku反射率(衰减后)  dBZ
  5  zka_ns                   近地面Ka反射率(衰减后)  dBZ
  6  dfr_ns                   近地面双频比(衰减后)    dB
  7  zku_max                  柱内Ku最大反射率(衰减后) dBZ
  8  zka_max                  柱内Ka最大反射率(衰减后) dBZ
  9  dfr_max                  柱内最大双频比(衰减后)  dB
 10  slope_zku                Ku反射率垂直斜率        dBZ/km
 11  slope_zka                Ka反射率垂直斜率        dBZ/km
 12  slope_dfr                双频比斜率              dB/km
 13  zku_max_height           Ku最大值高度            m
 14  zka_max_height           Ka最大值高度            m
 15  dfr_max_height           DFR最大值高度           m
 16  ipl                      冰相层厚度              m
 17  lpl                      液相层厚度              m
 32  heightBB                 融化层高度              km
 33  bb_thickness             融化层厚度              km
 36  dm_column_max            柱内最大Dm              mm
 37  nw_column_max            柱内最大Nw              log10(m^-3 mm^-1)

输出:
- new_featureset_quality.csv    逐文件逐列的质量报告
- new_featureset_quality.txt    汇总报告
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

import numpy as np
import pandas as pd

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cluster_hierarchical_kmeans_447 import find_matched_nearest_files

NEW_FEATURE_INDICES = list(range(18)) + [32, 33, 36, 37]

NEW_FEATURE_NAMES = [
    "precipRateNearSurface",
    "heightStormTop",
    "heightZeroDeg",
    "free_bottom_height",
    "zku_ns",
    "zka_ns",
    "dfr_ns",
    "zku_max",
    "zka_max",
    "dfr_max",
    "slope_zku",
    "slope_zka",
    "slope_dfr",
    "zku_max_height",
    "zka_max_height",
    "dfr_max_height",
    "ipl",
    "lpl",
    "heightBB",
    "bb_thickness",
    "dm_column_max",
    "nw_column_max",
]

NEW_FEATURE_UNITS = [
    "mm/hr", "m", "m", "m",
    "dBZ", "dBZ", "dB",
    "dBZ", "dBZ", "dB",
    "dBZ/km", "dBZ/km", "dB/km",
    "m", "m", "m",
    "m", "m",
    "km", "km",
    "mm", "log10(m^-3 mm^-1)",
]

NEW_FEATURE_EXPECTED_RANGE = [
    (0.0, 300.0),
    (0, 25000),
    (0, 7000),
    (0, 5000),
    (-30, 60),
    (-30, 50),
    (-10, 30),
    (-30, 70),
    (-30, 60),
    (-10, 35),
    (-10, 10),
    (-10, 10),
    (-10, 10),
    (0, 20000),
    (0, 15000),
    (0, 15000),
    (0, 20000),
    (0, 7000),
    (0, 6),
    (0, 2),
    (0, 5),
    (1, 6),
]

DEFAULT_MATCHED_NEAREST_INPUT_DIR = r"E:\julei\GPM_ERA5_Cluster_Project\results\matched_nearest"

OUTPUT_DIR = PROJECT_ROOT / "run" / "hierarchical_clustering"
OUTPUT_CSV = str(OUTPUT_DIR / "new_featureset_quality.csv")
OUTPUT_TXT = str(OUTPUT_DIR / "new_featureset_quality.txt")


def load_file_data(npz_path: str, feature_indices: List[int]) -> Tuple[np.ndarray, int, int, int]:
    block_data_list = []
    try:
        data = np.load(npz_path, allow_pickle=True)
        data_dict = dict(data)
    except Exception as e:
        print(f"  ERROR: 无法加载 {npz_path}: {e}")
        return np.empty((0, len(feature_indices))), 0, 0, 0

    block_keys = sorted([
        k for k in data_dict.keys()
        if k.startswith("block_")
        and not k.startswith("lat_")
        and not k.startswith("lon_")
        and not k.startswith("timestamp_")
        and not k.startswith("meta_")
    ])

    total_samples = 0
    total_nan = 0

    for block_key in block_keys:
        block_data = data_dict[block_key]
        if block_data.ndim != 2:
            continue

        max_idx = max(feature_indices)
        if block_data.shape[1] <= max_idx:
            continue

        selected = block_data[:, feature_indices]
        block_data_list.append(selected)
        total_samples += selected.shape[0]
        total_nan += np.sum(np.isnan(selected))

    if not block_data_list:
        return np.empty((0, len(feature_indices))), 0, 0, 0

    features = np.concatenate(block_data_list, axis=0)
    return features, total_samples, total_nan


def check_file_quality(npz_path: str) -> Dict:
    basename = os.path.basename(npz_path)

    features, total_samples, total_nan = load_file_data(npz_path, NEW_FEATURE_INDICES)

    if features.shape[0] == 0:
        return {
            "file": basename,
            "n_samples": 0,
            "n_total_nan": 0,
            "nan_ratio_total": 1.0,
        }

    col_stats = {}
    for i, name in enumerate(NEW_FEATURE_NAMES):
        col = features[:, i]
        nan_count = int(np.sum(np.isnan(col)))
        nan_ratio = nan_count / len(col) if len(col) > 0 else 1.0
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            col_stats[name] = {
                "nan_count": nan_count,
                "nan_ratio": nan_ratio,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "p01": np.nan,
                "p99": np.nan,
                "samples_with_any_nan": total_samples,
            }
            continue

        col_stats[name] = {
            "nan_count": nan_count,
            "nan_ratio": nan_ratio,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "median": float(np.median(valid)),
            "p01": float(np.percentile(valid, 1)),
            "p99": float(np.percentile(valid, 99)),
        }

    any_nan_per_row = np.isnan(features).any(axis=1)
    samples_clean = int(np.sum(~any_nan_per_row))
    samples_dirty = int(np.sum(any_nan_per_row))
    clean_ratio = samples_clean / total_samples if total_samples > 0 else 0.0

    return {
        "file": basename,
        "n_samples": total_samples,
        "n_total_nan": total_nan,
        "nan_ratio_total": total_nan / (total_samples * len(NEW_FEATURE_NAMES)) if total_samples > 0 else 1.0,
        "samples_clean": samples_clean,
        "samples_dirty": samples_dirty,
        "samples_clean_ratio": clean_ratio,
        "columns": col_stats,
    }


def main():
    print("=" * 70)
    print("GPM 新特征集 (22维) 数据质量检查")
    print(f"特征索引: {NEW_FEATURE_INDICES}")
    print(f"特征维度: {len(NEW_FEATURE_INDICES)}")
    print(f"数据目录: {DEFAULT_MATCHED_NEAREST_INPUT_DIR}")
    print("=" * 70)

    files = find_matched_nearest_files(DEFAULT_MATCHED_NEAREST_INPUT_DIR)
    print(f"\n找到 {len(files)} 个文件\n")

    all_results = []
    per_column_accum = {
        name: {"nan_total": 0, "samples_total": 0, "valid_values": []}
        for name in NEW_FEATURE_NAMES
    }

    total_all_samples = 0
    total_all_nan = 0
    total_clean_samples = 0
    total_dirty_samples = 0

    for idx, fp in enumerate(files):
        basename = os.path.basename(fp)
        print(f"[{idx+1}/{len(files)}] {basename} ...", end=" ", flush=True)

        result = check_file_quality(fp)
        all_results.append(result)

        n = result["n_samples"]
        total_all_samples += n
        total_all_nan += result["n_total_nan"]
        total_clean_samples += result["samples_clean"]
        total_dirty_samples += result["samples_dirty"]

        if "columns" in result:
            for name, stats in result["columns"].items():
                per_column_accum[name]["nan_total"] += stats["nan_count"]
                per_column_accum[name]["samples_total"] += n

        print(f"{n:,} samples | dirty={result['samples_dirty']:,} ({100-result['samples_clean_ratio']*100:.1f}%)")

    raw_rows = []
    for r in all_results:
        if "columns" not in r:
            continue
        for name in NEW_FEATURE_NAMES:
            if name not in r["columns"]:
                continue
            s = r["columns"][name]
            raw_rows.append({
                "file": r["file"],
                "n_samples": r["n_samples"],
                "feature": name,
                "nan_count": s["nan_count"],
                "nan_ratio": s["nan_ratio"],
                "min": s["min"],
                "max": s["max"],
                "mean": s["mean"],
                "std": s["std"],
                "median": s["median"],
                "p01": s["p01"],
                "p99": s["p99"],
            })

    df = pd.DataFrame(raw_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n逐列详细报告已保存: {OUTPUT_CSV}")

    lines = []
    lines.append("=" * 80)
    lines.append("GPM 新特征集 (22维) 数据质量汇总报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("## 特征集定义")
    lines.append("")
    lines.append(f"  索引: {NEW_FEATURE_INDICES}")
    lines.append(f"  维度: {len(NEW_FEATURE_INDICES)}")
    lines.append("  说明: GPM衰减后特征(0-17) + 融化层(32-33) + 柱内最大雨滴谱(36-37)")
    lines.append("  排除: 衰减前特征(18-29)、预留位(30-31)、2.5km雨滴谱(34-35)")
    lines.append("")

    lines.append("## 总体数据概况")
    lines.append("")
    lines.append(f"  文件总数:          {len(files)}")
    lines.append(f"  总样本数:          {total_all_samples:>15,}")
    lines.append(f"  总NaN数:           {total_all_nan:>15,}")
    lines.append(f"  全局NaN比例:       {total_all_nan/(total_all_samples*len(NEW_FEATURE_NAMES))*100:>14.2f}%")
    lines.append(f"  完全干净样本:      {total_clean_samples:>15,} ({total_clean_samples/total_all_samples*100:.2f}%)")
    lines.append(f"  含NaN样本:         {total_dirty_samples:>15,} ({total_dirty_samples/total_all_samples*100:.2f}%)")
    lines.append("")

    lines.append("## 逐列质量报告")
    lines.append("")
    header = f"  {'特征名':<24s} {'NaN数':>10s} {'NaN%':>7s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s} {'预期范围':>20s}"
    lines.append(header)
    lines.append("  " + "-" * 110)

    for i, name in enumerate(NEW_FEATURE_NAMES):
        acc = per_column_accum[name]
        nan_total = acc["nan_total"]
        samples_total = acc["samples_total"]
        nan_ratio_col = nan_total / samples_total if samples_total > 0 else 1.0

        col_mask = df["feature"] == name
        valid_df = df.loc[col_mask, ["min", "max", "mean", "std"]].dropna()

        if valid_df.empty:
            c_min, c_max, c_mean, c_std = "-", "-", "-", "-"
        else:
            c_min = f"{valid_df['min'].min():.2f}"
            c_max = f"{valid_df['max'].max():.2f}"
            c_mean = f"{valid_df['mean'].mean():.2f}"
            c_std = f"{valid_df['std'].mean():.2f}"

        expected = NEW_FEATURE_EXPECTED_RANGE[i]
        expected_str = f"[{expected[0]}, {expected[1]}]"

        lines.append(
            f"  {name:<24s} {nan_total:>10,} {nan_ratio_col*100:>6.2f}% "
            f"{c_min:>10s} {c_max:>10s} {c_mean:>10s} {c_std:>10s} {expected_str:>20s}"
        )

    lines.append("")
    lines.append("## 逐文件质量报告")
    lines.append("")
    lines.append(f"  {'文件名':<35s} {'样本数':>10s} {'干净%':>7s} {'含NaN%':>7s}")
    lines.append("  " + "-" * 65)

    nan_warnings = []
    for r in all_results:
        if r["n_samples"] == 0:
            continue
        clean_pct = r["samples_clean_ratio"] * 100
        dirty_pct = (1 - r["samples_clean_ratio"]) * 100
        flag = " !!!" if dirty_pct > 10 else ""
        lines.append(
            f"  {r['file']:<35s} {r['n_samples']:>10,} {clean_pct:>6.1f}% {dirty_pct:>6.1f}%{flag}"
        )
        if dirty_pct > 10:
            nan_warnings.append(f"  [{r['file']}] 含NaN样本比例 {dirty_pct:.1f}% > 10%")

    lines.append("")
    lines.append("## 质量评估结论")
    lines.append("")

    overall_clean_ratio = total_clean_samples / total_all_samples * 100 if total_all_samples > 0 else 0

    if nan_warnings:
        lines.append(f"  WARNING: 发现 {len(nan_warnings)} 个文件含NaN样本比例 > 10%:")
        lines.extend(nan_warnings)
    else:
        lines.append("  PASS: 所有文件含NaN样本比例均在 10% 以内")

    lines.append("")

    high_nan_cols = []
    for i, name in enumerate(NEW_FEATURE_NAMES):
        acc = per_column_accum[name]
        if acc["samples_total"] > 0:
            col_nan_ratio = acc["nan_total"] / acc["samples_total"]
            if col_nan_ratio > 0.10:
                high_nan_cols.append(f"  {name}: {col_nan_ratio*100:.1f}% NaN")
    if high_nan_cols:
        lines.append("  WARNING: 以下列NaN比例 > 10%:")
        lines.extend(high_nan_cols)
    else:
        lines.append("  PASS: 所有特征列NaN比例均在 10% 以内")

    lines.append("")
    lines.append(f"  整体干净样本比例: {overall_clean_ratio:.2f}%")
    if overall_clean_ratio > 95:
        strategy = "None (中位数填充即可)"
    elif overall_clean_ratio > 80:
        strategy = "'strict' 或中位数填充均可"
    else:
        strategy = "'strict' (删除含NaN样本)"
    lines.append(f"  建议聚类时使用的 nan_filter 策略: {strategy}")
    lines.append("")

    report_text = "\n".join(lines)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n汇总报告已保存: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()