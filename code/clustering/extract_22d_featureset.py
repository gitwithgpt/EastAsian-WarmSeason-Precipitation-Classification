"""
从 66 个月的 matched_nearest 文件中提取 22 维新特征集
逐文件处理，严格保留每样本的 lat/lon/year/month，不丢失任何元信息。

输出:
- N1_decayed_core_22d_clean.npz  (X, lat, lon, year, month)
- N1_decayed_core_22d_clean_info.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cluster_hierarchical_kmeans_447 import find_matched_nearest_files

NEW_FEATURE_INDICES = list(range(18)) + [32, 33, 36, 37]
INPUT_DIR = r"E:\julei\GPM_ERA5_Cluster_Project\results\matched_nearest"
OUTPUT_NPZ = PROJECT_ROOT / "run" / "hierarchical_clustering" / "N1_decayed_core_22d_clean.npz"
OUTPUT_INFO = PROJECT_ROOT / "run" / "hierarchical_clustering" / "N1_decayed_core_22d_clean_info.json"


def extract_one_file(npz_path: str) -> Optional[Tuple]:
    """
    读取单个文件，提取 22 维 + lat + lon，应用 strict NaN 过滤。

    Returns: (X, lat, lon, year, month, raw_row_idx)
        raw_row_idx: 干净样本在原始文件（块拼接后）中的行索引，用于对齐 ERA5 等原始列
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        data_dict = dict(data)
    except Exception as e:
        print(f"  ERROR: 无法加载文件: {e}")
        return None

    block_keys = sorted([
        k for k in data_dict.keys()
        if k.startswith("block_")
        and not k.startswith("lat_")
        and not k.startswith("lon_")
        and not k.startswith("timestamp_")
        and not k.startswith("meta_")
    ])

    year = int(data_dict.get("meta_year", 0))
    month = int(data_dict.get("meta_month", 0))

    chunks_x, chunks_lat, chunks_lon, chunks_year, chunks_month = [], [], [], [], []
    chunks_raw_idx = []
    total_in = 0
    total_out = 0
    local_offset = 0  # 当前块在文件内的原始行偏移

    for bk in block_keys:
        block_data = data_dict[bk]
        if block_data.ndim != 2:
            continue
        if block_data.shape[1] <= max(NEW_FEATURE_INDICES):
            continue

        feats = block_data[:, NEW_FEATURE_INDICES]
        n_rows = feats.shape[0]
        block_id = bk.split("_")[1]
        lat_key = f"lat_block_{block_id}"
        lon_key = f"lon_block_{block_id}"

        lat_raw = data_dict.get(lat_key)
        lon_raw = data_dict.get(lon_key)

        if lat_raw is not None and len(lat_raw) == n_rows:
            lat_arr = lat_raw
        else:
            if lat_raw is not None and len(lat_raw) != n_rows:
                print(f"    WARN {bk}: lat长度={len(lat_raw)}, 特征行数={n_rows}, 用NaN填充")
            lat_arr = np.full(n_rows, np.nan)

        if lon_raw is not None and len(lon_raw) == n_rows:
            lon_arr = lon_raw
        else:
            if lon_raw is not None and len(lon_raw) != n_rows:
                print(f"    WARN {bk}: lon长度={len(lon_raw)}, 特征行数={n_rows}, 用NaN填充")
            lon_arr = np.full(n_rows, np.nan)

        # strict: 删除 22 维中任意含 NaN 的行
        keep = ~np.isnan(feats).any(axis=1)
        n_dropped = int((~keep).sum())
        total_in += n_rows
        total_out += n_dropped

        raw_indices = np.arange(local_offset, local_offset + n_rows, dtype=np.int64)
        local_offset += n_rows

        chunks_x.append(feats[keep])
        chunks_lat.append(lat_arr[keep])
        chunks_lon.append(lon_arr[keep])
        chunks_year.append(np.full(int(keep.sum()), year, dtype=np.int32))
        chunks_month.append(np.full(int(keep.sum()), month, dtype=np.int32))
        chunks_raw_idx.append(raw_indices[keep])

    if not chunks_x:
        return None

    X = np.concatenate(chunks_x, axis=0).astype(np.float32)
    lat = np.concatenate(chunks_lat, axis=0).astype(np.float32)
    lon = np.concatenate(chunks_lon, axis=0).astype(np.float32)
    yr = np.concatenate(chunks_year)
    mo = np.concatenate(chunks_month)
    raw_idx = np.concatenate(chunks_raw_idx)

    assert X.shape[0] == lat.shape[0] == lon.shape[0] == yr.shape[0] == mo.shape[0] == raw_idx.shape[0], (
        f"长度不一致: X={X.shape[0]}, lat={lat.shape[0]}, lon={lon.shape[0]}, "
        f"yr={yr.shape[0]}, mo={mo.shape[0]}, raw_idx={raw_idx.shape[0]}"
    )

    print(f"  {year}-{month:02d}: {total_in:,} -> {X.shape[0]:,} (删除 {total_out:,}, {total_out/total_in*100:.1f}%)")
    return X, lat, lon, yr, mo, raw_idx


def main():
    print("=" * 70)
    print("提取 22 维新特征集 (逐文件, strict NaN过滤)")
    print(f"索引: {NEW_FEATURE_INDICES}")
    print("=" * 70)

    files = find_matched_nearest_files(INPUT_DIR)
    print(f"找到 {len(files)} 个文件\n")

    all_x, all_lat, all_lon, all_year, all_month = [], [], [], [], []
    all_raw_row_idx = []
    all_file_idx = []
    grand_in, grand_out = 0, 0

    for idx, fp in enumerate(files):
        result = extract_one_file(fp)
        if result is None:
            continue
        x, lat, lon, yr, mo, raw_idx = result
        n_clean = x.shape[0]

        all_x.append(x)
        all_lat.append(lat)
        all_lon.append(lon)
        all_year.append(yr)
        all_month.append(mo)
        all_raw_row_idx.append(raw_idx)
        all_file_idx.append(np.full(n_clean, idx, dtype=np.int32))

    X = np.concatenate(all_x, axis=0)
    lat = np.concatenate(all_lat, axis=0)
    lon = np.concatenate(all_lon, axis=0)
    year = np.concatenate(all_year)
    month = np.concatenate(all_month)
    raw_row_idx = np.concatenate(all_raw_row_idx)
    file_idx = np.concatenate(all_file_idx)

    assert X.shape[0] == lat.shape[0] == lon.shape[0] == year.shape[0] == month.shape[0] == raw_row_idx.shape[0] == file_idx.shape[0]

    print(f"\n总计: {X.shape[0]:,} 干净样本, {X.shape[1]} 维")
    print(f"lat 范围: [{np.nanmin(lat):.1f}, {np.nanmax(lat):.1f}] (NaN: {np.isnan(lat).sum():,})")
    print(f"lon 范围: [{np.nanmin(lon):.1f}, {np.nanmax(lon):.1f}] (NaN: {np.isnan(lon).sum():,})")
    print(f"年份范围: [{year.min()}, {year.max()}]")
    print(f"月份分布: {np.unique(month)}")

    nan_total = np.isnan(X).sum()
    print(f"NaN残留: {nan_total} (应为 0)")

    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        lat=lat,
        lon=lon,
        year=year,
        month=month,
        file_idx=file_idx,
        raw_row_idx=raw_row_idx,
        files=sorted(files),
    )
    print(f"\n数据已保存: {OUTPUT_NPZ}")

    info = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_name": "N1_decayed_core_22d",
        "feature_indices": NEW_FEATURE_INDICES,
        "feature_dim": len(NEW_FEATURE_INDICES),
        "nan_filter": "strict",
        "n_samples": int(X.shape[0]),
        "n_files": len(files),
        "lat_range": [float(lat.min()), float(lat.max())],
        "lon_range": [float(lon.min()), float(lon.max())],
        "year_range": [int(year.min()), int(year.max())],
        "file_size_mb": round(os.path.getsize(OUTPUT_NPZ) / (1024 * 1024), 1),
        "arrays": [
            "X (float32, n×22)",
            "lat (float32)",
            "lon (float32)",
            "year (int32)",
            "month (int32)",
            "file_idx (int32) — 指向 files 列表的索引，定位原始 .npz 文件",
            "raw_row_idx (int64) — 在原始文件块拼接矩阵中的行号",
            "files (list[str]) — 原始文件路径列表",
        ],
        "usage_era5": (
            "恢复原始 ERA5 特征: "
            "1) 取 file_idx[i] 定位 files[file_idx[i]] "
            "2) 加载该 .npz 文件的全部 447 列 "
            "3) 取 raw_row_idx[i] 行，索引 47:447 即为 ERA5 特征"
        ),
    }
    with open(OUTPUT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"统计信息: {OUTPUT_INFO}")


if __name__ == "__main__":
    main()