"""
协调式单轮敏感性分析 — 固定22维新特征集
============================================
数据源: N1_decayed_core_22d_clean.npz (已预提取, strict-NaN过滤, 9,618,893干净样本)
阶段1: k_over 优化（固定 K=5）
阶段2: K 值优化（使用最优 k_over）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_NPZ = str(PROJECT_ROOT / "N1_decayed_core_22d_clean.npz")
DEFAULT_OUTPUT_DIR = r"E:\julei\GPM_ERA5_Cluster_Project\results\hierarchical_clustering\sensitivity_22d"

NEW_FEATURE_INDICES = list(range(18)) + [32, 33, 36, 37]
NEW_FEATURE_NAME = "N1_decayed_core_22d"
NEW_FEATURE_DIM = len(NEW_FEATURE_INDICES)


def load_clean_data(npz_path: str, max_samples: Optional[int], random_state: int) -> np.ndarray:
    """加载预提取的干净数据，可选子采样。返回 (X, lat, lon)。"""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    lat = data["lat"]
    lon = data["lon"]
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        lat = lat[idx]
        lon = lon[idx]
    print(f"加载数据: {X.shape[0]:,} 样本, {X.shape[1]} 维")
    return X, lat, lon


def normalize(X: np.ndarray) -> np.ndarray:
    """MinMax 归一化 (不填充缺失值, 因为数据已 clean)。"""
    X = np.where(np.isinf(X), np.nan, X)
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = np.isnan(col)
        if np.any(mask):
            X[mask, i] = np.nanmedian(col) if not np.all(mask) else 0.0
    return MinMaxScaler().fit_transform(X)


def run_two_stage_clustering(
    X: np.ndarray,
    k_over: int,
    k_range: List[int],
    random_state: int = 42,
) -> Dict:
    X_norm = normalize(X)

    kmeans = KMeans(n_clusters=k_over, random_state=random_state, n_init=10, max_iter=300)
    stage1_labels = kmeans.fit_predict(X_norm)
    centroids = kmeans.cluster_centers_

    Z = linkage(centroids, method="ward")

    results = {}
    for n_clusters in k_range:
        centroids_labels = fcluster(Z, n_clusters, criterion="maxclust") - 1
        sample_labels = centroids_labels[stage1_labels]

        try:
            db_score = davies_bouldin_score(X_norm, sample_labels)
        except Exception:
            db_score = np.nan
        try:
            sil_score = silhouette_score(X_norm, sample_labels)
        except Exception:
            sil_score = np.nan
        try:
            ch_score = calinski_harabasz_score(X_norm, sample_labels)
        except Exception:
            ch_score = np.nan

        unique, counts = np.unique(sample_labels, return_counts=True)
        min_class_size = int(np.min(counts))
        max_class_size = int(np.max(counts))

        results[n_clusters] = {
            "davies_bouldin": db_score,
            "silhouette": sil_score,
            "calinski_harabasz": ch_score,
            "min_class_size": min_class_size,
            "max_class_size": max_class_size,
            "n_samples": X_norm.shape[0],
        }
        print(f"    K={n_clusters}: DB={db_score:.4f}, Sil={sil_score:.4f}, CH={ch_score:.1f}, min_class={min_class_size}")

    return results


def run_stage1_k_over_optimization(
    X: np.ndarray,
    output_dir: str,
    k_fixed: int = 5,
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("阶段1: k_over 优化")
    print("=" * 80)
    print(f"特征集: {NEW_FEATURE_NAME} ({NEW_FEATURE_DIM}维), 样本: {X.shape[0]:,}")
    print(f"固定 K: {k_fixed}")

    k_over_values = [50, 75, 100, 150, 200, 300]
    print(f"测试 k_over: {k_over_values}")

    rows = []
    for k_over in k_over_values:
        print(f"\n{'='*50}")
        print(f"  k_over = {k_over}")
        print(f"{'='*50}")
        res = run_two_stage_clustering(X, k_over=k_over, k_range=[k_fixed], random_state=random_state)
        m = res[k_fixed]
        constraint_ok = m["min_class_size"] >= 10
        rows.append({
            "k_over": k_over,
            "k_fixed": k_fixed,
            "davies_bouldin": m["davies_bouldin"],
            "silhouette": m["silhouette"],
            "calinski_harabasz": m["calinski_harabasz"],
            "min_class_size": m["min_class_size"],
            "constraint_satisfied": constraint_ok,
        })

    df = pd.DataFrame(rows)
    valid = df[df["constraint_satisfied"] == True]
    if valid.empty:
        print("警告: 无满足约束的结果，放宽约束选择")
        valid = df

    best_idx = valid["davies_bouldin"].idxmin()
    best_k_over = int(valid.loc[best_idx, "k_over"])
    print(f"\n{'='*60}")
    print(f"阶段1 结论: 最优 k_over={best_k_over}, DB={valid.loc[best_idx, 'davies_bouldin']:.3f}")
    print(f"{'='*60}")

    stage1_out = os.path.join(output_dir, "stage1_k_over_optimization")
    os.makedirs(stage1_out, exist_ok=True)
    df.to_csv(os.path.join(stage1_out, "stage1_results.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df["k_over"], df["davies_bouldin"], "o-", linewidth=2, markersize=8)
    ax1.axvline(x=best_k_over, color="red", linestyle="--", label=f"Optimal k_over={best_k_over}")
    ax1.set_xlabel("k_over"); ax1.set_ylabel("Davies-Bouldin"); ax1.set_title("Stage1: k_over vs DB")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(df["k_over"], df["silhouette"], "o-", linewidth=2, markersize=8, color="green")
    ax2.axvline(x=best_k_over, color="red", linestyle="--")
    ax2.set_xlabel("k_over"); ax2.set_ylabel("Silhouette"); ax2.set_title("Stage1: k_over vs Silhouette")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(stage1_out, "stage1_k_over_curves.png"), dpi=150)
    plt.close()

    return best_k_over, df


def run_stage2_k_optimization(
    X: np.ndarray,
    best_k_over: int,
    output_dir: str,
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("阶段2: 最终 K 值优化")
    print("=" * 80)
    print(f"特征集: {NEW_FEATURE_NAME} ({NEW_FEATURE_DIM}维), k_over={best_k_over}")

    k_range = [3, 4, 5, 6, 7, 8, 9, 10]
    print(f"测试 K: {k_range}")

    res = run_two_stage_clustering(X, k_over=best_k_over, k_range=k_range, random_state=random_state)

    rows = []
    for k, m in res.items():
        rows.append({"K": k, "davies_bouldin": m["davies_bouldin"], "silhouette": m["silhouette"],
                     "calinski_harabasz": m["calinski_harabasz"], "min_class_size": m["min_class_size"],
                     "max_class_size": m["max_class_size"]})
    df = pd.DataFrame(rows)

    best_db_idx = df["davies_bouldin"].idxmin()
    best_k_db = int(df.loc[best_db_idx, "K"])
    best_sil_idx = df["silhouette"].idxmax()
    best_k_sil = int(df.loc[best_sil_idx, "K"])

    if best_k_db == best_k_sil:
        best_k = best_k_db; method = "DB指数和轮廓系数一致"
    elif abs(best_k_db - best_k_sil) <= 2:
        best_k = 5 if 5 in [best_k_db, best_k_sil] else best_k_db
        method = f"综合判断 (DB推荐{best_k_db}, Sil推荐{best_k_sil})"
    else:
        best_k = best_k_db; method = "优先DB指数"

    print(f"\n 最优 K={best_k} ({method})")
    stage2_out = os.path.join(output_dir, "stage2_k_optimization")
    os.makedirs(stage2_out, exist_ok=True)
    df.to_csv(os.path.join(stage2_out, "stage2_results.csv"), index=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color, label in [(ax1, "davies_bouldin", "b", "Davies-Bouldin"),
                                     (ax2, "silhouette", "g", "Silhouette"),
                                     (ax3, "calinski_harabasz", "r", "Calinski-Harabasz")]:
        ax.plot(df["K"], df[col], f"{color}o-", linewidth=2, markersize=8)
        ax.axvline(x=best_k, color="red", linestyle="--")
        ax.set_xlabel("K"); ax.set_ylabel(label); ax.set_title(f"{label} vs K"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(stage2_out, "stage2_evaluation_curves.png"), dpi=150)
    plt.close()

    return best_k, df


def run_final_clustering(
    X: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    best_k_over: int, best_k: int, output_dir: str, random_state: int = 42,
):
    print("\n" + "=" * 80)
    print("最终聚类: 使用最优组合")
    print(f"特征集: {NEW_FEATURE_NAME}, k_over={best_k_over}, K={best_k}, 样本={X.shape[0]:,}")
    print("=" * 80)

    X_norm = normalize(X)
    kmeans = KMeans(n_clusters=best_k_over, random_state=random_state, n_init=10, max_iter=300)
    stage1_labels = kmeans.fit_predict(X_norm)
    centroids = kmeans.cluster_centers_
    Z = linkage(centroids, method="ward")
    sample_labels = fcluster(Z, best_k, criterion="maxclust") - 1
    sample_labels = sample_labels[stage1_labels]

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"cluster_results_K{best_k}.npz")
    np.savez_compressed(result_file, X=X, labels=sample_labels, lat=lat, lon=lon, n_clusters=best_k)
    print(f"结果保存: {result_file}")

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"feature_set": NEW_FEATURE_NAME, "feature_indices": NEW_FEATURE_INDICES,
                   "feature_dim": NEW_FEATURE_DIM, "k_over": best_k_over, "best_k": best_k,
                   "n_samples": int(X.shape[0])}, f, indent=2)


def generate_summary_report(output_dir, stage1_df, stage2_df, best_k_over, best_k):
    lines = []
    lines.append("# 单轮敏感性分析报告 (22维新特征集)\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"## 特征集: {NEW_FEATURE_NAME} ({NEW_FEATURE_DIM}维)\n")
    lines.append(f"索引: {NEW_FEATURE_INDICES}\n\n")
    lines.append("## 最优参数\n")
    lines.append(f"| 参数 | 最优值 |\n|------|--------|\n| k_over | {best_k_over} |\n| K | {best_k} |\n\n")
    lines.append("## 阶段1: k_over 优化\n")
    lines.append("| k_over | DB | Silhouette | CH | min_class |\n|--------|-----|------------|-----|----------|\n")
    for _, row in stage1_df.iterrows():
        c = "✓" if row["constraint_satisfied"] else "✗"
        lines.append(f"| {row['k_over']} | {row['davies_bouldin']:.3f} | {row['silhouette']:.3f} | {row['calinski_harabasz']:.1f} | {row['min_class_size']} {c} |\n")
    lines.append(f"\n**最优 k_over**: {best_k_over}\n\n")
    lines.append("## 阶段2: K 值优化\n")
    lines.append("| K | DB | Silhouette | CH | min_class | max_class |\n|---|-----|------------|-----|----------|----------|\n")
    for _, row in stage2_df.iterrows():
        m = " ← 最优" if row["K"] == best_k else ""
        lines.append(f"| {row['K']} | {row['davies_bouldin']:.3f} | {row['silhouette']:.3f} | {row['calinski_harabasz']:.1f} | {row['min_class_size']} | {row['max_class_size']}{m} |\n")
    lines.append(f"\n**最优 K**: {best_k}\n")
    with open(os.path.join(output_dir, "SENSITIVITY_REPORT.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n报告已保存: {os.path.join(output_dir, 'SENSITIVITY_REPORT.md')}")


def main():
    parser = argparse.ArgumentParser(description="单轮敏感性分析 (22维新特征集)")
    parser.add_argument("--input-npz", type=str, default=DEFAULT_NPZ, help=f"预提取数据 (默认: {DEFAULT_NPZ})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2], help="起始阶段 (1=k_over, 2=K)")
    parser.add_argument("--skip-final", action="store_true", help="跳过最终聚类")
    parser.add_argument("--max-samples", type=int, default=200000, help="灵敏度扫描采样数 (默认: 200000, None=全量)")
    parser.add_argument("--max-samples-final", type=int, default=None, help="最终聚类采样数 (默认: None=全量)")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 80)
    print("单轮敏感性分析 — 22维新特征集 (预提取数据)")
    print(f"数据: {args.input_npz}")
    print(f"输出: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载分析用数据（子采样）
    X_scan, _, _ = load_clean_data(args.input_npz, args.max_samples, args.random_state)

    # 阶段1
    if args.start_stage <= 1:
        best_k_over, stage1_df = run_stage1_k_over_optimization(X_scan, args.output_dir, random_state=args.random_state)
        with open(os.path.join(args.output_dir, "best_k_over.txt"), "w") as f:
            f.write(str(best_k_over))
    else:
        with open(os.path.join(args.output_dir, "best_k_over.txt"), "r") as f:
            best_k_over = int(f.read().strip())
        stage1_df = pd.read_csv(os.path.join(args.output_dir, "stage1_k_over_optimization", "stage1_results.csv"))
        print(f"\n从阶段1继续: k_over={best_k_over}")

    # 阶段2
    if args.start_stage <= 2:
        best_k, stage2_df = run_stage2_k_optimization(X_scan, best_k_over, args.output_dir, random_state=args.random_state)
        with open(os.path.join(args.output_dir, "best_k.txt"), "w") as f:
            f.write(str(best_k))
    else:
        with open(os.path.join(args.output_dir, "best_k.txt"), "r") as f:
            best_k = int(f.read().strip())
        stage2_df = pd.read_csv(os.path.join(args.output_dir, "stage2_k_optimization", "stage2_results.csv"))
        print(f"\n从阶段2继续: K={best_k}")

    generate_summary_report(args.output_dir, stage1_df, stage2_df, best_k_over, best_k)

    # 最终聚类（全量数据）
    if not args.skip_final:
        X_full, lat_full, lon_full = load_clean_data(args.input_npz, args.max_samples_final, args.random_state)
        run_final_clustering(X_full, lat_full, lon_full, best_k_over, best_k,
                            os.path.join(args.output_dir, "final_clustering"), random_state=args.random_state)

    print("\n" + "=" * 80)
    print("敏感性分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()