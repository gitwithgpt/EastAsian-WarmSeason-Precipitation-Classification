"""
从 N1 预提取数据重新生成干净树状图（不带 K 值横线）
聚类参数与 cluster_hierarchical_kmeans_447.py 一致: k_over=150, random_state=42
"""

import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NPZ_PATH = Path(__file__).resolve().parent / "N1_decayed_core_22d_clean.npz"
OUTPUT_PNG = r"E:\julei\GPM_ERA5_Cluster_Project\results\hierarchical_clustering\cluster_results_n1_K6\dendrogram_clean.png"

K_OVER = 150

print("加载数据...")
data = np.load(str(NPZ_PATH), allow_pickle=True)
X = data["X"].astype(np.float64)
print(f"样本: {X.shape[0]:,}, 维度: {X.shape[1]}")

print("MinMax 归一化...")
X = np.where(np.isinf(X), np.nan, X)
for i in range(X.shape[1]):
    col = X[:, i]
    mask = np.isnan(col)
    if np.any(mask):
        X[mask, i] = np.nanmedian(col) if not np.all(mask) else 0.0
X_norm = MinMaxScaler().fit_transform(X)

print(f"K-means 过度聚类 k={K_OVER}...")
kmeans = KMeans(n_clusters=K_OVER, random_state=42, n_init=10, max_iter=300)
kmeans.fit_predict(X_norm)
centroids = kmeans.cluster_centers_
print(f"质心: {centroids.shape}")

print("Ward 层次聚类...")
Z = linkage(centroids, method="ward")

print("生成树状图...")
plt.figure(figsize=(20, 10))
dendrogram(Z, truncate_mode="level", p=30, show_leaf_counts=True,
           leaf_font_size=8, color_threshold=0.7 * max(Z[:, 2]))
plt.title("Hierarchical Clustering Dendrogram — N1 22d (Stage 2)")
plt.xlabel("Cluster Index")
plt.ylabel("Ward Distance")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.close()

print(f"树状图已保存: {OUTPUT_PNG}")