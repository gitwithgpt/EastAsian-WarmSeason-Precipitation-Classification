"""
Standalone bootstrap stability script (fast, lightweight).
Loads pre-computed Z_matrix and runs a small bootstrap for illustration.
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style helpers (inline)
# ---------------------------------------------------------------------------
SERIES_WIDTH_INCHES = {"single": 3.54, "onehalf": 5.20, "double": 7.48, "full": 7.48}
EXPORT_SPECS = {
    "halftone": {"dpi": 300, "single_min_px": 1063, "full_min_px": 2244},
    "combo": {"dpi": 500, "single_min_px": 1772, "full_min_px": 3740},
    "lineart": {"dpi": 1000, "single_min_px": 3543, "full_min_px": 7480},
}
K_SELECTED = 4
K_OVER = 150
RANDOM_STATE = 42

def apply_publication_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.22,
        "axes.unicode_minus": False,
    })
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

def get_figure_size(width="double", aspect=0.62):
    width_in = SERIES_WIDTH_INCHES.get(width, SERIES_WIDTH_INCHES["double"])
    return width_in, width_in * aspect

def style_axis(ax):
    ax.grid(True, axis="y", alpha=0.22)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def label_panels(axes, start_letter='a', x=0.02, y=0.98):
    code = ord(start_letter)
    for ax in np.ravel(axes):
        ax.text(x, y, f"({chr(code)})", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=10)
        code += 1

def _ensure_minimum_pixels(fig, width, figure_type, requested_dpi=None):
    spec = EXPORT_SPECS.get(figure_type, EXPORT_SPECS["combo"])
    dpi = requested_dpi or spec["dpi"]
    width_px = fig.get_size_inches()[0] * dpi
    min_px = spec["full_min_px"] if width in {"double", "full"} else spec["single_min_px"]
    if width_px < min_px:
        dpi = int(np.ceil(min_px / fig.get_size_inches()[0]))
    return max(dpi, spec["dpi"])

def save_figure_bundle(fig, out_path, figure_type="combo", width="double",
                       raster_exts=(".png", ".tif"), vector_exts=(".pdf", ".svg"), dpi=None):
    export_dpi = _ensure_minimum_pixels(fig, width=width, figure_type=figure_type, requested_dpi=dpi)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Codex unified publication pipeline"}
    for ext in raster_exts:
        save_kwargs = {"dpi": export_dpi, "bbox_inches": "tight", "facecolor": "white"}
        if ext.lower() in {".tif", ".tiff"}:
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(str(out_path.with_suffix(ext)), **save_kwargs)
    for ext in vector_exts:
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext.lower() in {".pdf", ".svg"}:
            save_kwargs["metadata"] = metadata
        fig.savefig(str(out_path.with_suffix(ext)), **save_kwargs)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
NPZ_PATH = Path("E:/julei/run/run/hierarchical_clustering/N1_decayed_core_22d_clean.npz")
OUT_DIR = Path("E:/julei/run/run/hierarchical_clustering/publication_figures")

def load_and_normalize(npz_path, max_samples=100000):
    data = np.load(str(npz_path), allow_pickle=True)
    X = data["X"].astype(np.float64)
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
    print(f"  Loaded: {X.shape[0]:,} samples x {X.shape[1]} dims")
    X = np.where(np.isinf(X), np.nan, X)
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = np.isnan(col)
        if np.any(mask):
            X[mask, i] = np.nanmedian(col) if not np.all(mask) else 0.0
    return MinMaxScaler().fit_transform(X)

def build_z_matrix(X_norm, k_over=K_OVER):
    print(f"  Stage 1: K-means over-clustering k={k_over} ...")
    kmeans = KMeans(n_clusters=k_over, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    stage1_labels = kmeans.fit_predict(X_norm)
    centroids = kmeans.cluster_centers_
    print(f"  Centroids shape: {centroids.shape}")
    print("  Stage 2: Ward hierarchical linkage ...")
    Z = linkage(centroids, method="ward")
    return Z, stage1_labels

# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------
def calculate_subsample_stability(X_norm, sample_labels, n_bootstrap=3, sample_frac=0.8):
    n_samples = X_norm.shape[0]
    sample_size = int(sample_frac * n_samples)
    aris = []
    ari_vs_k = {k: [] for k in range(3, 9)}
    print(f"  Bootstrap stability: {n_bootstrap} iterations, {sample_frac*100:.0f}% subsample, n={n_samples}")
    for i in range(n_bootstrap):
        rng = np.random.RandomState(RANDOM_STATE + i * 100)
        idx = rng.choice(n_samples, sample_size, replace=False)
        X_sub = X_norm[idx]
        kmeans_sub = KMeans(n_clusters=K_OVER, random_state=RANDOM_STATE + i, n_init=10, max_iter=300)
        sub_stage1 = kmeans_sub.fit_predict(X_sub)
        sub_centroids = kmeans_sub.cluster_centers_
        Z_sub = linkage(sub_centroids, method="ward")
        for k in range(3, 9):
            sub_labels_centroid = fcluster(Z_sub, k, criterion="maxclust") - 1
            sub_labels = sub_labels_centroid[sub_stage1]
            ari = adjusted_rand_score(sample_labels[idx], sub_labels)
            ari_vs_k[k].append(ari)
        sub_labels_centroid = fcluster(Z_sub, K_SELECTED, criterion="maxclust") - 1
        sub_labels = sub_labels_centroid[sub_stage1]
        ari = adjusted_rand_score(sample_labels[idx], sub_labels)
        aris.append(ari)
        print(f"    Iter {i+1}: ARI(K={K_SELECTED}) = {ari:.4f}")
    return aris, ari_vs_k

def plot_stability_publication(aris, ari_vs_k, out_dir):
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=get_figure_size(width="double", aspect=0.35))
    ax1, ax2 = axes
    ks = sorted(ari_vs_k.keys())
    data_for_box = [ari_vs_k[k] for k in ks]
    bp = ax1.boxplot(data_for_box, tick_labels=[str(k) for k in ks], patch_artist=True,
                     widths=0.5, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="#C0392B", markeredgecolor="#C0392B", markersize=6))
    box_colors = ["#7AADD6", "#E8C58A", "#7FC97F", "#D68A8A", "#B8A9C9", "#A9D6E5"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#2C3E50")
        patch.set_linewidth(0.8)
    for whisker in bp["whiskers"]:
        whisker.set(color="#2C3E50", linewidth=0.8)
    for cap in bp["caps"]:
        cap.set(color="#2C3E50", linewidth=0.8)
    for median in bp["medians"]:
        median.set(color="#2C3E50", linewidth=1.5)
    for flier in bp["fliers"]:
        flier.set(marker="o", color="#7F8C8D", alpha=0.4, markersize=3)
    ax1.axvline(x=ks.index(K_SELECTED) + 1, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=0)
    ax1.set_xlabel("Number of clusters (K)", fontsize=9)
    ax1.set_ylabel("Adjusted Rand Index (ARI)", fontsize=9)
    ax1.set_title("Bootstrap stability across K values", fontsize=9)
    ax1.set_ylim(0.5, 1.02)
    style_axis(ax1)
    ax1.yaxis.grid(True, alpha=0.22)
    ax1.xaxis.grid(False)
    ari_k4 = ari_vs_k[K_SELECTED]
    mean_ari = np.mean(ari_k4)
    std_ari = np.std(ari_k4)
    ax2.hist(ari_k4, bins=8, color="#7AADD6", edgecolor="#2C3E50", alpha=0.7, linewidth=0.8)
    ax2.axvline(x=mean_ari, color="#C0392B", linestyle="-", linewidth=1.8, label=f"Mean ARI = {mean_ari:.3f}")
    ax2.axvline(x=mean_ari - std_ari, color="#7F8C8D", linestyle="--", linewidth=1.2)
    ax2.axvline(x=mean_ari + std_ari, color="#7F8C8D", linestyle="--", linewidth=1.2, label=f"±1 SD = {std_ari:.3f}")
    ax2.set_xlabel(f"Adjusted Rand Index (K = {K_SELECTED})", fontsize=9)
    ax2.set_ylabel("Frequency", fontsize=9)
    ax2.set_title(f"Stability distribution for K = {K_SELECTED}", fontsize=9)
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    style_axis(ax2)
    ax2.yaxis.grid(True, alpha=0.22)
    ax2.xaxis.grid(False)
    label_panels(axes, start_letter="a", x=0.04, y=0.96)
    plt.tight_layout(w_pad=1.2)
    out_path = out_dir / "fig_stability_publication"
    save_figure_bundle(fig, out_path, figure_type="combo", width="double",
                       raster_exts=(".png", ".tif"), vector_exts=(".pdf", ".svg"))
    print(f"  Saved stability bundle: {out_path}")
    return mean_ari, std_ari

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    print("\n[1] Loading data (n=200,000) and building Z-matrix ...")
    X_norm = load_and_normalize(NPZ_PATH, max_samples=200000)
    Z, stage1_labels = build_z_matrix(X_norm, k_over=K_OVER)

    ref_centroid_labels = fcluster(Z, K_SELECTED, criterion="maxclust") - 1
    sample_labels = ref_centroid_labels[stage1_labels]
    print(f"  Reference sample labels: {len(sample_labels)} samples, K={K_SELECTED}")

    print("\n[2] Running bootstrap stability analysis (100 iterations) ...")
    aris, ari_vs_k = calculate_subsample_stability(X_norm, sample_labels, n_bootstrap=100, sample_frac=0.8)
    mean_ari, std_ari = plot_stability_publication(aris, ari_vs_k, OUT_DIR)

    stability_df = pd.DataFrame({
        "K": list(ari_vs_k.keys()),
        "ARI_mean": [np.mean(v) for v in ari_vs_k.values()],
        "ARI_std": [np.std(v) for v in ari_vs_k.values()],
        "ARI_min": [np.min(v) for v in ari_vs_k.values()],
        "ARI_max": [np.max(v) for v in ari_vs_k.values()],
    })
    stability_df.to_csv(str(OUT_DIR / "stability_bootstrap_results.csv"), index=False)
    print(f"  Stability results saved: {OUT_DIR / 'stability_bootstrap_results.csv'}")
    print(f"\n  K={K_SELECTED} mean ARI = {mean_ari:.4f} ± {std_ari:.4f}")

    print("\n" + "=" * 80)
    print("Bootstrap stability figure generated.")
    print("=" * 80)

if __name__ == "__main__":
    main()
