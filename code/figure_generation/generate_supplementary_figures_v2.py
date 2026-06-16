import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, fcluster
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "figures" / "supplementary"
DATA_DIR = REPO_ROOT / "data"
Z_MATRIX_PATH = DATA_DIR / "Z_matrix.npy"
STABILITY_CSV_PATH = DATA_DIR / "stability_bootstrap_results.csv"
STORY_NPZ_PATH = DATA_DIR / "cluster4_story_dataset_plus_dsd25.npz"  # ~905 MB; not included in repo

CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
CLUSTER_NAMES = ["C0", "C1", "C2", "C3"]

K_SELECTED = 4


def setup_fonts():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    })


def save_all_formats(fig, out_path, dpi=300):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path.with_suffix(".png")), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path.with_suffix(".svg")), bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path.with_suffix(".tif")), dpi=dpi, format="tif", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Figure S1: K-selection metrics
# ---------------------------------------------------------------------------
def plot_figure_s1():
    setup_fonts()
    K = np.array([3, 4, 5, 6, 7, 8])
    db = np.array([1.1788, 1.1462, 1.2568, 1.1875, 1.2236, 1.2387])
    sil = np.array([0.3337, 0.3338, 0.3340, 0.3358, 0.3024, 0.2960])
    ch = np.array([103394.8, 88414.4, 81645.0, 71087.8, 71211.8, 63972.1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Davies-Bouldin Index
    ax = axes[0]
    ax.plot(K, db, "o-", color="#C0392B", linewidth=1.8, markersize=7,
            markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#C0392B", zorder=3)
    ax.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_xlabel("Number of clusters (K)", fontsize=10)
    ax.set_ylabel("Davies-Bouldin index", fontsize=10)
    ax.set_xticks(K)
    ax.set_ylim(1.14, 1.29)
    ax.text(0.02, 0.98, "(a) Davies-Bouldin Index", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=12)
    style_axis(ax)

    # (b) Silhouette Score
    ax = axes[1]
    ax.plot(K, sil, "s-", color="#27AE60", linewidth=1.8, markersize=7,
            markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#27AE60", zorder=3)
    ax.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_xlabel("Number of clusters (K)", fontsize=10)
    ax.set_ylabel("Silhouette score", fontsize=10)
    ax.set_xticks(K)
    ax.set_ylim(0.292, 0.340)
    ax.text(0.02, 0.98, "(b) Silhouette Score", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=12)
    style_axis(ax)

    # (c) Calinski-Harabasz Index
    ax = axes[2]
    ax.plot(K, ch, "^-", color="#2980B9", linewidth=1.8, markersize=7,
            markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#2980B9", zorder=3)
    ax.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_xlabel("Number of clusters (K)", fontsize=10)
    ax.set_ylabel("Calinski-Harabasz index", fontsize=10)
    ax.set_xticks(K)
    ax.set_ylim(60000, 110000)
    ax.text(0.02, 0.98, "(c) Calinski-Harabasz Index", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=12)
    style_axis(ax)

    plt.tight_layout(w_pad=1.2)
    save_all_formats(fig, OUT_DIR / "figure_s1_k_selection", dpi=300)
    print("Saved figure_s1_k_selection")


# ---------------------------------------------------------------------------
# Figure S2: Dendrogram
# ---------------------------------------------------------------------------
def plot_figure_s2():
    setup_fonts()
    Z = np.load(str(Z_MATRIX_PATH))
    n_leaves = Z.shape[0] + 1  # 150

    leaf_clusters = fcluster(Z, t=K_SELECTED, criterion="maxclust") - 1
    if len(leaf_clusters) != n_leaves:
        leaf_clusters = fcluster(Z, t=4.0, criterion="distance") - 1

    # Compute node colors for link_color_func
    node_color = np.empty(2 * n_leaves - 1, dtype=object)
    node_color[:] = "grey"
    for i in range(n_leaves):
        node_color[i] = CLUSTER_COLORS[leaf_clusters[i]]
    for i in range(n_leaves, 2 * n_leaves - 1):
        idx = i - n_leaves
        left = int(Z[idx, 0])
        right = int(Z[idx, 1])
        if node_color[left] == node_color[right] and node_color[left] != "grey":
            node_color[i] = node_color[left]
        else:
            node_color[i] = "grey"

    fig, ax = plt.subplots(figsize=(16, 8))
    R = dendrogram(
        Z, truncate_mode="level", p=30, show_leaf_counts=False,
        no_labels=True, leaf_font_size=6, color_threshold=4.0,
        above_threshold_color="grey",
        link_color_func=lambda k: node_color[k],
        ax=ax,
    )

    ax.axhline(y=4.0, color="#C0392B", linestyle="--", linewidth=1.5, zorder=5)
    ax.text(0.98, 4.05, "K = 4", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=10, color="#C0392B", fontweight="bold")

    ax.set_xlabel("Cluster index", fontsize=10)
    ax.set_ylabel("Ward distance", fontsize=10)
    style_axis(ax)

    # Legend in upper-right corner
    legend_handles = [Patch(facecolor=CLUSTER_COLORS[i], edgecolor="none", label=f"Cluster {CLUSTER_NAMES[i]}")
                      for i in range(4)]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9,
              facecolor="white", edgecolor="grey", fancybox=False)

    plt.tight_layout()
    save_all_formats(fig, OUT_DIR / "figure_s2_dendrogram", dpi=300)
    print("Saved figure_s2_dendrogram")


# ---------------------------------------------------------------------------
# Figure S4: Bootstrap stability
# ---------------------------------------------------------------------------
def plot_figure_s4():
    setup_fonts()
    df = pd.read_csv(str(STABILITY_CSV_PATH))
    K = df["K"].values
    ari_mean = df["ARI_mean"].values
    ari_std = df["ARI_std"].values
    ari_min = df["ARI_min"].values
    ari_max = df["ARI_max"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a)
    ax = axes[0]
    for i, k in enumerate(K):
        # thin range bar
        ax.plot([k, k], [ari_min[i], ari_max[i]], color="#2C3E50", linewidth=1.5,
                solid_capstyle="butt", zorder=2)
        # mean ± std thicker bar
        ax.plot([k, k], [ari_mean[i] - ari_std[i], ari_mean[i] + ari_std[i]],
                color="#C0392B", linewidth=4, solid_capstyle="butt", zorder=3)
        # mean marker
        ax.plot(k, ari_mean[i], "D", color="#C0392B", markersize=7, zorder=4)

    ax.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xlabel("Number of clusters (K)", fontsize=10)
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=10)
    ax.set_xticks(K)
    ax.set_ylim(0.5, 1.02)
    ax.text(0.02, 0.98, "(a) Bootstrap stability across K values", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=12)
    style_axis(ax)

    # Panel (b)
    ax = axes[1]
    k4_idx = np.where(K == K_SELECTED)[0][0]
    mean_val = ari_mean[k4_idx]
    std_val = ari_std[k4_idx]
    min_val = ari_min[k4_idx]
    max_val = ari_max[k4_idx]

    ax.plot([1, 1], [min_val, max_val], color="#2C3E50", linewidth=1.5,
            solid_capstyle="butt", zorder=2)
    ax.plot([1, 1], [mean_val - std_val, mean_val + std_val],
            color="#C0392B", linewidth=4, solid_capstyle="butt", zorder=3)
    ax.plot(1, mean_val, "D", color="#C0392B", markersize=7, zorder=4)

    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks([1])
    ax.set_xticklabels([f"K = {K_SELECTED}"])
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=10)
    ax.text(0.02, 0.98, "(b) Stability distribution for K = 4", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=12)
    style_axis(ax)

    # Text box at top-right
    ax.text(0.98, 0.98, f"Mean ARI = {mean_val:.3f} ± {std_val:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.9))

    plt.tight_layout(w_pad=1.2)
    save_all_formats(fig, OUT_DIR / "figure_s11_bootstrap_stability", dpi=300)
    print("Saved figure_s11_bootstrap_stability")


# ---------------------------------------------------------------------------
# Figure S8 / S9: Quasi-CFAD helpers
# ---------------------------------------------------------------------------
def compute_median_iqr(x_sub, y_sub, y_edges, min_count=40):
    mids = 0.5 * (y_edges[:-1] + y_edges[1:])
    med = np.full_like(mids, np.nan, dtype=float)
    q25 = np.full_like(mids, np.nan, dtype=float)
    q75 = np.full_like(mids, np.nan, dtype=float)
    for i in range(len(y_edges) - 1):
        mask = (y_sub >= y_edges[i]) & (y_sub < y_edges[i + 1])
        if np.sum(mask) >= min_count:
            med[i] = np.nanmedian(x_sub[mask])
            q25[i] = np.nanpercentile(x_sub[mask], 25)
            q75[i] = np.nanpercentile(x_sub[mask], 75)
    return med, q25, q75, mids


def plot_quasi_cfad(x_top, x_bottom, y_var, ylim, xlim_top, xlim_bottom, ylabel, out_stem):
    setup_fonts()
    data = np.load(str(STORY_NPZ_PATH), allow_pickle=True)
    labels = data["labels"]
    cids = [0, 1, 2, 3]
    n_rows = 2
    n_cols = 4

    y_edges = np.linspace(ylim[0], ylim[1], 41)
    x_edges_top = np.linspace(xlim_top[0], xlim_top[1], 41)
    x_edges_bottom = np.linspace(xlim_bottom[0], xlim_bottom[1], 41)

    # Precompute histograms and global vmax
    vmax = 0.0
    cached = []
    for row_idx, (x_var, x_edges) in enumerate([(x_top, x_edges_top), (x_bottom, x_edges_bottom)]):
        row_cached = []
        for cid in cids:
            mask = (labels == cid) & np.isfinite(x_var) & np.isfinite(y_var)
            x_sub = x_var[mask]
            y_sub = y_var[mask]
            H, _, _ = np.histogram2d(y_sub, x_sub, bins=[y_edges, x_edges])
            H = H / H.sum()
            vmax = max(vmax, np.nanmax(H))
            med, q25, q75, mids = compute_median_iqr(x_sub, y_sub, y_edges)
            row_cached.append((H, x_edges, y_edges, med, q25, q75, mids))
        cached.append(row_cached)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=False, sharey=True)

    meshes = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            H, x_edges, y_edges, med, q25, q75, mids = cached[row_idx][col_idx]
            mesh = ax.pcolormesh(x_edges, y_edges, H, cmap="Spectral_r", shading="auto", vmin=0, vmax=vmax)
            meshes.append(mesh)

            valid = np.isfinite(med)
            if np.any(valid):
                ax.plot(med[valid], mids[valid], color="black", linewidth=1.7)
                ax.plot(q25[valid], mids[valid], color="black", linewidth=0.85, linestyle="--", alpha=0.85)
                ax.plot(q75[valid], mids[valid], color="black", linewidth=0.85, linestyle="--", alpha=0.85)

            letter = chr(ord("a") + row_idx * n_cols + col_idx)
            cid = cids[col_idx]
            ax.text(0.02, 0.98, f"({letter}) {CLUSTER_NAMES[cid]}", transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold", fontsize=12, color="black")
            style_axis(ax)
            ax.set_aspect("auto")

            if row_idx == 0:
                ax.set_xlabel("Column-max Dm (mm)", fontsize=10)
                ax.set_xlim(xlim_top)
            else:
                ax.set_xlabel("Column-max log10(Nw)", fontsize=10)
                ax.set_xlim(xlim_bottom)

            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.set_ylim(ylim)

    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(meshes[-1], cax=cax)
    cbar.set_label("Probability density", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_all_formats(fig, OUT_DIR / out_stem, dpi=300)
    print(f"Saved {out_stem}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating Figure S1 ...")
    plot_figure_s1()

    print("Generating Figure S2 ...")
    plot_figure_s2()

    print("Generating Figure S4 ...")
    plot_figure_s4()

    print("Loading story dataset for S8/S9 ...")
    data = np.load(str(STORY_NPZ_PATH), allow_pickle=True)
    X = data["X"]
    labels = data["labels"]
    # Extract needed columns
    x_dm = X[:, 20]
    x_nw = X[:, 21]
    y_bb = X[:, 18]          # already in km (deg0l proxy for bright-band height)
    y_st = X[:, 1] / 1000.0  # storm-top height in km
    del X, data

    print("Generating Figure S8 ...")
    plot_quasi_cfad(
        x_top=x_dm,
        x_bottom=x_nw,
        y_var=y_bb,
        ylim=(0, 5.5),
        xlim_top=(0.5, 3.0),
        xlim_bottom=(26, 48),
        ylabel="Bright-band height (km)",
        out_stem="figure_s8_quasi_cfad_dsd_brightband",
    )

    print("Generating Figure S9 ...")
    plot_quasi_cfad(
        x_top=x_dm,
        x_bottom=x_nw,
        y_var=y_st,
        ylim=(2, 14),
        xlim_top=(0.5, 3.0),
        xlim_bottom=(26, 48),
        ylabel="Storm-top height (km)",
        out_stem="figure_s9_quasi_cfad_dsd_stormtop",
    )

    print("\nAll supplementary figures generated successfully.")


if __name__ == "__main__":
    main()
