"""
Generate publication-quality supplementary figures (S1, S2, S4, S8, S9)
for Atmospheric Research journal submission.
Matches Figure 5 style: Arial/Helvetica, white background, minimal borders,
panel labels (a), (b), ... in bold at top-left, compact colorbar at right edge.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, fcluster

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "figures" / "supplementary"
DATA_DIR = REPO_ROOT / "data"
Z_MATRIX_PATH = DATA_DIR / "Z_matrix.npy"
STORY_NPZ_PATH = DATA_DIR / "cluster4_story_dataset.npz"  # ~874 MB; not included in repo
STABILITY_CSV_PATH = DATA_DIR / "stability_bootstrap_results.csv"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
CLUSTER_COLORS = ["#7AADD6", "#E8C58A", "#7FC97F", "#D68A8A"]
CLUSTER_NAMES = ["C0", "C1", "C2", "C3"]
CFAD_CMAP = "Spectral_r"

SERIES_WIDTH_INCHES = {"single": 3.54, "onehalf": 5.20, "double": 7.48, "full": 7.48}

K_SELECTED = 4
K_OVER = 150
RANDOM_STATE = 42

LETTERS = list("abcdefghijklmnopqrstuvwxyz")


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
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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
        ax.text(x, y, f"({chr(code)})", transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", fontsize=10)
        code += 1


def save_figure_bundle(fig, out_path, dpi=600, raster_exts=(".png",), vector_exts=(".pdf",)):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in raster_exts:
        fig.savefig(str(out_path.with_suffix(ext)), dpi=dpi, bbox_inches="tight", facecolor="white")
    for ext in vector_exts:
        fig.savefig(str(out_path.with_suffix(ext)), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure S1: K-selection metrics
# ---------------------------------------------------------------------------
def plot_figure_s1():
    apply_publication_style()
    K = np.array([3, 4, 5, 6, 7, 8])
    db = np.array([1.1788, 1.1462, 1.2568, 1.1875, 1.2236, 1.2387])
    sil = np.array([0.3337, 0.3338, 0.3340, 0.3358, 0.3024, 0.2960])
    ch = np.array([103394.8, 88414.4, 81645.0, 71087.8, 71211.8, 63972.1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    # DB Index
    ax1.plot(K, db, "o-", color="#C0392B", linewidth=1.8, markersize=7,
             markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#C0392B", zorder=3)
    ax1.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=2)
    ax1.set_xlabel("Number of clusters (K)", fontsize=10)
    ax1.set_ylabel("Davies-Bouldin Index", fontsize=10)
    ax1.set_title("Davies-Bouldin Index (lower is better)", fontsize=10)
    ax1.set_xticks(K)
    style_axis(ax1)
    ax1.yaxis.grid(True, alpha=0.22)
    ax1.xaxis.grid(False)

    # Silhouette
    ax2.plot(K, sil, "s-", color="#27AE60", linewidth=1.8, markersize=7,
             markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#27AE60", zorder=3)
    ax2.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=2)
    ax2.set_xlabel("Number of clusters (K)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.set_title("Silhouette Score (higher is better)", fontsize=10)
    ax2.set_xticks(K)
    style_axis(ax2)
    ax2.yaxis.grid(True, alpha=0.22)
    ax2.xaxis.grid(False)

    # Calinski-Harabasz
    ax3.plot(K, ch, "^-", color="#2980B9", linewidth=1.8, markersize=7,
             markerfacecolor="white", markeredgewidth=1.2, markeredgecolor="#2980B9", zorder=3)
    ax3.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=2)
    ax3.set_xlabel("Number of clusters (K)", fontsize=10)
    ax3.set_ylabel("Calinski-Harabasz Index", fontsize=10)
    ax3.set_title("Calinski-Harabasz Index (higher is better)", fontsize=10)
    ax3.set_xticks(K)
    style_axis(ax3)
    ax3.yaxis.grid(True, alpha=0.22)
    ax3.xaxis.grid(False)

    fig.suptitle("Cluster number selection metrics", fontsize=14, fontweight="bold", y=1.02)
    label_panels(axes, start_letter="a", x=0.04, y=0.96)
    plt.tight_layout(w_pad=1.2)
    save_figure_bundle(fig, OUT_DIR / "figure_s1_k_selection", dpi=300)
    print("Saved figure_s1_k_selection")


# ---------------------------------------------------------------------------
# Figure S2: Dendrogram
# ---------------------------------------------------------------------------
def plot_figure_s2():
    apply_publication_style()
    Z = np.load(str(Z_MATRIX_PATH))
    n_leaves = Z.shape[0] + 1  # 150

    # Compute leaf cluster assignments using fcluster (maxclust=4)
    leaf_clusters = fcluster(Z, t=K_SELECTED, criterion="maxclust") - 1
    # Ensure length is n_leaves
    if len(leaf_clusters) != n_leaves:
        # Fallback: fcluster on Z with criterion distance
        leaf_clusters = fcluster(Z, t=4.5, criterion="distance") - 1

    # Compute dominant cluster for each link (internal node) in Z
    n_clusters = K_SELECTED
    cluster_counts = np.zeros((len(Z), n_clusters), dtype=int)
    link_cluster = np.empty(len(Z), dtype=int)

    for i, (left, right, dist, count) in enumerate(Z):
        left = int(left)
        right = int(right)
        if left < n_leaves:
            cluster_counts[i, leaf_clusters[left]] += 1
        else:
            cluster_counts[i, :] += cluster_counts[left - n_leaves, :]
        if right < n_leaves:
            cluster_counts[i, leaf_clusters[right]] += 1
        else:
            cluster_counts[i, :] += cluster_counts[right - n_leaves, :]
        link_cluster[i] = int(np.argmax(cluster_counts[i]))

    color_threshold = 4.5  # just above the 4-cluster merge distance
    above_color = "#7F8C8D"

    def link_color_func(link_id):
        # scipy dendrogram passes link_id as n_leaves + row_index for internal nodes
        if link_id >= n_leaves:
            row_idx = link_id - n_leaves
            if 0 <= row_idx < len(link_cluster):
                return CLUSTER_COLORS[link_cluster[row_idx]]
        else:
            # leaf node (if ever passed)
            if 0 <= link_id < len(leaf_clusters):
                return CLUSTER_COLORS[leaf_clusters[link_id]]
        return above_color

    fig, ax = plt.subplots(figsize=(16, 7))
    R = dendrogram(
        Z, truncate_mode="level", p=30, show_leaf_counts=False,
        no_labels=True, leaf_font_size=6, color_threshold=color_threshold,
        above_threshold_color=above_color,
        link_color_func=link_color_func,
        ax=ax,
    )
    ax.axhline(y=4.0, color="#C0392B", linestyle="--", linewidth=1.5, zorder=5)
    ax.text(0.98, 4.05, "K = 4", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=10, color="#C0392B", fontweight="bold")

    ax.set_title("Hierarchical clustering dendrogram (Ward linkage)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster index (leaf count shown in parentheses)", fontsize=10)
    ax.set_ylabel("Ward distance", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(False)

    # Add legend for cluster colors at upper right
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=CLUSTER_COLORS[i], edgecolor="none", label=f"Cluster {CLUSTER_NAMES[i]}")
                      for i in range(n_clusters)]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8,
              facecolor="white", edgecolor="#7F8C8D", fancybox=False)

    plt.tight_layout()
    save_figure_bundle(fig, OUT_DIR / "figure_s2_dendrogram", dpi=300)
    print("Saved figure_s2_dendrogram")


# ---------------------------------------------------------------------------
# Figure S4: Bootstrap stability
# ---------------------------------------------------------------------------
def plot_figure_s4():
    apply_publication_style()
    df = pd.read_csv(str(STABILITY_CSV_PATH))
    K = df["K"].values
    ari_mean = df["ARI_mean"].values
    ari_std = df["ARI_std"].values
    ari_min = df["ARI_min"].values
    ari_max = df["ARI_max"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax1, ax2 = axes

    # Panel (a): ARI stability across K
    for i, k in enumerate(K):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        # Min-max range line
        ax1.plot([k, k], [ari_min[i], ari_max[i]], color=color, linewidth=3, solid_capstyle="butt", zorder=2)
        # Mean ± std box
        ax1.plot([k, k], [ari_mean[i] - ari_std[i], ari_mean[i] + ari_std[i]],
                 color="#2C3E50", linewidth=2, zorder=3)
        # Mean diamond
        ax1.plot(k, ari_mean[i], "D", color="#C0392B", markersize=7, zorder=4)

    ax1.axvline(x=K_SELECTED, color="#7F8C8D", linestyle="--", linewidth=1.2, zorder=1)
    ax1.set_xlabel("Number of clusters (K)", fontsize=10)
    ax1.set_ylabel("Adjusted Rand Index (ARI)", fontsize=10)
    ax1.set_title("Bootstrap stability across K values", fontsize=10)
    ax1.set_ylim(0.5, 1.02)
    ax1.set_xticks(K)
    style_axis(ax1)
    ax1.yaxis.grid(True, alpha=0.22)
    ax1.xaxis.grid(False)

    # Panel (b): K=4 stability distribution (summary statistics)
    k4_idx = np.where(K == K_SELECTED)[0][0]
    mean_val = ari_mean[k4_idx]
    std_val = ari_std[k4_idx]
    min_val = ari_min[k4_idx]
    max_val = ari_max[k4_idx]

    # Draw a vertical thick bar from min to max
    ax2.plot([1, 1], [min_val, max_val], color="#7AADD6", linewidth=8, solid_capstyle="butt", zorder=2)
    # Mean ± std box
    ax2.plot([1, 1], [mean_val - std_val, mean_val + std_val],
             color="#2C3E50", linewidth=3, zorder=3)
    # Mean diamond
    ax2.plot(1, mean_val, "D", color="#C0392B", markersize=8, zorder=4)
    ax2.set_xlim(0.5, 1.5)
    ax2.set_ylim(0.5, 1.02)
    ax2.set_xticks([1])
    ax2.set_xticklabels([f"K = {K_SELECTED}"])
    ax2.set_ylabel("Adjusted Rand Index (ARI)", fontsize=10)
    ax2.set_title(f"Stability distribution for K = {K_SELECTED}", fontsize=10)
    style_axis(ax2)
    ax2.yaxis.grid(True, alpha=0.22)
    ax2.xaxis.grid(False)
    ax2.spines["bottom"].set_visible(True)
    ax2.spines["left"].set_visible(True)

    # Annotation text with white background box
    ax2.text(0.98, 0.98, f"Mean ARI = {mean_val:.3f} ± {std_val:.3f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2C3E50", alpha=0.9))

    label_panels(axes, start_letter="a", x=0.04, y=0.96)
    plt.tight_layout(w_pad=1.2)
    save_figure_bundle(fig, OUT_DIR / "figure_s4_bootstrap_stability", dpi=300)
    print("Saved figure_s4_bootstrap_stability")


# ---------------------------------------------------------------------------
# Figure S8 / S9: Quasi-CFAD
# ---------------------------------------------------------------------------
def robust_edges(values, n_bins=36, padding=0.03):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return np.linspace(0.0, 1.0, n_bins + 1)
    q = np.nanpercentile(values, [1, 99])
    q1, q99 = float(q[0]), float(q[1])
    span = max(q99 - q1, 1e-6)
    return np.linspace(q1 - padding * span, q99 + padding * span, n_bins + 1)


def normalized_hist2d(x, y, x_edges, y_edges):
    hist, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    row_sum = hist.sum(axis=1, keepdims=True)
    return np.divide(hist, row_sum, out=np.zeros_like(hist), where=row_sum > 0)


def add_profile(ax, x, y, y_edges):
    mids = 0.5 * (y_edges[:-1] + y_edges[1:])
    med = np.full_like(mids, np.nan, dtype=float)
    q25 = np.full_like(mids, np.nan, dtype=float)
    q75 = np.full_like(mids, np.nan, dtype=float)
    for i in range(len(y_edges) - 1):
        mask = (y >= y_edges[i]) & (y < y_edges[i + 1]) & np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) >= 40:
            med[i] = np.nanmedian(x[mask])
            q25[i] = np.nanpercentile(x[mask], 25)
            q75[i] = np.nanpercentile(x[mask], 75)
    valid = np.isfinite(med)
    if np.any(valid):
        ax.plot(med[valid], mids[valid], color="black", linewidth=1.7)
        ax.plot(q25[valid], mids[valid], color="black", linewidth=0.85, linestyle="--", alpha=0.85)
        ax.plot(q75[valid], mids[valid], color="black", linewidth=0.85, linestyle="--", alpha=0.85)


def plot_quasi_cfad(data, specs, out_stem, title):
    """
    specs: list of (x_data, y_data, xlabel, ylabel) per row.
    out_stem: e.g. 'figure_s8_quasi_cfad_dsd_brightband'
    title: figure suptitle
    """
    apply_publication_style()
    X = data["X"]
    labels = data["labels"]
    cids = sorted(np.unique(labels))
    n_rows = len(specs)
    n_cols = len(cids)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    
    # Compute shared vmax across all panels
    vmax = 0.0
    cached = []
    for row, (x, y, xlabel, ylabel) in enumerate(specs):
        x_edges = robust_edges(x)
        y_edges = robust_edges(y)
        row_cached = []
        for cid in cids:
            mask = (labels == cid) & np.isfinite(x) & np.isfinite(y)
            hist = normalized_hist2d(x[mask], y[mask], x_edges, y_edges)
            row_cached.append((cid, mask, hist, x_edges, y_edges))
            vmax = max(vmax, float(np.nanpercentile(hist, 99.5)))
        cached.append(row_cached)
    vmax = max(vmax, 0.05)

    # Plot
    meshes = []
    for row_idx, row_cached in enumerate(cached):
        for col_idx, (cid, mask, hist, x_edges, y_edges) in enumerate(row_cached):
            ax = axes[row_idx, col_idx]
            mesh = ax.pcolormesh(x_edges, y_edges, hist, cmap=CFAD_CMAP,
                                 shading="auto", vmin=0, vmax=vmax)
            add_profile(ax, x[mask], y[mask], y_edges)
            # Cluster name inside panel, top center, below the panel label area
            ax.text(0.5, 0.95, f"Cluster {CLUSTER_NAMES[cid]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, fontweight="bold", color="black")
            # Panel label top-left
            letter = LETTERS[row_idx * n_cols + col_idx]
            ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10, fontweight="bold", color="black")
            if col_idx == 0:
                ax.set_ylabel(specs[row_idx][3], fontsize=10)
            if row_idx == n_rows - 1:
                ax.set_xlabel(specs[row_idx][2], fontsize=10)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            meshes.append(mesh)

    # Single colorbar at the right edge of the figure
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(meshes[-1], cax=cax, shrink=0.8, aspect=20)
    cbar.set_label("Conditional normalized frequency", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    save_figure_bundle(fig, OUT_DIR / out_stem, dpi=300)
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
    data = dict(np.load(str(STORY_NPZ_PATH), allow_pickle=True, mmap_mode="r"))
    X = data["X"]
    labels = data["labels"]

    # S8: DSD vs Bright-band height
    print("Generating Figure S8 ...")
    x_dm = X[:, 20]
    y_bb = X[:, 18]  # already in km
    x_nw = X[:, 21]
    plot_quasi_cfad(
        data,
        specs=[
            (x_dm, y_bb, "Column-max Dm (mm)", "Bright-band height (km)"),
            (x_nw, y_bb, "Column-max log10(Nw)", "Bright-band height (km)"),
        ],
        out_stem="figure_s8_quasi_cfad_dsd_brightband",
        title="Supplementary quasi-CFAD: DSD extreme proxies versus bright-band height",
    )

    # S9: DSD vs Storm-top height
    print("Generating Figure S9 ...")
    y_st = X[:, 1] / 1000.0  # convert m to km
    plot_quasi_cfad(
        data,
        specs=[
            (x_dm, y_st, "Column-max Dm (mm)", "Storm-top height (km)"),
            (x_nw, y_st, "Column-max log10(Nw)", "Storm-top height (km)"),
        ],
        out_stem="figure_s9_quasi_cfad_dsd_stormtop",
        title="Supplementary quasi-CFAD: DSD extreme proxies versus storm-top height",
    )

    print("\nAll supplementary figures generated successfully.")


if __name__ == "__main__":
    main()
