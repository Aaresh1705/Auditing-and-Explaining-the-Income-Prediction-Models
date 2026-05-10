"""
k-NN Locality Test — Validate UMAP's manifold assumption
=========================================================
Tests whether the latent spaces of XGBoost (leaf embeddings) and NN
(last hidden layer) exhibit locally uniform Euclidean structure.

UMAP assumes the data lies on a manifold that is locally Euclidean.
If k-NN distances vary wildly across the space, the manifold assumption
is violated and UMAP projections may be misleading.

Tests performed:
  1. k-NN distance distribution — histogram of distances to the k-th
     nearest neighbor. A tight distribution suggests uniform local density.
  2. Distance ratio (k-th / 1st neighbor) — if locally Euclidean, this
     ratio should be roughly consistent across points.
  3. Spatial uniformity — splits the space into regions (via PCA quadrants)
     and compares k-NN distance distributions across regions.

Usage:
    python assignment2/knn_locality_test.py
    python assignment2/knn_locality_test.py --no-fnlwgt
"""

from pathlib import Path
import os
import sys
import argparse
import warnings

N_THREADS = str(max(1, os.cpu_count() // 2))
os.environ.setdefault("OMP_NUM_THREADS", N_THREADS)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", N_THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(int(N_THREADS))
tf.config.threading.set_inter_op_parallelism_threads(2)

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from concepts import (
    prepare_data, load_models, OUTPUT_DIR,
    C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GRAY, C_DARK, BG,
)


def extract_nn_last_hidden(nn_model, X_scaled):
    """Extract activations from the last hidden Dense layer."""
    _ = nn_model.predict(X_scaled[:1], verbose=0)
    inp = tf.keras.Input(shape=(X_scaled.shape[1],))
    x = inp
    last_hidden = None
    for layer in nn_model.layers:
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Dense) and layer != nn_model.layers[-1]:
            last_hidden = x
    extractor = tf.keras.Model(inputs=inp, outputs=last_hidden)
    return extractor.predict(X_scaled, verbose=0)


def compute_knn_stats(embeddings, k=15, sample_size=3000):
    """Compute k-NN distances and ratios for a sample of the embeddings."""
    rng = np.random.default_rng(42)
    n = len(embeddings)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample = StandardScaler().fit_transform(embeddings[idx].astype(float))

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(sample)
    distances, _ = nn.kneighbors(sample)

    # Skip self-distance (column 0)
    d1 = distances[:, 1]       # 1st neighbor
    dk = distances[:, k]       # k-th neighbor
    d_mean = distances[:, 1:].mean(axis=1)  # mean over all k neighbors
    ratio = dk / (d1 + 1e-10)  # k-th / 1st distance ratio

    return {
        "sample": sample,
        "d1": d1,
        "dk": dk,
        "d_mean": d_mean,
        "ratio": ratio,
        "all_distances": distances[:, 1:],
    }


def spatial_uniformity_test(sample, dk, n_quadrants=4):
    """Split space into PCA quadrants, compare k-NN distance distributions."""
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(sample)
    median_x = np.median(pca_2d[:, 0])
    median_y = np.median(pca_2d[:, 1])

    quadrant_labels = np.zeros(len(sample), dtype=int)
    quadrant_labels[(pca_2d[:, 0] >= median_x) & (pca_2d[:, 1] >= median_y)] = 0
    quadrant_labels[(pca_2d[:, 0] < median_x) & (pca_2d[:, 1] >= median_y)] = 1
    quadrant_labels[(pca_2d[:, 0] < median_x) & (pca_2d[:, 1] < median_y)] = 2
    quadrant_labels[(pca_2d[:, 0] >= median_x) & (pca_2d[:, 1] < median_y)] = 3

    quadrant_dists = [dk[quadrant_labels == q] for q in range(n_quadrants)]

    # Kruskal-Wallis test: are the distributions significantly different?
    stat, p_value = stats.kruskal(*quadrant_dists)

    return {
        "pca_2d": pca_2d,
        "quadrant_labels": quadrant_labels,
        "quadrant_dists": quadrant_dists,
        "kruskal_stat": stat,
        "kruskal_p": p_value,
    }


def plot_locality_report(knn_stats, spatial, model_name, k, filename):
    """4-panel diagnostic plot for one model's latent space."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor=BG)
    fig.suptitle(
        f"k-NN Locality Test: {model_name} (k={k})\n"
        f"Validating UMAP manifold assumption",
        fontsize=14, fontweight="bold", color=C_DARK,
    )

    dk = knn_stats["dk"]
    d1 = knn_stats["d1"]
    ratio = knn_stats["ratio"]

    # --- Panel 1: k-th neighbor distance distribution (log scale) ---
    ax = axes[0, 0]
    dk_log = np.log10(dk + 1e-10)
    ax.hist(dk_log, bins=60, color=C_BLUE, edgecolor="white", alpha=0.8)
    ax.axvline(np.log10(np.median(dk)), color=C_RED, ls="--", lw=2,
               label=f"Median: {np.median(dk):.2f}")
    ax.set_xlabel(f"log₁₀ Distance to {k}-th nearest neighbor", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{k}-NN Distance Distribution", fontsize=11, fontweight="bold")
    cv = np.std(dk) / np.mean(dk)
    ax.text(0.95, 0.95, f"CV = {cv:.3f}\nStd = {np.std(dk):.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(fontsize=9)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel 2: Distance ratio (k-th / 1st) (log scale) ---
    ax = axes[0, 1]
    ratio_log = np.log10(ratio + 1e-10)
    ax.hist(ratio_log, bins=60, color=C_ORANGE, edgecolor="white", alpha=0.8)
    ax.axvline(np.log10(np.median(ratio)), color=C_RED, ls="--", lw=2,
               label=f"Median: {np.median(ratio):.2f}")
    ax.set_xlabel(f"log₁₀ Distance ratio (d_{k} / d_1)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Local Expansion Ratio", fontsize=11, fontweight="bold")
    cv_ratio = np.std(ratio) / np.mean(ratio)
    ax.text(0.95, 0.95, f"CV = {cv_ratio:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(fontsize=9)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel 3: Spatial uniformity (PCA quadrants) ---
    ax = axes[1, 0]
    quad_colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]
    quad_names = ["Q1", "Q2", "Q3", "Q4"]
    pca_2d = spatial["pca_2d"]
    for q in range(4):
        mask = spatial["quadrant_labels"] == q
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], c=[quad_colors[q]],
                   s=4, alpha=0.3, label=quad_names[q])
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("PCA Quadrants", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, markerscale=3)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)

    # --- Panel 4: k-NN distance by quadrant (box plot) ---
    ax = axes[1, 1]
    bp = ax.boxplot(spatial["quadrant_dists"], labels=quad_names, patch_artist=True,
                    widths=0.6, showfliers=False)
    for patch, color in zip(bp["boxes"], quad_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(f"Distance to {k}-th neighbor", fontsize=10)
    ax.set_title("k-NN Distance by Region", fontsize=11, fontweight="bold")
    p = spatial["kruskal_p"]
    ax.text(0.95, 0.95,
            f"Kruskal-Wallis\nH = {spatial['kruskal_stat']:.1f}\np = {p:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def plot_quadrants_across_projections(knn_stats, spatial, model_name, filename):
    """Plot PCA quadrant labels on PCA, t-SNE, and UMAP projections."""
    sample = knn_stats["sample"]
    quadrant_labels = spatial["quadrant_labels"]
    pca_2d = spatial["pca_2d"]

    quad_colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]
    quad_names = ["Q1", "Q2", "Q3", "Q4"]

    # Run t-SNE
    print(f"    t-SNE for {model_name}...", flush=True)
    n_pca = min(30, sample.shape[1])
    pca_pre = PCA(n_components=n_pca, random_state=42).fit_transform(sample)
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30,
                   learning_rate="auto", init="pca", max_iter=1000).fit_transform(pca_pre)

    # Run UMAP
    projections = [("PCA", pca_2d), ("t-SNE", tsne_2d)]
    if HAS_UMAP:
        print(f"    UMAP for {model_name}...", flush=True)
        umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1).fit_transform(sample)
        projections.append(("UMAP", umap_2d))

    n_proj = len(projections)
    fig, axes = plt.subplots(1, n_proj, figsize=(6 * n_proj, 5.5), facecolor=BG,
                             constrained_layout=True)
    fig.suptitle(
        f"PCA Quadrants Mapped Across Projections: {model_name}",
        fontsize=14, fontweight="bold", color=C_DARK,
    )

    for ax, (method, coords) in zip(axes, projections):
        for q in range(4):
            mask = quadrant_labels == q
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[quad_colors[q]],
                       s=6, alpha=0.5, label=quad_names[q])
        ax.set_title(method, fontsize=12, fontweight="bold", color=C_DARK)
        ax.set_xlabel(f"{method} 1", fontsize=9)
        ax.set_ylabel(f"{method} 2", fontsize=9)
        ax.legend(fontsize=7, markerscale=3, loc="best")
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_box_aspect(1)

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def print_summary(knn_stats, spatial, model_name, k):
    """Print a text summary of the locality test results."""
    dk = knn_stats["dk"]
    ratio = knn_stats["ratio"]
    cv = np.std(dk) / np.mean(dk)
    cv_ratio = np.std(ratio) / np.mean(ratio)

    print(f"\n  {model_name}:")
    print(f"    {k}-NN distance — mean: {np.mean(dk):.3f}, "
          f"std: {np.std(dk):.3f}, CV: {cv:.3f}")
    print(f"    Expansion ratio (d_{k}/d_1) — mean: {np.mean(ratio):.2f}, "
          f"median: {np.median(ratio):.2f}, CV: {cv_ratio:.3f}")
    print(f"    Spatial uniformity (Kruskal-Wallis) — "
          f"H={spatial['kruskal_stat']:.1f}, p={spatial['kruskal_p']:.2e}")

    if cv < 0.3:
        print(f"    -> CV < 0.3: local density is fairly uniform")
    elif cv < 0.6:
        print(f"    -> CV 0.3-0.6: moderate density variation")
    else:
        print(f"    -> CV > 0.6: high density variation, manifold assumption may be weak")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fnlwgt", action="store_true",
                        help="Use models retrained without fnlwgt")
    parser.add_argument("--k", type=int, default=15,
                        help="Number of neighbors (default: 15, matches UMAP default)")
    parser.add_argument("--sample-size", type=int, default=3000,
                        help="Number of points to sample (default: 3000)")
    args = parser.parse_args()

    print("=" * 60)
    print("k-NN Locality Test — UMAP Manifold Assumption Validation")
    print("=" * 60)

    # Load data and models
    print("\nLoading data and models...")
    data = prepare_data(drop_fnlwgt=args.no_fnlwgt)
    xgb_model, nn_model, X_test_xgb = load_models(data, drop_fnlwgt=args.no_fnlwgt)

    # Extract embeddings
    print("\nExtracting embeddings...")
    nn_hidden = extract_nn_last_hidden(nn_model, data["X_test_scaled"])
    xgb_leaves = xgb_model.apply(X_test_xgb)
    print(f"  NN last hidden: {nn_hidden.shape}")
    print(f"  XGBoost leaves: {xgb_leaves.shape}")

    k = args.k
    sample_size = args.sample_size

    # Run tests
    for name, emb, fname, proj_fname in [
        ("Neural Network", nn_hidden, "knn_locality_nn.png", "knn_quadrants_nn.png"),
        ("XGBoost", xgb_leaves, "knn_locality_xgb.png", "knn_quadrants_xgb.png"),
    ]:
        print(f"\nRunning k-NN locality test for {name} (k={k})...")
        knn = compute_knn_stats(emb, k=k, sample_size=sample_size)
        spatial = spatial_uniformity_test(knn["sample"], knn["dk"])
        plot_locality_report(knn, spatial, name, k, fname)
        plot_quadrants_across_projections(knn, spatial, name, proj_fname)
        print_summary(knn, spatial, name, k)

    print("\n" + "=" * 60)
    print("Done. Plots saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
