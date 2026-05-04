"""
Assignment 2 — Technical Audit (expanded)
==========================================
Latent-space visualization (PCA + t-SNE + UMAP), concept probing (18 concepts),
TCAV for NN, concept sensitivity for XGBoost, CKA alignment, fairness metrics,
and proxy feature analysis.

Usage:
    python assignment2/technical_audit.py
    python assignment2/technical_audit.py --no-fnlwgt   # use retrained models
"""

from pathlib import Path
import os
import sys
import warnings
import argparse

# Limit CPU parallelism before importing heavy libraries
N_THREADS = str(max(1, os.cpu_count() // 2))
os.environ.setdefault("OMP_NUM_THREADS", N_THREADS)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", N_THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm

import xgboost as xgb
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(int(N_THREADS))
tf.config.threading.set_inter_op_parallelism_threads(2)

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from concepts import (
    prepare_data, load_models, get_all_concepts, STAKEHOLDER_CONCEPTS,
    OUTPUT_DIR, C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GRAY, C_DARK, BG,
    SENSITIVE_PREFIXES, drop_cols_by_prefix,
)

# Try importing umap; fall back gracefully
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. UMAP plots will be skipped.")
    print("  Install with: pip install umap-learn")


# ============================================================
# 1. Latent space extraction
# ============================================================
def extract_nn_hidden(nn_model, X_scaled):
    """Extract activations from each hidden Dense layer of the NN."""
    _ = nn_model.predict(X_scaled[:1], verbose=0)
    layer_outputs = {}
    inp = tf.keras.Input(shape=(X_scaled.shape[1],))
    x = inp
    for i, layer in enumerate(nn_model.layers):
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Dense) and layer != nn_model.layers[-1]:
            extractor = tf.keras.Model(inputs=inp, outputs=x)
            activations = extractor.predict(X_scaled, verbose=0)
            layer_outputs[f"dense_{i}_{layer.units}"] = activations
            print(f"  Extracted layer {layer.name}: shape {activations.shape}")
    return layer_outputs


def extract_xgb_leaves(xgb_model, X_test_xgb):
    """Get leaf node assignment embeddings from XGBoost."""
    leaves = xgb_model.apply(X_test_xgb)
    print(f"  XGBoost leaf embeddings: shape {leaves.shape}")
    return leaves


# ============================================================
# 2. Dimensionality reduction (PCA, t-SNE, UMAP)
# ============================================================
def run_dim_reduction(embeddings, name, sample_size=2000):
    """Run PCA, t-SNE, and UMAP on embeddings. Returns dict of 2D coords + sample indices."""
    n = len(embeddings)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample = embeddings[idx]

    scaled = StandardScaler().fit_transform(sample)

    results = {}

    # PCA (direct 2D)
    print(f"  PCA for {name}...", flush=True)
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(scaled)
    results["PCA"] = pca_2d

    # t-SNE (PCA pre-reduction then t-SNE)
    n_pca = min(30, scaled.shape[1])
    pca_pre = PCA(n_components=n_pca, random_state=42).fit_transform(scaled)
    print(f"  t-SNE for {name} ({len(idx)} samples)...", flush=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                learning_rate="auto", init="pca", max_iter=1000)
    results["t-SNE"] = tsne.fit_transform(pca_pre)

    # UMAP
    if HAS_UMAP:
        print(f"  UMAP for {name}...", flush=True)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        results["UMAP"] = reducer.fit_transform(scaled)

    return results, idx


def plot_latent_space_grid(dim_results, idx, demo_df, y_test, model_name, filename):
    """
    Create a grid: rows = dim-reduction methods (PCA, t-SNE, UMAP),
    cols = coloring attributes (income, sex, race, age).
    """
    methods = list(dim_results.keys())
    n_methods = len(methods)
    attrs = ["Income", "Sex", "Race", "Age Group"]

    fig, axes = plt.subplots(n_methods, 4, figsize=(20, 5.5 * n_methods),
                             facecolor=BG, gridspec_kw={"wspace": 0.35, "hspace": 0.35})
    if n_methods == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Latent Space: {model_name}\nInternal representations colored by demographic attributes",
                 fontsize=14, fontweight="bold", color=C_DARK, y=0.98)

    df_sample = demo_df.iloc[idx].copy()
    y_sample = y_test.iloc[idx]
    age_bins = [0, 25, 40, 60, 100]
    age_labels = ["<=25", "26-40", "41-60", "60+"]
    df_sample["age_group"] = pd.cut(df_sample["age"], bins=age_bins, labels=age_labels)

    coloring = [
        ("Income", y_sample.values, "RdYlGn", True),
        ("Sex", df_sample["sex"].values, None, False),
        ("Race", df_sample["race"].values, None, False),
        ("Age Group", df_sample["age_group"].astype(str).values, None, False),
    ]

    for row, method in enumerate(methods):
        z = dim_results[method]
        for col, (title, values, cmap, is_numeric) in enumerate(coloring):
            ax = axes[row, col]
            ax.set_facecolor(BG)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", color=C_DARK)
            ax.set_xlabel(f"{method} 1", fontsize=8)
            ax.set_ylabel(f"{method} 2", fontsize=8)

            if col == 0:
                ax.text(-0.15, 0.5, method, transform=ax.transAxes,
                        fontsize=12, fontweight="bold", color=C_DARK,
                        rotation=90, va="center", ha="center")

            if is_numeric:
                sc = ax.scatter(z[:, 0], z[:, 1], c=values, cmap=cmap, s=6, alpha=0.6)
                plt.colorbar(sc, ax=ax, label=">50K", shrink=0.7)
            else:
                unique_vals = sorted(set(values))
                colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_vals), 2)))
                for j, val in enumerate(unique_vals):
                    mask = values == val
                    ax.scatter(z[mask, 0], z[mask, 1], c=[colors[j]], s=6,
                               alpha=0.5, label=val)
                ax.legend(fontsize=6, markerscale=2, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 3. Concept probing (expanded: 18 concepts)
# ============================================================
def concept_probe_single(embeddings, concept_mask, concept_name, n_threads):
    """Train logistic regression probe for a binary concept."""
    mask = ~pd.isna(concept_mask)
    X_probe = embeddings[mask]
    y_probe = concept_mask[mask].astype(int)

    class_counts = pd.Series(y_probe).value_counts()
    if len(class_counts) < 2:
        return None
    cv_folds = min(5, class_counts.min())
    if cv_folds < 2:
        return None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    probe = LogisticRegression(max_iter=2000, n_jobs=n_threads)
    baseline = DummyClassifier(strategy="most_frequent")

    probe_scores = cross_val_score(probe, X_probe, y_probe, cv=cv, scoring="accuracy")
    baseline_scores = cross_val_score(baseline, X_probe, y_probe, cv=cv, scoring="accuracy")

    return {
        "concept": concept_name,
        "n_positive": int(y_probe.sum()),
        "n_total": len(y_probe),
        "probe_mean": probe_scores.mean(),
        "probe_std": probe_scores.std(),
        "baseline_mean": baseline_scores.mean(),
        "improvement": probe_scores.mean() - baseline_scores.mean(),
    }


def run_expanded_concept_probes(embeddings_dict, raw_test_df, model_name):
    """
    Run probes for all 18 concepts across all available embedding layers.
    embeddings_dict: {layer_name: numpy array}
    raw_test_df: raw (pre-encoding) test DataFrame for concept evaluation
    """
    all_concepts = get_all_concepts()
    results = []

    for layer_name, emb in tqdm(embeddings_dict.items(), desc=f"  Probing {model_name}", leave=False):
        for concept_name, concept_fn in all_concepts.items():
            try:
                concept_mask = concept_fn(raw_test_df).values.astype(float)
            except Exception:
                continue
            r = concept_probe_single(emb, concept_mask, concept_name, int(N_THREADS))
            if r:
                r["model"] = model_name
                r["layer"] = layer_name
                results.append(r)

    return results


def plot_concept_probe_heatmap(probe_df, filename):
    """Heatmap: concept x layer, colored by probe accuracy improvement."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), facecolor=BG)
    fig.suptitle("Linear Probing: Concept Encoding Strength Across Layers\n"
                 "(Improvement over majority-class baseline)",
                 fontsize=13, fontweight="bold", color=C_DARK, y=0.98)

    for ax_i, (model_name, color) in enumerate([("NN", C_BLUE), ("XGBoost", C_ORANGE)]):
        ax = axes[ax_i]
        df_m = probe_df[probe_df["model"] == model_name]
        if df_m.empty:
            ax.set_visible(False)
            continue

        pivot = df_m.pivot_table(index="concept", columns="layer", values="improvement", aggfunc="mean")
        pivot = pivot.reindex(sorted(pivot.index))

        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=-0.05, vmax=0.3)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.replace("dense_", "L") for c in pivot.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val) > 0.15 else C_DARK)

        ax.set_title(f"{model_name}", fontsize=12, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Improvement over baseline")
        ax.set_facecolor(BG)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def plot_concept_probe_bars(probe_df, filename):
    """Bar chart: probe accuracy vs baseline for each concept, grouped by model."""
    # Use only the last (deepest) layer per model for the summary bar chart
    all_concepts_list = list(get_all_concepts().keys())
    models = probe_df["model"].unique()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    x = np.arange(len(all_concepts_list))
    w = 0.35

    for i, model in enumerate(models):
        df_m = probe_df[probe_df["model"] == model]
        # Pick the last layer for each model
        last_layer = df_m["layer"].unique()[-1]
        df_layer = df_m[df_m["layer"] == last_layer]

        vals = []
        baselines = []
        for c in all_concepts_list:
            row = df_layer[df_layer["concept"] == c]
            vals.append(row["probe_mean"].values[0] if len(row) > 0 else np.nan)
            baselines.append(row["baseline_mean"].values[0] if len(row) > 0 else np.nan)

        color = C_BLUE if "NN" in model else C_ORANGE
        ax.bar(x + i * w, vals, w, label=f"{model} probe", color=color, edgecolor="white")
        ax.bar(x + i * w, baselines, w, color=C_GRAY, alpha=0.25, edgecolor="white")

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(all_concepts_list, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Concept Probing: Can concepts be recovered from model representations?",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.bar([], [], color=C_GRAY, alpha=0.25, label="Majority-class baseline")
    ax.legend(fontsize=9)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 4. TCAV for Neural Network
# ============================================================
def build_nn_forward_model(nn_model, X_scaled):
    """Build a functional model that outputs all hidden layers + pre-sigmoid logit.

    We use the pre-sigmoid logit (instead of the sigmoid output) because
    sigmoid saturates for confident predictions, producing near-zero gradients
    that make TCAV scores degenerate.
    """
    _ = nn_model.predict(X_scaled[:1], verbose=0)
    inp = tf.keras.Input(shape=(X_scaled.shape[1],))
    x = inp
    hidden_outputs = {}
    for i, layer in enumerate(nn_model.layers):
        if layer == nn_model.layers[-1] and isinstance(layer, tf.keras.layers.Dense):
            # Create a copy of the final Dense layer but with linear activation
            # to get the pre-sigmoid logit
            logit_layer = tf.keras.layers.Dense(
                layer.units, activation="linear", name="logit_output"
            )
            logit = logit_layer(x)
            # Build model first, then copy weights
            hidden_outputs_list = list(hidden_outputs.values())
            all_outputs = hidden_outputs_list + [logit]
            model = tf.keras.Model(inputs=inp, outputs=all_outputs)
            # Copy the original layer's weights (kernel + bias) to the logit layer
            logit_layer.set_weights(layer.get_weights())
            layer_names = list(hidden_outputs.keys())
            return model, layer_names
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Dense) and layer != nn_model.layers[-1]:
            hidden_outputs[f"dense_{i}_{layer.units}"] = x

    # Fallback (shouldn't reach here)
    all_outputs = list(hidden_outputs.values()) + [x]
    model = tf.keras.Model(inputs=inp, outputs=all_outputs)
    layer_names = list(hidden_outputs.keys())
    return model, layer_names


def compute_tcav_score(nn_model, X_scaled, concept_positive_idx, concept_negative_idx,
                       target_class=1, n_random=10):
    """
    Compute TCAV scores for each hidden layer of the NN.

    For each layer:
    1. Train a linear CAV (concept activation vector) on hidden activations
    2. Compute directional derivatives: fraction of target-class inputs
       where gradient w.r.t. hidden activations aligns with CAV direction

    Returns dict: {layer_name: {"tcav_score": float, "p_value": float}}
    """
    forward_model, layer_names = build_nn_forward_model(nn_model, X_scaled)

    X_tensor = tf.constant(X_scaled, dtype=tf.float32)

    # Get activations for concept-positive and concept-negative
    all_acts = forward_model(X_tensor, training=False)
    hidden_acts = {name: all_acts[i].numpy() for i, name in enumerate(layer_names)}

    # Get target-class samples (predicted >50K)
    predictions = all_acts[-1].numpy().ravel()
    target_idx = np.where(predictions >= 0.5)[0]
    if len(target_idx) < 10:
        target_idx = np.arange(len(X_scaled))  # fallback: use all

    results = {}

    for layer_idx, layer_name in enumerate(layer_names):
        acts = hidden_acts[layer_name]

        # Train CAV: linear classifier on concept-positive vs concept-negative activations
        pos_acts = acts[concept_positive_idx]
        neg_acts = acts[concept_negative_idx]

        # Balance classes
        n_min = min(len(pos_acts), len(neg_acts))
        if n_min < 5:
            results[layer_name] = {"tcav_score": np.nan, "p_value": 1.0}
            continue

        rng = np.random.default_rng(42)
        pos_sample = pos_acts[rng.choice(len(pos_acts), n_min, replace=False)]
        neg_sample = neg_acts[rng.choice(len(neg_acts), n_min, replace=False)]

        cav_X = np.vstack([pos_sample, neg_sample])
        cav_y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(cav_X, cav_y)
        cav_vector = clf.coef_[0]  # direction pointing toward concept-positive
        cav_vector = cav_vector / (np.linalg.norm(cav_vector) + 1e-10)

        # Compute directional derivatives for target-class samples
        # Use a subset to save compute
        subset_target = target_idx[:min(500, len(target_idx))]
        X_subset = tf.constant(X_scaled[subset_target], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_subset)
            outputs = forward_model(X_subset, training=False)
            hidden_out = outputs[layer_idx]
            final_out = outputs[-1]

        # Gradient of final output w.r.t. hidden layer activations
        grad = tape.gradient(final_out, hidden_out)
        if grad is None:
            results[layer_name] = {"tcav_score": np.nan, "p_value": 1.0}
            continue

        grad_np = grad.numpy()
        # Directional derivative = dot product of gradient with CAV
        directional_derivs = grad_np @ cav_vector
        tcav_score = (directional_derivs > 0).mean()

        # Statistical test: compare with random CAVs
        random_scores = []
        for _ in range(n_random):
            random_vec = rng.standard_normal(len(cav_vector))
            random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-10)
            rand_derivs = grad_np @ random_vec
            random_scores.append((rand_derivs > 0).mean())

        random_scores = np.array(random_scores)
        # Two-sided p-value: proportion of random scores as extreme as observed
        p_value = (np.abs(random_scores - 0.5) >= np.abs(tcav_score - 0.5)).mean()

        results[layer_name] = {"tcav_score": tcav_score, "p_value": p_value}

    return results


def run_tcav_analysis(nn_model, X_scaled, raw_test_df):
    """Run TCAV for all 18 concepts across NN layers."""
    all_concepts = get_all_concepts()
    results = []

    for concept_name, concept_fn in tqdm(all_concepts.items(), desc="  TCAV (NN)"):
        try:
            concept_mask = concept_fn(raw_test_df).values
        except Exception:
            continue

        pos_idx = np.where(concept_mask)[0]
        neg_idx = np.where(~concept_mask)[0]

        if len(pos_idx) < 5 or len(neg_idx) < 5:
            continue

        tcav_results = compute_tcav_score(nn_model, X_scaled, pos_idx, neg_idx)

        for layer_name, scores in tcav_results.items():
            results.append({
                "concept": concept_name,
                "layer": layer_name,
                "tcav_score": scores["tcav_score"],
                "p_value": scores["p_value"],
            })

    return pd.DataFrame(results)


def plot_tcav_by_stakeholder(tcav_df, filename):
    """Plot TCAV scores grouped by stakeholder, with bars per NN layer."""
    n_stakeholders = len(STAKEHOLDER_CONCEPTS)
    fig, axes = plt.subplots(1, n_stakeholders, figsize=(18, 7), facecolor=BG)
    fig.suptitle("TCAV Scores: Neural Network Layers\n"
                 "Score > 0.5 = concept positively influences >50K prediction",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.01)

    colors_layers = [C_BLUE, C_GREEN, C_PURPLE, C_ORANGE]
    layers = sorted(tcav_df["layer"].unique())

    for ax_i, (stakeholder, concepts) in enumerate(STAKEHOLDER_CONCEPTS.items()):
        ax = axes[ax_i]
        concept_names = list(concepts.keys())
        x = np.arange(len(concept_names))
        w = 0.8 / max(len(layers), 1)

        for l_i, layer in enumerate(layers):
            vals = []
            for c in concept_names:
                row = tcav_df[(tcav_df["concept"] == c) & (tcav_df["layer"] == layer)]
                vals.append(row["tcav_score"].values[0] if len(row) > 0 else np.nan)

            short_label = layer.replace("dense_", "L")
            ax.bar(x + l_i * w, vals, w, label=short_label,
                   color=colors_layers[l_i % len(colors_layers)], edgecolor="white")

        ax.axhline(0.5, color=C_GRAY, ls="--", lw=1, alpha=0.7)
        ax.set_xticks(x + w * (len(layers) - 1) / 2)
        ax.set_xticklabels(concept_names, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("TCAV Score", fontsize=9)
        ax.set_title(stakeholder, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 5. Concept sensitivity for XGBoost (TCAV analog)
# ============================================================
def compute_xgb_concept_sensitivity(xgb_model, X_test_xgb, xgb_leaves, raw_test_df):
    """
    For XGBoost, we cannot compute gradients. Instead we measure:
    (a) How well a linear probe on leaf embeddings can separate concept+/concept-
    (b) How much the CAV projection correlates with model prediction probability

    This gives a "concept sensitivity" score analogous to TCAV.
    """
    all_concepts = get_all_concepts()
    results = []

    xgb_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
    leaves_scaled = StandardScaler().fit_transform(xgb_leaves.astype(float))

    for concept_name, concept_fn in tqdm(all_concepts.items(), desc="  Concept sensitivity (XGB)"):
        try:
            concept_mask = concept_fn(raw_test_df).values
        except Exception:
            continue

        pos_idx = np.where(concept_mask)[0]
        neg_idx = np.where(~concept_mask)[0]

        if len(pos_idx) < 10 or len(neg_idx) < 10:
            continue

        # (a) Train CAV in leaf embedding space
        n_min = min(len(pos_idx), len(neg_idx))
        rng = np.random.default_rng(42)
        pos_sample = leaves_scaled[rng.choice(pos_idx, min(n_min, 2000), replace=False)]
        neg_sample = leaves_scaled[rng.choice(neg_idx, min(n_min, 2000), replace=False)]

        cav_X = np.vstack([pos_sample, neg_sample])
        cav_y = np.concatenate([np.ones(len(pos_sample)), np.zeros(len(neg_sample))])

        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(cav_X, cav_y)
        cav_accuracy = clf.score(cav_X, cav_y)

        cav_vector = clf.coef_[0]
        cav_vector = cav_vector / (np.linalg.norm(cav_vector) + 1e-10)

        # (b) Project all samples onto CAV, correlate with prediction probability
        projections = leaves_scaled @ cav_vector
        correlation = np.corrcoef(projections, xgb_proba)[0, 1]

        # Concept sensitivity score: fraction of positive-class predictions
        # where projection is above median (analogous to TCAV > 0.5)
        target_idx = np.where(xgb_proba >= 0.5)[0]
        if len(target_idx) > 0:
            median_proj = np.median(projections)
            sensitivity = (projections[target_idx] > median_proj).mean()
        else:
            sensitivity = np.nan

        results.append({
            "concept": concept_name,
            "cav_accuracy": cav_accuracy,
            "correlation_with_pred": correlation,
            "sensitivity_score": sensitivity,
        })

    return pd.DataFrame(results)


def plot_xgb_concept_sensitivity(xgb_sens_df, filename):
    """Bar chart of XGBoost concept sensitivity scores by stakeholder."""
    n_stakeholders = len(STAKEHOLDER_CONCEPTS)
    fig, axes = plt.subplots(1, n_stakeholders, figsize=(18, 7), facecolor=BG)
    fig.suptitle("XGBoost Concept Sensitivity (TCAV Analog)\n"
                 "Based on CAV projection in leaf embedding space",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.01)

    for ax_i, (stakeholder, concepts) in enumerate(STAKEHOLDER_CONCEPTS.items()):
        ax = axes[ax_i]
        concept_names = list(concepts.keys())

        vals = []
        corrs = []
        for c in concept_names:
            row = xgb_sens_df[xgb_sens_df["concept"] == c]
            vals.append(row["sensitivity_score"].values[0] if len(row) > 0 else np.nan)
            corrs.append(row["correlation_with_pred"].values[0] if len(row) > 0 else 0)

        x = np.arange(len(concept_names))
        colors = [C_RED if abs(c) > 0.15 else C_ORANGE for c in corrs]
        ax.bar(x, vals, color=colors, edgecolor="white")
        ax.axhline(0.5, color=C_GRAY, ls="--", lw=1, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(concept_names, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Sensitivity Score", fontsize=9)
        ax.set_title(stakeholder, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate correlation values
        for i, (v, c) in enumerate(zip(vals, corrs)):
            if not np.isnan(v):
                ax.text(i, v + 0.02, f"r={c:.2f}", ha="center", fontsize=6, color=C_DARK)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 6. CKA — Linear Centered Kernel Alignment
# ============================================================
def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def plot_cka_heatmap(nn_layers, xgb_leaves, filename):
    """CKA heatmap between all representation layers."""
    rng = np.random.default_rng(42)
    n = min(1000, len(xgb_leaves), min(v.shape[0] for v in nn_layers.values()))
    idx = rng.choice(min(len(xgb_leaves), min(v.shape[0] for v in nn_layers.values())),
                     size=n, replace=False)

    all_reps = {}
    for name, act in nn_layers.items():
        short_name = name.replace("dense_", "NN Layer ")
        all_reps[short_name] = StandardScaler().fit_transform(act[idx])
    all_reps["XGB Leaves"] = StandardScaler().fit_transform(xgb_leaves[idx].astype(float))

    names = list(all_reps.keys())
    m = len(names)
    cka_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            cka_matrix[i, j] = linear_cka(all_reps[names[i]], all_reps[names[j]])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    im = ax.imshow(cka_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)

    for i in range(m):
        for j in range(m):
            ax.text(j, i, f"{cka_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cka_matrix[i, j] > 0.5 else C_DARK)

    plt.colorbar(im, ax=ax, label="Linear CKA", shrink=0.8)
    ax.set_title("Representational Alignment (Linear CKA)\n"
                 "NN Hidden Layers vs XGBoost Leaf Embeddings",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_facecolor(BG)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 7. Fairness metrics
# ============================================================
def compute_fairness_metrics(y_true, y_pred, sensitive_attr, attr_name):
    groups = sorted(set(sensitive_attr))
    rows = []
    for g in groups:
        mask = sensitive_attr == g
        yt = y_true[mask]
        yp = y_pred[mask]
        n = mask.sum()
        approval_rate = yp.mean()
        accuracy = accuracy_score(yt, yp)
        pos_mask = yt == 1
        tpr = yp[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
        neg_mask = yt == 0
        fpr = yp[neg_mask].mean() if neg_mask.sum() > 0 else np.nan
        rows.append({
            "attribute": attr_name, "group": g, "n_samples": n,
            "approval_rate": approval_rate, "accuracy": accuracy,
            "tpr": tpr, "fpr": fpr,
        })

    df = pd.DataFrame(rows)
    rates = df["approval_rate"]
    if rates.max() > 0:
        df["disparate_impact_ratio"] = rates.min() / rates.max()
    else:
        df["disparate_impact_ratio"] = np.nan
    tpr_gap = df["tpr"].max() - df["tpr"].min()
    fpr_gap = df["fpr"].max() - df["fpr"].min()
    df["equalized_odds_gap"] = max(tpr_gap, fpr_gap)
    return df


def run_fairness_analysis(y_test, xgb_pred, nn_pred, demo_test):
    all_results = []
    for attr in ["sex", "race"]:
        attr_vals = demo_test[attr].values
        for model_name, preds in [("XGBoost", xgb_pred), ("NN", nn_pred)]:
            df_fair = compute_fairness_metrics(y_test.values, preds, attr_vals, attr)
            df_fair["model"] = model_name
            all_results.append(df_fair)
    return pd.concat(all_results, ignore_index=True)


def plot_fairness(fairness_df, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)
    fig.suptitle("Fairness Audit: Demographic Parity & Equalized Odds",
                 fontsize=14, fontweight="bold", color=C_DARK, y=0.98)

    for ax, attr in zip(axes, ["sex", "race"]):
        df_attr = fairness_df[fairness_df["attribute"] == attr]
        groups = df_attr["group"].unique()
        x = np.arange(len(groups))
        w = 0.35

        xgb_rates = [df_attr[(df_attr["group"] == g) & (df_attr["model"] == "XGBoost")]["approval_rate"].values[0]
                      for g in groups]
        nn_rates = [df_attr[(df_attr["group"] == g) & (df_attr["model"] == "NN")]["approval_rate"].values[0]
                     for g in groups]

        b1 = ax.bar(x - w/2, xgb_rates, w, label="XGBoost", color=C_ORANGE, edgecolor="white")
        b2 = ax.bar(x + w/2, nn_rates, w, label="Neural Network", color=C_BLUE, edgecolor="white")

        max_rate = max(max(xgb_rates), max(nn_rates))
        ax.axhline(max_rate * 0.8, color=C_RED, ls="--", lw=1.5, alpha=0.7, label="80% rule threshold")

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("Approval Rate", fontsize=10)
        ax.set_title(f"Approval Rate by {attr.title()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)

        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{bar.get_height():.1%}", ha="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 8. Proxy feature analysis
# ============================================================
def proxy_analysis(X_raw):
    from scipy.stats import chi2_contingency

    df = X_raw.copy()
    le_sex = LabelEncoder()
    df["sex_encoded"] = le_sex.fit_transform(df["sex"])
    le_race = LabelEncoder()
    df["race_encoded"] = le_race.fit_transform(df["race"])

    numeric_features = ["age", "fnlwgt", "education-num",
                        "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = ["marital-status", "relationship", "occupation", "workclass"]

    proxy_results = []
    for sensitive, encoded_col in [("sex", "sex_encoded"), ("race", "race_encoded")]:
        for feat in numeric_features:
            if feat not in df.columns:
                continue
            corr = df[encoded_col].corr(df[feat])
            proxy_results.append({
                "sensitive_feature": sensitive, "proxy_candidate": feat,
                "correlation": corr, "abs_correlation": abs(corr),
            })
        for cat_feat in categorical_features:
            contingency = pd.crosstab(df[sensitive], df[cat_feat])
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            n = contingency.sum().sum()
            k = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0
            proxy_results.append({
                "sensitive_feature": sensitive, "proxy_candidate": cat_feat,
                "correlation": cramers_v, "abs_correlation": cramers_v,
            })

    return pd.DataFrame(proxy_results).sort_values("abs_correlation", ascending=False)


def plot_proxy_analysis(proxy_df, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("Proxy Feature Analysis\n"
                 "How strongly do remaining features correlate with removed sensitive attributes?",
                 fontsize=13, fontweight="bold", color=C_DARK, y=0.99)

    for ax, sensitive in zip(axes, ["sex", "race"]):
        df_s = proxy_df[proxy_df["sensitive_feature"] == sensitive].sort_values(
            "abs_correlation", ascending=True)
        colors = [C_RED if v > 0.3 else C_ORANGE if v > 0.1 else C_GRAY
                  for v in df_s["abs_correlation"]]
        ax.barh(df_s["proxy_candidate"], df_s["abs_correlation"], color=colors, edgecolor="white")
        ax.axvline(0.3, color=C_RED, ls="--", lw=1, alpha=0.7, label="Strong proxy (>0.3)")
        ax.axvline(0.1, color=C_ORANGE, ls="--", lw=1, alpha=0.7, label="Moderate proxy (>0.1)")
        ax.set_title(f"Proxies for {sensitive.title()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Correlation / Cramer's V", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        for i, (_, row) in enumerate(df_s.iterrows()):
            ax.text(row["abs_correlation"] + 0.005, i, f"{row['abs_correlation']:.3f}",
                    va="center", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fnlwgt", action="store_true",
                        help="Use models retrained without fnlwgt")
    args = parser.parse_args()

    drop_fnlwgt = args.no_fnlwgt

    print("=" * 60)
    print("Assignment 2 — Technical Audit (expanded)")
    if drop_fnlwgt:
        print("  Mode: without fnlwgt")
    print("=" * 60)

    steps = tqdm(total=9, desc="Overall progress", position=0, leave=True)

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    steps.set_description("[1/9] Loading data")
    data = prepare_data(drop_fnlwgt=drop_fnlwgt)
    print(f"  Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")
    steps.update(1)

    # ----------------------------------------------------------
    # 2. Load models & predictions
    # ----------------------------------------------------------
    steps.set_description("[2/9] Loading models")
    xgb_model, nn_model, X_test_xgb = load_models(data, drop_fnlwgt=drop_fnlwgt)

    xgb_pred = xgb_model.predict(X_test_xgb)
    xgb_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
    nn_proba = nn_model.predict(data["X_test_scaled"], verbose=0).ravel()
    nn_pred = (nn_proba >= 0.5).astype(int)

    y_test = data["y_test"]
    demo_test = data["demo_test"]
    raw_test = data["raw_test"]

    print(f"  XGBoost — Acc: {accuracy_score(y_test, xgb_pred):.4f}, "
          f"AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
    print(f"  NN      — Acc: {accuracy_score(y_test, nn_pred):.4f}, "
          f"AUC: {roc_auc_score(y_test, nn_proba):.4f}")
    steps.update(1)

    # ----------------------------------------------------------
    # 3. Extract latent representations
    # ----------------------------------------------------------
    steps.set_description("[3/9] Extracting latent representations")
    nn_layers = extract_nn_hidden(nn_model, data["X_test_scaled"])
    xgb_leaves = extract_xgb_leaves(xgb_model, X_test_xgb)
    steps.update(1)

    # ----------------------------------------------------------
    # 4. Latent space visualization (PCA + t-SNE + UMAP)
    # ----------------------------------------------------------
    steps.set_description("[4/9] Dimensionality reduction & visualization")

    last_layer_name = list(nn_layers.keys())[-1]
    nn_last_hidden = nn_layers[last_layer_name]

    nn_dim, idx_nn = run_dim_reduction(nn_last_hidden, "NN hidden layer")
    plot_latent_space_grid(nn_dim, idx_nn, demo_test, y_test, "Neural Network", "nn_latent_space.png")

    xgb_dim, idx_xgb = run_dim_reduction(xgb_leaves, "XGBoost leaves")
    plot_latent_space_grid(xgb_dim, idx_xgb, demo_test, y_test, "XGBoost", "xgb_latent_space.png")
    steps.update(1)

    # ----------------------------------------------------------
    # 5. Expanded concept probing (18 concepts, all layers)
    # ----------------------------------------------------------
    steps.set_description("[5/9] Running expanded concept probes (18 concepts)")

    # Prepare embedding dicts for probing
    nn_emb_dict = {name: act for name, act in nn_layers.items()}
    xgb_leaves_scaled = StandardScaler().fit_transform(xgb_leaves.astype(float))
    xgb_emb_dict = {"xgb_leaves": xgb_leaves_scaled}

    print("\n  Neural Network probes:")
    nn_probes = run_expanded_concept_probes(nn_emb_dict, raw_test, "NN")
    print(f"  NN: {len(nn_probes)} probe results")

    print("\n  XGBoost probes:")
    xgb_probes = run_expanded_concept_probes(xgb_emb_dict, raw_test, "XGBoost")
    print(f"  XGBoost: {len(xgb_probes)} probe results")

    all_probes = nn_probes + xgb_probes
    probe_df = pd.DataFrame(all_probes)
    probe_df.to_csv(OUTPUT_DIR / "concept_probe_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'concept_probe_results.csv'}")

    plot_concept_probe_heatmap(probe_df, "concept_probe_heatmap.png")
    plot_concept_probe_bars(probe_df, "concept_probes_bar.png")
    steps.update(1)

    # ----------------------------------------------------------
    # 6. TCAV for Neural Network
    # ----------------------------------------------------------
    steps.set_description("[6/9] Running TCAV analysis (NN)")
    tcav_df = run_tcav_analysis(nn_model, data["X_test_scaled"], raw_test)
    tcav_df.to_csv(OUTPUT_DIR / "tcav_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'tcav_results.csv'}")

    if not tcav_df.empty:
        plot_tcav_by_stakeholder(tcav_df, "tcav_nn_layers.png")
    steps.update(1)

    # ----------------------------------------------------------
    # 7. Concept sensitivity for XGBoost
    # ----------------------------------------------------------
    steps.set_description("[7/9] Running XGBoost concept sensitivity")
    xgb_sens_df = compute_xgb_concept_sensitivity(xgb_model, X_test_xgb, xgb_leaves, raw_test)
    xgb_sens_df.to_csv(OUTPUT_DIR / "xgb_concept_sensitivity.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'xgb_concept_sensitivity.csv'}")

    if not xgb_sens_df.empty:
        plot_xgb_concept_sensitivity(xgb_sens_df, "xgb_concept_sensitivity.png")
    steps.update(1)

    # ----------------------------------------------------------
    # 8. CKA analysis
    # ----------------------------------------------------------
    steps.set_description("[8/9] Running CKA representational alignment")
    plot_cka_heatmap(nn_layers, xgb_leaves, "cka_heatmap.png")
    steps.update(1)

    # ----------------------------------------------------------
    # 9. Fairness metrics & proxy analysis
    # ----------------------------------------------------------
    steps.set_description("[9/9] Computing fairness metrics & proxy analysis")

    fairness_df = run_fairness_analysis(y_test, xgb_pred, nn_pred, demo_test)
    fairness_df.to_csv(OUTPUT_DIR / "fairness_metrics.csv", index=False)
    plot_fairness(fairness_df, "fairness_audit.png")

    proxy_df = proxy_analysis(data["X_raw"])
    proxy_df.to_csv(OUTPUT_DIR / "proxy_analysis.csv", index=False)
    plot_proxy_analysis(proxy_df, "proxy_analysis.png")

    steps.update(1)
    steps.close()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    suffix = " (no fnlwgt)" if drop_fnlwgt else ""
    print(f"\nModel Performance{suffix}:")
    print(f"  XGBoost — Acc: {accuracy_score(y_test, xgb_pred):.4f}, "
          f"AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
    print(f"  NN      — Acc: {accuracy_score(y_test, nn_pred):.4f}, "
          f"AUC: {roc_auc_score(y_test, nn_proba):.4f}")

    print("\nTop Concept Probe Results (improvement > 0.05):")
    if not probe_df.empty:
        top_probes = probe_df[probe_df["improvement"] > 0.05].sort_values("improvement", ascending=False)
        for _, row in top_probes.head(15).iterrows():
            print(f"  {row['model']:8s} | {row['layer']:20s} | {row['concept']:25s} | "
                  f"+{row['improvement']:.3f}")

    if not tcav_df.empty:
        print("\nTCAV Scores (NN, significant concepts):")
        sig_tcav = tcav_df[tcav_df["p_value"] < 0.5].sort_values("tcav_score", ascending=False)
        for _, row in sig_tcav.head(10).iterrows():
            direction = "+" if row["tcav_score"] > 0.5 else "-"
            print(f"  {row['layer']:20s} | {row['concept']:25s} | "
                  f"TCAV={row['tcav_score']:.3f} {direction} (p={row['p_value']:.2f})")

    print("\nFairness Metrics:")
    for attr in ["sex", "race"]:
        for model in ["XGBoost", "NN"]:
            df_sub = fairness_df[(fairness_df["attribute"] == attr) & (fairness_df["model"] == model)]
            di = df_sub["disparate_impact_ratio"].iloc[0]
            eo = df_sub["equalized_odds_gap"].iloc[0]
            print(f"  {model:8s} | {attr:4s} | DI: {di:.3f} "
                  f"{'(PASS)' if di >= 0.8 else '(FAIL < 0.8)'} | EO gap: {eo:.3f}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
