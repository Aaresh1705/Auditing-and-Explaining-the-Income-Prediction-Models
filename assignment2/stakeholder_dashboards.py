"""
Assignment 2 — Stakeholder Dashboards (v2)
===========================================
Generates 6 polished static dashboards (3 stakeholders x 2 models) with
concept-based explanations, TCAV/concept sensitivity scores, fairness metrics.

Dashboards:
  1. Head of Data Science — XGBoost
  2. Head of Data Science — Neural Network
  3. Company Director — XGBoost
  4. Company Director — Neural Network
  5. Loan Applicant — XGBoost
  6. Loan Applicant — Neural Network

Usage:
    python assignment2/stakeholder_dashboards.py
    python assignment2/stakeholder_dashboards.py --no-fnlwgt
"""

import os
import sys
import argparse

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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import tensorflow as tf
import lime.lime_tabular

tf.config.threading.set_intra_op_parallelism_threads(int(N_THREADS))
tf.config.threading.set_inter_op_parallelism_threads(2)

sys.path.insert(0, os.path.dirname(__file__))
from concepts import (
    prepare_data, load_models, STAKEHOLDER_CONCEPTS, OUTPUT_DIR,
    C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GRAY, C_DARK, BG,
)


# ============================================================
# Shared helpers
# ============================================================
def load_audit_csv(name):
    path = OUTPUT_DIR / name
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {path} not found. Run technical_audit.py first.")
    return pd.DataFrame()


def draw_gauge(ax, proba, accent_col):
    """Probability gauge (semicircle 0-100%)."""
    theta = np.linspace(np.pi, 0, 300)
    arc = plt.cm.RdYlGn(np.linspace(0, 1, 300))
    for i in range(len(theta) - 1):
        ax.plot(np.cos(theta[i:i + 2]), np.sin(theta[i:i + 2]),
                color=arc[i], lw=18, solid_capstyle="butt")
    ang = np.pi - proba * np.pi
    ax.annotate("", xy=(0.6 * np.cos(ang), 0.6 * np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.5, mutation_scale=18))
    ax.plot(0, 0, "o", color=C_DARK, markersize=7, zorder=5)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.2)
    ax.axis("off")
    ax.set_facecolor(BG)
    pct_color = plt.cm.RdYlGn(proba)
    ax.text(0, -0.18, f"{proba * 100:.0f}%", ha="center", fontsize=28,
            fontweight="black", color=pct_color)
    ax.text(0, -0.32, "Approval Probability", ha="center", fontsize=11, color=C_DARK)
    ax.text(-1.15, -0.05, "0%", ha="center", fontsize=9, color=C_GRAY)
    ax.text(1.15, -0.05, "100%", ha="center", fontsize=9, color=C_GRAY)


def draw_concept_bar(ax, concepts, scores, title, accent_col,
                     xlabel="Concept Influence Score", neutral_line=0.5):
    """Horizontal bar chart of concept scores — clean, square-friendly."""
    y = np.arange(len(concepts))
    colors = []
    for s in scores:
        if np.isnan(s):
            colors.append(C_GRAY)
        elif s > neutral_line + 0.1:
            colors.append(C_GREEN)
        elif s < neutral_line - 0.1:
            colors.append(C_RED)
        else:
            colors.append(accent_col)

    ax.barh(y, scores, color=colors, edgecolor="white", height=0.6)
    ax.axvline(neutral_line, color=C_GRAY, ls="--", lw=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(concepts, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK, pad=12)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1)

    for i, s in enumerate(scores):
        if not np.isnan(s):
            ax.text(s + 0.02, i, f"{s:.2f}", va="center", fontsize=9, color=C_DARK)


def get_concept_scores(stakeholder, model_name, tcav_df, xgb_sens_df):
    """Get concept scores for a stakeholder/model pair."""
    concepts = list(STAKEHOLDER_CONCEPTS[stakeholder].keys())
    scores = []

    for c in concepts:
        if model_name == "NN" and not tcav_df.empty:
            # Use first hidden layer TCAV (most informative)
            rows = tcav_df[tcav_df["concept"] == c]
            first_layer = sorted(rows["layer"].unique())[0] if len(rows) > 0 else None
            row = rows[rows["layer"] == first_layer] if first_layer else pd.DataFrame()
            scores.append(row["tcav_score"].values[0] if len(row) > 0 else np.nan)
        elif model_name == "XGBoost" and not xgb_sens_df.empty:
            row = xgb_sens_df[xgb_sens_df["concept"] == c]
            scores.append(row["sensitivity_score"].values[0] if len(row) > 0 else np.nan)
        else:
            scores.append(np.nan)

    return concepts, np.array(scores)


def get_probe_scores(stakeholder, model_name, probe_df):
    """Get probe improvement scores for a stakeholder/model pair."""
    concepts = list(STAKEHOLDER_CONCEPTS[stakeholder].keys())
    scores = []
    model_key = model_name if model_name == "NN" else "XGBoost"

    for c in concepts:
        df_m = probe_df[(probe_df["model"] == model_key) & (probe_df["concept"] == c)]
        if not df_m.empty:
            # Use the layer with highest improvement
            scores.append(df_m["improvement"].max())
        else:
            scores.append(0.0)

    return concepts, np.array(scores)


# Mapping from Loan Applicant concepts to the raw feature prefixes they depend on.
# After one-hot encoding, any column starting with these prefixes belongs to the concept.
CONCEPT_FEATURE_MAP = {
    "College educated": ["education"],
    "Full-time standard": ["hours-per-week"],
    "White collar occupation": ["occupation"],
    "Has investment activity": ["capital-gain", "capital-loss"],
    "Peak career": ["age", "hours-per-week", "education-num"],
    "Single / never married": ["marital-status"],
}


def explain_applicant_lime(model_name, data, xgb_model, nn_model, X_test_xgb,
                           app_idx):
    """
    Run LIME on a single applicant and aggregate feature contributions by concept.

    Returns dict: concept_name -> summed LIME contribution (positive = toward approval).
    """
    feature_names = data["feature_names"]

    if model_name == "XGBoost":
        training_data = data["X_train"].values
        instance = X_test_xgb.iloc[app_idx].values
        predict_fn = lambda x: xgb_model.predict_proba(
            pd.DataFrame(x, columns=feature_names)
            .reindex(columns=xgb_model.get_booster().feature_names, fill_value=0)
        )
    else:  # NN
        training_data = data["X_train_scaled"]
        instance = data["X_test_scaled"][app_idx]
        def predict_fn(x):
            p = nn_model.predict(x, verbose=0).ravel()
            return np.column_stack([1 - p, p])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=["<=50K", ">50K"],
        mode="classification",
        random_state=42,
    )

    explanation = explainer.explain_instance(
        instance, predict_fn, num_features=len(feature_names), labels=(1,),
    )

    # Get contributions for class 1 (>50K = approval)
    contrib_map = dict(explanation.as_list(label=1))

    # Map feature contributions to concepts
    # contrib_map keys are like "education_Bachelors > 0.50" — extract the feature name
    feature_contribs = {}
    for desc, weight in contrib_map.items():
        # LIME descriptions look like "feature_name <= 0.12" or "0.5 < feature_name <= 1.0"
        for fname in feature_names:
            if fname in desc:
                feature_contribs[fname] = feature_contribs.get(fname, 0) + weight
                break

    concept_scores = {}
    for concept_name, prefixes in CONCEPT_FEATURE_MAP.items():
        total = 0.0
        for fname, w in feature_contribs.items():
            if any(fname == p or fname.startswith(p + "_") for p in prefixes):
                total += w
        concept_scores[concept_name] = total

    return concept_scores


# ============================================================
# 1. Head of Data Science Dashboard
# ============================================================
def dashboard_data_scientist(model_name, accent_col, data, model_pred, model_proba,
                             y_test, demo_test):
    """Technical dashboard for model validation."""
    tcav_df = load_audit_csv("tcav_results.csv")
    xgb_sens_df = load_audit_csv("xgb_concept_sensitivity.csv")
    probe_df = load_audit_csv("concept_probe_results.csv")
    fairness_df = load_audit_csv("fairness_metrics.csv")
    proxy_df = load_audit_csv("proxy_analysis.csv")

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    fig.suptitle(f"Technical Audit — {model_name}\nHead of Data Science",
                 fontsize=15, fontweight="bold", color=accent_col, y=0.99)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                           left=0.08, right=0.95, top=0.93, bottom=0.04)

    acc = accuracy_score(y_test, model_pred)
    auc = roc_auc_score(y_test, model_proba)

    # --- Panel 1: Performance + confusion matrix ---
    ax_perf = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, model_pred)
    im = ax_perf.imshow(cm, cmap="Blues", aspect="equal")
    ax_perf.set_xticks([0, 1])
    ax_perf.set_yticks([0, 1])
    ax_perf.set_xticklabels(["Pred <=50K", "Pred >50K"], fontsize=9)
    ax_perf.set_yticklabels(["True <=50K", "True >50K"], fontsize=9)
    for i in range(2):
        for j in range(2):
            ax_perf.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if cm[i, j] > cm.max() / 2 else C_DARK)
    ax_perf.set_title(f"Confusion Matrix\nAcc: {acc:.1%}  |  AUC: {auc:.3f}",
                      fontsize=11, fontweight="bold", color=C_DARK, pad=10)
    ax_perf.set_facecolor(BG)

    # --- Panel 2: Concept sensitivity (TCAV or XGB analog) ---
    ax_tcav = fig.add_subplot(gs[0, 1])
    concepts, scores = get_concept_scores("Head of Data Science", model_name,
                                          tcav_df, xgb_sens_df)
    score_type = "TCAV Score" if model_name == "NN" else "Concept Sensitivity"
    draw_concept_bar(ax_tcav, concepts, scores,
                     f"Concept Influence ({score_type})", accent_col)

    # --- Panel 3: Concept probe heatmap (this model only) ---
    ax_probe = fig.add_subplot(gs[1, 0])
    if not probe_df.empty:
        model_key = model_name if model_name == "NN" else "XGBoost"
        df_m = probe_df[probe_df["model"] == model_key]
        if not df_m.empty:
            pivot = df_m.pivot_table(index="concept", columns="layer",
                                     values="improvement", aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.index))
            im2 = ax_probe.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                                  vmin=-0.05, vmax=0.5)
            ax_probe.set_xticks(range(len(pivot.columns)))
            col_labels = [c.replace("dense_", "Layer ").replace("xgb_leaves", "Leaf Embeddings")
                          for c in pivot.columns]
            ax_probe.set_xticklabels(col_labels, fontsize=9)
            ax_probe.set_yticks(range(len(pivot.index)))
            ax_probe.set_yticklabels(pivot.index, fontsize=9)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax_probe.text(j, i, f"{val:+.2f}", ha="center", va="center",
                                      fontsize=8, color="white" if abs(val) > 0.2 else C_DARK)
            plt.colorbar(im2, ax=ax_probe, shrink=0.6, label="Probe Accuracy - Baseline")

    ax_probe.set_title(f"Linear Probing — {model_name}: Which Concepts Are Encoded?",
                       fontsize=12, fontweight="bold", color=C_DARK, pad=10)
    ax_probe.set_facecolor(BG)

    # --- Panel 3b: ROC-AUC per demographic group ---
    ax_roc = fig.add_subplot(gs[1, 1])
    if not fairness_df.empty:
        model_key = model_name if model_name == "NN" else "XGBoost"
        df_f = fairness_df[fairness_df["model"] == model_key]
        if not df_f.empty:
            all_groups = []
            all_aucs = []
            all_colors = []
            attr_labels = []
            for attr, col in [("sex", C_BLUE), ("race", C_ORANGE)]:
                df_attr = df_f[df_f["attribute"] == attr]
                for _, row in df_attr.iterrows():
                    all_groups.append(row["group"])
                    all_aucs.append(row["roc_auc"])
                    all_colors.append(col)
                    attr_labels.append(attr)

            x = np.arange(len(all_groups))
            ax_roc.bar(x, all_aucs, color=all_colors, edgecolor="white")

            # 80% rule line based on max AUC
            valid_aucs = [a for a in all_aucs if not np.isnan(a)]
            if valid_aucs:
                ax_roc.axhline(max(valid_aucs) * 0.8, color=C_RED, ls="--", lw=1.5,
                               alpha=0.7, label="80% rule threshold")
                y_floor = max(0.5, min(valid_aucs) - 0.05)
                ax_roc.set_ylim(y_floor, 1.0)

            ax_roc.set_xticks(x)
            ax_roc.set_xticklabels(all_groups, fontsize=8, rotation=30, ha="right")
            ax_roc.set_ylabel("ROC-AUC", fontsize=10)

            # Add value labels on bars
            for i, a in enumerate(all_aucs):
                if not np.isnan(a):
                    ax_roc.text(i, a + 0.003, f"{a:.3f}", ha="center", fontsize=7)

            # Legend for attribute colours
            sex_patch = mpatches.Patch(color=C_BLUE, label="Sex")
            race_patch = mpatches.Patch(color=C_ORANGE, label="Race")
            ax_roc.legend(handles=[sex_patch, race_patch,
                          plt.Line2D([], [], color=C_RED, ls="--", label="80% rule")],
                          fontsize=8)

    ax_roc.set_title(f"Demographic Parity: ROC-AUC — {model_name}",
                     fontsize=11, fontweight="bold", color=C_DARK, pad=10)
    ax_roc.set_facecolor(BG)
    ax_roc.spines[["top", "right"]].set_visible(False)

    # --- Panel 4: Fairness — Approval rates by demographic group ---
    ax_fair = fig.add_subplot(gs[2, 0])
    if not fairness_df.empty:
        model_key = model_name if model_name == "NN" else "XGBoost"
        df_f = fairness_df[fairness_df["model"] == model_key]
        if not df_f.empty:
            all_groups = []
            all_rates = []
            all_colors = []
            for attr, col in [("sex", C_BLUE), ("race", C_ORANGE)]:
                df_attr = df_f[df_f["attribute"] == attr]
                for _, row in df_attr.iterrows():
                    all_groups.append(row["group"])
                    all_rates.append(row["approval_rate"])
                    all_colors.append(col)

            x = np.arange(len(all_groups))
            ax_fair.bar(x, all_rates, color=all_colors, edgecolor="white")
            max_rate = max(all_rates) if all_rates else 1
            ax_fair.axhline(max_rate * 0.8, color=C_RED, ls="--", lw=1.5, alpha=0.7)
            ax_fair.set_xticks(x)
            ax_fair.set_xticklabels(all_groups, fontsize=8, rotation=30, ha="right")
            ax_fair.set_ylabel("Approval Rate", fontsize=10)
            for i, r in enumerate(all_rates):
                ax_fair.text(i, r + 0.005, f"{r:.1%}", ha="center", fontsize=7)

            sex_patch = mpatches.Patch(color=C_BLUE, label="Sex")
            race_patch = mpatches.Patch(color=C_ORANGE, label="Race")
            ax_fair.legend(handles=[sex_patch, race_patch,
                           plt.Line2D([], [], color=C_RED, ls="--", label="80% rule")],
                           fontsize=8)

    ax_fair.set_title(f"Demographic Parity: Approval Rates — {model_name}",
                      fontsize=11, fontweight="bold", color=C_DARK, pad=10)
    ax_fair.set_facecolor(BG)
    ax_fair.spines[["top", "right"]].set_visible(False)

    # --- Panel 5: Proxy analysis ---
    ax_proxy = fig.add_subplot(gs[2, 1])
    if not proxy_df.empty:
        df_sex = proxy_df[proxy_df["sensitive_feature"] == "sex"].nlargest(8, "abs_correlation")
        y_pos = np.arange(len(df_sex))
        colors = [C_RED if v > 0.3 else C_ORANGE if v > 0.1 else C_GRAY
                  for v in df_sex["abs_correlation"]]
        ax_proxy.barh(y_pos, df_sex["abs_correlation"].values, color=colors, edgecolor="white")
        ax_proxy.set_yticks(y_pos)
        ax_proxy.set_yticklabels(df_sex["proxy_candidate"].values, fontsize=9)
        ax_proxy.axvline(0.3, color=C_RED, ls="--", lw=1, alpha=0.7, label="Strong proxy")
        ax_proxy.axvline(0.1, color=C_ORANGE, ls="--", lw=1, alpha=0.7, label="Moderate proxy")
        ax_proxy.legend(fontsize=8)
        for i, v in enumerate(df_sex["abs_correlation"].values):
            ax_proxy.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    ax_proxy.set_title("Proxy Features for Sex\n(features that encode removed sensitive data)",
                       fontsize=11, fontweight="bold", color=C_DARK, pad=10)
    ax_proxy.set_xlabel("Correlation / Cramer's V", fontsize=9)
    ax_proxy.set_facecolor(BG)
    ax_proxy.spines[["top", "right"]].set_visible(False)

    safe_name = model_name.lower().replace(" ", "_")
    path = OUTPUT_DIR / f"dashboard_data_scientist_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 2. Company Director Dashboard
# ============================================================
def dashboard_director(model_name, accent_col, data, model_pred, model_proba,
                       y_test, demo_test):
    """Non-technical dashboard for risk assessment and deployment decision."""
    tcav_df = load_audit_csv("tcav_results.csv")
    xgb_sens_df = load_audit_csv("xgb_concept_sensitivity.csv")
    fairness_df = load_audit_csv("fairness_metrics.csv")

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    fig.suptitle(f"AI Model Assessment — {model_name}\nFor the Board",
                 fontsize=15, fontweight="bold", color=accent_col, y=0.99)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                           left=0.08, right=0.95, top=0.93, bottom=0.04)

    acc = accuracy_score(y_test, model_pred)

    # --- Panel 1: Plain-language accuracy ---
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_acc.axis("off")
    ax_acc.set_title("How Accurate Is This AI?", fontsize=13, fontweight="bold",
                     color=C_DARK, pad=10)

    # Large accuracy number
    ax_acc.text(0.5, 0.7, f"{acc:.0%}", transform=ax_acc.transAxes,
                fontsize=60, fontweight="black", color=accent_col, ha="center")
    ax_acc.text(0.5, 0.48, "of decisions are correct", transform=ax_acc.transAxes,
                fontsize=13, color=C_DARK, ha="center")

    # Plain language explanation
    n_test = len(y_test)
    n_correct = int(acc * n_test)
    n_wrong = n_test - n_correct
    ax_acc.text(0.5, 0.3,
                f"Out of {n_test:,} test cases:\n"
                f"  {n_correct:,} correctly classified\n"
                f"  {n_wrong:,} incorrectly classified",
                transform=ax_acc.transAxes, fontsize=10, color=C_GRAY,
                ha="center", va="top", linespacing=1.6)

    # --- Panel 2: Fairness verdict ---
    ax_fair = fig.add_subplot(gs[0, 1])
    ax_fair.axis("off")
    ax_fair.set_title("Is This AI Fair?", fontsize=13, fontweight="bold",
                      color=C_DARK, pad=10)

    model_key = model_name if model_name == "NN" else "XGBoost"
    all_pass = True
    if not fairness_df.empty:
        y_pos = 0.85
        # --- Approval rate parity (sex & race) ---
        for attr in ["sex", "race"]:
            df_sub = fairness_df[(fairness_df["attribute"] == attr) &
                                 (fairness_df["model"] == model_key)]
            if df_sub.empty:
                continue
            di = df_sub["disparate_impact_ratio"].iloc[0]
            passes = di >= 0.8
            if not passes:
                all_pass = False

            status = "FAIR" if passes else "NEEDS ATTENTION"
            status_col = C_GREEN if passes else C_RED
            bg_col = status_col + "12"
            edge_col = status_col + "55"

            ax_fair.add_patch(mpatches.FancyBboxPatch(
                (0.04, y_pos - 0.16), 0.92, 0.15, boxstyle="round,pad=0.018",
                facecolor=bg_col, edgecolor=edge_col, linewidth=1.5,
                transform=ax_fair.transAxes, clip_on=False))

            if attr == "sex":
                explanation = f"Men and women are treated equally: {di:.0%} fairness ratio"
                if not passes:
                    explanation = f"Women receive different outcomes: {di:.0%} fairness ratio"
            else:
                explanation = f"All racial groups treated equally: {di:.0%} fairness ratio"
                if not passes:
                    explanation = f"Racial groups receive different outcomes: {di:.0%} fairness ratio"

            ax_fair.text(0.08, y_pos - 0.05, explanation,
                         transform=ax_fair.transAxes, fontsize=9, color=C_DARK)
            ax_fair.text(0.08, y_pos - 0.11, status,
                         transform=ax_fair.transAxes, fontsize=9,
                         fontweight="bold", color=status_col)
            y_pos -= 0.20

        # --- Accuracy parity across demographics ---
        df_model = fairness_df[fairness_df["model"] == model_key]
        if not df_model.empty:
            accs = df_model["accuracy"].values
            acc_gap = accs.max() - accs.min()
            acc_passes = acc_gap < 0.05  # less than 5pp gap
            if not acc_passes:
                all_pass = False

            status = "FAIR" if acc_passes else "NEEDS ATTENTION"
            status_col = C_GREEN if acc_passes else C_RED
            bg_col = status_col + "12"
            edge_col = status_col + "55"

            ax_fair.add_patch(mpatches.FancyBboxPatch(
                (0.04, y_pos - 0.16), 0.92, 0.15, boxstyle="round,pad=0.018",
                facecolor=bg_col, edgecolor=edge_col, linewidth=1.5,
                transform=ax_fair.transAxes, clip_on=False))

            explanation = f"Accuracy consistent across groups: {acc_gap:.1%} gap"
            if not acc_passes:
                explanation = f"Accuracy varies across groups: {acc_gap:.1%} gap"

            ax_fair.text(0.08, y_pos - 0.05, explanation,
                         transform=ax_fair.transAxes, fontsize=9, color=C_DARK)
            ax_fair.text(0.08, y_pos - 0.11, status,
                         transform=ax_fair.transAxes, fontsize=9,
                         fontweight="bold", color=status_col)
            y_pos -= 0.20

        # Overall verdict
        verdict_col = C_GREEN if all_pass else C_RED
        verdict_text = "LOW RISK" if all_pass else "HIGH RISK"
        ax_fair.add_patch(mpatches.FancyBboxPatch(
            (0.15, 0.02), 0.7, 0.13, boxstyle="round,pad=0.02",
            facecolor=verdict_col + "18", edgecolor=verdict_col,
            linewidth=2.5, transform=ax_fair.transAxes, clip_on=False))
        ax_fair.text(0.5, 0.085, f"Fairness Risk Level: {verdict_text}",
                     transform=ax_fair.transAxes, fontsize=11,
                     fontweight="bold", color=verdict_col, ha="center")

    # --- Panel 3: Concept influence (the whole point!) ---
    ax_concept = fig.add_subplot(gs[1, :])
    concepts, scores = get_concept_scores("Company Director", model_name,
                                          tcav_df, xgb_sens_df)
    # Re-center around 0: positive = pushes toward approval, negative = against
    centered_scores = np.array([s - 0.5 if not np.isnan(s) else 0.0 for s in scores])
    y = np.arange(len(concepts))
    bar_colors = []
    for s in centered_scores:
        if s > 0.1:
            bar_colors.append(C_GREEN)
        elif s < -0.1:
            bar_colors.append(C_RED)
        else:
            bar_colors.append(accent_col)

    ax_concept.barh(y, centered_scores, color=bar_colors, edgecolor="white", height=0.55)
    ax_concept.axvline(0, color=C_GRAY, ls="--", lw=1.5, alpha=0.7)
    ax_concept.set_yticks(y)
    ax_concept.set_yticklabels(concepts, fontsize=11)
    ax_concept.set_xlim(-0.5, 0.5)
    ax_concept.set_xlabel("Pushes against approval                                "
                          "                                Pushes toward approval",
                          fontsize=9)
    ax_concept.set_title("How Much Do These Concepts Influence AI Decisions?",
                         fontsize=13, fontweight="bold", color=C_DARK, pad=12)
    ax_concept.set_facecolor(BG)
    ax_concept.spines[["top", "right"]].set_visible(False)

    # Annotations with plain language
    for i, (c, s) in enumerate(zip(concepts, centered_scores)):
        if s > 0.1:
            note = "Favours approval"
        elif s < -0.1:
            note = "Works against approval"
        else:
            note = "Modest influence"
        # Place label on the extending end of the bar
        if s >= 0:
            ax_concept.text(s + 0.02, i, note,
                            va="center", fontsize=9, color=C_DARK)
        else:
            ax_concept.text(s - 0.02, i, note,
                            va="center", fontsize=9, color=C_DARK, ha="right")

    # --- Panel 4: Deployment recommendation ---
    ax_deploy = fig.add_subplot(gs[2, :])
    ax_deploy.axis("off")

    # Main recommendation box
    ax_deploy.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.15), 0.9, 0.8, boxstyle="round,pad=0.025",
        facecolor=C_ORANGE + "15", edgecolor=C_ORANGE, linewidth=2.5,
        transform=ax_deploy.transAxes, clip_on=False))
    ax_deploy.text(0.5, 0.85, "RECOMMENDATION: CONDITIONAL DEPLOYMENT",
                   transform=ax_deploy.transAxes, fontsize=14,
                   fontweight="bold", color=C_ORANGE, ha="center")
    ax_deploy.text(0.5, 0.76, "This model can be deployed, but only with the following safeguards in place:",
                   transform=ax_deploy.transAxes, fontsize=11,
                   color=C_DARK, ha="center")

    conditions = [
        "Ongoing fairness monitoring — check demographic parity monthly",
        "Human review of borderline cases (30-70% probability)",
        "Quarterly bias re-assessment with updated data",
        "Clear appeal mechanism for rejected applicants",
        "Investigate and mitigate proxy feature reliance (relationship, marital status)",
    ]
    y_pos = 0.66
    for i, cond in enumerate(conditions):
        ax_deploy.text(0.12, y_pos, f"{i+1}.", transform=ax_deploy.transAxes,
                       fontsize=10, fontweight="bold", color=accent_col)
        ax_deploy.text(0.16, y_pos, cond, transform=ax_deploy.transAxes,
                       fontsize=10, color=C_DARK)
        y_pos -= 0.09

    # Risk note at bottom
    ax_deploy.text(0.5, 0.20,
                   f"Note: This model currently FAILS the 80% fairness rule for both sex and race.",
                   transform=ax_deploy.transAxes, fontsize=10,
                   color=C_RED, ha="center", fontweight="bold")

    safe_name = model_name.lower().replace(" ", "_")
    path = OUTPUT_DIR / f"dashboard_director_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 3. Loan Applicant Dashboard
# ============================================================
def dashboard_applicant(model_name, accent_col, data, model_pred, model_proba,
                        y_test, demo_test, raw_test, xgb_model, nn_model,
                        X_test_xgb):
    """
    Applicant dashboard using LIME-based local concept explanations.
    Shows which concepts helped/hurt their specific application.
    """
    # Pick an approved applicant with moderate capital gains (0 < gain < 10000)
    cap_gains = raw_test["capital-gain"].values
    approved_mask = (model_proba >= 0.70) & (cap_gains > 0) & (cap_gains < 10000)
    if approved_mask.any():
        approved_idxs = np.where(approved_mask)[0]
        # Pick the one closest to 80% probability
        app_idx = approved_idxs[np.argmin(np.abs(model_proba[approved_idxs] - 0.80))]
    else:
        # Fallback: highest proba with any capital gain
        gain_mask = (cap_gains > 0) & (cap_gains < 10000)
        if gain_mask.any():
            app_idx = np.where(gain_mask)[0][np.argmax(model_proba[gain_mask])]
        else:
            app_idx = int(np.argmax(model_proba))

    app_proba = model_proba[app_idx]
    app_pred = model_pred[app_idx]
    pred_label = "APPROVED" if app_pred == 1 else "REJECTED"
    pred_color = C_GREEN if app_pred == 1 else C_RED

    # Determine which concepts apply to this applicant
    app_concepts = STAKEHOLDER_CONCEPTS["Loan Applicant"]
    app_raw = raw_test.iloc[app_idx]

    concept_status = {}
    for name, fn in app_concepts.items():
        try:
            concept_status[name] = bool(fn(raw_test).iloc[app_idx])
        except Exception:
            concept_status[name] = None

    # Get local concept scores via LIME
    print(f"    Running LIME explanation for applicant {app_idx}...")
    lime_scores = explain_applicant_lime(model_name, data, xgb_model, nn_model,
                                         X_test_xgb, app_idx)
    concepts_list = list(STAKEHOLDER_CONCEPTS["Loan Applicant"].keys())
    scores = np.array([lime_scores.get(c, 0.0) for c in concepts_list])

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    fig.suptitle(f"Your Loan Application Result — {model_name}",
                 fontsize=15, fontweight="bold", color=accent_col, y=0.99)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                           left=0.08, right=0.95, top=0.93, bottom=0.04)

    # --- Panel 1: Decision ---
    ax_decision = fig.add_subplot(gs[0, 0])
    ax_decision.axis("off")
    ax_decision.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.15), 0.9, 0.7, boxstyle="round,pad=0.03",
        facecolor=pred_color + "18", edgecolor=pred_color, linewidth=3,
        transform=ax_decision.transAxes, clip_on=False))
    ax_decision.text(0.5, 0.68, f"APPLICATION", transform=ax_decision.transAxes,
                     fontsize=14, color=C_DARK, ha="center")
    ax_decision.text(0.5, 0.52, pred_label, transform=ax_decision.transAxes,
                     fontsize=24, fontweight="black", color=pred_color, ha="center")
    ax_decision.text(0.5, 0.30,
                     "Based on the information you provided,\nthe AI has made this recommendation.",
                     transform=ax_decision.transAxes, fontsize=10,
                     color=C_GRAY, ha="center", linespacing=1.5)

    # --- Panel 2: Gauge ---
    ax_gauge = fig.add_subplot(gs[0, 1])
    draw_gauge(ax_gauge, app_proba, accent_col)

    # --- Panel 3: Concept-based explanation (main panel) ---
    ax_concepts = fig.add_subplot(gs[1, :])
    ax_concepts.axis("off")
    ax_concepts.set_title("Why Did the AI Make This Decision?\n"
                          "Based on concepts that describe your profile",
                          fontsize=13, fontweight="bold", color=C_DARK, pad=15)

    # Column headers
    header_y = 0.93
    ax_concepts.text(0.06, header_y, "Concept", transform=ax_concepts.transAxes,
                     fontsize=10, fontweight="bold", color=C_GRAY)
    ax_concepts.text(0.35, header_y, "Your Profile", transform=ax_concepts.transAxes,
                     fontsize=10, fontweight="bold", color=C_GRAY)
    ax_concepts.text(0.62, header_y, "Impact", transform=ax_concepts.transAxes,
                     fontsize=10, fontweight="bold", color=C_GRAY)
    ax_concepts.text(0.82, header_y, "Strength", transform=ax_concepts.transAxes,
                     fontsize=10, fontweight="bold", color=C_GRAY)
    # Header underline
    ax_concepts.plot([0.04, 0.96], [header_y - 0.03, header_y - 0.03],
                     color=C_GRAY + "55", lw=1, transform=ax_concepts.transAxes)

    # Normalize arrow strengths: map max absolute LIME score to 5 arrows
    max_abs = max(np.abs(scores).max(), 1e-6)

    y_pos = 0.85
    for i, (concept_name, score) in enumerate(zip(concepts_list, scores)):
        applies = concept_status.get(concept_name, None)
        influence = score  # LIME: positive = toward approval, negative = against

        # Determine impact for this applicant
        if applies is True and influence > 0:
            impact = "HELPED"
            impact_col = C_GREEN
        elif applies is True and influence <= 0:
            impact = "HURT"
            impact_col = C_RED
        elif applies is False and influence > 0:
            impact = "MISSED OPPORTUNITY"
            impact_col = C_ORANGE
        elif applies is False and influence <= 0:
            impact = "AVOIDED"
            impact_col = C_GREEN
        else:
            impact = "UNKNOWN"
            impact_col = C_GRAY

        # Strength: 1-5 arrows scaled relative to the strongest concept
        strength = min(5, max(1, round(abs(influence) / max_abs * 5)))
        if influence > 0:
            arrow = "\u25B2"   # up triangle
            arrow_col = C_GREEN
        else:
            arrow = "\u25BC"   # down triangle
            arrow_col = C_RED

        # Concept name
        ax_concepts.text(0.06, y_pos - 0.04, concept_name,
                         transform=ax_concepts.transAxes, fontsize=11,
                         fontweight="bold", color=C_DARK)

        # Applies to you?
        applies_text = "Applies to you" if applies else "Does not apply to you"
        ax_concepts.text(0.35, y_pos - 0.04, applies_text,
                         transform=ax_concepts.transAxes, fontsize=10,
                         color=C_DARK, style="italic")

        # Impact label
        ax_concepts.text(0.62, y_pos - 0.04, impact,
                         transform=ax_concepts.transAxes, fontsize=10,
                         fontweight="bold", color=impact_col)

        # Strength indicator: colored arrows + gray filled dots at fixed positions
        for k in range(5):
            x_pos = 0.82 + k * 0.01
            if k < strength:
                ax_concepts.text(x_pos, y_pos - 0.04, arrow,
                                 transform=ax_concepts.transAxes, fontsize=11,
                                 color=arrow_col, fontweight="bold", ha="center")
            else:
                ax_concepts.text(x_pos, y_pos - 0.04, "\u25CF",
                                 transform=ax_concepts.transAxes, fontsize=11,
                                 color=C_GRAY + "55", ha="center")

        y_pos -= 0.145

    # --- Panel 4: What can you change? ---
    ax_tips = fig.add_subplot(gs[2, 0])
    ax_tips.axis("off")
    ax_tips.set_title("What Can You Change?", fontsize=13,
                      fontweight="bold", color=C_DARK, pad=10)

    # Generate tips based on which concepts don't apply but have high influence
    tips = []
    for concept_name, score in zip(concepts_list, scores):
        applies = concept_status.get(concept_name, None)
        if applies is False and score > 0:
            if "college" in concept_name.lower() or "educated" in concept_name.lower():
                tips.append(("Get a qualification", "A degree or diploma could\nimprove your chances significantly."))
            elif "investment" in concept_name.lower():
                tips.append(("Start investing", "Even small capital gains show\nfinancial stability."))
            elif "white collar" in concept_name.lower():
                tips.append(("Career change", "Professional or managerial roles\ncarry more weight."))
            elif "full-time" in concept_name.lower():
                tips.append(("Work full-time", "Working 38-42 hours per week\nis viewed positively."))
            elif "peak career" in concept_name.lower():
                tips.append(("Build experience", "Established careers with full-time\nhours improve your profile."))

    if not tips:
        tips = [
            ("Education", "Higher education improves\nyour chances."),
            ("Investments", "Capital gains show financial\nstability."),
        ]

    cy = 0.88
    for title, desc in tips[:4]:
        ax_tips.add_patch(mpatches.FancyBboxPatch(
            (0.03, cy - 0.2), 0.94, 0.18, boxstyle="round,pad=0.02",
            linewidth=1.2, edgecolor=C_GREEN + "88", facecolor=C_GREEN + "12",
            transform=ax_tips.transAxes, clip_on=False))
        ax_tips.text(0.08, cy - 0.03, title, transform=ax_tips.transAxes,
                     fontsize=10, fontweight="bold", color=C_DARK, va="top")
        ax_tips.text(0.08, cy - 0.09, desc, transform=ax_tips.transAxes,
                     fontsize=9, color="#444", va="top")
        cy -= 0.24

    # --- Panel 5: Fairness statement ---
    ax_fair = fig.add_subplot(gs[2, 1])
    ax_fair.axis("off")
    ax_fair.set_title("About This Decision", fontsize=13,
                      fontweight="bold", color=C_DARK, pad=10)

    statements = [
        ("What this AI does NOT use:", C_GREEN,
         "This model does not use your gender,\n"
         "race, or nationality as direct inputs."),
        ("What you should know:", C_ORANGE,
         "Some factors (like marital status) may\n"
         "indirectly correlate with protected attributes.\n"
         "We are working to reduce this."),
        ("Your rights:", C_BLUE,
         "You have the right to appeal this decision.\n"
         "Ask for a human review if you believe\n"
         "the result is unfair."),
    ]

    cy = 0.85
    for title, col, text in statements:
        ax_fair.add_patch(mpatches.FancyBboxPatch(
            (0.03, cy - 0.22), 0.94, 0.20, boxstyle="round,pad=0.02",
            facecolor=col + "10", edgecolor=col + "44", linewidth=1.2,
            transform=ax_fair.transAxes, clip_on=False))
        ax_fair.text(0.08, cy - 0.03, title, transform=ax_fair.transAxes,
                     fontsize=10, fontweight="bold", color=col, va="top")
        ax_fair.text(0.08, cy - 0.09, text, transform=ax_fair.transAxes,
                     fontsize=9, color=C_DARK, va="top", linespacing=1.4)
        cy -= 0.28

    safe_name = model_name.lower().replace(" ", "_")
    path = OUTPUT_DIR / f"dashboard_applicant_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
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
    print("Assignment 2 — Stakeholder Dashboards (v2)")
    print("  6 dashboards: 3 stakeholders x 2 models")
    print("=" * 60)

    data = prepare_data(drop_fnlwgt=drop_fnlwgt)
    xgb_model, nn_model, X_test_xgb = load_models(data, drop_fnlwgt=drop_fnlwgt)

    xgb_pred = xgb_model.predict(X_test_xgb)
    xgb_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
    nn_proba = nn_model.predict(data["X_test_scaled"], verbose=0).ravel()
    nn_pred = (nn_proba >= 0.5).astype(int)

    y_test = data["y_test"]
    demo_test = data["demo_test"]
    raw_test = data["raw_test"]

    models = [
        ("XGBoost", C_ORANGE, xgb_pred, xgb_proba),
        ("NN", C_BLUE, nn_pred, nn_proba),
    ]

    for i, (model_name, color, pred, proba) in enumerate(models):
        print(f"\n[{i*3+1}/6] Data Scientist — {model_name}...")
        dashboard_data_scientist(model_name, color, data, pred, proba, y_test, demo_test)

        print(f"[{i*3+2}/6] Director — {model_name}...")
        dashboard_director(model_name, color, data, pred, proba, y_test, demo_test)

        print(f"[{i*3+3}/6] Applicant — {model_name}...")
        dashboard_applicant(model_name, color, data, pred, proba, y_test, demo_test, raw_test,
                            xgb_model, nn_model, X_test_xgb)

    print(f"\nDone! All 6 dashboards saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
