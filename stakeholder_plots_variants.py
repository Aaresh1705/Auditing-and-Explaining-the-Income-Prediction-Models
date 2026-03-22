import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

C_BLUE   = "#3A86FF"
C_ORANGE = "#FF6B35"
C_GREEN  = "#06D6A0"
C_RED    = "#EF476F"
C_PURPLE = "#845EC2"
C_GRAY   = "#8D99AE"
C_DARK   = "#1A1A2E"
BG       = "#FAFAFA"

VARIANT = "no_sensitive_no_country"

_DROPS = {
    "no_sensitive":            ["sex", "race"],
    "no_sensitive_no_country": ["sex", "race", "native-country"],
}
assert VARIANT in _DROPS, f"Unknown VARIANT '{VARIANT}'. Choose from: {list(_DROPS)}"
_dropped_prefixes = _DROPS[VARIANT]

# ======================================================================================================================
# 1. LOAD DATA
# ======================================================================================================================
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Fetching dataset from UCI repo ...")
adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy().iloc[:, 0]

X = X.replace("?", pd.NA)
y = y.astype(str).str.strip()

valid_target = y.isin(["<=50K", ">50K", "<=50K.", ">50K."])
X = X.loc[valid_target]
y = y.loc[valid_target]

valid_X = X.notna().all(axis=1)
X = X.loc[valid_X]
y = y.loc[valid_X]

y = y.map({
    "<=50K": 0,
    ">50K": 1,
    "<=50K.": 0,
    ">50K.": 1
})

sex_all  = X["sex"].values
race_all = X["race"].values

categorical_cols = X.select_dtypes(include=["object", "string"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Drop columns for the selected variant
_to_drop = [c for c in X.columns if any(c == p or c.startswith(p + "_") for p in _dropped_prefixes)]
X = X.drop(columns=_to_drop)

feat_names = X.columns.tolist()
print(f"  Variant: {VARIANT}  |  Total samples: {X.shape[0]}  Features: {len(feat_names)}")

X_train, X_test, y_train, y_test, sex_train, sex_test, race_train, race_test = train_test_split(
    X, y, sex_all, race_all,
    test_size=0.2, random_state=42, stratify=y
)

X_train_num = X_train.astype("float32").values
X_test_num  = X_test.astype("float32").values
y_train_num = y_train.values.astype(int)
y_test_num  = y_test.values.astype(int)

print(f"  Train: {X_train_num.shape}  Test: {X_test_num.shape}")

# Scaled version for the NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled  = scaler.transform(X_test_num)

# ======================================================================================================================
# 2. LOAD MODELS
# ======================================================================================================================
print("Loading models ...")

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"models/xgb_{VARIANT}.json")

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))
nn_model = tf.keras.models.load_model(f"models/nn_{VARIANT}.keras")

print("  Models loaded.")

xgb_pred  = xgb_model.predict(X_test_num)
xgb_proba = xgb_model.predict_proba(X_test_num)[:, 1]

def nn_predict_scalar(x):
    return nn_model.predict(x, verbose=0).ravel()

def nn_predict_proba(x):
    p1 = nn_predict_scalar(x)
    return np.column_stack([1 - p1, p1])

nn_proba = nn_predict_scalar(X_test_scaled)
nn_pred  = (nn_proba >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
xgb_acc = accuracy_score(y_test, xgb_pred)
nn_acc  = accuracy_score(y_test, nn_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)
nn_auc  = roc_auc_score(y_test, nn_proba)
cm_xgb  = confusion_matrix(y_test, xgb_pred)
cm_nn   = confusion_matrix(y_test, nn_pred)

print(f"  XGB  acc={xgb_acc:.3f}  auc={xgb_auc:.3f}")
print(f"  NN   acc={nn_acc:.3f}   auc={nn_auc:.3f}")

# ======================================================================================================================
# 4. LIME
# ======================================================================================================================
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    training_data=X_train_num,
    feature_names=feat_names,
    class_names=["<=50K", ">50K"],
    mode="classification"
)

# Pick an applicant whose XGBoost probability sits in the 40-60% uncertainty band
uncertain_mask = (xgb_proba >= 0.30) & (xgb_proba <= 0.40)
if uncertain_mask.any():
    # Among uncertain cases, pick the one closest to 50%
    uncertain_idxs = np.where(uncertain_mask)[0]
    rejected_idx = uncertain_idxs[np.argmin(np.abs(xgb_proba[uncertain_idxs] - 0.50))]
else:
    # Fallback: closest to 50% overall
    rejected_idx = int(np.argmin(np.abs(xgb_proba - 0.50)))
print(f"  Applicant idx={rejected_idx}  XGB proba={xgb_proba[rejected_idx]:.3f}  NN proba={nn_proba[rejected_idx]:.3f}")
x_applicant         = X_test_num[rejected_idx]
applicant_proba_xgb = xgb_proba[rejected_idx]
applicant_proba_nn  = nn_proba[rejected_idx]


def sort_lime(exp):
    lst    = exp.as_list()
    labels = [l for l, _ in lst]
    vals   = np.array([v for _, v in lst])
    order  = np.argsort(vals)
    return [labels[i] for i in order], vals[order]


print("Computing LIME for applicant XGBoost")
lime_exp_xgb = lime_explainer.explain_instance(
    data_row=x_applicant,
    predict_fn=xgb_model.predict_proba,
    num_features=10
)
lime_labels_xgb, lime_vals_xgb = sort_lime(lime_exp_xgb)

print("Computing LIME for applicant Neural Network")


def nn_predict_proba_lime(x):
    return nn_predict_proba(scaler.transform(x))


lime_exp_nn = lime_explainer.explain_instance(
    data_row=x_applicant,
    predict_fn=nn_predict_proba_lime,
    num_features=10
)
lime_labels_nn, lime_vals_nn = sort_lime(lime_exp_nn)



# ======================================================================================================================
# HELPER FUNCTIONS for applicant plots
# ======================================================================================================================
def draw_gauge(ax, proba, accent_col, bg=BG):
    theta = np.linspace(np.pi, 0, 300)
    arc = plt.cm.RdYlGn(np.linspace(0, 1, 300))
    for i in range(len(theta) - 1):
        ax.plot(np.cos(theta[i:i + 2]), np.sin(theta[i:i + 2]),
                color=arc[i], lw=16, solid_capstyle="butt")
    ang = np.pi - proba * np.pi
    ax.annotate("", xy=(0.62 * np.cos(ang), 0.62 * np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.5, mutation_scale=18))
    ax.plot(0, 0, "o", color=C_DARK, markersize=6, zorder=5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.15)
    ax.axis("off")
    ax.set_facecolor(BG)
    pct_color = plt.cm.RdYlGn(proba)
    ax.text(0, -0.22, f"{proba * 100:.0f}%", ha="center", fontsize=26, fontweight="black", color=pct_color)
    ax.text(0, -0.35, "Approval probability", ha="center", fontsize=10, color=C_DARK)
    ax.text(-1.12, -0.08, "0%",   ha="center", fontsize=8, color=C_GRAY)
    ax.text( 1.12, -0.08, "100%", ha="center", fontsize=8, color=C_GRAY)
    ax.set_title("Your Current Approval Probability", fontsize=11, fontweight="bold", color=accent_col)


def _clean_lime_label(raw):
    import re
    # Strip leading bound e.g. '0.00 < ' before extracting the feature name
    cleaned = re.sub(r"^[-\d.]+\s*[<>]=?\s*", "", raw).strip()
    name = re.split(r"\s*[<>=!]+\s*[-\d.]+", cleaned)[0].strip()
    parts = name.split("_")
    if len(parts) >= 2:
        prefix = parts[0]
        suffix = " ".join(p.capitalize() for p in parts[1:])
        short = {
            "native-country": "Country",
            "marital-status": "Marital",
            "occupation":     "Job",
            "workclass":      "Work type",
            "relationship":   "Relationship",
            "education":      "Education",
            "race":           "Race",
            "sex":            "Gender",
        }
        prefix_clean = short.get(prefix, prefix.replace("-", " ").title())
        label = f"{prefix_clean}: {suffix}"
    else:
        label = name.replace("-", " ").replace("_", " ").title()
    return label[:28] + "…" if len(label) > 28 else label


def draw_lime_waterfall(ax, labels, vals, applicant_row, model_name, pred_label, accent_col, bg=BG):
    clean_labels = [_clean_lime_label(l) for l in labels]
    display_vals = []
    for raw in labels:
        # LIME conditions can be  'feat <= 1.0'  or  '0.0 < feat <= 1.0'
        # Strip all numeric bounds and comparison operators to isolate the feature name
        import re
        feat = re.sub(r"[-\d.]+\s*[<>=]+\s*", "", raw).strip()
        feat = re.sub(r"\s*[<>=]+.*", "", feat).strip()
        if feat in applicant_row.index:
            v = applicant_row[feat]
            if isinstance(v, (int, float, np.number)) and float(v) in (0.0, 1.0) and "_" in feat:
                # Binary OHE column (has underscore) - Yes/No is more readable than 1/0
                display_vals.append("Yes" if float(v) == 1.0 else "No")
            elif isinstance(v, (int, float, np.number)):
                display_vals.append(str(int(v)) if float(v).is_integer() else f"{v:.2f}")
            else:
                display_vals.append("Yes" if v else "No")
        else:
            display_vals.append("?")

    bar_c = [C_RED if v < 0 else C_GREEN for v in vals]
    yp    = np.arange(len(clean_labels))
    bw    = ax.barh(yp, vals, color=bar_c, edgecolor="white", height=0.65)
    ax.axvline(0, color=C_DARK, lw=1.5)
    ax.set_yticks(yp)
    ax.set_yticklabels(clean_labels, fontsize=9)
    x_range = max(abs(vals.min()), abs(vals.max())) if len(vals) else 0.1
    ax.set_xlim(-x_range * 1.35, x_range * 1.35)
    ax.set_xlabel("← Pushes REJECTED                    Pushes APPROVED →",
                  fontsize=8.5, labelpad=6)
    ax.set_title(f"Why Did the AI Make This Decision? (LIME — {model_name})",
                 fontsize=11, fontweight="bold", color=accent_col)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    for bar, val, disp in zip(bw, vals, display_vals):
        bar_w = abs(val)
        if bar_w > x_range * 0.18:
            ax.text(val / 2, bar.get_y() + bar.get_height() / 2,
                    disp, va="center", ha="center", fontsize=7.5, color="white", fontweight="bold")
        else:
            offset = x_range * 0.04
            ax.text(val + (offset if val >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    disp, va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=7.5, color=C_DARK)
    ax.legend(handles=[
        mpatches.Patch(color=C_RED,   label="Worked against you"),
        mpatches.Patch(color=C_GREEN, label="Worked in your favour"),
    ], fontsize=9, loc="lower right")
    oc = C_RED if "REJECTED" in pred_label else C_GREEN
    ax.text(0.5, 1.04, pred_label, transform=ax.transAxes,
            ha="center", fontsize=12, fontweight="black", color=oc)


def outcome_label(pred_arr, idx):
    return "APPLICATION REJECTED" if pred_arr[idx] == 0 else "APPROVED"


actions = [
    ("Education level",    "Completing higher education\nboosts approval odds significantly."),
    ("Hours worked / week","Working closer to full-time (40 h)\nis viewed positively."),
    ("Investment income",  "Any capital gains strongly\nimprove your score."),
    ("Job type",           "Managerial or professional roles\ncarry more weight."),
]

# ----------------------------------------------------------------------------------------------------------------------
# PLOT 3a — LOAN APPLICANT: XGBoost
# ----------------------------------------------------------------------------------------------------------------------
print("Generating plot 3a: Loan Applicant (XGBoost) ...")
fig3a = plt.figure(figsize=(14, 10), facecolor=BG)
fig3a.suptitle(f"Your Loan Application — XGBoost Model  [{VARIANT}]",
               fontsize=16, fontweight="bold", color=C_ORANGE, y=0.99)
gs3a = gridspec.GridSpec(2, 2, figure=fig3a, hspace=0.45, wspace=0.35,
                         left=0.06, right=0.97, top=0.92, bottom=0.06)

ax_lime_xgb = fig3a.add_subplot(gs3a[:, 0])
draw_lime_waterfall(ax_lime_xgb, lime_labels_xgb, lime_vals_xgb,
                    X_test.iloc[rejected_idx], "XGBoost",
                    outcome_label(xgb_pred, rejected_idx), C_ORANGE)
ax_lime_xgb.set_facecolor(BG)

ax_act_xgb = fig3a.add_subplot(gs3a[0, 1])
ax_act_xgb.axis("off")
ax_act_xgb.set_title("What Can You Change to Improve\nYour Chances?",
                     fontsize=11, fontweight="bold", color=C_DARK, pad=10)
cy = 0.92
for title, desc in actions:
    ax_act_xgb.add_patch(mpatches.FancyBboxPatch(
        (0.02, cy - 0.17), 0.96, 0.16, boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor=C_GREEN + "88", facecolor=C_GREEN + "15",
        transform=ax_act_xgb.transAxes, clip_on=False))
    ax_act_xgb.text(0.07, cy - 0.02, title, transform=ax_act_xgb.transAxes,
                    fontsize=9, fontweight="bold", color=C_DARK, va="top")
    ax_act_xgb.text(0.07, cy - 0.08, desc, transform=ax_act_xgb.transAxes,
                    fontsize=7.5, color="#444", va="top")
    cy -= 0.20

ax_gauge_xgb = fig3a.add_subplot(gs3a[1, 1])
draw_gauge(ax_gauge_xgb, applicant_proba_xgb, C_ORANGE)
ax_gauge_xgb.set_facecolor(BG)
fig3a.text(0.5, 0.01,
           "Explanation: LIME on XGBoost. Factors outside your control are shown for transparency only.",
           ha="center", fontsize=7.5, color=C_GRAY, style="italic")

plt.savefig(f"plot_applicant_xgb_{VARIANT}.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  saved -> plot_applicant_xgb_{VARIANT}.png")

# ----------------------------------------------------------------------------------------------------------------------
# PLOT 3b — LOAN APPLICANT: Neural Network
# ----------------------------------------------------------------------------------------------------------------------
print("Generating plot 3b: Loan Applicant (Neural Network) ...")
fig3b = plt.figure(figsize=(14, 10), facecolor=BG)
fig3b.suptitle(f"Your Loan Application — Neural Network Model  [{VARIANT}]",
               fontsize=16, fontweight="bold", color=C_BLUE, y=0.99)
gs3b = gridspec.GridSpec(2, 2, figure=fig3b, hspace=0.45, wspace=0.35,
                         left=0.06, right=0.97, top=0.92, bottom=0.06)

ax_lime_nn = fig3b.add_subplot(gs3b[:, 0])
draw_lime_waterfall(ax_lime_nn, lime_labels_nn, lime_vals_nn,
                    X_test.iloc[rejected_idx], "Neural Network",
                    outcome_label(nn_pred, rejected_idx), C_BLUE)

ax_act_nn = fig3b.add_subplot(gs3b[0, 1])
ax_act_nn.axis("off")
ax_act_nn.set_title("What Can You Change to Improve\nYour Chances?",
                    fontsize=11, fontweight="bold", color=C_DARK, pad=10)
cy = 0.92
for title, desc in actions:
    ax_act_nn.add_patch(mpatches.FancyBboxPatch(
        (0.02, cy - 0.17), 0.96, 0.16, boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor=C_GREEN + "88", facecolor=C_GREEN + "15",
        transform=ax_act_nn.transAxes, clip_on=False))
    ax_act_nn.text(0.07, cy - 0.02, title, transform=ax_act_nn.transAxes,
                   fontsize=9, fontweight="bold", color=C_DARK, va="top")
    ax_act_nn.text(0.07, cy - 0.08, desc, transform=ax_act_nn.transAxes,
                   fontsize=7.5, color="#444", va="top")
    cy -= 0.20

ax_gauge_nn = fig3b.add_subplot(gs3b[1, 1])
draw_gauge(ax_gauge_nn, applicant_proba_nn, C_BLUE)
ax_gauge_nn.set_facecolor(BG)
fig3b.text(0.5, 0.01,
           "Explanation: LIME on Neural Network. Factors outside your control are shown for transparency only.",
           ha="center", fontsize=7.5, color=C_GRAY, style="italic")

plt.savefig(f"plot_applicant_nn_{VARIANT}.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  saved -> plot_applicant_nn_{VARIANT}.png")

print(f"\nDone! Two plots saved for variant [{VARIANT}]:")
print(f"  plot_applicant_xgb_{VARIANT}.png")
print(f"  plot_applicant_nn_{VARIANT}.png")