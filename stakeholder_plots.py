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

# Save raw demographic columns for fairness analysis later
sex_all  = X["sex"].values
race_all = X["race"].values

categorical_cols = X.select_dtypes(include=["object", "string"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

feat_names = X.columns.tolist()
print(f"  Total samples: {X.shape[0]}  Features: {len(feat_names)}")

X_train, X_test, y_train, y_test, sex_train, sex_test, race_train, race_test = train_test_split(
    X, y, sex_all, race_all,
    test_size=0.2, random_state=42, stratify=y
)

X_train_num = X_train.astype("float32").values
X_test_num  = X_test.astype("float32").values
y_train_num = y_train.values.astype(int)
y_test_num  = y_test.values.astype(int)

print(f"  Train: {X_train_num.shape}  Test: {X_test.shape}")

# Scaled version for the NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled  = scaler.transform(X_test)

# ======================================================================================================================
# 2. LOAD MODELS
# ======================================================================================================================
print("Loading models ...")

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgb_model.json")

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))
nn_model = tf.keras.models.load_model("models/nn_model.keras")

print("  Models loaded.")

xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

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
# 3. SHAP
# ======================================================================================================================
import shap

print("Computing SHAP for XGBoost ...")
xgb_explainer = shap.TreeExplainer(xgb_model)
np.random.seed(42)
shap_idx = np.random.choice(len(X_test_num), size=500, replace=False)
shap_vals_xgb = xgb_explainer.shap_values(X_test_num[shap_idx])

print("Computing SHAP for NN (KernelExplainer -- may take ~1 min) ...")
background = X_train_scaled[np.random.choice(len(X_train_scaled), 100, replace=False)]
nn_explainer = shap.KernelExplainer(nn_predict_scalar, background)
shap_vals_nn = nn_explainer.shap_values(X_test_scaled[shap_idx], nsamples=100)

mean_shap_xgb = pd.Series(np.abs(shap_vals_xgb).mean(axis=0), index=feat_names)
mean_shap_nn = pd.Series(np.abs(shap_vals_nn).mean(axis=0), index=feat_names)

# Signed mean SHAP for demographic feature columns, split by group
sex_shap_idx = sex_test[shap_idx]
sex_shap_idx_nn = sex_test[shap_idx]
race_shap_idx = race_test[shap_idx]
race_shap_idx_nn = race_test[shap_idx]

sex_feat_cols = [i for i, f in enumerate(feat_names) if "sex" in f.lower()]
race_feat_cols = [i for i, f in enumerate(feat_names) if "race" in f.lower()]


abs_mean_shap_vals_nn = np.abs(shap_vals_nn).mean(axis=0)
abs_nn_shap_n = abs_mean_shap_vals_nn / abs_mean_shap_vals_nn.sum() # fractional influence of global
mean_shap_vals_nn = shap_vals_nn.mean(axis=0)
nn_shap_n = mean_shap_vals_nn / max(min(mean_shap_vals_nn), max(mean_shap_vals_nn), key=abs)

abs_mean_shap_vals_xgb = np.abs(shap_vals_xgb).mean(axis=0)
abs_xgb_shap_n = abs_mean_shap_vals_xgb / abs_mean_shap_vals_xgb.sum() # fractional influence of global
mean_shap_vals_xgb = shap_vals_xgb.mean(axis=0)
xgb_shap_n = mean_shap_vals_xgb / max(min(mean_shap_vals_xgb), max(mean_shap_vals_xgb), key=abs)

xgb_sex_shap_n = {name: xgb_shap_n[idx] for name, idx in zip(sex_shap_idx, sex_feat_cols)}
nn_sex_shap_n = {name: nn_shap_n[idx] for name, idx in zip(sex_shap_idx, sex_feat_cols)}
xgb_race_shap_n = {name: xgb_shap_n[idx] for name, idx in zip(race_shap_idx, race_feat_cols)}
nn_race_shap_n = {name: nn_shap_n[idx] for name, idx in zip(race_shap_idx, race_feat_cols)}

# create pandas series containing normalized features (accross features) with values above 0.01
mean_shap_xgb = pd.Series(abs_xgb_shap_n, index=feat_names).sort_values(ascending=False)
mean_shap_nn = pd.Series(abs_nn_shap_n, index=feat_names).sort_values(ascending=False)
mean_shap_xgb = mean_shap_xgb[mean_shap_xgb > 0.01]
mean_shap_nn = mean_shap_nn[mean_shap_nn > 0.01]
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

rejected_idx       = np.where((xgb_pred == 0) & (y_test_num == 0))[0][0]
x_applicant        = X_test_num[rejected_idx]
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
# 5. FAIRNESS
# ======================================================================================================================
def approval_by_group(pred, group_arr):
    return {g: pred[group_arr == g].mean() for g in np.unique(group_arr)}

xgb_sex_rates  = approval_by_group(xgb_pred, sex_test)
xgb_race_rates = approval_by_group(xgb_pred, race_test)
nn_sex_rates   = approval_by_group(nn_pred,  sex_test)
nn_race_rates  = approval_by_group(nn_pred,  race_test)

def is_sensitive(name):
    return any(k in name.lower() for k in ("sex", "race"))


# ======================================================================================================================
# PLOT 1 - HEAD OF DATA SCIENCE
# ======================================================================================================================
print("Generating plot 1: Head of Data Science ...")
fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle("Model Audit Report -- Head of Data Science",
             fontsize=18, fontweight="bold", color=C_DARK, y=0.98)
outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

patch_sens = mpatches.Patch(color=C_RED, label="Sensitive attribute (sex/race)")

# (A) SHAP -- XGBoost
ax_xgb = fig.add_subplot(outer[0, 0])
c_xgb  = [C_RED if is_sensitive(f) else C_ORANGE for f in mean_shap_xgb.index]
bars_a = ax_xgb.barh(mean_shap_xgb.index[::-1], mean_shap_xgb.values[::-1],
                     color=c_xgb[::-1], edgecolor="white", linewidth=0.5)
ax_xgb.set_title("XGBoost -- Global SHAP Feature Importance\n(L1 norm of SHAP values)",
                 fontsize=10, fontweight="bold", color=C_DARK)
ax_xgb.set_xlabel("L1 of mean |SHAP value|", fontsize=8)
ax_xgb.tick_params(labelsize=7)
ax_xgb.set_facecolor(BG)
for bar, val in zip(bars_a, mean_shap_xgb.values[::-1]):
    ax_xgb.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=6.5)
ax_xgb.legend(handles=[patch_sens, mpatches.Patch(color=C_ORANGE, label="Non-sensitive")],
              fontsize=7, loc="lower right")

# (B) SHAP -- Neural Network
ax_nn = fig.add_subplot(outer[0, 1])
c_nn  = [C_RED if is_sensitive(f) else C_BLUE for f in mean_shap_nn.index]
bars_b = ax_nn.barh(mean_shap_nn.index[::-1], mean_shap_nn.values[::-1],
                    color=c_nn[::-1], edgecolor="white", linewidth=0.5)
ax_nn.set_title("Neural Network -- Global SHAP Feature Importance\n(L1 norm of SHAP values)",
                fontsize=10, fontweight="bold", color=C_DARK)
ax_nn.set_xlabel("L1 of mean |SHAP value|", fontsize=8)
ax_nn.tick_params(labelsize=7)
ax_nn.set_facecolor(BG)
for bar, val in zip(bars_b, mean_shap_nn.values[::-1]):
    ax_nn.text(val * 1.01, bar.get_y() + bar.get_height()/2,
               f"{val:.4f}", va="center", fontsize=6.5)
ax_nn.legend(handles=[patch_sens, mpatches.Patch(color=C_BLUE, label="Non-sensitive")],
             fontsize=7, loc="lower right")

# (C) Fairness audit — approval rate (left axis) + normalised signed SHAP (right axis)
#     Both sex and race groups shown; SHAP normalised independently per model
ax_fair = fig.add_subplot(outer[1, 0])
ax_fair_s = ax_fair.twinx()

groups = list(xgb_sex_rates.keys()) + list(xgb_race_rates.keys())
r_xgb = [xgb_sex_rates[g] for g in xgb_sex_rates] + [xgb_race_rates[g] for g in xgb_race_rates]
r_nn = [nn_sex_rates[g] for g in xgb_sex_rates] + [nn_race_rates[g] for g in xgb_race_rates]
shap_xgb_n = [xgb_sex_shap_n.get(g, 0) for g in xgb_sex_rates] + [xgb_race_shap_n.get(g, 0) for g in xgb_race_rates]
shap_nn_n = [nn_sex_shap_n.get(g, 0) for g in xgb_sex_rates] + [nn_race_shap_n.get(g, 0) for g in xgb_race_rates]

xp = np.arange(len(groups))
w = 0.20

# Approval rate bars
b1 = ax_fair.bar(xp - w * 1.5, r_xgb, w, label="Approval — XGBoost", color=C_ORANGE, edgecolor="white", zorder=3)
b2 = ax_fair.bar(xp - w * 0.5, r_nn, w, label="Approval — Neural Network", color=C_BLUE, edgecolor="white", zorder=3)

# Signed normalised SHAP bars
sc_xgb = [C_GREEN if v >= 0 else C_RED for v in shap_xgb_n]
sc_nn = [C_GREEN if v >= 0 else C_RED for v in shap_nn_n]
b3 = ax_fair_s.bar(xp + w * 0.5, shap_xgb_n, w, color=sc_xgb, edgecolor="white",
                   alpha=0.85, label="SHAP — XGBoost (norm.)", zorder=3)
b4 = ax_fair_s.bar(xp + w * 1.5, shap_nn_n, w, color=sc_nn, edgecolor="white",
                   alpha=0.85, hatch="///", label="SHAP — Neural Network (norm.)", zorder=3)
ax_fair_s.axhline(0, color=C_DARK, lw=0.8, ls="--", zorder=4)
ax_fair_s.set_ylim(-1.4, 1.4)

# Left axis
ax_fair.set_xticks(xp)
ax_fair.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
ax_fair.set_ylabel("Approval rate", fontsize=8)
ax_fair.set_ylim(0, 1.18)
ax_fair.set_facecolor(BG)
ax_fair.spines[["top"]].set_visible(False)

# Right axis
ax_fair_s.set_ylabel("MaxAbs Normalised SHAP", fontsize=7, color=C_GRAY)
ax_fair_s.tick_params(axis="y", labelcolor=C_GRAY, labelsize=7)
ax_fair_s.spines[["top"]].set_visible(False)

ax_fair.set_title("Fairness Audit: Approval Rate & SHAP by Demographic Group",
                  fontsize=10, fontweight="bold", color=C_DARK)

# Approval rate value labels
for bar in list(b1) + list(b2):
    ax_fair.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6)

# SHAP value labels
for bar, val in list(zip(list(b3) + list(b4), shap_xgb_n + shap_nn_n)):
    offset = 0.01 if val >= 0 else -0.01
    ax_fair_s.text(bar.get_x() + bar.get_width() / 2, val + offset,
                   f"{val:+.2f}", ha="center",
                   va="bottom" if val >= 0 else "top", fontsize=6, color=C_DARK)

ax_fair.legend(handles=[
    b1, b2,
    mpatches.Patch(color=C_GREEN, label="SHAP → toward approval"),
    mpatches.Patch(color=C_RED, label="SHAP → toward rejection"),
    mpatches.Patch(facecolor="white", edgecolor=C_DARK, label="XGB"),
    mpatches.Patch(facecolor="white", edgecolor=C_DARK, hatch="///", label="NN"),
], fontsize=6.5, loc="upper right", ncol=2)

# (D) Confusion matrices
cmap_cm = LinearSegmentedColormap.from_list("cm", ["white", C_BLUE])
inner   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 1], wspace=0.4)
for idx_m, (cm, acc, auc, name, col) in enumerate([
        (cm_xgb, xgb_acc, xgb_auc, "XGBoost",       C_ORANGE),
        (cm_nn,  nn_acc,  nn_auc,  "Neural Network", C_BLUE)]):
    ax_m = fig.add_subplot(inner[idx_m])
    ax_m.imshow(cm, interpolation="nearest", cmap=cmap_cm)
    ax_m.set_title(f"{name}\nAcc={acc:.3f}  AUC={auc:.3f}",
                   fontsize=9, fontweight="bold", color=col)
    tl = ["<=50K\n(Reject)", ">50K\n(Approve)"]
    ax_m.set_xticks([0,1]); ax_m.set_yticks([0,1])
    ax_m.set_xticklabels(tl, fontsize=7); ax_m.set_yticklabels(tl, fontsize=7)
    ax_m.set_xlabel("Predicted", fontsize=7); ax_m.set_ylabel("Actual", fontsize=7)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax_m.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                      fontsize=10, fontweight="bold",
                      color="white" if cm[i,j] > thresh else C_DARK)

"""fig.text(0.5, 0.01,
    "Red = sensitive demographic feature in top-10 SHAP  |  Dashed line = 80%-rule fairness threshold",
    ha="center", fontsize=8, color=C_DARK, style="italic")
"""
plt.savefig("plot_data_scientist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved -> plot_data_scientist.png")


# ======================================================================================================================
# PLOT 2 -- COMPANY DIRECTOR
# ======================================================================================================================
FRIENDLY = {
    "capital-gain":   "Investment income",
    "education-num":  "Education level",
    "age":            "Age",
    "hours-per-week": "Hours worked / week",
    "fnlwgt":         "Population weight",
    "capital-loss":   "Investment losses",
    "marital-status": "Marital status",
    "occupation":     "Job type",
    "sex":            "* Gender",
    "race":           "* Race / ethnicity",
    "relationship":   "Household role",
    "workclass":      "Employment type",
    "education":      "Education type",
    "native-country": "Country of origin",
}
def friendly(name):
    for k, v in FRIENDLY.items():
        if k in name:
            return v
    return name.replace("-", " ").replace("_", " ").title()

print("Generating plot 2: Company Director ...")
fig2 = plt.figure(figsize=(16, 10), facecolor="white")
fig2.suptitle("FutureFinance Corp -- AI Loan Model: Executive Trust Dashboard",
              fontsize=16, fontweight="bold", color=C_DARK, y=0.99)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35,
                        left=0.06, right=0.97, top=0.92, bottom=0.06)

# (A) Accuracy
ax_acc = fig2.add_subplot(gs2[0, 0])
brs = ax_acc.bar(["XGBoost", "Neural\nNetwork"],
                 [xgb_acc*100, nn_acc*100],
                 color=[C_ORANGE, C_BLUE], width=0.5, edgecolor="white")
ax_acc.set_ylim(0, 110)
ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
ax_acc.set_title("How Accurate Are the Models?", fontsize=11, fontweight="bold")
ax_acc.set_facecolor("white")
ax_acc.spines[["top","right"]].set_visible(False)
for bar in brs:
    ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center",
                fontsize=13, fontweight="bold", color=bar.get_facecolor())
ax_acc.set_facecolor(BG)

# (B) Top 5 SHAP factors — mirrored butterfly chart
#     LEFT  side: XGBoost's own top-5, normalised, descending
#     RIGHT side: Neural Network's own top-5, normalised, descending
#     Both sides are fully independent — 5 features each, own ranking
ax_feat = fig2.add_subplot(gs2[0, 1:])

# Each model's independent top-5, high→low, then reversed for bottom→top plot
xgb_top5 = mean_shap_xgb.head(5)
nn_top5 = mean_shap_nn.head(5)

xgb_feats = xgb_top5.index.tolist()[::-1]  # least→most important (bottom→top)
nn_feats = nn_top5.index.tolist()[::-1]

xgb_norm = xgb_top5.values[::-1]
nn_norm = nn_top5.values[::-1]

y = np.arange(5)
h = 0.55

max_val = max(xgb_norm.max(), nn_norm.max())

cx = [C_RED if is_sensitive(f) else C_ORANGE for f in xgb_feats]
cn = [C_RED if is_sensitive(f) else C_BLUE for f in nn_feats]

ax_feat.barh(y, [-v/max_val for v in xgb_norm], h, color=cx, edgecolor="white", zorder=1)
ax_feat.barh(y, nn_norm/max_val, h, color=cn, edgecolor="white", zorder=1)

ax_feat.axvline(0, color=C_DARK, lw=1.8, zorder=4)
ax_feat.set_xlim(-1.55, 1.55)
ax_feat.set_ylim(-0.7, 4.7)
ax_feat.axis("off")

# Left labels — XGBoost feature names, placed beyond xlim so no overlap with bars
for i, (feat, val) in enumerate(zip(xgb_feats, xgb_norm)):
    fc = C_RED if is_sensitive(feat) else C_DARK
    # Name: far left, well past the longest bar
    ax_feat.text(-1.12, i, friendly(feat),
                 ha="right", va="center", fontsize=9,
                 color=fc, fontweight="bold" if is_sensitive(feat) else "normal",
                 clip_on=False)
    # Value: centred inside the bar (only if bar is wide enough)
    if val > 0.01:
        ax_feat.text(-val / 2 / max_val, i, f"{val*100:.1f}%",
                     ha="center", va="center", fontsize=7.5,
                     color="white", fontweight="bold", clip_on=False)

# Right labels — Neural Network feature names, placed beyond xlim
for i, (feat, val) in enumerate(zip(nn_feats, nn_norm)):
    fc = C_RED if is_sensitive(feat) else C_DARK
    ax_feat.text(1.12, i, friendly(feat),
                 ha="left", va="center", fontsize=9,
                 color=fc, fontweight="bold" if is_sensitive(feat) else "normal",
                 clip_on=False)
    if val > 0.01:
        ax_feat.text(val / 2 / max_val, i, f"{val*100:.1f}%",
                     ha="center", va="center", fontsize=7.5,
                     color="white", fontweight="bold", clip_on=False)

# Model name banners
ax_feat.text(-0.54, 4.52, "← XGBoost", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=C_ORANGE)
ax_feat.text(0.54, 4.52, "Neural Network →", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=C_BLUE)

ax_feat.set_title("What Does the AI Pay Attention To?\n"
                  "(Each model's own top-5, normalised |SHAP value|)",
                  fontsize=11, fontweight="bold")

ax_feat.legend(handles=[
    mpatches.Patch(color=C_ORANGE, label="XGBoost top-5"),
    mpatches.Patch(color=C_BLUE, label="Neural Network top-5"),
    mpatches.Patch(color=C_RED, label="Sensitive demographic feature"),
], fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), framealpha=1)
ax_feat.set_facecolor(BG)

# (C) Gender fairness — approval rate (left axis) + signed sex SHAP (right axis)
ax_fair2 = fig2.add_subplot(gs2[1, :2])
ax_shap2 = ax_fair2.twinx()

sex_groups = sorted(xgb_sex_rates.keys())
gap_pct = abs(xgb_sex_rates.get("Male", 0) - xgb_sex_rates.get("Female", 0)) * 100
"""
x = np.arange(len(sex_groups))
w = 0.20

# Approval rate bars — XGB and NN
appr_xgb = [xgb_sex_rates[g] * 100 for g in sex_groups]
appr_nn = [nn_sex_rates[g] * 100 for g in sex_groups]
b1 = ax_fair2.bar(x - w * 1.1, appr_xgb, w, color=C_ORANGE, edgecolor="white",
                  label="Approval % — XGBoost", zorder=3)
b2 = ax_fair2.bar(x, appr_nn, w, color=C_BLUE, edgecolor="white",
                  label="Approval % — Neural Network", zorder=3)

# Signed SHAP bars — normalised across features sum
sv_xgb_raw = np.array([xgb_sex_shap_n.get(g, 0) for g in sex_groups])
sv_nn_raw = np.array([nn_sex_shap_n.get(g, 0) for g in sex_groups])

xgb_shap_scale = np.abs(sv_xgb_raw).max() or 1.0
nn_shap_scale = np.abs(sv_nn_raw).max() or 1.0

sv_xgb = sv_xgb_raw / xgb_shap_scale  # normalised, sign preserved
sv_nn = sv_nn_raw / nn_shap_scale

sc_xgb = [C_GREEN if v >= 0 else C_RED for v in sv_xgb]
sc_nn = [C_GREEN if v >= 0 else C_RED for v in sv_nn]
b3 = ax_shap2.bar(x + w * 1.1, sv_xgb, w, color=sc_xgb, edgecolor="white",
                  alpha=0.90, label="Sex SHAP — XGBoost (normalised)", zorder=3)
b4 = ax_shap2.bar(x + w * 2.2, sv_nn, w, color=sc_nn, edgecolor="white",
                  alpha=0.90, hatch="///", label="Sex SHAP — Neural Network (normalised)", zorder=3)
ax_shap2.axhline(0, color=C_DARK, lw=1.0, ls="--", zorder=4)
ax_shap2.set_ylim(-1.3, 1.3)

# Left axis — approval rate
ax_fair2.set_xticks(x + w * 0.55)
ax_fair2.set_xticklabels(sex_groups, fontsize=11, fontweight="bold")
ax_fair2.set_ylabel("Approval rate (%)", fontsize=9)
ax_fair2.set_ylim(0, 115)
ax_fair2.set_facecolor("white")
ax_fair2.spines[["top"]].set_visible(False)

# Right axis — SHAP
ax_shap2.set_ylabel("Normalised sex SHAP", fontsize=8, color=C_GRAY)
ax_shap2.tick_params(axis="y", labelcolor=C_GRAY, labelsize=8)
ax_shap2.spines[["top"]].set_visible(False)

# Value labels — approval rates
for bar in list(b1) + list(b2):
    ax_fair2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                  f"{bar.get_height():.0f}%", ha="center",
                  fontsize=9, fontweight="bold", color=bar.get_facecolor())

# Value labels — show normalised value
raw_vals = list(sv_xgb_raw) + list(sv_nn_raw)
norm_vals = list(sv_xgb) + list(sv_nn)
for bar, norm_val, raw_val in zip(list(b3) + list(b4), norm_vals, raw_vals):
    offset = 0.06 if norm_val >= 0 else -0.06
    ax_shap2.text(bar.get_x() + bar.get_width() / 2,
                  norm_val + offset,
                  f"{norm_val:+.2f}", ha="center",
                  va="bottom" if norm_val >= 0 else "top",
                  fontsize=7.5, color=C_DARK)

verdict = "ACTION REQUIRED" if gap_pct > 5 else "Within Tolerance"
ax_fair2.set_title(
    f"Fairness Check: Approval Rate & Sex Feature SHAP by Gender \n"
    f"Approval gap = {gap_pct:.1f}% - {verdict}",
    fontsize=11, fontweight="bold",
    color=C_RED if gap_pct > 5 else C_GREEN)

ax_fair2.legend(handles=[
    b1, b2,
    mpatches.Patch(color=C_GREEN, label="SHAP → pushes toward approval"),
    mpatches.Patch(color=C_RED, label="SHAP → pushes toward rejection"),
    mpatches.Patch(facecolor="white", edgecolor=C_DARK,
                   label="XGB SHAP"),
    mpatches.Patch(facecolor="white", edgecolor=C_DARK, hatch="///",
                   label="NN SHAP"),
], fontsize=7.5, loc="upper left", ncol=2)
ax_fair2.set_facecolor(BG)
"""
groups = list(xgb_sex_rates.keys()) + list(xgb_race_rates.keys())
r_xgb = [xgb_sex_rates[g] for g in xgb_sex_rates] + [xgb_race_rates[g] for g in xgb_race_rates]
r_nn = [nn_sex_rates[g] for g in xgb_sex_rates] + [nn_race_rates[g] for g in xgb_race_rates]
shap_xgb_n = [xgb_sex_shap_n.get(g, 0) for g in xgb_sex_rates] + [xgb_race_shap_n.get(g, 0) for g in xgb_race_rates]
shap_nn_n = [nn_sex_shap_n.get(g, 0) for g in xgb_sex_rates] + [nn_race_shap_n.get(g, 0) for g in xgb_race_rates]

xp = np.arange(len(groups))
w = 0.20

# Approval rate bars
b1 = ax_fair2.bar(xp - w * 1.5, r_xgb, w, label="Approval — XGBoost", color=C_ORANGE, edgecolor="white", zorder=3)
b2 = ax_fair2.bar(xp - w * 0.5, r_nn, w, label="Approval — Neural Network", color=C_BLUE, edgecolor="white", zorder=3)

# Signed normalised SHAP bars
sc_xgb = [C_GREEN if v >= 0 else C_RED for v in shap_xgb_n]
sc_nn = [C_GREEN if v >= 0 else C_RED for v in shap_nn_n]
b3 = ax_shap2.bar(xp + w * 0.5, shap_xgb_n, w, color=sc_xgb, edgecolor="white",
                   alpha=0.85, label="SHAP — XGBoost (norm.)", zorder=3)
b4 = ax_shap2.bar(xp + w * 1.5, shap_nn_n, w, color=sc_nn, edgecolor="white",
                   alpha=0.85, hatch="///", label="SHAP — Neural Network (norm.)", zorder=3)
ax_fair2.axhline(0, color=C_DARK, lw=0.8, ls="--", zorder=4)
ax_fair2.set_ylim(-1.4, 1.4)

# Left axis
ax_fair2.set_xticks(xp)
ax_fair2.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
ax_fair2.set_ylabel("Approval rate (%)", fontsize=8)
ax_fair2.set_ylim(0, 1.18)
ax_fair2.set_yticklabels([f"{v:.0f}%" for v in ax_fair2.get_yticks()*100])
ax_fair2.set_facecolor(BG)
ax_fair2.spines[["top"]].set_visible(False)

# Right axis
ax_shap2.set_ylabel("MaxAbs Normalised SHAP", fontsize=7, color=C_GRAY)
ax_shap2.tick_params(axis="y", labelcolor=C_GRAY, labelsize=7)
ax_shap2.spines[["top"]].set_visible(False)

ax_fair2.set_title("Fairness Audit: Approval Rate & SHAP by Demographic Group",
                  fontsize=10, fontweight="bold", color=C_DARK)

# Approval rate value labels
for bar in list(b1) + list(b2):
    ax_fair2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height()*100:.0f}", ha="center", va="bottom", fontsize=6)

# SHAP value labels
for bar, val in list(zip(list(b3) + list(b4), shap_xgb_n + shap_nn_n)):
    offset = 0.01 if val >= 0 else -0.01
    ax_shap2.text(bar.get_x() + bar.get_width() / 2, val + offset,
                   f"{val*100:.0f}%", ha="center",
                   va="bottom" if val >= 0 else "top", fontsize=6, color=C_DARK)

ax_fair2.legend(handles=[
    b1, b2,
    mpatches.Patch(color=C_GREEN, label="SHAP → toward approval"),
    mpatches.Patch(color=C_RED, label="SHAP → toward rejection"),
    mpatches.Patch(facecolor="white", edgecolor=C_DARK, label="XGB"),
    mpatches.Patch(facecolor="white", edgecolor=C_DARK, hatch="///", label="NN"),
], fontsize=6.5, loc="upper right", ncol=2)


# (D) Verdict card
ax_card = fig2.add_subplot(gs2[1, 2])
ax_card.axis("off")
rc = C_RED if gap_pct > 5 else C_GREEN
ax_card.add_patch(mpatches.FancyBboxPatch(
    (0.05,0.05), 0.90, 0.90, boxstyle="round,pad=0.04",
    linewidth=2, edgecolor=rc, facecolor=rc+"22",
    transform=ax_card.transAxes, clip_on=False))
ax_card.text(0.5, 0.90, "DEPLOYMENT VERDICT", ha="center",
             transform=ax_card.transAxes, fontsize=10, fontweight="bold", color=C_DARK)
ax_card.text(0.5, 0.74, "MEDIUM RISK" if gap_pct>5 else "LOW RISK",
             ha="center", transform=ax_card.transAxes,
             fontsize=22, color=rc, fontweight="black")
for li, line in enumerate([
    f"XGBoost accuracy:    {xgb_acc*100:.1f}%",
    f"Neural Net accuracy: {nn_acc*100:.1f}%",
    f"XGBoost AUC:         {xgb_auc:.3f}",
    F"Neural Net AUC:        {nn_auc:.3f}",
    "",
    "*Recommendation:*",
    "Mitigate bias before" if gap_pct>5 else "Ready for",
    "full deployment."    if gap_pct>5 else "pilot deployment.",
]):
    ax_card.text(0.5, 0.68 - li*0.075, line, ha="center",
                 transform=ax_card.transAxes, fontsize=8.5, color=C_DARK)
ax_card.set_facecolor(BG)

plt.savefig("plot_director.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  saved -> plot_director.png")


# ======================================================================================================================
# HELPER FUNCTIONS for applicant plots
# ======================================================================================================================
def draw_gauge(ax, proba, accent_col, bg=BG):
    # Full green-to-red arc always drawn as the static track (left=red, right=green)
    theta = np.linspace(np.pi, 0, 300)
    arc = plt.cm.RdYlGn(np.linspace(0, 1, 300))  # red→yellow→green left to right
    for i in range(len(theta) - 1):
        ax.plot(np.cos(theta[i:i + 2]), np.sin(theta[i:i + 2]),
                color=arc[i], lw=16, solid_capstyle="butt")

    # Arrow points to the probability position on the arc
    ang = np.pi - proba * np.pi  # π = left (0%), 0 = right (100%)
    ax.annotate("", xy=(0.62 * np.cos(ang), 0.62 * np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.5,
                                mutation_scale=18))

    # Small circle at needle base
    ax.plot(0, 0, "o", color=C_DARK, markersize=6, zorder=5)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.15)
    ax.axis("off")
    ax.set_facecolor(BG)
    # Percentage label coloured by position on the arc
    pct_color = plt.cm.RdYlGn(proba)
    ax.text(0, -0.22, f"{proba * 100:.0f}%", ha="center", fontsize=26, fontweight="black",
            color=pct_color)
    ax.text(0, -0.35, "Approval probability", ha="center", fontsize=10, color=C_DARK)
    ax.text(-1.12, -0.08, "0%", ha="center", fontsize=8, color=C_GRAY)
    ax.text(1.12, -0.08, "100%", ha="center", fontsize=8, color=C_GRAY)
    ax.set_title("Your Current Approval Probability", fontsize=11,
                 fontweight="bold", color=accent_col)


def _clean_lime_label(raw):
    """Convert a raw LIME condition label into a short readable feature name.
    E.g. 'capital-gain <= 0.000707' -> 'Capital Gain'
         'marital-status_Married-civ-spouse <= 0.00' -> 'Marital: Married'
    """
    import re
    # Strip the condition part (everything after <= >= < > =)
    name = re.split(r"\s*[<>=!]+\s*[-\d.]+", raw)[0].strip()
    # Remove one-hot suffix after the last underscore if it looks like a category
    parts = name.split("_")
    if len(parts) >= 2:
        prefix = parts[0]
        suffix = " ".join(p.capitalize() for p in parts[1:])
        # Shorten known prefixes
        short = {
            "native-country": "Country",
            "marital-status": "Marital",
            "occupation": "Job",
            "workclass": "Work type",
            "relationship": "Relationship",
            "education": "Education",
            "race": "Race",
            "sex": "Gender",
        }
        prefix_clean = short.get(prefix, prefix.replace("-", " ").title())
        label = f"{prefix_clean}: {suffix}"
    else:
        label = name.replace("-", " ").replace("_", " ").title()
    # Hard cap length
    return label[:28] + "…" if len(label) > 28 else label


def draw_lime_waterfall(ax, labels, vals, applicant_row, model_name, pred_label, accent_col, bg=BG):
    # Clean up LIME condition labels to short readable names
    clean_labels = [_clean_lime_label(l) for l in labels]

    display_vals = []

    for raw in labels:
        feat = raw.split("<=")[0].split(">=")[0].split(">")[0].split("<")[0].strip()

        if feat in applicant_row.index:
            v = applicant_row[feat]

            if isinstance(v, (int, float, np.number)):
                display_vals.append(str(int(v)) if float(v).is_integer() else f"{v:.2f}")
            else:
                display_vals.append(str(bool(v)))
        else:
            display_vals.append("?")

    bar_c = [C_RED if v < 0 else C_GREEN for v in vals]
    yp = np.arange(len(clean_labels))
    bw = ax.barh(yp, vals, color=bar_c, edgecolor="white", height=0.65)
    ax.axvline(0, color=C_DARK, lw=1.5)
    ax.set_yticks(yp)
    ax.set_yticklabels(clean_labels, fontsize=9)

    # Dynamic x-axis padding so value labels don't get clipped
    x_range = max(abs(vals.min()), abs(vals.max())) if len(vals) else 0.1
    ax.set_xlim(-x_range * 1.35, x_range * 1.35)

    # Two-line x-axis label so it doesn't crowd the tick numbers
    ax.set_xlabel("← Pushes REJECTED                    Pushes APPROVED →",
                  fontsize=8.5, labelpad=6)

    ax.set_title(f"Why Did the AI Make This Decision? (LIME — {model_name})",
        fontsize = 11, fontweight = "bold", color = accent_col)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    # Value labels: inside bar if wide enough, otherwise just outside
    for bar, val, disp in zip(bw, vals, display_vals):
        bar_w = abs(val)
        s = "+" if val >= 0 else ""
        if bar_w > x_range * 0.18:  # wide enough — put label inside
            ax.text(val / 2, bar.get_y() + bar.get_height() / 2,
                    disp, va="center", ha="center",
                    fontsize=7.5, color="white", fontweight="bold")
        else:  # too narrow — put label outside
            offset = x_range * 0.04
            ax.text(val + (offset if val >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    disp, va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=7.5, color=C_DARK)

    ax.legend(handles=[
        mpatches.Patch(color=C_RED, label="Worked against you"),
        mpatches.Patch(color=C_GREEN, label="Worked in your favour"),
    ], fontsize=9, loc="lower right")
    oc = C_RED if "REJECTED" in pred_label else C_GREEN
    ax.text(0.5, 1.04, pred_label, transform=ax.transAxes,
            ha="center", fontsize=12, fontweight="black", color=oc)


def outcome_label(pred_arr, idx):
    return "APPLICATION REJECTED" if pred_arr[idx] == 0 else "APPROVED"

# ----------------------------------------------------------------------------------------------------------------------
# PLOT 3a -- LOAN APPLICANT: XGBoost
# ----------------------------------------------------------------------------------------------------------------------
print("Generating plot 3a: Loan Applicant (XGBoost) ...")
fig3a = plt.figure(figsize=(14, 10), facecolor=BG)
fig3a.suptitle("Your Loan Application — XGBoost Model",
               fontsize=16, fontweight="bold", color=C_ORANGE, y=0.99)
gs3a = gridspec.GridSpec(2, 2, figure=fig3a, hspace=0.45, wspace=0.35,
                         left=0.06, right=0.97, top=0.92, bottom=0.06)

ax_lime_xgb = fig3a.add_subplot(gs3a[:, 0])
draw_lime_waterfall(ax_lime_xgb,
                    lime_labels_xgb,
                    lime_vals_xgb,
                    X_test.iloc[rejected_idx],
                    "XGBoost",
                    outcome_label(xgb_pred, rejected_idx),
                    C_ORANGE)
ax_lime_xgb.set_facecolor(BG)

ax_act_xgb = fig3a.add_subplot(gs3a[0, 1])
ax_act_xgb.axis("off")
ax_act_xgb.set_title("What Can You Change to Improve\nYour Chances?",
                     fontsize=11, fontweight="bold", color=C_DARK, pad=10)
actions = [
    ("Education level", "Completing higher education\nboosts approval odds significantly."),
    ("Hours worked / week", "Working closer to full-time (40 h)\nis viewed positively."),
    ("Investment income", "Any capital gains strongly\nimprove your score."),
    ("Job type", "Managerial or professional roles\ncarry more weight."),
]
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

plt.savefig("plot_applicant_xgb.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  saved -> plot_applicant_xgb.png")

# ----------------------------------------------------------------------------------------------------------------------
# PLOT 3b -- LOAN APPLICANT: Neural Network
# ----------------------------------------------------------------------------------------------------------------------
print("Generating plot 3b: Loan Applicant (Neural Network) ...")
fig3b = plt.figure(figsize=(14, 10), facecolor=BG)
fig3b.suptitle("Your Loan Application — Neural Network Model",
               fontsize=16, fontweight="bold", color=C_BLUE, y=0.99)
gs3b = gridspec.GridSpec(2, 2, figure=fig3b, hspace=0.45, wspace=0.35,
                         left=0.06, right=0.97, top=0.92, bottom=0.06)

ax_lime_nn = fig3b.add_subplot(gs3b[:, 0])
draw_lime_waterfall(ax_lime_nn,
                    lime_labels_nn,
                    lime_vals_nn,
                    X_test.iloc[rejected_idx],
                    "Neural Network",
                    outcome_label(nn_pred, rejected_idx),
                    C_BLUE)

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

plt.savefig("plot_applicant_nn.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  saved -> plot_applicant_nn.png")

print("\nDone! Four plots saved:")
print("  plot_data_scientist.png")
print("  plot_director.png")
print("  plot_applicant_xgb.png")
print("  plot_applicant_nn.png")
