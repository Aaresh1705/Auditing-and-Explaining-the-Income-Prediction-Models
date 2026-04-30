from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ADULT_DIR = BASE_DIR / "adult"
MODEL_PATH = BASE_DIR / "models" / "xgb_no_sensitive_no_country.json"
OUTPUT_DIR = BASE_DIR / "assignment2_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Dataset setup
# ============================================================

header = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

numeric_cols = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

categorical_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_raw_adult(split="test"):
    """
    Loads the raw UCI Adult dataset without integer-encoding categories.
    This is important because the saved XGBoost model expects one-hot
    encoded columns with category names, e.g. workclass_Private.
    """

    if split == "train":
        path = ADULT_DIR / "adult.data"
        df = pd.read_csv(path, names=header)

    elif split == "test":
        path = ADULT_DIR / "adult.test"
        df = pd.read_csv(path, names=header, skiprows=1)

    else:
        raise ValueError("split must be 'train' or 'test'")

    # Remove whitespace around text values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # Remove rows with missing categorical values
    df = df.replace("?", np.nan).dropna()

    # Test set labels have dots, e.g. <=50K. and >50K.
    target_map = {
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,
        ">50K.": 1,
    }

    df["income"] = df["income"].map(target_map)

    if df["income"].isna().any():
        raise ValueError("Unexpected income value found.")

    return df


def load_xgb_data(split="test", variant="no_sensitive_no_country"):
    """
    Creates the input format expected by the saved XGBoost model.

    Variants:
    - full: keeps all features
    - no_sensitive: removes sex and race
    - no_sensitive_no_country: removes sex, race, and native-country
    """

    df = load_raw_adult(split)
    y = df["income"].astype(int)

    feature_cols = numeric_cols + categorical_cols

    if variant == "no_sensitive":
        feature_cols = [
            c for c in feature_cols
            if c not in ["sex", "race"]
        ]

    elif variant == "no_sensitive_no_country":
        feature_cols = [
            c for c in feature_cols
            if c not in ["sex", "race", "native-country"]
        ]

    elif variant == "full":
        pass

    else:
        raise ValueError("Unknown variant.")

    X_raw = df[feature_cols].copy()

    cat_cols_used = [
        c for c in categorical_cols
        if c in feature_cols
    ]

    X = pd.get_dummies(
        X_raw,
        columns=cat_cols_used,
        drop_first=False
    )

    return X, y, df


# ============================================================
# Plotting helpers
# ============================================================

def save_scatter_numeric(z, values, title, colorbar_label, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        z[:, 0],
        z[:, 1],
        c=values,
        s=8,
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label=colorbar_label)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot: {output_path}")


def save_scatter_category(z, values, title, filename, category_order=None):
    values = pd.Series(values)

    if category_order is not None:
        values = pd.Categorical(
            values,
            categories=category_order,
            ordered=True
        )
        codes = values.codes
        uniques = category_order
    else:
        codes, uniques = pd.factorize(values)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        z[:, 0],
        z[:, 1],
        c=codes,
        s=8,
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(len(uniques)))
    cbar.set_ticklabels(uniques)

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot: {output_path}")


# ============================================================
# Concept probing
# ============================================================

probe_results = []


def concept_probe(embeddings, concept_values, concept_name):
    """
    Trains a simple logistic-regression probe to predict a demographic
    concept from the XGBoost leaf embeddings.

    If probe accuracy is much higher than the majority-class baseline,
    the model representation contains information about that concept.
    """

    mask = pd.Series(concept_values).notna().values

    X_probe = embeddings[mask]
    y_raw = pd.Series(concept_values).values[mask]

    le = LabelEncoder()
    y_probe = le.fit_transform(y_raw)

    class_counts = pd.Series(y_probe).value_counts()
    min_class_count = class_counts.min()
    cv_folds = min(5, min_class_count)

    if cv_folds < 2:
        print(f"\n{concept_name}: Not enough samples for cross-validation.")
        return

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )

    probe = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )

    baseline = DummyClassifier(strategy="most_frequent")

    probe_scores = cross_val_score(
        probe,
        X_probe,
        y_probe,
        cv=cv,
        scoring="accuracy"
    )

    baseline_scores = cross_val_score(
        baseline,
        X_probe,
        y_probe,
        cv=cv,
        scoring="accuracy"
    )

    result = {
        "concept": concept_name,
        "classes": ", ".join(map(str, le.classes_)),
        "probe_accuracy_mean": probe_scores.mean(),
        "probe_accuracy_std": probe_scores.std(),
        "baseline_accuracy_mean": baseline_scores.mean(),
        "baseline_accuracy_std": baseline_scores.std(),
        "improvement_over_baseline": probe_scores.mean() - baseline_scores.mean(),
    }

    probe_results.append(result)

    print(f"\nConcept probe: {concept_name}")
    print(f"Classes: {list(le.classes_)}")
    print(f"Probe accuracy: {probe_scores.mean():.3f} ± {probe_scores.std():.3f}")
    print(f"Baseline accuracy: {baseline_scores.mean():.3f} ± {baseline_scores.std():.3f}")
    print(f"Improvement over baseline: {probe_scores.mean() - baseline_scores.mean():.3f}")


# ============================================================
# Main audit
# ============================================================

def main():
    print("\n=== Assignment 2 XGBoost Audit ===")

    # ------------------------------------------------------------
    # 1. Load test data
    # ------------------------------------------------------------

    X_test, y_test, original_test_df = load_xgb_data(
        split="test",
        variant="no_sensitive_no_country"
    )

    print("\nX_test shape before alignment:", X_test.shape)

    # ------------------------------------------------------------
    # 2. Load saved XGBoost model
    # ------------------------------------------------------------

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print(f"Loaded model from: {MODEL_PATH}")

    # ------------------------------------------------------------
    # 3. Align columns with model features
    # ------------------------------------------------------------

    expected_features = model.get_booster().feature_names
    X_test = X_test.reindex(columns=expected_features, fill_value=0)

    print("X_test shape after alignment:", X_test.shape)
    print("Expected feature count:", len(expected_features))

    # ------------------------------------------------------------
    # 4. Evaluate model
    # ------------------------------------------------------------

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nFirst predictions:")
    print(y_pred[:10])

    print("\nFirst probabilities:")
    print(y_prob[:10])

    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    metrics_df = pd.DataFrame([
        {
            "model": "XGBoost no_sensitive_no_country",
            "accuracy": acc,
            "roc_auc": auc,
            "n_test_samples": len(y_test),
            "n_features": X_test.shape[1],
        }
    ])

    metrics_path = OUTPUT_DIR / "xgb_audit_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # ------------------------------------------------------------
    # 5. Leaf embeddings
    # ------------------------------------------------------------

    leaf_embeddings = model.apply(X_test)

    print("\nLeaf embedding shape:")
    print(leaf_embeddings.shape)

    # Save embeddings in case you need them later
    embeddings_path = OUTPUT_DIR / "xgb_leaf_embeddings.npy"
    np.save(embeddings_path, leaf_embeddings)
    print(f"Saved leaf embeddings: {embeddings_path}")

    # ------------------------------------------------------------
    # 6. Latent-space visualization with t-SNE
    # ------------------------------------------------------------

    print("\nRunning t-SNE. This may take a little while...")

    sample_size = min(800, len(X_test))

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=sample_size, replace=False)

    leaf_sample = leaf_embeddings[idx]
    y_sample = y_test.iloc[idx]
    df_sample = original_test_df.iloc[idx].copy()

    # Create age groups
    age_order = ["<=25", "26-40", "41-60", "60+"]

    df_sample["age_group"] = pd.cut(
        df_sample["age"],
        bins=[0, 25, 40, 60, 100],
        labels=age_order
    )

    df_sample["age_group"] = pd.Categorical(
        df_sample["age_group"],
        categories=age_order,
        ordered=True
    )

    # Scale and reduce before t-SNE for speed
    leaf_scaled = StandardScaler().fit_transform(leaf_sample)
    leaf_pca = PCA(n_components=30, random_state=42).fit_transform(leaf_scaled)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=20,
        learning_rate="auto",
        init="pca",
        max_iter=750,
        method="barnes_hut"
    )

    z = tsne.fit_transform(leaf_pca)

    # Save t-SNE coordinates
    tsne_df = pd.DataFrame({
        "tsne_1": z[:, 0],
        "tsne_2": z[:, 1],
        "income": y_sample.values,
        "sex": df_sample["sex"].values,
        "race": df_sample["race"].values,
        "age": df_sample["age"].values,
        "age_group": df_sample["age_group"].astype(str).values,
    })

    tsne_path = OUTPUT_DIR / "xgb_tsne_coordinates.csv"
    tsne_df.to_csv(tsne_path, index=False)
    print(f"Saved t-SNE coordinates: {tsne_path}")

    save_scatter_numeric(
        z=z,
        values=y_sample,
        title="t-SNE of XGBoost leaf embeddings colored by income/approval",
        colorbar_label="Income >50K / Approved",
        filename="xgb_tsne_income.png"
    )

    save_scatter_category(
        z=z,
        values=df_sample["sex"],
        title="t-SNE of XGBoost leaf embeddings colored by sex",
        filename="xgb_tsne_sex.png"
    )

    save_scatter_category(
        z=z,
        values=df_sample["race"],
        title="t-SNE of XGBoost leaf embeddings colored by race",
        filename="xgb_tsne_race.png"
    )

    save_scatter_category(
    z=z,
    values=df_sample["age_group"],
    title="t-SNE of XGBoost leaf embeddings colored by age group",
    filename="xgb_tsne_age_group.png",
    category_order=age_order
    )

    # ------------------------------------------------------------
    # 7. Concept probing
    # ------------------------------------------------------------

    print("\nRunning concept probes...")

    original_test_df["age_group"] = pd.cut(
        original_test_df["age"],
        bins=[0, 25, 40, 60, 100],
        labels=["<=25", "26-40", "41-60", "60+"]
    )

    concept_probe(leaf_embeddings, original_test_df["sex"], "sex")
    concept_probe(leaf_embeddings, original_test_df["race"], "race")
    concept_probe(leaf_embeddings, original_test_df["age_group"], "age group")

    probe_results_df = pd.DataFrame(probe_results)

    print("\nConcept probe summary:")
    print(probe_results_df)

    probe_results_path = OUTPUT_DIR / "xgb_concept_probe_results.csv"
    probe_results_df.to_csv(probe_results_path, index=False)

    print(f"\nSaved concept probe results: {probe_results_path}")

    print("\n=== Audit complete ===")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()