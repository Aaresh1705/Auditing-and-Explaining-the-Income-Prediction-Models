"""
Shared definitions for Assignment 2 — concepts, data loading, constants.
=========================================================================
This module is imported by train_models.py, technical_audit.py, and
stakeholder_dashboards.py.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ADULT_DIR = BASE_DIR / "adult"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Colors (consistent with assignment 1 dashboards)
# ============================================================
C_BLUE = "#3A86FF"
C_ORANGE = "#FF6B35"
C_GREEN = "#06D6A0"
C_RED = "#EF476F"
C_PURPLE = "#845EC2"
C_GRAY = "#8D99AE"
C_DARK = "#1A1A2E"
BG = "#FAFAFA"

# ============================================================
# Dataset schema
# ============================================================
HEADER = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

SENSITIVE_PREFIXES = ["sex", "race", "native-country"]


# ============================================================
# Stakeholder concepts (18 total: 6 per stakeholder)
# ============================================================
# Each concept is a callable: df_raw -> boolean Series
# df_raw has the original column names BEFORE one-hot encoding.

STAKEHOLDER_CONCEPTS = {
    "Head of Data Science": {
        "Female": lambda df: df["sex"] == "Female",
        "Racial minority": lambda df: df["race"] != "White",
        "Part-time worker": lambda df: (df["hours-per-week"] < 35) & (~df["workclass"].isin(["Never-worked", "Without-pay"])),
        "Has investment activity": lambda df: (df["capital-gain"] > 0) | (df["capital-loss"] > 0),
        "Unpaid / never worked": lambda df: df["workclass"].isin(["Without-pay", "Never-worked"]),
        "Caregiver proxy": lambda df: df["relationship"].isin(["Wife", "Own-child"]),
    },
    "Company Director": {
        "Female": lambda df: df["sex"] == "Female",
        "Immigrant": lambda df: df["native-country"] != "United-States",
        "College educated": lambda df: df["education"].isin(["Bachelors", "Masters", "Doctorate", "Prof-school", "Assoc-acdm", "Assoc-voc"]),
        "Married": lambda df: df["marital-status"] == "Married-civ-spouse",
        "White collar": lambda df: df["occupation"].isin(["Exec-managerial", "Prof-specialty", "Tech-support", "Adm-clerical"]),
        "Near retirement": lambda df: df["age"] >= 58,
    },
    "Loan Applicant": {
        "College educated": lambda df: df["education"].isin(["Bachelors", "Masters", "Doctorate", "Prof-school", "Assoc-acdm", "Assoc-voc"]),
        "Full-time standard": lambda df: (df["hours-per-week"] >= 38) & (df["hours-per-week"] <= 42),
        "White collar occupation": lambda df: df["occupation"].isin(["Exec-managerial", "Prof-specialty", "Tech-support", "Adm-clerical"]),
        "Has investment activity": lambda df: (df["capital-gain"] > 0) | (df["capital-loss"] > 0),
        "Peak career": lambda df: (df["age"] >= 35) & (df["age"] <= 55) & (df["hours-per-week"] >= 35) & (df["education-num"] >= 10),
        "Single / never married": lambda df: df["marital-status"] == "Never-married",
    },
}


def get_all_concepts():
    """Return a flat dict of unique concept_name -> callable (deduped)."""
    all_concepts = {}
    for stakeholder, concepts in STAKEHOLDER_CONCEPTS.items():
        for name, fn in concepts.items():
            if name not in all_concepts:
                all_concepts[name] = fn
    return all_concepts


# ============================================================
# Data loading & preprocessing
# ============================================================
def drop_cols_by_prefix(X, prefixes):
    """Drop all columns that exactly match or start with any of the given prefixes."""
    to_drop = [
        col for col in X.columns
        if any(col == p or col.startswith(p + "_") for p in prefixes)
    ]
    return X.drop(columns=to_drop, errors="ignore")


def prepare_data(drop_fnlwgt=False):
    """
    Replicate the exact preprocessing from Assignment 1's notebook:
    load from local CSV files, merge train+test, clean, one-hot encode
    with drop_first=True, then train/test split with random_state=42.

    Parameters
    ----------
    drop_fnlwgt : bool
        If True, also drop the fnlwgt column before encoding.
    """
    df_train = pd.read_csv(ADULT_DIR / "adult.data", names=HEADER, skipinitialspace=True)
    df_test = pd.read_csv(ADULT_DIR / "adult.test", names=HEADER, skipinitialspace=True, skiprows=1)
    df = pd.concat([df_train, df_test], ignore_index=True)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    X_raw = df.drop(columns=["income"])
    y_raw = df["income"]

    X_raw = X_raw.replace("?", pd.NA)
    y_raw = y_raw.astype(str).str.strip()

    valid_target = y_raw.isin(["<=50K", ">50K", "<=50K.", ">50K."])
    X_raw = X_raw.loc[valid_target]
    y_raw = y_raw.loc[valid_target]

    valid_X = X_raw.notna().all(axis=1)
    X_raw = X_raw.loc[valid_X]
    y_raw = y_raw.loc[valid_X]

    y = y_raw.map({"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1})

    # Save raw demographic columns BEFORE encoding
    demographics = X_raw[["sex", "race", "age"]].copy()
    # Save full raw features for concept evaluation
    X_raw_clean = X_raw.copy()

    if drop_fnlwgt:
        X_raw = X_raw.drop(columns=["fnlwgt"], errors="ignore")

    categorical_cols = X_raw.select_dtypes(include=["object", "string"]).columns
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

    # Drop sensitive features
    X_debiased = drop_cols_by_prefix(X_encoded, SENSITIVE_PREFIXES)

    X_train, X_test, y_train, y_test = train_test_split(
        X_debiased, y, test_size=0.2, random_state=42, stratify=y
    )

    demo_train, demo_test = train_test_split(
        demographics, test_size=0.2, random_state=42, stratify=y
    )

    # Split raw features for concept evaluation
    raw_train, raw_test = train_test_split(
        X_raw_clean, test_size=0.2, random_state=42, stratify=y
    )

    # Also split the full encoded set (with sensitive features) for proxy analysis
    X_full_encoded = pd.get_dummies(
        X_raw_clean if not drop_fnlwgt else X_raw_clean.drop(columns=["fnlwgt"], errors="ignore"),
        columns=X_raw_clean.select_dtypes(include=["object", "string"]).columns,
        drop_first=True,
    )
    X_full_train, X_full_test = train_test_split(
        X_full_encoded, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_full_train": X_full_train,
        "X_full_test": X_full_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "demo_train": demo_train,
        "demo_test": demo_test,
        "raw_train": raw_train,
        "raw_test": raw_test,
        "X_raw": X_raw_clean,
        "feature_names": X_train.columns.tolist(),
        "scaler": scaler,
    }


# ============================================================
# Model loading
# ============================================================
def load_models(data, drop_fnlwgt=False):
    """Load both saved models and align XGBoost features."""
    import xgboost as xgb
    import tensorflow as tf

    if drop_fnlwgt:
        xgb_path = MODEL_DIR / "xgb_no_fnlwgt.json"
        nn_path = MODEL_DIR / "nn_no_fnlwgt.keras"
    else:
        xgb_path = MODEL_DIR / "xgb_no_sensitive_no_country.json"
        nn_path = MODEL_DIR / "nn_no_sensitive_no_country.keras"

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    expected_features = xgb_model.get_booster().feature_names
    X_test_xgb = data["X_test"].reindex(columns=expected_features, fill_value=0)

    nn_model = tf.keras.models.load_model(str(nn_path))

    return xgb_model, nn_model, X_test_xgb


def limit_threads():
    """Limit CPU parallelism to avoid overloading the processor."""
    n = str(max(1, os.cpu_count() // 2))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", n)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    return int(n)
