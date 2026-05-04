"""
Assignment 2 — Retrain models without fnlwgt
=============================================
Trains XGBoost and Neural Network on the debiased dataset
(no sex, race, native-country, fnlwgt) and saves them to models/.

Usage:
    python assignment2/train_models.py
"""

import os
import sys

# Limit threads before importing heavy libraries
N_THREADS = str(max(1, os.cpu_count() // 2))
os.environ.setdefault("OMP_NUM_THREADS", N_THREADS)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", N_THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.threading.set_intra_op_parallelism_threads(int(N_THREADS))
tf.config.threading.set_inter_op_parallelism_threads(2)

sys.path.insert(0, os.path.dirname(__file__))
from concepts import prepare_data, MODEL_DIR


def main():
    print("=" * 60)
    print("Retraining models without fnlwgt")
    print("=" * 60)

    data = prepare_data(drop_fnlwgt=True)
    print(f"Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")
    print(f"Features: {data['X_train'].shape[1]}")

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]

    # ----------------------------------------------------------
    # XGBoost
    # ----------------------------------------------------------
    print("\n--- XGBoost ---")
    X_train_num = X_train.astype("float32")
    X_test_num = X_test.astype("float32")

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train_num, y_train.astype(int))

    xgb_path = MODEL_DIR / "xgb_no_fnlwgt.json"
    xgb_model.save_model(str(xgb_path))
    print(f"Saved: {xgb_path}")

    y_pred_xgb = xgb_model.predict(X_test_num)
    y_proba_xgb = xgb_model.predict_proba(X_test_num)[:, 1]
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_auc = roc_auc_score(y_test, y_proba_xgb)
    print(f"Accuracy: {xgb_acc:.4f}")
    print(f"ROC-AUC:  {xgb_auc:.4f}")
    print(classification_report(y_test, y_pred_xgb))

    # ----------------------------------------------------------
    # Neural Network
    # ----------------------------------------------------------
    print("\n--- Neural Network ---")
    nn_model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    nn_path = MODEL_DIR / "nn_no_fnlwgt.keras"
    nn_model.save(str(nn_path))
    print(f"Saved: {nn_path}")

    y_proba_nn = nn_model.predict(X_test_scaled, verbose=0).ravel()
    y_pred_nn = (y_proba_nn >= 0.5).astype(int)
    nn_acc = accuracy_score(y_test, y_pred_nn)
    nn_auc = roc_auc_score(y_test, y_proba_nn)
    print(f"Accuracy: {nn_acc:.4f}")
    print(f"ROC-AUC:  {nn_auc:.4f}")
    print(classification_report(y_test, y_pred_nn))

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("=" * 60)
    print("SUMMARY (no fnlwgt, no sex, no race, no native-country)")
    print("=" * 60)
    print(f"XGBoost — Acc: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}")
    print(f"NN      — Acc: {nn_acc:.4f}, AUC: {nn_auc:.4f}")
    print(f"Features: {X_train.shape[1]}")


if __name__ == "__main__":
    main()
