"""
Standalone Model Evaluation Script
====================================
Loads trained models and prints a full evaluation report.

Usage:
    python evaluate_models.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)

import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_preprocessing import load_and_preprocess

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")
STATIC_DIR = os.path.join(BASE, "static")

MODEL_FILES = {
    "DNN":          "dnn_model.keras",
    "CNN1D":        "cnn1d_model.keras",
    "BiLSTM":       "bilstm_model.keras",
    "AttentionDNN": "attentiondnn_model.keras",
}


def main():
    print("\n" + "═" * 60)
    print("  PHISHING DETECTION — MODEL EVALUATION")
    print("═" * 60)

    print("\nLoading dataset …")
    _, _, X_test, _, _, y_test, _ = load_and_preprocess()

    rows = []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {name}: model file not found at {path}")
            continue

        print(f"\n── {name} ──")
        model = tf.keras.models.load_model(path)
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(
            y_test, y_pred, target_names=["Safe", "Phishing"]
        )
        print(report)
        print(f"AUC-ROC: {auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        rows.append({
            "Model": name,
            "AUC":   round(auc, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        })

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  {'Model':<16}  {'AUC':>8}  {'TP':>6}  {'TN':>6}  {'FP':>6}  {'FN':>6}")
    for r in rows:
        print(f"  {r['Model']:<16}  {r['AUC']:>8}  {r['TP']:>6}  "
              f"{r['TN']:>6}  {r['FP']:>6}  {r['FN']:>6}")
    print("═" * 60)


if __name__ == "__main__":
    main()
