"""
PhishGuard Pro — Command-Line Predictor
========================================
Quickly check a single URL, email, or message from the terminal.

Usage:
    python predict_cli.py --type URL --input "http://paypa1.verify.xyz/account"
    python predict_cli.py --type Email --input "Dear customer, your account is suspended."
    python predict_cli.py --type Message --input "You won a prize! Click now."
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils.feature_extractor import extract_features
from utils.ensemble import weighted_ensemble, confidence_level

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")

MODEL_FILES = {
    "DNN":          "dnn_model.keras",
    "CNN1D":        "cnn1d_model.keras",
    "BiLSTM":       "bilstm_model.keras",
    "AttentionDNN": "attentiondnn_model.keras",
}


def load_models_and_scaler():
    import tensorflow as tf
    import joblib

    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)

    scaler = None
    sp = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(sp):
        scaler = joblib.load(sp)

    return models, scaler


def predict(input_text: str, input_type: str, models: dict, scaler) -> dict:
    features = extract_features(input_text, input_type)
    raw = np.array(features, dtype="float32").reshape(1, -1)
    scaled = scaler.transform(raw) if scaler else raw

    preds = {}
    for name, model in models.items():
        try:
            prob = float(model.predict(scaled, verbose=0).flatten()[0])
            preds[name] = prob
        except Exception:
            preds[name] = None

    ensemble = weighted_ensemble(preds)
    return {"individual": preds, "ensemble": ensemble}


def main():
    parser = argparse.ArgumentParser(description="PhishGuard Pro CLI")
    parser.add_argument("--type",  required=True, choices=["URL", "Email", "Message"])
    parser.add_argument("--input", required=True, help="Text to analyse")
    args = parser.parse_args()

    print("\n" + "═" * 58)
    print("  🛡️  PhishGuard Pro — CLI Predictor")
    print("═" * 58)
    print(f"  Type  : {args.type}")
    print(f"  Input : {args.input[:80]}{'…' if len(args.input) > 80 else ''}")
    print("─" * 58)

    print("  Loading models …", end=" ", flush=True)
    models, scaler = load_models_and_scaler()
    if not models:
        print("\n  ❌ No models found. Run: python train_models.py")
        sys.exit(1)
    print(f"Loaded {len(models)} models ✔")

    result = predict(args.input, args.type, models, scaler)
    ensemble = result["ensemble"]

    print("\n  Individual Model Predictions:")
    for name, prob in result["individual"].items():
        if prob is None:
            print(f"    {name:<16}  (failed)")
        else:
            verdict = "PHISHING" if prob >= 0.5 else "SAFE    "
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"    {name:<16}  {verdict}  [{bar}]  {prob*100:.1f}%")

    verdict_str = "⚠️  PHISHING" if ensemble >= 0.5 else "✅  SAFE"
    confidence  = confidence_level(ensemble)

    print("─" * 58)
    print(f"  Ensemble Score : {ensemble*100:.1f}%")
    print(f"  Confidence     : {confidence}")
    print(f"  Final Verdict  : {verdict_str}")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    main()
