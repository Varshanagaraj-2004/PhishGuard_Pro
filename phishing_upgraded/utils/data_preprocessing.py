"""
Data Preprocessing Pipeline
Loads PhishingData.csv, cleans, encodes, and splits for model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "PhishingData.csv")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")


def load_and_preprocess(data_path: str = DATA_PATH):
    """
    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    feature_names          : list[str]
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Drop index column if present
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    # Separate features / label
    X = df.drop(columns=["Result"])
    y = df["Result"].copy()

    # Encode labels: -1 → 0 (Safe), 1 → 1 (Phishing)
    y = y.replace(-1, 0).astype(np.float32)

    # Convert all features to float
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    feature_names = X.columns.tolist()
    X = X.values
    y = y.values

    # 70/15/15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✔ Scaler saved → {SCALER_PATH}")

    print(f"  Dataset: {len(X_train)+len(X_val)+len(X_test)} samples | "
          f"Features: {X_train.shape[1]} | "
          f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"  Label balance  –  Phishing: {int(y_train.sum())} | "
          f"Safe: {int((y_train==0).sum())}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def preprocess_single(features: list, scaler_path: str = SCALER_PATH) -> np.ndarray:
    """Scale a single sample using the saved scaler."""
    scaler = joblib.load(scaler_path)
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    return scaler.transform(arr)
