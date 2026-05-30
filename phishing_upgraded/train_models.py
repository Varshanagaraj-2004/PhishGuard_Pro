"""
Train 4 Deep Learning Models for Phishing Detection
=====================================================
Models:
  1. Deep Neural Network (DNN)          – Multi-layer dense network
  2. CNN-1D                             – Convolutional feature extractor
  3. Bidirectional LSTM                 – Sequential / temporal patterns
  4. Attention-DNN (Transformer-style)  – Self-attention on feature tokens

Run:
    python train_models.py

Outputs (saved to ./models/):
    dnn_model.keras
    cnn1d_model.keras
    bilstm_model.keras
    attention_model.keras
    scaler.pkl
    training_history.pkl
    evaluation_results.pkl
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
)

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D,
    Flatten, LSTM, Bidirectional, Reshape, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_preprocessing import load_and_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 64
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# 1. Deep Neural Network
# ─────────────────────────────────────────────────────────────

def build_dnn(input_dim: int) -> Model:
    """
    5-layer deep fully-connected network with residual shortcuts,
    batch normalisation, dropout, and L2 regularisation.
    """
    inp = Input(shape=(input_dim,), name="dnn_input")

    # Block 1
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block 2  (residual shortcut)
    h = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    h = BatchNormalization()(h)
    h = Dropout(0.3)(h)
    short = Dense(128)(x)          # project for skip
    x = Add()([h, short])

    # Block 3
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Block 4
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)

    out = Dense(1, activation="sigmoid", name="dnn_output")(x)
    model = Model(inp, out, name="DNN")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ─────────────────────────────────────────────────────────────
# 2. 1-D CNN
# ─────────────────────────────────────────────────────────────

def build_cnn1d(input_dim: int) -> Model:
    """
    Treats each feature as a 1-D channel and applies stacked
    Conv1D → MaxPool layers to extract local feature patterns.
    """
    inp = Input(shape=(input_dim,), name="cnn_input")
    x = Reshape((input_dim, 1))(inp)

    # Conv block 1
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    # Conv block 2
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    # Conv block 3
    x = Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    out = Dense(1, activation="sigmoid", name="cnn_output")(x)
    model = Model(inp, out, name="CNN1D")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ─────────────────────────────────────────────────────────────
# 3. Bidirectional LSTM
# ─────────────────────────────────────────────────────────────

def build_bilstm(input_dim: int) -> Model:
    """
    Stacked Bidirectional LSTM reads the feature vector as a
    pseudo-sequence, capturing forward and backward dependencies.
    """
    inp = Input(shape=(input_dim,), name="bilstm_input")
    x = Reshape((input_dim, 1))(inp)

    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2,
                            recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2,
                            recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)

    out = Dense(1, activation="sigmoid", name="bilstm_output")(x)
    model = Model(inp, out, name="BiLSTM")
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ─────────────────────────────────────────────────────────────
# 4. Attention-DNN (Transformer-inspired)
# ─────────────────────────────────────────────────────────────

def build_attention_dnn(input_dim: int) -> Model:
    """
    Projects each feature into an embedding space, applies
    Multi-Head Self-Attention (Transformer encoder), then uses
    GlobalAveragePooling → dense classifier.
    """
    inp = Input(shape=(input_dim,), name="att_input")

    # Embed each scalar feature into a D-dim vector (token = feature)
    D = 32
    x = Dense(D)(inp)
    x = Reshape((1, D))(x)          # (batch, 1-token, D)

    # Project input_dim features into input_dim tokens of size D
    x_seq = Dense(D)(Reshape((input_dim, 1))(inp))  # (batch, feat, D)

    # Multi-Head Attention (self-attention over features)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=D // 4)(x_seq, x_seq)
    attn_out = Add()([x_seq, attn_out])
    attn_out = LayerNormalization()(attn_out)

    # Feed-forward sub-layer
    ff = Dense(D * 2, activation="relu")(attn_out)
    ff = Dense(D)(ff)
    ff = Add()([attn_out, ff])
    ff = LayerNormalization()(ff)

    # Pool over feature tokens
    x = GlobalAveragePooling1D()(ff)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    out = Dense(1, activation="sigmoid", name="att_output")(x)
    model = Model(inp, out, name="AttentionDNN")
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

def get_callbacks(name: str):
    ckpt_path = os.path.join(MODELS_DIR, f"{name}_best.keras")
    return [
        EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True,
                      mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_auc", save_best_only=True,
                        mode="max", verbose=0),
    ]


# ─────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred,
                                   target_names=["Safe", "Phishing"],
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    print(f"\n{'─'*50}")
    print(f"  {name} – Test AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Safe", "Phishing"]))

    return {
        "name": name, "cm": cm, "report": report,
        "auc": auc, "fpr": fpr, "tpr": tpr,
        "precision_curve": prec, "recall_curve": rec,
        "y_prob": y_prob, "y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

PALETTE = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444"]

def plot_training_history(histories: dict):
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle("Training History – All Models", fontsize=16, fontweight="bold")

    for idx, (name, hist) in enumerate(histories.items()):
        color = PALETTE[idx % len(PALETTE)]
        ax_acc = axes[0, idx]
        ax_loss = axes[1, idx]

        ax_acc.plot(hist["accuracy"], color=color, label="Train")
        ax_acc.plot(hist["val_accuracy"], color=color, linestyle="--", label="Val")
        ax_acc.set_title(f"{name}\nAccuracy", fontsize=11)
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(); ax_acc.grid(alpha=0.3)

        ax_loss.plot(hist["loss"], color=color, label="Train")
        ax_loss.plot(hist["val_loss"], color=color, linestyle="--", label="Val")
        ax_loss.set_title(f"{name}\nLoss", fontsize=11)
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
        ax_loss.legend(); ax_loss.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved → {path}")


def plot_confusion_matrices(results: list):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")

    for ax, res, color in zip(axes, results, PALETTE):
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Safe", "Phishing"],
                    yticklabels=["Safe", "Phishing"],
                    ax=ax, linewidths=0.5,
                    annot_kws={"size": 14})
        ax.set_title(f"{res['name']}\nAUC={res['auc']:.4f}", fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved → {path}")


def plot_roc_curves(results: list):
    fig, ax = plt.subplots(figsize=(8, 7))
    for res, color in zip(results, PALETTE):
        ax.plot(res["fpr"], res["tpr"],
                label=f"{res['name']}  (AUC={res['auc']:.4f})",
                color=color, linewidth=2.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves – All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved → {path}")


def plot_model_comparison(results: list):
    names = [r["name"] for r in results]
    metrics = {
        "Accuracy":  [r["report"]["accuracy"] for r in results],
        "Precision": [r["report"]["Phishing"]["precision"] for r in results],
        "Recall":    [r["report"]["Phishing"]["recall"] for r in results],
        "F1-Score":  [r["report"]["Phishing"]["f1-score"] for r in results],
        "AUC-ROC":   [r["auc"] for r in results],
    }

    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Model Comparison – Key Metrics", fontsize=14, fontweight="bold")

    colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=colors[i], alpha=0.85)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved → {path}")


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  PHISHING DETECTION – DEEP LEARNING TRAINING PIPELINE")
    print("═" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1/3] Loading & preprocessing data …")
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = \
        load_and_preprocess()
    input_dim = X_train.shape[1]

    # ── Build models ───────────────────────────────────────────
    builders = [
        ("DNN",          build_dnn),
        ("CNN1D",        build_cnn1d),
        ("BiLSTM",       build_bilstm),
        ("AttentionDNN", build_attention_dnn),
    ]

    histories = {}
    results = []

    print("\n[2/3] Training models …")
    for name, builder in builders:
        print(f"\n  ┌─ {name} {'─'*(45-len(name))}")
        t0 = time.time()
        model = builder(input_dim)
        model.summary(print_fn=lambda x: None)   # suppress verbose output

        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(name),
            verbose=1,
        )
        elapsed = time.time() - t0
        print(f"  └─ Done in {elapsed:.1f}s")

        # Save model
        save_path = os.path.join(MODELS_DIR, f"{name.lower()}_model.keras")
        model.save(save_path)
        print(f"  ✔ Model saved → {save_path}")

        histories[name] = hist.history
        results.append(evaluate_model(model, X_test, y_test, name))

    # ── Save artefacts ─────────────────────────────────────────
    print("\n[3/3] Generating evaluation plots & artefacts …")
    with open(os.path.join(MODELS_DIR, "training_history.pkl"), "wb") as f:
        pickle.dump(histories, f)
    with open(os.path.join(MODELS_DIR, "evaluation_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    plot_training_history(histories)
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_model_comparison(results)

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("═" * 60)
    print(f"  {'Model':<16}  {'Accuracy':>9}  {'Precision':>10}  "
          f"{'Recall':>8}  {'F1':>8}  {'AUC':>8}")
    print("  " + "─" * 58)
    for r in results:
        acc  = r["report"]["accuracy"]
        prec = r["report"]["Phishing"]["precision"]
        rec  = r["report"]["Phishing"]["recall"]
        f1   = r["report"]["Phishing"]["f1-score"]
        auc  = r["auc"]
        print(f"  {r['name']:<16}  {acc:>9.4f}  {prec:>10.4f}  "
              f"{rec:>8.4f}  {f1:>8.4f}  {auc:>8.4f}")
    print("═" * 60)
    print("  Training complete! Run: streamlit run streamlit_app.py\n")


if __name__ == "__main__":
    main()
