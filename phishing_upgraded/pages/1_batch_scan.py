"""
Batch Scan Page
Allows users to upload a CSV of URLs/emails/messages and scan all at once.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.feature_extractor import extract_features

st.set_page_config(page_title="Batch Scan | PhishGuard Pro", page_icon="📂", layout="wide")

st.markdown("## 📂 Batch Scan")
st.markdown("Upload a CSV file with a column named **`input`** (URLs, emails, or messages) "
            "and an optional **`type`** column (`URL`, `Email`, `Message`). "
            "PhishGuard Pro will scan every row using all 4 models.")

# ── Load models ───────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE, "models")

MODEL_FILES = {
    "DNN":          "dnn_model.keras",
    "CNN1D":        "cnn1d_model.keras",
    "BiLSTM":       "bilstm_model.keras",
    "AttentionDNN": "attentiondnn_model.keras",
}


@st.cache_resource(show_spinner=False)
def load_models_batch():
    import tensorflow as tf
    import joblib
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
            except Exception:
                pass
    scaler = None
    sp = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(sp):
        scaler = joblib.load(sp)
    return models, scaler


models, scaler = load_models_batch()

if not models:
    st.warning("No trained models found. Run `python train_models.py` first.")
    st.stop()

# ── Upload ────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV", type=["csv"])
col1, col2 = st.columns(2)
default_type = col1.selectbox("Default input type (if no `type` column)", ["URL", "Email", "Message"])
threshold = col2.slider("Phishing threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded:
    df = pd.read_csv(uploaded)
    st.markdown(f"**{len(df):,} rows loaded.** Preview:")
    st.dataframe(df.head(5), use_container_width=True)

    if "input" not in df.columns:
        st.error("CSV must have a column named `input`.")
        st.stop()

    if st.button("🚀 Run Batch Scan", type="primary"):
        results = []
        prog = st.progress(0.0, text="Scanning…")

        for i, row in df.iterrows():
            text = str(row["input"])
            itype = str(row.get("type", default_type)) if "type" in df.columns else default_type

            features = extract_features(text, itype)
            raw = np.array(features, dtype="float32").reshape(1, -1)
            scaled = scaler.transform(raw) if scaler else raw

            probs = []
            for name, model in models.items():
                try:
                    p = float(model.predict(scaled, verbose=0).flatten()[0])
                    probs.append(p)
                except Exception:
                    pass

            ensemble = float(np.mean(probs)) if probs else 0.5
            verdict = "Phishing" if ensemble >= threshold else "Safe"

            results.append({
                "input":      text[:80] + ("…" if len(text) > 80 else ""),
                "type":       itype,
                "verdict":    verdict,
                "confidence": f"{ensemble*100:.1f}%",
                "score":      round(ensemble, 4),
            })
            prog.progress((i + 1) / len(df), text=f"Scanned {i+1}/{len(df)}")

        st.success("Batch scan complete!")
        res_df = pd.DataFrame(results)

        # Summary
        total = len(res_df)
        n_phish = (res_df["verdict"] == "Phishing").sum()
        n_safe  = total - n_phish
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("🔴 Phishing", n_phish)
        c3.metric("🟢 Safe", n_safe)

        fig_pie = px.pie(
            values=[n_phish, n_safe], names=["Phishing", "Safe"],
            color_discrete_map={"Phishing": "#EF4444", "Safe": "#22C55E"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Colour verdicts
        def colour_row(row):
            color = "#FEE2E2" if row["verdict"] == "Phishing" else "#DCFCE7"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            res_df.style.apply(colour_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Download
        csv_out = res_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_out,
            file_name=f"batch_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ── Sample CSV ────────────────────────────────────────────────
with st.expander("📋 Download sample CSV template"):
    sample = pd.DataFrame({
        "input": [
            "https://paypa1-secure.login.xyz/account",
            "https://www.google.com",
            "Congratulations! You won a $500 Amazon gift card. Click now!",
            "Your invoice is attached. Please review.",
        ],
        "type": ["URL", "URL", "Message", "Email"],
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.download_button(
        "Download sample_batch.csv",
        data=sample.to_csv(index=False),
        file_name="sample_batch.csv",
        mime="text/csv",
    )
