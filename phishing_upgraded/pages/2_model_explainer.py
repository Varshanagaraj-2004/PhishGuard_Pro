"""
Model Explainer Page
Shows per-feature contribution using gradient-based importance
(approximated without SHAP to avoid heavy dependencies).
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.feature_extractor import extract_features, get_feature_names

st.set_page_config(page_title="Explainer | PhishGuard Pro", page_icon="🔬", layout="wide")

st.markdown("## 🔬 Model Explainer")
st.markdown("""
Understand **why** the model flagged an input as phishing.  
Enter a URL, email, or message and see which features contributed most to the prediction.
""")

BASE = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE, "models")
MODEL_FILES = {
    "DNN":          "dnn_model.keras",
    "CNN1D":        "cnn1d_model.keras",
    "BiLSTM":       "bilstm_model.keras",
    "AttentionDNN": "attentiondnn_model.keras",
}


@st.cache_resource(show_spinner=False)
def load_models_explainer():
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


models, scaler = load_models_explainer()

if not models:
    st.warning("No trained models found. Run `python train_models.py` first.")
    st.stop()


def compute_gradient_importance(model, x_scaled: np.ndarray) -> np.ndarray:
    """Approximate feature importance via input × gradient."""
    import tensorflow as tf
    x_tensor = tf.Variable(x_scaled.astype("float32"))
    with tf.GradientTape() as tape:
        pred = model(x_tensor, training=False)
    grads = tape.gradient(pred, x_tensor).numpy().flatten()
    importance = np.abs(grads * x_scaled.flatten())
    return importance


# ── Input ─────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Enter URL / Email / Message", height=140,
                              placeholder="https://suspicious-login.paypa1.verify-now.xyz/account")
with col2:
    input_type = st.selectbox("Type", ["URL", "Email", "Message"])
    sel_model  = st.selectbox("Explain with model", list(models.keys()))
    run_btn    = st.button("🔬 Explain", type="primary", use_container_width=True)

if run_btn and user_input.strip():
    features   = extract_features(user_input.strip(), input_type)
    feat_names = get_feature_names(input_type)
    raw = np.array(features, dtype="float32").reshape(1, -1)
    scaled = scaler.transform(raw) if scaler else raw

    model = models[sel_model]
    prob  = float(model.predict(scaled, verbose=0).flatten()[0])
    importance = compute_gradient_importance(model, scaled)

    verdict = "⚠️ PHISHING" if prob >= 0.5 else "✅ SAFE"
    color   = "#EF4444" if prob >= 0.5 else "#22C55E"

    st.markdown(f"""
    <div style="background:{color}22; border:2px solid {color}; border-radius:12px;
                padding:1rem; text-align:center; margin-bottom:1.5rem;">
        <h2 style="color:{color}; margin:0">{verdict}</h2>
        <p style="margin:0.2rem 0">Phishing probability: <strong>{prob*100:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Feature table
    imp_df = pd.DataFrame({
        "Feature":    feat_names[:len(features)],
        "Value":      features[:len(feat_names)],
        "Importance": importance[:len(feat_names)],
    }).sort_values("Importance", ascending=False)

    st.markdown("### 📊 Feature Importance (Gradient × Input)")

    fig = go.Figure()
    top = imp_df.head(15)
    bar_colors = ["#EF4444" if v > 0 else "#22C55E" for v in top["Value"]]
    fig.add_trace(go.Bar(
        x=top["Importance"], y=top["Feature"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"val={v:.2f}" for v in top["Value"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top 15 Feature Contributions — {sel_model}",
        xaxis_title="Importance Score",
        height=460,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Colour table
    def style_row(row):
        if row["Value"] > 0.5:
            return ["background-color:#FEE2E2"] * len(row)
        elif row["Value"] < 0:
            return ["background-color:#DCFCE7"] * len(row)
        return [""] * len(row)

    st.markdown("### 📋 Full Feature Breakdown")
    st.dataframe(
        imp_df.style.apply(style_row, axis=1).format({"Value": "{:.3f}", "Importance": "{:.5f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Waterfall chart
    st.markdown("### 🌊 Feature Impact Waterfall")
    top_w = imp_df.head(10)
    signed = [v * i for v, i in zip(top_w["Value"], top_w["Importance"])]
    fig_wf = go.Figure(go.Waterfall(
        x=top_w["Feature"].tolist(),
        y=signed,
        connector={"line": {"color": "#CBD5E1"}},
        increasing={"marker": {"color": "#EF4444"}},
        decreasing={"marker": {"color": "#22C55E"}},
    ))
    fig_wf.update_layout(title="Signed Feature Contributions (top 10)", height=400)
    st.plotly_chart(fig_wf, use_container_width=True)

elif run_btn:
    st.warning("Please enter some text to analyse.")
