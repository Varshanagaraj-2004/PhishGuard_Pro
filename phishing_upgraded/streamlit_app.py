"""
PhishGuard Pro – Industry-Level Phishing Detection System
==========================================================
Streamlit multi-page application with:
  • Real-time prediction via 4 deep learning models
  • Ensemble voting with confidence scores
  • Detailed feature explanation
  • Model performance dashboard
  • History / audit log
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from utils.feature_extractor import extract_features, get_feature_names

# ─────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")
STATIC_DIR = os.path.join(BASE, "static")

MODEL_FILES = {
    "DNN":          "dnn_model.keras",
    "CNN1D":        "cnn1d_model.keras",
    "BiLSTM":       "bilstm_model.keras",
    "AttentionDNN": "attentiondnn_model.keras",
}
MODEL_COLORS = {
    "DNN":          "#4F46E5",
    "CNN1D":        "#10B981",
    "BiLSTM":       "#F59E0B",
    "AttentionDNN": "#EF4444",
}
MODEL_ICONS = {
    "DNN":          "🧠",
    "CNN1D":        "📡",
    "BiLSTM":       "🔄",
    "AttentionDNN": "👁️",
}
MODEL_DESCRIPTIONS = {
    "DNN": "5-layer residual network with batch normalisation and dropout regularisation.",
    "CNN1D": "1-D Convolutional network that treats features as local patterns.",
    "BiLSTM": "Stacked Bidirectional LSTM capturing forward and backward dependencies.",
    "AttentionDNN": "Transformer-inspired model with multi-head self-attention over feature tokens.",
}

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ──────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
    border-radius: 18px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(79,70,229,0.35);
}
.hero h1 { color: #ffffff; font-size: 2.8rem; font-weight: 700; margin: 0; }
.hero p  { color: #C7D2FE; font-size: 1.1rem; margin-top: 0.5rem; }

/* ── Result cards ─────────────────────────────────────── */
.result-phishing {
    background: linear-gradient(135deg, #FEF2F2, #FFE4E6);
    border: 2px solid #EF4444;
    border-radius: 16px; padding: 1.5rem; text-align: center;
    box-shadow: 0 8px 32px rgba(239,68,68,0.2);
}
.result-safe {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border: 2px solid #22C55E;
    border-radius: 16px; padding: 1.5rem; text-align: center;
    box-shadow: 0 8px 32px rgba(34,197,94,0.2);
}
.result-phishing h2 { color: #DC2626; font-size: 2rem; }
.result-safe h2     { color: #16A34A; font-size: 2rem; }

/* ── Metric cards ─────────────────────────────────────── */
.metric-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    border-top: 4px solid #4F46E5;
}

/* ── Model cards ──────────────────────────────────────── */
.model-card {
    background: #F8FAFC;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 5px solid #4F46E5;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #1E1B4B; }
[data-testid="stSidebar"] * { color: #E0E7FF !important; }
[data-testid="stSidebar"] .stRadio > label { color: #A5B4FC !important; }

/* ── Input styling ────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    border-radius: 10px !important;
    border: 2px solid #E2E8F0 !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    import tensorflow as tf
    import joblib

    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")

    scaler = None
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    return models, scaler


@st.cache_data(show_spinner=False)
def load_eval_results():
    path = os.path.join(MODELS_DIR, "evaluation_results.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_training_history():
    path = os.path.join(MODELS_DIR, "training_history.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ─────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────

def predict_all(models, scaler, features: list):
    """Run all models and return per-model probabilities + ensemble."""
    raw = np.array(features, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        scaled = scaler.transform(raw)
    else:
        scaled = raw

    preds = {}
    for name, model in models.items():
        try:
            prob = float(model.predict(scaled, verbose=0).flatten()[0])
            preds[name] = prob
        except Exception as e:
            preds[name] = None

    valid = [v for v in preds.values() if v is not None]
    ensemble = float(np.mean(valid)) if valid else None
    return preds, ensemble


def verdict(prob: float) -> tuple:
    if prob >= 0.80:
        return "⚠️ PHISHING", "phishing", "HIGH RISK"
    elif prob >= 0.55:
        return "⚠️ LIKELY PHISHING", "phishing", "MEDIUM RISK"
    elif prob >= 0.45:
        return "❓ UNCERTAIN", "safe", "LOW CONFIDENCE"
    else:
        return "✅ SAFE", "safe", "SAFE"


# ─────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ PhishGuard Pro")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Detect", "📊 Dashboard", "📜 History", "ℹ️ About Models"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("#### Model Status")

    models_loaded, scaler = load_models()
    for name in MODEL_FILES:
        loaded = name in models_loaded
        icon = "🟢" if loaded else "🔴"
        st.markdown(f"{icon} **{name}**")

    if not models_loaded:
        st.error("No models found.\nRun `python train_models.py` first.")

    st.markdown("---")
    st.caption("PhishGuard Pro v2.0  \nBuilt with TensorFlow & Streamlit")


# ══════════════════════════════════════════════════════════════
# PAGE 1 – Detect
# ══════════════════════════════════════════════════════════════
if page == "🏠 Detect":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ PhishGuard Pro</h1>
        <p>Enterprise-grade phishing detection powered by 4 deep learning models</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input section ─────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🔍 Analyse Input")
        input_type = st.selectbox(
            "Input Type",
            ["URL", "Email", "Message"],
            help="Select what you want to analyse",
        )
        type_hints = {
            "URL":     "https://suspicious-login.verify-now.xyz/paypal/account",
            "Email":   "Subject: Urgent! Your account will be suspended\n\nDear customer, click here to verify your identity...",
            "Message": "WINNER! You've been selected for a $1,000 gift card. Click NOW to claim your prize before it expires!",
        }
        user_input = st.text_area(
            f"Enter {input_type}",
            placeholder=type_hints[input_type],
            height=160,
        )
        analyse_btn = st.button("🚀 Analyse Now", use_container_width=True, type="primary")

    with col_right:
        st.subheader("📋 Quick Tips")
        tip_tabs = st.tabs(["URL Tips", "Email Tips", "Message Tips"])
        with tip_tabs[0]:
            st.markdown("""
- IP addresses in URLs (e.g. `http://192.168.1.1/login`)
- Suspicious TLDs: `.xyz`, `.top`, `.bid`
- Misspelled brands: `paypa1.com`
- URL shorteners: `bit.ly`, `tinyurl`
- Excessive hyphens or subdomains
""")
        with tip_tabs[1]:
            st.markdown("""
- Urgent / threatening language
- Requests for credentials
- Generic greetings ("Dear Customer")
- Mismatched sender address
- Suspicious attachments (`.exe`, `.zip`)
""")
        with tip_tabs[2]:
            st.markdown("""
- Prize / reward claims
- "Act now" / "Limited time" pressure
- Unexpected links or phone numbers
- Too-good-to-be-true offers
- Requests to call premium numbers
""")

    # ── Prediction ────────────────────────────────────────────
    if analyse_btn and user_input.strip():
        if not models_loaded:
            st.error("No models are loaded. Please run `python train_models.py` first.")
            st.stop()

        with st.spinner("Analysing with deep learning models …"):
            time.sleep(0.3)   # small UX pause
            features = extract_features(user_input.strip(), input_type)
            preds, ensemble = predict_all(models_loaded, scaler, features)

        if ensemble is None:
            st.error("Prediction failed. Please check model files.")
            st.stop()

        label, cls, risk = verdict(ensemble)
        is_phishing = cls == "phishing"

        st.markdown("---")
        st.subheader("🎯 Detection Result")

        r_col1, r_col2 = st.columns([2, 3])

        with r_col1:
            card_class = "result-phishing" if is_phishing else "result-safe"
            st.markdown(f"""
            <div class="{card_class}">
                <h2>{label}</h2>
                <p style="font-size:1.15rem; font-weight:600; margin:0.2rem 0">
                    Risk: <strong>{risk}</strong>
                </p>
                <p style="font-size:1.1rem; color:#374151">
                    Ensemble Confidence: <strong>{ensemble*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ensemble * 100,
                title={"text": "Phishing Probability %", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#EF4444" if is_phishing else "#22C55E"},
                    "steps": [
                        {"range": [0, 45],  "color": "#DCFCE7"},
                        {"range": [45, 55], "color": "#FEF9C3"},
                        {"range": [55, 100],"color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1E1B4B", "width": 3},
                        "thickness": 0.75, "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r_col2:
            st.markdown("#### Individual Model Predictions")
            for mname, prob in preds.items():
                if prob is None:
                    continue
                mlabel = "Phishing" if prob >= 0.5 else "Safe"
                mcolor = "#EF4444" if prob >= 0.5 else "#22C55E"
                fig_bar = go.Figure(go.Bar(
                    x=[prob * 100],
                    y=[f"{MODEL_ICONS[mname]} {mname}"],
                    orientation="h",
                    marker_color=mcolor,
                    text=[f"{prob*100:.1f}% – {mlabel}"],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    height=70, margin=dict(l=0, r=80, t=0, b=0),
                    xaxis=dict(range=[0, 110], showticklabels=False),
                    yaxis=dict(showticklabels=True),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # ── Feature importance view ────────────────────────────
        st.markdown("---")
        with st.expander("🔬 Feature Analysis (click to expand)", expanded=False):
            feat_names = get_feature_names(input_type)
            feat_df = pd.DataFrame({
                "Feature": feat_names[:len(features)],
                "Value":   features[:len(feat_names)],
            })
            feat_df["Contribution"] = feat_df["Value"].abs()
            feat_df = feat_df.sort_values("Contribution", ascending=False)

            fig_feat = px.bar(
                feat_df.head(15), x="Contribution", y="Feature",
                orientation="h", color="Value",
                color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
                title="Top 15 Feature Contributions",
            )
            fig_feat.update_layout(height=400)
            st.plotly_chart(fig_feat, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Active Risk Factors:**")
                risky = feat_df[feat_df["Value"] > 0]["Feature"].tolist()[:8]
                for f in risky:
                    st.markdown(f"🔴 `{f}`")
            with col_b:
                st.markdown("**Safe Indicators:**")
                safe_feats = feat_df[feat_df["Value"] <= 0]["Feature"].tolist()[:8]
                for f in safe_feats:
                    st.markdown(f"🟢 `{f}`")

        # ── Save to history ────────────────────────────────────
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type":      input_type,
            "input":     user_input[:120] + ("…" if len(user_input) > 120 else ""),
            "verdict":   label,
            "confidence": f"{ensemble*100:.1f}%",
            "risk":      risk,
        })

    elif analyse_btn:
        st.warning("Please enter some text to analyse.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 – Dashboard
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("## 📊 Model Performance Dashboard")

    eval_results = load_eval_results()
    train_history = load_training_history()

    if eval_results is None:
        st.info("No evaluation data found. Run `python train_models.py` to generate metrics.")
        st.stop()

    # ── KPI strip ──────────────────────────────────────────────
    st.markdown("### 🏆 Best Model Metrics")
    best = max(eval_results, key=lambda r: r["auc"])
    cols = st.columns(5)
    kpis = [
        ("Best Model",   best["name"],                                      "🥇"),
        ("Accuracy",     f"{best['report']['accuracy']*100:.2f}%",          "🎯"),
        ("AUC-ROC",      f"{best['auc']:.4f}",                              "📈"),
        ("F1-Score",     f"{best['report']['Phishing']['f1-score']:.4f}",   "⚡"),
        ("Precision",    f"{best['report']['Phishing']['precision']:.4f}",  "🔍"),
    ]
    for col, (title, val, icon) in zip(cols, kpis):
        with col:
            st.metric(f"{icon} {title}", val)

    st.markdown("---")

    # ── Comparison table ───────────────────────────────────────
    st.markdown("### 📋 Full Model Comparison")
    rows = []
    for r in eval_results:
        rows.append({
            "Model":     r["name"],
            "Accuracy":  f"{r['report']['accuracy']*100:.2f}%",
            "Precision": f"{r['report']['Phishing']['precision']:.4f}",
            "Recall":    f"{r['report']['Phishing']['recall']:.4f}",
            "F1-Score":  f"{r['report']['Phishing']['f1-score']:.4f}",
            "AUC-ROC":   f"{r['auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Charts ─────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📉 ROC Curves")
        fig_roc = go.Figure()
        for r, color in zip(eval_results, MODEL_COLORS.values()):
            fig_roc.add_trace(go.Scatter(
                x=r["fpr"], y=r["tpr"],
                name=f"{r['name']} (AUC={r['auc']:.3f})",
                line=dict(color=color, width=2.5),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(dash="dash", color="gray"),
        ))
        fig_roc.update_layout(
            xaxis_title="FPR", yaxis_title="TPR",
            legend=dict(x=0.55, y=0.15), height=400,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        st.markdown("### 🎯 Confusion Matrices")
        sel_model = st.selectbox("Select Model", [r["name"] for r in eval_results])
        res = next(r for r in eval_results if r["name"] == sel_model)
        fig_cm = px.imshow(
            res["cm"], text_auto=True,
            x=["Safe", "Phishing"], y=["Safe", "Phishing"],
            color_continuous_scale="Blues",
            title=f"{sel_model} – Confusion Matrix",
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Training history ───────────────────────────────────────
    if train_history:
        st.markdown("### 📈 Training History")
        model_sel = st.selectbox("Model", list(train_history.keys()), key="hist_sel")
        h = train_history[model_sel]

        fig_hist = go.Figure()
        color = list(MODEL_COLORS.values())[list(train_history.keys()).index(model_sel)]
        fig_hist.add_trace(go.Scatter(y=h["accuracy"],     name="Train Acc", line=dict(color=color)))
        fig_hist.add_trace(go.Scatter(y=h["val_accuracy"], name="Val Acc",   line=dict(color=color, dash="dash")))
        fig_hist.add_trace(go.Scatter(y=h["loss"],         name="Train Loss",line=dict(color="#F59E0B"), yaxis="y2"))
        fig_hist.add_trace(go.Scatter(y=h["val_loss"],     name="Val Loss",  line=dict(color="#F59E0B", dash="dash"), yaxis="y2"))
        fig_hist.update_layout(
            height=400,
            yaxis=dict(title="Accuracy"),
            yaxis2=dict(title="Loss", overlaying="y", side="right"),
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Static plots ───────────────────────────────────────────
    st.markdown("### 🖼️ Evaluation Plots")
    p_col1, p_col2 = st.columns(2)
    img_files = {
        "Model Comparison":  "model_comparison.png",
        "Training History":  "training_history.png",
        "Confusion Matrices":"confusion_matrices.png",
        "ROC Curves":        "roc_curves.png",
    }
    for (title, fname), col in zip(img_files.items(), [p_col1, p_col2, p_col1, p_col2]):
        path = os.path.join(STATIC_DIR, fname)
        if os.path.exists(path):
            with col:
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 – History
# ══════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.markdown("## 📜 Detection History")

    if not st.session_state.history:
        st.info("No analyses run yet. Go to **🏠 Detect** to start.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Summary stats
        total = len(df_hist)
        phishing_count = df_hist["verdict"].str.contains("PHISHING").sum()
        safe_count = total - phishing_count

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Analysed", total)
        c2.metric("🔴 Phishing Detected", phishing_count)
        c3.metric("🟢 Safe", safe_count)

        if total > 0:
            fig_pie = px.pie(
                values=[phishing_count, safe_count],
                names=["Phishing", "Safe"],
                color_discrete_map={"Phishing": "#EF4444", "Safe": "#22C55E"},
                title="Analysis Breakdown",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

        # Download
        csv = df_hist.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"phishguard_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4 – About Models
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About Models":
    st.markdown("## ℹ️ About the Models")
    st.markdown("""
    PhishGuard Pro uses **four specialised deep learning architectures**, each designed to
    capture different aspects of phishing patterns. An **ensemble majority vote** combines
    their outputs for the final verdict.
    """)

    for name in MODEL_FILES:
        color = MODEL_COLORS[name]
        icon  = MODEL_ICONS[name]
        desc  = MODEL_DESCRIPTIONS[name]
        loaded = name in models_loaded
        status = "🟢 Loaded" if loaded else "🔴 Not loaded"

        st.markdown(f"""
        <div class="model-card" style="border-left-color:{color}">
            <h3 style="margin:0">{icon} {name} &nbsp; <small style="color:#6B7280;font-size:0.8rem">{status}</small></h3>
            <p style="margin:0.4rem 0 0 0; color:#374151">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗃️ Dataset Information")
    data_path = os.path.join(BASE, "data", "PhishingData.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples",  f"{len(df):,}")
        col2.metric("Features",       f"{df.shape[1]-1}")
        phish_pct = (df["Result"] == 1).mean() * 100
        col3.metric("Phishing %",     f"{phish_pct:.1f}%")
        with st.expander("Feature Names"):
            feat_cols = [c.strip() for c in df.columns if c.strip() != "Result"]
            for i in range(0, len(feat_cols), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(feat_cols):
                        col.markdown(f"• `{feat_cols[i+j]}`")

    st.markdown("---")
    st.markdown("### 🔧 Feature Engineering")
    st.markdown("""
    The feature extractor analyses three input types:

    | Input Type | Feature Categories |
    |---|---|
    | **URL** | IP address, URL length, shorteners, HTTPS, suspicious TLDs, sub-domain depth, redirect count |
    | **Email** | Link safety, urgency keywords, brand misspellings, credential requests, sender spoofing |
    | **Message** | Reward/threat language, keyword density, capitalisation ratio, link presence |

    All 30 features are normalised using a `StandardScaler` fitted on the training set.
    """)
