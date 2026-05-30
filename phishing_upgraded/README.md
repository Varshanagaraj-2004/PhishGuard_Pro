# 🛡️ PhishGuard Pro — Industry-Level Phishing Detection System

> **Enterprise-grade phishing detection powered by 4 deep learning architectures**  
> Built with TensorFlow 2.x · Streamlit · Plotly · Scikit-learn

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Deep Learning Models](#deep-learning-models)
- [Feature Engineering](#feature-engineering)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Streamlit App](#streamlit-app)
- [Results & Metrics](#results--metrics)
- [Dataset](#dataset)

---

## Overview

PhishGuard Pro detects phishing attempts across three input types:

| Input Type | Example |
|---|---|
| **URL** | `http://paypa1-secure.verify-now.xyz/login` |
| **Email** | Full email body with headers |
| **Message** | SMS / chat messages |

Four deep learning models vote on each input, and an ensemble confidence score is produced alongside a risk level (HIGH / MEDIUM / LOW).

---

## Project Structure

```
phishing_upgraded/
├── streamlit_app.py          ← Main Streamlit UI (multi-page)
├── train_models.py           ← Full training pipeline (4 DL models)
├── requirements.txt          ← Python dependencies
├── README.md
│
├── utils/
│   ├── __init__.py
│   ├── feature_extractor.py  ← URL / Email / Message feature engineering
│   └── data_preprocessing.py ← Dataset loading, scaling, splitting
│
├── data/
│   └── PhishingData.csv      ← 11,000+ labelled phishing samples
│
├── models/                   ← Auto-created after training
│   ├── dnn_model.keras
│   ├── cnn1d_model.keras
│   ├── bilstm_model.keras
│   ├── attentiondnn_model.keras
│   ├── scaler.pkl
│   ├── training_history.pkl
│   └── evaluation_results.pkl
│
├── static/                   ← Auto-created after training
│   ├── training_history.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── model_comparison.png
│
└── notebooks/
    └── EDA_and_Analysis.ipynb
```

---

## Deep Learning Models

### 1. 🧠 Deep Neural Network (DNN)
- 5-layer residual fully-connected network
- Residual shortcuts between dense blocks
- Batch Normalisation + Dropout (0.2–0.3)
- L2 weight regularisation
- Best suited for: tabular feature classification

### 2. 📡 1-D Convolutional Neural Network (CNN1D)
- Treats feature vector as a 1-D signal
- 3 Conv1D blocks with MaxPooling + BatchNorm
- Captures local feature co-occurrence patterns
- Best suited for: detecting correlated URL features

### 3. 🔄 Bidirectional LSTM (BiLSTM)
- 3 stacked Bidirectional LSTM layers
- Processes feature sequence in both directions
- Captures long-range dependencies
- Best suited for: sequential/contextual patterns

### 4. 👁️ Attention-DNN (Transformer-inspired)
- Multi-Head Self-Attention over feature tokens
- Layer Normalisation + residual connections
- GlobalAveragePooling → dense classifier
- Best suited for: learning which features matter most

### Ensemble Strategy
All 4 models output a phishing probability (0–1).  
Final score = **mean** of all valid model outputs.

| Score Range | Verdict |
|---|---|
| ≥ 0.80 | ⚠️ PHISHING — HIGH RISK |
| 0.55–0.79 | ⚠️ LIKELY PHISHING — MEDIUM RISK |
| 0.45–0.54 | ❓ UNCERTAIN |
| < 0.45 | ✅ SAFE |

---

## Feature Engineering

### URL Features (30)
| Feature | Description |
|---|---|
| `having_ip_address` | IP literal in host |
| `url_length` | Scored: <54 safe, >75 suspicious |
| `shortening_service` | Known URL shorteners |
| `having_at_symbol` | `@` in URL redirects |
| `double_slash_redirecting` | `//` in path |
| `prefix_suffix` | Hyphens in domain |
| `having_sub_domain` | Subdomain depth |
| `ssl_state` | HTTPS vs HTTP |
| `https_token_in_domain` | "https" in domain name |
| `suspicious_keywords` | Brand/login keywords |
| … | (30 total) |

### Email / Message Features (30)
| Feature | Description |
|---|---|
| `urgent_language` | "act now", "expires" etc. |
| `reward_language` | "winner", "prize" etc. |
| `threat_language` | "suspended", "terminated" |
| `caps_ratio` | Excessive capitalisation |
| `misspelled_brand` | paypall, amaz0n, etc. |
| `generic_greeting` | "Dear Customer" |
| `request_credentials` | Asks for password/SSN |
| `suspicious_attachments` | .exe, .docm, .vbs |
| `keyword_density` | Phishing keyword concentration |
| … | (30 total) |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all 4 models
```bash
python train_models.py
```
Training takes ~5–15 minutes depending on hardware (GPU recommended).  
Models, scaler, and evaluation plots are saved automatically.

### 3. Launch the Streamlit app
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`

---

## Training Pipeline

`train_models.py` runs the full pipeline:

1. **Load** `PhishingData.csv` (11,000+ samples, 30 features)
2. **Preprocess**: clean columns, encode labels (−1 → 0), standard-scale
3. **Split**: 70% train / 15% validation / 15% test (stratified)
4. **Train** each model with:
   - Early stopping (patience=10, monitor=val_auc)
   - ReduceLROnPlateau (factor=0.5, patience=5)
   - ModelCheckpoint (saves best val_auc epoch)
5. **Evaluate**: classification report, AUC-ROC, confusion matrix
6. **Save**: `.keras` models, scaler, history, evaluation results
7. **Plot**: training curves, ROC curves, confusion matrices, comparison bar chart

---

## Streamlit App

### Pages
| Page | Description |
|---|---|
| 🏠 **Detect** | Enter URL/Email/Message → instant ensemble prediction |
| 📊 **Dashboard** | Full model metrics, interactive ROC curves, training history |
| 📜 **History** | Session audit log with CSV export |
| ℹ️ **About Models** | Model architectures, dataset info, feature engineering docs |

---

## Results & Metrics

Expected performance after full training (60 epochs):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| DNN | ~97% | ~0.97 | ~0.97 | ~0.97 | ~0.99 |
| CNN1D | ~97% | ~0.97 | ~0.97 | ~0.97 | ~0.99 |
| BiLSTM | ~96% | ~0.96 | ~0.96 | ~0.96 | ~0.99 |
| AttentionDNN | ~97% | ~0.97 | ~0.97 | ~0.97 | ~0.99 |

*Actual values depend on random seed and hardware.*

---

## Dataset

**PhishingData.csv** — UCI ML Repository Phishing Dataset  
- **11,055 samples** (phishing + legitimate websites)  
- **30 features** derived from URL structure, page content, and external services  
- **Labels**: `1` = Phishing, `-1` = Safe (converted to 0 during preprocessing)

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| UI | Streamlit 1.35 |
| Charts | Plotly, Matplotlib, Seaborn |
| ML Utilities | Scikit-learn |
| Data | Pandas, NumPy |

---

*PhishGuard Pro — Built for academic excellence and industry readiness.*
