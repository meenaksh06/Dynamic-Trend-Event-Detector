# Dynamic Trend & Event Detector

**An Advanced Neuro-Statistical Pipeline for Semantic Topic Discovery, Hybrid Temporal Forecasting, and Deep Learning Model Interpretability.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow%20/%20PyTorch-orange.svg)](https://tensorflow.org)
[![XAI](https://img.shields.io/badge/Insights-SHAP%20/%20Attention-green.svg)](https://github.com/slundberg/shap)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED.svg)](https://docs.docker.com/compose/)

---

## Overview
The **Dynamic Trend & Event Detector** is a high-fidelity system designed to identify, track, and forecast evolving news narratives. By fusing probabilistic generative models with contextual transformers and sequence architectures, it provides deep insights into the "pulse" of global media.

The project spans **3 phases** — from raw data processing (Phase 1), through deep learning topic modeling and forecasting (Phase 2), to a **synergistic hybrid ARIMA + BiLSTM architecture** with diagnostic ablation studies (Phase 3).

### The Full Pipeline
```mermaid
graph LR
    A[Raw News Corpus] --> B[Feature Engineering]
    B --> C[S-BERT / LDA Topics]
    C --> D[ARIMA Linear Trend]
    D --> E[BiLSTM Residual Correction]
    E --> F[Hybrid Forecast]
    F --> G[SHAP & Attention XAI]
    G --> H[Intelligence Dashboard]
```

---

## Core Technical Pillars

### 1. Semantic Evolution Tracking
Utilizes **BERTopic** (Sentence-BERT + UMAP + HDBSCAN) to capture nuanced shifts in news vocabulary and narrative clusters with state-of-the-art precision.

### 2. Multivariate Temporal Forecasting
A **Bidirectional LSTM (BiLSTM)** architecture designed for high-dimensional time-series forecasting, predicting topic proportions across complex temporal slices.

### 3. Hybrid Innovation (Phase 3)
A **synergistic ARIMA + BiLSTM** hybrid model where:
- **ARIMA(2,1,2)** captures linear trends and seasonal patterns (ML component)
- **BiLSTM** learns non-linear residual dynamics that ARIMA misses (DL component)
- **Additive Fusion**: `y_hybrid = y_ARIMA + residual_BiLSTM`

> [!IMPORTANT]
> Full architecture diagram with tensor shapes available at **[Phase 3 Architecture](docs/phase3_architecture.md)**.

### 4. Explainable AI (XAI)
Transparency-first design using **SHAP**, **Temporal Attention Maps**, and **Gradient Saliency** to validate model decisions and reveal the semantic triggers behind trend shifts.

### 5. Diagnostic Ablation Studies (Phase 3)
Rigorous component-level analysis proving the **necessity** of the hybrid architecture:
- ML-Only (ARIMA) vs DL-Only (BiLSTM) vs Hybrid across multiple topics
- Exact % degradation when each component is removed
- Statistical justification that the whole > sum of parts

---

## Mathematical Rigor
This project is built on first-principles engineering. Comprehensive mathematical derivations are documented in our **[Foundations Guide](docs/mathematical_foundations.md)**.

> [!NOTE]
> **Key Highlights:**
> - **Manifold Learning**: Topological preservation via Fuzzy Simplicial Sets.
> - **Gradient Flow**: Solving vanishing gradients through Forget Gate Calculus.
> - **Optimization**: Analysis of the Loss Landscape $(\mathcal{J}(\theta))$ and Hessian-based generalization.

---

## Repository Roadmap

```text
├── backend/               # FastAPI Analytics Service + Dockerfile
├── frontend/              # Next.js Intelligence Dashboard + Dockerfile
├── data/                  # Processed news & topic embeddings
│   ├── raw/               # Original News_Category_Dataset_v3.json
│   └── processed/         # processed_featured_data.csv
├── docs/                  # SOTA Reviews, Math Foundations & Architecture
│   ├── dl-sota-literature-review.md
│   ├── mathematical_foundations.md
│   └── phase3_architecture.md    # [NEW] Publication-ready diagrams
├── notebook-Phase-1/      # EDA, Feature Engineering, Baseline Models
├── notebook-Phase-2/      # BERTopic, LSTM Forecasting, XAI
├── notebook-Phase-3/      # [NEW] Hybrid Forecasting & Ablation Studies
│   ├── Hybrid_Forecasting.ipynb
│   ├── Ablation_Studies.ipynb
│   └── requirements.txt
├── project-reports/       # Phase 1 & 2 PDF reports
├── docker-compose.yml     # [NEW] One-command platform launch
├── start.sh               # Shell-based launch script
└── README.md
```

---

## Phase Summary

### Phase 1: Data Processing & Feature Engineering
- Raw news corpus cleaning and text preprocessing
- Temporal feature extraction (year, month, day of week)
- VADER sentiment analysis (neg, neu, pos, compound)
- Text statistics (word count, character count)

### Phase 2: Deep Learning & Topic Modeling
- **BERTopic**: Semantic topic discovery using S-BERT + UMAP + HDBSCAN
- **BiLSTM**: Bidirectional LSTM for temporal trend forecasting
- **XAI**: SHAP values, Temporal Attention Maps, Gradient Saliency

### Phase 3: Hybrid Innovation & Ablation
- **Hybrid Model**: Synergistic ARIMA + BiLSTM (neuro-statistical decomposition)
- **Ablation Studies**: Diagnostic analysis across 5 topics with 4 metrics (RMSE, MAE, R2, MAPE)
- **Architecture Docs**: Publication-ready Mermaid diagrams with tensor shapes
- **Dockerization**: Turn-key reproducibility via `docker-compose`

---

## Intelligence Platform

The project features a premium, full-stack intelligence platform for real-time visualization and forecasting.

### Key Features
- **Dashboard**: High-level KPIs and thematic evolution charts.
- **Predictive Analytics**: Interactive LSTM forecasts for emerging themes.
- **Event Explorer**: Deep dive into individual news reports with sentiment tracking.
- **Glassmorphic UI**: State-of-the-art dark mode design for optimal data clarity.

### Quick Start

**Option 1: Docker (Recommended)**
```bash
docker-compose up --build
```

**Option 2: Manual**
```bash
chmod +x start.sh
./start.sh
```

- **Dashboard**: `http://localhost:3000`
- **Analytics API**: `http://localhost:8000`

---

## Documentation

| Document | Description |
|---|---|
| [Phase 3 Architecture](docs/phase3_architecture.md) | Publication-ready pipeline & fusion diagrams |
| [Mathematical Foundations](docs/mathematical_foundations.md) | First-principles derivations (Linear Algebra, Calculus, Optimization) |
| [SOTA Literature Review](docs/dl-sota-literature-review.md) | Deep Learning state-of-the-art in topic modeling & event detection |
| [Phase 1 Report](project-reports/) | Data processing & EDA findings |
| [Phase 2 Report](project-reports/) | Deep Learning implementation & results |

---

> [!TIP]
> For a deep dive into the state-of-the-art, see our **[SOTA Literature Review](docs/dl-sota-literature-review.md)**.
