# Dynamic Trend & Event Detector 🚀

**An Advanced Pipeline for Semantic Topic Discovery, Temporal Trend Forecasting, and Deep Learning Model Interpretability.**

---

## 📈 Project Overview
The **Dynamic Trend & Event Detector** is a state-of-the-art system designed to identify, track, and forecast evolving news narratives. By combining probabilistic generative models (LDA) with contextual deep learning architectures (BERTopic & BiLSTM), the system provides high-fidelity insights into media trends and breaking events.

### 🌟 Key Advanced Features
- **Semantic Evolution Tracking**: Using Transformer-based embeddings to capture nuanced shifts in news vocabulary.
- **Multivariate Forecasting**: A Bidirectional LSTM (BiLSTM) architecture tuned to predict topic proportions across monthly temporal slices.
- **Explainable AI (XAI)**: A dedicated interpretability suite using **SHAP**, **Temporal Attention Maps**, and **Gradient Saliency** to validate model decisions.
- **Mathematical Rigor**: Model designs derived from first-principles **Linear Algebra** and **Multivariate Calculus**, with documented analysis of **Loss Landscape Geometry**.

---

## 🛠 Model Pipeline

The project follows a structured evolution of complexity, from statistical baselines to interpretable deep learning:

| Phase | Model | Core Technique | Purpose |
|-------|-------|----------------|---------|
| **A** | TF-IDF | Frequency extraction | Statistical Baseline |
| **B** | LDA | Variational Bayes (EM) | Latent Distribution Discovery |
| **C** | BERTopic | Sentence-BERT + UMAP | Deep Semantic Clustering |
| **D** | Forecast | BiLSTM + Attention | Temporal Pattern Prediction |

---

## 🔍 Model Interpretability & XAI
Ensuring reliability in event detection requires transparency. Our interpretability suite validates that models learn semantically valid features:

- **[SHAP (Shapley Additive Explanations)](notebook-Phase-2/Model_Interpretability.ipynb)**: Quantifies the global and local impact of historical topic counts on future forecasts.
- **[Temporal Attention Maps](notebook-Phase-2/Model_Interpretability.ipynb)**: Visualizes the internal "memory" of the sequence model, highlighting which lookback months drive the current prediction.
- **[Gradient Saliency](notebook-Phase-2/Model_Interpretability.ipynb)**: Uses partial derivatives to identify the specific words in headlines that trigger cluster assignments.

---

## 📐 Theoretical Foundations
We connect our technical implementation to academic first-principles. Detailed derivations are available in the **[Mathematical Foundations](docs/mathematical_foundations.md)** document.

### Highlights:
- **Optimization Geometry**: Analysis of the **Loss Landscape $(\mathcal{J}(\theta))$** and the role of the **Hessian Matrix** in finding flat vs. sharp minima for better generalization.
- **Forget Gate Calculus**: A mathematical proof of how LSTMs solve the vanishing gradient problem via identity bypasses.
- **Linear Algebra of Attention**: Deriving Query-Key-Value projections as dynamic transformations in high-dimensional vector spaces.

---

## 📂 Project Structure
```text
Dynamic-Trend-Event-Detector/
├── data/
│   └── processed/             # Featured news and topic embeddings
├── docs/
│   ├── literature-review.md   # SOTA analysis
│   └── mathematical_foundations.md # Derivations of LA & Calculus
├── models/                    # Saved weights (.h5, .safetensors)
├── notebook-Phase-2/
│   ├── Forecasting_LSTM.ipynb # Trend prediction pipeline
│   ├── Model_BERTopic.ipynb   # Semantic discovery
│   └── Model_Interpretability.ipynb # XAI (SHAP, Attention, Saliency)
└── app/                       # (Experimental) Visualization Dashboard
```

---

## 🚀 Setup & Installation

1. **Clone & Environment**:
   ```bash
   git clone https://github.com/meenaksh06/Dynamic-Trend-Event-Detector.git
   pip install -r requirements.txt
   ```

2. **Run Interpretability Analysis**:
   Generate the latest validation suite:
   ```bash
   python create_interpretability_nb.py
   ```

3. **Explore Foundations**:
   Review our theoretical discussion in `docs/mathematical_foundations.md`.

---

> [!TIP]
> Use the **Model_Interpretability.ipynb** notebook to verify "Sharp" vs "Flat" minima gradients on your own local loss surface simulations.

> [!IMPORTANT]
> This project requires a GPU for efficient BERTopic embedding generation and BiLSTM training.
