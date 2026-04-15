# Dynamic Trend & Event Detector

**Automated Detection and Tracking of Evolving Topics in News Media Using Probabilistic Topic Modeling, Temporal Feature Engineering, and Embedding-Based Clustering**

## Problem Statement

The exponential growth of digital news platforms generates an unprecedented volume of textual data daily. Organizations in journalism, policy-making, and social media analytics require automated systems that can:

1. **Detect emerging narratives** and breaking events from large news corpora.
2. **Track how topics evolve** over time — not just identify static topic clusters.
3. **Distinguish event-driven topic spikes** from seasonal or unrelated fluctuations.

Existing methods fail on one or more of these requirements:
- **Static topic models** (e.g., standard LDA) treat the entire corpus as a single time slice and cannot capture temporal evolution.
- **Keyword-based methods** rely on high-frequency terms, producing vague event descriptions that miss evolving semantics.
- **Pure embedding methods** (e.g., BERTopic alone) achieve high coherence but cannot separate event-driven changes from seasonal noise.

This project addresses these gaps by building a **multi-model temporal topic modeling pipeline** that combines probabilistic topic modeling with domain-specific trend features — all derived directly from the corpus without requiring external data sources.

## Approach & Methodology

We employ a structured **three-model ablation pipeline** to systematically evaluate increasingly sophisticated approaches:

| Phase | Model | Method | Purpose |
|-------|-------|--------|---------|
| **A** | TF-IDF Top-K | Frequency-based keyword extraction | Baseline lower bound — no topic structure |
| **B** | LDA (Latent Dirichlet Allocation) | Probabilistic generative topic model | Advanced ML — discovers latent topic distributions |
| **C** | BERTopic | BERT embeddings + UMAP + HDBSCAN | Deep Learning — semantic topic discovery |

Each model is evaluated using **coherence score (c_v)** and **perplexity** — the standard metrics for unsupervised topic models where classification metrics (F1, accuracy) do not apply.

**Key methodological contributions:**
- **Monthly temporal slicing** to track topic evolution across the 10-year corpus (2012–2022).
- **Article velocity** — a domain-specific feature capturing trend momentum (articles per category per time window).
- **Headline novelty score** — quantifies breaking-news vocabulary by measuring the inverse document frequency of words appearing in short time windows.
- All temporal signals are derived **directly from the corpus**, unlike approaches that require external data (e.g., Google Trends).

### Theoretical Foundation

The LDA model operates on the following generative process:

- `θ_d ~ Dir(α)` — Document-topic distribution (Dirichlet prior)
- `β_k ~ Dir(η)` — Topic-word distribution (Dirichlet prior)
- `z_dn ~ Mult(θ_d)` — Topic assignment per word (Multinomial)
- `w_dn ~ Mult(β_z)` — Word drawn from assigned topic (Multinomial)

Inference is performed via the **EM algorithm (variational Bayes)**. We acknowledge the **i.i.d. assumption violation** inherent in LDA when applied to temporally structured data and mitigate it via monthly slicing — an approximation of Dynamic Topic Models (Blei & Lafferty, 2006) at lower computational cost.

## Dataset

**HuffPost News Category Dataset v3** (Misra, 2022)

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) |
| **Total Articles** | 209,527 |
| **Categories** | 42 unique news categories |
| **Time Span** | January 2012 – September 2022 |
| **Format** | JSON (one record per line) |
| **Fields** | `link`, `headline`, `category`, `short_description`, `authors`, `date` |

### Dataset Characteristics (from EDA)

- **Non-Stationarity:** Article volume fluctuates significantly around major events (US Elections 2016/2020, COVID-19 2020, MeToo Movement 2017).
- **Coverage Drop:** 93.7% decrease in article volume post-2017, indicating the model is most robust on 2012–2017 data.
- **Category Imbalance:** 35.1× ratio between the most frequent category (POLITICS) and the least (EDUCATION).
- **Text Statistics:**
  - Headline length: ~9.6 words (approximately normal distribution).
  - Description length: ~19.7 words (right-skewed, skewness = 1.47).
  - *Hapax legomena* (words appearing only once): 45.7% — indicating rich, diverse vocabulary.

## Project Structure

```
Dynamic-Trend-Event-Detector/
├── README.md                          # This file
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── News_Category_Dataset_v3.json    # Raw dataset (209,527 articles)
│   └── processed/
│       ├── processed_news.csv               # Intermediate processed data
│       └── featured_news.parquet            # Efficient optimized processed data
│
├── notebook/
│   ├── EDA.ipynb                      # Exploratory Data Analysis
│   ├── Model.ipynb                    # Model training & evaluation (TF-IDF, LDA, BERTopic)
│   └── feature_engineering.py         # Feature engineering pipeline script
│
├── docs/
│   └── literature-review.md           # Comprehensive literature review
│
└── research-papers/
    ├── A_Survey_on_Event_Tracking_in_Social_Media_Data_Streams.pdf
    └── Explainable_Topic_Modeling_for_Tracking_User_Interests_Related_to_Social_Events.pdf
```

## Setup & Installation

### Prerequisites

- Python 3.8+
- [Google Colab](https://colab.research.google.com/)

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn gensim pyLDAvis nltk tqdm
```

For BERTopic (Model C, GPU recommended):
```bash
pip install bertopic
```

### Running the Pipeline

1. **Clone the repository:**
   ```bash
   git clone https://github.com/meenaksh06/Dynamic-Trend-Event-Detector.git
   cd Dynamic-Trend-Event-Detector
   ```

2. **Data Preprocessing & EDA:**
   Open `notebook/EDA.ipynb` in Google Colab or Jupyter Notebook and run all cells. This performs:
   - Data loading and inspection
   - Missing value analysis and handling
   - Temporal trend visualization with event annotations
   - Category distribution analysis
   - Text characteristics analysis (word count, vocabulary richness)

3. **Feature Engineering:**
   Run the feature engineering script:
   ```bash
   python notebook/feature_engineering.py
   ```
   This extracts:
   - **Temporal features:** year, month, day_of_week
   - **Text length features:** word_count, char_count
   - **Sentiment features:** VADER sentiment scores (neg, neu, pos, compound)

4. **Model Training & Evaluation:**
   Open `notebook/Model.ipynb` in Google Colab (GPU runtime recommended) and run all cells to train and evaluate:
   - Model A: TF-IDF Baseline
   - Model B: LDA with hyperparameter tuning
   - Model C: BERTopic (optional, requires GPU)

## Pipeline Overview

### Phase 1: Data Preprocessing & EDA

Comprehensive exploratory analysis covering:

- **Data Quality:** Identification and handling of missing values (6 missing headlines, 9.41% missing descriptions, 17.86% missing authors).
- **Temporal Analysis:** Monthly article volume visualization with annotations for major world events (US Elections, COVID-19, MeToo Movement).
- **Distribution Analysis:** Category frequency distributions, text length distributions with statistical moments (mean, std, skewness, kurtosis).
- **Stationarity Assessment:** Identification of regime shifts and non-stationary behavior in article volume time series.

### Phase 2: Feature Engineering

Engineered features designed to capture trend signals:

| Feature | Type | Description |
|---------|------|-------------|
| `year` | Temporal | Year extracted from article date |
| `month` | Temporal | Month extracted from article date |
| `day_of_week` | Temporal | Day of week (0=Monday, 6=Sunday) |
| `word_count` | Text | Number of words in cleaned text |
| `char_count` | Text | Number of characters in cleaned text |
| `sentiment_neg` | Sentiment (VADER) | Negative sentiment score |
| `sentiment_neu` | Sentiment (VADER) | Neutral sentiment score |
| `sentiment_pos` | Sentiment (VADER) | Positive sentiment score |
| `sentiment_compound` | Sentiment (VADER) | Compound sentiment score (-1 to +1) |

### Phase 3: Model Application

Three-model structured ablation study with increasing complexity.

## Models

### Model A — TF-IDF Baseline

- **Method:** Term Frequency–Inverse Document Frequency (TF-IDF) with top-K keyword extraction per monthly time slice.
- **Parameters:** `max_features=3000`, `ngram_range=(1, 2)`
- **Purpose:** Lower bound baseline. Surfaces event keywords but **cannot group them into coherent topics**.
- **Limitation:** No topic structure, ignores word co-occurrence semantics.

### Model B — LDA (Advanced ML)

- **Method:** Latent Dirichlet Allocation via Gensim.
- **Hyperparameter Tuning:** Systematic search over `k ∈ {10, 15, 20, ..., 50}` using coherence score (c_v).
- **Vocabulary Filtering:** `no_below=15, no_above=0.5` to remove rare and overly common terms.
- **Inference:** 5-pass variational Bayes with automatic α and η optimization.
- **Evaluation:**
  - **Coherence (c_v):** PMI-based semantic similarity of top words per topic.
  - **Perplexity:** Model's ability to generalize to unseen data.
- **Strength:** Provides probabilistic interpretation of topic assignments, enabling temporal tracking of topic distributions.

### Model C — BERTopic (Deep Learning)

- **Method:** Sentence-BERT embeddings → UMAP dimensionality reduction → HDBSCAN density-based clustering → class-based TF-IDF (c-TF-IDF) for topic representation.
- **Purpose:** Captures deep semantic relationships that TF-IDF and LDA miss.
- **Evaluation:** Coherence comparison against LDA baseline.
- **Advantage:** Higher coherence scores on news corpora (Grootendorst, 2022).

## Results & Evaluation

### Evaluation Metrics

| Metric | Description | Applicable To |
|--------|-------------|--------------|
| **Coherence (c_v)** | PMI-based semantic similarity of top words in each topic | LDA, BERTopic |
| **Perplexity** | Negative log-likelihood of held-out documents | LDA |

### Structured Ablation

| Model | Method | Topic Structure | Temporal Tracking | Semantic Depth |
|-------|--------|:-:|:-:|:-:|
| A — TF-IDF | Frequency baseline | ✗ | ✗ | ✗ |
| B — LDA | Probabilistic model | ✓ | ✓ (via monthly slicing) | Partial |
| C — BERTopic | Neural embeddings | ✓ | ✓ | ✓ |

The ablation shows progressive improvement from simple frequency methods to probabilistic models to deep learning approaches, justifying the added complexity at each stage.

## Key Findings

1. **Topic Non-Stationarity:** Topic distributions shift dramatically around major world events, validating temporal slicing as a necessary modeling choice.
2. **Event Detection via Velocity:** Article velocity (volume per category per time window) reliably captures trend momentum for event-driven topics.
3. **Headline Novelty:** Rare words appearing in short time windows (e.g., "omicron" in late 2021, "impeachment" in 2019) provide strong breaking-news signals.
4. **LDA vs BERTopic:** BERTopic achieves higher coherence than LDA but lacks probabilistic interpretability — the hybrid approach combines strengths of both.
5. **Self-Contained Temporal Signals:** All temporal features derived from the corpus itself, without requiring external data sources.

## Literature Review

A comprehensive literature review is available in [`docs/literature-review.md`](docs/literature-review.md), covering:

1. **Probabilistic Topic Models:** LDA (Blei et al., 2003), Dynamic Topic Models (Blei & Lafferty, 2006), Topics over Time (Wang & McCallum, 2006).
2. **Neural & Embedding-Based Approaches:** BERT (Devlin et al., 2019), BERTopic (Grootendorst, 2022), BERTrend (Balagopalan et al., 2024), GLIPCA + RAG (Le et al., 2026).
3. **Event Tracking in Social Media:** Survey by Han et al. (2024) identifying key failure modes in existing methods.
4. **Direct Dataset Benchmarks:** Rajan et al. (2024) benchmarking LDA, BERTopic, NMF, and Top2Vec on this exact HuffPost dataset — but only statically, without temporal analysis.
5. **Research Gaps:** Explicit identification of 6 limitations in prior work and how this project addresses each.

## Research Papers

The following research papers (included in `research-papers/`) informed this project:

1. **Han et al. (2024)** — *A Survey on Event Tracking in Social Media Data Streams.* Big Data Mining and Analytics, 7(1), 217–243.
2. **Le et al. (2026)** — *Explainable Topic Modeling for Tracking User Interests Related to Social Events.* IEEE Access, 14, 36548–36563.

## Contributors

| Contributor | Role | Key Contributions |
|------------|------|-------------------|
| **Meenaksh** | Data & Feature Engineering | Dataset curation, data preprocessing, feature engineering pipeline |
| **Abhishek** | Modeling & Analysis | EDA, model implementation (TF-IDF, LDA, future scope (BERTopic)), literature review, research paper curation |

## References

1. Blei, D.M., Ng, A.Y., Jordan, M.I. (2003). *Latent Dirichlet Allocation.* JMLR, 3, 993–1022.
2. Blei, D.M., Lafferty, J.D. (2006). *Dynamic Topic Models.* ICML 2006.
3. Wang, X., McCallum, A. (2006). *Topics over Time.* ACM SIGKDD.
4. Devlin, J. et al. (2019). *BERT.* NAACL 2019.
5. Grootendorst, M. (2022). *BERTopic.* arXiv:2203.05794.
6. Balagopalan, A. et al. (2024). *BERTrend.* ACL FutureD Workshop.
7. Han, Z. et al. (2024). *A Survey on Event Tracking in Social Media Data Streams.* Big Data Mining and Analytics, 7(1), 217–243.
8. Rajan, A. et al. (2024). *Comparative Study of Topic Modelling on News Articles.* Springer DaSET.
9. Le, H.H., Harakawa, R., Iwahashi, M. (2026). *Explainable Topic Modeling for Tracking User Interests.* IEEE Access, 14, 36548–36563.
10. Yao, Z. et al. (2018). *Dynamic Word Embeddings.* WSDM 2018.
11. Misra, R. (2022). *News Category Dataset.* [doi:10.48550/arXiv.2209.11429](https://doi.org/10.48550/arXiv.2209.11429).

## License

This project is developed for academic purposes. The HuffPost News Category Dataset is publicly available on Kaggle under its respective license.
