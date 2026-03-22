# Literature Review Summary

## 1. Introduction

The exponential growth of digital news platforms and social media has led to an
unprecedented volume of textual information being generated daily. Organizations
in journalism, policy-making, and social media analytics need automated systems
to detect emerging narratives, track evolving topics, and identify breaking events
from large news corpora. Understanding how topics evolve over time is essential
for analyzing what drives public discourse, societal shifts, and information trends.

**Blei et al. (2003)** established that documents can be modeled as probabilistic
mixtures of topics using Latent Dirichlet Allocation (LDA), providing the
foundational framework for automated topic discovery in large text collections.
**Han et al. (2024)** emphasize that event tracking in social media involves three
critical steps — event detection, event propagation, and event evolution — and
that existing topic-based methods fail to capture multi-dimensional event
characteristics. **Le et al. (2026)** demonstrate that combining temporal trend
analysis with topic modeling enables the tracking of user interests before, during,
and after social events.

This project builds on those insights by using the **HuffPost News Category Dataset
(Misra, 2022)** — 209,527 articles across 42 categories spanning 2012–2022 — to
detect and track evolving topics over time using a pipeline of probabilistic topic
modeling, temporal feature engineering, and embedding-based clustering.

## 2. Probabilistic Topic Models and Baseline Methods

**Blei et al. (2003)** introduced LDA as a generative probabilistic model where
each document is a mixture of topics and each topic is a distribution over words.
The model uses Dirichlet priors (α for document-topic, η for topic-word) and
performs inference via the Expectation-Maximization (EM) algorithm.

**Blei and Lafferty (2006)** extended LDA to Dynamic Topic Models (DTM), which
model topic evolution over time using Kalman filtering — allowing topic vocabulary
to change across time slices. **Wang and McCallum (2006)** proposed Topics over
Time (ToT), treating timestamps as observed variables drawn from Beta distributions
parameterized by topic mixtures.

### Key Insight from Literature
Static topic models (LDA) cannot capture how topics evolve — they treat the entire
corpus as a single time slice. As noted by **Han et al. (2024)**, traditional
topic model-based event detection requires manual configuration of prior parameters
and relies on high-frequency keywords, resulting in vague descriptions that miss
evolving event semantics.

### Application in Our Project
We use:
- **TF-IDF top-K** as Model A (frequency baseline — no topic structure)
- **LDA with tuned k** as Model B (Advanced ML — probabilistic topic model)
- Monthly temporal slicing to approximate DTM's time-based topic evolution
- Coherence (c_v) and perplexity as evaluation metrics — standard for unsupervised
  topic models where classification metrics like F1 do not apply

## 3. Neural and Embedding-Based Approaches

**Devlin et al. (2019)** introduced BERT, which produces contextual embeddings
where the same word has different representations depending on context — a
fundamental improvement over TF-IDF's context-free bag-of-words representation.

**Grootendorst (2022)** proposed BERTopic, combining sentence-BERT embeddings,
UMAP dimensionality reduction, and HDBSCAN density-based clustering. A class-based
TF-IDF (c-TF-IDF) extracts representative words per cluster. BERTopic consistently
achieves higher coherence than LDA on news corpora.

**Balagopalan et al. (2024)** extended BERTopic with temporal trend tracking
(BERTrend), applying it to sliding time windows to measure topic signal strength.
However, BERTrend does not provide a probabilistic interpretation of trend
confidence and cannot separate event-related topic fluctuations from unrelated
seasonal variations.

**Le et al. (2026)** proposed GLIPCA + RAG for explainable topic modeling,
combining graphical lasso-guided PCA to separate event-related trends from noise,
with Retrieval-Augmented Generation (RAG) to reduce LLM hallucination in topic
labeling. Their method outperformed BERTopic on Purity (0.5444 vs 0.2679) for
event-related topic extraction.

### Key Insight from Literature
As **Le et al. (2026)** demonstrate, BERTopic alone cannot separate topic
fluctuations caused by social events from unrelated temporal variations. A
dedicated temporal component is essential for event-driven topic tracking.

### Application in Our Project
We use:
- **BERT-based sentence embeddings** as the foundation for Phase 3 BERTopic
- **BERTopic** as Model C (Deep Learning) for semantic topic discovery
- **Monthly temporal slicing** as our temporal component — derived directly from
  the corpus, requiring no external data source (unlike Le et al. who use Google Trends)
- Phase 3 hybrid architecture: LDA probabilistic assignments + BERTopic semantic
  embeddings + temporal evolution logic

## 4. Applied Event Tracking and News Analytics

**Han et al. (2024)** provide a comprehensive survey of event detection,
propagation, and evolution in social networks. They identify two critical
failure modes in existing topic-based event detection:

1. Methods rely on high-frequency keywords and ignore multi-dimensional
   event characteristics, leading to vague topic descriptions
2. Methods treat all topic keywords as independent features, ignoring
   correlations — making it difficult to explore deep semantic expression
   mechanisms of major events

The survey categorizes topic models into standard (LDA, PLSA), clustering-based
(BTM, GSDMM), self-aggregating, and deep learning-based methods — positioning
LDA as the established baseline and deep learning hybrids as the frontier.

**Rajan et al. (2024)** benchmark LDA, BERTopic, NMF, and Top2Vec on the
**exact same HuffPost dataset** used in this project. Key findings: BERTopic
achieves highest coherence, LDA is most interpretable, NMF is fastest. However,
their evaluation is entirely **static** — no temporal slicing, no event-driven
analysis, no trend tracking across the 10-year span.

**Yao et al. (2018)** demonstrate that word semantics shift over time using
dynamic word embeddings aligned via orthogonal Procrustes — showing words like
"cell" and "gay" change meaning dramatically over decades.

### Key Insight from Literature
**Rajan et al. (2024)** evaluated these models on our dataset statically.
**Han et al. (2024)** identify the keyword-only limitation of existing methods.
Neither work introduces domain-specific trend features to capture momentum
and novelty signals.

### Application in Our Project
We use:
- **Rajan et al. (2024)** as our quantitative baseline — our LDA coherence
  (0.3354) is directly comparable to their benchmark results
- **Han et al. (2024)**'s identified limitations to justify our two original
  features: **article velocity** (captures trend momentum) and **headline
  novelty score** (captures breaking-news vocabulary)
- **Yao et al. (2018)** to motivate the novelty score — rare words in global
  corpus appearing in a short time window signal semantic novelty (e.g.,
  "omicron" in late 2021, "impeachment" in 2019)

## 5. Research Gaps Identified in Prior Work

| **Limitation in Prior Work** | **How Our Project Addresses It** |
|------------------------------|----------------------------------|
| Rajan et al. (2024) evaluate LDA/BERTopic statically — no temporal slicing | We apply monthly temporal slicing to track topic evolution 2012–2022 |
| Han et al. (2024) identify keyword-only methods produce vague event descriptions | We add article velocity + novelty score as domain-specific trend features |
| Le et al. (2026) require external Google Trends data for temporal signals | We derive all temporal signals directly from the news corpus itself |
| Le et al. (2026) operate on Vietnamese single-language corpus only | We apply to English multi-category HuffPost corpus with 42 categories |
| BERTopic (Grootendorst 2022) cannot separate event vs seasonal topic shifts | Our LDA temporal slicing + velocity features explicitly model event-driven spikes |
| Blei 2003 LDA assumes static topics — i.i.d. documents | We acknowledge this assumption violation and mitigate via monthly slicing |

## 6. Our Contribution

| **Previous Studies** | **Our Study (This Project)** |
|----------------------|------------------------------|
| LDA evaluated on static corpora (Blei 2003, Rajan 2024) | LDA applied with monthly temporal slicing on 10-year news corpus |
| Event tracking surveys identify keyword-only limitations (Han 2024) | Article velocity + novelty score features address this limitation empirically |
| BERTopic achieves high coherence but no temporal structure (Grootendorst 2022) | Phase 3 hybrid: BERTopic embeddings + LDA temporal assignments |
| Explainable temporal modeling requires external trend data (Le 2026) | All temporal signals derived from corpus — no external data needed |
| No ablation study on HuffPost with temporal + static comparison (Rajan 2024) | Structured ablation: TF-IDF baseline vs LDA vs Hybrid with coherence + perplexity |
| DTM models temporal evolution but is computationally expensive (Blei 2006) | Monthly slicing approximates DTM at lower computational cost in Phase 1 |

## 7. Conclusion

This literature review establishes the theoretical and empirical foundation for
dynamic trend and event detection in news media. By combining the probabilistic
topic modeling framework of **Blei et al. (2003)**, the temporal evolution
motivation of **Blei and Lafferty (2006)**, the embedding-based improvements
of **Grootendorst (2022)**, the event tracking context of **Han et al. (2024)**,
the recent explainable temporal modeling of **Le et al. (2026)**, and the direct
dataset benchmarks of **Rajan et al. (2024)**, this study aims to:

- Detect and track evolving topics month-by-month across 10 years of HuffPost news
- Identify event-driven spikes using article velocity and headline novelty features
- Compare probabilistic baseline (LDA) against deep learning (BERTopic) in a
  structured ablation study
- Provide a reproducible temporal topic modeling pipeline on English multi-category
  news data

This approach ensures both **academic grounding** in established topic modeling
literature and **practical contribution** by filling the temporal analysis gap
identified across all surveyed works.

## References

1. Blei, D.M., Ng, A.Y., Jordan, M.I. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993–1022.
2. Blei, D.M., Lafferty, J.D. (2006). Dynamic Topic Models. *ICML 2006*.
3. Wang, X., McCallum, A. (2006). Topics over Time. *ACM SIGKDD*.
4. Devlin, J. et al. (2019). BERT. *NAACL 2019*.
5. Grootendorst, M. (2022). BERTopic. *arXiv:2203.05794*.
6. Balagopalan, A. et al. (2024). BERTrend. *ACL FutureD Workshop*.
7. Han, Z. et al. (2024). A Survey on Event Tracking in Social Media Data Streams. *Big Data Mining and Analytics*, 7(1), 217–243.
8. Rajan, A. et al. (2024). Comparative Study of Topic Modelling on News Articles. *Springer DaSET*.
9. Le, H.H., Harakawa, R., Iwahashi, M. (2026). Explainable Topic Modeling for Tracking User Interests. *IEEE Access*, 14, 36548–36563.
10. Yao, Z. et al. (2018). Dynamic Word Embeddings. *WSDM 2018*.
