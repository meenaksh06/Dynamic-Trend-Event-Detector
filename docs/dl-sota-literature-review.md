# Deep Learning State of the Art (SOTA) Literature Review

## 1. Introduction: The Deep Learning Shift in Topic Modeling and Event Tracking

The automated detection and tracking of evolving topics in large news corpora has traditionally been dominated by probabilistic models like Latent Dirichlet Allocation (LDA) (Blei et al., 2003) and its temporal variant, Dynamic Topic Models (DTM) (Blei & Lafferty, 2006). While robust, these methods rely on bag-of-words assumptions, ignoring the semantic context of vocabulary, which is particularly limiting when analyzing fast-moving, nuanced news streams and breaking events.

In recent years, the State of the Art (SOTA) has shifted decisively toward Deep Learning (DL)—specifically, self-attention mechanisms and Transformer architectures. By mapping words and sentences to dense semantic vector spaces, DL models capture syntactic structures and contextual meaning. However, applying DL to *dynamic event tracking over time* remains an open challenge.

This comprehensive review traces the lineage of Deep Learning in topic and event detection, evaluating the strengths and limitations of current SOTA methods. We conclude by positioning our multi-model temporal methodology (combining probabilistic temporal slicing, embedding-based clustering, and corpus-derived velocity features) as the logical next step to address the unresolved gap in distinguishing semantic event signals from seasonal noise.

## 2. The Foundation: Contextual Embeddings and Neural Representations

The transition to Deep Learning in NLP was catalyzed by word embeddings (Word2Vec, GloVe), which Yao et al. (2018) successfully made dynamic to track semantic shifts over time. However, these embeddings were static for a given word.

The breakthrough for topic modeling came with **BERT (Devlin et al., 2019)** and **Sentence-BERT (Reimers & Gurevych, 2019)**. These models generate *contextual embeddings*, where the semantic representation of a word depends on its surrounding sentence. In news media analysis, this contextualization is crucial: the word "bank" in a financial crisis narrative is correctly isolated from "bank" in a geographic context without relying heavily on co-occurrence probabilities alone.

While Transformers provided superior representations, using them directly for document clustering posed dimensionality challenges, leading to the development of specialized neural topic models.

## 3. High-Coherence Semantic Clustering: Top2Vec and BERTopic

The current undisputed SOTA for *static* semantic topic discovery is clustering-based embedding approaches. 

**Top2Vec (Angelov, 2020)** pioneered this space by jointly embedding documents and words, using UMAP for dimensionality reduction and HDBSCAN for density-based clustering. It bypassed the need for pre-defined topic counts (unlike LDA) and generated highly coherent topic representations.

**BERTopic (Grootendorst, 2022)** advanced this architecture by decoupling the embedding phase from clustering and introducing class-based TF-IDF (c-TF-IDF). This allowed BERTopic to extract highly representative terms for clusters. As benchmarked by **Rajan et al. (2024)** explicitly on the HuffPost dataset, BERTopic achieves significantly higher topic coherence (c_v) than LDA and NMF. It captures deep semantic relationships that traditional probabilistic models miss.

**Limitation (The Temporal Gap):** Both Top2Vec and standard BERTopic are fundamentally static. When applied to 10 years of longitudinal news data, they compress events (like the 2016 and 2020 US Elections) into single clusters, failing to track the momentum, birth, propagation, and death of real-world narratives as they happen in time.

## 4. Deep Learning for Temporal and Event Tracking

To address the temporal gap, researchers have attempted to force DL models into dynamic frameworks. 

**Dynamic BERTopic** and variations like **BERTrend (Balagopalan et al., 2024)** attempt to track topic signal strength by applying attention-based weighting across sliding time windows. These approaches work well for coarse-grained trends but struggle to identify breaking events. They fail to explicitly separate an "event-driven topic spike" (e.g., a sudden pandemic) from a "seasonal fluctuation" (e.g., annual holiday news).

Furthermore, as highlighted in a recent survey on event tracking in data streams by **Han et al. (2024)**, while deep learning excels at representation, it often obscures the *mechanisms* of event propagation. Neural networks treat semantic spaces robustly but discard specific domain signals—like the sudden acceleration of article publication (velocity) or the introduction of uniquely novel vocabulary (hapax legomena)—that journalism analytics rely upon.

## 5. Emerging SOTA: Hybrid Architectures and LLMs

The bleeding edge of research has recognized that purely unsupervised clustering of embeddings is insufficient for accurate event tracking. 

**Le et al. (2026)** introduced **GLIPCA + RAG**, a methodology attempting "Explainable Topic Modeling." They utilized Graphical Lasso-guided PCA to filter noise from event-related trends, supplemented by Retrieval-Augmented Generation (RAG) using Large Language Models to interpret and label the clustered events. While achieving high purity scores, their methodology introduces massive computational overhead and relies heavily on external data anchors (like Google Trends) to guide the temporal signals.

This introduces a critical constraint for production systems: reliance on external continuous data streams or massive LLM calls per document breaks down when processing hundreds of thousands of rapidly streaming news articles.

## 6. Identifying the Research Gap

From this lineage—from static probabilistic models to highly coherent but static deep learning (BERTopic), and finally to external-data-dependent LLM trackers (GLIPCA+RAG)—three fundamental gaps persist:

1. **The Inability to Filter Noise Dynamically:** High-coherence DL models struggle to separate event-driven spikes from regular temporal drifts.
2. **Missing Domain Features:** Black-box neural models ignore critical metadata (time, velocity, novelty) inherently present in the news stream.
3. **Reliance on External Data:** SOTA temporal models often require external signals (like search trends) to align their topics over time.

## 7. Our Position in the Research Lineage: The Logical Next Step

This project is meticulously positioned to bridge these exact gaps. We view the integration of **probabilistic temporal tracking**, **Deep Learning semantic clustering**, and **corpus-derived signal processing** as the necessary, logical next step in SOTA.

Instead of discarding probabilistic structuring or relying on external foundational models for temporal alignment, we propose a **structured multi-model pipeline**:

1. **Temporal Slicing & Feature Engineering:** We extract domain-specific, corpus-internal signals—*Article Velocity* (capturing trend momentum) and *Headline Novelty Scores* (capturing breaking vocabulary). This handles the mechanics of event momentum without external data (addressing the gap in Le et al., 2026).
2. **Phase-Ablated Benchmarking:** We apply LDA over monthly slices to approximate Dynamic Topic Models, providing a probabilistic foundation of how underlying topic distributions shift over time.
3. **Semantic Refinement (BERTopic):** Finally, we integrate BERTopic (Grootendorst, 2022) precisely where DL shines: semantic clustering. But we condition its input based on our engineered temporal slices.

By separating the **mechanism of time** (handled via dataset slicing and velocity features) from the **mechanism of meaning** (handled via Sentence-BERT and HDBSCAN), our approach resolves the static limitation of BERTopic while avoiding the external-data dependency and computational unscalability of LLM-based event trackers. 

This project establishes that state-of-the-art event detection does not require more monolithic, end-to-end neural networks; rather, it requires integrating Deep Learning representations with domain-aware temporal architectures.
