# NLP-Homework-02-Word-Embeddings

## Overview

This notebook explores word embeddings through training, comparison, bias analysis, and text classification. Models are trained on Wikipedia data and evaluated against pre-trained GloVe and Word2Vec embeddings across multiple NLP tasks.


## Tasks

### 2.2 — Training Word Embeddings
Two Word2Vec models are trained from scratch on the Simple English Wikipedia dataset (`wikimedia/wikipedia`, `20231101.simple`):
- **CBOW** (Continuous Bag of Words) — `sg=0`
- **Skip-gram** — `sg=1`

Both models use `vector_size=100`, `window=5`, and `min_count=5`. Models are saved to disk with an `os.path.exists` check to avoid retraining on reruns.

### 2.3 — Comparing Word Embeddings
The trained models are compared against pre-trained embeddings:
- `GloVe-Wiki-Gigaword-100`
- `GloVe-Wiki-Gigaword-300`
- `Word2Vec-Google-News-300`

Comparisons are made using similarity queries (e.g. *guitar*, *actor*) and analogy queries (e.g. *Ankara : Germany :: Turkey : ?*).

### 2.4 — Bias in Word Embeddings
Religious bias is measured using the **WEAT** (Word Embedding Association Test) metric from the [WEFE](https://wefe.readthedocs.io/) library. The test compares associations between:
- **Target sets:** Christianity-related words vs. Islam-related words
- **Attribute sets:** Positive words vs. Negative words

All four models (CBOW, Skip-gram, GloVe-100, GloVe-300) are evaluated. The `lost_vocabulary_threshold` is set to `0.5` to handle out-of-vocabulary words.

### 2.5 — Text Classification: Sparse vs. Dense
Sentiment classification is performed on the [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) dataset using two approaches:

| Model | Representation |
|-------|---------------|
| **Model A** | TF-IDF (sparse, `max_features=50000`) + Logistic Regression |
| **Model B** | Average word embeddings (dense) + Logistic Regression |

Model B is evaluated with all four embedding models. Results are compared using `classification_report`.

---

## Dependencies

```bash
pip install datasets gensim nltk wefe scikit-learn numpy
```

---

## Dataset
- **Training:** [Simple English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) — `20231101.simple`
- **Classification:** [TweetEval Sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval)

---
