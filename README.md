# 🧠 Deep Learning for NLP — Tweet Classification

> A 3-part progression through NLP techniques for tweet sentiment/topic classification, from classical ML baselines to fine-tuned Transformer models. Developed for the **AI-2: Deep Learning for NLP** course competition.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Homework 1 — Classical ML Baseline](#homework-1--classical-ml-baseline)
- [Homework 2 — Neural Network with GloVe Embeddings](#homework-2--neural-network-with-glove-embeddings)
- [Homework 3 — Transformer Fine-tuning (BERT & DistilBERT)](#homework-3--transformer-fine-tuning-bert--distilbert)
- [Progression Summary](#progression-summary)
- [Setup & Requirements](#setup--requirements)

---

## Overview

Each homework builds on the previous, exploring increasingly powerful approaches to text classification on tweet data:

| Homework | Approach | Key Libraries |
|---|---|---|
| HW1 | TF-IDF + Logistic Regression | `sklearn` |
| HW2 | GloVe Embeddings + Custom MLP + Optuna | `PyTorch`, `gensim`, `optuna` |
| HW3 | BERT / DistilBERT Fine-tuning + Optuna | `transformers`, `PyTorch`, `optuna` |

All three tasks use the same tweet dataset structure: `train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`.

---

## Homework 1 — Classical ML Baseline

**Notebook:** `ai-2-homework-1.ipynb`

### Approach

A classical NLP pipeline using **TF-IDF vectorisation** and **Logistic Regression** as a strong baseline before moving to deep learning.

```
Raw Tweet Text
      │
      ▼
┌─────────────────────┐
│   Text Preprocessing│
│  - lowercase        │
│  - remove URLs      │
│  - expand slang     │  ← e.g. "gr8" → "great", "b4" → "before"
│  - normalise spaces │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   TF-IDF Vectoriser │  ← word/n-gram frequency weighting
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Logistic Regression│  ← tuned with GridSearchCV
│  (Pipeline + CV)    │
└────────┬────────────┘
         │
         ▼
    submission.csv
```

### Key Design Decisions

- **Slang expansion**: A custom dictionary maps SMS/Twitter abbreviations to full words (e.g. `btw` → `but the way`, `h8` → `hate`) before vectorisation, improving vocabulary coverage.
- **GridSearchCV**: Hyperparameters (C, max_features, ngram_range) are tuned via cross-validation on the training set.
- **Evaluation metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix are all computed on the validation set.

---

## Homework 2 — Neural Network with GloVe Embeddings

**Notebook:** `ai-hw2.ipynb`

### Approach

Replaces the sparse TF-IDF representation with **dense GloVe word embeddings** (Twitter-trained, 200d), then trains a custom feedforward neural network with **Optuna hyperparameter optimisation**.

```
Raw Tweet Text
      │
      ▼
┌──────────────────────┐
│  Text Preprocessing  │
│  + NLTK Tokenisation │
│  + Lemmatisation     │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  GloVe Twitter 27B   │  ← pre-trained 200-dim embeddings
│  (loaded via gensim) │
│                      │
│  tokens → avg vector │  ← mean-pool all token embeddings
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│          Custom Net (PyTorch)        │
│                                      │
│  Input (200d)                        │
│     │                                │
│  BatchNorm1d                         │
│     │                                │
│  Linear → BatchNorm → Act → Dropout  │  block 1 (H1 units)
│     │                                │
│  Linear → BatchNorm → Act → Dropout  │  block 2 (H2 units)
│     │                                │
│  Linear (1) → sigmoid                │  binary output
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│   Optuna (25 trials) │
│  - optimizer: Adam/  │
│    AdamW             │
│  - learning_rate     │
│  - dropout_rate      │
│  - weight_decay      │
│  - activation fn     │  ← LeakyReLU / ReLU / SELU
│  - batch_size        │
│  - H1, H2 sizes      │
└──────────────────────┘
```

### Best Hyperparameters Found

```python
{
  'optimizer_name': 'AdamW',
  'learning_rate':  0.004169673329375278,
  'dropout_rate':   0.45,
  'weight_decay':   0.0010285156143153202,
  'activation':     'LeakyReLU',
  'batch_size':     128,
  'H1':             32,
  'H2':             96
}
```

### Key Design Decisions

- **GloVe Twitter embeddings** (27B tokens, 200d) were chosen over generic GloVe because the dataset is tweets — Twitter-specific embeddings encode slang and hashtag patterns better.
- **Mean pooling** over token embeddings gives a fixed-size document representation regardless of tweet length.
- **BatchNorm before activation** stabilises training on the relatively small embedding inputs.
- **StandardScaler** is applied to the embedding vectors before feeding into the network — important since GloVe vectors have inconsistent magnitudes across dimensions.
- Optuna's **TPE sampler** efficiently navigates the hyperparameter space; after 25 trials the best config was hardcoded for the final submission run.

---

## Homework 3 — Transformer Fine-tuning (BERT & DistilBERT)

**Notebooks:** `ai-2-hw3-bert.ipynb` · `ai-2-hw3-distilbert.ipynb`

### Approach

Fine-tunes pre-trained Transformer models on the tweet classification task using HuggingFace `transformers`. Two model variants were explored and compared.

```
Raw Tweet Text
      │
      ▼
┌──────────────────────────┐
│   HuggingFace Tokeniser  │
│  - WordPiece tokenisation│
│  - [CLS] + [SEP] tokens  │
│  - padding / truncation  │
│  - max_length: 64        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│           TweetDataset (PyTorch)             │
│  encode_plus() → input_ids + attention_mask  │
└────────────────────┬─────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│  BERT-base-     │   │  DistilBERT-base-    │
│  uncased        │   │  uncased             │
│  (110M params)  │   │  (66M params, ~40%   │
│                 │   │   smaller/faster)    │
└────────┬────────┘   └──────────┬───────────┘
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Fine-tuning Loop     │
         │                       │
         │  AdamW optimiser      │
         │  Linear LR warmup     │
         │  Gradient clipping    │
         │  BCEWithLogitsLoss    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Optuna (25 trials)   │
         │  - learning_rate      │
         │  - batch_size         │
         │  - epochs             │
         │  - warmup_ratio       │
         │  - grad_clip_norm     │
         │  - dropout rates      │
         │  - optimizer_eps      │
         │  MedianPruner         │  ← early-stops bad trials
         └───────────────────────┘
```

### Best Hyperparameters Found

**BERT:**
```python
{
  "learning_rate":    1.739e-05,
  "batch_size":       16,
  "epochs":           2,
  "max_length":       64,
  "warmup_ratio":     0.01705,
  "grad_clip_norm":   1.2664,
  "weight_decay":     0.00714,
  "attention_dropout":0.193,
  "hidden_dropout":   0.189
}
```

**DistilBERT:**
```python
{
  "learning_rate":    1.786e-05,
  "batch_size":       8,
  "epochs":           2,
  "max_length":       64,
  "warmup_ratio":     0.0574,
  "grad_clip_norm":   1.302,
  "weight_decay":     0.00832,
  "attention_dropout":0.113,
  "hidden_dropout":   0.184
}
```

### Key Design Decisions

- **max_length = 64**: Tweets are short by nature (≤280 chars). Truncating to 64 tokens covers the vast majority of tweets while significantly reducing memory and compute cost vs the default 512.
- **Gradient clipping** (`clip_grad_norm_`) prevents exploding gradients during fine-tuning, especially important with very small learning rates and small batch sizes.
- **Linear warmup scheduler**: Gradually increases LR from 0 during the warmup phase, then linearly decays — standard practice for stable Transformer fine-tuning.
- **MedianPruner in Optuna**: Automatically terminates unpromising trials early based on intermediate validation F1, saving GPU time.
- **BERT vs DistilBERT trade-off**: DistilBERT is ~40% smaller and ~60% faster with minimal accuracy loss (~3%), making it practical when GPU time is limited. Both were evaluated to identify the better option for the leaderboard.

---

## Progression Summary

```
HW1: TF-IDF + LogReg          → strong baseline, interpretable
         │
         │  (+dense semantics)
         ▼
HW2: GloVe + MLP + Optuna     → captures word meaning, tweet-aware embeddings
         │
         │  (+contextual representations)
         ▼
HW3: BERT / DistilBERT        → full contextual understanding,
     + Optuna fine-tuning        state-of-the-art NLP
```

Each step addresses a limitation of the previous:

- HW1 → HW2: TF-IDF treats words as independent; embeddings capture semantic similarity
- HW2 → HW3: Mean-pooled embeddings lose word order; Transformers model full context with attention

---

## Setup & Requirements

```bash
pip install torch pandas numpy scikit-learn gensim nltk optuna transformers matplotlib seaborn
```

### Running the Notebooks

The notebooks were developed on **Kaggle** (GPU environment). To run locally:

1. Update `source_dir` to point to your local dataset path
2. For HW2, download [GloVe Twitter 27B](https://nlp.stanford.edu/projects/glove/) (200d variant)
3. For HW3, models are downloaded automatically via HuggingFace

To re-run Optuna hyperparameter search, uncomment **Section 7** in the HW2/HW3 notebooks. The hardcoded `best_params` reflect the results of a completed 25-trial study.

---

*Course: AI-2 — Deep Learning for NLP*
