# Toxic Comment Detector

This repository contains our final project for **CSCI 4050U – Machine Learning**.  
The goal is to detect toxic online comments and assign one or more toxicity labels to each comment.

Given a raw text comment, the models predict six binary labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

We compare a classical machine-learning baseline to two neural networks, then deploy the best model in an interactive demo.

---

## 1. Project Structure

```text
.
├─ src/
│  ├─ model0.py      # Baseline: TF-IDF + Logistic Regression (One-vs-Rest)
│  ├─ model1.py      # Neural model: Embedding + Average Pooling + MLP
│  ├─ model2.py      # Neural model: Embedding + Bi-LSTM (best model)
│  └─ demo.py        # Interactive terminal demo using the trained Bi-LSTM
├─ data/
│  └─ raw/
│     └─ train.csv   # Kaggle toxic comment dataset (NOT committed)
├─ saved_models/     # Trained model weights (NOT committed)
├─ notebooks/        # Optional analysis / results notebooks
├─ requirements.txt  # Python dependencies
└─ .gitignore
