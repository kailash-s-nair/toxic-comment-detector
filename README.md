# Toxic Comment Detector

This repository contains the final project for **CSCI 4050U – Machine Learning**.  
The goal is to detect toxic online comments and assign one or more toxicity labels to each comment.

Given a raw text comment, the models predict six binary labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

We compare a classical machine-learning baseline to two neural networks, then use the best model in an interactive demo script.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Models](#models)  
4. [Repository Structure](#repository-structure)  
5. [Getting the Code](#getting-the-code)  
6. [Setup & Installation](#setup--installation)  
7. [How to Run](#how-to-run)  
   - [Baseline: TF-IDF + Logistic Regression](#baseline-tf-idf--logistic-regression-model0py)  
   - [Neural Model 1: Embedding + Avg Pooling + MLP](#neural-model-1-embedding--average-pooling--mlp-model1py)  
   - [Neural Model 2: Embedding + Bi-LSTM](#neural-model-2-embedding--bi-lstm-model2py)  
   - [Interactive Demo](#interactive-demo-demopy)  
8. [Results (Template)](#results-template)  
9. [Extending the Project](#extending-the-project)  
10. [Troubleshooting](#troubleshooting)  
11. [Acknowledgements](#acknowledgements)

---

## Project Overview

Online platforms are filled with user-generated comments, some of which can be toxic or harmful.  
This project builds and compares several models for **multi-label toxic comment classification**:

- A **classical baseline** using bag-of-words style features (TF-IDF) + **Logistic Regression**.
- A **simple neural network** with word embeddings and an MLP classifier.
- A **Bi-LSTM neural network** that captures word order and context.

Each model outputs six binary labels for a given comment, allowing a single comment to be tagged with multiple toxicity types at once.

---

## Dataset

This project is designed around the **Kaggle “Toxic Comment Classification Challenge”** style dataset:

- Each row: a user comment (text) + up to 6 binary labels.
- Labels match the six classes listed above.
- Train/validation/test splits are created from the original training data.

> **Note:**  
> Place your dataset files (e.g., `train.csv`, `test.csv`) in a `data/` folder, or update the paths inside the model scripts if you use a different layout.

---

## Models

We compare three main approaches:

1. **Model 0 – Baseline (TF-IDF + Logistic Regression)**  
   - Vectorizes text into TF-IDF features.  
   - Trains a One-vs-Rest Logistic Regression classifier for each label.  
   - Fast to train and serves as a strong, interpretable baseline.

2. **Model 1 – Embedding + Average Pooling + MLP**  
   - Uses a learned embedding layer to map tokens to dense vectors.  
   - Averages token embeddings over the sequence.  
   - Feeds the pooled vector into a small MLP (fully-connected network) for prediction.  
   - Captures more semantic information than TF-IDF while staying simple and efficient.

3. **Model 2 – Embedding + Bi-LSTM (Best Model)**  
   - Uses embeddings followed by a Bi-LSTM to capture word order and context from both directions.  
   - The final hidden states (or a pooled representation) go through a fully-connected layer for multi-label classification.  
   - Typically achieves the best performance among the three.

---

## Repository Structure

```text
.
├─ data/                 # Dataset files (train/test CSVs, etc.) – not committed by default
├─ models/               # (Optional) Saved model weights/checkpoints
├─ notebooks/            # (Optional) Jupyter notebooks for exploration
├─ src/
│  ├─ model0.py          # Baseline: TF-IDF + Logistic Regression (One-vs-Rest)
│  ├─ model1.py          # Neural model: Embedding + Average Pooling + MLP
│  ├─ model2.py          # Neural model: Embedding + Bi-LSTM (best model)
│  ├─ demo.py            # Interactive terminal demo using the best model
│  ├─ utils.py           # (Optional) Shared utilities: preprocessing, metrics, etc.
│  └─ config.py          # (Optional) Central config: paths, hyperparameters
├─ requirements.txt      # Python dependencies
├─ .gitignore
└─ README.md             # This file
