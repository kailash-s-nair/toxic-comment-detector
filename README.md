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

## 1. Project Overview

This project implements a **multi-label text classification** system for toxic comment detection.

We:

1. Load and preprocess the Kaggle toxic comments dataset.  
2. Train three models:  
   - Baseline: TF-IDF + Logistic Regression (One-vs-Rest)  
   - Neural Model 1: Embedding + Average Pooling + MLP  
   - Neural Model 2: Embedding + Bi-LSTM (main model)  
3. Evaluate each model with appropriate metrics.  
4. Provide an **interactive demo**:  
   - CLI demo (terminal)  
   - Optional web demo: Flask API + React frontend

---

## 2. Dataset

We use the dataset from the **Jigsaw Toxic Comment Classification Challenge** (Kaggle).

Each row contains:

- `id`: unique comment ID  
- `comment_text`: the raw text of the comment  
- Six target columns (0/1):  
  - `toxic`  
  - `severe_toxic`  
  - `obscene`  
  - `threat`  
  - `insult`  
  - `identity_hate`

You must download the CSVs yourself from Kaggle and place them under `data/` as described below.

---

## 3. Problem Formulation

- **Task type**: Multi-label classification  
  - Each comment can have zero, one, or multiple labels active.  
- **Input**: Raw English text (user comments).  
- **Output**: A 6-dimensional vector of probabilities or binary labels indicating the presence of each toxicity category.  
- **Evaluation**: We focus on metrics suitable for multi-label problems, such as:  
  - Macro / micro F1-score  
  - ROC-AUC (optional)  
  - Per-label F1

---

## 4. Repository Structure

```text
.
├─ data/
│  ├─ raw/
│  │  ├─ train.csv          # Kaggle training data
│  │  └─ test.csv           # (optional) Kaggle test data
│  └─ processed/            # any preprocessed or cached files
│
├─ saved_models/
│  ├─ model0_baseline.joblib   # example: baseline model artifacts
│  ├─ tfidf_vectorizer.joblib  # example: TF-IDF vectorizer
│  └─ model2_bilstm.pt         # Bi-LSTM model weights
│
├─ src/
│  ├─ model0.py      # Baseline: TF-IDF + Logistic Regression (One-vs-Rest)
│  ├─ model1.py      # Neural model: Embedding + Average Pooling + MLP
│  ├─ model2.py      # Neural model: Embedding + Bi-LSTM (main model)
│  ├─ demo.py        # Shared inference + CLI demo
│  └─ webDemo/
│     ├─ api.py      # Flask API exposing /predict
│     └─ frontend/   # Vite + React web demo
│        ├─ package.json
│        └─ src/App.jsx
│
├─ requirements.txt  # Python dependencies
└─ README.md
```

You may not have all files if you have not trained all models yet. The important pieces are:

- `data/raw/train.csv` (dataset)  
- `src/model*.py` (training scripts)  
- `src/demo.py` + `src/webDemo/api.py` (inference & API)  
- `src/webDemo/frontend` (web demo)

---

## 5. Getting the Code

You have two main options:

### Option A: Clone with Git (recommended)

```bash
git clone https://github.com/<your-username>/toxic-comment-detector.git
cd toxic-comment-detector
```

### Option B: Download as ZIP

1. Go to the GitHub repository page in your browser.  
2. Click **Code → Download ZIP**.  
3. Extract the ZIP file somewhere on your computer.  
4. Open a terminal in that extracted folder.

---

## 6. Setup & Installation

### 6.1. Requirements

- Python 3.9+ (3.10/3.11 are ideal)  
- `pip` (Python package manager)  
- Optional but recommended: `virtualenv` or `conda`  
- Enough RAM to train small neural networks (8GB+ is comfortable)

### 6.2. Create and Activate a Virtual Environment (recommended)

From the project root:

```bash
python -m venv .venv
```

Activate (Windows):

```bash
.\.venv\Scriptsctivate
```

Activate (macOS / Linux):

```bash
source .venv/bin/activate
```

You should now see the environment name in your terminal prompt.

### 6.3. Install Python Dependencies

Using `requirements.txt` in the repo root:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install (at minimum):

- `numpy`, `pandas`  
- `scikit-learn`  
- `torch` (PyTorch)  
- `Flask`, `Flask-Cors`  
- `colorama`  
- `tqdm`

### 6.4. Dataset Setup

This project is designed for the **"Toxic Comment Classification Challenge"** dataset (Kaggle).

1. Go to Kaggle and download the dataset CSV files.  
2. Create the following folder structure (if not already there):

```text
data/
  raw/
    train.csv
    test.csv          # optional
  processed/
```

3. Place `train.csv` (and `test.csv` if available) inside `data/raw/`.

Some scripts may automatically create `data/processed/` when you run them.

---

## 7. How to Run

All commands below assume you are in the **project root directory** and your virtual environment (if any) is activated.

### 7.1. Baseline: TF-IDF + Logistic Regression (`model0.py`)

```bash
cd src
python model0.py
```

### 7.2. Neural Model 1: Embedding + Average Pooling + MLP (`model1.py`)

```bash
cd src
python model1.py
```

### 7.3. Neural Model 2: Embedding + Bi-LSTM (`model2.py`)

```bash
cd src
python model2.py
```

### 7.4. Interactive CLI Demo (`demo.py`)

```bash
cd src
python demo.py
```

### 7.5. Web API (`api.py`) + React Frontend

**Flask API:**

From the project root:

```bash
python -m src.webDemo.api
```

This exposes a `/predict` endpoint at `http://localhost:8000`.

**React frontend:**

```bash
cd src/webDemo/frontend
npm install
npm run dev
```

Open the printed `http://localhost:5173` URL in your browser and use the UI to send comments to the model.

---

## 8. Results (Template)

Use this section to summarize your final results (replace placeholders with your actual numbers).

### 8.1. Overall Performance

| Model                                   | Macro F1 | Micro F1 | ROC-AUC | Notes                    |
|----------------------------------------|----------|----------|---------|--------------------------|
| Baseline: TF-IDF + Logistic Regression | 0.xx     | 0.xx     | 0.xx    | Simple, fast baseline    |
| Neural Model 1: Emb + AvgPool + MLP    | 0.xx     | 0.xx     | 0.xx    |                          |
| Neural Model 2: Emb + Bi-LSTM          | 0.xx     | 0.xx     | 0.xx    | Best overall performance |

### 8.2. Per-Label F1 Scores (Example)

| Label         | Baseline F1 | Model 1 F1 | Model 2 F1 |
|---------------|-------------|------------|------------|
| toxic         | 0.xx        | 0.xx       | 0.xx       |
| severe_toxic  | 0.xx        | 0.xx       | 0.xx       |
| obscene       | 0.xx        | 0.xx       | 0.xx       |
| threat        | 0.xx        | 0.xx       | 0.xx       |
| insult        | 0.xx        | 0.xx       | 0.xx       |
| identity_hate | 0.xx        | 0.xx       | 0.xx       |

---

## 9. Extending the Project

Ideas to push this further:

- Pretrained embeddings (GloVe, FastText)  
- Transformer-based models (e.g., BERT, DistilBERT)  
- Better text preprocessing (URLs, emojis, mentions, leetspeak)  
- Class imbalance handling (class weights, focal loss, oversampling)  
- Explainability (LIME/SHAP, attention visualizations)  
- Real-time moderation demos (chat bot, comment filter, etc.)

---

## 10. Troubleshooting

- **Dataset not found**:  
  - Check that `train.csv` is in `data/raw/`.  
  - Print the current working directory in your script to debug paths:

    ```python
    import os
    print(os.getcwd())
    ```

- **Out-of-memory errors**:  
  - Reduce `batch_size`, `max_seq_len`, `embedding_dim`, or `hidden_dim`.  
  - Close other heavy applications.

- **CUDA / GPU issues**:  
  - If you don’t have CUDA, force CPU mode.  
  - If you do, ensure your PyTorch install matches your CUDA version.

- **Import errors / module not found**:  
  - Run commands from the project root.  
  - Make sure `src/` is in the Python path (using `python -m src.webDemo.api` rather than running files directly sometimes helps).  
  - Confirm your virtual environment is activated.

---

## 11. Acknowledgements

- **Course**: CSCI 4050U – Machine Learning (Ontario Tech University)  
- **Dataset**: Jigsaw Toxic Comment Classification Challenge (Kaggle)  
- **Libraries / Tools**:  
  - PyTorch, scikit-learn, pandas, NumPy  
  - Flask, Flask-Cors  
  - React, Vite  
- **People**: Course instructor and TAs for project guidance and feedback.

If you use or build on this project, please consider citing the original dataset and giving credit to the contributors of this repository.
