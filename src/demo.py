import os
import re
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

DATA_CSV_PATH = os.path.join("data", "raw", "train.csv")

MAX_VOCAB_SIZE = 20000
MAX_LEN = 100
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


if not os.path.exists(DATA_CSV_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_CSV_PATH}. "
        f"Make sure train.csv is at data/raw/train.csv"
    )

df = pd.read_csv(DATA_CSV_PATH)

texts = df["comment_text"].fillna("").tolist()
y = df[LABEL_COLUMNS].values.astype(np.float32)

# same split as model2 (so vocab is built on train only)
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y[:, 0],
)

def simple_tokenize(text):
    """
    Same simple tokenizer as model2.py
    """
    text = text.lower()
    text = re.sub(r"[^a-z]+", " ", text)
    tokens = text.strip().split()
    return tokens


counter = collections.Counter()
for t in X_train_texts:
    counter.update(simple_tokenize(t))

most_common = counter.most_common(MAX_VOCAB_SIZE - 2)

itos = ["<pad>", "<unk>"] + [w for (w, _) in most_common]
stoi = {w: i for i, w in enumerate(itos)}

PAD_IDX = stoi["<pad>"]
UNK_IDX = stoi["<unk>"]
vocab_size = len(itos)


def encode_text(text, max_len=MAX_LEN):
    """
    Convert text to fixed-length list of token IDs.
    """
    tokens = simple_tokenize(text)
    ids = [stoi.get(tok, UNK_IDX) for tok in tokens]

    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return np.array(ids, dtype=np.int64)

class ToxicBiLSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_labels,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * hidden_dim, num_labels)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)     # (batch, seq_len, embed_dim)
        outputs, (h_n, c_n) = self.lstm(emb)
        # last layer's forward and backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        x = self.dropout(h_cat)
        logits = self.fc(x)
        return logits
    
model_path = os.path.join("saved_models", " model2_bilstm.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Could not find {model_path}. "
        f"Train model2.py first so it saves the weights."
    )

model = ToxicBiLSTMModel(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_labels=len(LABEL_COLUMNS),
    pad_idx=PAD_IDX,
).to(DEVICE)

state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("Loaded model2_bilstm.pt")
print("Vocab size:", vocab_size)
print("Ready for interactive demo.\n")

def predict_comment(text, threshold=0.5):
    """
    Encode text, run model, return label probabilities and predictions.
    """
    ids = encode_text(text, MAX_LEN)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, seq_len)

    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # shape: (6,)

    predictions = (probs >= threshold).astype(int)
    return probs, predictions


def pretty_print_predictions(text, probs, preds, threshold=0.5):
    print("\nInput comment:")
    print(text)
    print("\nPredicted labels (threshold = {:.2f}):".format(threshold))
    for label, p, pred in zip(LABEL_COLUMNS, probs, preds):
        status = "YES" if pred == 1 else "no"
        print(f"  {label:13s}  ->  {status:3s}  (p = {p:.3f})")
    print("-" * 50)


if __name__ == "__main__":
    print("Type a comment to classify. Type 'quit' to exit.\n")

    while True:
        user_input = input("Your comment: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Exiting demo.")
            break

        if not user_input:
            print("Please type something.\n")
            continue

        probs, preds = predict_comment(user_input, threshold=0.5)
        pretty_print_predictions(user_input, probs, preds, threshold=0.5)