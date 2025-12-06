import os
import re
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

DATA_CSV_PATH = os.path.join("data", "raw", "train.csv")

MAX_VOCAB_SIZE = 20000
MAX_LEN = 100
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

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

print("Total samples:", len(texts))
print("Label matrix shape:", y.shape)

X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y[:,0],
)

print("Train samples:", len(X_train_texts))
print("Val samples:", len(X_val_texts))


def simple_tokenize(text):
    """
    Very simple whitespace + punctuation tokenizer.
    Lowercases and splits on non-letter characters.
    """
    text = text.lower()
    # replace non-letters with space
    text = re.sub(r"[^a-z]+", " ", text)
    tokens = text.strip().split()
    return tokens

counter = collections.Counter()
for t in X_train_texts:
    counter.update(simple_tokenize(t))

most_common = counter.most_common(MAX_VOCAB_SIZE - 2)
itos = ["<pad>", "<unk>"] + [w  for (w,_) in most_common]
stoi = {w : i for i, w in enumerate(itos)}

PAD_IDX = stoi["<pad>"]
UNK_IDX = stoi["<unk>"]

vocab_size = len(itos)
print("Vocab size:", vocab_size)


def encode_text(text, max_len=MAX_LEN):
    """
    Convert raw text to a fixed-length list of token IDs.
    Unknown words -> UNK_IDX, pad/truncate to max_len.
    """
    tokens = simple_tokenize(text)
    ids = [stoi.get(tok, UNK_IDX) for tok in tokens]

    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return np.array(ids, dtype = np.int64)


class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = encode_text(self.texts[idx], self.max_len)
        label_vec = self.labels[idx]
        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(label_vec, dtype=torch.float32)
    
train_ds = ToxicCommentsDataset(X_train_texts, y_train, max_len = MAX_LEN)
val_ds = ToxicCommentsDataset(X_val_texts, y_val, max_len = MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

print("Batches (train):", len(train_loader))
print("Batches (val):", len(val_loader))

class ToxicBiLSTMModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim,
            num_layers,
            num_labels,
            pad_idx = 0,

    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * hidden_dim, num_labels)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        outputs, (h_n, c_n) = self.lstm(emb)
        h_forward = h_n[-2,:,:]
        h_backward = h_n[-1,:,:]
        h_cat = torch.cat([h_forward, h_backward], dim = 1)
        x = self.dropout(h_cat)
        logits = self.fc(x)
        return logits
    
model = ToxicBiLSTMModel(
    vocab_size = vocab_size,
    embed_dim = EMBED_DIM,
    hidden_dim = HIDDEN_DIM,
    num_layers = NUM_LAYERS,
    num_labels = len(LABEL_COLUMNS),
    pad_idx = PAD_IDX, 
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def evaluate(model, loader, criterion, device, threshold = 0.5):
    model.eval()
    total_loss  = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_pred.append(probs)
            all_true.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    y_true = np.vstack(all_true)
    y_scores = np.vstack(all_pred)
    y_hat = (y_scores >= threshold).astype(int)

    f1_micro = f1_score(y_true, y_hat, average = "micro", zero_division=0)
    f1_macro = f1_score(y_true, y_hat, average="macro", zero_division=0)

    return avg_loss, f1_micro, f1_macro, y_hat, y_true

best_val_f1 = 0.0

for epoch in range(1,EPOCHS + 1):
    train_loss = train_one_epoch(model,train_loader, optimizer, criterion, DEVICE)
    val_loss, f1_micro, f1_macro, y_true, y_hat = evaluate(
        model, val_loader, criterion, DEVICE 
    )
    print(
        f"Epoch {epoch:02d} | "
        f"train loss: {train_loss:.4f} | "
        f"val loss: {val_loss:.4f} | "
        f"F1 micro: {f1_micro:.4f} | "
        f"F1 macro: {f1_macro:.4f}"
    )
if f1_micro > best_val_f1:
    best_val_f1 = f1_micro
    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", " model2_bilstm.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  -> New best model saved to {save_path}")

print("\nClassification report (last epoch):")
print(classification_report(
    y_true, y_hat, target_names=LABEL_COLUMNS, zero_division=0
))