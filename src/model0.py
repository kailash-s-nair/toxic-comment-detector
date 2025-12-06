import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("data/raw/train.csv")

label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

texts = df["comment_text"].fillna("").tolist()
y = df[label_columns].values

X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y[:, 0],
)

vectorizer = TfidfVectorizer(
    max_features=100_000,      
    ngram_range=(1, 2),
    stop_words="english",
    lowercase=True,
)

X_train = vectorizer.fit_transform(X_train_texts)  
X_val = vectorizer.transform(X_val_texts)

base_clf = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
)

clf = OneVsRestClassifier(base_clf)

clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_val)
y_pred = (y_proba >= 0.5).astype(int)

print("F1 micro:", f1_score(y_val, y_pred, average="micro"))
print("F1 macro:", f1_score(y_val, y_pred, average="macro"))
print(classification_report(y_val, y_pred, target_names=label_columns))
