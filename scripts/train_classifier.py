"""
Trains the SOS message triage classifier.

Approach: use a frozen, pretrained DistilBERT to turn each message into a
768-dim embedding (mean-pooled over tokens), then train two lightweight
classifier heads on top of those embeddings:
  - category head: 6-class (trapped_injured, medical, structural_rescue,
    water_food, evacuation, informational)
  - severity head: 5-class (severity 1-5)

This is fast on CPU (embeddings are a single forward pass per example,
then the classifier heads train in seconds) while still being genuine
model training on our own labeled data, not a rules lookup table.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/sos_messages.csv"
MODELS_DIR = "models"


def embed_texts(texts: list[str], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    """Mean-pool DistilBERT token embeddings into one vector per message."""
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64,
                             return_tensors="pt")
            out = model(**enc)
            hidden = out.last_hidden_state                       # (B, T, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()    # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts                          # (B, 768)
            all_embeddings.append(mean_pooled.numpy())
    return np.vstack(all_embeddings)


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"{len(df)} labeled examples across {df['category'].nunique()} categories")

    print(f"Loading {MODEL_NAME} (frozen, for embeddings only) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    for p in model.parameters():
        p.requires_grad = False

    print("Computing embeddings for all messages ...")
    X = embed_texts(df["text"].tolist(), tokenizer, model)
    y_cat = df["category"].values
    y_sev = df["severity"].values

    X_train, X_test, ycat_train, ycat_test, ysev_train, ysev_test = train_test_split(
        X, y_cat, y_sev, test_size=0.15, random_state=7, stratify=y_cat
    )

    print("\nTraining category classifier ...")
    cat_clf = LogisticRegression(max_iter=2000, C=2.0)
    cat_clf.fit(X_train, ycat_train)
    cat_pred = cat_clf.predict(X_test)
    print(f"Category accuracy: {accuracy_score(ycat_test, cat_pred):.3f}")
    print(classification_report(ycat_test, cat_pred, zero_division=0))

    print("Training severity classifier ...")
    sev_clf = LogisticRegression(max_iter=2000, C=2.0)
    sev_clf.fit(X_train, ysev_train)
    sev_pred = sev_clf.predict(X_test)
    print(f"Severity accuracy: {accuracy_score(ysev_test, sev_pred):.3f}")
    print(classification_report(ysev_test, sev_pred, zero_division=0))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(cat_clf, f"{MODELS_DIR}/category_classifier.joblib")
    joblib.dump(sev_clf, f"{MODELS_DIR}/severity_classifier.joblib")
    print(f"\nSaved classifier heads to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
