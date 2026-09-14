"""
Trains the SOS message triage classifier using TF-IDF features + Logistic
Regression heads for category and severity.

Why TF-IDF instead of a transformer here: with ~200 labeled examples,
TF-IDF + a linear classifier is a legitimate, well-established choice
(large pretrained embeddings need more data to pay off their extra
complexity) and it has the advantage of being fully interpretable --
you can inspect exactly which words push a message toward
"trapped_injured" vs "informational", which is a strong thing to show
in a demo. See scripts/train_classifier.py for the DistilBERT-embedding
version if you want to run that instead on a machine with internet
access to huggingface.co.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_PATH = "data/sos_messages.csv"
MODELS_DIR = "models"


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"{len(df)} labeled examples across {df['category'].nunique()} categories")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams ("chest pain", "rising fast")
        min_df=2,
        max_features=2000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["text"])
    y_cat = df["category"].values
    y_sev = df["severity"].values

    X_train, X_test, ycat_train, ycat_test, ysev_train, ysev_test, idx_train, idx_test = \
        train_test_split(X, y_cat, y_sev, df.index, test_size=0.15, random_state=7,
                          stratify=y_cat)

    print("\nTraining category classifier ...")
    cat_clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    cat_clf.fit(X_train, ycat_train)
    cat_pred = cat_clf.predict(X_test)
    print(f"Category accuracy: {accuracy_score(ycat_test, cat_pred):.3f}")
    print(classification_report(ycat_test, cat_pred, zero_division=0))

    print("Training severity classifier ...")
    sev_clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    sev_clf.fit(X_train, ysev_train)
    sev_pred = sev_clf.predict(X_test)
    print(f"Severity accuracy: {accuracy_score(ysev_test, sev_pred):.3f}")
    print(classification_report(ysev_test, sev_pred, zero_division=0))

    # show the most predictive words per category - good for the demo/README
    print("\nTop words per category (interpretability):")
    feature_names = np.array(vectorizer.get_feature_names_out())
    for i, cls in enumerate(cat_clf.classes_):
        top_idx = np.argsort(cat_clf.coef_[i])[-8:][::-1]
        print(f"  {cls:18s}: {', '.join(feature_names[top_idx])}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(cat_clf, f"{MODELS_DIR}/category_classifier.joblib")
    joblib.dump(sev_clf, f"{MODELS_DIR}/severity_classifier.joblib")
    print(f"\nSaved vectorizer + classifier heads to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
