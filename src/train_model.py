"""
train_model.py
---------------
Trains the intent-classification model used by the chatbot.

Pipeline:
    raw patterns (data/intents.json)
        -> text preprocessing (src/nlp_utils.py)
        -> TF-IDF vectorization (unigrams + bigrams)
        -> Logistic Regression classifier (multi-class, one-vs-rest)
        -> saved to models/ as .joblib artifacts

Run:
    python src/train_model.py
"""

import json
import os
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(__file__))
from nlp_utils import preprocess_for_vectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.joblib")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.txt")


def load_training_data(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            texts.append(preprocess_for_vectorizer(pattern))
            labels.append(tag)
    return texts, labels


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    texts, labels = load_training_data()
    print(f"Loaded {len(texts)} training examples across {len(set(labels))} intents.")

    # Stratified split isn't reliable with very few examples per class in a
    # small demo dataset, so we train on the full set and additionally
    # report a held-out split where possible for a sanity check.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=10, class_weight="balanced")),
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        print(f"Held-out accuracy: {acc:.2f}")
        print(report)
        with open(REPORT_PATH, "w") as f:
            f.write(f"Held-out accuracy: {acc:.2f}\n\n{report}")
    except ValueError as e:
        # Falls back gracefully if any class has too few samples to split.
        print(f"Skipping held-out evaluation ({e}). Training on full dataset.")

    # Final model is always trained on the FULL dataset for best coverage.
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
