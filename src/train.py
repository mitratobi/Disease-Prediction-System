import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Allow running from project root or from src/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import load_and_preprocess

MODEL_PATH = "model/random_forest_model.pkl"
META_PATH = "model/model_meta.pkl"


def train():
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, symptom_cols, le = load_and_preprocess()

    print(f"Features (symptoms): {len(symptom_cols)}")
    print(f"Classes (diseases):  {len(le.classes_)}")
    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    print("\nTraining RandomForestClassifier (n_estimators=100)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump({"symptom_cols": symptom_cols, "label_encoder": le}, META_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    train()
