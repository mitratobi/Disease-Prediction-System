import os
import joblib
import numpy as np

MODEL_PATH = "model/random_forest_model.pkl"
META_PATH = "model/model_meta.pkl"

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Run train.py first."
            )
        _model = joblib.load(MODEL_PATH)
        _meta = joblib.load(META_PATH)


def get_symptom_list():
    """Return the list of valid symptom names the model was trained on."""
    _load()
    return _meta["symptom_cols"]


def predict(symptoms: list[str]) -> str:
    """
    Given a list of symptom strings, return the predicted disease name.

    Unknown symptoms are ignored with a warning.
    """
    _load()
    symptom_cols = _meta["symptom_cols"]
    le = _meta["label_encoder"]

    vector = np.zeros(len(symptom_cols), dtype=int)
    for s in symptoms:
        s = s.strip().lower().replace(" ", "_")
        if s in symptom_cols:
            vector[symptom_cols.index(s)] = 1
        else:
            print(f"  [warning] '{s}' is not a recognised symptom — skipped.")

    pred_idx = _model.predict([vector])[0]
    return le.inverse_transform([pred_idx])[0]
