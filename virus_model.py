# virus_model.py
from __future__ import annotations
from pathlib import Path
import re
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "virus_model_protein.joblib"
AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

_model = None

def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Virus model not found at {MODEL_PATH}. "
                "Train it first with train_virus_model_protein.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_influenza_like_protein(protein_seq: str) -> float:
    seq = AA_RE.sub("", protein_seq.upper())
    if len(seq) < 50:
        raise ValueError("Protein sequence too short after cleaning (need >= 50 aa).")
    model = _load_model()
    p = float(model.predict_proba([seq])[0][1])
    return p
