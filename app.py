# app.py
from __future__ import annotations

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from virus_data import VIRUS_DB

app = FastAPI()


class PredictPayload(BaseModel):
    location: str
    virus_name: Optional[str] = None
    protein_sequence: Optional[str] = None


def _clean_protein(seq: str) -> str:
    # Keep amino letters only, allow FASTA headers to be passed but strip them out for matching.
    lines = seq.splitlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        out.append(line.upper())
    joined = "".join(out)
    # filter invalid letters
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    joined = "".join([c for c in joined if c in allowed])
    return joined


@app.post("/predict")
def predict(payload: PredictPayload) -> Dict[str, Any]:
    # Require location
    location = (payload.location or "").strip()
    if not location:
        raise HTTPException(status_code=400, detail="location is required")

    # Case 1: virus_name
    if payload.virus_name and payload.virus_name.strip():
        key = payload.virus_name.strip().lower()

        if key not in VIRUS_DB:
            # 404 used by your frontend to show "Not Found"
            raise HTTPException(status_code=404, detail="Not Found")

        # Your real model logic can go here; this is just a stable response shape.
        return {
            "location": location,
            "input_type": "virus_name",
            "virus_key": key,
            "top_viruses": [
                {"name": key, "score": 1.0},
            ],
        }

    # Case 2: protein_sequence
    if payload.protein_sequence and payload.protein_sequence.strip():
        seq = _clean_protein(payload.protein_sequence)
        if len(seq) < 25:
            raise HTTPException(status_code=400, detail="protein_sequence too short")

        # Example: naive matching by exact sequence (or prefix match) against VIRUS_DB
        # Replace with your embedding / ML logic.
        matches: List[Dict[str, Any]] = []
        for k, db_seq in VIRUS_DB.items():
            db_clean = _clean_protein(db_seq)
            if seq == db_clean:
                matches.append({"name": k, "score": 1.0})

        if not matches:
            # Not found by protein match
            raise HTTPException(status_code=404, detail="Not Found")

        return {
            "location": location,
            "input_type": "protein_sequence",
            "top_viruses": matches[:5],
        }

    raise HTTPException(status_code=400, detail="Provide virus_name or protein_sequence")
