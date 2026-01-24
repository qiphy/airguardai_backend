# backend/train_virus_model_protein.py
from __future__ import annotations
from pathlib import Path
import re
import random
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score

AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")  # standard 20 aa

def read_fasta(path: Path) -> list[str]:
    seqs = []
    cur = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.upper())
        if cur:
            seqs.append("".join(cur))
    return seqs

def clean_protein(seq: str) -> str | None:
    seq = AA_RE.sub("", seq.upper())
    # remove too-short proteins (noise) and extremely long oddities
    if len(seq) < 50 or len(seq) > 5000:
        return None
    return seq

def load_class(faa_paths: list[Path], label: int, max_seqs: int | None = None) -> tuple[list[str], list[int]]:
    all_seqs = []
    for p in faa_paths:
        all_seqs.extend(read_fasta(p))
    all_seqs = [s for s in (clean_protein(s) for s in all_seqs) if s is not None]

    # downsample for speed if huge
    if max_seqs is not None and len(all_seqs) > max_seqs:
        random.shuffle(all_seqs)
        all_seqs = all_seqs[:max_seqs]

    y = [label] * len(all_seqs)
    return all_seqs, y

def main():
    random.seed(42)
    base = Path(__file__).resolve().parent.parent  # project root if backend/ is inside
    # Adjust if your layout is different:
    pos = [Path(".//data/ncbi/pos_fluA/ncbi_dataset/data/protein.faa")]
    neg1 = [Path("./data/ncbi/neg_adeno/ncbi_dataset/data/protein.faa")]
    neg2 = [Path("./data/ncbi/neg_fluB/ncbi_dataset/data/protein.faa")]

    if not pos:
        raise SystemExit("No positive protein.faa found. Did you unzip Influenza A download?")
    if not neg1 and not neg2:
        raise SystemExit("No negatives found. Download at least one negative class (e.g., Coronaviridae).")

    X_pos, y_pos = load_class(pos, label=1, max_seqs=5000)
    X_neg1, y_neg1 = load_class(neg1, label=0, max_seqs=5000) if neg1 else ([], [])
    X_neg2, y_neg2 = load_class(neg2, label=0, max_seqs=5000) if neg2 else ([], [])

    X = X_pos + X_neg1 + X_neg2
    y = y_pos + y_neg1 + y_neg2

    if len(X) < 2000:
        raise SystemExit(f"Too little data: {len(X)} protein sequences total. Download more genomes or reduce filters.")

    y_arr = np.array(y, dtype=int)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    # Protein k-mer TF-IDF (works well in hackathons)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=False,
        max_features=200000
        )),
        ("clf", SGDClassifier(loss="log_loss", max_iter=20, tol=1e-3))
    ])

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds, digits=4))

    out = Path(__file__).resolve().parent / "models" / "virus_model_protein.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
