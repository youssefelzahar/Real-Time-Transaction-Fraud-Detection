"""Shared helpers for the Phase 2 model audit: data, splits, metrics, score cache.

Scores (one float32 per row) are cached outside the repository, in
$AUDIT_SCORES_DIR (default ~/.cache/nilepay-model-audit), so later scripts can
recompute metrics from raw predictions without re-scoring.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import TARGET, chronological_split, load_data  # noqa: E402
from src.evaluate import evaluate_at_threshold  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "fraud_data.csv"
SCORES_DIR = Path(os.getenv("AUDIT_SCORES_DIR", Path.home() / ".cache" / "nilepay-model-audit"))
EXPECTED_BOUNDARIES = {"train": [1, 280], "val": [281, 354], "test": [355, 743]}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_splits():
    df = load_data(DATA)
    train, val, test, bounds = chronological_split(df)
    assert bounds == EXPECTED_BOUNDARIES, bounds
    assert train["step"].max() < val["step"].min() and val["step"].max() < test["step"].min()
    return df, train, val, test, bounds


def metrics(y, p, threshold):
    """src.evaluate metrics (now incl. accuracy and F1) plus counts as plain ints."""
    m = evaluate_at_threshold(np.asarray(y), np.asarray(p), threshold)
    m["n"] = int(len(y))
    m["fraud"] = int(np.sum(y))
    return m


def save_scores(name, **arrays):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES_DIR / f"{name}.npz", **arrays)


def load_scores(name):
    path = SCORES_DIR / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing: run score_artifacts.py / rerun_notebook.py first")
    return dict(np.load(path))


def write_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, default=float) + "\n")


def compare(recorded, reproduced, keys=None, tol=1e-9):
    """Field-by-field comparison; ints must match exactly, floats within tol."""
    out = {}
    for k in keys or recorded:
        if k not in reproduced:
            continue
        r, q = recorded[k], reproduced[k]
        if isinstance(r, (int, np.integer)) and not isinstance(r, bool):
            ok = int(r) == int(q)
        else:
            ok = abs(float(r) - float(q)) <= tol
        out[k] = {"recorded": r, "reproduced": q, "match": bool(ok)}
    return out
