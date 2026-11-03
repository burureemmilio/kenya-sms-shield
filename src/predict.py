"""Load the saved model and score a message."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .preprocessing import warning_indicators
from .train import scam_score

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def load_model(path: Path | None = None):
    p = path or ROOT / "models" / "final_model.joblib"
    if p not in _CACHE:
        _CACHE[p] = joblib.load(p)
    return _CACHE[p]


def top_terms(bundle, text: str, k: int = 8):
    """Terms in THIS message that push hardest towards scam (+) or legit (-), via coef * feature value."""
    pipe = bundle["pipeline"]
    clf = pipe.named_steps["clf"]
    feats = pipe.named_steps["features"]
    names = np.array(feats.get_feature_names_out())
    x = feats.transform([text])
    if hasattr(clf, "coef_"):
        w = clf.coef_.ravel()
    else:  # Naive Bayes: log-odds of each feature
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    contrib = x.multiply(w).tocsr()
    idx = contrib.indices
    vals = contrib.data
    order = np.argsort(vals)
    pos = [(names[idx[i]].split("__")[-1], float(vals[i])) for i in order[::-1][:k] if vals[i] > 0]
    neg = [(names[idx[i]].split("__")[-1], float(vals[i])) for i in order[:k] if vals[i] < 0]
    return pos, neg


def predict(text: str, threshold: float | None = None, path: Path | None = None) -> dict:
    bundle = load_model(path)
    thr = bundle["config"]["threshold"] if threshold is None else threshold
    s = float(scam_score(bundle["pipeline"], [text])[0])
    pos, neg = top_terms(bundle, text)
    return dict(score=s, threshold=thr, label="scam" if s >= thr else "legit",
                indicators=warning_indicators(text), scam_terms=pos, legit_terms=neg)
