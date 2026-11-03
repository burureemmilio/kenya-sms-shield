"""Evaluation helpers shared by notebooks 03-05."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from .train import scam_score

DEFAULT_THRESHOLD = {"nb": 0.5, "lr": 0.5, "svm": 0.0}


def metrics(y, score, thr) -> dict:
    """Scam = positive class. recall = share of scams caught; precision = share of flagged that are scams."""
    pred = (np.asarray(score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, score), tp=tp, fp=fp, fn=fn, tn=tn)


def cv_fold_metrics(pipe, X, y, folds, model_key, thr=None) -> pd.DataFrame:
    """Fit a fresh clone on each training fold, score the held-out fold. `folds` = list of (train_idx, test_idx)."""
    X = np.asarray(X, dtype=object); y = np.asarray(y)
    thr = DEFAULT_THRESHOLD[model_key] if thr is None else thr
    rows = []
    for k, (a, b) in enumerate(folds):
        p = clone(pipe).fit(X[a], y[a])
        rows.append(dict(fold=k, **metrics(y[b], scam_score(p, X[b]), thr)))
    return pd.DataFrame(rows)


def mean_sd(df: pd.DataFrame, cols=("accuracy", "recall", "precision", "f1", "roc_auc")) -> pd.Series:
    out = {}
    for c in cols:
        out[c] = f"{df[c].mean():.3f} ± {df[c].std():.3f}"
    return pd.Series(out)


def oof_scores(pipe, X, y, folds, n) -> np.ndarray:
    """Out-of-fold scam scores: each row scored by a model that never saw it (or its group)."""
    X = np.asarray(X, dtype=object); y = np.asarray(y)
    s = np.full(n, np.nan)
    for a, b in folds:
        s[b] = scam_score(clone(pipe).fit(X[a], y[a]), X[b])
    return s


def threshold_table(y, score, thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = metrics(y, score, t)
        rows.append(dict(threshold=t, recall=m["recall"], precision=m["precision"], f1=m["f1"],
                         missed_scams=m["fn"], false_alarms=m["fp"]))
    return pd.DataFrame(rows)


def choose_threshold(y, score, target_recall=0.95) -> float:
    """Highest-precision threshold that still keeps scam recall >= target (computed on validation/OOF scores).
    Among candidate cut-points (the observed scores), pick the LARGEST threshold meeting the recall target:
    raising the threshold can only lower recall and raise (or keep) precision, so the largest feasible one
    gives the fewest false alarms for the required level of protection."""
    y = np.asarray(y); score = np.asarray(score)
    best = None
    for t in np.unique(score):
        if (score[y == 1] >= t).mean() >= target_recall:
            best = t if best is None else max(best, t)
    return float(best)
