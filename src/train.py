"""Pipelines for the three models (+ feature variants) and a CLI to train the final model.

    python -m src.train          # trains the selected configuration on ALL clean data and saves models/final_model.joblib
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.svm import LinearSVC

from .preprocessing import HandcraftedFeatures, model_text, RANDOM_STATE, clean_dataset, load_raw

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline(model: str = "lr", ngram=(1, 2), hand: bool = False, **params) -> Pipeline:
    """model in {'nb','lr','svm'}.  hand=True appends the rule-based features.

    The TF-IDF vocabulary/IDF weights and the scaler are fitted ONLY when .fit() is called, i.e. on
    whatever training fold the caller supplies."""
    tfidf = TfidfVectorizer(lowercase=False, preprocessor=model_text, ngram_range=ngram, sublinear_tf=True,
                            min_df=params.pop("min_df", 2), strip_accents=None,
                            token_pattern=r"(?u)\b\w+\b")
    # NB needs non-negative inputs -> MinMax; linear models -> MaxAbs (keeps sparsity, handles zeros)
    scaler = MinMaxScaler() if model == "nb" else MaxAbsScaler()
    if hand:
        feats = FeatureUnion([("tfidf", tfidf),
                              ("hand", Pipeline([("h", HandcraftedFeatures()), ("scale", scaler)]))])
    else:
        feats = FeatureUnion([("tfidf", tfidf)])
    if model == "nb":
        clf = MultinomialNB(alpha=params.get("alpha", 0.5))
    elif model == "lr":
        clf = LogisticRegression(C=params.get("C", 4.0), max_iter=2000, class_weight=params.get("class_weight"),
                                 solver=params.get("solver", "liblinear"), random_state=RANDOM_STATE)
    elif model == "svm":
        clf = LinearSVC(C=params.get("C", 0.5), class_weight=params.get("class_weight"), random_state=RANDOM_STATE,
                        max_iter=10000)
    else:
        raise ValueError(model)
    return Pipeline([("features", feats), ("clf", clf)])


def scam_score(pipe: Pipeline, X) -> np.ndarray:
    """Continuous 'scam-ness' score. Probability for NB/LR, signed margin for SVM. Threshold is applied to this."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def train_final(config_path: Path | None = None) -> Path:
    cfg = json.loads((config_path or ROOT / "models" / "final_config.json").read_text())
    clean, _ = clean_dataset(load_raw(ROOT / "data" / "scam_detection_labeled.csv"))
    pipe = build_pipeline(cfg["model"], tuple(cfg["ngram"]), cfg["hand"], **cfg["params"])
    pipe.fit(clean["message_content"], clean["y"])
    out = ROOT / "models" / "final_model.joblib"
    joblib.dump({"pipeline": pipe, "config": cfg}, out)
    return out


if __name__ == "__main__":
    print("saved", train_final())


# ----------------------------------------------------------------------------------------
# Hyper-parameter search (used in notebook 03).  Small, explicit grids: with ~500 training messages a
# huge search would just overfit the validation folds.
# ----------------------------------------------------------------------------------------
GRIDS = {
    "nb":  {"alpha": [0.1, 0.3, 0.5, 1.0]},
    "lr":  {"C": [0.5, 1.0, 4.0, 16.0], "class_weight": [None, "balanced"]},
    "svm": {"C": [0.1, 0.3, 1.0, 3.0], "class_weight": [None, "balanced"]},
}
FEATURE_OPTIONS = [((1, 1), False), ((1, 2), False), ((1, 1), True), ((1, 2), True)]


def tune(model: str, X, y, folds, return_table: bool = False):
    """Exhaustive search over (n-gram range, hand-crafted features on/off, model hyper-parameters),
    scored by mean ROC-AUC over `folds` (training data only). Ties (within 0.002) go to the simpler option:
    no hand-crafted features, unigrams, stronger regularisation."""
    from itertools import product
    from sklearn.model_selection import ParameterGrid
    from .evaluation import cv_fold_metrics
    rows = []
    for (ng, hand), p in product(FEATURE_OPTIONS, list(ParameterGrid(GRIDS[model]))):
        r = cv_fold_metrics(build_pipeline(model, ng, hand, **p), X, y, folds, model)
        rows.append(dict(model=model, ngram=ng, hand=hand, params=p, roc_auc=r.roc_auc.mean(), f1=r.f1.mean()))
    tab = pd.DataFrame(rows)
    best_auc = tab.roc_auc.max()
    near = tab[tab.roc_auc >= best_auc - 0.002].copy()
    near["complexity"] = near.hand.astype(int) * 2 + near.ngram.map(lambda n: n[1] - 1)
    best = near.sort_values(["complexity", "roc_auc"], ascending=[True, False]).iloc[0]
    cfg = dict(model=model, ngram=list(best.ngram), hand=bool(best.hand), params=best.params)
    return (cfg, tab) if return_table else cfg
