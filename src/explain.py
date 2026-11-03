"""SHAP-style explanations for the (linear) KenyaSMS Shield models.

Why this module exists
----------------------
Both deployed classifiers are Logistic Regressions on a FeatureUnion (word TF-IDF + char TF-IDF + rule-based
Kenya-aware features).  For a linear model f(x) = w.x + b the SHAP value of feature i under the
*interventional* (independent-features) formulation has a closed form:

        phi_i(x) = w_i * (x_i - E_background[x_i])          and        f(x) = E[f] + sum_i phi_i(x)

which is exactly what ``shap.LinearExplainer(model, background)`` returns.  Computing it directly keeps the
notebook fast and lets it run without the optional ``shap`` dependency; the notebook still cross-checks
against the ``shap`` library whenever it is installed (``crosscheck_with_shap``).

All values are on the model's *log-odds* (logit) scale - the scale on which the model is additive.  The deployed
"risk score" is sigmoid(logit) x 100 and the deployed threshold 0.32 corresponds to logit ln(0.32/0.68).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse

# Colour-blind-safe pair used throughout: blue = pushes towards LEGIT, orange/red = pushes towards SCAM
C_SCAM = "#d1495b"
C_LEGIT = "#2e6f9e"
C_GREY = "#8d99a6"
CMAP = LinearSegmentedColormap.from_list("legit_scam", [C_LEGIT, "#e9e4f0", C_SCAM])

FAMILY_ORDER = ["word n-gram", "char n-gram", "lexicon count", "structural", "Kenya-aware rule"]
FAMILY_COLORS = {"word n-gram": "#4c78a8", "char n-gram": "#9ecae9", "lexicon count": "#f58518",
                 "structural": "#b6992d", "Kenya-aware rule": "#54a24b"}
LEX_FEATURES = ("lex_",)


# ----------------------------------------------------------------------------------------------
# Feature naming
# ----------------------------------------------------------------------------------------------
def feature_family(raw_name: str) -> str:
    """Map a FeatureUnion output name (e.g. 'char__ ur', 'hand__ke_mpesa_terms') to an interpretable family."""
    if raw_name.startswith("word__"):
        return "word n-gram"
    if raw_name.startswith("char__"):
        return "char n-gram"
    short = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
    if short.startswith("ke_"):
        return "Kenya-aware rule"
    if short.startswith(LEX_FEATURES):
        return "lexicon count"
    return "structural"


def pretty_name(raw_name: str) -> str:
    """Readable label: words in plain text, char n-grams in quotes (spaces shown as '_'), rules in [brackets]."""
    fam = feature_family(raw_name)
    short = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
    if fam == "word n-gram":
        return short
    if fam == "char n-gram":
        return "chars ‘" + short.replace(" ", "_") + "’"
    return f"[{short}]"


# ----------------------------------------------------------------------------------------------
# SHAP values for a linear Pipeline(features -> clf)
# ----------------------------------------------------------------------------------------------
@dataclass
class LinearShap:
    values: np.ndarray        # (n, p) dense SHAP values, log-odds scale
    data: np.ndarray          # (n, p) feature values in MODEL space (after weighting/scaling)
    base_value: float         # E_background[f(x)]  (mean logit over the background set)
    logit: np.ndarray         # (n,) model output f(x) on the logit scale
    names: np.ndarray         # raw feature names
    pretty: np.ndarray        # readable feature names
    family: np.ndarray        # family per feature

    @property
    def prob(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit))

    def additivity_error(self) -> float:
        return float(np.max(np.abs(self.values.sum(1) + self.base_value - self.logit)))

    def global_importance(self) -> pd.DataFrame:
        """Mean |SHAP| per feature, mean signed SHAP, and the share of messages in which the feature is present (x != 0)."""
        a = np.abs(self.values)
        df = pd.DataFrame({"feature": self.pretty, "raw_name": self.names, "family": self.family,
                           "mean_abs_shap": a.mean(0), "mean_shap": self.values.mean(0),
                           "present_in": (self.data != 0).mean(0)})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def family_contribution(self) -> pd.DataFrame:
        """Per-message sum of SHAP values within each family (n x families)."""
        out = {}
        for fam in FAMILY_ORDER:
            m = self.family == fam
            out[fam] = self.values[:, m].sum(1)
        return pd.DataFrame(out)


def _logit_parameters(pipe, class_index=None):
    clf = pipe.named_steps["clf"]
    coef, intercept = np.atleast_2d(clf.coef_), np.atleast_1d(clf.intercept_)
    k = 0 if (coef.shape[0] == 1 or class_index is None) else int(class_index)
    # sklearn binary LR stores ONE row (the positive class); a multinomial LR stores one row per class.
    return coef[k], float(intercept[k])


def linear_shap(pipe, X_background, X_explain, class_index=None) -> LinearShap:
    """Exact interventional SHAP values for a fitted Pipeline([('features', FeatureUnion), ('clf', LogisticRegression)]).

    class_index : for a multinomial model, which class's logit to explain (ignored for binary models).
    """
    feats = pipe.named_steps["features"]
    w, b = _logit_parameters(pipe, class_index)
    Xb = feats.transform(np.asarray(X_background, dtype=object))
    Xe = feats.transform(np.asarray(X_explain, dtype=object))
    mu = np.asarray(Xb.mean(0)).ravel() if sparse.issparse(Xb) else Xb.mean(0)
    Xe = Xe.toarray() if sparse.issparse(Xe) else np.asarray(Xe)
    values = (Xe - mu) * w
    names = np.asarray(feats.get_feature_names_out())
    return LinearShap(values=values, data=Xe, base_value=float(mu @ w + b), logit=Xe @ w + b, names=names,
                      pretty=np.array([pretty_name(n) for n in names]),
                      family=np.array([feature_family(n) for n in names]))


def crosscheck_with_shap(pipe, X_background, X_explain, ours: LinearShap, class_index=None, tol=1e-6) -> dict:
    """Compare our closed-form values with ``shap.LinearExplainer`` (only if the optional `shap` package is installed)."""
    import shap  # noqa: optional dependency
    feats = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    Xb = feats.transform(np.asarray(X_background, dtype=object))
    Xe = feats.transform(np.asarray(X_explain, dtype=object))
    if np.atleast_2d(clf.coef_).shape[0] > 1:  # multinomial: explain one class's logit with an equivalent binary view
        from sklearn.linear_model import LogisticRegression
        w, b = _logit_parameters(pipe, class_index)
        lr = LogisticRegression(); lr.coef_ = w[None, :]; lr.intercept_ = np.array([b]); lr.classes_ = np.array([0, 1])
        clf = lr
    ex = shap.LinearExplainer(clf, Xb, feature_perturbation="interventional")
    sv = ex.shap_values(Xe)
    sv = sv.toarray() if sparse.issparse(sv) else np.asarray(sv)
    return dict(shap_version=shap.__version__, max_abs_diff=float(np.max(np.abs(sv - ours.values))),
                base_diff=float(abs(np.ravel(ex.expected_value)[0] - ours.base_value)),
                agrees=bool(np.max(np.abs(sv - ours.values)) < tol))


# ----------------------------------------------------------------------------------------------
# Plots (matplotlib, so they render identically with or without the `shap` package)
# ----------------------------------------------------------------------------------------------
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def plot_bar(sv: LinearShap, top=20, ax=None, title="Mean |SHAP value| (log-odds)"):
    """Global importance bar, coloured by feature family."""
    g = sv.global_importance().head(top).iloc[::-1]
    ax = ax or plt.subplots(figsize=(7, 0.32 * top + 1.2))[1]
    ax.barh(g.feature, g.mean_abs_shap, color=[FAMILY_COLORS[f] for f in g.family])
    ax.set_xlabel(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER]
    ax.legend(handles, FAMILY_ORDER, loc="lower right", frameon=False, fontsize=8)
    _style(ax)
    return ax


def plot_beeswarm(sv: LinearShap, top=15, ax=None, seed=0):
    """SHAP summary (beeswarm): one dot per message; x = effect on log-odds, colour = feature value (low->high)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-np.abs(sv.values).mean(0))[:top][::-1]
    ax = ax or plt.subplots(figsize=(7.5, 0.42 * top + 1.4))[1]
    for row, j in enumerate(order):
        phi, x = sv.values[:, j], sv.data[:, j]
        # vertical jitter proportional to local density of points along x
        bins = np.digitize(phi, np.histogram_bin_edges(phi, bins=40))
        dens = pd.Series(bins).map(pd.Series(bins).value_counts()).to_numpy()
        jit = (rng.uniform(-1, 1, len(phi)) * 0.38 * np.minimum(dens / max(dens.max(), 1), 1))
        rng_x = np.ptp(x)
        c = (x - x.min()) / rng_x if rng_x > 0 else np.full(len(x), 0.5)
        ax.scatter(phi, row + jit, c=c, cmap=CMAP, vmin=0, vmax=1, s=9, alpha=0.85, linewidths=0)
    ax.axvline(0, color="#444", lw=0.8)
    # robust x-range: a single extrapolating message (e.g. a count far above anything seen in training) must not flatten the rest
    allv = sv.values[:, order].ravel()
    lo, hi = np.percentile(allv, [0.5, 99.5]); pad = 0.08 * (hi - lo)
    out = int(((allv < lo - pad) | (allv > hi + pad)).sum())
    ax.set_xlim(min(lo - pad, -0.1), hi + pad)
    if out:
        ax.text(0.99, 0.01, f"{out} point(s) beyond the axis (largest {np.abs(allv).max():+.1f})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color="#555")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(sv.pretty[order])
    ax.set_xlabel("SHAP value  (← towards LEGIT      towards SCAM →)")
    sm = plt.cm.ScalarMappable(cmap=CMAP); sm.set_array([0, 1])
    cb = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.04, ticks=[0, 1]); cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("feature value", fontsize=8); cb.outline.set_visible(False)
    _style(ax)
    return ax


def plot_waterfall(sv: LinearShap, i: int, top=10, ax=None, threshold_logit=None, title=None):
    """Waterfall for message i: starts at E[f(x)] and walks, feature by feature, to this message's f(x)."""
    phi = sv.values[i]
    order = np.argsort(-np.abs(phi))
    shown = order[:top]
    rest = phi[order[top:]].sum()
    n_rest = int((np.abs(phi[order[top:]]) > 1e-12).sum())
    labels = [f"{sv.pretty[j]}" for j in shown] + ([f"{n_rest} other features"] if n_rest else [])
    deltas = list(phi[shown]) + ([rest] if n_rest else [])
    ax = ax or plt.subplots(figsize=(7.2, 0.36 * len(deltas) + 1.5))[1]
    start = sv.base_value
    cum = start
    for k, d in enumerate(deltas):
        ax.barh(k, d, left=cum, color=C_SCAM if d > 0 else C_LEGIT, height=0.62)
        span = max(abs(sv.logit[i] - sv.base_value), 1.0)
        if abs(d) > 0.45 * span:   # long bar: write the value inside it
            ax.text(cum + d / 2, k, f"{d:+.2f}", va="center", ha="center", fontsize=8, color="white")
        else:
            ax.text(cum + d + (0.04 * span if d >= 0 else -0.04 * span), k, f"{d:+.2f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8)
        cum += d
    ax.axvline(start, color=C_GREY, ls=":", lw=1); ax.axvline(cum, color="#222", lw=1)
    if threshold_logit is not None:
        ax.axvline(threshold_logit, color="#222", ls="--", lw=1)
        ax.text(threshold_logit, 0.02, "flag threshold ", transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="bottom", fontsize=7.5, color="#222")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel(f"log-odds of scam   (start = average message {start:+.2f}  →  this message {cum:+.2f}, "
                  f"score {100 / (1 + np.exp(-cum)):.0f}/100)", fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    return ax
