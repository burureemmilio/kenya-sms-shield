"""Data loading, cleaning, grouping and hand-crafted features for the SMS scam project.

Design principle: anything that is *learned from data* (TF-IDF vocabulary, scalers, ...)
lives inside a scikit-learn Pipeline and is fitted on training data only.  Everything
here is either (a) deterministic cleaning, (b) a fixed, rule-based feature extractor
(nothing is fitted), or (c) a helper used only to decide how rows are *split*.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

LABEL_MAP = {"legit": 0, "scam": 1}   # positive class = scam (so recall/precision are "scam recall/precision")
RANDOM_STATE = 42


# ----------------------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Key used ONLY to detect repeated messages (not fed to the model).
    Lower-cases, collapses all whitespace/newlines and strips the ends, so that two messages
    that differ only by spacing/case are recognised as the same message."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def model_text(s: str) -> str:
    """Text actually shown to the TF-IDF model: lower-cased, with the export artefact '\\n' (a literal
    backslash + n found in ~31% of messages) and real line breaks turned into spaces.
    Without this, '\\nYOUR' would be tokenised as the junk word 'nyour'."""
    return str(s).replace("\\n", " ").replace("\n", " ").lower()


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["y"] = df["label"].map(LABEL_MAP)
    df["text_key"] = df["message_content"].map(normalize_text)
    return df


def clean_dataset(df: pd.DataFrame):
    """Return (clean_df, audit_dict).

    Decisions (explained in notebook 01):
      1. Texts that appear with BOTH labels are removed entirely - their true label is unknowable.
      2. Remaining repeated texts are collapsed to one row (keep first occurrence).
    """
    n_label = df.groupby("text_key")["label"].nunique()
    conflict_keys = set(n_label[n_label > 1].index)
    no_conflict = df[~df["text_key"].isin(conflict_keys)]
    clean = no_conflict.drop_duplicates("text_key", keep="first").reset_index(drop=True)
    audit = dict(
        raw_rows=len(df),
        conflict_texts=len(conflict_keys),
        conflict_rows=int(df["text_key"].isin(conflict_keys).sum()),
        rows_after_conflict_removal=len(no_conflict),
        clean_rows=len(clean),
    )
    return clean, audit


# ----------------------------------------------------------------------------------------
# Near-duplicate grouping (used for SPLITTING only)
# ----------------------------------------------------------------------------------------
def near_duplicate_groups(texts, threshold: float = 0.8) -> np.ndarray:
    """Connected components of the 'cosine similarity >= threshold' graph over char n-gram TF-IDF.

    SMS scams/ads are mostly sent from templates ("Dear Customer, reply with N to set ... as
    your Skiza tune") - two messages can differ by one digit and still leak into train/test.
    Grouping them and splitting by group prevents that.  This vectoriser is fitted on ALL texts
    but only decides which rows travel together; it is never used as a model feature, and it
    uses no labels, so it cannot leak label information."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform([normalize_text(t) for t in texts])
    sim = (X @ X.T).tocoo()
    parent = list(range(X.shape[0]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, v in zip(sim.row, sim.col, sim.data):
        if i < j and v >= threshold:
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(X.shape[0])])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


# ----------------------------------------------------------------------------------------
# Hand-crafted structural / lexical features
# ----------------------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|ke|co\.ke|net|org|io|me|ly|link|xyz|top|info)\b(?:/\S*)?|\bbit\.ly/\S+)", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?254|0)[17]\d{8}(?!\d)|\+\d{10,13}")
USSD_RE = re.compile(r"\*\d[\d*]*#")
MONEY_RE = re.compile(r"\b(?:ksh?s?|kes|ugx|tzs|usd|\$)\.?\s?[\d,]+(?:\.\d+)?", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
WORD_RE = re.compile(r"[a-z']+")

# Lexicons are deliberately SHORT and generic (domain knowledge, not tuned on the labels).
LEXICONS = {
    "urgency": ["urgent", "urgently", "immediately", "now", "today", "expire", "expires", "expired", "hurry",
                "asap", "deadline", "limited", "last chance", "within", "final notice", "quickly"],
    "threat": ["suspended", "blocked", "locked", "deactivated", "deactivate", "terminated", "restricted",
               "legal action", "arrest", "penalty", "unlock", "verify your", "verification"],
    "reward": ["congratulations", "congratulation", "won", "win", "winner", "prize", "bonus", "free",
               "reward", "jackpot", "credited", "cash", "claim", "earn", "salary", "hourly", "daily"],
    "payment_request": ["send", "pay", "payment", "deposit", "fee", "registration", "activation", "till",
                        "paybill", "transfer", "refund", "processing fee"],
    "job_offer": ["recruit", "recruiter", "vacancy", "hiring", "shortlisted", "applicant", "job", "part-time",
                  "whatsapp", "telegram", "interview"],
    "kiswahili_sheng": ["tafadhali", "pesa", "ndugu", "asante", "karibu", "jibu", "bonyeza", "mungu", "ujumbe",
                        "shukran", "lipa", "kutuma", "ushindi", "bure", "haraka", "wewe",
                        "yako", "tuma", "mkopo", "pata"],
    "reply_stop": ["reply stop", "reply with stop", "to stop", "unsubscribe", "opt out", "dnd"],
}
HAND_FEATURES = ["n_chars_log", "n_words_log", "n_urls", "n_digits_ratio", "n_phones", "n_ussd", "n_money",
                 "upper_ratio", "n_exclaim", "n_emoji"] + [f"lex_{k}" for k in LEXICONS]


def _lex_count(text_lower: str, tokens: set, phrases: list[str]) -> int:
    c = 0
    for p in phrases:
        if " " in p or "-" in p:
            c += p in text_lower
        else:
            c += p in tokens
    return c


def extract_hand_features(text: str) -> dict:
    t = str(text)
    low = model_text(t)
    toks = set(WORD_RE.findall(low))
    n_words = len(low.split())
    f = {
        "n_chars_log": np.log1p(len(t)),
        "n_words_log": np.log1p(n_words),
        "n_urls": len(URL_RE.findall(t)),
        "n_digits_ratio": sum(ch.isdigit() for ch in t) / max(len(t), 1),
        "n_phones": len(PHONE_RE.findall(t)),
        "n_ussd": len(USSD_RE.findall(t)),
        "n_money": len(MONEY_RE.findall(t)),
        "upper_ratio": sum(ch.isupper() for ch in t) / max(sum(ch.isalpha() for ch in t), 1),
        "n_exclaim": t.count("!"),
        "n_emoji": len(EMOJI_RE.findall(t)),
        "n_newlines": t.count("\n") + t.count("\\n"),   # EDA only - NOT a model feature (export artefact)
    }
    for k, phrases in LEXICONS.items():
        f[f"lex_{k}"] = _lex_count(low, toks, phrases)
    return f


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Rule-based features; nothing is learned, so there is nothing to leak. Scaling is done by a
    separate (fitted) scaler later in the pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [extract_hand_features(t) for t in np.asarray(X).ravel()]
        return pd.DataFrame(rows, columns=HAND_FEATURES).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(HAND_FEATURES)


def warning_indicators(text: str) -> list[str]:
    """Human-readable, rule-based warning flags for the app (descriptive, not model output)."""
    f = extract_hand_features(text)
    out = []
    if f["n_urls"]: out.append(f"Contains {int(f['n_urls'])} link(s) - do not click links from unknown senders.")
    if f["n_phones"]: out.append("Contains a phone number to call/WhatsApp.")
    if f["n_ussd"]: out.append("Contains a USSD code (*...#) - dialling these can subscribe you to paid services.")
    if f["lex_urgency"]: out.append("Urgency wording (e.g. now / today / expires).")
    if f["lex_threat"]: out.append("Threat or account-restriction wording (suspended / locked / verify).")
    if f["lex_reward"]: out.append("Reward / prize / earnings language.")
    if f["lex_payment_request"]: out.append("Mentions paying or sending money / fees.")
    if f["lex_job_offer"]: out.append("Job-offer / recruitment wording.")
    if f["n_money"]: out.append("Mentions a money amount.")
    if f["n_emoji"] >= 3: out.append("Many emojis.")
    return out
