"""Two-stage Kenya-focused SMS scam classification.

Stage 1: binary legitimate/scam classifier.
Stage 2: for messages flagged as scams, classify scam severity using the source
dataset's prelabel tiers (moderate vs high). The three-way user-facing output is
SAFE / SUSPICIOUS / HIGH-RISK SCAM.

Scores are model scores, not verified probabilities. Do not use as a payment
authorization or automated blocking system.
"""
from __future__ import annotations
import re, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score

from .preprocessing import model_text, clean_dataset, load_raw, extract_hand_features, HAND_FEATURES, warning_indicators

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "models" / "kenya_sms_shield.joblib"
RISK_NAMES = {0: "SAFE", 1: "SUSPICIOUS", 2: "HIGH-RISK SCAM"}
SCAM_TYPES = ["moderate-risk scam", "high-risk scam"]

KENYA_PATTERNS = {
    "mpesa_reference": r"\b[a-z0-9]{10}\b",
    "mpesa_terms": r"\b(mpesa|m-pesa|safaricom|fuliza|pochi|bonga|till number|paybill)\b",
    "bank_terms": r"\b(kcb|equity|coop(?:erative)? bank|absa|ncba|stanbic|family bank|dtb|i&m)\b",
    "money_direction_send": r"\b(send|tuma|lipa|pay|transfer|nitumie|nitumizie)\b",
    "money_direction_receive": r"\b(received|you have received|credited|imelipwa|umepokea|imeingia)\b",
    "reversal_request": r"\b(reverse|reversal|wrong number|sent by mistake|mistakenly sent|rudisha|nirudishie|rejesha)\b",
    "credential_request": r"\b(pin|password|passcode|otp|one.time.password|verification code)\b",
    "identity_impersonation": r"\b(customer care|safaricom agent|bank officer|mpesa agent|support team)\b",
    "domain_tld_risk": r"\.(?:xyz|top|click|work|zip|mov|gq|tk|ml|cf)\b",
    "whatsapp_redirect": r"\b(whatsapp|telegram|wa\.me)\b",
    "mpesa_receipt_structure": r"(?is)(?=.*\b[a-z0-9]{10}\b)(?=.*\bconfirmed\b)(?=.*\bbalance\b)(?=.*\btransaction cost\b)",
}
class AdvancedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows=[]
        for raw in np.asarray(X,dtype=object).ravel():
            t=str(raw); low=model_text(t)
            base=extract_hand_features(t)
            row=[base.get(k,0) for k in HAND_FEATURES]
            row += [int(bool(re.search(p,low,re.I))) for p in KENYA_PATTERNS.values()]
            rows.append(row)
        return np.asarray(rows,dtype=float)
    def get_feature_names_out(self,input_features=None):
        return np.array(HAND_FEATURES+["ke_"+k for k in KENYA_PATTERNS])

def build_advanced_pipeline(multiclass=False):
    word=TfidfVectorizer(preprocessor=model_text,lowercase=False,ngram_range=(1,2),min_df=2,
                         sublinear_tf=True,token_pattern=r"(?u)\b\w+\b")
    char=TfidfVectorizer(preprocessor=model_text,lowercase=False,analyzer="char_wb",
                         ngram_range=(2,5),min_df=2,sublinear_tf=True,max_features=50000)
    feats=FeatureUnion([
        ("word",word),("char",char),
        ("hand",Pipeline([("features",AdvancedFeatures()),("scale",MaxAbsScaler())]))
    ],transformer_weights={"word":1.0,"char":0.8,"hand":0.7})
    clf=LogisticRegression(C=4.0,max_iter=3000,class_weight="balanced",
                           solver="lbfgs" if multiclass else "liblinear",
                           random_state=42)
    return Pipeline([("features",feats),("clf",clf)])

def train_and_evaluate():
    raw=load_raw(ROOT/"data"/"scam_detection_labeled.csv")
    clean,audit=clean_dataset(raw)
    clean["tier"]=clean["prelabel"].astype(str).str[0].astype(int)
    split=pd.read_csv(ROOT/"data"/"split.csv").drop_duplicates("text_key")[["text_key","split","group"]]
    d=clean.merge(split,on="text_key",how="inner",suffixes=("","_split"))
    train=d[d.split=="train"].copy(); test=d[d.split=="test"].copy()

    synth_train=pd.read_csv(ROOT/"data"/"synthetic_legitimate_development.csv")
    synth_holdout=pd.read_csv(ROOT/"data"/"synthetic_legitimate_holdout.csv")

    Xtr=np.concatenate([train.message_content.to_numpy(),synth_train.message_content.to_numpy()])
    ytr=np.concatenate([train.y.to_numpy(),np.zeros(len(synth_train),dtype=int)])
    Xte=test.message_content.to_numpy()
    yte=test.y.to_numpy()
    binary=build_advanced_pipeline(False).fit(Xtr,ytr)
    bscore=binary.predict_proba(Xte)[:,1]
    threshold=0.32
    bpred=(bscore>=threshold).astype(int)
    synthetic_scores=binary.predict_proba(synth_holdout.message_content.to_numpy())[:,1]

    # True three-class risk model: source tier 0=legit, 1=moderate-risk scam, 2=high-risk scam.
    # Synthetic legitimate development examples augment class 0 only.
    y3_train=np.concatenate([train.tier.to_numpy(),np.zeros(len(synth_train),dtype=int)])
    severity=build_advanced_pipeline(True).fit(Xtr,y3_train)
    y3_test=test.tier.to_numpy()
    pred3=severity.predict(test.message_content.to_numpy())
    metrics={
      "audit":audit,"train_rows":len(train),"test_rows":len(test),
      "binary_threshold":threshold,
      "synthetic_legitimate_holdout":{"n":int(len(synth_holdout)),
          "false_alarms":int(np.sum(synthetic_scores>=threshold)),
          "false_alarm_rate":float(np.mean(synthetic_scores>=threshold)),
          "warning":"Synthetic stress test only; not a representative real-world estimate."},
      "binary_test":{"classification_report":classification_report(yte,bpred,output_dict=True,zero_division=0),
                     "roc_auc":float(roc_auc_score(yte,bscore))},
      "multiclass_risk_test":{"classification_report":classification_report(y3_test,pred3,labels=[0,1,2],
                       target_names=["safe","suspicious_moderate_scam","high_risk_scam"],output_dict=True,zero_division=0),
                       "macro_f1":float(f1_score(y3_test,pred3,labels=[0,1,2],average="macro",zero_division=0)),
                       "confusion_matrix":confusion_matrix(y3_test,pred3,labels=[0,1,2]).tolist()},
      "risk_policy":{"safe_below":round(threshold*0.65,4),"suspicious_from":round(threshold*0.65,4),
                     "high_risk_from":threshold,
                     "note":"Raw Logistic Regression score; not calibrated probability. Severity tier is a second model."}
    }
    # Refit for deployment only after recording untouched test results. Synthetic development
    # examples augment the binary model; synthetic holdout is never used for fitting.
    binary_full=build_advanced_pipeline(False).fit(
        np.concatenate([clean.message_content.to_numpy(),synth_train.message_content.to_numpy()]),
        np.concatenate([clean.y.to_numpy(),np.zeros(len(synth_train),dtype=int)]))
    y3_all=np.concatenate([clean.tier.to_numpy(),np.zeros(len(synth_train),dtype=int)])
    severity_full=build_advanced_pipeline(True).fit(
        np.concatenate([clean.message_content.to_numpy(),synth_train.message_content.to_numpy()]),y3_all)
    joblib.dump({"binary":binary_full,"severity":severity_full,
                 "config":{"threshold":threshold,"risk_policy":metrics["risk_policy"],
                           "binary_classes":["legit","scam"],"severity_classes":["safe","suspicious / moderate-risk scam","high-risk scam"],
                           "training_rows":len(clean),"synthetic_binary_augmentation":len(synth_train),
                           "feature_families":["word TF-IDF 1-2","character TF-IDF char_wb 2-5","Kenya-aware structural features"],
                           "score_warning":"Scores are not calibrated probabilities."}},BUNDLE_PATH)
    (ROOT/"reports"/"advanced_evaluation.json").write_text(json.dumps(metrics,indent=2))
    return metrics

def explain_linear(pipe,text,k=8,class_index=1,reverse=False):
    """Exact additive feature contributions on the model's linear decision scale."""
    feats=pipe.named_steps["features"]; clf=pipe.named_steps["clf"]
    names=np.asarray(feats.get_feature_names_out())
    x=feats.transform([text])
    coef=clf.coef_
    if coef.shape[0]==1: w=coef[0]
    else: w=coef[class_index]
    contrib=x.multiply(w).tocsr()
    pairs=[]
    for idx,val in zip(contrib.indices,contrib.data):
        if abs(float(val))>1e-10:
            name=str(names[idx])
            pairs.append({"feature":name,"contribution":float(val)})
    pairs.sort(key=lambda z:abs(z["contribution"]),reverse=True)
    return pairs[:k]

def assess_message(text,bundle=None):
    if bundle is None: bundle=joblib.load(BUNDLE_PATH)
    bp=bundle["binary"]; sp=bundle["severity"]; threshold=bundle["config"]["threshold"]
    score=float(bp.predict_proba([text])[0,1])
    # Borderline buffer is intentionally explicit and configurable.
    safe_cut=threshold*0.65
    multiclass_prediction=int(sp.predict([text])[0])
    if score < safe_cut and multiclass_prediction == 0:
        tier=0; risk="SAFE"
    elif score < threshold:
        tier=1; risk="SUSPICIOUS"
    else:
        # A binary scam flag is never downgraded to SAFE by the second-stage model.
        tier=2 if multiclass_prediction==2 else 1
        risk=RISK_NAMES[tier]
    binary_explain=explain_linear(bp,text,8)
    class_idx=2 if tier==2 else 1
    severity_explain=explain_linear(sp,text,6,class_index=class_idx) if tier>0 else []
    features=AdvancedFeatures().transform([text])[0]
    names=AdvancedFeatures().get_feature_names_out()
    structured={str(n):float(v) for n,v in zip(names,features) if v>0}
    return {"risk_tier":tier,"risk_label":risk,"risk_score":round(score*100,1),
            "binary_score":score,"threshold":threshold,"indicators":warning_indicators(text),
            "kenya_features":structured,"binary_explanation":binary_explain,
            "severity_explanation":severity_explain,
             "severity_prediction":("safe" if tier==0 else
                "high-risk scam" if tier==2 else
                "suspicious / moderate-risk scam" if multiclass_prediction==1 else
                "suspicious / manual review (model disagreement)"),
            "disclaimer":"Risk score is an uncalibrated model score, not the probability that a message is fraudulent."}


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
