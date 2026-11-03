"""Export the key figures used in the report and slides to reports/figures/."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from .evaluation import threshold_table
from .train import scam_score

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "reports" / "figures"; OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
d = pd.read_csv(ROOT / "data" / "split.csv"); te = d[d.split == "test"]
dev = joblib.load(ROOT / "models" / "dev_models.joblib"); lr = dev["lr"]
s = scam_score(lr["pipeline"], te.message_content.values); thr = lr["config"]["threshold"]

# 1 leakage
leak = pd.read_csv(ROOT / "reports" / "leakage_experiment.csv")
order = ["P1 naive (raw, random)", "P2 exact-dedup groups (raw)", "P3 deduped, random folds", "P4 deduped + near-dup groups"]
fig, ax = plt.subplots(figsize=(8, 3.8)); sns.barplot(data=leak, x="protocol", y="f1", hue="model", order=order, ax=ax, palette="Set2")
ax.set_ylim(0.82, 0.95); ax.set_xlabel(""); ax.set_ylabel("cross-validated F1"); ax.set_title("Stricter evaluation lowers F1 (leakage inflation)"); ax.tick_params(axis="x", rotation=12, labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_leakage.png", dpi=150); plt.close(fig)

# 2 confusion
fig, ax = plt.subplots(figsize=(3.8, 3.6)); ConfusionMatrixDisplay.from_predictions(te.y, (s >= thr).astype(int), display_labels=["legit", "scam"], ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Logistic Regression - held-out test\n(deployed threshold)", fontsize=10); ax.grid(False); fig.tight_layout(); fig.savefig(OUT / "fig_confusion.png", dpi=150); plt.close(fig)

# 3 coefficients
pipe = lr["pipeline"]; names = np.array([n.replace("tfidf__", "").replace("hand__", "[feat] ") for n in pipe.named_steps["features"].get_feature_names_out()])
coef = pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=names).sort_values(); top = pd.concat([coef.head(12), coef.tail(12)])
fig, ax = plt.subplots(figsize=(6, 5)); ax.barh(top.index, top.values, color=np.where(top.values > 0, "#c8553d", "#4c9f70")); ax.set_title("Strongest features (right = scam, left = legit)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_coefficients.png", dpi=150); plt.close(fig)

# 4 threshold
tt = threshold_table(te.y.values, s, np.linspace(0.02, 0.98, 97))
fig, ax = plt.subplots(figsize=(6.4, 3.6)); ax.plot(tt.threshold, tt.missed_scams, color="#c8553d", label="missed scams"); ax.plot(tt.threshold, tt.false_alarms, color="#336", label="false alarms")
ax.axvline(thr, color="k", ls=":", label=f"deployed threshold {thr:.2f}"); ax.set_xlabel("threshold"); ax.set_ylabel("count (128 test messages)"); ax.legend(); ax.set_title("Threshold trade-off (descriptive, test set)")
fig.tight_layout(); fig.savefig(OUT / "fig_threshold.png", dpi=150); plt.close(fig)

# 5 CV model comparison
nf = pd.read_csv(ROOT / "reports" / "nested_cv_folds.csv"); g = nf.groupby("model").roc_auc.agg(["mean", "std"]).rename(index={"nb": "Naive Bayes", "lr": "Logistic Reg.", "svm": "Linear SVM"})
fig, ax = plt.subplots(figsize=(4.6, 3.4)); ax.bar(g.index, g["mean"], yerr=g["std"], color=["#e1812c", "#3274a1", "#3a923a"], capsize=4); ax.set_ylim(0.85, 1); ax.set_ylabel("nested-CV ROC-AUC"); ax.set_title("Model comparison (training data)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_model_comparison.png", dpi=150); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
