# Explainable Machine Learning for SMS Scam Detection in the Kenyan Mobile Money Context

Binary classification (scam vs legit) of Kenyan SMS messages with Naive Bayes, Logistic Regression and Linear SVM on TF-IDF + rule-based features, with leakage-aware evaluation, threshold analysis, explainability and a Streamlit app.

**Headline result of the baseline study** (128 held-out messages, evaluated once in notebook 04; the advanced extension below re-uses the same split, see its snapshot): the selected Logistic Regression catches 96% of scams (recall 0.96, precision 0.88, F1 0.92, ROC-AUC 0.93). Conclusions are about *predicting this dataset's labels*, which are annotations rather than verified truth.

## Read the notebooks in order - each one explains every decision in Markdown
| Notebook | Phase | What it decides / shows |
|---|---|---|
| `notebooks/01_data_audit.ipynb` | 1 Data audit | 736 vs 738 discrepancy, duplicates (89 vs 91), 2 label-conflict texts, near-duplicate groups, `prelabel` leakage, `\n` artefact → cleaned dataset |
| `notebooks/02_eda.ipynb` | 2 EDA & features | Freezes the test split first; class/length/structure/lexicon analysis on training data; final feature plan |
| `notebooks/03_model_training.ipynb` | 3 Modelling | Grouped CV, feature experiment, nested-CV tuning, pre-declared model-selection rule, **leakage experiment**, threshold selection |
| `notebooks/04_evaluation_explainability.ipynb` | 4 Evaluation | One-time test evaluation, bootstrap CIs, coefficients + stability, error analysis, limitation probes, final model |
| `notebooks/05_optional_three_class.ipynb` | Optional | Label-consistency check and 3-class (legit / moderate / high) model; not deployed |
| `notebooks/06_shap_explainability.ipynb` | 6 Explainability | SHAP for the advanced binary + severity models: global importance, beeswarm, feature families, waterfalls, error analysis, app examples, bootstrap stability of the explanations |

Notebooks are shipped **already executed** (outputs and charts included). They must be run in order because each saves files used by the next.

## Structure
```
data/        scam_detection_labeled.csv (original) · cleaned_messages.csv · split.csv (train/test flag)
notebooks/   01-05 (above)
src/         preprocessing.py (cleaning, grouping, features) · train.py (pipelines, tuning) · evaluation.py · predict.py · make_figures.py · explain.py (SHAP values + plots)
models/      final_model.joblib (deployed) · dev_models.joblib (train-only, 3 models) · final_config.json · selection.json · model_card.json
reports/     final_report.md/.docx · presentation.pptx · CSV results · figures/
app.py       Streamlit app
```

## Run
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb   # optional: re-run everything, in order
python -m src.train          # re-fits the deployed model from models/final_config.json
streamlit run app.py
```

## Key decisions (details and evidence in the notebooks)
* Target `label` (scam = positive class). `prelabel` is never a feature (it contains the answer).
* Removed the 4 rows of 2 texts that carry both labels; kept one copy of other repeated texts → 643 messages.
* Splits are stratified and **grouped by near-duplicate template family**; the 20% test set was frozen before EDA and used once.
* All learned preprocessing is inside scikit-learn Pipelines (fitted on training folds only).
* Model chosen by nested-CV ROC-AUC with a pre-declared tie rule (within 1 SE → LR > SVM > NB). Threshold = highest value keeping ≥95% out-of-fold scam recall (0.32).
* Explanations use coefficient × feature value (exact for linear models). Notebook 06 adds SHAP for the advanced models (exact linear SHAP in `src/explain.py`; it is cross-checked against the `shap` package if installed).

## Caveats
* Streamlit could not be installed in the build environment, so `app.py` was syntax-checked and its prediction logic (`src/predict.py`) tested, but the UI itself has not been launched.
* The literature review in the report relies on what the project brief says about prior work; add formal citations before submission.
* The dataset contains real-looking phone numbers and names; handle it accordingly and check its licence before redistributing.


---

## KenyaSMS Shield: advanced two-stage classification

The current advanced application adds a sequential decision system:

1. **Binary classifier:** legitimate vs scam.
2. **Risk policy:** SAFE, SUSPICIOUS, or HIGH-RISK SCAM.
3. **Severity classifier:** if the binary gate flags a scam, a second classifier predicts moderate-risk vs high-risk using the source dataset's `prelabel` severity annotations.
4. **Explainability:** additive Logistic Regression feature contributions for the binary decision and severity decision, plus separate rule-based warning indicators.
5. **Kenya-aware feature engineering:** M-Pesa/Safaricom/Fuliza/paybill/till terms, Kenyan bank names, KES/KSh amounts, Kenyan phone formats, reversal language, PIN/OTP requests, USSD patterns, suspicious TLDs, and WhatsApp/Telegram redirection.
6. **NLP:** word TF-IDF (1–2 grams) + character `char_wb` TF-IDF (2–5 grams) + structural features.

### Run the upgraded app

```bash
pip install -r requirements.txt
python -m src.advanced
streamlit run app.py
```

`python -m src.advanced` trains the models and writes the evaluation report. Alternatively run:

```bash
python -c "from src.advanced import train_and_evaluate; print(train_and_evaluate())"
```

The trained bundle is saved to `models/kenya_sms_shield.joblib`; held-out metrics are written to `reports/advanced_evaluation.json`.

### Important label and score interpretation

The three output levels are not three independently verified fraud truths. `SAFE` means the binary model's score is below a conservative review band; `SUSPICIOUS` includes borderline messages and the dataset's moderate-risk scam tier; `HIGH-RISK SCAM` corresponds to the dataset's high-risk scam tier. The moderate/high labels originate from the source dataset's `prelabel` annotations. They are severity labels, not scam-family categories.

The displayed 0–100 risk score is the binary Logistic Regression score multiplied by 100. It is **not a calibrated probability**. The threshold and review band are explicit policy choices, not universal safety boundaries. Calibration and independent external validation remain necessary before operational use.

### Evaluation snapshot

The held-out split contains 128 messages. In the latest advanced pipeline run, binary ROC-AUC was about 0.940 and binary scam-class precision was about 0.839, recall 0.973 and F1 0.901 at the inherited 0.32 threshold. On all 128 held-out messages (53 safe / 16 moderate-risk / 59 high-risk), the three-class risk model achieved macro-F1 about 0.872 (the baseline three-class model of notebook 05 scored 0.84 on the same split). The binary model was augmented with 80 synthetic legitimate development messages; on a separate 40-message synthetic legitimate holdout it raised 0 false alarms (0%). This is only a synthetic stress test and must not be presented as real-world false-alarm performance. The moderate class has only 16 held-out examples, so this estimate is uncertain. See `reports/advanced_evaluation.json` for full precision/recall/support and confusion matrix. These advanced metrics were produced on the same test split already used by the baseline study, so they are exploratory and not a fresh untouched estimate. They predict source annotations, not verified real-world fraud.

### Synthetic Kenyan SMS examples

`data/synthetic_kenya_sms_examples.csv` contains 120 clearly marked synthetic examples across safe, suspicious, and high-risk scam patterns. `data/synthetic_legitimate_development.csv` (80 messages) is used only to augment binary-model training; `data/synthetic_legitimate_holdout.csv` (40 messages) is kept separate for a synthetic false-alarm stress test. It is a demonstration and manual QA resource only. It is **not** merged into the labelled research dataset, model training, or held-out evaluation. Synthetic templates cannot substitute for consented, independently verified Kenyan SMS.

### Safety, privacy, and scope

- Never share an M-Pesa PIN, bank PIN, password, or OTP.
- A text that resembles an M-Pesa receipt is not proof that a transaction occurred. Verify in the official app/account statement.
- Do not click links or call numbers supplied by a suspicious SMS. Independently find the institution's official contact route.
- The app does not verify sender IDs, inspect live domain registration/age, resolve URLs, or access telecom/bank systems.
- Do not automatically block, report, or take financial action based on this prototype.
