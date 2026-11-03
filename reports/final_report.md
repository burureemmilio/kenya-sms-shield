# Explainable Machine Learning for SMS Scam Detection in the Kenyan Mobile Money Context

*Final project report. All numbers come from the executed notebooks in `notebooks/` and the CSV files in `reports/`.*

## 1. Summary
We built and compared three text classifiers (Multinomial Naive Bayes, Logistic Regression, Linear SVM) that predict whether an SMS in a Kenyan dataset is labelled *scam* or *legit*. The main emphasis was **rigorous, leakage-aware evaluation** and **interpretable predictions**. On a held-out test set of 128 messages, the selected Logistic Regression caught **96% of scams** (recall 0.96) with precision 0.88, F1 0.92 and ROC-AUC 0.93. The three models were statistically hard to tell apart except that Naive Bayes was weaker at the chosen operating point. Evaluating naively (with duplicate messages) overstated F1 by about 1.3-1.6 points. Many remaining "errors" look like label noise or very short link-free scams. The conclusions concern *predicting this dataset's labels*; real-world effectiveness needs further validation.

## 2. Research questions and where they are answered
| # | Research question | Answer | Notebook |
|---|---|---|---|
| 1 | How well can ML separate scam from legit SMS in this dataset? | Test recall 0.96, precision 0.88, F1 0.92, AUC 0.93 (F1 95% CI 0.87-0.96) | 04 |
| 2 | Which of NB / LR / Linear SVM is best under identical splits? | LR ~ SVM (nested-CV AUC 0.952 each) > NB (0.924). LR vs SVM indistinguishable; LR chosen for interpretability | 03, 04 |
| 3 | How to balance detecting scams vs false alarms? | Threshold chosen on validation data to keep >=95% recall: 0.32. On test: 3 missed scams, 10 false alarms (default 0.5: 6 and 7) | 03, 04 |
| 4 | Which words/features drive predictions? | Link count, urgency and threat wording, account/free/gift words push to scam; informal personal-chat vocabulary and emojis push to legit; stable under bootstrap | 02, 04 |
| 5 | How reliable on unseen messages with duplicates/leakage controlled? | Naive protocol inflates F1 by ~1.5 points; conclusions robust to the near-duplicate threshold; remaining errors analysed | 03, 04 |

## 3. Related work (as given in the project brief)
The brief notes that this dataset comes from an existing research project on Kenyan SMS scam detection that used Logistic Regression, XGBoost and SHAP. This project builds on it with a different question - *how trustworthy are the results once evaluation is made rigorous?* - rather than reproducing its models. (Literature was not re-searched in this offline build; a full literature review with citations should be added before submission.)

## 4. Data and preparation (Notebook 01)
* **File:** 736 rows, 3 columns (`message_content`, `prelabel`, `label`); 425 scam / 311 legit. The published description reports 738 / 427; the 2-scam difference could not be reconciled offline and is documented as a limitation.
* **Duplicates:** 89 fully duplicated rows and 91 repeated texts; the gap of 2 is two texts annotated with *both* labels. Those 4 rows were removed (true label unknowable). Other repeats were collapsed to one copy. **Result: 643 unique messages (376 scam, 267 legit).**
* **Near-duplicates:** template families (e.g. the same promotion with a different number) were grouped with character n-gram cosine similarity >= 0.8 (599 groups) and kept together in every split.
* **Other findings:** a literal `\n` export artefact (31% of messages; 42% of scams vs 14% of legit) was neutralised so the model cannot use it as a shortcut; `prelabel` perfectly determines `label`, so it was never used as a feature; three template families carry inconsistent labels (label noise).
* **Class imbalance (58/42)** is mild; no resampling was used.

## 5. Exploratory analysis and features (Notebook 02)
The test set (20%, stratified and grouped) was frozen *before* exploration. On training data: scams are longer, contain links, money amounts, capitals, exclamation marks, urgency and reward wording far more often; emojis, opt-out wording and Skiza-tune/inspirational vocabulary are more common in legit messages; Kiswahili/Sheng usage is about equal in both classes. M-Pesa-type words mostly occur in scams in this data, so genuine receipts are at risk of false alarms. Features: TF-IDF word unigrams/bigrams and 17 rule-based features (length, links, digits, phone numbers, USSD codes, money amounts, capitals, exclamations, emojis, and seven lexicon families). Prelabel, the newline artefact, stop-word removal and stemming were deliberately excluded.

## 6. Methodology (Notebook 03)
* **Pipelines:** all learned preprocessing (TF-IDF, scaling) lives inside scikit-learn `Pipeline`s, fitted only on training folds.
* **Validation:** stratified + group 5-fold CV, repeated; nested CV for an honest estimate of "tune then fit".
* **Tuning:** small explicit grids over n-gram range, handcrafted features on/off, and model hyper-parameters; criterion ROC-AUC, ties to the simpler option. Every model chose **unigrams + handcrafted features**; bigrams did not pay off with ~500 messages.
* **Selection rule (pre-declared):** best nested-CV AUC; within one standard error prefer LR > SVM > NB. SVM was nominally best (0.9523 vs 0.9516) but within 1 SE, so **LR was selected**.
* **Threshold:** the highest threshold with >=95% out-of-fold scam recall (0.32 for LR), because in a warn-only app a missed scam costs more than a false alarm.

## 7. Results (Notebook 04)
| Model (selected threshold) | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Naive Bayes | 0.96 | 0.80 | 0.87 | 0.91 |
| **Logistic Regression** | **0.96** | **0.88** | **0.92** | **0.93** |
| Linear SVM | 0.96 | 0.88 | 0.92 | 0.94 |

Test set: 128 messages, evaluated once. LR minus NB in F1 is +0.044 (95% CI 0.010 to 0.082); in AUC the gap is borderline (+0.025, CI -0.005 to 0.059). LR and SVM are indistinguishable.

**Leakage study (RQ5).** Cross-validated F1 for LR: 0.923 (naive, with duplicates) -> 0.915 (grouped by exact text) -> 0.909 (de-duplicated) -> 0.908 (de-duplicated + near-duplicate groups). The inflation is real but modest because only ~13% of rows are exact repeats and the models are linear. Results barely change across near-duplicate thresholds 0.6-0.95.

**Explainability.** The strongest scam features are link count, urgency wording, threat wording and words such as `system`, `free`, `account`, `https`, `gift`; the strongest legit features are informal personal-conversation words (`niko`, `nitumie`, `leo`...) and emoji count. Some lexicon words (`send`, `job`) point to *legit* because context matters. Kiswahili is not itself a signal. Ninety per cent of the 30 strongest features keep their sign in >=90% of bootstrap refits.

**Error analysis.** Missed scams are short, contain no link or money amount, and 60% are "moderate-risk" (vs 24% of all scams). Many false alarms look like genuine scams labelled legit (prize promotions, fake M-Pesa credits, "download TG" job offers), suggesting label noise; measured precision is therefore probably conservative.

**Optional three-class model (Notebook 05).** Test macro-F1 0.84 (baseline 0.21); the moderate-risk class is weakest (F1 0.73, only 16 test examples). It was not deployed.

## 8. Deployment (app.py)
A Streamlit app lets a user paste an SMS and see the predicted class, the model score against the threshold, rule-based warning indicators, the words that moved this prediction (coefficient x feature value, the exact decomposition for a linear model) and a limitations panel. The final model is refitted on all 643 messages with the validation-chosen threshold.

## 9. Limitations
* Labels are annotations, not verified ground truth; conclusions are about predicting them.
* Only 643 distinct messages; the test set has 128 so intervals are wide.
* The class ratio is a collection artefact; real-world precision would be lower.
* Genuine M-Pesa receipts and bank alerts may be flagged; short link-free social-engineering scams may be missed.
* The "legit" class is partly a specific family (Skiza/devotional promos), which the model can exploit; scammers can imitate it.
* Hyper-parameters and the threshold were tuned on the same training messages, so out-of-fold scores are slightly optimistic.
* Static text only: no sender, time, or user context. Scam wording drifts quickly.
* The 736 vs 738 discrepancy with the published description is unresolved. SHAP was not used (not installable in the build environment); coefficients are the exact decomposition for the chosen linear model.

## 10. Ethical considerations
* **Privacy:** messages contain names, phone numbers and account references. The app stores nothing; the dataset should not be redistributed publicly beyond what its licence allows.
* **Harm of errors:** false alarms can harm legitimate senders and users' trust; missed scams can cost money. The app therefore says "possibly a scam" and never blocks.
* **Fairness:** the model must not treat Kiswahili or Sheng as suspicious; the analysis shows language is not the signal, but this should be re-checked on new data.
* **Over-reliance:** a "legit" verdict is not a guarantee of safety.
* **Dual use:** published features could help scammers evade detection; the system is for user-side warning, with limitations stated openly.

## 11. Future work
Verified multi-annotator labels and more genuine M-Pesa/bank messages; sender and context features; comparison with XGBoost and SHAP; evaluation on a time-ordered split; multilingual and character-level models; a documented definition of the moderate-risk tier; a user study of the warnings.


## 12. Advanced KenyaSMS Shield extension (2026-11)

The upgraded application now uses a sequential two-stage design. Stage 1 performs binary legitimate/scam detection. A policy layer presents low-scoring messages as SAFE and borderline messages as SUSPICIOUS. When the binary score crosses the scam threshold, Stage 2 predicts the source annotation severity: moderate-risk scam or high-risk scam. Thus the user-facing output has three risk levels, but the underlying architecture is binary-first followed by scam-severity classification.

The advanced model combines word TF-IDF (1–2 grams), character `char_wb` TF-IDF (2–5 grams), and Kenya-aware features for M-Pesa/Safaricom/Fuliza, Kenyan banks, KES/KSh, phone numbers, reversal requests, PIN/OTP requests, USSD, selected TLD patterns, WhatsApp/Telegram redirects, and a structured M-Pesa receipt pattern. Receipt structure is a learned feature, never a whitelist or automatic safe override.

The binary model is augmented with 80 synthetic legitimate development messages. A physically separate 40-message synthetic legitimate holdout produced 0 false alarms at the 0.32 threshold (0/40). This is a synthetic stress test only and must not be described as a real-world false-alarm rate. On the original 128-message grouped test set, the updated binary classifier achieved ROC-AUC 0.940, scam precision 0.839, scam recall 0.973 and scam F1 0.901. The three-class risk model achieved macro-F1 0.872 on the 128-message test set (safe support 53; moderate-risk support 16; high-risk support 59). Full details are in `advanced_evaluation.json`.

The Streamlit app now shows the risk tier, raw score, local linear-model feature contributions, Kenyan structural indicators and practical safety guidance. An optional FastAPI service is included. The risk score is uncalibrated and no output verifies a sender, transaction or URL. The model card and methodology report document these boundaries.

**Reconciliation note.** The baseline study (sections 1-11) and this extension are two different systems evaluated on the same 128-message test split. Baseline Logistic Regression (`models/final_model.joblib`, notebook 04): recall 0.96, precision 0.878, F1 0.917, ROC-AUC 0.933, accuracy 0.898. Advanced binary model (`models/kenya_sms_shield.joblib`, `reports/advanced_evaluation.json`): recall 0.973, precision 0.839, F1 0.901, ROC-AUC 0.940, accuracy 0.875. Because the test split was already used once by the baseline, the advanced figures are exploratory. The three-class macro-F1 is 0.84 for the baseline notebook-05 model and 0.872 for the advanced model. Notebook 05 concluded that tier labels were not reliable enough to show end users; the advanced app shows them only as an experimental output, and the moderate-risk tier rests on just 16 test messages.
