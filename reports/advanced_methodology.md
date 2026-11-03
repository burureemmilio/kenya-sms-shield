# KenyaSMS Shield – Advanced Methodology and Evaluation

## System design

The upgraded prototype uses a two-stage classifier with a three-level risk presentation.

- Stage 1: binary Logistic Regression separates the source labels `legit` and `scam`.
- Stage 2: a second, three-class Logistic Regression (`0` safe / `1` moderate-risk scam / `2` high-risk scam, from the source `prelabel` tiers) is trained on the same features. In the code (`assess_message`) it is evaluated for every message, and the policy layer below combines its output with the binary score. A binary scam flag is never downgraded to SAFE by this model.
- Decision policy: a low binary score is presented as SAFE; a configurable buffer below the scam threshold is presented as SUSPICIOUS; messages above the threshold receive the severity model's moderate/high result.

The binary score is the primary gate; the three-class model only refines the severity wording (moderate vs high) and can keep a borderline message out of the SAFE band. The 80 synthetic legitimate development messages are added to class 0 of both models.

## Feature representation

The feature union combines word TF-IDF (unigrams/bigrams), character-boundary TF-IDF (`char_wb`, 2–5 character n-grams), and deterministic structural features. Character n-grams are useful for spelling variation and obfuscation, but they do not guarantee robust handling of all languages or adversarial text.

Kenya-aware structural features cover M-Pesa/Safaricom/Fuliza and paybill/till references, common Kenyan bank names, Kenyan phone number formats, KES/KSh amounts, reversal requests, PIN/OTP requests, USSD codes, selected suspicious TLDs, and WhatsApp/Telegram redirection. These are indicators, not proof of fraud. A domain TLD alone must never determine a risk class.

## Explainability

For linear models, the app reports additive feature contributions on the model's decision scale (`feature value × coefficient`). Positive binary contributions move the decision toward scam; negative contributions move away from scam. The severity model has its own class-specific contributions. The app separately reports fixed rule-based warning indicators.

These explanations are local model mechanics, not causal explanations. Correlated terms can share or redistribute contribution. The feature list should be read alongside the original message and domain context.

## Held-out evaluation

The existing frozen, group-aware split is reused: 515 training rows and 128 test rows. **Important:** this is the same test set that the baseline study (notebooks 03-04) already used once, so the advanced results are a second look at it and should be read as exploratory rather than as a fresh, untouched estimate. The binary model is evaluated against the original binary labels (53 legit, 75 scam). The three-class risk model is evaluated on all 128 test messages against `prelabel` tiers 0, 1 and 2 (support 53 / 16 / 59). Models are refitted on all clean labelled messages only after held-out evaluation, for app inference.

The current run reports:
- Binary ROC-AUC: 0.940
- Binary scam-class precision: 0.839
- Binary scam-class recall: 0.973
- Binary scam-class F1: 0.901
- Binary accuracy: 0.875 (at the inherited 0.32 threshold)
- Three-class risk macro-F1 on all 128 test messages: 0.872 (confusion matrix rows safe/moderate/high: [47, 0, 6], [2, 13, 1], [4, 2, 53])

The test set contains only 16 moderate-risk and 59 high-risk scam messages, so the moderate-class estimate is uncertain. See `advanced_evaluation.json` for the complete metrics and confusion matrix.

## Risk score semantics

The app shows the binary model score on a 0–100 display scale. It is not calibrated probability. A score of 80 does not mean an 80% chance of fraud. Before using probability language, fit and evaluate calibration (e.g., sigmoid calibration with group-aware out-of-fold predictions), report Brier score and reliability curves, and validate on a separate, naturally sampled dataset.

## Limitations and next steps

1. Acquire consented, privacy-scrubbed, independently verified Kenyan SMS from M-Pesa, banks, telcos, personal messages, and scam reports.
2. Have at least two annotators label binary fraud and severity separately; report Cohen's kappa and adjudicate disagreements.
3. Build a taxonomy of scam families (phishing, impersonation, reversal fraud, fake jobs, prize scams, fake Fuliza, delivery scams) only when there are enough independently labelled examples per family. Current tier labels are severity, not scam family.
4. Run grouped repeated CV and a prospective time-based test on newly collected messages.
5. Calibrate scores on a representative validation sample and assess false-alarm rate on genuine transactional SMS.
6. Consider XGBoost/LightGBM and multilingual sentence embeddings as comparisons, not assumed improvements.
7. Domain age, URL reputation and sender identity require trusted external data and safe URL handling; they are not currently implemented.
8. Do not auto-block messages or make financial decisions from this prototype.
