# Model Card: KenyaSMS Shield

## Intended use
Educational and portfolio research prototype that analyses pasted SMS text and provides a non-binding risk warning in a Kenyan mobile-money context.

## Not intended for
Automatic SMS blocking, payment approval, account access, law-enforcement decisions, credit decisions, or definitive fraud verification.

## Architecture
- Binary Logistic Regression: `legit` vs `scam`.
- Three-class Logistic Regression: `safe` vs `moderate-risk scam` vs `high-risk scam`, used alongside the binary gate: the binary score decides whether a message is flagged, and this model refines the SAFE / SUSPICIOUS / HIGH-RISK wording (it never downgrades a binary scam flag to SAFE).
- Policy layer: SAFE / SUSPICIOUS / HIGH-RISK SCAM.
- Features: word TF-IDF, character `char_wb` TF-IDF, and hand-engineered Kenyan structural indicators.

## Data
- Original cleaned labelled data: 643 unique messages after removing conflicting-label texts and collapsing duplicates.
- Binary model training: original clean messages plus 80 synthetic legitimate development messages.
- Synthetic legitimate holdout: 40 messages, never used in fitting.
- Separate synthetic demonstration corpus: 120 examples, never used in model fitting or held-out evaluation.
- Source labels are annotations, not verified fraud outcomes. Three-class risk labels derive from `prelabel`.

## Performance snapshot
See `reports/advanced_evaluation.json`. Current run (on the same 128-message test split the baseline already used once, so exploratory): binary ROC-AUC approximately 0.940; scam-class F1 approximately 0.901; three-class risk macro-F1 approximately 0.872 on the 128-message held-out set. Synthetic legitimate holdout false-alarm rate is 0/40; this is not a real-world estimate.

## Scores and thresholds
The 0–100 score is the raw binary model score multiplied by 100. It is not calibrated probability. Thresholds are policy settings selected for this prototype and require calibration/validation on representative data.

## Known limitations
Small dataset; label noise; limited independent Kenyan legitimate SMS; no sender identity; no live URL reputation, domain age, or telecom/bank verification; no prospective time-based evaluation; severity tiers are not scam-family categories; synthetic examples may not reflect natural message distributions.

## Ethical and safety considerations
Do not treat a SAFE output as a guarantee. Never share PINs, passwords, or OTPs. Verify receipts in official account records. Contact institutions through independently obtained official channels.
