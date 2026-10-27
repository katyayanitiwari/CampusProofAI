# Student-facing financial scam message model

- Dataset: Mendeley Data v2, 520 rows after duplicate/conflict cleanup; CC BY 4.0.
- Scholarship-related keyword rows: 5; insufficient for scholarship-specific validation.
- Model: `LogisticRegression_wordonly` with word-only TF-IDF; `wordonly` vocabulary profile (844 fitted features), threshold selected on validation data.
- Model footprint: 844 fitted features and 12,475 bytes; standard reference has 9,850 features and 105,023 bytes.
- One local single-message benchmark (50 runs): selected median 2.354 ms vs standard median 4.323 ms; timing depends on hardware and system load.
- Held-out scam precision: 0.822; recall: 0.968.
- Intended use: experimental general financial scam-message triage in English/Bangla.
- Not intended for scholarship/fellowship authenticity decisions; score is not a probability or verdict.
- Attribution: Ahmed, A. R. & Islam, G. F. (2025), Financial scams detection dataset, v2, doi:10.17632/znsk27yk3h.2.
- Full metrics and limitations: `artifacts/metrics/financial_message_scam_metrics.json`.

