# CampusProof job-posting warning model

- Model: `LinearSVC` with TF-IDF features and a validation-selected threshold.
- Training data: DIFrauD Job Scams at revision `26e18fecb5396f2eb9adb63d42cf916923043478`; MIT as declared by the publisher.
- Intended use: a warning signal for English employment/job postings only.
- Not intended for scholarship, fellowship, or award verification.
- The model score is not a probability or a verdict.
- See `artifacts/metrics/job_scam_model_metrics.json` for split sizes and held-out metrics.

