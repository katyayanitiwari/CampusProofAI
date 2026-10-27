# Scholarship dataset review

Reviewed 27 October 2026 for CampusProof's scholarship-offer verification task.

## Finding

No publicly downloadable, labeled dataset of genuine and fraudulent scholarship advertisements or offers was identified in this review. A recent study, “Ensuring Scholarship Integrity: Detecting and Classifying Fraud in Digital Applications Using Machine Learning,” reports a 1,000-row dataset for detecting fraudulent digital scholarship applications. That is a different task: detecting suspicious applicant records does not establish whether a scholarship offer or its publisher is genuine. The paper page describes the dataset but no downloadable dataset or repository was located during this review.

A separate Mendeley Data corpus is downloadable under CC BY 4.0 and contains 523 broad English/Bangla financial scam and legitimate messages. After removing three duplicate-message rows, 520 remained. Five messages matched scholarship/grant/tuition keywords, too few and too generic to test scholarship-offer classification. A separate experimental message-triage model was trained on the full corpus with a fixed stratified 65/15/20 split; the held-out test split had 104 messages.

The U.S. Department of Education Office of Inspector General publishes scholarship-scam warnings and red flags, including upfront fees and unsolicited awards. This is useful guidance for human review, but it is not a labeled corpus suitable for defensible supervised training.

## Model decision

The original TF-IDF + Linear SVM model was trained on DIFrauD English job postings. Its held-out test scam precision is 96.8% and recall is 50.0% (60 scam examples in the test split). These metrics apply only to that job-posting benchmark. A second, experimental logistic-regression model now provides a general financial-message text signal; on its 104-message test set, scam precision was 82.2% and recall was 96.8% (13 legitimate messages were falsely flagged). Its results are uncertain because the dataset is small and class/domain coverage is limited. It is explicitly not a scholarship-authenticity result or probability. CampusProof compares claims with source text supplied by the user and states that this text is not independently authenticated.

No scholarship-specific classifier was trained from synthetic, guessed, or unrelated labels. That would give misleading performance claims. Instead, CampusProof now runs transparent scholarship safety checks for fee requests, unsolicited award claims, urgency, credential/identity requests, social-media “easy award” claims, and guaranteed-award wording. These checks are prompts for independent source verification, not fraud verdicts. See `artifacts/reports/scholarship-safety-guidelines.md`.

## Sources

- [Scholarship-integrity study (2026)](https://doi.org/10.1016/j.procs.2026.04.152)
- [U.S. Department of Education OIG: Scholarship Scams](https://oig.ed.gov/resources/free-materials/student-loan-scams/scholarship-scams)
- [DIFrauD dataset card](https://huggingface.co/datasets/difraud/difraud)
- [Mendeley Financial scams detection dataset v2](https://data.mendeley.com/datasets/znsk27yk3h/2)

