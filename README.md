# CampusProof AI

Student opportunity discovery, eligibility matching, and evidence receipts. Web search opens in the user's browser. CampusProof does not crawl or fetch pages; users can paste source text for a transparent comparison.

## Setup

Python 3.11+ and Node.js 20+ recommended. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

## Train the CampusProof job-posting warning model

The repository includes a downloader for the DIFrauD `job_scams` split, whose dataset card declares an MIT license. Download the pinned split files, then train and evaluate the model:

```powershell
python scripts/download_difraud_job_scams.py
cd backend
python -m app.ml.train_difraud
```

The run compares TF-IDF + Logistic Regression and Linear SVM, selects the model and warning threshold using validation data, then reports results once on the held-out test split. The trained model is written to `models/active.joblib`, metadata to `models/metadata.json`, and test metrics to `artifacts/metrics/job_scam_model_metrics.json`.

**Scope limit:** DIFrauD's job-scams split contains employment listings, not scholarships or fellowships. This model is a warning signal for English job postings only. It is not a scholarship-fraud detector, a probability, or a verdict. Source comparisons use text supplied by the user and do not authenticate the publisher.

Dataset source: [DIFrauD on Hugging Face](https://huggingface.co/datasets/difraud/difraud), pinned to revision `26e18fecb5396f2eb9adb63d42cf916923043478`. See `data/raw/difraud-job-scams/DATASET_SOURCE.md` after download for attribution and limitations.

## General scam-message model

The Verify receipt also includes an experimental general financial-message triage signal. It is trained on Mendeley Data's 523 English/Bangla scam/ham messages under CC BY 4.0; after deduplicating, 520 rows remain. Only five rows match scholarship/grant/tuition keywords, so this is **not** a scholarship-offer detector. Held-out metrics are uncertain because the test split contains only 104 examples. The score is not a probability and should never replace checking the publisher and terms.

From the project root, download the dataset with `python scripts/download_mendeley_financial_scams.py`. From `backend/`, train using `python -m app.ml.train_financial_messages`. The separate artifact is `models/financial_message_scam.joblib`; metrics and limitations are in `artifacts/metrics/financial_message_scam_metrics.json` and `artifacts/reports/financial-message-model-card.md`. Attribution: Ahmed and Islam (2025), [Mendeley Data v2](https://data.mendeley.com/datasets/znsk27yk3h/2), CC BY 4.0.

No publicly downloadable labeled dataset for genuine versus fraudulent scholarship advertisements was found. A scholarship-application fraud study reports a dataset, but application fraud is a different task and its dataset was not downloadable from the sources reviewed; see `artifacts/reports/scholarship-dataset-review.md`.

For scholarship offers, verification receipts also use transparent checks for upfront fees, unsolicited awards, urgency, sensitive credential/identity requests, easy-award social ads, guaranteed-award claims, and Indian government claims linked from outside official domains. These are safety prompts based on published guidance, not a learned classifier or a fraud verdict. See `artifacts/reports/scholarship-safety-guidelines.md`.

### Optional CSV training pipeline

For a different properly licensed job-posting CSV, check the exact copy's license and attribution and pass its path to the existing generic pipeline. It expects a binary `fraudulent` label.

## Prepare, train, evaluate

Run after activation from `backend/`:

```powershell
python -m app.ml.dataset ../data/raw/EMSCAD.csv --out ../data/processed/prepared.csv
python -m app.ml.train ../data/raw/EMSCAD.csv
python -m app.ml.evaluate ../data/raw/EMSCAD.csv
```

The pipeline removes exact duplicates, validates binary labels, creates stratified 70/15/15 train/validation/test splits, fits TF-IDF + class-weighted Logistic Regression and compares Linear SVM, chooses threshold using validation F1, then reports test precision, recall, F1, average precision, ROC-AUC, confusion matrix, and class report. Artifacts go to `models/` and `artifacts/`. Near-duplicate grouping is not implemented. The model is for job posting text only, not validated for Indian scholarships. Its decision score is not a probability.

Single prediction, from `backend/`: `python -c "from app.ml.predict import predict_text; print(predict_text('Job message text'))"`

## Run app

Terminal 1 from `backend/`: `uvicorn app.main:app --reload --port 8000`

Terminal 2 from `frontend/`: `npm install; npm run dev`

Open http://localhost:5173; API docs at http://localhost:8000/docs. Discovery opens a Google search in a new browser tab. Verification compares the submitted message with optional source text pasted by the user; no third-party search key or crawler is used. Optional `VITE_API_URL` is a public frontend backend URL, never a secret.

## Tests and integration

From `backend/`: `python -m pytest -q`. Frontend: `cd frontend; npm install; npm run build`.

## API

`GET /api/health`, `GET /api/model/status`, `POST /api/verify` (`text`, optional `url`), `POST /api/opportunities/search`, `POST /api/eligibility/match-requirements` (`requirements`, `profile`), `GET /api/reports`, `GET /api/reports/{report_id}`.

Evidence quote matching is a deterministic text-overlap heuristic, not semantic fact-checking; nuanced contradiction detection is not yet implemented, so ambiguous results remain insufficient. Eligibility rules surface unknown criteria for manual confirmation. No optional LLM or Agent API is enabled. API errors are surfaced in the UI. SQLite stores local receipts.

## Judge walkthrough

1. Start backend/frontend.
2. Search for an opportunity in the browser and open the publisher's official page independently.
3. Paste the opportunity text and the relevant official-page text into Verify. Inspect exact text matches, uncertainty, and safety indicators.
4. Confirm the publisher and current terms directly on its official site.
5. Explain that the job-post classifier is a warning signal only, not a scholarship fraud model or probability.
6. If live API is unavailable, show the explicit unavailable status; `data/sample/demo_scenarios.json` contains illustrative prompts only, not findings.

## Deployment

`docker compose up --build` starts the backend. Supply `.env` via platform secret manager; persist `artifacts/reports`, and provide both `models/active.joblib` plus `models/metadata.json` and `models/financial_message_scam.joblib` plus its metadata as built artifacts. Deploy frontend separately with `VITE_API_URL` set and configure production CORS. Deployment does not train a model.

