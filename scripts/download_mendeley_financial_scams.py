"""Download Mendeley Data's CC BY 4.0 English/Bangla financial-message corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "mendeley-financial-scams"
DATASET_URL = "https://data.mendeley.com/datasets/znsk27yk3h/2"
METADATA_URL = "https://data.mendeley.com/public-api/datasets/znsk27yk3h"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "CampusProof research project/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read(2_000_000)


def main() -> None:
    metadata = json.loads(fetch(METADATA_URL))
    if metadata.get("data_licence", {}).get("short_name") != "CC BY 4.0":
        raise SystemExit("Dataset license changed; inspect it before downloading or training.")
    file_info = next((item for item in metadata.get("files", []) if item.get("filename", "").lower().endswith(".csv")), None)
    if not file_info:
        raise SystemExit("No CSV file is listed in the dataset metadata.")
    content = fetch(file_info["content_details"]["download_url"])
    if len(content) != int(file_info["size"]):
        raise SystemExit(f"Downloaded size mismatch: expected {file_info['size']}, received {len(content)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "financial-scams-dataset.csv"
    output.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    source_note = (
        "# Dataset source and limits\n\n"
        "- Dataset: Mendeley Data, Financial scams detection dataset, version 2.\n"
        f"- DOI: [10.17632/znsk27yk3h.2]({DATASET_URL})\n"
        "- License: CC BY 4.0. Attribution: Al Rafi Ahmed and Gazi Faizul Islam (2025).\n"
        "- Publisher description: manually labeled English/Bangla scam and ham messages; broad financial scams, including loan/job/OTP/banking scams.\n"
        f"- File SHA-256: `{digest}`\n"
        "- Limit: this is not a scholarship-offer dataset. Only a handful of rows mention grants/tuition, so it cannot validate scholarship fraud detection.\n"
        "- Model use: experimental general scam-message triage only; never a probability or a verdict.\n"
    )
    (DATA_DIR / "DATASET_SOURCE.md").write_text(source_note, encoding="utf-8")
    print(json.dumps({"path": str(output), "bytes": len(content), "sha256": digest, "version": metadata["version"]}, indent=2))


if __name__ == "__main__":
    main()

