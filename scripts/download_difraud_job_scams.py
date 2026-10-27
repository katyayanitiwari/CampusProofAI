"""Download the MIT-licensed DIFrauD job-scams split from a pinned Hub revision."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "difraud-job-scams"
REVISION = "26e18fecb5396f2eb9adb63d42cf916923043478"
BASE_URL = f"https://huggingface.co/datasets/difraud/difraud/resolve/{REVISION}/job_scams"
SPLITS = ("train", "validation", "test")
MAX_SPLIT_BYTES = 32 * 1024 * 1024


def validate_jsonl(payload: bytes, split: str) -> int:
    count = 0
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {split}.jsonl line {line_number}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"Unexpected record in {split}.jsonl line {line_number}")
        if item.get("label") not in (0, 1):
            raise ValueError(f"Invalid label in {split}.jsonl line {line_number}")
        count += 1
    if count == 0:
        raise ValueError(f"The {split} split is empty")
    return count


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        url = f"{BASE_URL}/{split}.jsonl"
        request = urllib.request.Request(url, headers={"User-Agent": "CampusProofAI/1.0 (dataset training)"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(MAX_SPLIT_BYTES + 1)
        except urllib.error.URLError as exc:
            raise SystemExit(f"Could not download {split} split from Hugging Face: {exc}") from exc
        if len(payload) > MAX_SPLIT_BYTES:
            raise SystemExit(f"{split} split exceeded the {MAX_SPLIT_BYTES} byte safety limit")
        rows = validate_jsonl(payload, split)
        destination = DATA_DIR / f"{split}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        summary[split] = {"rows": rows, "bytes": len(payload), "url": url}

    source_note = DATA_DIR / "DATASET_SOURCE.md"
    source_note.write_text(
        "# Dataset source and scope\n\n"
        "- Dataset: DIFrauD, `job_scams` configuration\n"
        "- Publisher: `difraud/difraud` on Hugging Face\n"
        f"- Pinned revision: `{REVISION}`\n"
        "- License declared by the dataset publisher: MIT\n"
        "- Underlying task: English employment/job-ad scam classification\n"
        "- Limitation: this is not scholarship, fellowship, or internship-award data; "
        "the resulting model is only a job-posting warning signal.\n"
        "- Dataset card: https://huggingface.co/datasets/difraud/difraud\n"
        "- Dataset paper: https://aclanthology.org/2024.lrec-main.468/\n"
        "- Original EMSCAD paper: https://doi.org/10.3390/fi9010006\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

