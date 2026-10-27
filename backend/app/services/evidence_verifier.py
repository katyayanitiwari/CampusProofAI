from datetime import datetime, timezone
import os
import re
import uuid

from app.schemas.models import Claim, Source, VerifyResponse
from app.services.risk_assessment import risk_signals
from app.services.report_service import save_report


def verify(text, url=None, source_text=""):
    """Compare claims only with source text explicitly pasted by the user."""
    now = datetime.now(timezone.utc).isoformat()
    supplied_source = (source_text or "").strip()
    sources = []
    limits = [
        "No external search or page fetch is performed. Source text is user-provided and has not been independently authenticated.",
        "Safety checks are prompts, not proof of fraud or legitimacy.",
    ]
    if supplied_source:
        sources.append(
            Source(
                url=str(url) if url else "",
                title="Source text provided by user",
                retrieved_at=now,
                status="user-provided",
                text=supplied_source[:18000],
            )
        )
    else:
        limits.append(
            "No source text was provided. Open the source independently and paste its relevant text to compare claims."
        )

    candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if len(sentence.strip()) > 15
    ][:8]
    claims = []
    for sentence in candidates:
        normalized = re.sub(r"\W+", " ", sentence.casefold()).strip()
        match = None
        for source in sources:
            for passage in re.split(r"(?<=[.!?])\s+|\n+", source.text):
                normalized_passage = re.sub(r"\W+", " ", passage.casefold()).strip()
                if normalized_passage == normalized and len(normalized.split()) >= 4:
                    match = (source, passage.strip())
                    break
            if match:
                break
        claims.append(
            Claim(
                claim=sentence[:500],
                status="supports" if match else "insufficient",
                source_url=(match[0].url or None) if match else None,
                evidence_quote=match[1] if match else None,
                uncertainty=(
                    "Exact text match in user-provided source; verify the page owner and current context independently."
                    if match
                    else "No exact text match was found in the source text provided. This does not establish that the claim is false."
                ),
            )
        )
    if not claims:
        claims = [
            Claim(claim="Opportunity details", status="insufficient", uncertainty="No factual claim could be extracted.")
        ]

    status = (
        "supports"
        if sources and all(claim.status == "supports" for claim in claims)
        else "insufficient"
    )
    result = VerifyResponse(
        report_id=str(uuid.uuid4()),
        evidence_status=status,
        summary=(
            "Some or all wording matches the source text you supplied; confirm the source and current terms independently."
            if status == "supports"
            else "There is not enough user-provided source evidence to support the claims."
        ),
        claims=claims,
        sources=sources,
        risk_signals=risk_signals(text, url),
        limitations=limits,
        retrieved_at=now,
        live=False,
    )
    if os.getenv("CAMPUSPROOF_PUBLIC_MODE") != "1":
        save_report(result.model_dump())
    return result

