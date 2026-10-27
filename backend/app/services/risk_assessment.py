def risk_signals(text):
    t=text.lower(); signals=[]
    if any(x in t for x in ("pay registration fee","processing fee","upfront fee","send money")): signals.append("Message requests payment before an award or placement; verify through an independently found official channel.")
    if any(x in t for x in ("act now","within 24 hours","limited slots","urgent")): signals.append("Urgency language may pressure applicants; confirm deadlines on the publisher’s site.")
    if any(x in t for x in ("guaranteed selection","100% guaranteed")): signals.append("Guarantee language warrants further checking against official programme terms.")
    return signals
import re
from urllib.parse import urlsplit


SCHOLARSHIP_CONTEXT = re.compile(
    r"\b(scholarship|fellowship|bursary|student grant|education grant|tuition aid|financial aid|award money)\b",
    re.I,
)


def risk_signals(text, url=None):
    """Surface evidence-based warning signs; these are cues, never fraud verdicts."""
    original = text or ""
    t = original.casefold()
    signals = []

    # These general payment/urgency checks also remain useful for internships and jobs.
    if re.search(r"\b(pay|send|transfer|wire)\b.{0,45}\b(money|fee|deposit|charge|payment)\b|\b(processing|registration|redemption|release|courier) fee\b|\bupfront fee\b", t):
        signals.append("Payment requested before an award or placement: verify the fee directly on the programme’s independently located official site before paying.")
    if re.search(r"\b(act now|within 24 hours|limited slots?|urgent|last chance|immediately)\b", t):
        signals.append("Urgency language may pressure applicants; confirm deadlines through an official programme page.")
    if re.search(r"\b(guaranteed selection|100% guaranteed|guaranteed scholarship|guaranteed grant)\b", t):
        signals.append("Guaranteed-award wording warrants checking against published eligibility and selection rules.")

    if SCHOLARSHIP_CONTEXT.search(original):
        if re.search(r"\b(selected|winner|chosen|approved)\b.{0,70}\b(scholarship|fellowship|grant|award)\b|\b(claim|collect|receive)\b.{0,50}\b(your|the)\b.{0,30}\b(scholarship|fellowship|grant|award)\b", t):
            signals.append("The message announces or invites a scholarship award claim; if you did not apply, verify the award through the sponsor’s official contact details.")
        if re.search(r"\b(otp|one.time password|pin number|password|passcode|cvv|security code)\b", t) and re.search(r"\b(send|share|reply|text|provide|enter|whatsapp)\b", t):
            signals.append("The message asks you to send or enter an OTP, password, PIN, or card security code; do not share these in response to an offer.")
        if re.search(r"\b(bank account|account number|debit card|credit card|aadhaar|social security)\b", t) and re.search(r"\b(reply|send|share|whatsapp|text us|message us|click here)\b", t):
            signals.append("The message requests sensitive identity or financial details through a message or link; first verify the sponsor and portal independently.")
        if re.search(r"\b(no essay|no documents|no requirements|five.minute|5.minute|instant approval|everyone qualifies|easy scholarship)\b", t) and re.search(r"\b(instagram|facebook|social media|whatsapp|telegram|tiktok|ad)\b", t):
            signals.append("A social-media offer promises an unusually easy or minimal-requirement award; locate the sponsor’s official programme page before applying.")
        government_claim = re.search(r"\b(government of india|ministry|national scholarship portal|\bNSP\b|central sector scheme|PM[- ]scholarship)\b", original, re.I)
        if government_claim and url:
            host = (urlsplit(url).hostname or "").casefold().rstrip(".")
            official_india_host = host in {"gov.in", "nic.in"} or host.endswith((".gov.in", ".nic.in"))
            if host and not official_india_host:
                signals.append("The text claims an Indian government scholarship but the supplied link is not on a .gov.in or .nic.in domain; cross-check the exact scheme through the official ministry or National Scholarship Portal.")

    # Remove any duplicate warnings when one text matches overlapping phrases.
    return list(dict.fromkeys(signals))

