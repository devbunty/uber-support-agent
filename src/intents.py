"""
Intent taxonomy for @Uber_Support.

Methodology (see decision_log.md for the full reasoning):
This taxonomy was drafted from domain knowledge of what @Uber_Support
handles (rider fare disputes, driver complaints, lost items, account
issues, safety reports, general feedback) and is DESIGNED TO BE REVISED
once real data is loaded: run `scripts/refine_taxonomy.py` (open-coding
pass over a random sample of real inbound tweets) after you drop twcs.csv
into data/raw/, and update this file if new clusters emerge or two
intents turn out to never be distinguishable in practice.

Kept deliberately small (8 + OTHER). Fine-grained micro-intents look
impressive but are harder to label consistently and harder to trust in
an escalation decision — a handful of well-separated buckets is easier
to audit.
"""
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    SAFETY_INCIDENT = "safety_incident"
    DRIVER_BEHAVIOR = "driver_behavior"
    FARE_BILLING_DISPUTE = "fare_billing_dispute"
    TRIP_CANCELLATION = "trip_cancellation"
    LOST_ITEM = "lost_item"
    ACCOUNT_ACCESS = "account_access"
    REFUND_REQUEST = "refund_request"
    FEEDBACK_PRAISE = "feedback_praise"
    OTHER = "other"


@dataclass(frozen=True)
class IntentSpec:
    intent: Intent
    description: str
    default_severity: str  # high | medium | low -- prior used by the escalation policy
    example_seed_phrases: tuple


TAXONOMY = {
    Intent.SAFETY_INCIDENT: IntentSpec(
        Intent.SAFETY_INCIDENT,
        "Physical safety, assault, harassment, accident, or the rider explicitly saying they feel unsafe.",
        "high",
        ("accident", "assaulted", "harassed", "scared for my safety", "hit my car", "crashed"),
    ),
    Intent.DRIVER_BEHAVIOR: IntentSpec(
        Intent.DRIVER_BEHAVIOR,
        "Complaint about driver conduct that is NOT an emergency (rude, bad route, smoking in car).",
        "medium",
        ("rude", "yelled at me", "bad attitude", "went the wrong way", "smoking in the car"),
    ),
    Intent.FARE_BILLING_DISPUTE: IntentSpec(
        Intent.FARE_BILLING_DISPUTE,
        "Disagreement about what was charged: surge, wrong fare, promo/credit not applied, double charge.",
        "medium",
        ("overcharged", "double charged", "promo didn't work", "surge price", "wrong fare"),
    ),
    Intent.TRIP_CANCELLATION: IntentSpec(
        Intent.TRIP_CANCELLATION,
        "Driver cancelled or never arrived, or a dispute about a cancellation fee.",
        "low",
        ("driver cancelled", "never showed up", "cancellation fee", "waited 20 minutes"),
    ),
    Intent.LOST_ITEM: IntentSpec(
        Intent.LOST_ITEM,
        "Rider left a physical item in the vehicle.",
        "low",
        ("left my phone", "lost my", "forgot my bag", "still in the car"),
    ),
    Intent.ACCOUNT_ACCESS: IntentSpec(
        Intent.ACCOUNT_ACCESS,
        "Can't log in, app crashing, can't add/remove a payment method, account deactivated.",
        "medium",
        ("can't log in", "app keeps crashing", "won't accept my card", "account deactivated", "reset password"),
    ),
    Intent.REFUND_REQUEST: IntentSpec(
        Intent.REFUND_REQUEST,
        "Explicit refund ask not already captured by a billing dispute or cancellation fee.",
        "low",
        ("refund", "money back", "want my money"),
    ),
    Intent.FEEDBACK_PRAISE: IntentSpec(
        Intent.FEEDBACK_PRAISE,
        "Compliment or general feedback with no action needed.",
        "low",
        ("great driver", "thank you", "love the app", "best ride"),
    ),
    Intent.OTHER: IntentSpec(
        Intent.OTHER,
        "Doesn't cleanly fit another bucket, is too ambiguous, or genuinely needs a human to read the thread.",
        "medium",
        (),
    ),
}

ALL_INTENTS = [i.value for i in Intent]
