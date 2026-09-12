"""
Decides whether the drafted reply can be auto-sent or must be escalated to
a human agent, and always states why.

Design choices (see decision_log.md -- this is one of the most important
files in the whole project, since "trust" ultimately lives here):

1. Intent severity is a HARD gate, not a weighted factor. safety_incident
   ALWAYS escalates regardless of classifier confidence or draft quality --
   a confident, well-grounded reply to a safety report is still the wrong
   thing to auto-send. Confidence scores should never be allowed to
   override a hard safety rule.

2. Self-reported LLM classifier confidence is treated as a weak signal,
   not ground truth (LLMs are frequently overconfident). We discount it
   and require retrieval grounding (top_similarity) as a SEPARATE,
   independent signal. A confident classification with no good precedent
   to ground the reply in still escalates -- confidence about *what* the
   issue is doesn't mean we know *how* this brand has resolved it.

3. Thresholds here are initial, documented guesses, NOT tuned on the
   golden set yet. eval/eval_harness.py reports escalation
   precision/recall against golden-set human judgments so these thresholds
   can be justified or revised with evidence -- see report.md.
"""
from dataclasses import dataclass
from src.intents import TAXONOMY, Intent

# Intents that escalate unconditionally, independent of confidence/similarity.
HARD_ESCALATE_INTENTS = {Intent.SAFETY_INCIDENT.value}

CONFIDENCE_THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.35


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def decide(intent: str, classifier_confidence: float, top_similarity: float,
           used_fallback_classifier: bool = False) -> EscalationDecision:
    if intent in HARD_ESCALATE_INTENTS:
        return EscalationDecision(True, f"Hard policy: '{intent}' always escalates regardless of confidence.")

    if used_fallback_classifier:
        return EscalationDecision(True, "Classifier fell back to keyword rules (LLM call failed/malformed) -- "
                                         "don't auto-handle on a degraded signal.")

    severity = TAXONOMY[Intent(intent)].default_severity
    if severity == "high":
        return EscalationDecision(True, f"Intent '{intent}' has high default severity.")

    if classifier_confidence < CONFIDENCE_THRESHOLD:
        return EscalationDecision(True, f"Classifier confidence {classifier_confidence:.2f} "
                                         f"is below threshold {CONFIDENCE_THRESHOLD}.")

    if top_similarity < SIMILARITY_THRESHOLD:
        return EscalationDecision(True, f"No sufficiently similar historical precedent found "
                                         f"(top similarity {top_similarity:.2f} < {SIMILARITY_THRESHOLD}) -- "
                                         f"draft reply is not well-grounded.")

    return EscalationDecision(False, f"Intent '{intent}' ({severity} severity), classifier confidence "
                                      f"{classifier_confidence:.2f}, and grounding similarity "
                                      f"{top_similarity:.2f} all clear the auto-handle bar.")


if __name__ == "__main__":
    print(decide("safety_incident", 0.95, 0.9))
    print(decide("lost_item", 0.8, 0.6))
    print(decide("lost_item", 0.4, 0.6))
    print(decide("fare_billing_dispute", 0.8, 0.2))
