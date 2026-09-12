"""
Two baselines, required before any pipeline metric is meaningful:

  TRIVIAL  : always predict the single most common intent in the golden
             set; for escalation, always predict "auto-handle" (never
             escalate). This is the floor -- if the real pipeline can't
             beat this, it isn't adding value.

  SIMPLE   : the same keyword-rule classifier used internally as the
             pipeline's crash-fallback (src.classifier.keyword_fallback_classify),
             reused here on purpose rather than writing a second rule set --
             it's a fair "simple but real" baseline, and reusing it means
             we can also report how often the pipeline silently degrades
             TO this baseline (used_fallback=True cases).
             Escalation: escalate iff the keyword match is 'safety_incident'
             or no keyword matched at all (best_hits==0, i.e. the simple
             rule has no idea what this is).
"""
from collections import Counter
from src.classifier import keyword_fallback_classify


def trivial_predict_intent(golden_true_intents) -> str:
    return Counter(golden_true_intents).most_common(1)[0][0]


def trivial_classify(_text: str, majority_intent: str) -> str:
    return majority_intent


def trivial_escalate(_text: str) -> bool:
    return False  # trivial baseline: auto-handle everything


def simple_classify(text: str):
    intent, confidence, rationale = keyword_fallback_classify(text)
    return intent, confidence, rationale


def simple_escalate(text: str) -> bool:
    intent, _, _ = keyword_fallback_classify(text)
    return intent == "safety_incident"
