"""
Intent classification. Talks to the LLM through src.llm_client so the same
code path works for the mock backend (dev/test) and any real model.

Design choices (see decision_log.md):
- We ask for JSON with {intent, confidence, rationale} in one call rather
  than a separate confidence-estimation pass -- cheaper, and self-reported
  LLM confidence is treated as a *prior*, not ground truth (the escalation
  policy in src/escalation.py discounts it -- see that file's docstring).
- If the model returns malformed JSON or an intent label outside our
  taxonomy, we do NOT silently guess. We fall back to a transparent
  keyword-rule classifier and mark result.used_fallback=True, so the eval
  harness can report how often the "real" classifier actually failed.
"""
import json
import re
from dataclasses import dataclass
from src.intents import TAXONOMY, ALL_INTENTS
from src.llm_client import LLMClient

SYSTEM_PROMPT = """CLASSIFY_TASK
You are an intent classifier for @Uber_Support customer messages on Twitter.
Classify the message into EXACTLY ONE of these intents:
{intent_list}

Respond with ONLY a JSON object: {{"intent": "<one of the intents above>", "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}}
No other text.
"""


def _build_system_prompt() -> str:
    lines = [f'- {i.value}: {spec.description}' for i, spec in TAXONOMY.items()]
    return SYSTEM_PROMPT.format(intent_list="\n".join(lines))


def keyword_fallback_classify(text: str):
    """Transparent, inspectable rule-based classifier. Used as (a) a safety
    net when the LLM call fails/returns garbage, and (b) the 'simple baseline'
    in the eval harness -- same function, two purposes, deliberately."""
    text_l = text.lower()
    best_intent, best_hits = "other", 0
    for intent, spec in TAXONOMY.items():
        hits = sum(1 for phrase in spec.example_seed_phrases if phrase in text_l)
        if hits > best_hits:
            best_intent, best_hits = intent.value, hits
    confidence = min(0.5 + 0.15 * best_hits, 0.9) if best_hits else 0.3
    return best_intent, confidence, f"keyword rule matched {best_hits} phrase(s)"


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    rationale: str
    used_fallback: bool


def classify(text: str, llm: LLMClient) -> ClassificationResult:
    system = _build_system_prompt()
    try:
        raw = llm.generate(system=system, user=text, temperature=0.0)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        intent = payload["intent"].strip().lower()
        confidence = float(payload["confidence"])
        rationale = str(payload.get("rationale", ""))
        if intent not in ALL_INTENTS:
            raise ValueError(f"intent '{intent}' not in taxonomy")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence {confidence} out of range")
        return ClassificationResult(intent, confidence, rationale, used_fallback=False)
    except Exception as e:
        intent, confidence, rationale = keyword_fallback_classify(text)
        return ClassificationResult(intent, confidence, f"[fallback: {e}] {rationale}", used_fallback=True)


if __name__ == "__main__":
    from src.llm_client import get_llm_client
    llm = get_llm_client()
    tests = [
        "my driver just crashed into another car, im shaking right now",
        "why am i being charged twice for the same ride",
        "left my sunglasses in the backseat, how do i get them back",
        "just wanted to say my driver tonight was awesome!!",
    ]
    for t in tests:
        r = classify(t, llm)
        print(f"{t!r}\n  -> {r}\n")
