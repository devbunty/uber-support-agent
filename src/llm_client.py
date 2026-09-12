"""
Pluggable LLM client. Every downstream component (classifier, retrieval
re-ranking, reply drafting, judge) talks to `LLMClient.generate()` and never
to a specific vendor SDK -- so swapping models is a one-line change.

Backends:
  - "mock"      : deterministic, keyword-grounded, no API key required.
                  This is NOT a real LLM -- it exists so the pipeline is
                  fully runnable and testable with zero setup. Accuracy
                  numbers produced with this backend are a floor, not a
                  claim about LLM quality (see report.md "what's misleading").
  - "anthropic" : calls the real Claude API. Requires ANTHROPIC_API_KEY
                  in your environment. This is the "bring your own model"
                  slot -- swap this class for an OpenAI/local-model client
                  and nothing else in the codebase needs to change as long
                  as you implement the same .generate() signature.

Select via env var LLM_BACKEND=mock|anthropic (defaults to mock).
"""
import os
import re
import json
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return raw text completion for the given system+user prompt."""
        raise NotImplementedError


class MockLLM(LLMClient):
    """
    Deterministic stand-in used for development/testing without an API key.
    It does simple keyword/rule matching against the taxonomy's seed phrases
    and template-based drafting -- good enough to prove the *pipeline logic*
    (routing, escalation, evaluation harness) works, not a substitute for
    real model quality. Every place that uses this backend is logged in
    eval output so headline numbers are never silently mock-generated.
    """
    def __init__(self):
        from src.intents import TAXONOMY
        self.taxonomy = TAXONOMY

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        # Two things this mock needs to handle: classification prompts and
        # drafting prompts. We detect which by a marker the caller includes.
        if "CLASSIFY_TASK" in system:
            return self._classify(user)
        if "DRAFT_TASK" in system:
            return self._draft(user)
        if "JUDGE_TASK" in system:
            return self._judge(user)
        return json.dumps({"note": "mock backend has no handler for this task"})

    def _classify(self, user: str) -> str:
        text = user.lower()
        best_intent, best_hits = "other", 0
        for intent, spec in self.taxonomy.items():
            hits = sum(1 for phrase in spec.example_seed_phrases if phrase in text)
            if hits > best_hits:
                best_intent, best_hits = intent.value, hits
        confidence = min(0.55 + 0.15 * best_hits, 0.95) if best_hits else 0.35
        return json.dumps({
            "intent": best_intent,
            "confidence": round(confidence, 2),
            "rationale": f"mock keyword match, {best_hits} seed phrase(s) hit" if best_hits
                         else "no seed phrases matched, defaulted to OTHER",
        })

    def _draft(self, user: str) -> str:
        m = re.search(r"PRECEDENT_REPLY:\s*(.+)", user, re.DOTALL)
        precedent = m.group(1).strip().split("\n")[0] if m else \
            "Thanks for reaching out -- please DM us more details so we can help."
        return json.dumps({"draft_reply": precedent})

    def _judge(self, user: str) -> str:
        return json.dumps({"score": 3, "reasoning": "mock judge: neutral default score"})


class AnthropicLLM(LLMClient):
    """Real backend. Requires `pip install anthropic` and ANTHROPIC_API_KEY set."""
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


def get_llm_client(backend: str = None) -> LLMClient:
    backend = backend or os.environ.get("LLM_BACKEND", "mock")
    if backend == "mock":
        return MockLLM()
    if backend == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"Unknown LLM_BACKEND '{backend}'. Use 'mock' or 'anthropic', "
                      f"or add your own LLMClient subclass here.")
