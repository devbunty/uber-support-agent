"""
Drafts a reply to a new customer message, grounded in the most similar
historically-resolved precedent(s) from retrieval.py.

Design choice: the LLM is explicitly instructed to ADAPT the precedent
reply, not invent a new resolution path from scratch. This is the core
"grounded in how the brand has historically resolved similar issues"
requirement -- it also makes the draft auditable (a human reviewer can
see exactly which past reply the draft is based on) and gives the
escalation policy a cheap signal: low retrieval similarity = the draft
is not well-grounded = lean toward escalating (see escalation.py).
"""
import json
import re
from dataclasses import dataclass
from typing import List
from src.retrieval import RetrievedPrecedent
from src.llm_client import LLMClient

SYSTEM_PROMPT = """DRAFT_TASK
You are drafting a customer support reply for @Uber_Support on Twitter.
You are given the new customer message and 1-3 precedent examples of how
@Uber_Support has actually replied to similar past messages.

Rules:
- Base your reply's ACTION (what you ask the customer to do, e.g. DM trip ID,
  use an in-app flow, contact the safety line) on the precedent replies.
  Do not invent a new resolution path that isn't reflected in the precedents.
- Match @Uber_Support's tone: brief, empathetic, action-oriented, no more
  than 2 sentences, Twitter-appropriate.
- If the precedents disagree with each other, prefer the one from the
  highest-similarity precedent (listed first).
- Respond with ONLY JSON: {"draft_reply": "<reply text>"}
"""


@dataclass
class DraftResult:
    draft_reply: str
    grounded_on: List[RetrievedPrecedent]
    top_similarity: float


def _format_precedents(precedents: List[RetrievedPrecedent]) -> str:
    lines = []
    for i, p in enumerate(precedents, 1):
        lines.append(f"PRECEDENT_{i} (similarity={p.similarity:.2f}):\n"
                      f"  past customer message: {p.customer_text}\n"
                      f"  PRECEDENT_REPLY: {p.brand_reply}")
    return "\n".join(lines)


def draft_reply(customer_message: str, precedents: List[RetrievedPrecedent], llm: LLMClient) -> DraftResult:
    user = f"New customer message: {customer_message}\n\n{_format_precedents(precedents)}"
    raw = llm.generate(system=SYSTEM_PROMPT, user=user, temperature=0.2)
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        text = str(payload["draft_reply"]).strip()
    except Exception:
        # Never fail silently into a blank reply -- fall back to the top
        # precedent verbatim, which is always a *safe* (if generic) reply.
        text = precedents[0].brand_reply if precedents else \
            "Thanks for reaching out -- please DM us more details so we can help."
    top_sim = precedents[0].similarity if precedents else 0.0
    return DraftResult(draft_reply=text, grounded_on=precedents, top_similarity=top_sim)


if __name__ == "__main__":
    from src.llm_client import get_llm_client
    from src.retrieval import TfidfRetriever
    llm = get_llm_client()
    retriever = TfidfRetriever()
    msg = "I left my sunglasses in the back of my uber last night, how do I get them back?"
    precedents = retriever.retrieve(msg, k=2)
    result = draft_reply(msg, precedents, llm)
    print(f"Customer: {msg}")
    print(f"Draft: {result.draft_reply}")
    print(f"Top similarity: {result.top_similarity:.2f}")
