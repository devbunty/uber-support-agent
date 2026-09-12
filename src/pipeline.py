"""
End-to-end agent pipeline: one customer message in, one AgentResponse out.
This is the single entry point the eval harness (and, eventually, any real
integration) calls -- keeping it thin and dependency-injected (llm, retriever
passed in) makes it trivial to swap backends or test in isolation.
"""
from dataclasses import dataclass, asdict
from src.classifier import classify
from src.retrieval import TfidfRetriever
from src.reply_generator import draft_reply
from src.escalation import decide
from src.llm_client import LLMClient


@dataclass
class AgentResponse:
    customer_message: str
    intent: str
    intent_confidence: float
    intent_rationale: str
    used_fallback_classifier: bool
    draft_reply: str
    top_similarity: float
    grounded_on_thread_id: str
    escalate: bool
    escalation_reason: str

    def to_dict(self):
        return asdict(self)


def run_pipeline(customer_message: str, llm: LLMClient, retriever: TfidfRetriever,
                  intent_filter_retrieval: bool = False, k: int = 3) -> AgentResponse:
    clf = classify(customer_message, llm)

    intent_mask = None
    if intent_filter_retrieval:
        intent_mask = (retriever.df["predicted_intent"] == clf.intent) \
            if "predicted_intent" in retriever.df.columns else None

    precedents = retriever.retrieve(customer_message, k=k, intent_filter=intent_mask)
    draft = draft_reply(customer_message, precedents, llm)
    esc = decide(clf.intent, clf.confidence, draft.top_similarity, clf.used_fallback)

    return AgentResponse(
        customer_message=customer_message,
        intent=clf.intent,
        intent_confidence=clf.confidence,
        intent_rationale=clf.rationale,
        used_fallback_classifier=clf.used_fallback,
        draft_reply=draft.draft_reply,
        top_similarity=draft.top_similarity,
        grounded_on_thread_id=precedents[0].thread_id if precedents else "",
        escalate=esc.escalate,
        escalation_reason=esc.reason,
    )


if __name__ == "__main__":
    import json
    from src.llm_client import get_llm_client

    llm = get_llm_client()
    retriever = TfidfRetriever()

    messages = [
        "my driver just got into an accident, we're both shaken up",
        "I left my backpack in the back seat, how do I get it back?",
        "why is my fare $40 when the estimate said $22",
        "you guys are the worst, app never works",
    ]
    for m in messages:
        r = run_pipeline(m, llm, retriever)
        print(json.dumps(r.to_dict(), indent=2))
        print("-" * 60)
