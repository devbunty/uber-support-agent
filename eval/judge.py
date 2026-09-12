"""
LLM-as-judge for drafted reply quality.

Rubric (1-5, single holistic score with three named criteria the judge must
consider -- kept to one score rather than three sub-scores because with
only ~188 golden examples, splitting into sub-metrics would make the
judge-vs-human agreement analysis too noisy to trust; see decision_log.md):

  1 = Wrong or unsafe: factually wrong action, ignores the actual issue,
      or gives an action that could make things worse.
  2 = Off-target: on-topic but the wrong resolution path for this specific
      issue (e.g. billing-dispute answer to a lost-item question).
  3 = Generic but harmless: reasonable-sounding but generic, doesn't use
      the specific detail in the customer's message.
  4 = Good: correct resolution path, appropriately brief and on-brand.
  5 = Excellent: correct resolution path AND acknowledges the specific
      detail/emotion in the customer's message.

IMPORTANT LIMITATION (also in report.md): judge-vs-human agreement here is
measured against labels I (the developer) assigned myself reading each
reply, NOT an independent second annotator. This is a known weakness --
self-agreement is an upper bound on true inter-rater reliability, not a
substitute for it. Documented, not hidden -- see report.md "what's
misleading".
"""
import json
import re
from dataclasses import dataclass
from src.llm_client import LLMClient

SYSTEM_PROMPT = """JUDGE_TASK
Score the AGENT_REPLY to the CUSTOMER_MESSAGE on a 1-5 scale:
1 = wrong/unsafe action
2 = off-target resolution path
3 = generic but harmless
4 = good: correct resolution path, brief, on-brand
5 = excellent: correct resolution path AND acknowledges the specific detail/emotion in the message

Respond with ONLY JSON: {"score": <1-5 integer>, "reasoning": "<one sentence>"}
"""


@dataclass
class JudgeResult:
    score: int
    reasoning: str


def judge_reply(customer_message: str, agent_reply: str, llm: LLMClient) -> JudgeResult:
    user = f"CUSTOMER_MESSAGE: {customer_message}\nAGENT_REPLY: {agent_reply}"
    raw = llm.generate(system=SYSTEM_PROMPT, user=user, temperature=0.0)
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        score = int(payload["score"])
        if not (1 <= score <= 5):
            raise ValueError("score out of range")
        return JudgeResult(score, str(payload.get("reasoning", "")))
    except Exception as e:
        return JudgeResult(3, f"[judge parse failure, defaulted to neutral 3: {e}]")
