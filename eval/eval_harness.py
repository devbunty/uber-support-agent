"""
Runs the trivial baseline, simple baseline, and the real pipeline over the
golden set, and reports:
  - intent classification accuracy + macro-F1 (all three)
  - escalation precision/recall/F1 against ideal_escalate (all three)
  - LLM-judge quality score distribution for the pipeline's drafted replies
  - fallback-classifier rate (how often the "real" classifier degraded)

Judge-vs-human agreement is a SEPARATE step (see eval/judge_agreement.py)
because it requires a human-labeled subset that doesn't exist until you've
looked at pipeline output at least once -- see that file's docstring for why.

Usage: python -m eval.eval_harness
"""
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.llm_client import get_llm_client
from src.retrieval import TfidfRetriever
from src.pipeline import run_pipeline
from eval.baselines import trivial_predict_intent, trivial_classify, trivial_escalate, \
    simple_classify, simple_escalate
from eval.judge import judge_reply


def run(golden_path="eval/golden_set.csv", out_path="eval/predictions.csv"):
    golden = pd.read_csv(golden_path)
    golden["ideal_escalate"] = golden["ideal_escalate"].astype(bool)

    llm = get_llm_client()
    retriever = TfidfRetriever()
    majority_intent = trivial_predict_intent(golden["true_intent"])

    rows = []
    for _, g in golden.iterrows():
        msg = g["customer_message"]

        pred = run_pipeline(msg, llm, retriever)
        simple_intent, _, _ = simple_classify(msg)
        j = judge_reply(msg, pred.draft_reply, llm)

        rows.append({
            "id": g["id"], "customer_message": msg, "true_intent": g["true_intent"],
            "ideal_escalate": g["ideal_escalate"], "source": g["source"],

            "trivial_intent": trivial_classify(msg, majority_intent),
            "trivial_escalate": trivial_escalate(msg),

            "simple_intent": simple_intent,
            "simple_escalate": simple_escalate(msg),

            "pipeline_intent": pred.intent,
            "pipeline_confidence": pred.intent_confidence,
            "pipeline_used_fallback": pred.used_fallback_classifier,
            "pipeline_escalate": pred.escalate,
            "pipeline_escalate_reason": pred.escalation_reason,
            "pipeline_draft_reply": pred.draft_reply,
            "pipeline_top_similarity": pred.top_similarity,
            "judge_score": j.score,
            "judge_reasoning": j.reasoning,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} predictions -> {out_path}\n")
    print_metrics(df)
    return df


def print_metrics(df: pd.DataFrame):
    print("=" * 70)
    print("INTENT CLASSIFICATION")
    print("=" * 70)
    for name, col in [("Trivial (majority class)", "trivial_intent"),
                       ("Simple (keyword rule)", "simple_intent"),
                       ("Pipeline", "pipeline_intent")]:
        acc = accuracy_score(df["true_intent"], df[col])
        f1 = f1_score(df["true_intent"], df[col], average="macro", zero_division=0)
        print(f"  {name:28s} accuracy={acc:.3f}  macro-F1={f1:.3f}")

    print(f"\n  Pipeline classifier fell back to keyword rules on "
          f"{df['pipeline_used_fallback'].mean()*100:.1f}% of examples.")

    print("\n" + "=" * 70)
    print("ESCALATION DECISION (positive class = escalate)")
    print("=" * 70)
    for name, col in [("Trivial (never escalate)", "trivial_escalate"),
                       ("Simple (safety keyword only)", "simple_escalate"),
                       ("Pipeline", "pipeline_escalate")]:
        p = precision_score(df["ideal_escalate"], df[col], zero_division=0)
        r = recall_score(df["ideal_escalate"], df[col], zero_division=0)
        f1 = f1_score(df["ideal_escalate"], df[col], zero_division=0)
        print(f"  {name:28s} precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")

    print("\n" + "=" * 70)
    print("DRAFT REPLY QUALITY (LLM judge, 1-5)")
    print("=" * 70)
    print(f"  mean={df['judge_score'].mean():.2f}  "
          f"median={df['judge_score'].median():.1f}  "
          f"% scoring <=2 (bad)={(df['judge_score']<=2).mean()*100:.1f}%")

    print("\n" + "=" * 70)
    print("BREAKDOWN: sampled vs hand-authored (edge cases) examples")
    print("=" * 70)
    for source, g in df.groupby("source"):
        acc = accuracy_score(g["true_intent"], g["pipeline_intent"])
        print(f"  {source:14s} n={len(g):3d}  pipeline intent accuracy={acc:.3f}  "
              f"mean judge score={g['judge_score'].mean():.2f}")


if __name__ == "__main__":
    run()
