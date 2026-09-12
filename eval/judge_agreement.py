"""
Measures agreement between the LLM judge (eval/judge.py) and a human-labeled
subset (eval/human_quality_labels.csv, n=40).

LIMITATION, stated plainly: the "human" labels are mine (the developer's),
assigned by reading each customer_message/draft_reply pair against the
rubric in judge.py, NOT an independent second annotator. This measures
whether the judge tracks MY reading of the rubric, which is a weaker claim
than true inter-rater reliability -- but it's the honest floor: if the
judge can't even agree with the person who wrote the rubric, it can't be
trusted to grade at scale. A real second annotator is listed in report.md
"what I'd do with one more week".

Usage: python -m eval.judge_agreement
  (run eval_harness.py first so eval/predictions.csv exists)
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def run(predictions_path="eval/predictions.csv", human_path="eval/human_quality_labels.csv"):
    preds = pd.read_csv(predictions_path)[["id", "judge_score", "judge_reasoning", "pipeline_draft_reply", "customer_message"]]
    human = pd.read_csv(human_path)
    merged = human.merge(preds, on="id", how="left")

    exact_match = (merged["judge_score"] == merged["human_score"]).mean()
    within_1 = (merged["judge_score"] - merged["human_score"]).abs().le(1).mean()
    mae = (merged["judge_score"] - merged["human_score"]).abs().mean()

    judge_is_constant = merged["judge_score"].nunique() == 1
    if judge_is_constant:
        kappa = 0.0
        kappa_note = ("undefined in the strict sense (judge output has zero variance -- "
                       "every score is the same value) -- reported as 0.0 by convention, "
                       "but the honest reading is 'the judge carries no signal at all'.")
    else:
        kappa = cohen_kappa_score(merged["judge_score"], merged["human_score"], weights="quadratic")
        kappa_note = ""

    print(f"n = {len(merged)} hand-labeled examples")
    print(f"Exact match rate:     {exact_match*100:.1f}%")
    print(f"Within +/-1 point:    {within_1*100:.1f}%")
    print(f"Mean absolute error:  {mae:.2f}")
    print(f"Quadratic weighted kappa: {kappa:.3f} {kappa_note}")
    print(f"Judge output distinct values: {sorted(merged['judge_score'].unique())}")
    print(f"Human output distinct values: {sorted(merged['human_score'].unique())}")

    if judge_is_constant:
        print("\n*** FINDING: the LLM judge is running on the mock backend and returns a constant "
              "score (3) for every input by construction. This agreement analysis correctly shows "
              "near-zero agreement -- that is the expected, honest result of grading a stub. "
              "This number becomes meaningful only after switching LLM_BACKEND=anthropic (or your "
              "chosen model) for both the agent AND the judge. See report.md 'what's misleading'. ***")

    merged.to_csv("eval/judge_agreement_detail.csv", index=False)
    return merged


if __name__ == "__main__":
    run()
