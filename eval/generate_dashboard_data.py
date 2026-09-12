"""
Computes the same metrics as eval/eval_harness.py but writes them to
eval/dashboard_data.json for the frontend to render, instead of printing
to stdout. Run this after eval_harness.py / judge_agreement.py.
"""
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def build(predictions_path="eval/predictions.csv", golden_path="eval/golden_set.csv",
          judge_detail_path="eval/judge_agreement_detail.csv", out_path="eval/dashboard_data.json"):
    df = pd.read_csv(predictions_path)
    df["ideal_escalate"] = df["ideal_escalate"].astype(bool)

    def clf_metrics(col):
        return {
            "accuracy": round(accuracy_score(df["true_intent"], df[col]), 3),
            "macro_f1": round(f1_score(df["true_intent"], df[col], average="macro", zero_division=0), 3),
        }

    def esc_metrics(col):
        return {
            "precision": round(precision_score(df["ideal_escalate"], df[col], zero_division=0), 3),
            "recall": round(recall_score(df["ideal_escalate"], df[col], zero_division=0), 3),
            "f1": round(f1_score(df["ideal_escalate"], df[col], zero_division=0), 3),
        }

    intent_dist = df["true_intent"].value_counts().to_dict()

    breakdown = []
    for source, g in df.groupby("source"):
        breakdown.append({
            "source": source,
            "n": int(len(g)),
            "pipeline_accuracy": round(accuracy_score(g["true_intent"], g["pipeline_intent"]), 3),
            "mean_judge_score": round(g["judge_score"].mean(), 2),
        })

    judge_agreement = {}
    try:
        jd = pd.read_csv(judge_detail_path)
        judge_agreement = {
            "n": int(len(jd)),
            "exact_match_pct": round((jd["judge_score"] == jd["human_score"]).mean() * 100, 1),
            "within_1_pct": round((jd["judge_score"] - jd["human_score"]).abs().le(1).mean() * 100, 1),
            "mae": round((jd["judge_score"] - jd["human_score"]).abs().mean(), 2),
            "judge_is_constant": bool(jd["judge_score"].nunique() == 1),
        }
    except FileNotFoundError:
        pass

    worst_cases = df.sort_values("judge_score").head(8)[
        ["id", "customer_message", "true_intent", "pipeline_intent",
         "pipeline_draft_reply", "judge_score", "pipeline_escalate", "pipeline_escalate_reason"]
    ].to_dict(orient="records")

    data = {
        "n_golden": int(len(df)),
        "intent_distribution": intent_dist,
        "intent_metrics": {
            "trivial": clf_metrics("trivial_intent"),
            "simple": clf_metrics("simple_intent"),
            "pipeline": clf_metrics("pipeline_intent"),
        },
        "escalation_metrics": {
            "trivial": esc_metrics("trivial_escalate"),
            "simple": esc_metrics("simple_escalate"),
            "pipeline": esc_metrics("pipeline_escalate"),
        },
        "judge_quality": {
            "mean": round(df["judge_score"].mean(), 2),
            "pct_bad": round((df["judge_score"] <= 2).mean() * 100, 1),
        },
        "fallback_rate_pct": round(df["pipeline_used_fallback"].mean() * 100, 1),
        "breakdown_by_source": breakdown,
        "judge_agreement": judge_agreement,
        "worst_cases": worst_cases,
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote dashboard data -> {out_path}")
    return data


if __name__ == "__main__":
    build()
