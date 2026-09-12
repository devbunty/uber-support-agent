"""
Builds eval/golden_set.csv.

METHODOLOGY (goes in report.md verbatim):
  1. Stratified random sample of ~180 examples from the synthetic corpus,
     proportional to each intent's natural frequency but with a floor of
     5 examples for rare intents (safety_incident) so rare-but-critical
     cases aren't evaluated on n=1. Ground truth intent for these comes
     from data/synthetic/true_labels.csv, which is legitimate ground truth
     HERE ONLY because we generated the synthetic data ourselves and know
     which template produced each message.
  2. ~20 HAND-AUTHORED adversarial examples (written directly in this file,
     not sampled) targeting known taxonomy weak points: sarcasm, mixed/
     dual intent, no explicit ask, very short messages, off-topic. These
     were labelled by reading each one and deciding true_intent and
     ideal_escalate myself -- see the HAND_AUTHORED list below for the
     reasoning behind each label.
  3. `ideal_escalate` for the sampled examples is derived mechanically from
     the escalation policy's stated rules (safety->escalate,
     confidence/similarity thresholds don't apply to ground truth by
     definition, so we use: severity==high -> escalate, else -> auto)
     EXCEPT where a human judgment call overrides it (see
     `ideal_escalate_reason`; overrides are manual and documented per row).

*** CRITICAL CAVEAT (also in report.md "what's misleading"): ***
This golden set's intent labels are cheap to get right because they come
from the generator that wrote the messages. On the REAL twcs.csv, you do
not get this shortcut -- you must read each sampled real tweet and label
it yourself, blind to any model prediction. Re-run this script's sampling
logic (stratify by a quick keyword pre-pass so you don't have to eyeball
all 3M rows) against real data, then hand-label the sample in a
spreadsheet before trusting any number in this report.
"""
import random
import pandas as pd
from src.intents import ALL_INTENTS

random.seed(42)

HAND_AUTHORED = [
    # (customer_message, true_intent, ideal_escalate, reason)
    ("great, ANOTHER driver who cancels on me right as I'm walking out the door. love this app",
     "trip_cancellation", False, "Sarcastic tone but the underlying issue is a routine cancellation -- "
     "escalating every sarcastic tweet would swamp humans with low-severity noise."),
    ("my driver was texting the whole ride AND then tried to charge me a cleaning fee for nothing",
     "driver_behavior", True, "Mixed intent (unsafe distracted driving + billing dispute); the safety-adjacent "
     "half means this should go to a human even though 'driver_behavior' alone is medium severity."),
    ("...",
     "other", True, "No content at all -- nothing to classify or ground a reply in, must go to a human."),
    ("still nothing???",
     "other", True, "Almost certainly a follow-up to an earlier unresolved thread, but with no context "
     "attached a bot cannot know what 'nothing' refers to."),
    ("uber pls",
     "other", True, "Too little information to classify confidently; auto-handling here risks a generic, "
     "unhelpful reply to what could be anything from a joke to a real complaint."),
    ("the driver dropped me at the wrong location and then it was pitch black and unfamiliar, I had to walk "
     "20 min alone at night",
     "safety_incident", True, "Not an assault/accident, but a real safety consequence of an operational "
     "mistake -- hard policy on safety_incident should catch this, testing whether the classifier does too."),
    ("can you guys just fix your app already, 3rd time crashing today",
     "account_access", False, "Frustrated tone but a clear, common, low-severity technical complaint with a "
     "well-established resolution path (DM for troubleshooting)."),
    ("I was double charged AND my driver was incredibly rude about it when I asked him to explain",
     "fare_billing_dispute", True, "Mixed billing + behavior; billing precedent alone wouldn't address the "
     "behavior complaint, so grounding would be incomplete -- should escalate rather than under-serve half the issue."),
    ("thank you for refunding my trip so quickly earlier, appreciate it!",
     "feedback_praise", False, "Clearly closed-loop positive feedback, no action needed, safe to auto-ack."),
    ("is this the right account to ask about a job driving for uber?",
     "other", False, "Off-topic for rider support but low-stakes; a generic 'wrong department, try X' reply "
     "is safe to auto-send."),
    ("I think my ex is using my old uber account to see where I am, is that possible??",
     "safety_incident", True, "Potential stalking/privacy-safety issue disguised as an account question -- "
     "a good test of whether the classifier over-indexes on the word 'account' and misses the real signal."),
    ("charged me twice for one trip, this is ridiculous, fix it now",
     "fare_billing_dispute", False, "Angry tone but a completely standard, low-severity billing dispute with "
     "a clear precedent -- tone alone shouldn't force escalation."),
    ("my card was charged for a ride I never took, I wasn't even in that city",
     "fare_billing_dispute", True, "Possible fraud/unauthorized account use, not a routine fare dispute -- "
     "should escalate even though surface keywords look like an ordinary billing complaint."),
    ("left my medication in the car, I need it back today, it's not optional",
     "lost_item", True, "Surface-level identical to a routine lost-item case, but the stated urgency (medical "
     "need) means the standard 'check the app' reply is inadequate -- tests whether severity is content-aware "
     "or purely intent-label-based."),
    ("does the driver see my full name or just my first name?",
     "other", False, "Simple factual/privacy question, no dispute, safe to auto-answer or route generically."),
]

TARGET_SAMPLED = 190
MIN_PER_INTENT = 5


def stratified_sample(units_path="data/processed/message_units.csv",
                       labels_path="data/synthetic/true_labels.csv") -> pd.DataFrame:
    units = pd.read_csv(units_path)
    labels = pd.read_csv(labels_path)
    merged = units.merge(labels, left_on="customer_tweet_id", right_on="customer_tweet_id", how="inner")

    parts = []
    n_intents = merged["true_intent"].nunique()
    per_intent_target = max(MIN_PER_INTENT, TARGET_SAMPLED // n_intents)
    for intent, group in merged.groupby("true_intent"):
        n = min(len(group), per_intent_target)
        parts.append(group.sample(n=n, random_state=42))
    sampled = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return sampled


def derive_ideal_escalate(intent: str) -> bool:
    from src.intents import TAXONOMY, Intent
    return TAXONOMY[Intent(intent)].default_severity == "high"


def build(out_path="eval/golden_set.csv"):
    sampled = stratified_sample()
    rows = []
    for _, r in sampled.iterrows():
        rows.append({
            "id": f"S{r['customer_tweet_id']}",
            "customer_message": r["customer_clean_text"],
            "true_intent": r["true_intent"],
            "ideal_escalate": derive_ideal_escalate(r["true_intent"]),
            "ideal_escalate_reason": "mechanical: severity==high" if derive_ideal_escalate(r["true_intent"])
                                      else "mechanical: severity!=high",
            "source": "sampled",
        })

    for i, (msg, intent, esc, reason) in enumerate(HAND_AUTHORED):
        rows.append({
            "id": f"H{i+1:03d}",
            "customer_message": msg,
            "true_intent": intent,
            "ideal_escalate": esc,
            "ideal_escalate_reason": reason,
            "source": "hand_authored",
        })

    df = pd.DataFrame(rows)
    assert set(df["true_intent"]).issubset(set(ALL_INTENTS)), "unknown intent label in golden set"
    df.to_csv(out_path, index=False)
    print(f"Golden set: {len(df)} examples ({(df['source']=='sampled').sum()} sampled, "
          f"{(df['source']=='hand_authored').sum()} hand-authored) -> {out_path}")
    print(df["true_intent"].value_counts())
    return df


if __name__ == "__main__":
    build()
