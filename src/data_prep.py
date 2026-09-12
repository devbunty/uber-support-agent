"""
Loads a twcs.csv-shaped file (real Kaggle file or the synthetic stand-in),
filters to a single brand, and reconstructs
    (customer_message -> brand_reply [-> customer_follow_up])
units using response_tweet_id / in_response_to_tweet_id.

Design choices (see decision_log.md):
- We only keep customer tweets that got an actual brand reply. Unanswered
  tweets can't teach us "how the brand historically resolved" anything, and
  including them would corrupt the retrieval corpus.
- response_tweet_id can be a comma-separated list in the real dataset
  (a tweet can have multiple replies); we take the first one authored by
  the brand.
- @mentions and URLs are stripped for the *classifier/retrieval* view but
  the ORIGINAL raw text is preserved in `raw_text` for the reply drafter
  and for audit/eval, since stripping can remove meaningful signal (e.g.
  "@Uber_Support @UberEats" both mentioned).
"""
import re
import pandas as pd

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")

EXPECTED_COLS = {"tweet_id", "author_id", "inbound", "created_at", "text",
                  "response_tweet_id", "in_response_to_tweet_id"}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing expected columns: {missing}. "
                          f"Are you sure this is a twcs.csv-shaped file?")
    df["inbound"] = df["inbound"].isin(["True", "true", "1"])
    return df


def clean_for_model(text: str) -> str:
    t = MENTION_RE.sub("", text)
    t = URL_RE.sub("", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def _first_id(cell: str):
    ids = [x for x in str(cell).split(",") if x and x.lower() != "nan"]
    return ids[0] if ids else None


def build_message_units(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """
    Returns one row per (customer message that got a brand reply), with:
      customer_tweet_id, customer_raw_text, customer_clean_text,
      brand_reply_id, brand_reply_text,
      followup_raw_text (or None), thread_id
    """
    by_id = df.set_index("tweet_id", drop=False)

    inbound = df[df["inbound"] & df["text"].str.contains(
        rf"@{re.escape(brand_handle)}\b", case=False, regex=True, na=False
    )].copy()
    inbound = inbound[inbound["text"].apply(lambda t: len(clean_for_model(t)) > 0)]

    records = []
    for _, row in inbound.iterrows():
        reply_row = None
        rid = _first_id(row["response_tweet_id"])
        # a customer tweet can list multiple response ids; walk them to find
        # the one actually authored by the brand
        candidates = [x for x in str(row["response_tweet_id"]).split(",") if x and x.lower() != "nan"]
        for cid in candidates:
            if cid in by_id.index and by_id.loc[cid, "author_id"] == brand_handle:
                reply_row = by_id.loc[cid]
                break
        if reply_row is None:
            continue  # no brand reply -> not usable for grounding/eval

        # optional customer follow-up to the brand reply
        followup_text = None
        fid = _first_id(reply_row["response_tweet_id"])
        if fid and fid in by_id.index and by_id.loc[fid, "inbound"]:
            followup_text = by_id.loc[fid, "text"]

        records.append({
            "thread_id": row["tweet_id"],
            "customer_tweet_id": row["tweet_id"],
            "customer_raw_text": row["text"],
            "customer_clean_text": clean_for_model(row["text"]),
            "brand_reply_id": reply_row["tweet_id"],
            "brand_reply_text": reply_row["text"],
            "followup_raw_text": followup_text,
            "created_at": row["created_at"],
        })

    return pd.DataFrame.from_records(records)


def run(raw_path: str, brand_handle: str, out_path: str) -> pd.DataFrame:
    df = load_raw(raw_path)
    units = build_message_units(df, brand_handle)
    units.to_csv(out_path, index=False)
    return units


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/synthetic/twcs_uber_synthetic.csv")
    ap.add_argument("--brand", default="Uber_Support")
    ap.add_argument("--out", default="data/processed/message_units.csv")
    args = ap.parse_args()
    units = run(args.raw, args.brand, args.out)
    print(f"{len(units)} usable (customer -> brand reply) units written to {args.out}")
    print(units[["customer_clean_text", "brand_reply_text"]].head(5).to_string())
