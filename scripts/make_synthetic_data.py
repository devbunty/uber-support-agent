"""
Generates a synthetic stand-in for the Kaggle "Customer Support on Twitter"
dataset (thoughtvector/customer-support-on-twitter), scoped to @Uber_Support,
so the whole pipeline can be built and tested before you download the real
~3M-row twcs.csv.

Schema matches the real file exactly:
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

Swap this file's output for the real twcs.csv in data/raw/ and nothing else
in the pipeline changes -- see README "Step 2: use the real dataset".

This is NOT the golden eval set. It exists purely so classify/retrieve/draft/
escalate have something realistic to run against during development.
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(7)

BRAND = "Uber_Support"

# (intent, customer template list, historical brand-reply template list, severity)
# Reply templates encode how @Uber_Support has *actually historically resolved*
# this kind of issue on Twitter (ask to DM, ask for trip ID, point to in-app flow,
# apologize + escalate to safety line, etc.) -- these are what the retrieval /
# grounding step will later be built to surface.
SCENARIOS = {
    "safety_incident": {
        "severity": "high",
        "customer": [
            "my driver just got into an accident on the highway, we're both okay but this was terrifying {mention}",
            "the driver kept swerving and going way too fast, i genuinely felt unsafe the whole ride {mention}",
            "I was harassed by my driver during my ride tonight, this needs to be looked at immediately {mention}",
        ],
        "reply": [
            "We're so sorry to hear this and want to make sure you're safe. Please reach out to our safety line so a specialist can help right away: {safety_link}",
            "This is incredibly concerning and we want to help immediately. Can you please contact our dedicated safety team here: {safety_link}",
        ],
    },
    "driver_behavior": {
        "severity": "medium",
        "customer": [
            "my driver was so rude to me the entire trip, wouldn't even say hi {mention}",
            "driver took a completely different route than the app showed and it cost me extra {mention}",
            "the driver was smoking in the car with the windows up, it was so uncomfortable {mention}",
        ],
        "reply": [
            "We're sorry to hear about your experience. Please share your trip ID via DM so we can look into this with the driver.",
            "That's not the experience we want for you. Could you DM us the trip details so we can follow up?",
        ],
    },
    "fare_billing_dispute": {
        "severity": "medium",
        "customer": [
            "why was I charged double for the same trip?? {mention}",
            "the fare estimate said $12 but I got charged $27, what happened {mention}",
            "my promo code didn't apply and I got charged full price {mention}",
        ],
        "reply": [
            "Sorry about that! Please DM us your trip ID and email on file so we can review the charge.",
            "We'd like to take a closer look. Can you send us the trip ID via DM so we can check the fare breakdown?",
        ],
    },
    "trip_cancellation": {
        "severity": "low",
        "customer": [
            "driver cancelled on me after waiting 15 minutes and now I got charged a fee {mention}",
            "my driver never showed up and I still got charged for cancelling {mention}",
        ],
        "reply": [
            "Sorry for the trouble! Please DM your trip ID and we'll take a look at the cancellation fee.",
            "That shouldn't happen. Send us the trip details in a DM and we'll review the charge.",
        ],
    },
    "lost_item": {
        "severity": "low",
        "customer": [
            "I think I left my phone in the back seat of my uber, how do I get it back {mention}",
            "left my jacket in the car from my ride tonight, help! {mention}",
        ],
        "reply": [
            "No worries, we can help with that! Please use the 'Find Lost Item' option in your app under trip history to contact your driver directly.",
            "You can report this and contact your driver directly through the app: go to your trip in the app and tap 'I lost an item'.",
        ],
    },
    "account_access": {
        "severity": "medium",
        "customer": [
            "the app keeps crashing every time I try to request a ride {mention}",
            "my account got deactivated and I have no idea why {mention}",
            "I can't add my new card, it keeps saying payment failed {mention}",
        ],
        "reply": [
            "Sorry for the inconvenience! Please DM us the email on your account so we can look into this further.",
            "We'd like to help sort this out -- can you send us your account email via DM?",
        ],
    },
    "refund_request": {
        "severity": "low",
        "customer": [
            "I want a refund for my last trip, it was a mess {mention}",
            "can I get my money back for this ride, it was way overpriced {mention}",
        ],
        "reply": [
            "We understand. Please DM your trip ID and we'll review it for a possible refund.",
            "Sorry to hear that -- send us the trip ID in a DM so we can take a look.",
        ],
    },
    "feedback_praise": {
        "severity": "low",
        "customer": [
            "shoutout to my driver tonight, best ride I've had in a while! {mention}",
            "just wanted to say the new app update is so much smoother, nice work {mention}",
        ],
        "reply": [
            "This made our day! Thank you for sharing -- we'll be sure to pass it along.",
            "Love to hear it, thank you for the kind words!",
        ],
    },
    "other": {
        "severity": "medium",
        "customer": [
            "does uber operate in Cape Town yet? {mention}",
            "when is uber eats getting more restaurant options in my area {mention}",
        ],
        "reply": [
            "Thanks for reaching out! Availability varies by city -- check the app to see current service in your area.",
            "Great question -- restaurant availability updates regularly, keep an eye on the app!",
        ],
    },
}

FOLLOW_UPS = ["sent!", "just dm'd you", "ok thank you", "still waiting to hear back", "done, sent the details"]

WEIGHTS = {
    "safety_incident": 4, "driver_behavior": 14, "fare_billing_dispute": 18,
    "trip_cancellation": 12, "lost_item": 10, "account_access": 14,
    "refund_request": 10, "feedback_praise": 10, "other": 8,
}


def noisy(text: str) -> str:
    if random.random() < 0.15:
        text = text.replace(" you ", " u ").replace(" are ", " r ")
    if random.random() < 0.2:
        text = text.rstrip(".") + random.choice(["", "!", "...", " 😡", " 😞", " 🙏"])
    return text


def main(n_threads: int = 260, out_dir: str = "data/synthetic"):
    rows = []
    true_labels = []  # (customer_tweet_id, true_intent) -- known because WE generated it
    tweet_id = 100000
    start = datetime(2023, 1, 1)
    intents = list(WEIGHTS.keys())
    weights = list(WEIGHTS.values())

    for _ in range(n_threads):
        intent = random.choices(intents, weights=weights, k=1)[0]
        scenario = SCENARIOS[intent]
        cust_id = str(random.randint(10**7, 10**8))
        ts = start + timedelta(minutes=random.randint(0, 500000))

        # inbound customer tweet
        cust_text = noisy(random.choice(scenario["customer"]).format(mention=f"@{BRAND}"))
        cust_tid = str(tweet_id); tweet_id += 1
        rows.append([cust_tid, cust_id, "True", ts.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                     cust_text, "", ""])
        true_labels.append((cust_tid, intent))

        # brand reply
        reply_text = random.choice(scenario["reply"]).format(safety_link="https://help.uber.com/safety")
        reply_tid = str(tweet_id); tweet_id += 1
        ts2 = ts + timedelta(minutes=random.randint(2, 90))
        rows.append([reply_tid, BRAND, "False", ts2.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                     reply_text, "", cust_tid])
        # backfill response_tweet_id on the customer row
        rows[-2][5] = reply_tid

        # ~45% of threads have a short follow-up turn
        if random.random() < 0.45:
            fu_text = random.choice(FOLLOW_UPS)
            fu_tid = str(tweet_id); tweet_id += 1
            ts3 = ts2 + timedelta(minutes=random.randint(1, 60))
            rows.append([fu_tid, cust_id, "True", ts3.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                         fu_text, "", reply_tid])
            rows[-2][5] = fu_tid  # brand reply's response_tweet_id points to follow-up

    random.shuffle(rows)  # real twcs.csv is not neatly ordered either

    path = f"{out_dir}/twcs_uber_synthetic.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "author_id", "inbound", "created_at", "text",
                    "response_tweet_id", "in_response_to_tweet_id"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic tweets across {n_threads} threads -> {path}")

    labels_path = f"{out_dir}/true_labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_tweet_id", "true_intent"])
        w.writerows(true_labels)
    print(f"Wrote ground-truth intent labels (KNOWN because this data is synthetic) -> {labels_path}\n"
          f"NOTE: this file is a synthetic-data-only convenience. Once you load the real twcs.csv, "
          f"there is no such file -- you must actually hand-label the golden set yourself. "
          f"See scripts/build_golden_set.py docstring.")


if __name__ == "__main__":
    main()
