# Uber_Support AI Agent

An AI support agent for **@Uber_Support** (Twitter customer support) that:
1. Classifies incoming messages into a 9-intent taxonomy (`src/intents.py`)
2. Drafts a reply grounded in how @Uber_Support has historically resolved similar issues (retrieval + drafting)
3. Decides auto-handle vs. escalate-to-human, with a stated reason (`src/escalation.py`)

Full writeup: **`report/REPORT.md`**. Non-obvious decisions and why: **`decision_log.md`**.

## Reproduce the headline results in under 15 minutes

```bash
git clone <this-repo> && cd uber-support-agent
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# 1. Generate synthetic data (schema-identical to the real Kaggle twcs.csv)
python3 scripts/make_synthetic_data.py

# 2. Ingest: filter to Uber_Support, reconstruct (customer -> brand reply) pairs
python3 src/data_prep.py

# 3. Build the golden evaluation set (188 examples: stratified sample + hand-authored edge cases)
python3 -m scripts.build_golden_set

# 4. Run the full evaluation harness: baselines + pipeline, all metrics
python3 -m eval.eval_harness

# 5. Judge-vs-human agreement check (mandatory "how good is your judge" evidence)
python3 -m eval.judge_agreement
```

All five steps run in well under a minute on the synthetic data (no API key needed —
defaults to `LLM_BACKEND=mock`, a deterministic keyword-based stand-in so the *pipeline
plumbing* is fully testable with zero setup). **Read `report/REPORT.md`, section
"What's misleading about my headline number" before trusting any number this produces
on the mock backend** — it's the most important section in this repo.

## Using the real Kaggle dataset

1. Download `twcs.csv` from [Kaggle: Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (requires a free Kaggle account) and place it at `data/raw/twcs.csv`.
2. Re-run ingestion against the real file:
   ```bash
   python3 src/data_prep.py --raw data/raw/twcs.csv --brand Uber_Support --out data/processed/message_units.csv
   ```
3. **You must now hand-label a golden set yourself.** `scripts/build_golden_set.py`'s
   stratified-sampling logic is reusable, but its ground-truth intents come from
   `data/synthetic/true_labels.csv`, which only exists because we generated the synthetic
   data and know the answer. On real tweets there is no such file — sample ~180-200 real
   examples (the script's `stratified_sample()` function works on any `message_units.csv`;
   you'll need a quick keyword pre-pass per intent to stratify since you don't have true
   labels yet), export to a spreadsheet, and label `true_intent` / `ideal_escalate` by
   reading each one yourself, blind to any model prediction. Keep the 15 hand-authored
   adversarial examples in `scripts/build_golden_set.py::HAND_AUTHORED` — they're
   deliberately brand-agnostic-ish and still useful as stress tests.
4. Everything downstream (`eval_harness.py`, `judge_agreement.py`) is unchanged.

## Dashboard & live demo

A small FastAPI app renders the eval results as a case-file-styled ledger,
and lets you file a "case" (type a message) and watch the agent classify,
retrieve, draft, and decide in real time.

```bash
# after running the 5 reproduction steps above, refresh the dashboard data:
python3 -m eval.generate_dashboard_data

# then launch the app:
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**:
- `/` — the ledger: intent/escalation metrics vs. baselines, judge-agreement
  results, and the lowest-scoring cases, all pulled live from `eval/dashboard_data.json`
- `/demo` — file a new case and see the full pipeline decision, including the
  stated escalation reason and which historical precedent the draft is grounded in

Re-run `python3 -m eval.generate_dashboard_data` any time after re-running
the eval harness (e.g. after switching `LLM_BACKEND`) to refresh the ledger.

## Using a real LLM instead of the mock

The mock backend (`src/llm_client.py::MockLLM`) is a deterministic keyword matcher —
useful for testing pipeline logic with zero setup, **not a claim about real model
quality**. To get real classification/drafting/judging:

```bash
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=<your key>
python3 -m eval.eval_harness
```

`AnthropicLLM` in `src/llm_client.py` is ~15 lines — swap it for OpenAI, a local model,
or whatever you have access to; every other file talks to `LLMClient.generate()` and
doesn't know or care which backend is behind it.

## Repo layout

```
src/
  intents.py         intent taxonomy (9 intents incl. OTHER)
  llm_client.py       pluggable LLM backend (mock | anthropic | bring-your-own)
  data_prep.py        ingest twcs.csv -> (customer, brand_reply) pairs
  classifier.py       intent classification + keyword fallback
  retrieval.py        TF-IDF nearest-precedent retrieval
  reply_generator.py  drafts a reply grounded in retrieved precedent
  escalation.py       auto-handle vs escalate policy, with stated reason
  pipeline.py         wires the above into one run_pipeline() call
scripts/
  make_synthetic_data.py   synthetic twcs.csv-shaped dataset for dev/test
  build_golden_set.py      golden eval set: stratified sample + hand-authored cases
eval/
  golden_set.csv                the 188-example golden set
  baselines.py                  trivial + simple (keyword) baselines
  judge.py                      LLM-as-judge rubric
  human_quality_labels.csv      40 hand-labeled examples (by the developer)
  judge_agreement.py            judge-vs-human agreement metrics
  eval_harness.py               runs everything, prints the metrics table
report/
  REPORT.md            problem framing, results, failure analysis, next steps
app/
  main.py              FastAPI app: ledger dashboard + live demo
  templates/           Jinja2 templates (dashboard.html, demo.html)
  static/style.css     case-file visual theme
decision_log.md        10-15 non-obvious decisions and why
```
