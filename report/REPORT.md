# Report: AI Support Agent for @Uber_Support

## 1. Problem framing

**What "good" means for this brand.** @Uber_Support handles a high volume of
short, often emotionally-charged, public messages where the wrong public
reply is worse than no reply. "Good" here is not "maximize auto-handle
rate" — it's: **never auto-handle a message where the resolution requires
judgment the system doesn't actually have**, and be honest, in a stated
reason, about why each decision was made. A system that auto-handles 90%
of tickets but occasionally sends a generic billing reply to a safety
report is not good, regardless of its aggregate accuracy number. Concretely,
"good" means:

- Intent classification that's not just accurate on average, but
  *reliably escalates on the categories where an auto-reply carries real
  risk* (safety, fraud-adjacent billing, anything ambiguous).
- Replies grounded in a specific historical precedent a human could audit,
  not free-generated text that merely sounds plausible.
- An escalation decision with a stated, falsifiable reason — "confidence
  0.35 < threshold 0.6" is auditable and can be argued with; "the model
  felt uncertain" is not.

**What I chose not to build.**
- **No fine-grained intent taxonomy** (e.g. Banking77-style 77 classes).
  For a Twitter support triage system, 8 well-separated categories that a
  human can reason about beat 30+ categories that blur into each other and
  are expensive to label consistently.
- **No fully autonomous resolution** (e.g. actually issuing a refund).
  The agent's job is triage + drafting, not taking irreversible account
  actions — that boundary is a deliberate scope cut, not a missing feature.
- **No multi-turn dialogue management.** The pipeline scores one customer
  message at a time using thread context already resolved (was there a
  brand reply, was there a follow-up) rather than holding conversational
  state across turns. Real deployment would need this; it was cut to keep
  the evaluation surface (one message in, one decision out) auditable.
- **No embedding-based retrieval, no fine-tuning.** Both are natural next
  steps (see §5) but weren't built first on purpose — see decision log
  #5 and #11: prove the simple, auditable version works (or document
  exactly how it fails) before reaching for a more powerful, less
  inspectable one.

## 2. Results vs. baselines

All numbers below are from the synthetic dataset (188-example golden set:
173 stratified-sampled + 15 hand-authored adversarial cases) with
`LLM_BACKEND=mock`. **Read §4 before treating any of these as a claim about
real-world model quality — several of them are structurally unable to be
that.**

### Intent classification

| Method | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always predict majority class) | 0.138 | 0.027 |
| Simple (keyword rule) | 0.628 | 0.644 |
| Pipeline (mock backend) | 0.628 | 0.644 |

### Escalation decision (positive class = escalate)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Trivial (never escalate) | 0.000 | 0.000 | 0.000 |
| Simple (escalate only on safety keyword) | 1.000 | 0.368 | 0.538 |
| Pipeline | 0.178 | 0.947 | 0.300 |

### Reply quality (LLM judge, 1-5)

Mean 3.00, 0% scoring ≤2 — **this number is fake by construction under the
mock backend**; see §4.

### Judge-vs-human agreement (n=40 hand-labeled)

Exact match 2.5%, within ±1 point 82.5%, quadratic-weighted kappa 0.000.
The kappa of exactly 0 isn't a coincidence — see §4.

## 3. Failure analysis: top 5 failure modes

**1. Paraphrase blindness in retrieval and classification.**
Both TF-IDF retrieval and the keyword-rule classifier only match literal
words. "I got charged twice" retrieves "driver never showed up and I still
got charged for cancelling" (similarity 0.35) instead of the actual
fare-dispute precedent, because "charged twice" shares no n-grams with
"double charged." *Hypothesis:* fixed by embedding-based retrieval and a
real LLM classifier that understands paraphrase — this is the single
highest-leverage fix available (decision log #5).

**2. Escalation thresholds are badly miscalibrated for the mock backend's
confidence distribution.** Pipeline recall is 0.947 but precision is only
0.178 — it escalates almost everything. Example: "left my phone in the
back seat of my uber" is correctly retrieved and drafted, but the mock
classifier's confidence for it lands at 0.35 (below the 0.6 threshold), so
it escalates a routine, low-severity, well-grounded case anyway.
*Hypothesis:* the 0.6/0.35 thresholds (decision log #8) were initial
guesses tuned for nothing; they need a real precision-recall curve fit to
this golden set, and confidence needs to be recalibrated per-backend since
a real LLM's confidence distribution will look nothing like a keyword
counter's.

**3. Vague/short messages can trigger a confidently wrong resolution
path.** `H004`, "still nothing???" (a likely follow-up with no visible
context) drew a full safety-escalation reply pointing the customer to the
safety line — human quality score: 1/5. *Hypothesis:* the drafter adapts
whatever precedent retrieval hands it without checking whether the message
actually contains enough information to justify that action; short,
context-free messages need a distinct "insufficient information, must
escalate" path rather than being routed through the same
retrieve-and-adapt logic as everything else.

**4. Severity is keyed to intent category, not message content, so
per-message stakes are invisible.** `H014`, "left my medication in the
car, I need it back today, it's not optional," is intent-correct
(`lost_item`) but gets the standard low-severity self-service reply —
human quality score: 2/5 — because `lost_item` has a fixed "low" severity
regardless of what was actually lost or how urgent it is. *Hypothesis:*
severity needs a second, orthogonal signal (an urgency/stakes estimate)
rather than being purely a lookup on intent label; a fraud-adjacent
billing dispute and a routine one currently get identical treatment for
the same underlying reason (see `S100xxx` fraud example in the golden
set's hand-authored cases).

**5. Duplicate/templated synthetic phrasing inflates the sampled-split
accuracy.** Many golden-set examples are near-identical strings ("does
uber operate in Cape Town yet?" appears verbatim multiple times) because
the synthetic generator has only 2-3 templates per intent. Pipeline
accuracy is 0.653 on sampled examples vs. 0.333 on hand-authored
adversarial ones — a 2x gap that exists purely because the sampled split
is artificially easy. *Hypothesis:* real tweets will look much more like
the hand-authored split (varied phrasing, sarcasm, typos, mixed intent)
than the templated split, so 0.628 overall accuracy is likely an
overestimate of real-world performance, possibly a large one.

## 4. What is misleading about my headline number

This is the most important section in this report, so it's blunt:

- **Pipeline intent accuracy (0.628) is bit-for-bit identical to the
  simple keyword baseline** — because under `LLM_BACKEND=mock`, the
  "pipeline classifier" *is* a keyword matcher (see `src/llm_client.py::
  MockLLM._classify`). This number proves the pipeline's plumbing is
  correct. It proves nothing about whether an LLM adds value over a
  keyword rule, because no LLM was actually classifying anything.

- **The reply-quality mean of 3.00 is not a measurement, it's a
  constant.** `MockLLM._judge` always returns 3. The judge-agreement
  script caught this directly: kappa is exactly 0.000 because the judge's
  output has literally zero variance across all 40 hand-checked examples.
  Any report that presented "mean quality 3.00" as a finding without this
  caveat would be actively misleading.

- **97% escalation recall sounds safe. It is not, on its own, a useful
  number.** Paired with 18% precision, it describes a system that
  escalates almost everything — which is safe in the narrow sense of
  "rarely auto-sends something wrong," but fails the actual business goal
  of reducing human load. A policy that escalates 90%+ of tickets isn't a
  triage agent, it's a very expensive way to forward tickets.

- **"Hand-labelled golden set" undersells how easy 173/188 of these
  labels were to get right.** Their ground truth comes from the synthetic
  generator's own metadata — I know the answer because I wrote the
  template that produced the message, not because I did the annotation
  work a real golden set requires. The 15 hand-authored adversarial
  examples are genuinely hand-labeled by reading and judgment; the other
  173 are a convenience that will not exist once real tweets are involved
  (see README "Using the real Kaggle dataset").

- **Judge-vs-human agreement was measured against my own labels, not an
  independent annotator.** Even a well-behaved judge could show inflated
  agreement here simply because I wrote both the rubric and the "human"
  labels — the true test (a second person, blind to the judge's output)
  hasn't been run yet.

- **Small per-class n makes per-intent numbers fragile.** `safety_incident`
  has n=12 in the golden set; a single misclassification moves that
  class's F1 by roughly 8 points. Don't read the per-intent breakdown as
  precise.

## 5. What I'd do next with one more week

1. **Swap `LLM_BACKEND=anthropic`** for the classifier, drafter, and judge,
   and re-run the entire harness. This is the single change that would
   make every number in §2 mean something about actual model quality
   rather than pipeline plumbing.
2. **Recalibrate escalation thresholds against a real precision-recall
   curve** on the golden set once real classifier confidence is available,
   and consider per-intent thresholds instead of one global cutoff — the
   current 97%-recall/18%-precision split is a symptom of an untuned,
   backend-mismatched threshold, not a fundamental limit.
3. **Replace TF-IDF retrieval with embedding-based retrieval** to fix the
   paraphrase-blindness failure mode (§3.1), which is currently the
   single highest-leverage, cheapest fix available.
4. **Get a genuine second annotator** — for both golden-set intent labels
   and the judge-agreement human labels — to convert "agreement with
   myself" into an actual inter-rater reliability number.
5. **Download the real `twcs.csv` and re-derive the taxonomy** via an
   actual open-coding pass over a sample of real @Uber_Support tweets,
   rather than the domain-knowledge-informed guess this taxonomy
   currently is, then re-build a real (non-synthetic-shortcut) golden set.
6. **Add an orthogonal urgency/stakes signal**, separate from intent
   category, to fix failure mode #4 (severity currently can't see that
   "left my medication, need it today" is higher-stakes than "left my
   jacket").
7. **Expand the golden set toward 250** with more adversarial diversity —
   multi-lingual messages, heavy typo/slang density, and multi-turn
   context-dependent messages, which the current hand-authored set only
   partially covers.
8. **Track cost and latency per ticket** once a real API is wired in —
   trustworthiness in production depends on more than accuracy; a system
   that's accurate but too slow or expensive to run on 100% of volume
   needs a different rollout plan (e.g. auto-handle only above a
   confidence bar high enough to also be cheap to get right).
