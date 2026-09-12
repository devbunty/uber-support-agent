# Decision Log

Non-obvious decisions made while building this, and why. Ordinary choices
(use Python, use pandas) are omitted.

1. **Kept the intent taxonomy to 8 + OTHER, not 20+.** More intents look
   more sophisticated but are harder to label consistently and harder to
   audit an escalation decision against. A handful of well-separated
   buckets you can actually reason about beats fine-grained categories
   that blur into each other.

2. **`safety_incident` is a hard escalation gate, not a weighted factor.**
   No confidence score or grounding-quality score can override it. This
   was a deliberate choice to stop confidence from ever substituting for a
   safety rule — a model can be very sure about the wrong thing.

3. **Classifier confidence and retrieval similarity are treated as two
   independent signals, not one.** A confident intent label doesn't mean
   we know how the brand resolves it, and a good precedent match doesn't
   mean we've correctly identified the issue. Escalation requires both
   bars to clear, not an average of the two.

4. **The keyword-rule classifier is reused in three places on purpose**:
   as the crash-fallback inside `classifier.py`, as the "simple baseline"
   in `eval/baselines.py`, and (structurally) as the entire brain of the
   mock LLM backend. This wasn't laziness — it means "how often did the
   real classifier degrade to the simple baseline" is a real, reportable
   number (`pipeline_used_fallback` in eval output), not a hidden failure mode.

5. **TF-IDF retrieval before embeddings.** Free, deterministic, fully
   offline, and — critically — auditable: you can see exactly which words
   drove a match. Its known failure mode (paraphrase blindness, e.g.
   "charged twice" vs. "double charged") is treated as a documented
   limitation and the first upgrade path, not hidden.

6. **Retrieval intent-filtering is a toggle, not baked in.** It's tempting
   to assume "only retrieve from the same predicted intent" obviously
   helps. It's a real design choice with a real failure mode (a wrong
   intent classification poisons the whole retrieval), so the harness can
   measure its effect rather than assume it.

7. **The reply drafter is instructed to adapt the precedent's ACTION, not
   invent a new resolution path.** This is what makes "grounded in how the
   brand has historically resolved similar issues" true rather than
   aspirational, and it's what makes a draft auditable — a reviewer can
   see exactly which past reply a draft is based on.

8. **Escalation thresholds (confidence 0.6, similarity 0.35) are stated
   as initial guesses, not tuned values**, and the eval harness reports
   precision/recall against the golden set specifically so they can be
   revised with evidence instead of vibes. Current pipeline numbers show
   they're miscalibrated (0.95 recall, 0.18 precision) — see report.md.

9. **The golden set is 173 stratified-sampled + 15 deliberately
   hand-authored adversarial examples**, not 188 sampled. A golden set
   built entirely from the same templates that generated the training
   corpus would only prove the pipeline can solve problems shaped exactly
   like its own training data. The hand-authored cases (sarcasm, mixed
   intent, no clear ask, urgency hidden inside a routine-looking message)
   exist specifically to break that.

10. **Ground-truth intent labels for the sampled 173 come "for free" from
    the synthetic generator, and this is flagged as a shortcut that
    doesn't exist on real data**, not presented as if it were a real
    labeling exercise. See README "Using the real Kaggle dataset" for
    what actually has to happen once real tweets are involved.

11. **The LLM-judge rubric is a single 1-5 holistic score, not several
    sub-scores** (groundedness, tone, correctness scored separately).
    With only ~188 golden examples, splitting into sub-metrics would make
    judge-vs-human agreement too noisy per sub-metric to draw any
    conclusion from.

12. **Judge-vs-human agreement is measured against the developer's own
    labels, not an independent second annotator**, and this is stated as
    a real limitation rather than dressed up as inter-rater reliability.
    It's still a meaningful floor: if the judge can't agree with the
    person who wrote the rubric, it can't be trusted at scale.

13. **`LLMClient` is an abstract interface every component talks to,
    never a specific SDK.** This was worth the extra indirection because
    it means the entire evaluation harness, escalation policy, and report
    structure are provable and reviewable before a single real API call
    is made — the mock backend exists to de-risk the architecture, not to
    produce trustworthy numbers.

14. **The mock backend's judge always returns a flat score (3), and the
    judge-agreement script actively detects and calls this out** rather
    than silently reporting a kappa/agreement number that would look
    plausible but mean nothing. Discovering "your evaluation number is
    fake" is exactly the kind of thing this project is supposed to catch,
    including when the fakeness is in a tool we built ourselves.

15. **Unanswered customer tweets are dropped during ingestion**, not kept
    with an empty reply. They can't teach the retrieval corpus "how the
    brand resolved this," and including them would silently corrupt every
    downstream precedent match with empty/garbage grounding.
