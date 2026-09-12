"""
Retrieval: given a new customer message, find the most similar
historically-resolved (customer_message -> brand_reply) pair from
data/processed/message_units.csv to ground the drafted reply in.

Design choices (see decision_log.md):
- TF-IDF + cosine similarity, not embeddings. This is a deliberate "simple
  baseline first" choice: it's free, deterministic, fully offline, and
  auditable (you can see exactly which words drove the match). It's also
  a documented limitation -- see report.md failure analysis -- since it
  misses paraphrases ("charged twice" vs "double charged"). Swapping in
  a real embedding model is a single-function change (see
  `EmbeddingRetriever` stub at the bottom) and is the first thing listed
  in "what I'd do with one more week".
- We optionally filter the candidate pool to the same predicted intent
  before ranking, because two messages can be lexically similar but need
  different resolutions (e.g. "my driver was rude" vs "my driver crashed").
  This filter is a toggle, not baked in, so the eval harness can measure
  its effect directly instead of assuming it helps.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedPrecedent:
    similarity: float
    customer_text: str
    brand_reply: str
    thread_id: str


class TfidfRetriever:
    def __init__(self, corpus_path: str = "data/processed/message_units.csv"):
        self.df = pd.read_csv(corpus_path)
        self.df = self.df.dropna(subset=["customer_clean_text", "brand_reply_text"]).reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.df["customer_clean_text"])

    def retrieve(self, query: str, k: int = 3, intent_filter: Optional[pd.Series] = None):
        pool = self.df
        matrix = self.matrix
        if intent_filter is not None:
            mask = intent_filter.values
            if mask.sum() >= k:  # only filter if enough candidates remain
                pool = self.df[mask].reset_index(drop=True)
                matrix = self.matrix[mask]

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            results.append(RetrievedPrecedent(
                similarity=float(sims[i]),
                customer_text=pool.iloc[i]["customer_clean_text"],
                brand_reply=pool.iloc[i]["brand_reply_text"],
                thread_id=str(pool.iloc[i]["thread_id"]),
            ))
        return results


class EmbeddingRetriever:
    """
    Stub for the upgrade path: swap TfidfRetriever for this once you have an
    embedding API/model wired up. Same .retrieve() signature so nothing else
    in the pipeline needs to change. Left unimplemented on purpose -- see
    report.md 'what I'd do with one more week'.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Plug in your embedding model/API here.")


if __name__ == "__main__":
    r = TfidfRetriever()
    for q in ["I got charged twice for one ride", "left my phone in the uber"]:
        print(f"\nQuery: {q!r}")
        for p in r.retrieve(q, k=2):
            print(f"  sim={p.similarity:.2f} | precedent={p.customer_text!r} -> {p.brand_reply!r}")
