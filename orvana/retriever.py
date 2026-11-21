"""Nearest-neighbour lookup over the cleaned TRAIN segments (never the gold set).

Used for dynamic few-shot prompting (Stage 2 v3): the LLM sees the k most similar labelled
segments. Character n-grams handle Arabic, Franco and English and are robust to typos.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import settings


class Retriever:
    def __init__(self, texts: list[str], labels: list[str]):
        self.texts, self.labels = texts, labels
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(texts)

    def max_similarity(self, texts: list[str]) -> np.ndarray:
        """Cosine similarity of each text to its closest training segment: low = new phrasing (out of
        distribution), where a fine-tuned model's confidence cannot be trusted."""
        return np.asarray((self.vec.transform(texts) @ self.matrix.T).max(axis=1).todense()).ravel()

    def nearest(self, text: str, k: int = 8) -> list[tuple[str, str]]:
        sims = (self.matrix @ self.vec.transform([text]).T).toarray().ravel()
        order = np.argsort(-sims)
        out, seen = [], set()
        for i in order:  # skip duplicates so the k examples are informative
            if self.texts[i] in seen:
                continue
            seen.add(self.texts[i])
            out.append((self.texts[i], self.labels[i]))
            if len(out) == k:
                break
        return out[::-1]  # most similar last = closest to the question


@lru_cache(maxsize=1)
def load_retriever() -> Retriever:
    seg = pd.read_csv(settings.data_dir / "processed" / "segments.csv")
    seg = seg[(seg.split == "train") & seg.label_clean.notna() & seg.text.notna()].drop_duplicates("text")
    return Retriever(seg.text.tolist(), seg.label_clean.tolist())
