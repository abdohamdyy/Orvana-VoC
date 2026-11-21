"""Production backend: the cheap fine-tuned model answers; a segment goes to the LLM only when there
is a reason to doubt it. Three routing signals (chosen on suite-dev, see docs/STAGE3_REPORT.md):
  - confidence below a threshold (after temperature scaling)
  - disagreement with a second cheap model (TF-IDF): two different models agreeing is strong evidence
  - low similarity to the training data: new phrasing, where confidence is known to be over-optimistic
"""
import numpy as np

from ..schema import Category


class CascadeBackend:
    def __init__(self, primary, fallback, threshold: float = 0.8, second_opinion=None, retriever=None,
                 min_similarity: float | None = None):
        self.primary, self.fallback, self.threshold = primary, fallback, threshold
        self.second_opinion, self.retriever, self.min_similarity = second_opinion, retriever, min_similarity
        rules = [f"conf<{threshold}"] + (["disagree"] if second_opinion else []) + \
                ([f"sim<{min_similarity}"] if min_similarity is not None else [])
        self.name = f"cascade({primary.name} -> {fallback.name} if {' or '.join(rules)})"
        self.routed = 0  # segments sent to the fallback (monitoring / cost)
        self.seen = 0

    def route_mask(self, segments: list[str], preds) -> np.ndarray:
        mask = np.array([conf < self.threshold for _, conf in preds])
        if self.second_opinion is not None:
            other = self.second_opinion.classify(segments)
            mask |= np.array([a[0] != b[0] for a, b in zip(preds, other)])
        if self.min_similarity is not None and self.retriever is not None:
            mask |= self.retriever.max_similarity(segments) < self.min_similarity
        return mask

    def classify(self, segments: list[str]) -> list[tuple[Category, float]]:
        preds = self.primary.classify(segments)
        self.seen += len(segments)
        unsure = [i for i, r in enumerate(self.route_mask(segments, preds)) if r] if segments else []
        if unsure:
            try:
                second = self.fallback.classify([segments[i] for i in unsure])
                for i, (cat, _) in zip(unsure, second):
                    # confidence 0.0 marks "the cheap model was overruled, the LLM decided" (a softmax over
                    # 9 classes never goes below 1/9, so 0.0 is unambiguous); monitoring counts these
                    preds[i] = (cat, 0.0)
                self.routed += len(unsure)
            except Exception:  # LLM down: keep the primary answer (still a valid label) rather than fail
                pass
        return preds
