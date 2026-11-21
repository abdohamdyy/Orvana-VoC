"""Cheap baseline: character n-gram TF-IDF + logistic regression on the cleaned train segments.

It is the floor any fancier model has to beat (Stage 3 comparison). Trains in seconds on CPU.
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from ..config import settings
from ..schema import Category

MODEL_PATH = settings.model_dir.parent / "tfidf.joblib"


class TfidfBackend:
    name = "tfidf-logreg"

    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> "TfidfBackend":
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
        )
        pipe.fit(texts, labels)
        return cls(pipe)

    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, MODEL_PATH)

    @classmethod
    def load(cls) -> "TfidfBackend":
        return cls(joblib.load(MODEL_PATH))

    def classify(self, segments: list[str]) -> list[tuple[Category, float]]:
        proba = self.pipe.predict_proba(segments)
        classes = self.pipe.classes_
        return [(Category(classes[i]), float(p[i])) for p, i in zip(proba, np.argmax(proba, axis=1))]
