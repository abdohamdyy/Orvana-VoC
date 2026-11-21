"""Scoring helpers shared by scripts/error_analysis.py and scripts/run_suite.py.
`scores` reproduces evaluate.py exactly (same normalisation, same weights)."""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

WEIGHTS = {"category_macro_f1": .40, "secondary_category_accuracy": .10, "product_exact_match": .125,
           "store_exact_match": .125, "sentiment_accuracy": .05, "urgency_accuracy": .20}
FIELDS = {"secondary_category": "secondary_category_accuracy", "product_mentioned": "product_exact_match",
          "store_mentioned": "store_exact_match", "sentiment": "sentiment_accuracy", "urgency": "urgency_accuracy"}


def norm(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "none").astype(str).str.strip().str.lower() \
        .replace({"": "none", "nan": "none"}).str.replace(r"\.0$", "", regex=True)


def scores(g: pd.DataFrame, p: pd.DataFrame) -> dict:
    r = {"category_macro_f1": f1_score(norm(g.category), norm(p.category), average="macro", zero_division=0),
         "category_accuracy": accuracy_score(norm(g.category), norm(p.category))}
    for f, k in FIELDS.items():
        r[k] = accuracy_score(norm(g[f]), norm(p[f]))
    r["overall"] = sum(r[k] * w for k, w in WEIGHTS.items())
    return r


def bootstrap_ci(g: pd.DataFrame, p: pd.DataFrame, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    g, p = g.reset_index(drop=True), p.reset_index(drop=True)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(g), len(g))
        vals.append(scores(g.iloc[idx].reset_index(drop=True), p.iloc[idx].reset_index(drop=True))["overall"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; stable for small slices."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(centre - half, 3), round(centre + half, 3)
