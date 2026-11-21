"""Error analysis for one or more prediction files against the dev gold set.

    python scripts/error_analysis.py predictions/stage2_qwen3-4b_v1.csv [more.csv ...] [--examples 8]

Prints, per file: the evaluate.py metrics with a bootstrap 95% CI on the overall score, per-class
F1, the category confusions, and a few wrong examples per field. Also writes outputs/metrics/<name>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent.parent
W = {"category_macro_f1": .40, "secondary_category_accuracy": .10, "product_exact_match": .125,
     "store_exact_match": .125, "sentiment_accuracy": .05, "urgency_accuracy": .20}
FIELDS = ["category", "secondary_category", "product_mentioned", "store_mentioned", "sentiment", "urgency"]


def norm(s):
    return s.fillna("none").astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)


def scores(g: pd.DataFrame, p: pd.DataFrame) -> dict:
    """Same formulas as evaluate.py."""
    r = {"category_macro_f1": f1_score(norm(g.category), norm(p.category), average="macro"),
         "category_accuracy": accuracy_score(norm(g.category), norm(p.category))}
    for f, k in [("secondary_category", "secondary_category_accuracy"), ("product_mentioned", "product_exact_match"),
                 ("store_mentioned", "store_exact_match"), ("sentiment", "sentiment_accuracy"), ("urgency", "urgency_accuracy")]:
        r[k] = accuracy_score(norm(g[f]), norm(p[f]))
    r["overall"] = sum(r[k] * w for k, w in W.items())
    return r


def bootstrap_ci(g, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(g), len(g))
        vals.append(scores(g.iloc[idx].reset_index(drop=True), p.iloc[idx].reset_index(drop=True))["overall"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analyse(path: Path, gold: pd.DataFrame, n_examples: int) -> dict:
    p = pd.read_csv(path, dtype=str).set_index("ticket_id").reindex(gold.index).reset_index()
    g = gold.reset_index()
    s = scores(g, p)
    lo, hi = bootstrap_ci(g, p)
    print(f"\n=== {path.name}\n" + "  ".join(f"{k}={v:.3f}" for k, v in s.items()) + f"\n  overall 95% CI [{lo:.3f}, {hi:.3f}]")
    rep = classification_report(norm(g.category), norm(p.category), output_dict=True, zero_division=0)
    print("  per-class F1: " + ", ".join(f"{c}={rep[c]['f1-score']:.2f}" for c in sorted(rep) if c not in ("accuracy", "macro avg", "weighted avg")))
    conf = pd.crosstab(norm(g.category), norm(p.category))
    off = [(a, b, int(conf.loc[a, b])) for a in conf.index for b in conf.columns if a != b and conf.loc[a, b] > 0]
    print("  category confusions (gold -> pred): " + ", ".join(f"{a}->{b}:{n}" for a, b, n in sorted(off, key=lambda x: -x[2])))
    errors = {}
    for f in FIELDS:
        bad = norm(g[f]) != norm(p[f])
        errors[f] = int(bad.sum())
        if bad.any() and n_examples:
            print(f"  -- {f}: {bad.sum()} wrong")
            for _, r in g[bad].head(n_examples).iterrows():
                print(f"     gold={r[f]!s:16} pred={p.loc[_, f]!s:16} | {str(r.message_text)[:90]}")
    out = {"file": path.name, **{k: round(v, 4) for k, v in s.items()}, "overall_ci95": [round(lo, 4), round(hi, 4)],
           "per_class_f1": {c: round(rep[c]["f1-score"], 3) for c in rep if c not in ("accuracy", "macro avg", "weighted avg")},
           "errors_per_field": errors}
    (ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "metrics" / f"{path.stem}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--gold", default=str(ROOT / "data" / "tickets_dev_gold.csv"),
                    help="reference file (default: dev gold; e.g. tests/fixtures/challenge_set.csv)")
    args = ap.parse_args()
    gold = pd.read_csv(args.gold, dtype=str).set_index("ticket_id")
    for f in args.files:
        analyse(Path(f), gold, args.examples)


if __name__ == "__main__":
    sys.exit(main())
