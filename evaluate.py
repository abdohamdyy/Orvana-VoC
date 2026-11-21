"""Score your predictions against the dev gold set.

Usage:  python evaluate.py predictions.csv [data/tickets_dev_gold.csv]

predictions.csv must have the columns:
ticket_id, category, secondary_category, product_mentioned, store_mentioned,
sentiment, urgency
Empty cells mean "none". product_mentioned = brand (as in products_catalog.csv),
store_mentioned = store_id (as in store_names.csv).
"""
import sys, pandas as pd
from sklearn.metrics import f1_score, accuracy_score

pred_path = sys.argv[1]
gold_path = sys.argv[2] if len(sys.argv) > 2 else "data/tickets_dev_gold.csv"
g = pd.read_csv(gold_path).set_index("ticket_id")
p = pd.read_csv(pred_path).set_index("ticket_id")
missing = g.index.difference(p.index)
print(f"tickets in gold: {len(g)} | predicted: {len(g)-len(missing)} | missing: {len(missing)}")
p = p.reindex(g.index)
norm = lambda s: s.fillna("none").astype(str).str.strip().str.lower()
res = {
    "category_macro_f1": f1_score(norm(g.category), norm(p.category), average="macro"),
    "category_accuracy": accuracy_score(norm(g.category), norm(p.category)),
    "secondary_category_accuracy": accuracy_score(norm(g.secondary_category), norm(p.secondary_category)),
    "product_exact_match": accuracy_score(norm(g.product_mentioned), norm(p.product_mentioned)),
    "store_exact_match": accuracy_score(norm(g.store_mentioned), norm(p.store_mentioned)),
    "sentiment_accuracy": accuracy_score(norm(g.sentiment), norm(p.sentiment)),
    "urgency_accuracy": accuracy_score(g.urgency.astype(str), p.urgency.fillna(-1).astype(int).astype(str)),
}
w = {"category_macro_f1": .40, "secondary_category_accuracy": .10, "product_exact_match": .125, "store_exact_match": .125,
     "sentiment_accuracy": .05, "urgency_accuracy": .20}
for k, v in res.items(): print(f"{k:30s} {v:.3f}")
print(f"{'OVERALL (weighted)':30s} {sum(res[k]*w[k] for k in w):.3f}")
