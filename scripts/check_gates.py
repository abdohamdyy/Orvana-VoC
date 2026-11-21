"""Regression gates for rule/extractor changes (plan revision 4, critique #4).

    python scripts/check_gates.py --save      # record the current behaviour as the baseline
    python scripts/check_gates.py             # compare against the baseline; exit 1 on a regression

Gate 1: the deterministic parts still reproduce the dev gold exactly (oracle category).
Gate 2: behaviour on ALL 9k historical tickets does not drift: how often messages are split into two
        problems, and how often each extractor / flag fires. The history is template-based, so these
        rates are known-good; a jump means a new rule fires where it should not (false positives).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.config import settings  # noqa: E402
from orvana.extractors import (double_charge_amount, find_product, find_store, has_safety_issue,  # noqa: E402
                               is_repeat_complaint, split_segments)
from orvana.normalize import clean_message  # noqa: E402
from orvana.rules import sentiment, urgency  # noqa: E402
from orvana.schema import Category  # noqa: E402

BASELINE = settings.outputs_dir / "metrics" / "gates_baseline.json"
TOLERANCE = 0.002  # absolute change in a rate allowed on 9k tickets (~18 tickets)


def measure() -> dict:
    g = pd.read_csv(settings.data_dir / "tickets_dev_gold.csv")
    none = lambda v: None if pd.isna(v) else v  # noqa: E731
    ok = {"product": 0, "store": 0, "secondary": 0, "urgency": 0, "sentiment": 0}
    for r in g.itertuples():
        c = clean_message(r.message_text)
        segs = split_segments(c.text)
        cat = Category(r.category)
        ok["product"] += find_product(c.text) == none(r.product_mentioned)
        ok["store"] += find_store(c.text) == none(r.store_mentioned)
        ok["secondary"] += (len(segs) == 2) == (not pd.isna(r.secondary_category))
        ok["urgency"] += urgency(cat, segs[0], c.text) == r.urgency
        ok["sentiment"] += sentiment(cat).value == r.sentiment
    gold = {k: v / len(g) for k, v in ok.items()}

    t = pd.read_csv(settings.data_dir / "tickets.csv").message_text
    cleaned = [clean_message(x) for x in t]
    per = pd.DataFrame({
        "split_two": [len(split_segments(c.text)) == 2 for c in cleaned],
        "product": [find_product(c.text) or "" for c in cleaned],
        "store": [find_store(c.text) or "" for c in cleaned],
        "safety_flag": [has_safety_issue(c.text) for c in cleaned],
        "repeat_flag": [is_repeat_complaint(c.text) for c in cleaned],
        "double_charge_flag": [double_charge_amount(c.text) is not None for c in cleaned],
        "injection_flag": [c.injection_flag for c in cleaned],
        "invalid": [not c.is_valid for c in cleaned],
    })
    rates = {k: round(float(per[k].mean() if per[k].dtype == bool else (per[k] != "").mean()), 4) for k in per}
    return {"gold": gold, "history_rates": rates, "_per_ticket": per, "_text": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    now = measure()
    per, text = now.pop("_per_ticket"), now.pop("_text")
    print(json.dumps(now, indent=1))
    per_path = BASELINE.with_suffix(".csv")
    if args.save:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(now, indent=1))
        per.to_csv(per_path, index=False)
        print("baseline saved")
        return
    base = json.loads(BASELINE.read_text())
    old = pd.read_csv(per_path, keep_default_na=False, dtype=str)
    failures = [f"gold {k}: {v:.3f} < 1.0" for k, v in now["gold"].items() if v < 1.0]
    for k, v in now["history_rates"].items():
        if abs(v - base["history_rates"][k]) > TOLERANCE:
            failures.append(f"history {k}: {base['history_rates'][k]} -> {v}")
    # show WHICH historical tickets changed, so each change can be judged (fix or false positive?)
    for col in per.columns:
        changed = per[col].astype(str) != old[col].astype(str)
        if changed.any():
            print(f"\n-- {col}: {int(changed.sum())} historical tickets changed")
            for i in per.index[changed][:8]:
                print(f"   {old.at[i, col]!s:>14} -> {per.at[i, col]!s:<14} | {text[i][:100]}")
    print("\n" + ("\n".join(failures) if failures else "ALL GATES PASS"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
