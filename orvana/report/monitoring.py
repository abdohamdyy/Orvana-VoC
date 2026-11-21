"""Weekly model-health signals (docs/RISKS.md section D). Written into the report and into
outputs/metrics/monitoring_<week>.json so a trend can be followed over time."""
import math
import re

import pandas as pd

from ..config import settings

_STORE_CONTEXT = re.compile(r"فرع|far3|branch", re.IGNORECASE)


def psi(current: dict[str, float], reference: dict[str, float]) -> float:
    """Population Stability Index between two share distributions (in %). > 0.2 = significant shift."""
    total = 0.0
    for k in reference:
        p, q = max(current.get(k, 0.0), 0.1) / 100, max(reference[k], 0.1) / 100
        total += (p - q) * math.log(p / q)
    return round(total, 3)


def signals(df: pd.DataFrame, categories: list[dict]) -> dict:
    valid = df[df.is_valid]
    n = max(len(valid), 1)
    # confidence 0.0 = the cascade sent the ticket to the LLM (models disagreed); < 0.5 = model unsure
    low_conf = valid.confidence < 0.5
    unmatched_store = valid.message_text.str.contains(_STORE_CONTEXT) & valid.store_mentioned.isna()
    out = {
        "low_confidence_share_pct": round(100 * float(low_conf.sum()) / n, 1),
        "sent_to_llm_pct": round(100 * float((valid.confidence == 0.0).sum()) / n, 1),
        "unmatched_store_mentions": int(unmatched_store.sum()),
        "category_psi_vs_history": psi({c["category"]: c["share_pct"] for c in categories},
                                       {c["category"]: c["hist_share_pct"] for c in categories}),
        "injection_share_pct": round(100 * float(df.injection_flag.sum()) / max(len(df), 1), 1),
        "invalid_share_pct": round(100 * float((~df.is_valid).sum()) / max(len(df), 1), 1),
    }
    alerts = []
    if out["low_confidence_share_pct"] > 35:  # dev-suite baseline: ~25% of new-phrasing segments are sent to the LLM
        alerts.append("many tickets the fine-tuned model could not settle alone: new phrasing? consider labelling + retraining")
    if out["category_psi_vs_history"] > 0.2:
        alerts.append("category mix shifted vs history: check whether it is a real incident or model drift")
    if out["unmatched_store_mentions"] > 3:
        alerts.append("store mentions not in store_names.csv: new branch or new spelling? update the catalogue")
    out["alerts"] = alerts
    return out
