"""Phase D: when should the cheap fine-tuned model hand a segment to the LLM?

    python scripts/routing_analysis.py --ft-dir models/stage3

1. Temperature scaling: fit T on the internal dev set (held-out templates) by minimising NLL;
   saved to <ft-dir>/calibration.json and used by FinetunedBackend.
2. On the suite DEV split (never the holdout), for every message's primary segment we get the
   fine-tuned prediction + calibrated confidence, the TF-IDF prediction, the similarity to the
   training data, and the LLM (v3) prediction.
3. Each routing rule gives (share sent to the LLM, category accuracy of the combined system).
   Pre-registered choice: the CHEAPEST rule whose accuracy is within 1 point of the best rule.
Writes outputs/metrics/routing_analysis.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.backends.finetuned import FinetunedBackend  # noqa: E402
from orvana.backends.llm_ollama import OllamaBackend  # noqa: E402
from orvana.backends.tfidf import TfidfBackend  # noqa: E402
from orvana.config import settings  # noqa: E402
from orvana.extractors import split_segments  # noqa: E402
from orvana.normalize import clean_message  # noqa: E402
from orvana.retriever import load_retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fit_temperature(ft: FinetunedBackend) -> float:
    seg = pd.read_csv(settings.data_dir / "processed" / "segments.csv")
    dev = seg[(seg.split == "dev") & seg.label_clean.notna() & seg.text.notna()].drop_duplicates("text")
    logits = ft.logits(dev.text.tolist())
    y = np.array([ft.labels.index(l) for l in dev.label_clean])
    best = (1e9, 1.0)
    for t in np.arange(0.5, 5.01, 0.05):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        nll = float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean())
        best = min(best, (nll, float(round(t, 2))))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", default=str(settings.model_dir))
    ap.add_argument("--llm", default="qwen3:4b")
    args = ap.parse_args()
    ft_dir = Path(args.ft_dir)

    ft = FinetunedBackend.load(ft_dir)
    ft.temperature = 1.0
    t = fit_temperature(ft)
    (ft_dir / "calibration.json").write_text(json.dumps({"temperature": t, "fitted_on": "internal dev (held-out templates)"}))
    ft.temperature = t
    print(f"temperature = {t}")

    base = pd.read_csv(ROOT / "tests" / "suite" / "base.csv", dtype=str, keep_default_na=False)
    base = base[(base.split == "dev") & (base.ambiguous == "0")]
    segs = [split_segments(clean_message(x).text)[0] for x in base.message_text]
    gold = base.category.to_numpy()

    ft_pred = ft.classify(segs)
    ft_cat = np.array([c.value for c, _ in ft_pred])
    conf = np.array([p for _, p in ft_pred])
    tf_cat = np.array([c.value for c, _ in TfidfBackend.load().classify(segs)])
    sim = load_retriever().max_similarity(segs)
    llm_cat = np.array([c.value for c, _ in OllamaBackend(model=args.llm, prompt_version="v3").classify(segs)])

    rules = {"never (fine-tuned only)": np.zeros(len(segs), bool), "always (LLM only)": np.ones(len(segs), bool),
             "disagree with TF-IDF": ft_cat != tf_cat}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        rules[f"conf<{tau}"] = conf < tau
        rules[f"conf<{tau} or disagree"] = (conf < tau) | (ft_cat != tf_cat)
    for q in (0.1, 0.2, 0.3):
        sigma = float(np.quantile(sim, q))
        rules[f"sim<{sigma:.2f} (q{int(q * 100)})"] = sim < sigma
        rules[f"conf<0.8 or disagree or sim<{sigma:.2f}"] = (conf < 0.8) | (ft_cat != tf_cat) | (sim < sigma)

    rows = []
    ft_wrong = ft_cat != gold
    for name, route in rules.items():
        final = np.where(route, llm_cat, ft_cat)
        rows.append({"rule": name, "routed_pct": round(100 * route.mean(), 1),
                     "category_acc": round(float((final == gold).mean()), 4),
                     "ft_errors_caught": f"{int((route & ft_wrong).sum())}/{int(ft_wrong.sum())}"})
    df = pd.DataFrame(rows).sort_values("routed_pct")
    best = df.category_acc.max()
    eligible = df[(df.category_acc >= best - 0.01) & ~df.rule.str.startswith("always")]
    choice = eligible.sort_values(["routed_pct", "category_acc"], ascending=[True, False]).iloc[0].to_dict()
    print(df.to_string(index=False))
    print(f"\nchosen (cheapest within 1 point of the best {best:.4f}): {choice}")
    out = {"temperature": t, "n": len(segs), "rules": rows, "choice": choice,
           "ft_acc": float((ft_cat == gold).mean()), "llm_acc": float((llm_cat == gold).mean()),
           "tfidf_acc": float((tf_cat == gold).mean())}
    (settings.outputs_dir / "metrics" / "routing_analysis.json").write_text(json.dumps(out, indent=1))
    # the chosen rule as machine-readable config next to the model (read by backends.make_cascade)
    name = choice["rule"]
    cfg = {"threshold": float(name.split("conf<")[1].split()[0]) if "conf<" in name else 0.0,
           "disagree": "disagree" in name,
           "min_similarity": float(name.split("sim<")[1].split()[0]) if "sim<" in name else None,
           "chosen_rule": name, "chosen_on": "suite dev split"}
    if name.startswith("never"):
        cfg.update(threshold=0.0, disagree=False, min_similarity=None)
    (ft_dir / "routing.json").write_text(json.dumps(cfg, indent=1))
    print("routing config:", cfg)


if __name__ == "__main__":
    main()
