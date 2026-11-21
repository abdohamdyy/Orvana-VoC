"""Run the behavioural test suite against one backend.

    python scripts/run_suite.py --backend tfidf --split dev
    python scripts/run_suite.py --backend llm --model qwen2.5:7b-instruct --split dev
    python scripts/run_suite.py --backend cascade --split holdout      # ONCE, at the very end

Reports (outputs/suite/<tag>_<split>.json + a printed summary):
  base      evaluate.py metrics overall (+ bootstrap CI) and per marginal slice (language, phenomenon),
            strict (non-ambiguous rows) and lenient (ambiguous rows: any acceptable category counts)
  perturb   accuracy and INVARIANCE (same answer as for the clean text) per perturbation type
  security  label-unaffected rate (primary), injection-flag rate, HTML escaping, must_not (with --replies)
  cost      ms per ticket, share of segments sent to the LLM (cascade)
  contamination  suite texts too similar (cosine > 0.9) to training/retrieval texts
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.backends import get_backend  # noqa: E402
from orvana.config import settings  # noqa: E402
from orvana.eval.metrics import bootstrap_ci, norm, scores, wilson  # noqa: E402
from orvana.eval.perturb import perturb_all  # noqa: E402
from orvana.pipeline import process_frame  # noqa: E402

SUITE = Path(__file__).resolve().parent.parent / "tests" / "suite"
LABELS = ["category", "secondary_category", "product_mentioned", "store_mentioned", "sentiment", "urgency"]


def predict(df: pd.DataFrame, backend) -> pd.DataFrame:
    recs = process_frame(df[["ticket_id", "message_text"]], backend)
    return pd.DataFrame([{**r.eval_row(), "is_valid": r.is_valid, "injection_flag": r.injection_flag,
                          "confidence": r.confidence} for r in recs])


def slice_table(g: pd.DataFrame, p: pd.DataFrame, by: str) -> list[dict]:
    out = []
    for key, idx in g.groupby(by).groups.items():
        gs, ps = g.loc[idx], p.loc[idx]
        k = int((norm(gs.category) == norm(ps.category)).sum())
        out.append({by: key, "n": len(gs), "overall": round(scores(gs, ps)["overall"], 3),
                    "category_acc": round(k / len(gs), 3), "category_acc_ci95": wilson(k, len(gs)),
                    "urgency_acc": round(float((norm(gs.urgency) == norm(ps.urgency)).mean()), 3),
                    "entities_acc": round(float(((norm(gs.product_mentioned) == norm(ps.product_mentioned)) &
                                                 (norm(gs.store_mentioned) == norm(ps.store_mentioned))).mean()), 3)})
    return sorted(out, key=lambda r: r["overall"])


def contamination(texts: list[str]) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    seg = pd.read_csv(settings.data_dir / "processed" / "segments.csv")
    train = seg.text.dropna().drop_duplicates().tolist()
    extra = settings.data_dir / "processed" / "synthetic.csv"
    if extra.exists():
        train += pd.read_csv(extra).text.dropna().tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1).fit(train + texts)
    sims = (vec.transform(texts) @ vec.transform(train).T).max(axis=1).toarray().ravel()
    hits = [(t[:80], round(float(s), 3)) for t, s in zip(texts, sims) if s > 0.9]
    return {"suite_texts": len(texts), "near_duplicates_of_training": len(hits), "examples": hits[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tfidf", choices=["tfidf", "llm", "ft", "cascade"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="v3")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--replies", action="store_true", help="also generate suggested replies for security rows")
    args = ap.parse_args()
    if args.split == "holdout":
        print("WARNING: holdout run. This should happen once, after all choices are final.")

    kw = {"model": args.model, "prompt_version": args.prompt} if args.backend in ("llm", "cascade") else {}
    backend = get_backend(args.backend, **kw)
    tag = args.tag or backend.name.replace(":", "_").replace("/", "_").replace(" ", "")
    report = {"backend": backend.name, "split": args.split}

    # ---------------------------------------------------------------- base cases
    base = pd.read_csv(SUITE / "base.csv", dtype=str, keep_default_na=False)
    base = base[base.split == args.split].reset_index(drop=True)
    t0 = time.perf_counter()
    pred = predict(base, backend)
    report["ms_per_ticket"] = round(1000 * (time.perf_counter() - t0) / len(base), 1)
    strict = base.ambiguous == "0"
    g, p = base[strict].reset_index(drop=True), pred[strict].reset_index(drop=True)
    s = scores(g, p)
    report["base"] = {"n": len(g), **{k: round(v, 4) for k, v in s.items()}, "overall_ci95": bootstrap_ci(g, p),
                      "by_lang": slice_table(g, p, "lang"), "by_phenomenon": slice_table(g, p, "phenomenon")}
    amb = base[~strict]
    ok = [pc in gc.split("|") for pc, gc in zip(pred[~strict].category, amb.acceptable_categories)]
    report["base"]["ambiguous_lenient_category_acc"] = round(sum(ok) / max(len(ok), 1), 3)
    report["base"]["errors"] = [
        {"id": r.ticket_id, "text": r.message_text[:90], **{f: (r[f], q[f]) for f in LABELS if norm(pd.Series([r[f]]))[0] != norm(pd.Series([q[f]]))[0]}}
        for (_, r), (_, q) in zip(g.iterrows(), p.iterrows())
        if any(norm(pd.Series([r[f]]))[0] != norm(pd.Series([q[f]]))[0] for f in LABELS)]

    # ---------------------------------------------------------------- perturbations
    variants = perturb_all(list(zip(g.ticket_id, g.message_text)))
    vdf = pd.DataFrame(variants, columns=["ticket_id", "perturbation", "message_text"])
    vpred = predict(vdf, backend)
    vdf["base_id"] = vdf.ticket_id.str.split("~").str[0]
    gold_by_id, clean_by_id = g.set_index("ticket_id"), p.assign(ticket_id=g.ticket_id).set_index("ticket_id")
    rows = []
    for name, idx in vdf.groupby("perturbation").groups.items():
        ids = vdf.loc[idx, "base_id"]
        vp = vpred.loc[idx].reset_index(drop=True)
        gold = gold_by_id.loc[ids].reset_index(drop=True)
        clean = clean_by_id.loc[ids].reset_index(drop=True)
        inv = float((norm(vp.category) == norm(clean.category)).mean())
        rows.append({"perturbation": name, "n": len(idx), "overall": round(scores(gold, vp)["overall"], 3),
                     "category_acc": round(float((norm(vp.category) == norm(gold.category)).mean()), 3),
                     "invariance": round(inv, 3)})
    report["perturb"] = sorted(rows, key=lambda r: r["invariance"])

    # ---------------------------------------------------------------- security
    sec = pd.read_csv(SUITE / "security.csv", dtype=str, keep_default_na=False)
    sec = sec[sec.split == args.split].reset_index(drop=True)
    spred = predict(sec, backend)
    pure = sec.position == "pure"
    unaffected = [(not v or c == "inquiry") if is_pure else c == e for c, v, e, is_pure in
                  zip(spred.category, spred.is_valid, sec.expected_category, pure)]
    sec["unaffected"], sec["flagged"] = unaffected, spred.injection_flag
    from orvana.report.render import _env
    html_ok = all("<script" not in _env.from_string("{{ t }}").render(t=t) for t in sec.message_text)
    report["security"] = {
        "n": len(sec), "label_unaffected_rate": round(sum(unaffected) / len(sec), 3),
        "flag_rate_on_attacks": round(float(spred.injection_flag.mean()), 3), "html_escaped": html_ok,
        "by_type": sec.groupby("attack_type").agg(n=("unaffected", "size"), unaffected=("unaffected", "mean"),
                                                  flagged=("flagged", "mean")).round(3).reset_index().to_dict("records"),
        "failures": sec[~sec.unaffected][["ticket_id", "attack_type", "position", "expected_category"]]
        .assign(pred=spred.category[~sec.unaffected].values).to_dict("records")}
    if args.replies:
        from orvana.backends.llm_ollama import OllamaBackend
        from orvana.pipeline import process_text
        from orvana.replies import suggest_reply
        writer = OllamaBackend(model=args.model, prompt_version="reply")
        leaks = []
        for r in sec[sec.must_not != ""].itertuples():
            reply, _, _ = suggest_reply(process_text(r.message_text, backend), r.message_text, writer)
            bad = [m for m in r.must_not.split("|") if m and m.lower() in reply.lower()]
            if bad:
                leaks.append({"id": r.ticket_id, "found": bad, "reply": reply[:160]})
        report["security"]["reply_must_not_violations"] = leaks

    routed = getattr(backend, "routed", None)
    if routed is not None:
        report["llm_routed_segments"] = routed
    report["contamination"] = contamination(base.message_text.tolist())

    out = settings.outputs_dir / "suite"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_{args.split}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    b = report["base"]
    print(f"\n== {backend.name} | {args.split} | {report['ms_per_ticket']} ms/ticket")
    print(f"base strict n={b['n']}: overall {b['overall']:.3f} CI{tuple(round(x, 3) for x in b['overall_ci95'])} "
          f"| category F1 {b['category_macro_f1']:.3f} | urgency {b['urgency_accuracy']:.3f} | product {b['product_exact_match']:.3f} "
          f"| store {b['store_exact_match']:.3f} | ambiguous lenient cat acc {b['ambiguous_lenient_category_acc']}")
    print("by language:   " + "  ".join(f"{r['lang']}={r['overall']} (n={r['n']})" for r in b["by_lang"]))
    print("by phenomenon: " + "  ".join(f"{r['phenomenon']}={r['overall']} (n={r['n']})" for r in b["by_phenomenon"]))
    print("perturbations (invariance): " + "  ".join(f"{r['perturbation']}={r['invariance']}" for r in report["perturb"]))
    sc = report["security"]
    print(f"security n={sc['n']}: label unaffected {sc['label_unaffected_rate']} | flagged {sc['flag_rate_on_attacks']} | html escaped {sc['html_escaped']}")
    print(f"contamination: {report['contamination']['near_duplicates_of_training']} suite texts > 0.9 similar to training")
    print(f"errors on strict base: {len(b['errors'])}")


if __name__ == "__main__":
    main()
