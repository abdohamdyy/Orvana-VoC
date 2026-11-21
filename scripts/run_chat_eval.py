"""Run the frozen multi-turn chat scenarios (tests/chat/scenarios.jsonl) and score them.

    python scripts/run_chat_eval.py --split dev --llm off      # no LLM anywhere: fine-tuned classifier + templates
    python scripts/run_chat_eval.py --split dev --llm on       # production: cascade classifier + LLM phrasing
    python scripts/run_chat_eval.py --split holdout --llm on --final   # ONCE, after every choice is made

Writes outputs/chat/eval_<split>_<llm>.json. Expectation keys are documented in tests/chat/make_scenarios.py.
"""
import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orvana.chat.manager import ChatBot  # noqa: E402
from orvana.chat.orders import DemoOrderService  # noqa: E402
from orvana.config import settings  # noqa: E402
from orvana.eval.metrics import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "tests" / "chat" / "scenarios.jsonl"


def backends(use_llm: bool):
    if not use_llm:
        from orvana.backends.finetuned import FinetunedBackend
        return FinetunedBackend.load(), None
    from orvana.report.weekly import production_backends
    classifier, llm = production_backends(cache=False)  # exactly as in the live app
    if llm is None:
        sys.exit("--llm on needs Ollama running")
    return classifier, llm


def ar_share(text: str) -> float:
    letters = re.findall(r"[^\W\d_]", text)
    return sum("؀" <= ch <= "ۿ" for ch in letters) / max(len(letters), 1)


def score(sc: dict, st, results) -> dict:
    """Returns {check_name: passed} for every expectation of the scenario + the global invariants."""
    e, out = sc["expect"], {}
    replies = [r.reply for r in results]
    plans = [r.plan for r in results]
    facts = {f for p in plans for f in p.fact_ids()}
    real = [c for c in st.cases if c.category != "inquiry"]
    first = real[0] if real else None
    if "category" in e:
        out["category"] = (first is None) if e["category"] == "none" else (first is not None and first.category == e["category"])
    if "facts_any" in e:
        out["facts_any"] = bool(facts & set(e["facts_any"]))
    if "facts_none" in e:
        out["facts_none"] = not facts & set(e["facts_none"])
    if "unknown" in e:
        out["unknown"] = any(p.kind == "unknown" for pl in plans for p in pl.parts) == e["unknown"]
    if "order_verified" in e:
        out["order_verified"] = any(c.order is not None for c in st.cases) == e["order_verified"]
    for slot in ("store", "product"):
        if slot in e:
            out[slot] = first is not None and getattr(first, slot) == e[slot]
    if "voucher" in e:
        out["voucher"] = max([c.outcome.get("voucher_egp", 0) for c in st.cases] or [0]) == e["voucher"]
    if "can_cancel" in e:
        out["can_cancel"] = any(c.outcome.get("can_cancel") for c in st.cases) == e["can_cancel"]
    if "refund_days" in e:
        out["refund_days"] = next((c.outcome["refund_days"] for c in st.cases if "refund_days" in c.outcome), None) == e["refund_days"]
    if "escalate" in e:
        esc = {x for p in plans for x in p.escalate}
        out["escalate"] = (not esc) if e["escalate"] is None else e["escalate"] in esc
    if "handoff" in e:
        out["handoff"] = (st.status == "handed_off") == e["handoff"]
    if "ticket" in e:
        out["ticket"] = any(c.ticket_id for c in st.cases) == e["ticket"]
    if "status" in e:
        out["status"] = st.status == e["status"]
    if "must" in e:
        out["must"] = all(any(re.search(rx, r, re.IGNORECASE) for r in replies) for rx in e["must"])
    if "must_not" in e:
        # ids are masked first: a forbidden "1000" must not match inside "ORV-10001" (evaluator fix, made
        # before the holdout was ever run; see docs/CHATBOT.md)
        masked = [re.sub(r"ORV-\d{5}|CH-[A-Z]{6}", " ", r) for r in replies]
        out["must_not"] = not any(re.search(rx, r, re.IGNORECASE) for rx in e["must_not"] for r in masked)
    # global invariants
    asks = [p.ask for p in plans if p.ask]
    out["inv_no_loop"] = all(asks.count(a) <= 2 for a in set(asks))
    out["inv_nonempty"] = all(r.strip() for r in replies)
    want_en = sc["lang"] == "en"
    out["inv_language"] = all((ar_share(r) <= 0.1) if want_en else (ar_share(r) >= 0.5) for r in replies)
    return out


def contamination(scen: list[dict], limit: float = 0.9) -> list[dict]:
    """Scenario turns that are near-copies of a knowledge-base example question (cosine > limit on the
    KB's own representation): such a turn would test memorisation, not understanding."""
    from orvana.chat.kb import content_key, load_kb
    kb = load_kb()
    hits = []
    for sc in scen:
        for t in sc["turns"]:
            sims = (kb.matrix @ kb.vec.transform([content_key(t)]).T).toarray().ravel()
            if sims.max() > limit:
                hits.append({"id": sc["id"], "turn": t, "kb_doc": kb.docs[int(sims.argmax())], "cosine": round(float(sims.max()), 3)})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--llm", default="off", choices=["on", "off"])
    ap.add_argument("--ids", default="", help="comma-separated scenario ids (debugging)")
    ap.add_argument("--final", action="store_true", help="required for the holdout: it is evaluated once")
    ap.add_argument("--show", action="store_true", help="print every conversation")
    ap.add_argument("--phrasing", default=None, choices=["template", "opener", "rewrite"], help="default: settings.chat_phrasing")
    ap.add_argument("--judge", default="", help="judge model (a different one than the bot), e.g. qwen2.5:7b-instruct")
    args = ap.parse_args()
    if args.split != "dev" and not args.final:
        sys.exit("the holdout is evaluated ONCE, after all choices are made: add --final")

    scen = [json.loads(line) for line in SCEN.read_text(encoding="utf-8").splitlines()]
    if args.ids:
        scen = [s for s in scen if s["id"] in args.ids.split(",")]
    elif args.split != "all":
        scen = [s for s in scen if s["split"] == args.split]
    classifier, llm = backends(args.llm == "on")
    settings.chat_use_llm = args.llm == "on"
    phrasing = args.phrasing or settings.chat_phrasing
    judge = None
    if args.judge:
        from orvana.backends.llm_ollama import OllamaBackend
        from orvana.eval.chat_judge import KEYS, judge_conversation
        judge = OllamaBackend(model=args.judge, prompt_version="chat-judge", cache=False)

    rows, latencies, sources, rejected = [], [], [], 0
    for sc in scen:
        bot = ChatBot(classifier, llm, orders=DemoOrderService(), phrasing=phrasing)
        st = bot.new_state(sc["customer"])
        results = [bot.turn(st, t) for t in sc["turns"]]
        checks = score(sc, st, results)
        ok = all(checks.values())
        latencies += [r.latency_ms for r in results]
        sources += [r.source for r in results]
        rejected += sum(bool(r.problems) for r in results)
        rows.append({"id": sc["id"], "type": sc["type"], "lang": sc["lang"], "pass": ok,
                     "failed": [k for k, v in checks.items() if not v], "checks": checks,
                     "turns": [{"customer": t, "bot": r.reply, "source": r.source, "action": r.plan.action,
                                "facts": r.plan.fact_ids(), "problems": r.problems} for t, r in zip(sc["turns"], results)],
                     "final": st.snapshot()})
        if judge is not None:
            try:
                rows[-1]["judge"] = judge_conversation(judge, [(t, r.reply) for t, r in zip(sc["turns"], results)])
            except RuntimeError as e:  # one failed judge call must not lose the whole run; it is reported
                print(f"  judge failed on {sc['id']}: {e}")
        if args.show or (not ok and args.split == "dev"):
            print(f"\n[{'PASS' if ok else 'FAIL'}] {sc['id']} ({sc['type']}/{sc['lang']}) failed={rows[-1]['failed']}")
            for t, r in zip(sc["turns"], results):
                print(f"  C: {t}\n  B[{r.plan.action}/{r.source}]: {r.reply}")

    n, k = len(rows), sum(r["pass"] for r in rows)
    lo, hi = wilson(k, n)

    def rate(pred):
        sel = [r for r in rows if pred(r)]
        return {"n": len(sel), "pass": round(sum(r["pass"] for r in sel) / len(sel), 3) if sel else None}

    def key_acc(key):
        sel = [r["checks"][key] for r in rows if key in r["checks"]]
        return {"n": len(sel), "acc": round(sum(sel) / len(sel), 3) if sel else None}

    summary = {
        "split": args.split, "llm": args.llm, "classifier": getattr(classifier, "name", "?"),
        "llm_model": getattr(llm, "name", None), "when": datetime.now().isoformat(timespec="seconds"),
        "task_success": {"n": n, "rate": round(k / n, 3), "ci95": [round(lo, 3), round(hi, 3)]},
        "by_type": {t: rate(lambda r, t=t: r["type"] == t) for t in sorted({r["type"] for r in rows})},
        "by_lang": {g: rate(lambda r, g=g: r["lang"] == g) for g in sorted({r["lang"] for r in rows})},
        "checks": {key: key_acc(key) for key in sorted({c for r in rows for c in r["checks"]})},
        "contamination_kb_vs_turns": contamination(scen),
        "must_not_violations": sum(1 for r in rows if r["checks"].get("must_not") is False),
        "phrasing": phrasing if args.llm == "on" else "template",
        "judge": ({k: round(sum(r["judge"][k] for r in rows if "judge" in r) / max(1, sum("judge" in r for r in rows)), 2) for k in KEYS}
                  | {"model": args.judge, "judged": sum("judge" in r for r in rows)}) if judge else None,
        "turns": len(latencies), "llm_phrased_share": round(sources.count("llm") / max(len(sources), 1), 3),
        "turns_with_rejected_llm_draft": rejected,
        "latency_ms": {"p50": round(statistics.median(latencies), 1),
                       "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)},
    }
    out = settings.outputs_dir / "chat"
    out.mkdir(parents=True, exist_ok=True)
    tag = ("subset" if args.ids else args.split) + (f"_{phrasing}" if args.llm == "on" else "")
    (out / f"eval_{tag}_{args.llm}.json").write_text(json.dumps({"summary": summary, "scenarios": rows}, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    if args.split == "holdout":
        with (out / "holdout_runs.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
