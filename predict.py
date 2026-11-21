"""Classify every complaint in a CSV and write predictions in evaluate.py format.

  python predict.py data/tickets_dev_gold.csv --backend llm --prompt v3 -o predictions/stage2_v3.csv
  python predict.py new_tickets.csv --backend ft -o predictions/live.csv
  python evaluate.py predictions/stage2_v3.csv

The input only needs ticket_id and message_text columns (any extra columns, e.g. gold labels, are ignored).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from orvana.backends import get_backend
from orvana.pipeline import process_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--backend", default="ft", choices=["llm", "tfidf", "ft", "cascade"])
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "llm_only"])
    ap.add_argument("--prompt", default="v3", help="LLM prompt version (v1 needs --mode llm_only)")
    ap.add_argument("--model", default=None, help="Ollama model tag, e.g. qwen2.5:7b-instruct")
    ap.add_argument("-o", "--output", default="predictions/predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"ticket_id": str})
    df["message_text"] = df["message_text"].fillna("")
    df = df.drop_duplicates("ticket_id")

    kw = {"model": args.model, "prompt_version": args.prompt} if args.backend in ("llm", "cascade") else {}
    backend = get_backend(args.backend, **kw)

    t0 = time.perf_counter()
    records = process_frame(df, backend, mode=args.mode)
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.eval_row() for r in records]).to_csv(out, index=False)
    with open(out.with_suffix(".records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    timing = {"backend": backend.name, "tickets": len(records), "total_s": round(elapsed, 2),
              "per_ticket_ms": round(1000 * elapsed / max(len(records), 1), 1)}
    calls = getattr(backend, "call_seconds", [])
    if calls:
        timing.update(llm_calls=len(calls), call_p50_ms=round(1000 * float(np.percentile(calls, 50)), 1),
                      call_p95_ms=round(1000 * float(np.percentile(calls, 95)), 1))
    out.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
