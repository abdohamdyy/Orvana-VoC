"""B1 evaluation: how do we know the suggested replies are good?

    python scripts/eval_replies.py [--n 40] [--judge qwen2.5:7b-instruct]

1. Automatic policy checks on every draft (orvana.replies.check_reply): Arabic, length, no cash
   compensation, no number that is not in the policy or the complaint, safety cases escalated.
   We report how often the LLM's FIRST draft passes, after one retry, and how often the safe
   template had to be used.
2. Optional LLM judge (a DIFFERENT model than the writer) scoring policy consistency, tone and
   helpfulness 1-5 with a rubric.
3. A review sheet (outputs/replies/replies_review.csv) with empty columns for a CS agent to score
   20 replies by hand. The human scores are the final word.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.backends.llm_ollama import OllamaBackend  # noqa: E402
from orvana.backends.tfidf import TfidfBackend  # noqa: E402
from orvana.config import settings  # noqa: E402
from orvana.pipeline import process_frame  # noqa: E402
from orvana.replies import check_reply, policy_for, suggest_reply  # noqa: E402

JUDGE = """You review a customer-service reply for an Egyptian retailer. Score 1-5 each:
policy (only promises allowed by the policy lines given; no cash compensation),
tone (polite, empathetic, natural Egyptian Arabic), helpful (addresses the complaint, clear next step).
Return JSON {"policy": int, "tone": int, "helpful": int, "comment": "..."}."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--judge", default="")
    args = ap.parse_args()

    w = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(settings.incoming_dir.glob("*.csv"))])
    w = w.dropna(subset=["message_text"]).drop_duplicates("ticket_id")
    recs = {r.ticket_id: r for r in process_frame(w, TfidfBackend.load())}
    w["category"] = w.ticket_id.map(lambda t: recs[t].category.value)
    w = w[w.ticket_id.map(lambda t: recs[t].is_valid)]
    per = max(1, args.n // 9)
    sample = w.groupby("category", group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=0))

    writer = OllamaBackend(prompt_version="reply")
    judge = OllamaBackend(model=args.judge, prompt_version="judge") if args.judge else None
    rows = []
    for _, r in sample.iterrows():
        rec = recs[r.ticket_id]
        reply, source, first_problems = suggest_reply(rec, r.message_text, writer)
        row = {"ticket_id": r.ticket_id, "category": rec.category.value, "urgency": rec.urgency,
               "complaint": r.message_text, "reply": reply, "source": source,
               "llm_draft_problems": "; ".join(first_problems),
               "final_passes_checks": not check_reply(reply, rec, r.message_text)}
        if judge:
            user = json.dumps({"policy_lines": policy_for(rec, r.message_text), "complaint": r.message_text,
                               "reply": reply}, ensure_ascii=False)
            schema = {"type": "object", "properties": {"policy": {"type": "integer"}, "tone": {"type": "integer"},
                      "helpful": {"type": "integer"}, "comment": {"type": "string"}},
                      "required": ["policy", "tone", "helpful", "comment"]}
            row.update(judge._chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": user}], schema))
        row.update(human_policy_1to5="", human_tone_1to5="", human_helpful_1to5="", human_notes="")
        rows.append(row)

    df = pd.DataFrame(rows)
    out = settings.outputs_dir / "replies"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "replies_review.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n": len(df),
        "llm_reply_used_pct": round(100 * (df.source == "llm").mean(), 1),
        "template_fallback_pct": round(100 * (df.source == "template").mean(), 1),
        "final_replies_passing_checks_pct": round(100 * df.final_passes_checks.mean(), 1),
        "problems_seen_in_rejected_drafts": df.llm_draft_problems[df.llm_draft_problems != ""].tolist(),
    }
    if judge:
        summary.update({k: round(float(df[k].mean()), 2) for k in ("policy", "tone", "helpful")})
    (out / "replies_eval.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
