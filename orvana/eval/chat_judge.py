"""LLM judge for chatbot conversations. Use a DIFFERENT model than the bot (bot: qwen3:4b; judge:
qwen2.5:7b-instruct), so it does not grade its own style. A judge is a noisy instrument: use it to
compare variants under the same judge, and keep a human review sheet as the final word."""
from pathlib import Path

POLICY = (Path(__file__).resolve().parents[2] / "docs" / "cs_policy_en.md").read_text(encoding="utf-8")
KEYS = ("coherence", "naturalness", "helpfulness", "faithfulness")
JUDGE = f"""You review a customer-service chatbot conversation for an Egyptian retailer. The retailer's policy:
{POLICY}
Score the BOT's replies from 1 (bad) to 5 (excellent) on each criterion:
- coherence: every sentence makes sense and fits the conversation (nothing confusing or contradictory)
- naturalness: sounds like a polite human agent writing in the customer's language (Egyptian Arabic or English)
- helpfulness: moves the customer's problem forward
- faithfulness: everything said agrees with the policy; no invented promises, amounts or dates
Answer with JSON only."""
SCHEMA = {"type": "object", "properties": {k: {"type": "integer"} for k in KEYS}, "required": list(KEYS)}


def judge_conversation(llm, turns: list[tuple[str, str]]) -> dict:
    """turns = [(customer, bot), ...] -> {criterion: 1..5}"""
    transcript = "\n".join(f"CUSTOMER: {c}\nBOT: {b}" for c, b in turns)
    out = llm._chat([{"role": "system", "content": JUDGE}, {"role": "user", "content": transcript}], SCHEMA, use_cache=False)
    return {k: max(1, min(5, int(out[k]))) for k in KEYS}
