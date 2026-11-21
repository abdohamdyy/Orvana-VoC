# Bonus

## B1: Suggested replies in Egyptian Arabic (`orvana/replies.py`)

**How a reply is made**
- The local LLM (`qwen3:4b`) receives the record plus **only the policy lines relevant to it** (from `docs/cs_policy_en.md`).
- The **voucher amount is decided in code**, not by the LLM. The LLM once offered 100 EGP for a 7-day delay; the policy says "more than 7 days".
- Internal rules (no cash compensation, 24 h first reply) are enforced by the checker and are not shown to the model. Otherwise the model repeated "مفيش تعويض نقدي" to customers.

**Automatic checks on every draft** (`check_reply`; a failure triggers one regeneration, then a safe template):
- the reply is mainly Arabic;
- it has a reasonable length;
- it says **nothing about cash**;
- it contains **no number that is not in the policy or the complaint** (Franco digits such as "3aref" are not counted as numbers);
- safety cases **mention escalation to the Quality team**.

**How we know the replies are good** (`python scripts/eval_replies.py --judge qwen2.5:7b-instruct`, 36 tickets stratified by category; `outputs/replies/`):

| Measure | Result |
|---|---|
| Final replies passing all policy checks | **100%** |
| Replies written by the LLM (rest: safe template) | 91.7% |
| Typical rejected drafts | invented numbers ("خلال 24 ساعة" with no policy basis; computed "25 points") |
| LLM judge, **a different model** (qwen2.5:7b), 1–5 | policy **3.56**, tone **3.97**, helpfulness **2.97** |
| Human review | `outputs/replies/replies_review.csv` has columns for a CS agent to score all 36 replies (policy, tone, helpfulness, notes). **Not done yet: the columns are still empty.** This review should be the final word. |

**Honest reading**
- The replies are **safe**: no cash, no invented numbers, safety cases escalated.
- They are only **moderately helpful**: a 4B model's Egyptian Arabic is limited.
- They are drafts for an agent to edit, not messages to send automatically.
- A larger local model would improve fluency, but in our tests it also invented details, so the checks would stay.

## B2: UI (`web/`, served by `app.py`)
`python app.py` → http://127.0.0.1:8000.

It is a React + Vite app. The **build is committed** (`web/dist`), so running it needs no Node, and it works offline (the Arabic font is bundled). A sidebar leads to two pages:

**Classification**
- **One complaint:**
  - the structured record (category / secondary with icon + label, urgency, sentiment, brand, branch name, confidence);
  - a **"Why this decision"** card: the cleaned text, the split into primary and secondary with each part's category and confidence, the rules that fired (urgency reason, safety, double charge, repeat), and any injection removed;
  - the suggested reply with copy.
- **CSV upload** (or the two weeks in `incoming/`):
  - a dashboard with KPI tiles, ranked category bars, urgency, and top branches and brands. The charts use one hue; identity is an icon + label, never colour alone;
  - a filterable table;
  - **Download predictions** in evaluate.py format, with CSV formula injection neutralised.
  - The counts for week 1 match the weekly email exactly: the same pipeline.

**Customer chat:** the B3 chatbot. Beside the conversation, the **agent panel** shows what the code decided: cases, order verified, voucher and refund days, urgency, ticket, the policy lines applied, and the demo orders.

**Server** (still the standard library):
- serves only files inside `web/dist` (path traversal → 404);
- a strict Content-Security-Policy (no external requests);
- request caps: 20 KB, or 5 MB for a CSV;
- never logs bodies, never sends stack traces.

Payload logic lives in `orvana/api.py` (tested in `tests/test_api.py`). For example, the explanation's record is identical to `process_text` on all 150 gold texts.

## B3 (extension): a customer chatbot (`orvana/chat/`, Customer chat page)
B1 drafts one reply for an agent. B3 **talks to the customer**:
- it answers policy questions;
- it asks for the missing details of a complaint (order number, branch, product);
- it looks the order up and works out the entitlement **in code** (50/100 EGP voucher, cancellation, refund days);
- it escalates safety cases immediately and hands over to a human when needed.

The full design and results are in [CHATBOT.md](CHATBOT.md).

**Result:** on the frozen multi-turn holdout (evaluated once), **97.2%** of conversations were fully correct (91.7% with no LLM at all), with 0 forbidden-content violations. Vouchers, refund days, safety escalation and hand-over were 100% correct.

In short:
- **The code decides, the LLM only words.** A state machine makes every decision.
- **Templates are the default wording.** On dev, an LLM rewrite scored no better with a separate judge, took ~4.5 s instead of 7 ms, and once turned an apology into an implied compensation promise.
- **Evaluation:**
  - 91 multi-turn scenarios, written and frozen **before** the dialogue manager, with a holdout evaluated once;
  - 40 simulated customers played by a different model, checked for invariants;
  - a judge model;
  - a sheet for a human reviewer.
- **Privacy:**
  - no transcripts are stored;
  - the LLM disk cache is off for anything live;
  - one redacted ticket row is written per complaint (no phone numbers, emails, cards or health details).
