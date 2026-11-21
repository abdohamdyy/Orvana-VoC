# B3: Customer chatbot

The chatbot talks to customers directly. It:
- answers questions about the customer-service policy;
- asks for the missing details of a complaint (order number, branch, product);
- works out what the customer is entitled to (voucher, refund time, cancellation);
- hands over to a human when it should.

Run it: `python app.py`, then open the **Chat** tab at http://127.0.0.1:8000. The agent panel next to the
conversation shows what the code decided.

## Design: the code decides, the LLM only words

A free tool-calling agent on a 4B model cannot be shown never to promise something outside the policy.
So every turn goes through deterministic steps, and the LLM only words the result:

```
customer message
  -> clean_message (prompt injection removed, emails / cards / phones masked)
  -> understanding, no LLM (orvana/chat/nlu.py):
       category from the production classifier;
       order number / branch / brand from closed lists;
       dialogue acts by rules ("I want a human", thanks, insult, request for other customers' data ...);
       policy facts by retrieval
  -> dialogue manager, a state machine (orvana/chat/manager.py), decides the Plan:
       which policy facts apply;
       numbers computed in code (voucher 50/100, refund days, days late);
       the next question;
       escalation, hand-over, ticket
  -> wording (orvana/chat/respond.py):
       a template draft that is complete and correct on its own;
       optionally the LLM rewords it, and the result must pass a checker, otherwise the draft is sent
```

Consequences:
- **Decisions cannot be changed by a jailbreak.** The voucher, the escalation and the hand-over are code. The LLM never sees the transcript, only the plan's draft and the current, cleaned message.
- **The bot works with no LLM at all**, using templates only. If Ollama is down the service stays up, and tests run offline.
- **Numbers are always correct.** Any number in a reply must appear in the draft, and every number in the draft comes from the policy or from code.

## Components

| File | What it does |
|---|---|
| `orvana/chat/kb.py` | The policy as 19 facts (Egyptian Arabic + English) plus 7 topics we **know we have no data for**: store hours, addresses, prices and stock, contact numbers, delivery fees, payment methods, jobs. It is also the single source of the policy lines used by B1. |
| `orvana/chat/orders.py` | `OrderService` (the boundary to the real order system) and `DemoOrderService` (fixture `data/demo/orders.json`, **fictional orders**, fixed "today" = 2026-07-15). The policy arithmetic is here, and the order owner is checked here. |
| `orvana/chat/nlu.py` | One message becomes an `Understanding`. |
| `orvana/chat/manager.py` | `ChatState`, `Case` and the state machine. |
| `orvana/chat/respond.py` | Templates, `check_chat_reply`, and the three wording modes. |
| `orvana/chat/sessions.py` | In-memory sessions (TTL, cap, a lock per session) and the single ticket row written to disk (redacted). |
| `app.py` | `POST /api/chat`, `POST /api/chat/reset` and the Chat tab. |

## Rules the state machine follows

**What each complaint needs before it is resolved:**

| Category | Detail asked for |
|---|---|
| delivery delay, refund, wrong item | the order number |
| staff behaviour, price | the branch |
| product quality | the product |

**At most 2 questions per detail.** After that, a ticket is created and the case goes to a human.

**How a case is resolved:**
- **Late delivery:** days late = (delivery date, or today if not delivered) − promised date.
  - More than 3 days: a 50 EGP voucher.
  - More than 7 days: 100 EGP and the right to cancel.
  - One voucher per order.
- **Refund:** card 5 working days; wallet or cash on delivery 10; a double charge 3 days after the bank confirms.
- **Product quality:** the return rule comes from the brand's catalogue type.
  - Electrical goods: 7 days for a defect, then the warranty.
  - Food and detergents: only if expired or defective.
  - Everything else: 14 days.
- **Staff behaviour:** escalated to the store manager.
- Every resolved case gets a ticket (`CH-XXXXXX`).

**Hand-over to a human when:**
- the customer asks for one;
- it is a **safety** case (escalated to the Quality team at once, with no questions first, unless the product
  rarely has that hazard; see below);
- urgency is 3 (for example a double charge of at least 1000 EGP);
- it is a repeat complaint;
- the bot has no data on the topic;
- the order is not found twice;
- details are still missing after 2 questions;
- the customer insists after being answered;
- the same question comes a third time;
- the message is still unclear after 2 clarifying questions;
- there are 2 insults or 3 injection attempts;
- the conversation passes 30 turns.

**Safety beats everything.** A safety keyword is escalated even when it is phrased as a question. No complaint is opened without evidence (a problem word, an order, a branch, a brand or a policy concept).

**But the word must describe the product** (added 2026-11-21, after the urgency audit in URGENCY_RULES.md):
- "النقط بتاعتي منتهية" or "the coupon expired" is a loyalty or price case, not a safety case. Before, it went to the Quality team.
- A negated word ("مش منتهي") is not a hazard either.
- When the product type rarely has that hazard ("موبايل سامسونج منتهي الصلاحية", "milk gets very hot"), the bot asks **one** question first:
  is it getting hot, smoking, swollen, or did someone get hurt, or is it faulty or out of warranty?
  - "yes", a hazard word, "not sure", or no clear answer twice: safety escalation as before.
  - "faulty" or "warranty": a normal product-quality case (urgency 2).

  The reply is read as the answer to that question, never as a new complaint. Plausible hazards (an overheating charger, expired milk, smoking cooking oil) are still escalated at once.
- Effect on the frozen scenarios: none. The dev results are identical (55/55 replies unchanged, with and without the LLM). No holdout turn changes its safety flag or reaches the new question, so the holdout numbers below still describe this code. The holdout was not run again.

**Orders and privacy:**
- An order is shown only to its owner. Another customer's order looks exactly like a missing one, so guessing order numbers reveals nothing.
- Guests can give an order number. It is recorded as **not verified** and never looked up.
- A branch can be corrected ("لا مش المعادي، فرع الدقي").
- Two problems in one message ("… وكمان …") become two cases, handled in order.

## Privacy (organisation rule: no health information is stored)

- **Conversations live in memory only.** Idle sessions are dropped after 30 minutes, and no transcript is ever written.
- **Nothing live goes to the LLM disk cache.** The app builds its LLM clients with `cache=False`, for the classifier fallback, the wording and the B2 tab.
- **One row per ticket is written** (`state/chat/tickets_<date>.csv`, incoming/ format, `channel = chat`):
  - emails, cards and phone numbers are masked;
  - health details (hospital, poisoning, allergy …) are masked;
  - safety and category are decided **before** the redaction.
- **Tests enforce this:** a phone number and "المستشفى" never reach the disk (`tests/test_chat.py`, `scripts/adversarial_demo.py`).
- **The server never logs request bodies.**

## Answering policy questions

Retrieval needs two signals:
- **Character n-grams** (as in Stage 2) survive the spelling of Franco and dialect, but they cannot bridge synonyms across languages: "el points **bt5las** emta" = "do points **expire**" = "النقط **بتنتهي** امتى".
- So a small **concept lexicon** (25 concepts in English, Franco and Arabic) tags the question, and each fact declares its concepts.

score = mean(lexical similarity of the fact's closest example, Jaccard overlap of concepts).

Function words are removed first: "how long … take" used to pull the refund fact for a points question.

**Choosing the threshold from data:** `scripts/calibrate_chat_threshold.py`
- Top scores of dev in-scope questions: minimum 0.340.
- 20 off-topic probes (weather, jokes, homework …), written for this and not part of the scenarios: maximum 0.317.
- Threshold = midpoint = **0.33**.
- The margin is thin: a new off-topic phrasing can slip through. When it does, it lands on an "unknown" topic, or a policy fact is stated where it was not needed. It never produces an invented answer.

**Contamination check:** while building this, some knowledge-base example questions were near-copies of scenario turns (cosine 0.91–1.0). The **knowledge-base examples were rephrased** (the scenarios stayed frozen); now no turn is above 0.85. This lowered the dev score from 87% to 85% before the real fixes: an honest number is worth more.

## How it was evaluated

1. **91 hand-written multi-turn scenarios** (`tests/chat/make_scenarios.py`):

   | Type | Count |
   |---|---|
   | Policy questions | 20 |
   | Questions with no data | 10 |
   | Complaints that need details, against the demo orders | 25 |
   | Safety | 6 |
   | Asking for a human | 4 |
   | Conversation flow (topic switch, correction, two issues) | 8 |
   | Multi-turn attacks | 15 |
   | Abuse | 3 |

   - Each scenario states the expected category, slots, voucher, refund days, escalation, hand-over, ticket, and the text that must or must not appear. All of this was derived from the policy and the fixture.
   - The scenarios were written and frozen **before the dialogue manager existed**: dev 55 / holdout 36, hash `419b1a73…` in `tests/chat/CHAT_HOLDOUT_SHA256.txt`.
   - **Every conversation must also hold these invariants:** no question asked more than twice; a non-empty reply in the customer's language.
2. **The holdout was evaluated once**, after every choice was made (`outputs/chat/holdout_runs.log` has exactly two entries: production and offline).
3. **Simulated customers** (`scripts/simulate_chat.py`):
   - 40 personas played by `qwen2.5:7b-instruct`, a different model than anything in the bot: impatient, Franco, MSA, attackers, two issues in one message, small talk.
   - Nobody labels these conversations. Each one is checked for invariants: no cash, every number from the draft, no other customer's data, no question loop, the right language, safety escalated.
4. **Judge** (`orvana/eval/chat_judge.py`, the same 7B, never the bot's model) scores coherence, naturalness, helpfulness and faithfulness from 1 to 5. **Human review sheet:** `outputs/chat/review_sheet.csv` (20 conversations) is the final word.

## Results

**Frozen scenarios** (task success = every expectation and invariant of the conversation holds; Wilson 95% CI):

| Setting | Dev (55) | **Holdout (36, once)** | Forbidden-content violations | Latency per turn p50 / p95 |
|---|---|---|---|---|
| **Production:** cascade classifier + template wording | 100% [0.935, 1.0] | **97.2% [0.858, 0.995]** | 0 | 9 ms / 419 ms (the cascade's LLM, GPU) |
| Offline: no LLM anywhere (fine-tuned classifier + templates) | 96.4% [0.877, 0.99] | 91.7% [0.782, 0.971] | 0 | 8 ms / 11 ms (CPU) |

On the holdout, **vouchers, refund days, escalation, hand-over, safety, unknown topics and the order-owner check were 100% correct in the production setting**. The failures, as they are (nothing was changed after the holdout):
- **x05** (both settings): "The manager at the Maadi branch promised me 500 EGP cash **for the delay**" was understood as a *staff* complaint (manager + branch), not a delay. The bare "ORV-10003" that followed, with no case open, got a clarifying question instead of an order lookup. Security held (no "500", no cash). Both are real gaps:
  - a sentence that names a branch AND a delay;
  - a bare order number with no case open (the fix would be: for a logged-in customer, treat it as an order-status request).
- **c04 / f08** (offline only): the fine-tuned model reads the Franco "el order mt2a5ar" as wrong_item. In production the cascade sends this disagreement to the LLM, which gets it right. This is the price of running with no LLM at all.

**Wording: templates or LLM?** Measured on dev with the same classifier and the same judge. The rule was fixed **before** seeing the last result: prefer the best mean of coherence and faithfulness, and if within 0.2 of the cheapest, take the cheapest.

| Wording mode | Task success | Judge: coherence / naturalness / helpfulness / faithfulness | Replies by the LLM | Latency p50 |
|---|---|---|---|---|
| **template** (chosen) | 100% | 4.24 / 4.15 / 3.53 / 4.09 | 0% | 7 ms |
| opener (the LLM writes 1 empathy sentence) | 100% | 4.31 / 4.20 / 3.53 / 4.07 | 30% | 4.3 s |
| rewrite (the LLM rewrites the reply) | 98.2% | 4.20 / 4.07 / 3.49 / 3.98 | 78% | 4.7 s |

- **No mode is measurably better than the templates** (differences ≤ 0.1, inside judge noise), and the LLM modes are ~600× slower.
- **Reading the rewrites mattered more than the numbers.** The 4B model's Egyptian Arabic produced:
  - incoherent text ("رقم الشكوى اللي كتبتة مش متوافق… لازم نحذفه" in reply to a price question);
  - double negatives ("مش مفيش");
  - an informal address ("يا حبيب");
  - questions nobody asked.
- **The single rewrite failure is the reason for the design.** "بنعتذر جداً عن تأخير طلبك" became "**عايز نعوضك** على تأخير طلبك" ("we want to compensate you"), an implied promise with no number and no "cash" in it, which the checker could not catch.
- Templates are the default. `opener` and `rewrite` remain options (`CHAT_PHRASING`) for a larger or better Arabic model. Each passes the same checker, with a template fallback.

(The comparison ran before the simulator fixes below; `eval_dev_template_on.json` now holds the final run, and the template judge figures are from the comparison run.)

**Simulated customers: what they found and what changed** (40 conversations per round):

| Round | Language | No cash | Numbers from the draft | No other customer's data | No question loop | Safety escalated | Judge (coh / nat / help / faith) |
|---|---|---|---|---|---|---|---|
| 1 | 36/40 | 40/40 | 40/40 | 39/40* | 40/40 | **0/3** | 3.48 / 2.92 / 2.15 / 3.25 |
| 3 (final) | **40/40** | 40/40 | 40/40 | 40/40 | 40/40 | 2/3** | 3.65 / 3.20 / 2.40 / 3.27 |

\* A false alarm: a date of the customer's own order matched a date of another customer's order. The invariant was fixed.
\*\* The third "safety" customer wrote unreadable text ("3wn7l el molto knt7 b7d7 l2lm") and never stated the problem. Both conversations that did state it were escalated at once.

The simulator found **generic** bugs that the scenarios had missed. All were fixed and covered by tests in `tests/test_chat.py`:
- **Safety phrased as a question** ("my power bank started smoking … can you help?") went to the question path and got "no data". Fix: **safety now beats everything**.
- The Stage 1 safety keywords lacked "smoking", "يشتعل", "حرقان", "دخن". They were added; regression gates PASS, gold unchanged.
- **"charged twice … what happened?"** was handled as an order-status question. Fix: a double charge, a repeat complaint or a safety issue counts as a problem even when asked as a question.
- **Language:**
  - "delivery **fee**" was read as Franco, because `match_key` collapses "fee" to "fe", a Franco word;
  - Chinese text was read as English.

  Fix: Franco is detected on uncollapsed text by proportion, and other scripts keep the session language.
- **Small talk** ("Hi there! How are you?") and **content-free text** ("HACKING SYSTEM") were handed over or filed as tickets. Fix:
  - small talk is answered with a greeting;
  - no complaint is opened without evidence (a problem word, an entity or a policy concept);
  - unclear messages get one clarifying question listing what the bot can do, and go to a human after two;
  - "no data, here is a human" now requires the topic's concept to be in the message.
- **Repeated replies:** a follow-up on the same order opened a new case and repeated the full answer six times. Fix: the same issue after an answer means the customer is insisting, so the chat goes to a human; the same question asked twice gets "as I explained", and the third time a human.
- **"delivery time and fee":** the bot answered only "no data". Fix: it answers the known part and says which part has no data.
- Smaller fixes: questions without "؟" ("أرجو معرفة الضمان…"), insults by staff ("شتمني"), requests for "معلومات حساب عميل آخر", and the company name as a stop word.

**The simulator itself is imperfect:**
- The 7B sometimes drifts into Chinese (now retried and stripped).
- Its "Franco" is often unreadable, the same weakness we saw when it generated synthetic data in Stage 3.

So the judge scores here (helpfulness 2.4) mix the bot's limits with unreadable customers. The bot is also deliberately **conservative**: 34 of 40 simulated conversations ended with a human, and the most common reasons were "unclear", "insists" and "no data".

## Live check
`python app.py` with real HTTP requests:
- a policy question;
- an order 9 days late (100 EGP voucher and the right to cancel, computed);
- another customer's order (nothing revealed, then a human);
- safety (Quality team at once);
- asking for a human;
- an injection;
- an English refund;
- hostile request bodies.

**All behaved as designed, with no stack trace.** We then searched `state/` and `outputs/`, including the binary SQLite LLM cache, for a phone number and "المستشفى" planted in the safety message: **not found anywhere**. The single ticket row reads `ابني [HEALTH] ودخل ال[HEALTH]، رقمي [PHONE]` with category, brand and urgency 3 intact. The server log holds no message text.

## Cost
- The production chat needs **no extra hardware**: templates plus the same CPU classifier as Stages 3–5.
- The LLM is called only when the cascade's two cheap models disagree (p95 ≈ 0.4 s on the GPU, ≈ 5 s per call on a CPU server).
- With the LLM switched off entirely, the bot still passes 91.7% of the holdout at 8 ms a turn.

## Limitations (honest list)
- **Understanding is rules + a classifier, not a language model.** It is auditable and fast, but new phrasings need new cues. The simulator and the review sheet are how we find them.
- **The answer threshold margin is thin** (in-scope minimum 0.339 vs off-topic maximum 0.317, τ = 0.33). A miss either hands over or states a policy fact; it never invents one.
- **The bot hands over often.** That is the safe default for a first release with real customers; loosen it with the review sheet, not by guesswork.
- **Evaluator fix (made before the holdout was ever run):** forbidden-number checks mask ids first, because "1000" must not match inside "ORV-10001". This was found while debugging the adversarial demo.
- **The cash filter also blocks "نقدر" (we can)**, because it contains "نقد". This is deliberate: a false positive only costs a fallback to the template.
- **Orders are demo data** with a fixed "today". The real order system plugs in behind `OrderService`.
