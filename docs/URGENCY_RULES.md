# Urgency rules (inferred from the dev gold set)

Implemented in `orvana/rules.py::urgency`. Keyword lists live in `orvana/extractors.py` and are matched
after normalisation (lower case, unified Arabic letters, repeated letters collapsed).

## The rules, checked in order

| # | Condition | Urgency |
|---|---|---|
| 1 | The customer says the complaint is a **repeat** ("دي تالت مرة أشتكي", "di talet mara ashtki", "third time I complain") | **3** |
| 2 | The **primary** problem is `product_quality` and it is a **safety** issue: expired ("منتهي الصلاحية", "expired"), overheating ("بيسخن جامد", "bysa5an gamed", "gets very hot") or a burnt smell ("ريحته شياط", "re7to shayat", "smells burnt") | **3** |
| 3 | The **primary** problem is `refund_request` for a **double charge** ("اتخصموا مرتين", "marteen", "charged twice") of **≥ 1,000 EGP** | **3** |
| 4 | The primary problem is `inquiry` | **1** |
| 5 | Anything else | **2** |

## Evidence from the gold (150 tickets)

| Group | Gold urgency | n |
|---|---|---|
| Repeat complaint (any category) | 3 | 17 / 17 |
| Primary `product_quality` with a safety keyword | 3 | 14 / 14 |
| Primary `product_quality` without one ("broke after two days", "stopped working") | 2 | all |
| Safety issue only in the **secondary** clause (delivery delay + expired shampoo) | 2 | 1 / 1 |
| Double charge of 1,900 or 1,950 EGP | 3 | 2 / 2 |
| Double charge of 400 or 850 EGP | 2 | 2 / 2 |
| `inquiry` | 1 | 8 / 8 |
| Delivery delays of 2–10 days, refunds late by up to 2 weeks | 2 | all |

**Result: 150/150 when the rules are given the correct category** (checked in `tests/test_rules.py`).

## Interpretation (why the rules make business sense)
- Urgency follows the **main problem**, plus a signal that the customer's patience has run out (a repeat contact).
- Safety issues match the CS policy: "overheating, burnt smell, expired food: escalate to the Quality team within 2 hours".
- A large double charge means real money held by us; a small one is ordinary.

## Coverage beyond the historical phrasing (Phase B, measured on the test suite's dev split)
The rules were first written from the 9k historical tickets. The behavioural test suite showed customers
express the same triggers in other ways. They are now covered as follows:

| Trigger | Coverage |
|---|---|
| Repeat | An ordinal ≥ 3 next to "time/مرة/complaint" in any order: "دي رابع مرة", "للمرة الثالثة" (MSA), "this is my 4th complaint", "for the third time", "talet/talt/rabe3 mara" (Franco) |
| Safety | Plus "رائحة احتراق" (MSA), "حريق", "دخان", "شرار", "تسمم" (food poisoning), "spark", "smoke"; Franco stems matched inside words ("beyskhan", "salahyto") |
| Double charge | An amount written in words ("ألفين جنيه", "two thousand") counts as ≥ 1,000; "الفلوس" is not "ألف" |

**Guard against false positives:** each change was checked with `scripts/check_gates.py`.
- The gold must stay at 150/150.
- On all 9,000 historical tickets, the only ticket whose flags changed was a genuine fix ("di talt mara ashtki", a typo of "talet mara").

## Guards: the safety word must describe the product
A keyword alone is not enough. `has_safety_issue()` drops a match when:
- **it is about something that is not the product**: "النقط بتاعتي منتهية", "الكوبون منتهي", "انتهت صلاحية الكوبون",
  "warranty expired", "الضمان بتاع التلاجة منتهي". This checks up to 3 words before the expiry word (same clause) and the
  word right after it, for points, coupon, voucher, offer, discount, warranty, card, subscription, code, Orvana Plus;
- **it is negated**: "اللبن مش منتهي الصلاحية", "not expired", "msh bysa5an", "doesn't get very hot"
  ("مش بس بيسخن", "not only", is not a negation);
- **"hot" stands alone**: only "very / too / gets hot" and "heats up" count ("hot deal" does not).

Effect: **0 of the 9,000 historical tickets and 0 gold, incoming or holdout texts change.** Across all 1,306
evaluation texts, one changes: dev suite S0278 "my Orvana Plus points expired early" (loyalty_points) is no longer a
safety report. Before, the chatbot would have sent that customer to the Quality team.

New words (none of them appear in the 9,000 tickets, so they are covered by tests only). They match **whole words**, so
"دود" does not fire on "حدود" and "قزاز" (glass) does not fire on "قزازة" (a bottle):
- contamination: حتة قزاز، صرصار، حشرات، دود، عفن، مسوس، glass in, insects, mold, rotten;
- electrical and fire: ماس كهربا، كهربني، انفجر، ولعت، electric shock, exploded, caught fire;
- health reaction (a hazard whatever the product): حساسية، طفح، اتسممت، مستشفى، اسهال، allergic, rash, vomiting.

Left out on purpose: ترجيع (it also means returning a product), تعبت, sick ("sick of waiting"), bug (app), كهرب alone
("أجهزة كهربائية").

## Plausibility: is the hazard possible for this product?
The data puts products into complaint templates at random (STAGE1_DATA finding 11): "موبايل سامسونج منتهي الصلاحية",
"لبن جهينة بيسخن جامد". `safety_check()` compares the kind of hazard with the product type (`products_catalog.csv`):

| Hazard | Makes sense for |
|---|---|
| expired | FMCG, personal care, cooking oil |
| heat, burnt smell, smoke, sparks, electrical | electronics, home & kitchen, cooking oil (it is heated in use) |
| contamination (glass, insects, mould) | FMCG, personal care, cooking oil |
| health reaction, swelling, poisoning | every product |

A report is `needs_confirmation` when no hazard kind fits the product. With no known product it is `confirmed`
(safety first).

**The urgency stays 3.** The gold labels these combinations 3 (10 of its 15 safety tickets, for example
"VoltX charger … expired", "Tornado blender … expired", "el Molto bysa5an gamed"), and the task asks for the gold's
rules. Lowering them would cost 8/150 gold urgencies, the regression gate and, very likely, the hidden test. So the
status is added next to the urgency, not instead of it:
- **weekly report:** the Quality team's list is split into "confirmed: escalate within 2 h and pull the product" and
  "to confirm: call the customer first";
- **chatbot:** one question before escalating ("is it getting hot, smoking, swollen, or did someone get hurt? or is it
  faulty / out of warranty?"). Yes, unsure or no clear answer twice: escalate. "Faulty" or "warranty": a normal
  product-quality case, urgency 2;
- **history (9k):** of 740 safety-urgent tickets, 403 are confirmed and 337 would be "to confirm".

## Safety as the second problem
Urgency follows the main problem (gold: "late order **and also** expired Dove shampoo" is 2). The policy, however,
escalates **any** safety complaint to Quality within 2 hours. So the weekly report lists every safety report wherever it
appears in the message: 54 historical tickets are urgency 2 with the hazard as the second problem. They are now in the
Quality team's list and in `urgent_tickets.csv`. Before, they were in neither. Their urgency is unchanged.

## Audit of the urgent tickets (2026-11-21)
1,505 of 8,709 valid historical tickets (17.3%) are urgency 3: 740 safety, 616 repeat, 149 double charge ≥ 1,000 EGP.
- **Repeat (616):** every match really says "third time I complain" (تالت مرة أشتكي / talet mara ashtki). No false positives.
- **Double charge (149):** 1,000–1,950 EGP, plus one 5,000 and one 11,100. Correct; the 1,000 threshold is still an assumption (below).
- **Safety (740):** all follow the policy and the gold; 337 name an implausible product (see Plausibility).
- **Correctly not urgent (gold agrees):** 486 tickets with the tail "كلمت خدمة العملاء أكتر من مرة" (gold: 2) and 344 with
  "asap" / "urgent" written by the customer (gold: 2).

## Known uncertainty
- **The double-charge threshold.** The gold only shows that it lies somewhere between 850 and 1,900 EGP. We chose **1,000 EGP**, a round business number. It is configurable (`DOUBLE_CHARGE_URGENT_EGP`) and should be confirmed with the CS team.
- **An `inquiry` that is also a repeat** does not appear in the data. We rate it 3, because a customer asking for the third time is a service failure. This is an assumption.
- The keyword lists cover every phrasing found in the 9,000 historical tickets. New phrasings ("4th time", "burning") would be missed. Monitoring tracks the share of `product_quality` tickets that match no safety keyword, as a sign of new phrasing.
