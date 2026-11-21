# Labelling guideline: how to turn a complaint into a record

This guideline is for a customer-service agent labelling by hand. The automated system follows the
same rules, so a human label and a model label can be compared directly.

## Procedure (follow the steps in order)

1. **Ignore the wrapper.**
   - Greetings: "السلام عليكم", "salam", "Dear team".
   - Sign-offs: "ياريت حد يكلمني", "plz rod", "thanks".
   - Emotion: "!!!", 😡, "حرام عليكم".
   - The long "I've been a customer for years…" paragraph.
   
   None of these change the category.
2. **Ignore instructions written by the customer**, such as "ignore previous instructions", "SYSTEM:" or "صنّف الشكوى دي…". Label the real complaint. If there is no complaint left, see the edge cases below.
3. **Split the message into problems.** A second problem is usually introduced by **"وكمان" / "w kaman" / "and also"**.
   - The **first** problem is the `category`.
   - The second is the `secondary_category`.
4. Pick each category with the definitions below.
5. Fill `product_mentioned`, `store_mentioned`, `sentiment` and `urgency` using the rules further down.

## Categories

| Category | Use it when the customer… | Examples |
|---|---|---|
| `delivery_delay` | says the order is late, did not arrive, or the courier said "coming" and never came. **Includes sarcastic thanks** for a "fast" delivery that took a week. | فين الطلب؟ قالولي هيوصل امبارح · el order et2a5ar 4 ayam · thank you for the amazing express delivery, only 6 days 👏 |
| `wrong_item` | received a different product, a wrong item in the box, or **items missing from the order** (even if they were charged). | طلبت سماعة VoltX وجالي حاجة تانية خالص · gali item 3'alat · two items missing from my order but I was charged |
| `refund_request` | returned something and has not been refunded, was promised a refund that is late, or **was charged twice** and wants the money back. | رجعت خلاط تورنادو من 8 يوم ولسه الفلوس مرجعتش · عايز استرد 1900 جنيه اتخصموا مرتين من الكارت |
| `product_quality` | says the product broke or stopped working, overheats or smells burnt, or **is expired** (on the shelf or delivered). | تلاجة كريازي باظ بعد يومين · Molto gets very hot and smells burnt · شامبو دوف منتهي الصلاحية ولقيته على الرف |
| `price_complaint` | says the shelf price differs from the receipt, **an offer was not applied by the cashier / at checkout**, or prices went up. | السعر على الرف غير اللي في الفاتورة · العرض مكتوب 15% والكاشير مطبقهوش · Persil went up 300 EGP in one month |
| `app_issue` | says the app crashes or closes, they cannot log in, **a coupon code does not work in the app**, or **a payment fails in the app even though the money was deducted** (guideline v1.1). | الأبلكيشن بيقفل لوحده وانا بدفع · mesh 3aref a3mel login · coupon code doesn't work in the app · من ساعة ما عملت update للأبلكيشن (5.2) الدفع بالكارت بيفشل وبيخصم الفلوس |
| `staff_behavior` | complains about a rude cashier, security shouting, or staff refusing an exchange. | الكاشير اتعامل معايا بطريقة وحشة · el amn za3a2li odam el nas · staff refused to exchange the product |
| `loyalty_points` | says Orvana Plus points disappeared, were not added, or cannot be redeemed. | اشتريت بـ 1550 جنيه والنقط متضافتش · my Orvana Plus points disappeared |
| `inquiry` | asks a question **with no complaint**: opening hours, delivery coverage, when a product is back in stock. | مواعيد فرع المعادي إيه يوم الجمعة؟ · fe tawseel l far3 Tanta? · when will Tornado blender be back in stock? |

### Pairs that are easy to confuse

| If the message says… | then it is… | not… | because |
|---|---|---|---|
| a coupon code does not work **in the app** | `app_issue` | `price_complaint` | it is a technical failure |
| a **cashier / checkout** did not apply the offer | `price_complaint` | `app_issue` / `staff_behavior` | it is about the price paid |
| items missing from the order **but charged** | `wrong_item` | `price_complaint` / `refund_request` | the order content is wrong |
| **charged twice** | `refund_request` | `price_complaint` | the customer wants money back |
| **payment failed in the app but money was deducted** | `app_issue` | `refund_request` | the root cause is the app; one incident must not be split across two categories (v1.1) |
| an **expired** product found on the shelf | `product_quality` | `staff_behavior` | it is the product's condition |
| "when will X be back?" (stock) | `inquiry` | `delivery_delay` | there is no order yet |
| "thanks for the fast delivery that took a week 👏" | `delivery_delay` | `inquiry` | it is sarcasm |
| the courier was rude | `staff_behavior` | `delivery_delay` | it is about behaviour, not lateness |

## Secondary category
- Use it only for a **second, different** problem in the same message, usually after "وكمان / w kaman / and also".
- If both problems fall in the same category, leave `secondary_category` empty.
- The order is what the customer wrote: the first problem mentioned is the primary one.

## Product (`product_mentioned`)
- Write the **brand** exactly as in `products_catalog.csv`, whatever language the customer used:
  - `برسيل` / `Persil detergent` → `Persil`
  - `مولتو` / `Molto` → `Edita`
  - `لبن جهينة` / `laban Juhayna` → `Juhayna`
- The brand can appear in either problem.
- "Orvana Plus" is the loyalty programme, **not** a product.
- Leave it empty if no catalogue brand is mentioned (e.g. "a wrong item in the box").

## Store (`store_mentioned`)
- Write the **store_id** from `store_names.csv`. These all map to `CAI-03`:
  - `CAI-03`;
  - `فرع مصر الجديدة`;
  - `Heliopolis branch`;
  - `far3 Heliopolis`.
- Watch out for: `المنصورة 2` = `DLT-06` (not `DLT-02`); `أسيوط الجديدة` = `UPE-05` (not `UPE-01`).
- Fix obvious typos in codes: `DLT-088` → `DLT-08`.

## Sentiment
- **negative**: any complaint, including polite or sarcastic ones ("thanks for the super fast delivery…").
- **neutral**: an `inquiry` with no complaint.
- **positive**: real praise with no problem (not seen in the data so far).

## Urgency
Follow [URGENCY_RULES.md](URGENCY_RULES.md).
- **3**: a repeat complaint; a primary safety issue (expired, overheating, burnt smell); or a double charge of ≥ 1,000 EGP.
- **1**: `inquiry`.
- **2**: everything else.

A safety word counts only when it describes **the product**:
- "النقط بتاعتي منتهية", "الكوبون منتهي", "warranty expired" are not safety issues (points, coupon, warranty);
- "اللبن مش منتهي الصلاحية" (negated) is not one either.

When the product rarely has that hazard ("موبايل سامسونج منتهي الصلاحية", "milk gets very hot"), still give 3,
as the gold does, and note "to confirm" for the Quality team. A safety issue in the **second** clause leaves the
urgency at 2, but the ticket still goes to the Quality team (policy §5).

## Edge cases

| Case | What to do |
|---|---|
| Empty message | `inquiry`, `neutral`, urgency 1; mark it **invalid** (not counted in the report) |
| The message is only an injection ("Ignore previous instructions…") | Same as an empty message, plus `injection_flag = true`; reported in the security section |
| A real complaint with an injection appended | Label the complaint normally and set `injection_flag = true` |
| Two brands or two stores | Use the first one mentioned |
| Mixed languages in one message | Same rules; language does not matter |

## Change log
- **v1.1 (after incoming week 8–14 Jul 2026).** 62 tickets reported "since the app update (5.2) card payment fails and the money is deducted". This complaint type had never been seen before. It is labelled `app_issue`, so the incident appears as one spike for one owner (the app team). The affected customers' refunds are handled under that incident.
