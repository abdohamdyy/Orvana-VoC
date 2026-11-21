# Memo: what to deploy for the Voice-of-the-Customer system

**To:** CTO · **From:** AI Engineering · **Re:** complaint classification and the weekly management report

## Recommendation
Deploy the **cascade** on **one CPU server**, with no GPU and no external API: clean-up and prompt-injection removal → a fine-tuned **multilingual-e5-small** classifier (~4 ms per ticket) → rules for brand, store, urgency and sentiment → a local LLM (**qwen3:4b**) only when two cheap models disagree. The same LLM writes the weekly Arabic summary, and every number in it is checked against the data. The customer chatbot is an optional phase 2: same server, no extra cost.

## Why (measured on a frozen holdout that nothing was tuned on)
| Option | Quality [95% CI] | Per ticket | If the LLM is down |
|---|---|---|---|
| LLM only | 0.960 [0.932, 0.984] | ~6 s | stops |
| Fine-tuned only | 0.935 [0.902, 0.966] | ~4 ms | works |
| **Cascade** | **0.959 [0.929, 0.983]** | ~4 ms; LLM for ~1% of tickets | **works** |

The cascade matches the LLM's quality with about 1% of its calls.

## Monthly cost (400 complaints a week)
**About 1,600 EGP a month on-premises** (one 8-core / 32 GB server written off over 3 years, plus power), or about 7,700 EGP in the cloud. On top of that, a CS agent spends about 4 hours a month reviewing tickets. External AI API: **0**. The same server can handle 10× the volume.

## Risks and how they are handled
- **New kinds of complaint:** the week-2 app-update bug (62 tickets) was flagged as a new issue at the top of the email.
- **Prompt injection:** removed before classification; 95.7% of attacks leave the label unchanged. The summary LLM never sees customer text.
- **A wrong number in the email:** every number must exist in the data and sit next to the right category, otherwise a fixed template is sent.
- **Safety:** any report of an expired, overheating or burning product goes to Quality within 2 hours, even when it is the second problem in the message.
  - The word must describe the product ("my points expired" is not a safety case).
  - A hazard the product rarely has (an "expired" phone) stays urgency 3 but is marked "to confirm".
- **Operations:** no double sends; a crashed send raises an alert. Credentials live only in `.env`.
- **Privacy:** data stays on our server; phone numbers, emails and card numbers are masked.
- **Business conventions to sign off:** the 1,000 EGP double-charge threshold, and "payment failed, money taken" → app_issue.

## When the model needs updating (shown in every weekly email)
| Signal | Threshold | Action |
|---|---|---|
| Tickets sent to the LLM | > 35% for 2 weeks | Label 200 tickets, retrain (~4 min) |
| Category mix vs 13 weeks (PSI) | > 0.2, no known incident | Check for drift |
| Unknown store or brand | > 3 a week | Add an alias; no retraining |
| Weekly review of 50 tickets | < 90% correct | Retrain |

A new model ships only if it keeps the gold at ≥ 0.999, does no worse on a fresh holdout, and passes all tests. Details: [RISKS.md](RISKS.md).
