# Orvana: Voice of the Customer

This system turns any customer complaint (Egyptian Arabic, MSA, Franco-Arabic or English) into a structured
record, and turns each week's complaints into an email report for management. Everything runs on
Orvana's own machines with open-weights models: **no external AI API, no API keys**.

```
complaint ─► clean (Unicode tricks, prompt injection, PII) ─► split on "وكمان / w kaman / and also / وبرضو / plus …"
          ─► fine-tuned e5-small on CPU ──(disagrees with TF-IDF?)──► local LLM qwen3:4b decides
          ─► dictionaries: brand + store   ─► rules: urgency + sentiment   ─► validated record
weekly file ─► records (idempotent) ─► pandas facts + new-incident detection ─► Arabic LLM summary (numbers verified) ─► email
chat (B3)  ─► same understanding ─► state machine decides (policy facts, voucher/refund in code, escalation, hand-over) ─► reply
```

## Results

**Dev gold set** (150 tickets, `evaluate.py`): every final system scores 0.999–1.000. The gold set reuses
the historical templates, so it cannot tell the systems apart.

**Behavioural test suite** (287 hand-labelled complaints in 6 language styles + 82 attacks + 16 perturbation types).
The holdout was frozen before any change and **evaluated once** ([ROBUSTNESS_REPORT](docs/ROBUSTNESS_REPORT.md)):

| System | Gold | **Holdout** [95% CI] | Attack: label unaffected | Per ticket on a CPU server |
|---|---|---|---|---|
| Baseline: TF-IDF + logistic regression | 1.000 | 0.924 [0.886, 0.956] | 0.913 | 0.4 ms |
| Stage 2: local LLM `qwen3:4b` (rules + retrieved few-shot) | 0.999 | 0.960 [0.932, 0.984] | 0.957 | ~6 s |
| Stage 3: fine-tuned `multilingual-e5-small` (ONNX int8) | 1.000 | 0.935 [0.902, 0.966] | 0.957 | ~4 ms |
| **Production: cascade (Stage 3 + LLM only on disagreement)** | **1.000** | **0.959** [0.929, 0.983] | 0.957 | ~4 ms; the LLM sees ~1% of real tickets |

**B3 customer chatbot** (91 multi-turn scenarios frozen before the code, holdout evaluated once; [CHATBOT](docs/CHATBOT.md)):
- **97.2%** of holdout conversations fully correct with the production classifier (91.7% with no LLM at all);
- 0 forbidden-content violations;
- vouchers, refund times, safety escalation and hand-over 100% correct on the holdout;
- 8–9 ms per turn.

Full stories: [STAGE1_DATA](docs/STAGE1_DATA.md) · [STAGE2_REPORT](docs/STAGE2_REPORT.md) ·
[STAGE3_REPORT](docs/STAGE3_REPORT.md) · [RISKS](docs/RISKS.md) · [CTO_MEMO](docs/CTO_MEMO.md).

## Demo video (4 min 39 s)

**[Watch on Google Drive](DRIVE_LINK)**. It is recorded from the local UI (`python app.py`), with no external API.

| Time | What it shows |
|---|---|
| 0:00 | **Customer chat, Arabic:** "الشاحن صلاحيته انتهت" is escalated as a safety case. The bot creates a ticket and hands over to a colleague. The agent panel shows what the code decided (case, urgency 3, the policy line applied). |
| 0:50 | **Customer chat, English:** a vague message ("where is my laptop?"), then clarifying questions, the product (Samsung), the electronics return policy and a ticket. |
| 2:10 | **Classification of one complaint** with "Why this decision" (cleaned text, split, category, rules) and the suggested reply (B1). Three complaints: "الفشار اتحرق", "بقالي 7 أيام مستني اللعبة" (delivery delay, voucher policy) and "الشاحن فرقع" (safety, urgency 3). |
| 3:35 | **CSV upload** of the dev gold file (150 tickets): dashboard, filterable table, download in `evaluate.py` format. |
| 4:10 | **A real week** (1–7 Jul, 320 tickets, 44 urgent): the same numbers as that week's email. |

The weekly emails themselves are in `outputs/emails/` (open the `.html` in a browser).

## Quick start (CPU only, no GPU needed)

Use **Python 3.11**, the version the code and the Docker image were built and tested with. The local mail sink
`scripts/smtp_sink.py` needs ≤ 3.11.

```bash
git lfs install                                          # once: the 118 MB model is stored with Git LFS
git clone <this repo> && cd <repo>                       # already cloned without LFS? run: git lfs pull
python -m venv .venv && .venv/Scripts/activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                                # 415 tests
python predict.py data/tickets_dev_gold.csv --backend ft -o predictions/stage3_final_ft.csv
python evaluate.py predictions/stage3_final_ft.csv
```

`models/stage3/model.int8.onnx` (118 MB) is above GitHub's 100 MB file limit, so it is stored with **Git LFS**. If
the file is a small text pointer instead of the model, run `git lfs pull`. It can also be rebuilt with Stage 3 below
in ~4 minutes on a GPU. No API key is needed for anything: all models are open-weights and run locally.

## How to run each stage

| Stage | Command | Output |
|---|---|---|
| 1. Data and schema | `python scripts/build_train_set.py`<br>`jupyter nbconvert --execute notebooks/01_eda.ipynb` | `data/processed/`, `outputs/metrics/label_noise.json` |
| 2. LLM | `ollama pull qwen3:4b`<br>`python predict.py data/tickets_dev_gold.csv --backend llm --prompt v3 -o predictions/stage2_qwen3-4b_v3.csv` | predictions + `.timing.json` |
| 3. Fine-tune | `python scripts/gen_synthetic.py` (needs `qwen2.5:7b-instruct` + `qwen3:4b`)<br>`python scripts/train_transformer.py --data data/processed/segments.csv --model intfloat/multilingual-e5-small --extra-data data/processed/synthetic.csv --out models/e5small_syn`<br>`python scripts/compare_models.py`<br>`python scripts/routing_analysis.py --ft-dir models/e5small_syn` | `models/*/metrics.json`, `routing.json` |
| Suite | `python scripts/run_suite.py --backend cascade --split dev` (the holdout only once, at the end) | `outputs/suite/*.json`, `docs/ROBUSTNESS_REPORT.md` |
| 4. Weekly report | `docker compose up -d`, then open http://localhost:8025 (Mailpit)<br>Without Docker: `python scripts/smtp_sink.py` + `python -m orvana.scheduler`<br>One-off: `python scripts/run_weekly.py`<br>Real mailbox: copy `.env.example` to `.env` and set `SMTP_HOST/PORT/USER/PASSWORD`, `SMTP_STARTTLS=true`, `MAIL_TO`. For Gmail: `smtp.gmail.com`, port 587, an App Password. The certificate is verified. | `outputs/emails/*.eml/.html`, `outputs/mailbox/` |
| 5. Production | `python scripts/adversarial_demo.py` · `python scripts/check_gates.py` | [RISKS](docs/RISKS.md), [CTO_MEMO](docs/CTO_MEMO.md) |
| Bonus | **UI:** `python app.py` → http://127.0.0.1:8000 (pages: Customer chat, Classification with CSV upload + dashboard; the built UI is included, no Node needed)<br>`python scripts/eval_replies.py` (B1)<br>`python scripts/run_chat_eval.py --split dev --llm off` · `python scripts/simulate_chat.py` (B3)<br>Change the UI: `cd web && npm install && npm run dev` (proxies `/api` to :8000) · `npm run build` | `outputs/replies/`, `outputs/chat/` |
| Live test | see [LIVE_TEST_RUNBOOK](docs/LIVE_TEST_RUNBOOK.md) | |

Stage 3 was trained on the local RTX 5050 (torch 2.10 cu128, separate `.venv-train`). The same script runs on a
free Colab/Kaggle T4: `notebooks/03_train_kaggle.ipynb`.

## Key decisions and why

1. **Remove the gold before anything else.** All 150 gold tickets are hidden in `tickets.csv`, and 94 more rows share their text after normalisation. They are dropped by id **and** by text.
2. **Clean the labels.** Staff labels are 15% noisy. The majority label of each template core is 100% correct on the gold, and it also labels the 6,000 unlabelled tickets.
3. **Rules where the task is deterministic, models where it is not.**
   - Brand and store come from closed lists: 100% on the gold and on the suite holdout.
   - Urgency follows rules that reproduce 150/150 gold labels.
   - Models only choose the category of each segment.
4. **Do not trust the gold set alone.** It reuses the history's templates; a TF-IDF model scores 1.000 on it. So we wrote a behavioural test suite (other dialects, phrasing, noise, attacks) with a **frozen holdout**, and every rule change also passed regression gates on all 9,000 historical tickets.
5. **Synthetic phrasings + a small model beat a bigger model.**
   - Synthetic data: +2 points on new text.
   - XLM-R base (2.4× bigger) generalised worse than e5-small.
   - The 7B LLM was no better than the 4B.
   - We read the synthetic data and dropped the generator's unusable Franco.
6. **Route on disagreement, not on confidence.** The fine-tuned model is over-confident on new text: confidence-based routing caught 7 of 19 errors, disagreement with TF-IDF caught 17 of 19. The cascade matches the LLM's quality and calls it for ~1% of real tickets.
7. **The report must help management act.**
   - **New-incident detection:** terms never seen in the history, here the app 5.2 payment bug in 62 tickets in week 2.
   - Safety items grouped by product with the 2-hour policy.
   - Week-over-week spikes; history compared by share, because the history is a sample.
8. **Every number is correct.**
   - Numbers are computed by pandas.
   - The LLM summary sees aggregates only, never customer text.
   - Each number must exist in the data **and** sit next to the right category.
   - After 3 failed tries we send a template summary.
9. **Idempotent by construction.**
   - A file is classified once.
   - The email for a week is claimed through a primary key.
   - A crash during sending raises an alert instead of a second email.
10. **The chatbot (B3): the code decides, the LLM only words.**
    - A state machine computes the voucher, the refund time, escalation and hand-over, so a jailbreak cannot change them.
    - An order is shown only to its owner.
    - Questions we have no data for (store hours, prices) are answered "no data, here is a human", never guessed.
    - Templates are the default wording. An LLM rewrite scored no better with a separate judge, took ~4.5 s instead of 7 ms, and once turned an apology into an implied compensation promise.
    - No transcript is stored.

## Documents
Stage 1:
- [STAGE1_DATA](docs/STAGE1_DATA.md)
- [SCHEMA](docs/SCHEMA.md) · [schema.json](docs/schema.json)
- [LABELING_GUIDE](docs/LABELING_GUIDE.md) (v1.1)
- [URGENCY_RULES](docs/URGENCY_RULES.md)

Stage 2: [STAGE2_REPORT](docs/STAGE2_REPORT.md). Stage 3: [STAGE3_REPORT](docs/STAGE3_REPORT.md). Suite: [ROBUSTNESS_REPORT](docs/ROBUSTNESS_REPORT.md).

Stage 5: [RISKS](docs/RISKS.md) · [CTO_MEMO](docs/CTO_MEMO.md) · [LIVE_TEST_RUNBOOK](docs/LIVE_TEST_RUNBOOK.md).

Bonus: [BONUS](docs/BONUS.md) (replies: 100% pass the policy checks; separate-model judge scores policy 3.6, tone 4.0, helpfulness 3.0 out of 5) · B3 chatbot: [CHATBOT](docs/CHATBOT.md).

Also: [AI_USAGE](AI_USAGE.md).

## Repository map
```
orvana/        normalize, extractors, rules, schema, pipeline, retriever, replies,
               backends/ (tfidf, llm_ollama, finetuned, cascade), report/ (state, stats, emerging, monitoring,
               summary_llm, validate_numbers, render, mailer, weekly), eval/ (metrics, perturb, chat_judge), scheduler,
               chat/ (kb, orders, nlu, manager, respond, sessions): the B3 chatbot
predict.py     any CSV -> evaluate.py format (dev, hidden test, live test)
app.py         serves the UI + JSON API (standard library HTTP server); payloads in orvana/api.py
web/           React + Vite UI source (src/) and its committed build (dist/)
scripts/       build_train_set, gen_synthetic, train_tfidf, train_transformer, compare_models, routing_analysis,
               run_suite, check_gates, error_analysis, run_weekly, smtp_sink, adversarial_demo, eval_replies,
               run_chat_eval, simulate_chat, calibrate_chat_threshold, ...
tests/         415 tests + suite/ (base.csv, security.csv, frozen HOLDOUT_SHA256.txt) + chat/ (91 frozen
               multi-turn scenarios, CHAT_HOLDOUT_SHA256.txt) + fixtures/
data/          inputs + processed/ (segments, messages, synthetic) + aliases.csv (extra spellings, no code)
               + demo/orders.json (FICTIONAL orders for the chatbot demo)
models/stage3  production model: tokenizer, labels, calibration.json, routing.json (+ ONNX via LFS)
predictions/   predictions for every solution on the dev gold (+ timing); challenge_*.csv are on our own 47-case
               challenge set: python evaluate.py predictions/challenge_tfidf.csv tests/fixtures/challenge_set.csv
outputs/       emails for both weeks, mailbox copies, suite results, metrics, training logs
docs/          everything above
```

## Time log
About **17 hours** of work between 19 and 22 November (58 hours elapsed of the 72-hour window): about 12 hours
of logged sessions with the coding assistant (19 Nov 1.9 h, 20 Nov 4.6 h, 21 Nov 4.9 h, 22 Nov 0.7 h) plus about
5 hours of reading, reviewing and testing. The blocks, in order:

| Block | What |
|---|---|
| 1 | Brief, data exploration (leakage, noise, templates, urgency rules), plan |
| 2 | Extractors, rules, label cleaning, Stage 1 docs |
| 3 | Stage 2: LLM backend v1–v3; Qwen3 thinking, the Windows `localhost` delay and model downloads |
| 4 | Stage 4: report, idempotent state, scheduler, emerging-issue detection, 180+ tests |
| 5 | Plan revision 4 (self-critique): behavioural suite with a frozen holdout; Phase B hardening behind regression gates |
| 6 | Stage 3: torch cu128 on the local GPU, synthetic data (read and filtered), 4-model ablation, routing analysis |
| 7 | Holdout (once), final emails, CTO memo, reports |
| 8 | B3 chatbot: scenarios frozen first, state machine, retrieval with a concept lexicon, phrasing ablation with a judge, simulated customers, chat holdout (once) |
| 9 | Data and urgency audit: fields are independent in the generated data (STAGE1_DATA 11–15); safety words must describe the product (guards, new words); "to confirm" status for implausible hazards (urgency unchanged, gold 150/150); safety as a second problem reaches the Quality team; both emails rebuilt |
| 10 | Production polish: one-page CTO memo, redesigned email, STARTTLS certificate check, real delivery tested through Gmail, full re-check (tests, gates, demo, every prediction file, notebook, links) |
