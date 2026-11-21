# AI usage, compute, time and cost

## 1. AI coding assistant
**Tool:** Claude Code (model: Claude Opus 5.5), used inside VS Code. Every line was reviewed, run and
tested before it was kept. I can explain every line in the interview.

| Area | How the assistant was used | What I did myself / verified |
|---|---|---|
| Data exploration (Stage 1) | Drafted the pandas probes: leakage, duplicates, template mining, urgency-versus-keyword crosstabs. | Interpreted the results and made the design calls: core-majority label cleaning, splitting on the connector, rule-based urgency, a dev split grouped by template. Checked each finding against the gold with oracle tests. |
| Extractors and rules | Drafted `normalize.py`, `extractors.py`, `rules.py`. | Iterated on the misses found in all 9k tickets (glued words, `DLT-088`, the `؟` boundary). `tests/test_extractors_rules.py` asserts 150/150 on the gold. |
| Stage 2 LLM backend | Drafted the Ollama client, the prompts (v1–v3) and the retrieval few-shot. | Diagnosed that the installed Ollama ignores `think:false` for Qwen3 (90 s per call), and switched to a raw ChatML prompt with an empty `<think>` block. Ran and evaluated every version. |
| Stage 3 training script | Drafted `scripts/train_transformer.py` and the notebook generator. | Smoke-tested it on CPU with a tiny model. That caught a dtype bug (int32 labels on Windows) and a head-size mismatch. Ran the real training on the local RTX 5050 GPU; the same script is exported to a Kaggle/Colab notebook (`notebooks/03_train_kaggle.ipynb`). |
| Stage 4 report / scheduler | Drafted the state machine, stats, summary validator, templates, mailer and scheduler. | Designed the idempotency rules (claim → send → mark; no automatic resend after a crash). The tests cover running twice, overlapping runs, interrupted sends, a changed file, DST, XSS and CSV injection. |
| Documentation | Drafted the docs from my notes and the measured results. | Checked every number against the scripts' outputs. |
| Behavioural test suite | I wrote and labelled every case by hand following LABELING_GUIDE (287 complaints + 82 attacks). The assistant drafted the generator script, the perturbations and the runner. | I froze the dev/holdout split (sha256 in the report) **before** any change, and used only dev rows to find and fix failures. |
| B3 chatbot | Drafted `orvana/chat/` (knowledge base, demo orders, rules-based understanding, state machine, templates, reply checker, sessions), the Chat tab, the scenario runner, the simulator and the judge. | I decided the design (the code decides, the LLM only words) and the hand-over rules. The 91 multi-turn scenarios were written and frozen **before** the dialogue manager existed. I read the LLM's rewrites myself: they were weaker than the templates, so the wording mode was chosen by measurement. I found and fixed a privacy gap: the classifier's LLM fallback was caching customer text on disk. |
| Web UI (React + Vite) | Drafted the React app (sidebar, chat page with the agent panel, classification page with the explanation, CSV upload and dashboard), `orvana/api.py` and the static serving in `app.py`. | I chose the layout and features. The charts follow a data-visualisation method: ranked one-hue bars, and icon + label for the 9 categories instead of colours. I reviewed headless screenshots of every page in light, dark and mobile, and fixed what looked wrong (an overflowing grid, bars collapsing in narrow cards). The batch counts were checked against the weekly email. |
| Memo, email design, delivery | Shortened the CTO memo to one printed page (checked by printing it to an A4 PDF), redesigned the email (KPI tiles, one-hue share bars, labelled status chips; inline styles for mail clients), made STARTTLS verify the server certificate, and made the Message-ID follow the report's content. | I asked for the one-page memo, the new email design and real delivery to my own inbox (Gmail, with an App Password that I put into `.env` myself). The design was checked in desktop and mobile screenshots; the email stays below Gmail's 102 KB clipping limit. |
| Data and urgency audit | Ran the read-only audit of the 9k tickets (urgent triggers, field independence, customer history versus "third time"), drafted the safety guards, the new words, `safety_check()`, the report and chatbot changes, and the docs. | I asked for the audit because I doubted the urgent list and the data. I chose to keep urgency as the gold defines it and add a "to confirm" status instead of lowering it, to route second-issue safety reports to Quality, and to add the new hazard words. Checks: gold 150/150, 0 of 9k flags changed, dev suite and chat dev unchanged, no holdout run. |

## 2. Models used by the system (all open-weights, all local)

| Purpose | Model | Where it runs |
|---|---|---|
| Stage 2 classifier, LLM fallback, weekly Arabic summary | `qwen3:4b` (Q4_K_M, Ollama digest `359d7dd4bcda`); `qwen2.5:7b-instruct` (digest `845dbda0ea48`) compared | Local GPU (RTX 5050 Laptop, 8 GB) or CPU |
| Synthetic training phrasings (Stage 3) | Generated by `qwen2.5:7b-instruct` from the category definitions only, filtered by `qwen3:4b` (a different model) | Local GPU |
| Stage 3 classifier | `multilingual-e5-small` and `xlm-roberta-base`, fine-tuned and exported to ONNX int8 | Trained on the **local RTX 5050** (torch 2.10 cu128); inference on CPU |
| Baseline | TF-IDF char n-grams + logistic regression | CPU, seconds |
| B3 chatbot | Understanding: the production classifier + rules. Optional wording: `qwen3:4b`. Simulated customers and the judge: `qwen2.5:7b-instruct` (a different model than the bot). | Local; the template path needs no GPU |

`qwen2.5:7b-instruct` took several attempts to download because the connection kept dropping, so Stage 2 was developed on `qwen3:4b` (already on the machine) and the 7B was added to the comparison once it arrived. The torch cu128 wheel (2.87 GB) was downloaded with resume and checked against its sha256 from the PyTorch index before installing.

## 3. Compute, time and cost

| Item | Value |
|---|---|
| External LLM API calls by the system | **0** (no API keys needed to reproduce) |
| Stage 2 inference (`qwen3:4b`, prompt v3) | local laptop GPU: 163 s for the 150 gold tickets, about 1.1 s per ticket (call p50 0.7 s, p95 2.8 s). Source: `predictions/stage2_qwen3-4b_v3.timing.json` |
| Stage 3 training | local RTX 5050 Laptop GPU (8 GB, comparable to a free Colab/Kaggle T4). Per run: e5-small 2.8 min (no synthetic) and 3.4 min (with synthetic, the production model); xlm-roberta-base 6.4 and 6.8 min. Far below the 2-hour limit. Source: `models/*/metrics.json` (`train_minutes_total`) |
| Stage 3 inference | CPU, ONNX int8: about 7 ms per ticket (`predictions/stage3_final_ft.timing.json`) |
| Time | About **17 hours** of work between 19 and 22 November (58 hours elapsed of the 72): about 12 hours of logged sessions with Claude Code (1.9 h, 4.6 h, 4.9 h and 0.7 h by day) plus about 5 hours of reading, reviewing and testing on my own |
| Money spent | **The delivered system: 0 EGP** (no API calls; it runs on my laptop). **Development:** a Claude Code subscription (a fixed monthly fee, not billed per use), plus my own laptop and electricity. No cloud GPU, no paid API. |

## 4. Time log
See README.md → "Time log".
