# Live test runbook (interview)

Goal: score a new, unseen CSV in under a minute, offline, without any API key.

## 0. Before the call
```bash
.venv/Scripts/activate                      # Linux/Mac: source .venv/bin/activate
pytest -q                                   # all green
ollama list                                 # qwen3:4b present (only needed for the LLM fallback)
```

## 1. Score their file
The input needs only `ticket_id` and `message_text`; any other columns are ignored.
```bash
# production system: fine-tuned model on CPU + rules, LLM only for uncertain segments
python predict.py their_file.csv --backend cascade -o predictions/live.csv

# fully offline / no Ollama: fine-tuned model only (CPU, ~ms per ticket)
python predict.py their_file.csv --backend ft -o predictions/live_ft.csv

# if they bring labels in evaluate.py format
python evaluate.py predictions/live.csv their_file.csv
python scripts/error_analysis.py predictions/live.csv --gold their_file.csv
```
Outputs:
- `predictions/live.csv` (evaluate.py format);
- `.records.jsonl` (full records with `is_valid`, `injection_flag`, `confidence`, `model_version`);
- `.timing.json`.

**Or in the UI:** `python app.py` → http://127.0.0.1:8000 → **Classification → CSV upload**.
- Drop their file to get the dashboard (categories, urgency, top branches and brands) and a filterable table.
- **Download predictions** gives the same evaluate.py format.
- The file is processed in memory; nothing is stored.

## 2. Show one complaint end to end
`python app.py` → http://127.0.0.1:8000 → **Classification → One complaint**: the record, **why** (the split, each part's category and confidence, the rules that fired, any injection removed), and the suggested reply.
The **Customer chat** page shows the chatbot, with the agent panel listing what the code decided.

## 3. If they try attacks live
- An injection inside a complaint is cut before classification. `injection_flag=true` in the record.
- A pure attack with no complaint gives `is_valid=false`. It is excluded from the report.
- The summary LLM never sees customer text. Replies are checked for cash promises, invented numbers and missing safety escalation.

## 4. If something looks wrong
| Symptom | Check |
|---|---|
| Brand or store missing | `python -c "from orvana.extractors import find_store, find_product; print(find_store('...'), find_product('...'))"`. A new spelling is one line in `data/aliases.csv` (no code change). |
| Category wrong | `records.jsonl` → `confidence`. Low confidence means the segment went to the LLM in the cascade. |
| Ollama not running | `--backend ft` works without it; the weekly summary falls back to the template. |
