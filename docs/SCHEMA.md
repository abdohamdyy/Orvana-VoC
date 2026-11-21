# Output schema: one record per complaint

Defined in code as `orvana/schema.py::TicketRecord` (Pydantic). Every output is validated against
it. The JSON Schema is in [schema.json](schema.json).

## Scored fields (evaluate.py format)

| Field | Type | Allowed values | "none" |
|---|---|---|---|
| `ticket_id` | string | as given in the input | never |
| `category` | enum | `delivery_delay`, `wrong_item`, `refund_request`, `product_quality`, `price_complaint`, `app_issue`, `staff_behavior`, `loyalty_points`, `inquiry` | never: always exactly one |
| `secondary_category` | enum or empty | same 9 values; **must differ from `category`** | empty when the message has a single problem |
| `product_mentioned` | string or empty | a **brand** exactly as in `products_catalog.csv`: `VoltX`, `Tornado`, `Juhayna`, `Samsung`, `Xiaomi`, `Persil`, `Orvana Select`, `Fresh`, `Dove`, `Kiriazi`, `Edita` | empty when no catalogue brand is mentioned |
| `store_mentioned` | string or empty | a **store_id** exactly as in `store_names.csv` (e.g. `CAI-03`) | empty when no store is mentioned |
| `sentiment` | enum | `negative`, `neutral`, `positive` | never |
| `urgency` | int | `1`, `2`, `3` (3 = most urgent) | never |

In the CSV an empty cell means "none" (this is how `evaluate.py` reads it). In JSON, "none" is `null`.

## Operational fields (not scored; used by the weekly report and by monitoring)

| Field | Type | Meaning |
|---|---|---|
| `is_valid` | bool | `false` for an empty message or a pure prompt injection. These tickets are kept for audit but **excluded from the report's statistics**. They get `inquiry` / `neutral` / urgency `1`. |
| `injection_flag` | bool | An injection attempt was found and removed before classification. |
| `confidence` | float 0–1 | The classifier's probability for the least certain segment. LLM backends report 1.0. Used to route uncertain cases to the LLM and to monitor drift. |
| `model_version` | string | The backend and version that produced the record (e.g. `ollama:qwen2.5:7b-instruct@845dbda0ea48:v3`), so every number in a report can be traced. |

## Mapping rules (what the extractors guarantee)
- **Product → brand.** Arabic, English and Franco names all map to the brand: `مولتو`/`Molto` → `Edita`, `لبن جهينة` → `Juhayna`, `تلاجة كريازي` → `Kiriazi`. "Orvana Plus" is the loyalty programme, not a product.
- **Store → code.** The code (`CAI-03`, also typos such as `DLT-088`), the Arabic name (`فرع مصر الجديدة`), the English name (`Heliopolis branch`) and Franco (`far3 Heliopolis`) all map to the same `store_id`. When two names overlap, the longest one wins (`المنصورة 2` → `DLT-06`, not `DLT-02`).
- **More than one brand or store** (not seen in the data so far): the first one mentioned is used.
