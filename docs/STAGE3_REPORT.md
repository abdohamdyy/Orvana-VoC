# Stage 3: our own model, and was fine-tuning worth it?

## What we trained
| | |
|---|---|
| Task | Classify one complaint **segment** into the 9 categories. Brands, stores, urgency and sentiment stay rule-based (they were already at 100% on the gold). |
| Data | 1. **Cleaned historical segments**: the staff labels were replaced by their template's majority label, gold removed, about 9.9k segments, 99.5% labelled.<br>2. **Synthetic new phrasings** (`scripts/gen_synthetic.py`): `qwen2.5:7b` wrote messages from the category definitions only, in Egyptian Arabic / MSA / English / mixed / voice-note style. `qwen3:4b` (a different model) kept a message only when it agreed on the label. **717 kept** of 1,066 (76.7% agreement). |
| Quality filter found by reading the data | The generator's **Franco was gibberish** ("3ny 5lm 7la 2mlik…") and some messages drifted into Chinese. Franco synthetic data was dropped entirely (194 messages) and foreign-script messages removed. The agreement rate alone would not have shown this. |
| Augmentation | Typos, glued words, punctuation removal, greeting/sign-off padding, lower-casing. These break the templates' surface cues: in the history every inquiry ends with "؟". |
| Selection | Internal dev = **held-out templates** (never seen in training), plus 15% held-out synthetic data. Never the gold, never the test suite. |
| Hardware | **Local RTX 5050 Laptop, 8 GB** (torch 2.10 cu128). The same script runs on a free Colab/Kaggle T4 (`notebooks/03_train_kaggle.ipynb`). |
| Export | ONNX int8 for **CPU** inference. Agreement with the torch model on 500 texts: **1.0000** for every model. |

## Ablation: model size × synthetic data
Same data, same augmentation, same preprocessing; only the base model and the synthetic data change.

| Model | Synthetic | Internal dev F1 (held-out templates) | Gold overall | **Suite-dev overall** [95% CI] | CPU ms/ticket | Size | Train time |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | no | 1.000 | 1.000 | 0.924 [0.894, 0.949] | 2.6 | 118 MB | 2.8 min |
| **multilingual-e5-small** | **yes** | 0.992 | **1.000** | **0.946** [0.920, 0.967] | 2.3 | 118 MB | 3.4 min |
| xlm-roberta-base | no | 0.951 | 1.000 | 0.938 [0.913, 0.958] | 6.7 | 279 MB | 6.4 min |
| xlm-roberta-base | yes | 0.951 | 1.000 | 0.956 [0.935, 0.974] | 6.8 | 279 MB | 6.8 min |
| *TF-IDF baseline (reference)* | – | 0.922 | 1.000 | 0.913 [0.881, 0.941] | 0.9 | – | seconds |

What we learn from it:
- **Synthetic phrasings help both models by about +2 points** on new text, at no inference cost. Franco went up too (e5: 0.82 → 0.90) even though no Franco synthetic data was used.
- **The larger model is not better where it matters.** XLM-R base memorises the training data fastest (loss 0.03 at epoch 2) but generalises *worse* to held-out templates (0.951 vs 0.992). It costs 2.5× more CPU and size. e5 was pre-trained contrastively for sentence meaning, which is a plausible reason it transfers better.
- **Pre-registered choice:** score = mean(suite-dev, internal-dev), gold ≥ 0.999, then the cheapest model within 0.01 of the best → **e5-small + synthetic** (score 0.969).
  - *Transparency note:* the first run of `compare_models.py` broke the tie between the two e5 models (identical architecture and cost) by row order and picked the lower-scoring one.
  - The tie-break "equal cost → higher score" was then written down explicitly. It is the only change to the rule.

## Production design: a cascade, and how it routes
Routing analysis on the suite **dev** split (`scripts/routing_analysis.py`), primary-category accuracy, n = 161:

| Rule (when to ask the LLM) | Sent to LLM | Accuracy | Fine-tuned errors caught |
|---|---|---|---|
| never (fine-tuned only) | 0% | 0.882 | 0 / 19 |
| confidence < 0.95 (after temperature scaling) | 7.5% | 0.907 | 7 / 19 |
| similarity to training data < q20 | 19.9% | 0.894 | 3 / 19 |
| **fine-tuned ≠ TF-IDF** | **25.5%** | **0.950** | **17 / 19** |
| always (LLM only) | 100% | 0.944 | – |

- **The fine-tuned model's confidence is not a usable routing signal on new text.** It is over-confident exactly where it is wrong. Temperature scaling fitted on in-distribution data made it worse (T = 0.5 sharpens the probabilities).
- **Disagreement between two different cheap models is the signal that works.** It catches 17 of 19 errors while sending a quarter of the segments to the LLM. The cascade is **more accurate than the LLM alone** and 4× cheaper in LLM calls.
- The rule is saved in `models/stage3/routing.json` and read by every entry point (predict, weekly job, UI).

## Which LLM as the fallback? (suite dev, primary category, n = 161)
| Fallback | LLM alone | Inside the cascade | Latency per call (GPU) |
|---|---|---|---|
| **qwen3:4b** | **0.944** | **0.950** | ~0.7 s |
| qwen2.5:7b-instruct | 0.938 | 0.938 | ~0.9 s |

The bigger model is not better here, so the 4B stays. The same happened for the weekly Arabic summary:
the 7B was more fluent but **invented a detail** ("for an hour after the update"), which no number check can
catch, and left out parts of the requested content. The 4B was plainer but faithful and complete.

## Final comparison: the frozen HOLDOUT (evaluated once, after every choice above)
| System | Gold overall | **Holdout overall** [95% CI] | Security: label unaffected | Per ticket (CPU server) | Without the LLM |
|---|---|---|---|---|---|
| Stage 2: LLM v3 (`qwen3:4b`) | 0.999 | 0.960 [0.932, 0.984] | 0.957 | ~6 s (measured: 5.0 s per LLM call on the Ryzen 7 CPU, 1.15 calls per ticket) | stops |
| Stage 3: fine-tuned e5-small | 1.000 | 0.935 [0.902, 0.966] | 0.957 | ~4 ms | works |
| **Production: cascade** | **1.000** | **0.959** [0.929, 0.983] | 0.957 | ~4 ms + LLM for the disagreements | falls back to the fine-tuned model |
| *TF-IDF baseline* | 1.000 | 0.924 [0.886, 0.956] | 0.913 | 0.4 ms | works |

How often the cascade actually calls the LLM:
- **Real traffic** (both incoming weeks): **7 of 723 segments (1%)**.
- **The deliberately unusual wording of the test suite:** ~25% of segments.

Holdout scores are ~1.5 points below the dev scores for the cascade (0.974 → 0.959), the expected price of
having made choices on dev; the holdout numbers are the ones to quote.

## Was fine-tuning worth it?
**Yes, but not for the reason the gold set would suggest.**
- On the gold set nothing beats TF-IDF: it is already at 1.000, because the gold reuses the historical templates. Judged only on the gold, fine-tuning would look pointless.
- On genuinely new text (other dialects, phrasings, noise, attacks), the fine-tuned model is better than TF-IDF:
  - dev 0.946 vs 0.913;
  - holdout 0.935 vs 0.924.
- It runs on a CPU in about 4 ms, more than **1,000× cheaper than the LLM** on the same CPU.
- It trains in **3.4 minutes**.
- On its own it is not as good as the LLM on unusual wording (holdout 0.935 vs 0.960).
- Its real value is that it makes the cascade possible:
  - **the LLM's quality** (holdout 0.959 vs 0.960);
  - **the fine-tuned model's cost** on real traffic: the LLM is consulted for ~1% of segments;
  - **the fine-tuned model's resilience**: classification keeps working if the LLM is down.
- What did **not** pay off: the larger XLM-R model, the 7B LLM, and confidence-based routing.
