# Robustness report: behavioural test suite

**Why this exists.** The dev gold reuses the historical templates, so a model that memorised them
scores ~1.0. The reviewers will test on other data: other dialects, phrasings and attacks. This suite
measures exactly that.

## Method (CheckList-style)
- `tests/suite/base.csv`: 288 hand-written complaints labelled with
  docs/LABELING_GUIDE.md.
  - **Tags:** language (Egyptian Arabic, MSA, Franco, English, mixed, voice-note style) and
    phenomenon (paraphrase, multi-issue, entity spellings, urgency edges, sentiment edges, out of scope, incident, style noise).
  - **Ambiguous rows** (the label depends on a convention we had to guess) are scored separately.
- `tests/suite/security.csv`: prompt-injection, delimiter, role-play, obfuscation, exfiltration, reply/summary attacks,
  HTML/CSV/SQL payloads, PII, very long text.
  - **Primary metric:** *label unaffected* (the attack changes nothing).
  - **Secondary metric:** *flagged* (it is also detected).
- **Perturbations** (`orvana/eval/perturb.py`): 16 label-preserving transforms (typos, glued words,
  tashkeel, zero-width, bidi, Arabic-Indic digits, injection suffix, PII, HTML, padding…). *Invariance* = the answer is unchanged.
- **Protocol against overfitting the suite:**
  - The rows are split **dev 60% / holdout 40%** (stratified by language × phenomenon).
  - Fixes and model choices use **dev only**.
  - The holdout is frozen before any change (sha256 `814f10844500cabb…`) and evaluated **once**, at the end.
  - The suite is never used for training, retrieval or prompts.
  - `run_suite.py` checks that no training text is a near-duplicate (cosine > 0.9) of a suite text.
- **Timing caveat:** `ms/ticket` for LLM-based backends on the dev split is too low, because dev texts were
  evaluated several times and answers came from the LLM cache. Use the holdout timings (evaluated once) and
  the latency measurements in STAGE2/STAGE3_REPORT and CTO_MEMO.
- **Perturbation caveat:** the training augmentation (typos, glued words, punctuation removal, greeting
  padding, lower-casing) overlaps with some perturbation types, so invariance on those types is partly
  *trained*. Tashkeel, zero-width, bidi, Arabic-Indic digits, HTML, PII and injection suffixes are handled by
  normalisation, not by training.

## Effect of the Phase B hardening (TF-IDF backend, dev split)
Values before Phase B come from the first run logs (`run_suite.py --backend tfidf --split dev`, before the
normalisation/extractor changes); after = the table below.

| | Before | After |
|---|---|---|
| Overall | 0.896 | 0.913 |
| Product / store | 0.981 / 0.994 | 1.000 / 1.000 |
| Security: label unaffected | 0.644 | 0.932 |
| Attacks flagged | 0.153 | 0.881 |
| Invariance to tashkeel / zero-width | 0.851 / 0.862 | 1.000 / 1.000 |

Every rule change also had to keep the dev gold at 150/150 and leave all 9,000 historical tickets unchanged
(`scripts/check_gates.py`). Two candidate rules were rejected by that gate: "plus" fired on "Orvana Plus",
and fuzzy store matching turned "maksoura" (broken) into Mansoura.

## Results

### Dev split (used for decisions)

| Backend | n | Overall [95% CI] | Category F1 | Urgency | Product | Store | Security: label unaffected | Attacks flagged | ms/ticket |
|---|---|---|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 161 | **0.974** [0.956, 0.989] | 0.949 | 0.988 | 1.000 | 1.000 | 0.932 | 0.881 | 3.1 |
| `ft_e5small_nosyn` | 161 | **0.924** [0.894, 0.949] | 0.836 | 0.969 | 1.000 | 1.000 | 0.932 | 0.881 | 2.6 |
| `ft_e5small_syn` | 161 | **0.946** [0.920, 0.967] | 0.885 | 0.975 | 1.000 | 1.000 | 0.932 | 0.881 | 2.3 |
| `ft_xlmrb_nosyn` | 161 | **0.938** [0.913, 0.958] | 0.871 | 0.969 | 1.000 | 1.000 | 0.949 | 0.881 | 6.7 |
| `ft_xlmrb_syn` | 161 | **0.956** [0.935, 0.974] | 0.907 | 0.988 | 1.000 | 1.000 | 0.949 | 0.881 | 6.8 |
| `llm_v3_qwen3-4b` | 161 | **0.971** [0.951, 0.986] | 0.944 | 0.981 | 1.000 | 1.000 | 0.932 | 0.881 | 24.5 |
| `tfidf-logreg` | 161 | **0.913** [0.881, 0.941] | 0.827 | 0.944 | 1.000 | 1.000 | 0.932 | 0.881 | 0.4 |

**Overall by language** (n in brackets)

| Backend | egy | en | franco | mixed | msa | voice |
|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 0.97 (53) | 1.00 (40) | 0.91 (27) | 1.00 (13) | 1.00 (19) | 1.00 (9) |
| `ft_e5small_nosyn` | 0.93 (53) | 0.92 (40) | 0.82 (27) | 1.00 (13) | 1.00 (19) | 0.92 (9) |
| `ft_e5small_syn` | 0.95 (53) | 0.90 (40) | 0.90 (27) | 1.00 (13) | 0.98 (19) | 1.00 (9) |
| `ft_xlmrb_nosyn` | 0.94 (53) | 0.91 (40) | 0.90 (27) | 1.00 (13) | 0.98 (19) | 0.92 (9) |
| `ft_xlmrb_syn` | 0.97 (53) | 0.95 (40) | 0.89 (27) | 1.00 (13) | 1.00 (19) | 0.92 (9) |
| `llm_v3_qwen3-4b` | 0.97 (53) | 0.99 (40) | 0.91 (27) | 1.00 (13) | 1.00 (19) | 1.00 (9) |
| `tfidf-logreg` | 0.93 (53) | 0.88 (40) | 0.93 (27) | 0.95 (13) | 0.85 (19) | 0.92 (9) |

**Overall by phenomenon** (n in brackets)

| Backend | entities | incident | multi_issue | out_of_scope | paraphrase | sentiment_edge | style_noise | urgency_edge |
|---|---|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 0.81 (20) | 1.00 (2) | 0.89 (17) | 1.00 (5) | 0.98 (94) | 1.00 (3) | 1.00 (6) | 1.00 (14) |
| `ft_e5small_nosyn` | 0.81 (20) | 1.00 (2) | 0.82 (17) | 0.73 (5) | 0.93 (94) | 1.00 (3) | 0.78 (6) | 1.00 (14) |
| `ft_e5small_syn` | 0.85 (20) | 1.00 (2) | 0.85 (17) | 0.73 (5) | 0.95 (94) | 1.00 (3) | 0.78 (6) | 1.00 (14) |
| `ft_xlmrb_nosyn` | 0.86 (20) | 1.00 (2) | 0.88 (17) | 1.00 (5) | 0.94 (94) | 1.00 (3) | 0.80 (6) | 0.66 (14) |
| `ft_xlmrb_syn` | 0.84 (20) | 1.00 (2) | 0.93 (17) | 1.00 (5) | 0.95 (94) | 1.00 (3) | 1.00 (6) | 0.88 (14) |
| `llm_v3_qwen3-4b` | 0.81 (20) | 1.00 (2) | 0.89 (17) | 1.00 (5) | 0.97 (94) | 1.00 (3) | 1.00 (6) | 1.00 (14) |
| `tfidf-logreg` | 0.84 (20) | 1.00 (2) | 0.80 (17) | 0.51 (5) | 0.93 (94) | 1.00 (3) | 1.00 (6) | 0.69 (14) |

**Invariance under label-preserving perturbations** (share of answers unchanged vs the clean text)

| Backend | arabic_digits | bidi_marks | elongate | emoji_punct | glue_words | html_wrap | injection_suffix | no_punct | padding | pii_suffix | tashkeel | tatweel | typo_double | typo_drop | upper_latin | zero_width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 0.909 | 1.0 | 0.963 | 0.975 | 0.969 | 0.963 | 0.975 | 0.932 | 0.907 | 0.963 | 1.0 | 1.0 | 0.975 | 0.944 | 0.964 | 0.994 |
| `ft_e5small_nosyn` | 0.886 | 0.969 | 0.925 | 0.925 | 0.956 | 0.913 | 0.957 | 0.864 | 0.708 | 0.87 | 0.957 | 0.978 | 0.901 | 0.919 | 0.94 | 0.975 |
| `ft_e5small_syn` | 0.955 | 0.994 | 0.944 | 0.969 | 0.943 | 0.963 | 0.963 | 0.886 | 0.863 | 0.938 | 0.989 | 1.0 | 0.944 | 0.938 | 0.988 | 0.987 |
| `ft_xlmrb_nosyn` | 0.75 | 0.888 | 0.863 | 0.876 | 0.874 | 0.882 | 0.894 | 0.795 | 0.752 | 0.907 | 0.936 | 0.876 | 0.863 | 0.826 | 0.892 | 0.918 |
| `ft_xlmrb_syn` | 0.955 | 0.938 | 0.938 | 0.913 | 0.925 | 0.919 | 0.957 | 0.886 | 0.901 | 0.944 | 0.989 | 0.966 | 0.944 | 0.894 | 0.94 | 0.962 |
| `llm_v3_qwen3-4b` | 0.955 | 1.0 | 0.95 | 0.957 | 0.969 | 0.963 | 0.981 | 0.909 | 0.888 | 0.944 | 1.0 | 1.0 | 0.981 | 0.957 | 0.952 | 1.0 |
| `tfidf-logreg` | 0.909 | 1.0 | 0.95 | 0.981 | 0.975 | 0.957 | 0.969 | 0.955 | 0.882 | 0.957 | 1.0 | 1.0 | 0.95 | 0.901 | 1.0 | 1.0 |

### Holdout split (evaluated once, after all decisions)

| Backend | n | Overall [95% CI] | Category F1 | Urgency | Product | Store | Security: label unaffected | Attacks flagged | ms/ticket |
|---|---|---|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 107 | **0.959** [0.929, 0.983] | 0.917 | 0.981 | 1.000 | 1.000 | 0.957 | 0.913 | 102.6 |
| `ft_e5small_syn` | 107 | **0.935** [0.902, 0.966] | 0.866 | 0.963 | 1.000 | 1.000 | 0.957 | 0.913 | 4.0 |
| `llm_v3_qwen3-4b` | 107 | **0.960** [0.932, 0.984] | 0.920 | 0.981 | 1.000 | 1.000 | 0.957 | 0.913 | 461.6 |
| `tfidf-logreg` | 107 | **0.924** [0.886, 0.956] | 0.844 | 0.963 | 1.000 | 1.000 | 0.913 | 0.913 | 0.4 |

**Overall by language** (n in brackets)

| Backend | egy | en | franco | mixed | msa | voice |
|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 0.94 (36) | 0.98 (27) | 0.86 (16) | 0.92 (8) | 0.99 (13) | 1.00 (7) |
| `ft_e5small_syn` | 0.91 (36) | 0.89 (27) | 0.86 (16) | 0.91 (8) | 0.97 (13) | 1.00 (7) |
| `llm_v3_qwen3-4b` | 0.95 (36) | 0.98 (27) | 0.87 (16) | 0.92 (8) | 0.96 (13) | 1.00 (7) |
| `tfidf-logreg` | 0.93 (36) | 0.91 (27) | 0.93 (16) | 0.92 (8) | 0.95 (13) | 0.81 (7) |

**Overall by phenomenon** (n in brackets)

| Backend | entities | incident | multi_issue | out_of_scope | paraphrase | sentiment_edge | style_noise | urgency_edge |
|---|---|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 0.96 (13) | 1.00 (1) | 0.80 (13) | 1.00 (3) | 0.96 (61) | 1.00 (2) | 1.00 (2) | 0.91 (12) |
| `ft_e5small_syn` | 1.00 (13) | 1.00 (1) | 0.76 (13) | 0.68 (3) | 0.94 (61) | 1.00 (2) | 1.00 (2) | 0.88 (12) |
| `llm_v3_qwen3-4b` | 0.96 (13) | 1.00 (1) | 0.84 (13) | 1.00 (3) | 0.98 (61) | 1.00 (2) | 1.00 (2) | 0.91 (12) |
| `tfidf-logreg` | 0.91 (13) | 1.00 (1) | 0.82 (13) | 0.68 (3) | 0.91 (61) | 1.00 (2) | 1.00 (2) | 1.00 (12) |

**Invariance under label-preserving perturbations** (share of answers unchanged vs the clean text)

| Backend | arabic_digits | bidi_marks | elongate | emoji_punct | glue_words | html_wrap | injection_suffix | no_punct | padding | pii_suffix | tashkeel | tatweel | typo_double | typo_drop | upper_latin | zero_width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cascade_e5syn_qwen3-4b` | 1.0 | 1.0 | 0.972 | 0.963 | 0.953 | 0.972 | 0.981 | 0.947 | 0.907 | 0.981 | 1.0 | 1.0 | 0.981 | 0.962 | 1.0 | 1.0 |
| `ft_e5small_syn` | 0.862 | 0.963 | 0.915 | 0.953 | 0.925 | 0.888 | 0.963 | 0.947 | 0.869 | 0.953 | 0.953 | 0.984 | 0.953 | 0.934 | 0.98 | 0.99 |
| `llm_v3_qwen3-4b` | 1.0 | 1.0 | 0.981 | 0.953 | 0.953 | 0.991 | 0.981 | 0.947 | 0.869 | 0.963 | 1.0 | 1.0 | 0.981 | 0.972 | 0.961 | 1.0 |
| `tfidf-logreg` | 0.966 | 1.0 | 0.962 | 0.981 | 0.953 | 0.944 | 0.972 | 0.842 | 0.86 | 0.981 | 1.0 | 1.0 | 0.991 | 0.934 | 1.0 | 1.0 |

