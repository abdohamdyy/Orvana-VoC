"""Synthetic training data for Stage 3: new PHRASINGS the historical templates never contain.

    python scripts/gen_synthetic.py [--per 20]

- Generator: qwen2.5:7b-instruct, given ONLY the category definitions + a style description
  (never the test suite, never the gold set).
- Filter: qwen3:4b with the v3 classifier prompt (a different model, so their mistakes are less
  correlated): a message is kept only if the filter agrees with the intended category.
- Leakage guard: anything with cosine > 0.9 to a test-suite text is dropped.
Output: data/processed/synthetic.csv (text, label, style, kept, filter_label)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.backends.llm_ollama import OllamaBackend  # noqa: E402
from orvana.backends.prompts import CATEGORY_DEFINITIONS  # noqa: E402
from orvana.config import settings  # noqa: E402
from orvana.normalize import clean_message  # noqa: E402
from orvana.schema import CATEGORIES  # noqa: E402

STYLES = {
    "egy": "Egyptian colloquial Arabic written in Arabic script, like a WhatsApp message",
    "msa": "formal Modern Standard Arabic (فصحى), like an email",
    "franco": "Franco-Arabic / Arabizi: Egyptian Arabic written in Latin letters with numbers (3=ع, 7=ح, 2=ء, 5=خ)",
    "en": "casual English written by an Egyptian customer",
    "mixed": "Egyptian Arabic in Arabic script mixed with English words (code-switching), e.g. 'الـ order اتأخر'",
    "voice": "a voice note transcribed to Arabic text: no punctuation, filler words (يعني، بص، ايوه)",
}
NO_COMPLAINT = ("messages that contain NO complaint at all: greetings only, thanks, a general question about "
                "the company, a job request, a question about another shop, spam or random text")

SYSTEM = """You write realistic customer messages sent to Orvana, an Egyptian retail chain (supermarkets, electronics, home appliances; app and home delivery; loyalty programme "Orvana Plus").
Each message must describe exactly ONE problem of the requested category. Vary length (3 to 30 words), tone (angry, polite, sarcastic, tired), vocabulary and sentence structure. Do not copy the definition wording. Do not number the messages.
Category definitions:
""" + CATEGORY_DEFINITIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20, help="messages per (category, style)")
    ap.add_argument("--generator", default="qwen2.5:7b-instruct")
    ap.add_argument("--filter", default="qwen3:4b")
    args = ap.parse_args()

    gen = OllamaBackend(model=args.generator, prompt_version="synthetic-v1")
    schema = {"type": "object", "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
              "required": ["messages"]}
    rows = []
    jobs = [(c, s) for c in CATEGORIES for s in STYLES] + [("inquiry", s) for s in ("egy", "en", "franco")]
    for i, (cat, style) in enumerate(jobs):
        no_complaint = i >= len(CATEGORIES) * len(STYLES)
        what = NO_COMPLAINT if no_complaint else f"the category '{cat}'"
        for batch in range(max(1, args.per // 10)):
            user = f"Write 10 different messages for {what}. Style: {STYLES[style]}. Return JSON {{\"messages\": [...]}}."
            try:
                out = gen._chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], schema,
                                temperature=0.9, seed=1000 + 31 * i + batch)
            except RuntimeError as e:
                print("skip", cat, style, e)
                continue
            for m in out.get("messages", [])[:10]:
                if isinstance(m, str) and 8 <= len(m) <= 400:
                    rows.append({"text": m.strip(), "label": cat, "style": style, "no_complaint": no_complaint})
        print(f"{i + 1}/{len(jobs)} {cat}/{style}: {len(rows)} so far", flush=True)
    # free the GPU for the filter model / training
    requests.post(f"{settings.ollama_url}/api/generate", json={"model": args.generator, "keep_alive": 0}, timeout=30)

    df = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    df["text"] = df.text.map(lambda t: clean_message(t).text)
    filt = OllamaBackend(model=args.filter, prompt_version="v3")
    df["filter_label"] = [c.value for c, _ in filt.classify(df.text.tolist())]
    df["kept"] = df.filter_label == df.label
    # quality filters found by READING the output (not visible in the agreement rate):
    # - the 7B sometimes drifts into Chinese/other scripts mid-message
    # - its Franco is gibberish ("3ny 5lm 7la 2mlik khshsh ma3lomet"): Franco is dropped entirely and
    #   Franco robustness comes from the historical data (29% Franco) + typo augmentation instead
    foreign = df.text.str.contains(r"[Ѐ-ӿ぀-ヿ㐀-鿿가-힯]")
    df["drop_reason"] = ""
    df.loc[foreign, "drop_reason"] = "foreign script"
    df.loc[df["style"] == "franco", "drop_reason"] = "franco generation unusable"
    df.loc[df.text.str.split().str.len() < 3, "drop_reason"] = "too short"
    df.loc[df.drop_reason != "", "kept"] = False

    # leakage guard: never train on something that looks like a test-suite message
    suite = pd.read_csv(Path(__file__).resolve().parent.parent / "tests" / "suite" / "base.csv").message_text.tolist()
    sec = pd.read_csv(Path(__file__).resolve().parent.parent / "tests" / "suite" / "security.csv").message_text.tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit(df.text.tolist() + suite + sec)
    sim = (vec.transform(df.text) @ vec.transform(suite + sec).T).max(axis=1).toarray().ravel()
    df["max_sim_to_suite"] = sim.round(3)
    df.loc[sim > 0.9, "kept"] = False
    df.loc[sim > 0.9, "drop_reason"] = "too close to the test suite"

    out = settings.data_dir / "processed" / "synthetic.csv"
    df.to_csv(out, index=False)
    print(json.dumps({"generated": len(df), "kept": int(df.kept.sum()),
                      "agreement_rate": round(float((df.filter_label == df.label).mean()), 3),
                      "dropped_as_too_close_to_suite": int((sim > 0.9).sum()),
                      "dropped_by_quality_filter": df.drop_reason[df.drop_reason != ""].value_counts().to_dict(),
                      "kept_per_label": df[df.kept].label.value_counts().to_dict()}, indent=1))


if __name__ == "__main__":
    main()
