"""Build the training / internal-dev data from data/tickets.csv.

Steps (see docs/STAGE1_DATA.md for why):
 1. Remove every dev-gold ticket: by ticket_id AND by normalised text (28 gold texts repeat elsewhere).
 2. Reduce each segment to a "core": greeting, sign-off, intensifiers, brand/store names and numbers
    are stripped, so 'السلام عليكم، شامبو دوف باظ بعد يومين شكرا' and 'برسيل باظ بعد يومين !!!' share a core.
 3. Staff labels are noisy (~15-20% random flips). Take the MAJORITY staff label of each core as the
    clean label and propagate it to unlabeled tickets and to second segments with the same core.
 4. Typo variants that form their own tiny core get the label of their nearest labelled neighbour
    (only when the char n-gram cosine similarity is >= 0.75).
 5. Split cores (not rows) into train / internal-dev, so near-duplicates never cross the split.

Outputs: data/processed/segments.csv, data/processed/messages.csv, outputs/metrics/label_noise.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.config import settings  # noqa: E402
from orvana.extractors import load_catalog, split_segments  # noqa: E402
from orvana.normalize import clean_message, match_key  # noqa: E402

OUT = settings.data_dir / "processed"
SEED = 42

GREETINGS = ["السلام عليكم", "مساء الخير", "لو سمحت", "يا جماعة", "hi", "hello", "dear team", "salam",
             "law sama7t", "ya gama3a"]
SIGN_OFFS = ["ياريت حد يكلمني", "ارجو الرد", "شكرا", "حرام عليكم", "مش معقول كده", "دي تالت مرة أشتكي",
             "7ad yekalemni", "plz rod", "shokran", "7aram 3aleiko", "mesh ma32ool", "di talet mara ashtki",
             "thanks", "please reply", "asap", "unacceptable", "third time i complain", "!!!", "😡"]
TAILS = ["أنا عميل عندكم", "ana 3amil 3andoko", "i've been a loyal customer"]


def _strip_list(words):
    keys = sorted({match_key(w) for w in words}, key=len, reverse=True)
    return "|".join(re.escape(k) for k in keys)


_GREET_RE = re.compile(rf"^(?:{_strip_list(GREETINGS)})\s*[,،]?\s*")
_SIGN_RE = re.compile(rf"(?:\s*(?:{_strip_list(SIGN_OFFS)}))+\s*$")
_TAIL_RE = re.compile(rf"(?:{_strip_list(TAILS)}).*$")


# generic product nouns (catalogue names + their Franco spellings): "Persil detergent" and
# "Tornado blender" must give the same core, otherwise each product forms its own tiny core
_PRODUCT_WORDS = None


def _product_words() -> list[str]:
    global _PRODUCT_WORDS
    if _PRODUCT_WORDS is None:
        p = pd.read_csv(settings.data_dir / "products_catalog.csv")
        words = set()
        for name in list(p.product_name_ar) + list(p.product_name_en):
            words.update(w for w in match_key(name).split() if len(w) > 2)
        words.update(match_key(w) for w in ["sama3et", "sha7en", "talaga", "makwa", "laban", "zeit", "mobile",
                                            "power bank", "blender", "shampoo", "phone", "earbuds"])
        _PRODUCT_WORDS = sorted(words, key=len, reverse=True)
    return _PRODUCT_WORDS


def core_key(segment: str) -> str:
    cat = load_catalog()
    k = _TAIL_RE.sub("", match_key(segment)).strip()
    for _ in range(3):
        k = _GREET_RE.sub("", k).strip()
        k = _SIGN_RE.sub("", k).strip()
    for alias, _ in cat.store_aliases:
        k = k.replace(alias, "<S>")
    for alias, _ in cat.brand_aliases:
        k = k.replace(alias, "<P>")
    for w in _product_words():
        k = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", "<P>", k)
    k = re.sub(r"(?:<P>\s*)+", "<P> ", k)  # 'سماعه <P>' / '<P> detergent' -> '<P>'
    k = re.sub(r"\b(?:el|the)\s+(?=<P>)", "", k)
    k = re.sub(r"\d+", "<N>", k)
    return re.sub(r"\s+", " ", re.sub(r"[^\w<>\s%؟?]", "", k)).strip()


def main():
    tickets = pd.read_csv(settings.data_dir / "tickets.csv")
    gold = pd.read_csv(settings.data_dir / "tickets_dev_gold.csv")

    gold_keys = set(gold.message_text.map(match_key))
    leak_id = tickets.ticket_id.isin(gold.ticket_id)
    leak_text = tickets.message_text.map(match_key).isin(gold_keys)
    df = tickets[~leak_id & ~leak_text].copy()
    print(f"removed {leak_id.sum()} gold ids + {(leak_text & ~leak_id).sum()} rows with gold text -> {len(df)} rows")

    # ---- segments and cores
    df["clean"] = df.message_text.map(lambda t: clean_message(t).text)
    df["segments"] = df.clean.map(split_segments)
    seg = df[["ticket_id", "segments", "label_category"]].explode("segments").rename(columns={"segments": "text"})
    seg["position"] = seg.groupby("ticket_id").cumcount()
    seg["core"] = seg.text.map(core_key)

    # ---- majority staff label per core, using only PRIMARY segments (staff label = main problem)
    prim = seg[(seg.position == 0) & seg.label_category.notna()]
    votes = prim.groupby("core").label_category.agg(
        n_labeled="size",
        majority=lambda s: s.value_counts().index[0],
        share=lambda s: s.value_counts(normalize=True).iloc[0],
    )
    reliable = votes[(votes.n_labeled >= 3) & (votes.share >= 0.5)]
    seg = seg.merge(reliable[["majority"]], left_on="core", right_index=True, how="left")
    seg = seg.rename(columns={"majority": "label_clean"}).reset_index(drop=True)
    seg["label_source"] = np.where(seg.label_clean.notna(), "core_majority", "")

    # ---- typo variants form their own tiny cores: give them the label of the most similar
    #      labelled segment, but only when the match is very close (char n-gram cosine >= 0.75)
    #      (compared on the CORE, so long shared sign-offs/tails cannot dominate the similarity)
    seg["text"] = seg.text.fillna("")
    known, unknown = seg[seg.label_clean.notna()], seg[seg.label_clean.isna() & (seg.core.str.len() > 0)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(known.core)
    sims = vec.transform(unknown.core) @ vec.transform(known.core).T
    best = np.asarray(sims.argmax(axis=1)).ravel()
    best_sim = np.asarray(sims.max(axis=1).todense()).ravel()
    take = best_sim >= 0.75
    idx = unknown.index[take]
    seg.loc[idx, "label_clean"] = known.label_clean.to_numpy()[best[take]]
    seg.loc[idx, "label_source"] = "nearest_neighbour"
    print(f"nearest-neighbour labels: {take.sum()} of {len(unknown)} unlabeled segments")

    # ---- noise estimate on primary segments whose core has a reliable majority
    chk = seg[(seg.position == 0) & seg.label_category.notna() & (seg.label_source == "core_majority")]
    noise = float((chk.label_category != chk.label_clean).mean())
    print(f"staff-label disagreement with core majority: {noise:.1%} (n={len(chk)})")
    print(f"segments with a clean label: {seg.label_clean.notna().mean():.1%} of {len(seg)}")

    # ---- group split by core: ~12% of cores -> internal dev
    rng = np.random.default_rng(SEED)
    cores = seg.core.unique()
    dev_cores = set(rng.choice(cores, size=int(0.12 * len(cores)), replace=False))
    seg["split"] = np.where(seg.core.isin(dev_cores), "dev", "train")

    # message-level view (primary/secondary clean labels) for prompt/model selection
    wide = seg.pivot_table(index="ticket_id", columns="position", values="label_clean", aggfunc="first")
    msgs = df[["ticket_id", "message_text", "label_category"]].merge(
        wide.rename(columns={0: "primary_clean", 1: "secondary_clean"}), left_on="ticket_id", right_index=True, how="left")
    msgs = msgs.merge(seg[seg.position == 0][["ticket_id", "split"]], on="ticket_id")

    OUT.mkdir(parents=True, exist_ok=True)
    seg.drop(columns=["label_category"]).to_csv(OUT / "segments.csv", index=False)
    msgs.to_csv(OUT / "messages.csv", index=False)
    stats = {
        "rows_after_leak_removal": len(df), "gold_ids_removed": int(leak_id.sum()),
        "gold_text_rows_removed": int((leak_text & ~leak_id).sum()),
        "segments": len(seg), "distinct_cores": int(seg.core.nunique()),
        "reliable_cores": len(reliable), "segments_with_clean_label": float(seg.label_clean.notna().mean()),
        "staff_label_noise_estimate": noise,
        "label_distribution_clean": seg.label_clean.value_counts().to_dict(),
        "dev_messages": int((msgs.split == "dev").sum()),
    }
    (settings.outputs_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (settings.outputs_dir / "metrics" / "label_noise.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
