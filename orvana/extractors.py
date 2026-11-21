"""Deterministic extractors: product (brand), store (store_id), segment split, urgency flags.

Product and store are closed lists, so a dictionary lookup is more reliable, cheaper and easier
to audit than asking a model. Matching runs on match_key() text so typos/letter variants still hit.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz

from .config import settings
from .normalize import match_key

_LETTER = re.compile(r"[^\W\d_]")
_AR_PREFIX = "وبلف"  # Arabic one-letter prefixes that can be glued to a name (لطنطا, بفرع ...)


def _is_letter(ch: str) -> bool:
    return bool(_LETTER.match(ch))


def _find_alias(key: str, alias: str) -> int:
    """Position of alias in key with word boundaries (allowing one glued Arabic prefix), or -1."""
    start = 0
    while (i := key.find(alias, start)) != -1:
        before = key[i - 1] if i > 0 else " "
        before2 = key[i - 2] if i > 1 else " "
        after = key[i + len(alias)] if i + len(alias) < len(key) else " "
        ok_before = not _is_letter(before) and not before.isdigit() or (
            before in _AR_PREFIX and not _is_letter(before2)
        )
        ok_after = not _is_letter(after) and not after.isdigit()
        if ok_before and ok_after:
            return i
        start = i + 1
    return -1


@dataclass
class Catalog:
    brand_aliases: list[tuple[str, str]]  # (alias_key, brand), longest first
    store_aliases: list[tuple[str, str]]  # (alias_key, store_id), longest first
    store_ids: set[str]
    store_ctx_aliases: list[tuple[str, str]] = ()  # Franco spellings, only next to a branch word (see find_store)
    brand_category: dict[str, str] = field(default_factory=dict)  # brand -> product type (Electronics, FMCG, Cooking oil ...)


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    products = pd.read_csv(settings.data_dir / "products_catalog.csv")
    stores = pd.read_csv(settings.data_dir / "store_names.csv")

    brands: dict[str, str] = {}
    for r in products.itertuples():
        brands[match_key(r.brand)] = r.brand
        last_ar = r.product_name_ar.split()[-1]
        if re.search(r"[ء-ي]", last_ar):  # Arabic brand word, e.g. 'تورنادو'
            brands[match_key(last_ar)] = r.brand
        if r.brand.lower() not in r.product_name_en.lower():  # product sold under another name (Molto -> Edita)
            brands[match_key(r.product_name_en)] = r.brand

    store_alias: dict[str, str] = {}
    for r in stores.itertuples():
        for name in (r.store_id, r.name_ar, r.name_ar.replace("فرع ", ""), r.name_en, r.name_en.replace(" branch", "")):
            store_alias[match_key(name)] = r.store_id

    # extra spellings maintained as data (no code change needed to add one): data/aliases.csv
    extra_path = settings.data_dir / "aliases.csv"
    store_ctx: dict[str, str] = {}
    if extra_path.exists():
        for r in pd.read_csv(extra_path).itertuples():
            target = brands if r.kind == "brand" else store_ctx if r.kind == "store_ctx" else store_alias
            target[match_key(r.alias)] = r.value

    # product type for safety_check(); cooking oil is food that is heated in use, so smoke / a burnt smell is real
    types = {r.brand: "Cooking oil" if re.search(r"\boil\b", r.product_name_en, re.IGNORECASE) else r.category
             for r in products.itertuples()}
    by_len = lambda d: sorted(d.items(), key=lambda kv: -len(kv[0]))
    return Catalog(by_len(brands), by_len(store_alias), set(stores.store_id), by_len(store_ctx), types)


def _first_match(key: str, aliases: list[tuple[str, str]]) -> str | None:
    """Earliest mention wins; at the same position the longest alias wins ('المنصوره 2' beats 'المنصوره').
    Pass 1 needs word boundaries. Pass 2 (only if pass 1 found nothing) accepts long aliases glued to
    neighbouring words, a common typo in the data ('far3Downtown', 'Moltohayerga3')."""
    for strict in (True, False):
        best = None  # (position, -len, value)
        for alias, value in aliases:
            if strict:
                pos = _find_alias(key, alias)
            else:
                pos = key.find(alias) if len(alias) >= 5 else -1
            if pos != -1:
                cand = (pos, -len(alias), value)
                best = cand if best is None or cand < best else best
        if best:
            return best[2]
    return None


def _fuzzy_match(key: str, aliases: list[tuple[str, str]], threshold: int = 88) -> str | None:
    """Typo fallback (a dropped/swapped letter): compare 1-3 word windows with aliases of 5+ chars."""
    words = re.sub(r"[^\w\s-]", " ", key).split()
    windows = [" ".join(words[i : i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)]
    best = (0, None)
    for alias, value in aliases:
        if len(alias) < 5:
            continue
        for w in windows:
            if abs(len(w) - len(alias)) <= 2:
                score = fuzz.ratio(w, alias)
                if score >= threshold and score > best[0]:
                    best = (score, value)
    return best[1]


def find_product(text: str) -> str | None:
    cat = load_catalog()
    key = match_key(text)
    return _first_match(key, cat.brand_aliases) or _fuzzy_match(key, cat.brand_aliases)


_CODE_RE = re.compile(r"\b(CAI|GIZ|ALX|DLT|UPE|CNL)\s*-*\s*(\d{1,3})\b", re.IGNORECASE)


def _store_from_code(text: str, valid: set[str]) -> str | None:
    """Store codes, tolerant to typos such as 'DLT-088' (duplicated digit) or 'cai-3'."""
    for m in _CODE_RE.finditer(text):
        prefix, digits = m.group(1).upper(), m.group(2)
        candidates = [digits, digits.zfill(2), re.sub(r"(\d)\1", r"\1", digits)]
        for d in candidates:
            code = f"{prefix}-{d.zfill(2)}"
            if code in valid:
                return code
    return None


_STORE_CONTEXT = re.compile(r"فرع|far3|fr3|branch|store|محل")


def find_store(text: str) -> str | None:
    cat = load_catalog()
    key = match_key(text)
    name_hit = _first_match(key, cat.store_aliases)
    if name_hit:
        return name_hit
    code = _store_from_code(text, cat.store_ids)
    if code:
        return code
    # Franco spellings ('far3 el ma3adi') and typo-tolerant matching only when the customer is clearly
    # talking about a branch (otherwise 'maksoura' = broken would match 'Mansoura')
    if not _STORE_CONTEXT.search(key):
        return None
    return _first_match(key, cat.store_ctx_aliases) or _fuzzy_match(key, cat.store_aliases, threshold=85)


# Customers join two problems with 'وكمان' / 'w kaman' / 'and also' (28/28 multi-issue gold tickets).
# Typos drop one letter ('كمان', 'وكما', 'kaman'), so those variants are accepted too; 'وكان' ("and was")
# is deliberately NOT a connector. Other explicit connectors found in real Egyptian/MSA/English writing
# are accepted too. A split is only kept if BOTH sides have >= 2 words, so "والأسعار كمان غليت"
# ("and prices also went up") is not cut into a one-word fragment.
_AR = r"ء-ي"
_SPLIT_RE = re.compile(
    rf"\s*(?:و+ك+م+ا+ن+|(?<![{_AR}])ك+م+ا+ن+(?![{_AR}])|و+ك+م+ا+(?![{_AR}])|و+ك+م+ن+(?![{_AR}])"
    rf"|و\s?(?:برضو|برضه|برده|بردو)(?![{_AR}])|و?غير\s+كد[اه](?![{_AR}])|و?مش\s+بس\s+كد[اه](?![{_AR}])"
    rf"|كما\s+[أا]ن(?![{_AR}])|علاو[ةه]\s+على\s+ذلك|وكذلك|و[أا]يض[اً]+"
    r"|\b(?:w+\s*)?k+a*m+a*n+\b|\bw\s?bard[ou]\b|\b(?:and|nd|an)\s+(?:also|aslo|alos|als|lso|aso)\b"
    r"|(?<=[.,;!])\s*also\b|(?<!orvana\s)(?<!orvana)\bplus\b|\bin\s+addition\b|\badditionally\b|\bon\s+top\s+of\s+that\b"
    r"|\s\+\s|\n+)\s*",
    re.IGNORECASE)


def _words(s: str) -> int:
    return len(re.findall(r"[^\W\d_]{2,}", s))


def split_segments(text: str) -> list[str]:
    """[primary] or [primary, secondary]: the first issue mentioned is the main one."""
    for m in _SPLIT_RE.finditer(text):
        left, right = text[: m.start()].strip(" ,،.;"), text[m.end():].strip(" ,،.;")
        if m.group().strip().lower() == "plus" and re.search(r"\bor\w*$", left.lower()):
            continue  # "Orvana Plus" (and typos like "Orvna Plus") is the loyalty programme, not "plus"
        if _words(left) >= 2 and _words(right) >= 2:
            return [left, right]
    return [text.strip()] if text.strip() else [text]


def _kw(*words: str) -> re.Pattern:
    """Keyword alternation. Latin words get a left word boundary (and a right one when short,
    so 'hot' does not fire on 'hotline'); Arabic stems match inside words ('حرار' -> 'حرارة').
    A leading '*' marks a Franco stem that must also match inside a word ('*skhan' in 'beyskhan')."""
    parts = []
    for w in words:
        stem = w.startswith("*")
        k = re.escape(match_key(w.lstrip("*")))
        if re.match(r"[a-z0-9]", k) and not stem:
            k = rf"(?<![a-z]){k}" + (r"(?![a-z])" if len(w) <= 4 else "")
        parts.append(k)
    return re.compile("|".join(parts))


def _phrases(*phrases: str) -> re.Pattern:
    """Whole-word phrases (unlike _kw, an Arabic word never matches inside a longer one: 'دود' must not
    fire on 'حدود', 'قزاز' (glass) not on 'قزازة' (bottle)). Arabic words may carry one glued prefix
    (و ف ب ل) and the article. A trailing '*' allows any ending ('explod*' -> 'exploded', 'explosion')."""
    parts = []
    for p in phrases:
        stem = p.endswith("*")
        k = re.escape(match_key(p.rstrip("*")))
        if re.match(r"[ء-ي]", k):
            parts.append(rf"(?<![ء-ي])(?:[وفبل]|لل)?(?:ال)?{k}" + ("" if stem else r"(?![ء-ي])"))
        else:
            parts.append(rf"(?<![a-z0-9]){k}" + ("" if stem else r"(?![a-z0-9])"))
    return re.compile("|".join(parts))


# Safety keywords, grouped by the kind of hazard: whether a hazard makes sense depends on the product
# (an expired charger, an overheating carton of milk), see safety_check(). Lists mined from all 9k tickets and
# widened with synonyms of the policy's triggers ("overheating, smelling burnt, expired food"); matched on
# match_key() text. See docs/URGENCY_RULES.md.
_EXPIRY = _kw("منتهي الصلاحية", "منتهي", "انتهت صلاحي", "صلاحيته", "expired", "expiry", "*salahy")
_HEAT = [
    _kw("بيسخن", "سخن", "حرار", "شياط", "محروق", "احتراق", "احترق", "شتعل", "حريق", "شرار", "دخان", "دخن", "*sa5an",
        "*skhan", "*so5n", "*sokhn", "*shayat", "*ma7roo2", "*do5an", "overheat", "burnt", "burning", "smoke", "smoking",
        "spark", "melted"),
    # "hot" alone is not a hazard ("hot deal"): it must describe the product heating up
    re.compile(r"(?<![a-z])(?:very|so|to|extremely|realy|super) hot(?![a-z])"
               r"|(?<![a-z])(?:gets?|geting|got|becomes?|became) (?:very |so |to |realy )?hot(?![a-z])"
               r"|(?<![a-z])heat(?:s|ing)? up(?![a-z])"),
    _phrases("ماس كهربا", "ماس كهربائي", "ماس كهربي", "كهربتني", "كهربني", "بيكهرب", "بتكهرب", "مسكتني كهربا",
           "مسكني كهربا", "ضربتني كهربا", "ضربني كهربا", "لسعتني كهربا", "لسعني كهربا", "انفجر", "انفجرت", "فرقع", "فرقعت",
           "ولعت", "مسك نار", "مسكت نار", "طلع نار", "طلعت نار", "kahrabetni", "kahrabni", "beykahrab", "betkahrab",
           "mas kahraba", "enfagar", "enfagaret", "wala3et", "electric shock", "electrical shock", "shocked me",
           "gave me a shock", "electrocut*", "short circuit", "explod*", "explosion", "caught fire", "on fire",
           "burst into flames", "flames"),
]
_CONTAMINATION = _phrases(
    "حته قزاز", "حتت قزاز", "قطعه قزاز", "قطع قزاز", "قزاز مكسور", "حته ازاز", "حتت ازاز", "قطعه ازاز", "ازاز مكسور",
    "قطعه زجاج", "قطع زجاج", "زجاج مكسور", "شظايا", "حشره", "حشرات", "صرصار", "صراصير", "دود", "دوده", "سوس", "مسوس",
    "مسوسه", "عفن", "عفنه", "معفن", "معفنه", "متعفن", "متعفنه", "7etet ezaz", "7eta ezaz", "7ashra", "7asharat", "sarsar",
    "sarasir", "dood", "3afan", "me3afen", "met3afen", "mesawes", "piece of glass", "pieces of glass", "glass in",
    "broken glass", "glass shard*", "insect*", "cockroach*", "roach*", "worm*", "maggot*", "mold", "moldy", "mould",
    "mouldy", "rotten")
# a person or the pack is affected: a hazard whatever the product
_ALWAYS = [
    _kw("حرقان", "منفوخ", "تسمم", "swollen", "poison"),
    _phrases("حساسيه", "طفح", "اتسممت", "اتسمم", "اتسمموا", "مستشفي", "اسهال", "مغص", "استفراغ", "حروق", "7asasiya",
           "7asaseya", "tasamom", "etsamemt", "mostashfa", "eshal", "maghas", "allerg*", "rash", "food poisoning",
           "hospital*", "vomit*", "diarh*", "threw up", "throwing up", "stomach ache"),
]
SAFETY_KINDS: dict[str, list[re.Pattern]] = {"expiry": [_EXPIRY], "heat": _HEAT, "contamination": [_CONTAMINATION],
                                             "always": _ALWAYS}
# product types (products_catalog.csv) where each kind of hazard makes sense; "always" fits every product
_PLAUSIBLE = {"expiry": {"FMCG", "Personal Care", "Cooking oil"}, "heat": {"Electronics", "Home & Kitchen", "Cooking oil"},
              "contamination": {"FMCG", "Personal Care", "Cooking oil"}}

# Guards: the keyword must describe the product. "النقط بتاعتي منتهية" (my points expired), "الكوبون منتهي",
# "warranty expired" are not hazards, and neither is a negated one ("اللبن مش منتهي الصلاحية", "not expired").
_NOT_PRODUCT = re.compile(
    r"^[وف]?(?:ال|لل)?(?:نقط|نقاط|بوينت|كوبون|فاوتشر|فوتشر|قسيم|عرض|عروض|خصم|ضمان|كارت|كارد|بطاق|فيزا|اشتراك|عضوي|"
    r"كود|بروموكود|لينك|رصيد|باق)"
    r"|^(?:el)?(?:points?|coupons?|copon|kobon|kupon|vouchers?|ofers?|3ard|deals?|discounts?|5asm|khasm|daman|"
    r"waranty|guarantee|cards?|visa|subscriptions?|eshterak|codes?|promo\w*|balance|raseed|membership|plus|link)$")
_NEGATION = re.compile(r"^(?:مش|مو|غير|مكانش|ماكانش|مكنش|مكانتش|مكنتش|لا|ليس|ليست|مفيش|not|never|no|msh|mesh|mish|mosh|"
                       r"(?:is|was|are|were|does|do|did|has|have|had|wo|ca|could|should|would)n'?t)$")
_NEG_FILLER = {"yet", "even", "realy", "actualy", "لسه", "اصلا", "ابدا", "خالص"}
_PUNCT = ".,،!?؟;:()\"'"


def _tokens_before(key: str, start: int) -> list[str]:
    before = key[:start]
    toks = before.split()
    if toks and not before[-1:].isspace():
        toks = toks[:-1]  # the match starts inside a word: drop its prefix ('بي' of 'بيسخن')
    return toks


def _guarded(key: str, m: re.Match, kind: str) -> bool:
    before = _tokens_before(key, m.start())
    last = [t.strip(_PUNCT) for t in before[-2:]]
    if last and (_NEGATION.match(last[-1]) or len(last) == 2 and _NEGATION.match(last[0]) and last[1] in _NEG_FILLER):
        return True
    if kind != "expiry":
        return False
    clause = []
    for t in reversed(before[-3:]):  # up to 3 words back, without crossing a comma or full stop
        if t[-1:] in _PUNCT:
            break
        clause.append(t)
    if any(_NOT_PRODUCT.match(t.strip(_PUNCT)) for t in clause):
        return True
    rest = key[m.end():]
    after = rest.split(maxsplit=2)
    if after and not rest[:1].isspace():
        after = after[1:]  # the match ended inside a word ('صلاحي' + 'ه')
    return bool(after) and not after[0].startswith("و") and bool(_NOT_PRODUCT.match(after[0].strip(_PUNCT)))


def safety_kinds(segment: str) -> set[str]:
    """Kinds of hazard the text reports (expiry, heat, contamination, always), after the guards."""
    key = match_key(segment)
    return {kind for kind, patterns in SAFETY_KINDS.items()
            if any(not _guarded(key, m, kind) for rx in patterns for m in rx.finditer(key))}
# "repeat complaint": an ordinal >= 3 next to 'time/مرة/complaint' in Egyptian, MSA, Franco or English
_ORD_AR = r"(?:تالت|ثالث|رابع|خامس|سادس)"
REPEAT_RE = re.compile(
    rf"{_ORD_AR}\w*\s+(?:مر|شكو)|مر[هة]\s+(?:ال)?{_ORD_AR}"
    r"|\b(?:third|fourth|fifth|3rd|4th|5th)\s+(?:time|complaint|request|message)|\bfor\s+the\s+(?:third|fourth|fifth|3rd|4th|5th)\b"
    r"|\b(?:talet|tal[e]?t|rabe3|5ames|khames)\s+mar+a")
DOUBLE_CHARGE_RE = _kw("مرتين", "marteen", "twice", "double charge", "charged double")
_THOUSANDS = re.compile(r"(?<![ء-ي])(?:الف|الفين|الاف|تلاف)(?![ء-ي])|\b(?:thousand|alf|alfen|alfein)\b")


def has_safety_issue(segment: str) -> bool:
    return bool(safety_kinds(segment))


def safety_check(segment: str, brand: str | None = None) -> str | None:
    """None (no hazard), 'confirmed', or 'needs_confirmation' when the product type rarely has this hazard
    ('Samsung phone expired', 'Juhayna milk gets very hot'). The urgency rules do NOT use this: the gold labels
    such tickets 3, so they stay 3; it only tells the Quality team / the chatbot to confirm with the customer."""
    kinds = safety_kinds(segment)
    if not kinds:
        return None
    product_type = load_catalog().brand_category.get(brand or find_product(segment) or "")
    if product_type is None or "always" in kinds or any(product_type in _PLAUSIBLE[k] for k in kinds):
        return "confirmed"
    return "needs_confirmation"


def is_repeat_complaint(text: str) -> bool:
    return bool(REPEAT_RE.search(match_key(text)))


def double_charge_amount(segment: str) -> int | None:
    """Amount in EGP if the segment describes a double charge, else None.
    Amounts written in words ('ألفين جنيه', 'two thousand') count as >= 1000."""
    key = match_key(segment)
    if not DOUBLE_CHARGE_RE.search(key):
        return None
    amounts = [int(n) for n in re.findall(r"\d+", key) if int(n) >= 20]
    if amounts:
        return max(amounts)
    return 1000 if _THOUSANDS.search(key) else 0
