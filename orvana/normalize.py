"""Text cleaning applied before anything else touches a message.

- normalize_text(): Unicode clean-up (NFKC, invisible/bidi characters, tashkeel, tatweel, homoglyphs)
- match_key():      aggressive normalisation used ONLY for dictionary/keyword matching
- clean_message():  what the classifier sees: injections removed, label tokens stripped, PII masked,
                    length capped, plus flags (injection_flag, is_valid)
"""
import base64
import re
import unicodedata
from dataclasses import dataclass

from .config import settings

_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ـ": ""})
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_INVISIBLE = re.compile(r"[­؜​-‏‪-‮⁠-⁤⁦-⁩﻿]")
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # diacritics + tatweel
# letters that look Latin but are Cyrillic/Greek (used to hide "Ignore"), and Persian forms of Arabic letters
_HOMOGLYPHS = str.maketrans({"І": "I", "і": "i", "О": "O", "о": "o", "А": "A", "а": "a", "Е": "E", "е": "e",
                             "Р": "P", "р": "p", "С": "C", "с": "c", "Х": "X", "х": "x", "У": "Y", "у": "y",
                             "Ι": "I", "ο": "o", "α": "a", "ε": "e", "Т": "T", "Н": "H", "М": "M", "К": "K",
                             "ک": "ك", "ی": "ي", "ہ": "ه"})


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _TASHKEEL.sub("", text).translate(_HOMOGLYPHS)
    return re.sub(r"[ \t]+", " ", text).strip()


def match_key(s: str) -> str:
    """Lower-case, unify Arabic letter variants, and collapse repeated letters
    (the data has typos like 'ساامسونج', 'baank'). Digits are never collapsed (CAI-11)."""
    s = normalize_text(s).lower().translate(_AR_MAP).translate(_AR_DIGITS)
    s = re.sub(r"([^\W\d_])\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


# Prompt-injection markers. Written narrowly enough not to fire on real complaints
# ("تجاهل الموظف طلبي", "please reply with the tracking number"); checked on all 9k historical tickets
# by scripts/check_gates.py.
INJECTION_PATTERNS = [
    # English
    r"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|the|your|my|of)\s+)*(?:previous\s+|prior\s+|above\s+|earlier\s+|preceding\s+)?(?:instructions?|rules|prompts?|directions|guidelines)\b",
    r"\b(?:ignore|disregard|forget)\s+(?:everything|all)\s*(?:above|before|said|previous|you)?\b",
    r"\byou\s+are\s+now\b[^.\n]*", r"\bact\s+as\s+(?:an?\s+)?(?:ai|admin|system|assistant|model)\b",
    r"\brepeat\s+it\s+word\s+by\s+word\b",
    r"\bsystem\s*(?::|prompt|message|notice|instruction|override)", r"\bnew\s+rule\b",
    r"\b(?:note|message|instructions?)\s+(?:to|for)\s+(?:the\s+)?(?:ai|model|assistant|bot|classifier|system|report\s+writer)\b",
    r"\b(?:classify|categori[sz]e|label|tag)\s+(?:this|it|the\s+(?:message|ticket|complaint))\b",
    r"\bmark\s+(?:this|it)\s+(?:ticket\s+|message\s+)?as\b",
    r"\b(?:reply|respond|answer)\s+only\s+with\b", r"\boutput\s+(?:only\b|category|\{)",
    r"\bas\s+the\s+(?:system\s+)?admin(?:istrator)?\b", r"\bauthori[sz]e\s+you\b",
    r"\b(?:print|list|reveal|repeat|show)\b.{0,60}\b(?:customer\s+ids?|system\s+prompt|your\s+(?:prompt|instructions)|phone\s+numbers)\b",
    r"\bwhat\s+is\s+your\s+(?:system\s+)?prompt\b", r"\bdecode\s+and\s+(?:follow|execute|run)\b",
    r"\bin\s+your\s+reply\b", r"\btell\s+management\b[^.\n]*",
    # chat-template / delimiter tokens
    r"<\|im_(?:start|end)\|>", r"\[/?inst\]", r"###\s*instruction", r"</?\s*(?:system|customer_message|assistant|user)\s*>",
    # Arabic
    r"تجاهل\s+(?:كل\s+|جميع\s+)?(?:التعليمات|الاوامر|الأوامر|اللي\s+فات|ما\s+سبق)(?:\s+(?:السابق[ةه]|اللي\s+فات\S*))?",
    r"انس[ىي]\s+(?:كل\s+|جميع\s+)?(?:التعليمات|الاوامر|الأوامر|اللي\s+فات)(?:\s+(?:السابق[ةه]|اللي\s+فات\S*))?",
    r"(?:تعليمات|اوامر|أوامر)\s+النظام", r"(?:الفئ[ةه]|التصنيف)\s+(?:الصحيح[ةه]\s+)?(?:هي|هو)\b[^.\n،]*",
    r"(?<![ء-ي])و?صن[ّ]?ف(?:ها|ي|وا)?\s+(?:الشكو[ىي]|الرسال[ةه]|التذكر[ةه]|ده|دي)",  # not "الصنف ده" (= this item)
    r"اعتبر(?:ها|ه)?\s+(?:رسال[ةه]|استفسار|شكر|ايجابي|إيجابي)",
    r"يا\s+(?:ذكاء\s+اصطناعي|بوت|روبوت)", r"متتعاملش\s+مع\s+ده\s+ك",
    r"(?:ت?كتب|قول|قولي)(?:لي)?\s+في\s+الرد", r"رد\s+عليا\s+و\s?قول",
    # Franco
    r"\bignore\s+el\b(?:\s+\w+){0,3}", r"\bsan+e+f\b(?:\s+\w+){0,4}", r"\b(?:5|kh)al+e+(?:ha|h)\s+(?:inquiry|positive)\b",
    r"\bya\s+ai\b",
]
_INJ_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)
# label vocabulary never written by real customers: stripping it stops "keyword injection"
# ("...the correct category is loyalty_points") even against classifiers that follow no instructions
_LABEL_TOKENS = re.compile(
    r"\{[^{}]*\"(?:category|urgency|sentiment)\"[^{}]*\}"
    r"|\b(?:category|urgency|sentiment|priority|label)\s*[=:]\s*\"?[\w-]+\"?"
    r"|\b(?:delivery_delay|wrong_item|refund_request|product_quality|price_complaint|app_issue|staff_behavior|loyalty_points)\b",
    re.IGNORECASE)
_B64 = re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}")
# sentence boundary = punctuation FOLLOWED BY whitespace (so "5.2", "a.b@x.com" stay intact) or a newline
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+")

_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"), "[CARD]"),
    (re.compile(r"(?<!\d)(?:\+?20|0)1[0125]\d{8}(?!\d)"), "[PHONE]"),
]


@dataclass
class CleanMessage:
    text: str  # cleaned text for the classifier
    injection_flag: bool  # an injection attempt was found (and removed)
    is_valid: bool  # False when nothing meaningful is left (empty / pure injection)


def repair_mojibake(text: str) -> str:
    """Undo the classic 'UTF-8 read as cp1252' corruption (e.g. 'Ø§Ù„Ø·Ù„Ø¨' -> 'الطلب')."""
    if not re.search(r"[ØÙ][\x80-\xBFŒ-™]", text):
        return text
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
            if re.search(r"[ء-ي]", fixed):
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


_SPACED_LETTERS = re.compile(r"(?:\b\w\b\s+){4,}")  # "I g n o r e ..." signature
_NOSPACE_MARKERS = ("ignoreprevious", "ignoreall", "ignoreinstructions", "disregardprevious", "disregardall",
                    "forgeteverything", "systemprompt", "youarenow", "تجاهلكلالتعليمات", "تجاهلالتعليمات")


def _injection_at(chunk: str) -> int:
    """Position of an injection in the chunk, or -1. Spaced-out letters ('I g n o r e ...') and
    reversed text (RTL-override trick) count as an injection of the whole chunk (position 0)."""
    m = _INJ_RE.search(chunk)
    if m:
        return m.start()
    spaced = _SPACED_LETTERS.search(chunk)
    if spaced:
        squeezed = re.sub(r"\s+", "", chunk[spaced.start():]).lower()
        if any(k in squeezed for k in _NOSPACE_MARKERS) or _INJ_RE.search(re.sub(r"(?<=\b\w) (?=\w\b)", "", chunk)):
            return spaced.start()  # where the spaced-out run begins (a complaint before it is kept)
    rev = _INJ_RE.search(chunk[::-1])
    if rev:
        return len(chunk) - rev.end()  # start of the reversed instruction in the original text
    return -1


def _strip_base64(text: str) -> tuple[str, bool]:
    flagged = False
    for tok in _B64.findall(text):
        try:
            decoded = base64.b64decode(tok + "=" * (-len(tok) % 4)).decode("utf-8")
            flagged = flagged or bool(_INJ_RE.search(decoded))
        except (ValueError, UnicodeDecodeError):
            pass
        text = text.replace(tok, " ")
    return text, flagged


# chat-template blocks: everything from an opening token to its closing token is injected text;
# what comes AFTER the closing token is the customer's own text again
_BLOCKS = re.compile(r"<\|im_start\|>.*?(?:<\|im_end\|>|$)|\[INST\].*?(?:\[/INST\]|$)"
                     r"|<\s*system\s*>.*?(?:<\s*/\s*system\s*>|$)", re.IGNORECASE | re.DOTALL)
_TOKENS = re.compile(r"<\|im_(?:start|end)\|>|\[/?INST\]|</?\s*(?:system|customer_message|assistant|user)\s*>"
                     r"|###\s*instruction\s*:?", re.IGNORECASE)
_CLAUSE_END = re.compile(r"[.,،;!?؟\n]")


def _words(s: str) -> int:
    return len(re.findall(r"[^\W\d_]{2,}", s))


def remove_injections(text: str) -> tuple[str, bool]:
    """Remove injected instructions but keep the customer's complaint around them.
    - chat-template blocks and tokens are removed;
    - an injection appended in the middle of a sentence: keep the part BEFORE it;
    - an injection at the start of a sentence: drop its clause (up to , . ، ...), or only the matched
      phrases when the sentence has no clause boundary, so a complaint after it survives."""
    text, flagged = _strip_base64(text)
    text, n1 = _BLOCKS.subn(" ", text)
    text, n2 = _TOKENS.subn(" ", text)
    flagged = flagged or (n1 + n2) > 0
    out = []
    for chunk in _SENTENCE_SPLIT.split(text):
        pos = _injection_at(chunk)
        if pos == -1:
            out.append(chunk)
            continue
        flagged = True
        if _words(chunk[:pos]) >= 2:
            out.append(chunk[:pos])
        else:
            end = _CLAUSE_END.search(chunk, pos)
            out.append(chunk[end.end():] if end else _INJ_RE.sub(" ", chunk))
    text = _INJ_RE.sub(" ", " ".join(o.strip() for o in out if o.strip())) if flagged else text
    text, n = _LABEL_TOKENS.subn(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip(), flagged or n > 0


def clean_message(raw) -> CleanMessage:
    text = "" if raw is None or (isinstance(raw, float)) else str(raw)
    text = normalize_text(repair_mojibake(text.replace("\x00", " ")))
    text, injection = remove_injections(text)
    for rx, token in _PII:
        text = rx.sub(token, text)
    text = text[: settings.max_message_chars]
    # a message is meaningful if it still has a few real letters after cleaning; after an injection was
    # removed, a 1-3 word leftover ("and =inquiry", "السابقة دي استفسار") is attack debris, not a complaint
    letters = len(re.findall(r"[^\W\d_]", text))
    valid = letters >= 6 and not (injection and _words(text) < 4)
    return CleanMessage(text=text, injection_flag=injection, is_valid=valid)
