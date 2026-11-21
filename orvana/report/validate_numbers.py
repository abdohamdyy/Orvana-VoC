"""Guarantee that every number in the LLM-written summary comes from the computed facts.

- Arabic-Indic digits (٣٢٠) are converted before checking, so they cannot slip through.
- Numbers written as words (ثلاثة، عشرين، ألف ...) are rejected: they cannot be verified.
- Each number must equal (after rounding to 1 decimal) a number present in the facts,
  or its absolute value (the summary may say "انخفض 5" for a change of -5).
"""
import re

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
NUMBER_WORDS = {
    "واحد", "واحدة", "اثنين", "اتنين", "اثنان", "ثلاث", "ثلاثة", "تلاتة", "اربع", "أربع", "اربعة", "أربعة",
    "خمس", "خمسة", "ستة", "سبعة", "ثمانية", "تمانية", "تسعة", "عشر", "عشرة", "عشرين", "عشرون", "ثلاثين",
    "اربعين", "أربعين", "خمسين", "مية", "مائة", "مئة", "مئات", "الف", "ألف", "آلاف", "نصف", "ربع", "ضعف", "ضعفين",
}


def _numbers_in(obj, out: set[float]) -> set[float]:
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.update({round(float(obj), 1), round(abs(float(obj)), 1)})
    elif isinstance(obj, str):
        for n in _NUM_RE.findall(obj.translate(_AR_DIGITS)):  # dates like 2026-07-01, store codes like CAI-03
            out.add(round(float(n), 1))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _numbers_in(k, out)  # keys such as "urgency_3" also carry numbers
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)
    return out


def check_summary(summary: str, facts: dict) -> list[str]:
    """Return a list of problems (empty list = the summary is safe to send)."""
    problems = []
    allowed = _numbers_in(facts, set())
    text = summary.translate(_AR_DIGITS)
    for n in _NUM_RE.findall(text):
        if round(float(n), 1) not in allowed:
            problems.append(f"number {n} is not in the data")
    for token in re.findall(r"[ء-ي]+", summary):
        base = token[1:] if token[:1] in "وب" and len(token) > 3 else token
        if token in NUMBER_WORDS or base in NUMBER_WORDS:
            problems.append(f"number written as a word: {token}")
    if not summary.strip():
        problems.append("empty summary")
    return problems
