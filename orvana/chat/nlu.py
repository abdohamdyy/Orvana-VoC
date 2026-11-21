"""Understanding one customer message: what the customer wants and which details they gave.

No LLM here. The category comes from the production classifier (process_text); everything else is a
closed set (order ids, stores, brands) or a small dialogue act (ask for a human, thanks, insult ...),
which rules handle more reliably and auditably than a 4B model.
"""
import re
from dataclasses import dataclass, field

from ..extractors import double_charge_amount, find_product, find_store, has_safety_issue, is_repeat_complaint, split_segments
from ..normalize import clean_message, match_key, normalize_text
from ..pipeline import process_text
from ..schema import TicketRecord
from .kb import Fact, concepts, load_kb
from .orders import find_order_ids


def _re(*parts: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)


# all patterns run on match_key() text: lower case, أإآ->ا, ة->ه, ى->ي, repeated letters collapsed
_PERSON = r"موظف|حد|بني ?ادم|انسان|مسول|خدمه العملاء|شخص|مدير"  # مسؤول -> match_key -> مسول
HUMAN_RE = _re(
    rf"(?:اكلم|اتكلم|كلمني|حولني|حولوني|وصلني|اتواصل مع)\s+(?:\S+\s+){{0,2}}?(?:{_PERSON})",
    rf"(?:عايز|عاوز|محتاج|اريد|ابغي)\s+(?:موظف|بني ?ادم|انسان|حد حقيقي|حد من خدمه العملاء)", r"حولني\s+ل?حد",
    r"\b(?:talk|speak|chat)\s+(?:to|with)\s+(?:a\s+|an\s+|the\s+|real\s+|some\s*one\s+)*(?:human|person|agent|representative|someone|manager|customer\s+service)",
    r"\b(?:human|live|real)\s+(?:agent|person)\b", r"\b(?:transfer|connect)\s+me\b", r"^\W*(?:agent|human|representative)\W*$",
    r"\b(?:akalem|aklem|at2alem|atkalem|kalemny|7awelny|7wlny|wasalny)\s+(?:\w+\s+){0,2}?(?:mowazaf|mwzf|7ad|insan|bany\s?adam|modir|modeer|customer\s+service|el\s?5edma|el\s?khedma)",
    r"\b(?:3ayez|3ayz|3awez|3awz)\s+(?:mowazaf|mwzf|insan|bany\s?adam)")
GREET_RE = _re(r"^\W*(?:السلام عليكم|سلام عليكم|اهلا|مرحبا|هاي|هلو|صباح الخير|مساء الخير|ازيك|ازيكم|ازاي حضرتك)",
               r"^\W*(?:hi|hello|hey|salam|ahlan|good\s+(?:morning|evening|afternoon)|ezayak|izayak)\b",
               # small talk anywhere in the message ("Hi there! How are you doing today?")
               r"عامل\s+ايه|اخبارك|(?<![ء-ي])ازيك|\bhow\s+(?:are|r)\s+(?:you|u)\b|\bhow\s+is\s+it\s+going\b|\b(?:3amel\s+eh|akhbarak)\b")
# answers to "is there a safety risk, or is it just faulty / out of warranty?" (asked when the product type
# rarely has the hazard the customer named, e.g. an "expired" phone)
YES_RE = _re(r"^\W*(?:اه|ايوه|ايوا|ايو|اكيد|نعم|طبعا|فعلا)(?![ء-ي])",
             r"^\W*(?:yes|yeah|yep|yup|sure|definitely|aywa|aiwa|akid|aked|tab3an)\b")
NO_RE = _re(r"^\W*(?:لا|لاء|مش|مفيش)(?![ء-ي])", r"^\W*(?:no|nope|nah|not|la|la2|msh|mesh|mafish)\b")
NOT_HAZARD_RE = _re(r"ضمان|بايظ|باظ|بطل|وقف|مش شغال|مبيشتغلش|عطلان|عطل|خربان",
                    r"\b(?:waranty|guarantee|broke|broken|stoped|not working|doesn'?t work|faulty|dead|daman|bawaz|ba2ez|"
                    r"wa2af|batal|3otl|3atlan)\b")
UNSURE_RE = _re(r"مش متاكد|مش عارف|معرفش|مش فاكر|(?<![ء-ي])(?:ممكن|يمكن|جايز|احتمال)(?![ء-ي])",
                r"\b(?:not sure|unsure|maybe|possibly|perhaps|don'?t know|no idea|msh met2aked|mesh met2aked|"
                r"mesh 3aref|msh 3aref|ma3rafsh|momken|yemken|gayez)\b")


def hazard_answer(text: str) -> str | None:
    """'yes' = a real hazard, 'no' = not a hazard (faulty, warranty), 'unsure' (safety first), None = unclear."""
    if has_safety_issue(text):
        return "yes"
    key = match_key(text)
    if UNSURE_RE.search(key):
        return "unsure"
    if NO_RE.search(key) or NOT_HAZARD_RE.search(key):
        return "no"
    return "yes" if YES_RE.search(key) else None


THANKS_RE = _re(r"شكرا|متشكر|تسلم|الف شكر|مشكور", r"\b(?:thanks|thank\s+you|thx|shokran|merci|tslm|teslam)\b")
BYE_RE = _re(r"مع السلامه|\bباي\b", r"\b(?:bye|goodbye|ma3a\s?el\s?salama|ma3 elsalama)\b")
DONT_KNOW_RE = _re(r"مش فاكر|مش عارف|معرفش|مش معايا|ناسي|مش عايز اديك|مش هديك|مش هقول",
                   r"\b(?:don'?t|do\s+not)\s+(?:know|remember|have\s+it)\b|\bno\s+idea\b|\bforgot\b",
                   r"\b(?:msh|mesh)\s+(?:fakr|faker|3aref|3arf)\b|\bma3rafsh\b")
ABUSE_RE = _re(r"يا\s+(?:غبي|حمار|حيوان|زفت|كلب|متخلف|عبيط|اهبل)", r"(?:انت|انتو|انتي)\s+(?:غبي|اغبيا|حمير|حيوان|متخلف)",
               r"\b(?:stupid|idiot|dumb|moron)\b", r"\bf+u+c+k|\bshit\b",
               r"\bya\s+(?:ghaby|8aby|7mar|7omar|7ayawan|kalb|zeft|a7bal)\b")
COMPENSATION_RE = _re(r"تعويض|يعوض|هتعوضو|عوضوني", r"compensat", r"\bt[a3]*3?we+d|ta3wid")
DATA_REQUEST_RE = _re(r"(?:ارقام|رقم|بيانات|اسماء|ايميلات|عناوين|معلومات|تفاصيل|حسابات|حساب)\s+(?:\S+\s+){0,2}?"
                      r"(?:العملاء|عملاء|الناس|عميل اخر|عميل تاني|عملاء تانيين|حد تاني)",
                      r"(?:phone\s+numbers?|data|details|names|e-?mails?|addresses|ids?)\s+(?:\w+\s+){0,2}?(?:other\s+)?customers",
                      r"customer\s+(?:list|ids|data|phone)")
# match_key turns ى into ي, so the patterns use the ي form ("امتي")
QUESTION_RE = _re(r"[؟?]", r"^(?:\S+\s+){0,2}?(?:ايه|اقدر|ينفع)(?![ء-ي])",
                  r"(?<![ء-ي])(?:هل|ازاي|امتي|فين|ليه|كام|بكام|قد ايه|عايز اعرف|عاوز اعرف|محتاج اعرف|ممكن اعرف|نفسي اعرف|"
                  r"ارجو معرفه|اود معرفه|اريد معرفه)(?![ء-ي])",
                  r"^(?:how|when|where|what|why|which|can|could|is|are|do|does)\b",
                  r"\b(?:i\s+want|i'?d\s+like|i\s+would\s+like)\s+to\s+know\b|\bplease\s+tell\s+me\b",
                  r"\b(?:ezay|emta|fen|feen|leh|bkam|ad\s+eh|3ayez\s+a3raf)\b")
# words that mean something went wrong (a complaint), as opposed to a pure policy question
PROBLEM_RE = _re(r"(?<![ء-ي])مش(?![ء-ي])|(?<![ء-ي])ما\s*\S+ش(?![ء-ي])|لسه|اتاخر|متاخر|تاخير|غلط|بايظ|باظ|وحش|مشكل|عطل|اتخصم|"
                 r"سخن|شياط|ريحت|اتعامل|زعق|موصلش|ماوصلش|مارجعتش|منتهي|تالف|مكسور|عيب|حرامي|نصاب|بيقفل|بيفصل|واقع|ناقص|"
                 r"شتم|اهان|بهدل",
                 r"\b(?:not|never|still|late|wrong|broken|problem|issue|charged|rude|missing|damaged|defect\w*|crash\w*|"
                 r"insult\w*|shout\w*|yell\w*|curs\w*|"
                 r"didn'?t|hasn'?t|haven'?t|isn'?t|won'?t|can'?t)\b",
                 r"\b(?:msh|mesh|lessa|lsa|mt2a5ar|et2a5ar|wa7esh|ghalat|bayez|mshkla|moshkela|3eib|3eb|bt2fel|na2es|ma\w+sh)\b")
VAGUE_RE = _re(r"^\W*(?:عندي|في|فيه|حصلت)?\s*(?:مشكله|شكوي|استفسار|سؤال)\W*$",
               r"^\W*i\s+have\s+(?:a|an)\s+(?:problem|complaint|question|issue)\W*$", r"^\W*(?:3andy|3ndy)\s+(?:moshkela|mshkla|so2al)\W*$")
CORRECTION_RE = _re(r"(?<![ء-ي])(?:لا|لاء|مش|قصدي|اقصد|غلطت)(?![ء-ي])", r"\b(?:no|not|i\s+mean|sorry|la2|msh|a2sod)\b")
_FRANCO_MIX = re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*[2357])[a-z0-9]+\b")  # "bt5las", "3ayez", "la2"
_FRANCO_WORDS = set("el ana enta enty mesh msh 3ayez fen feen ezay izay leh keda kda lessa lsa kaman bta3y bta3 delwa2ty "
                    "ma3aya eh momken aywa tamam shokran mafeesh gedan awy ely fe w ya da dy di 7ad".split())


def detect_lang(text: str) -> str | None:
    """'ar' (Arabic script), 'franco' (Arabic in Latin letters) or 'en'; None = keep the session language
    (too short to tell, or mostly another script)."""
    text = re.sub(r"ORV\s*-?\s*\d*|CH-[A-Z]{6}", " ", text, flags=re.IGNORECASE)
    if len(re.findall(r"[^\W\d_]{2,}", text)) < 2:  # "ORV-10003", "رقمه 10002"
        return None
    letters = re.findall(r"[^\W\d_]", text)
    ar = sum("؀" <= ch <= "ۿ" for ch in letters)
    lat = sum("a" <= ch.lower() <= "z" for ch in letters)
    if ar + lat < 0.5 * len(letters):  # e.g. Chinese: not a language we answer in
        return None
    if ar / (ar + lat) >= 0.4:
        return "ar"
    # on lower-cased text WITHOUT letter collapsing: match_key turns "fee" into "fe" (a Franco word)
    low = normalize_text(text).lower()
    tokens = re.findall(r"[a-z0-9']+", low)
    hits = sum(t in _FRANCO_WORDS for t in tokens)
    return "franco" if _FRANCO_MIX.search(low) or (hits and hits >= 0.2 * len(tokens)) else "en"


@dataclass
class Understanding:
    text: str  # cleaned: injection removed, PII masked
    valid: bool
    injection: bool
    lang: str | None
    record: TicketRecord | None = None  # classifier output for the whole message
    segments: list[str] = field(default_factory=list)
    acts: set[str] = field(default_factory=set)  # human, greet, thanks, bye, dont_know, abuse, compensation, data, question, problem, vague, correction
    facts: list[tuple[Fact, float]] = field(default_factory=list)  # policy retrieval
    order_ids: list[str] = field(default_factory=list)
    store: str | None = None
    product: str | None = None
    amount: int | None = None  # double-charge amount
    safety: bool = False
    repeat: bool = False
    concepts: set[str] = field(default_factory=set)  # policy concepts mentioned (kb.concepts)

    def has_evidence(self) -> bool:
        """Something concrete to open a complaint on: a problem word, an entity or a policy concept.
        Without it ("HACKING SYSTEM", "REFRESH TOKEN") we ask instead of filing whatever the classifier says."""
        return bool("problem" in self.acts or self.concepts or self.order_ids or self.product or self.store)

    def content_words(self) -> int:
        """Words left after removing greetings / thanks / goodbyes (is there anything else to act on?)."""
        rest = match_key(self.text)
        for rx in (GREET_RE, THANKS_RE, BYE_RE):
            rest = rx.sub(" ", rest)
        return len(re.findall(r"[^\W\d_]{2,}", rest))


def _last_store(text: str) -> str | None:
    """The LAST branch mentioned wins ('مش المعادي، فرع الدقي' -> Dokki)."""
    for chunk in reversed([c for c in re.split(r"[،,.؛;!؟?\n]|قصدي|i mean", text) if c.strip()]):
        if (code := find_store(chunk)):
            return code
    return find_store(text)


def understand(raw: str, backend) -> Understanding:
    c = clean_message(raw)
    key = match_key(c.text)
    u = Understanding(text=c.text, valid=c.is_valid, injection=c.injection_flag, lang=detect_lang(c.text))
    checks = {"human": HUMAN_RE, "greet": GREET_RE, "thanks": THANKS_RE, "bye": BYE_RE, "dont_know": DONT_KNOW_RE,
              "abuse": ABUSE_RE, "compensation": COMPENSATION_RE, "data": DATA_REQUEST_RE, "question": QUESTION_RE,
              "problem": PROBLEM_RE, "vague": VAGUE_RE, "correction": CORRECTION_RE}
    u.acts = {name for name, rx in checks.items() if rx.search(key)}
    u.safety, u.repeat = has_safety_issue(c.text), is_repeat_complaint(c.text)
    u.amount, u.concepts = double_charge_amount(c.text), concepts(c.text)
    if u.safety or u.repeat or u.amount is not None:  # a problem even when phrased as a question
        u.acts.add("problem")
    if c.injection_flag:
        # after an attack is removed, keep the leftover only if it describes a problem or asks a real question
        # ("الطلب بتاعي اتأخر" is a complaint; "en el category inquiry" is attack debris)
        letters = len(re.findall(r"[^\W\d_]", c.text))
        u.valid = letters >= 6 and ("problem" in u.acts or "question" in u.acts and u.content_words() >= 3)
    # closed-set details first: a bare "ORV-10003" is too short to count as a valid message
    u.order_ids = find_order_ids(c.text)
    u.store, u.product = _last_store(c.text), find_product(c.text)
    if not u.valid:
        return u
    if u.content_words() >= 2:  # a bare order number or branch name needs no classifier call
        u.record = process_text(c.text, backend)  # cleaned text: an attack around a complaint must not void it
    u.segments = split_segments(c.text)
    u.facts = load_kb().search(c.text)
    return u
