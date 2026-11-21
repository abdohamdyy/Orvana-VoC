"""Knowledge base for the chatbot: the CS policy (docs/cs_policy_en.md) as atomic facts.

Each fact has the customer-facing sentence in Egyptian Arabic and English, plus example questions
(Arabic / Franco / English) used for retrieval. "unknown" entries are topics we KNOW we have no data
for (store hours, addresses, prices, stock ...): retrieving one makes the bot say so honestly and hand
over to an agent instead of inventing an answer.

This module is also the single source of the policy lines used by B1 (orvana/replies.py).
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..normalize import match_key


@dataclass(frozen=True)
class Fact:
    id: str
    ar: str
    en: str
    questions: tuple[str, ...] = field(default=())
    known: bool = True  # False = a topic we have no data for

    def text(self, lang: str) -> str:
        return self.en if lang == "en" else self.ar


_F = [
    # 1. Delivery
    Fact("delivery.times",
         "التوصيل في القاهرة والجيزة خلال 48 ساعة من تأكيد الطلب، وباقي المحافظات خلال 4 أيام عمل.",
         "Delivery is within 48 hours of order confirmation in Cairo and Giza, and within 4 working days in other governorates.",
         ("التوصيل بياخد قد ايه", "الطلب بيوصل امتى", "التوصيل للمحافظات بياخد كام يوم", "مدة الشحن كام يوم",
          "el delivery byakhod ad eh", "el order hywsal emta", "how long does shipping usually take",
          "when will my order arrive")),
    Fact("delivery.voucher_50", "لو التأخير أكتر من 3 أيام عن الميعاد: قسيمة شراء 50 جنيه.",
         "If the delay is more than 3 days past the promised date, you get a 50 EGP voucher.",
         ("لو الطلب اتأخر باخد تعويض", "ليا تعويض عن التأخير", "law el order et2a5ar ba5od ta3weed",
          "do i get compensation if delivery is late", "is there a voucher for late delivery")),
    Fact("delivery.voucher_100",
         "لو التأخير أكتر من 7 أيام: قسيمة 100 جنيه، والعميل من حقه يلغي الطلب ويسترد فلوسه كاملة.",
         "If the delay is more than 7 days, you get a 100 EGP voucher and may cancel the order for a full refund.",
         ("ينفع الغي الطلب لو اتأخر اسبوع", "اقدر الغي الاوردر المتاخر", "momken alghy el order el mt2a5ar",
          "the order is more than a week late what do i get")),
    Fact("delivery.one_voucher", "قسيمة واحدة بس لكل طلب.", "One voucher per order only.",
         ("ينفع اخد اكتر من قسيمة", "can i get two vouchers for the same order")),
    # 2. Returns and refunds
    Fact("returns.window", "الإرجاع خلال 14 يوم من الاستلام والمنتج بحالته الأصلية.",
         "Returns are accepted within 14 days of delivery, with the product in its original condition.",
         ("اقدر ارجع المنتج", "مدة الاسترجاع كام يوم", "سياسة الارجاع ايه", "momken arga3 el montag",
          "what is the return policy", "how many days do i have to return an item")),
    Fact("returns.electronics",
         "الأجهزة الكهربائية: إرجاع خلال 7 أيام لو فيه عيب صناعة، وبعدها الضمان سنة عند الوكيل المعتمد.",
         "Electronics: return within 7 days for a manufacturing defect; after that, the 1-year warranty is handled by the authorised dealer.",
         ("الخلاط فيه عيب اعمل ايه", "الضمان على الاجهزة قد ايه", "ارجاع جهاز كهربائي", "el daman kam sana",
          "return a defective charger", "how long is the guarantee on appliances")),
    Fact("returns.food", "الأكل والمنظفات المفتوحة ما بترجعش إلا لو منتهية الصلاحية أو فيها عيب.",
         "Opened food and detergents cannot be returned unless expired or defective.",
         ("ينفع ارجع اكل مفتوح", "ارجاع المنظفات بعد الفتح", "can i return opened food",
          "return an opened detergent")),
    Fact("refund.timing",
         "استرداد الفلوس على الكارت خلال 5 أيام عمل، وعلى المحفظة أو الدفع عند الاستلام خلال 10 أيام عمل.",
         "Card refunds take up to 5 working days; wallet or cash-on-delivery refunds take up to 10 working days.",
         ("الفلوس بترجع امتى", "الاسترداد بياخد قد ايه", "الريفاند بيرجع على الكارت امتى", "el felos btrga3 emta",
          "how long does a refund take", "when will i get my refund")),
    Fact("refund.double_charge", "الخصم المكرر بيرجع خلال 3 أيام عمل بعد التأكيد مع البنك.",
         "A double charge is refunded within 3 working days after confirmation with the bank.",
         ("الفلوس اللي اتسحبت زيادة بترجع", "خصم مكرر", "et5asam marteen", "i was charged twice when do i get it back")),
    # 3. Prices and promotions
    Fact("prices.shelf", "لو سعر الرف أقل من سعر الفاتورة، العميل بيدفع السعر الأقل.",
         "If the shelf price is lower than the receipt price, you pay the lower price.",
         ("السعر على الرف غير الفاتورة", "اتحاسبت باكتر من سعر الرف", "el se3r 3al raf a2al",
          "the shelf price was lower than what i paid")),
    Fact("prices.promos", "العروض بتتطبق أوتوماتيك على الكاشير وفي الأبلكيشن خلال فترة العرض بس.",
         "Promotions are applied automatically at checkout and in the app, during the promotion period only.",
         ("العرض ما اتطبقش", "الخصم بيتحسب ازاي", "el 3ard ma et6ba2sh", "how do promotions work",
          "the discount was not applied")),
    # 4. Orvana Plus
    Fact("loyalty.earn", "نقطة لكل 10 جنيه، والنقط بتظهر في الحساب خلال 72 ساعة من الشراء.",
         "You earn 1 point per 10 EGP, and points appear in your account within 72 hours of purchase.",
         ("النقط بتنزل امتى", "النقط ما ظهرتش", "بكسب كام نقطة", "orvana plus بيشتغل ازاي", "el points btnzl emta",
          "when do my points show up", "how many points do i earn", "how does orvana plus work")),
    Fact("loyalty.value", "النقط صالحة 12 شهر، و100 نقطة = خصم 10 جنيه.",
         "Points are valid for 12 months, and 100 points = a 10 EGP discount.",
         ("صلاحية النقط", "النقطة بكام", "el no2at bt2a sal7a ad eh", "do points expire", "how much are points worth")),
    # 5. Compensation and escalation
    Fact("escalation.safety",
         "أي شكوى عن منتج بيسخن أو ريحته شياط أو أكل منتهي الصلاحية بتتصعد لفريق الجودة خلال ساعتين والمنتج بيتسحب من الفرع.",
         "Any complaint about a product overheating, smelling burnt, or expired food is escalated to the Quality team within 2 hours, and the product is pulled from the store."),
    Fact("escalation.staff", "شكاوى سلوك الموظفين بتتصعد لمدير الفرع خلال 24 ساعة.",
         "Staff behaviour complaints are escalated to the store manager within 24 hours.",
         ("اشتكي من موظف ازاي", "how do i complain about a staff member")),
    Fact("escalation.first_reply", "أول رد على أي شكوى بيكون خلال 24 ساعة.",
         "Every complaint gets a first reply within 24 hours.",
         ("هتردوا عليا امتى", "الشكوى بتاخد قد ايه", "when will you reply to my complaint")),
    Fact("compensation.vouchers", "التعويض عن طريق الشات بيكون قسايم شراء بس، حسب قواعد السياسة.",
         "Compensation through chat is by vouchers only, within the policy rules.",
         ("عايز تعويض", "هتعوضوني بكام", "ta3weed", "i want compensation", "what compensation do i get")),
    # commitments used by B1 replies (not separate policy bullets)
    Fact("app.followup",
         "فريقنا التقني بيتابع المشكلة، ولو اتخصمت فلوس من غير ما الطلب يكمل بتترجع حسب مواعيد الاسترداد.",
         "Our technical team is following up, and if money was deducted without the order completing, it is refunded within the refund timelines."),
    Fact("wrong_item.replace", "هنستبدل المنتج الغلط أو نكمل الناقص، والإرجاع خلال 14 يوم من الاستلام.",
         "We will replace the wrong item or complete the missing one; returns are accepted within 14 days of delivery."),
    # Topics we have NO data for: the bot must say so, never guess
    Fact("unknown.store_hours", "", "", ("مواعيد الفرع ايه", "الفرع بيفتح الساعة كام", "بتقفلوا امتى", "الفرع فاتح الجمعة",
         "mawa3eed el far3 eh", "el far3 byefta7 emta", "what are the store opening hours", "what time do you close"), False),
    Fact("unknown.store_address", "", "", ("عنوان الفرع فين", "الفرع فين بالظبط", "اقرب فرع ليا فين", "3enwan el far3",
         "where is the branch", "what is the address of the store"), False),
    Fact("unknown.price_stock", "", "", ("سعر الخلاط كام", "المنتج ده متوفر", "عندكم شاحن", "بكام اللبن", "el charger bkam",
         "how much is", "is it in stock", "do you have this product available"), False),
    Fact("unknown.contact", "", "", ("رقم خدمة العملاء كام", "الخط الساخن", "ايميل الشركة", "ra2am el khedma",
         "customer service phone number", "hotline number"), False),
    Fact("unknown.delivery_fee", "", "", ("التوصيل عليه رسوم", "التوصيل بفلوس", "el shahn bkam", "how much is delivery fee",
         "is shipping free"), False),
    Fact("unknown.payment_methods", "", "", ("بتقبلوا فيزا", "ينفع ادفع بانستاباي", "فوري متاح", "payment methods",
         "can i pay with a credit card", "momken adfa3 visa"), False),
    Fact("unknown.jobs", "", "", ("عايز اشتغل عندكم", "فيه وظايف", "jobs", "are you hiring", "3ayez shoghl"), False),
]
FACTS: dict[str, Fact] = {f.id: f for f in _F}

# category -> policy facts (B1 replies use these lines; delivery_delay picks its voucher line in code)
CATEGORY_FACTS = {
    "delivery_delay": ["delivery.times", "delivery.voucher_50", "delivery.voucher_100", "delivery.one_voucher"],
    "refund_request": ["refund.timing", "refund.double_charge"],
    "product_quality": ["returns.window", "returns.electronics", "returns.food"],
    "price_complaint": ["prices.shelf", "prices.promos"],
    "loyalty_points": ["loyalty.earn", "loyalty.value"],
    "staff_behavior": ["escalation.staff"],
    "app_issue": ["app.followup"],
    "wrong_item": ["wrong_item.replace"],
    "inquiry": [],
}


# question/function words in English, Franco and Egyptian/MSA Arabic (match_key form). Without this,
# "How long do points take to appear?" matched "how long does a refund take" on "how long ... take".
_STOP = set("""a an the my i me you your we our it its this that is are was were be been do does did how long what
when where why which who can could will would should to of for in on at by with from about as and or if there
they them get got take takes much many any some please also just still yet ever so
el eh ana enta enty 3ayez 3ayz 3awez a3raf a3rf fe fi 3ala 3la men mn law lw kam ad momken leh emta ezay fen feen
ely ele da dy di w ya bta3y bta3 lehad l7d la7ad howa hwa heya kda keda
ايه امتي ازاي هل في علي من عن بعد قد كام لو انا ده دي عايز عاوز اعرف ممكن بتاعي بتاعتي ليه لحد اللي الي و يا هو
هي انت حضرتك ما متي ماذا كيف اين لماذا سيتم عند مع كل بس كده اعمل نعمل a3mel ne3mel
orvana اورفانا work works working help hi hello hey there doing today thanks thank please salam ahlan
اهلا مرحبا السلام عليكم شكرا""".split())  # company name, generic verbs, greetings: no topical information


# Policy concepts in English / Franco / Egyptian-MSA Arabic (whole words, match_key form). Character
# n-grams catch spelling variants but cannot bridge synonyms across languages ("bt5las" = "expire" =
# "بتنتهي"); the concept overlap does. Each fact declares its own concepts in FACT_CONCEPTS.
_CONCEPTS = {
    "expire": r"expir\w*|valid\w*|bt5las|btkhlas|te5las|btentehy|tentehy|sal7a|sala7\w*|صالح\w*|صلاحي\w*|\w*نتهي\w*|بتخلص|تخلص",
    "points": r"points?|no2at|no2ta|نقط\w*|نقاط|plus",
    "appear": r"appear\w*|show\w*|nzl\w*|btnzl|tenzel|\w*ظهر\w*|\w*نزل\w*",
    "deliver": r"deliver\w*|shipping|ship|arriv\w*|ywsal|yewsal|hywsal|twsel|tawsel|shahn|\w*شحن|\w*توصيل|\w*يوصل",
    "fee": r"fees?|cost\w*|مصاريف|رسوم|تكلف\w*",
    "price": r"prices?|se3r|bkam|\w*سعر\w*|بكام|how\s?much",
    "shelf": r"shelf|raf|رف|الرف|receipt|invoice|فاتور\w*|\w*فاتوره",
    "promo": r"promo\w*|offers?|discount\w*|3ard|3orood|خصم|عرض|العرض|عروض|اوكازيون",
    "refund": r"refund\w*|money|felos|flos|btrga3|\w*رجع\w*|استرداد|فلوس\w*",
    "return": r"return\w*|arga3|rag3\w*|ارجع|\w*رجاع|استرجاع|مرتجع",
    "food": r"food|opened|fata7to|fata7t|فتحت\w*|مفتوح\w*|اكل|منظف\w*|detergents?|persil|برسيل",
    "warranty": r"warrant\w*|guarant\w*|daman|\w*ضمان",
    "appliance": r"blender|خلاط|khalat|5alat|charger|شاحن|sha7en|مكواه|iron|تلاجه|tallaga|fridge|refrigerator|موبايل|mobile|"
                 r"powerbank|باور|سماعه|earbuds|headphones?|appliances?|electronics?|electric\w*|جهاز|اجهزه|كهرب\w*|\w*لكترون\w*|"
                 r"voltx|samsung|xiaomi|tornado|fresh|kiriazi|سامسونج|شاومي|تورنادو|فريش|كريازي",
    "defect": r"defect\w*|faulty|broken|3eib|3eb|عيب|بايظ|bayez|باظ|عطلان|مكسور|تالف",
    "cancel": r"cancel\w*|alghy|elgha2|\w*لغي|الغاء",
    "double": r"twice|double\w*|marteen|مرتين|مكرر\w*|\w*اتسحب\w*",
    "comp": r"compensat\w*|ta3weed|vouchers?|تعويض|\w*عوض\w*|قسيم\w*",
    "staff": r"staff|employee\w*|rude|mowazaf|\w*موظف\w*|كاشير|kashier|cashier",
    "reply": r"repl\w*|respon\w*|هتردوا|يردوا|ترد|الرد|ردكم|\w*الرد",
    "hours": r"hours|opening|close[sd]?|closing|byefta7|byfta7|fate7|fat7|ya2fel|byafel|مواعيد|\w*فتح|\w*قفل\w*|فاتح",
    "address": r"address\w*|location|located|3enwan|عنوان\w*",
    "stock": r"stock|availab\w*|mawgood|متوفر\w*|موجود\w*",
    "phone": r"phone|hotline|contact|email|telephone|تليفون|ساخن|ايميل|ra2am|رقم|خط",
    "pay": r"pay\w*|visa|instapay|fawry|apple|فيزا|انستاباي|فوري|ادفع|adfa3",
    "jobs": r"jobs?|hiring|vacanc\w*|shoghl|wazifa|وظاي\w*|وظيف\w*|شغل|اشتغل",
}


def _collapse(s: str) -> str:
    """Same letter-collapsing as match_key ('appear' -> 'apear'), applied to the lexicon itself."""
    return re.sub(r"([^\W\d_])\1+", r"\1", s)


_CONCEPT_RE = {tag: re.compile("^(?:" + "|".join(_collapse(a) for a in rx.split("|")) + ")$") for tag, rx in _CONCEPTS.items()}
_STOP = {_collapse(w) for w in _STOP}
_ARTICLE = re.compile(r"^(?:وال|بال|فال|كال|لل|ال)(?=\S{2})")
FACT_CONCEPTS = {
    "delivery.times": {"deliver"}, "delivery.voucher_50": {"deliver", "comp"},
    "delivery.voucher_100": {"deliver", "comp", "cancel"}, "delivery.one_voucher": {"comp"},
    "returns.window": {"return"}, "returns.electronics": {"return", "warranty", "appliance", "defect"},
    "returns.food": {"return", "food"},
    "refund.timing": {"refund"}, "refund.double_charge": {"refund", "double"},
    "prices.shelf": {"price", "shelf"}, "prices.promos": {"promo"},
    "loyalty.earn": {"points", "appear"}, "loyalty.value": {"points", "expire"},
    "escalation.staff": {"staff"}, "escalation.first_reply": {"reply"}, "compensation.vouchers": {"comp"},
    "unknown.store_hours": {"hours"}, "unknown.store_address": {"address"}, "unknown.price_stock": {"price", "stock"},
    "unknown.contact": {"phone"}, "unknown.delivery_fee": {"deliver", "fee"}, "unknown.payment_methods": {"pay"},
    "unknown.jobs": {"jobs"},
}


def content_key(text: str) -> str:
    words = [w for w in re.findall(r"[^\W_]+", match_key(text)) if w not in _STOP]
    return " ".join(words) or match_key(text)


def concepts(text: str) -> set[str]:
    words = re.findall(r"[^\W_]+", match_key(text))
    joined = " ".join(words)
    forms = words + [_ARTICLE.sub("", w) for w in words]  # "النقط" -> "نقط"
    found = {tag for tag, rx in _CONCEPT_RE.items() if any(rx.match(w) for w in forms)}
    if re.search(r"how much", joined):
        found.add("price")
    return found


class KnowledgeBase:
    """Character n-gram retrieval (robust to Arabic/Franco spelling) over facts + example questions,
    on content words only."""

    def __init__(self, facts: list[Fact]):
        self.docs, self.owner = [], []
        for f in facts:
            for d in (f.ar, f.en, *f.questions):
                if d:
                    self.docs.append(content_key(d))
                    self.owner.append(f.id)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.docs)

    def search(self, text: str, k: int = 3) -> list[tuple[Fact, float]]:
        """Best facts for the text. Score = lexical similarity of the fact's closest document (fact text or
        example question), averaged with the Jaccard overlap of concepts when the text has any."""
        sims = (self.matrix @ self.vec.transform([content_key(text)]).T).toarray().ravel()
        lexical: dict[str, float] = {}
        for i, fid in enumerate(self.owner):
            lexical[fid] = max(lexical.get(fid, 0.0), float(sims[i]))
        q = concepts(text)
        scores = {}
        for fid, lex in lexical.items():
            fc = FACT_CONCEPTS.get(fid, set())
            scores[fid] = 0.5 * (lex + len(q & fc) / len(q | fc)) if q else lex
        best = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(FACTS[fid], s) for fid, s in best]


@lru_cache(maxsize=1)
def load_kb() -> KnowledgeBase:
    return KnowledgeBase(_F)
