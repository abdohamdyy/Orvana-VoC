"""Behavioural test suite (CheckList style). Every case is hand-written and labelled with
docs/LABELING_GUIDE.md. Writes tests/suite/base.csv with a FROZEN dev/holdout split.

    python tests/suite/make_suite.py          # (re)writes base.csv; refuses to change a frozen holdout

Rules for using the suite (see docs/ROBUSTNESS_REPORT.md):
- never used for training, retrieval, prompts or threshold tuning;
- 'dev' rows may drive fixes; 'holdout' rows are evaluated ONCE at the end;
- 'ambiguous' rows (label depends on a convention we had to guess) are reported separately.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CASES = []


def c(text, cat, lang, phen, sec="", prod="", store="", sent=None, urg=None, amb=0, acc=""):
    """One labelled case. Defaults follow the guideline: inquiry -> neutral/1, complaint -> negative/2."""
    CASES.append(dict(message_text=text, category=cat, secondary_category=sec, product_mentioned=prod,
                      store_mentioned=store, sentiment=sent or ("neutral" if cat == "inquiry" else "negative"),
                      urgency=urg or (1 if cat == "inquiry" else 2), lang=lang, phenomenon=phen,
                      ambiguous=amb, acceptable_categories=acc or cat))


P = "paraphrase"
# ---------------------------------------------------------------- delivery_delay
c("بقالي أسبوع مستني الأوردر ومحدش معبرني", "delivery_delay", "egy", P)
c("الطلب كان المفروض يوصل الحد وإحنا النهارده الخميس", "delivery_delay", "egy", P)
c("لم يصلني الطلب حتى الآن رغم مرور خمسة أيام على موعد التسليم", "delivery_delay", "msa", P)
c("el order bta3y lesa mawsalsh mn yom el sabt", "delivery_delay", "franco", P)
c("ana mestany el talab men esboo kamel w mafish rad", "delivery_delay", "franco", P)
c("Still waiting for my delivery, it's been 9 days", "delivery_delay", "en", P)
c("الـ delivery اتأخر جدا والـ tracking مش بيتحرك من يومين", "delivery_delay", "mixed", P)
c("ايوه يا فندم انا طالب اوردر من يوم السبت ولحد دلوقتي ماجاش ولا حد كلمني", "delivery_delay", "voice", P)
c("Where is my order? It said out for delivery yesterday", "delivery_delay", "en", P)
# ---------------------------------------------------------------- wrong_item
c("جالي شاحن غير اللي طلبته خالص", "wrong_item", "egy", P)
c("الطلبية ناقصة كيس كامل وأنا دافع تمنه", "wrong_item", "egy", P)
c("استلمت منتجًا مختلفًا عن الذي طلبته والمذكور في الفاتورة", "wrong_item", "msa", P)
c("el order gali na2es 7agten men el list", "wrong_item", "franco", P)
c("You sent me the wrong size and the wrong color", "wrong_item", "en", P)
c("طلبت Dove شامبو جالي conditioner بداله", "wrong_item", "mixed", P, prod="Dove")
c("ana talabt zeit w gali samna badalo", "wrong_item", "franco", P)
c("الكرتونه وصلت فيها حاجات مش بتاعتي اصلا", "wrong_item", "voice", P)
# ---------------------------------------------------------------- refund_request
c("رجعت الحاجة من عشر أيام وفلوسي لسه ما رجعتش على الكارت", "refund_request", "egy", P)
c("عايز فلوسي اللي اتخصمت على أوردر اتلغى", "refund_request", "egy", P)
c("أرجو استرداد المبلغ المدفوع للطلب الملغي في أقرب وقت", "refund_request", "msa", P)
c("3ayez arga3 felousy 3ala el order elly et3amalo cancel", "refund_request", "franco", P)
c("I returned the Kiriazi fridge two weeks ago and still no refund", "refund_request", "en", P, prod="Kiriazi")
c("Please refund the 350 EGP you charged me twice", "refund_request", "en", P)
c("الـ refund بتاعي متأخر أوي ومحدش بيرد", "refund_request", "mixed", P)
c("المرتجع اتسلم للمندوب من اسبوعين والفلوس ماجتش لحد دلوقتي", "refund_request", "voice", P)
# ---------------------------------------------------------------- product_quality
c("التلاجة كريازي مش بتبرد خالص من أول يوم", "product_quality", "egy", P, prod="Kiriazi")
c("السماعة VoltX صوتها بيقطع كل شوية", "product_quality", "egy", P, prod="VoltX")
c("تعطلت المكواة بعد أسبوع واحد من الاستخدام", "product_quality", "msa", P)
c("el blender Tornado sotoh 3aly w bey2af lewa7do", "product_quality", "franco", P, prod="Tornado")
c("The Persil detergent left white stains on all my clothes", "product_quality", "en", P, prod="Persil")
c("Samsung phone screen started flickering after 3 days", "product_quality", "en", P, prod="Samsung")
c("الـ power bank شاومي مش بيشحن خالص", "product_quality", "mixed", P, prod="Xiaomi")
c("الخلاط اشتغل يومين وبعدين الموتور وقف ومش راضي يدور تاني", "product_quality", "voice", P)
# ---------------------------------------------------------------- price_complaint
c("الخصم اللي في الإعلان مش متطبق في الفرع", "price_complaint", "egy", P)
c("اتحاسبت على زيت Orvana Select بسعر أعلى من اللي على الرف", "price_complaint", "egy", P, prod="Orvana Select")
c("السعر المعروض على الرف يختلف عن السعر المسجل في الفاتورة", "price_complaint", "msa", P)
c("el se3r 3al raf 120 w 7asbouny 150", "price_complaint", "franco", P)
c("Prices in your stores are much higher than other supermarkets", "price_complaint", "en", P)
c("The buy-one-get-one offer wasn't applied at the cashier", "price_complaint", "en", P)
c("الـ discount اللي على الـ shelf مش ظاهر في الـ receipt", "price_complaint", "mixed", P)
c("الأسعار بقت نار كل أسبوع حاجة تغلى من غير سبب", "price_complaint", "voice", P)
# ---------------------------------------------------------------- app_issue
c("الأبلكيشن بيعلق عند صفحة الدفع", "app_issue", "egy", P)
c("مش عارف أغير العنوان من الأبلكيشن خالص", "app_issue", "egy", P)
c("لا يمكنني تسجيل الدخول إلى التطبيق منذ أمس", "app_issue", "msa", P)
c("el app msh bey3mel load khales", "app_issue", "franco", P)
c("The app keeps logging me out every few minutes", "app_issue", "en", P)
c("Checkout button does nothing in the app", "app_issue", "en", P)
c("الـ OTP مش بيوصل على الموبايل علشان أعمل login", "app_issue", "mixed", P)
c("كل ما افتح البرنامج يقفل لوحده ويطلعني بره", "app_issue", "voice", P)
# ---------------------------------------------------------------- staff_behavior
c("الكاشير زعق في وشي علشان سألت على العرض", "staff_behavior", "egy", P)
c("موظف خدمة العملاء قفل السكة في وشي", "staff_behavior", "egy", P)
c("تعامل الموظف معي بأسلوب غير لائق في فرع المعادي", "staff_behavior", "msa", P, store="CAI-02")
c("el security f far3 Tanta mesh mo7taram khales", "staff_behavior", "franco", P, store="DLT-01")
c("The delivery guy was rude and demanded a tip", "staff_behavior", "en", P)
c("Staff at the Nasr City branch ignored me for 20 minutes", "staff_behavior", "en", P, store="CAI-01")
c("الـ supervisor في فرع سموحة كان rude جدا معايا", "staff_behavior", "mixed", P, store="ALX-01")
c("الراجل اللي على الباب اتريق عليا قدام الناس", "staff_behavior", "voice", P)
# ---------------------------------------------------------------- loyalty_points
c("النقط بتاعة الشهر اللي فات ما نزلتش", "loyalty_points", "egy", P)
c("حاولت أستبدل النقط في الكاشير قالولي مش هينفع", "loyalty_points", "egy", P)
c("لم تُضف نقاط أورفانا بلس إلى حسابي بعد عملية الشراء", "loyalty_points", "msa", P)
c("el points bta3ty sab2a esboo w lesa ma nezletsh", "loyalty_points", "franco", P)
c("My Orvana Plus balance dropped to zero without any redemption", "loyalty_points", "en", P)
c("Points from my 900 EGP purchase never showed up", "loyalty_points", "en", P)
c("الـ points بتاعتي اتمسحت من غير ما أستخدمها", "loyalty_points", "mixed", P)
c("انا كنت مجمع نقط كتير وفجأة لقيتها صفر", "loyalty_points", "voice", P)
# ---------------------------------------------------------------- inquiry
c("هو فرع الزمالك بيفتح الساعة كام؟", "inquiry", "egy", P, store="CAI-06")
c("ما هي مواعيد عمل فرع المنصورة يوم الجمعة؟", "inquiry", "msa", P, store="DLT-02")
c("emta el Samsung el gedid hayenzel 3andoko?", "inquiry", "franco", P, prod="Samsung")
c("Do you have Juhayna milk in stock at the Dokki branch?", "inquiry", "en", P, prod="Juhayna", store="GIZ-02")
c("What payment methods do you accept on delivery?", "inquiry", "en", P)
c("ينفع أدفع بالـ Instapay عند الاستلام؟", "inquiry", "mixed", P)
c("لو سمحت عايز اعرف فيه فرع قريب من فيصل", "inquiry", "voice", P, store="GIZ-04", amb=1)
c("إزاي أشترك في Orvana Plus؟", "inquiry", "egy", P, amb=1, acc="inquiry|loyalty_points")

M = "multi_issue"
c("الأوردر متأخر 6 أيام وبرضو الأبلكيشن مش بيفتح", "delivery_delay", "egy", M, sec="app_issue")
c("الكاشير كان قليل الذوق و كمان العرض ما اتطبقش", "staff_behavior", "egy", M, sec="price_complaint")
c("استلمت منتج غلط، وغير كده النقط ما اتضافتش", "wrong_item", "egy", M, sec="loyalty_points")
c("My refund is late. Also the app crashes when I open my orders.", "refund_request", "en", M, sec="app_issue")
c("Order arrived 5 days late plus one item was missing", "delivery_delay", "en", M, sec="wrong_item")
c("el order et2a5ar w bardo el points ma nezletsh", "delivery_delay", "franco", M, sec="loyalty_points")
c("بالإضافة لتأخير الطلب، الموظف اتكلم معايا بطريقة وحشة", "delivery_delay", "egy", M, sec="staff_behavior")
c("التلاجة باظت بعد يومين. والأسعار كمان غليت", "product_quality", "egy", M, sec="price_complaint")
c("الطلب ما وصلش\nوالفلوس اتخصمت مرتين 1500 جنيه", "delivery_delay", "egy", M, sec="refund_request")
c("app crashes, wrong item delivered, and no points added", "app_issue", "en", M, sec="wrong_item")
c("طلبت شامبو دوف جالي برسيل وكمان المندوب كان بيزعق", "wrong_item", "egy", M, sec="staff_behavior", prod="Dove")
c("law sama7t el app bey2fel w kaman el order na2es", "app_issue", "franco", M, sec="wrong_item")
c("the cashier refused to apply my points and also she was rude", "loyalty_points", "en", M, sec="staff_behavior")
c("الزيت منتهي الصلاحية وكمان سعره أغلى من الرف", "product_quality", "egy", M, sec="price_complaint", urg=3)
c("The Molto was expired and also the receipt price was wrong", "product_quality", "en", M, sec="price_complaint", prod="Edita", urg=3)
c("I also ordered milk but the whole order is 4 days late", "delivery_delay", "en", M)
c("عايز أطلب كمان حاجة بس الأوردر الأولاني لسه ما وصلش", "delivery_delay", "egy", M)
c("لقد تأخر الطلب كثيرًا، كما أن التطبيق لا يعمل", "delivery_delay", "msa", M, sec="app_issue")
c("مش لاقي النقط بتاعتي + الكاشير ماطبقش الخصم", "loyalty_points", "egy", M, sec="price_complaint")
c("el refund mesh rage3 w el app kaman wa2ef", "refund_request", "franco", M, sec="app_issue")

E = "entities"
c("السعر غلط في فرع ٦ أكتوبر", "price_complaint", "egy", E, store="GIZ-05")
c("staff at the October branch were rude", "staff_behavior", "en", E, store="GIZ-05")
c("f far3 el Maadi el se3r mesh mazboot", "price_complaint", "franco", E, store="CAI-02")
c("فرع المنصوره ٢ الكاشير زعقلي", "staff_behavior", "egy", E, store="DLT-06")
c("cai-3 store has wrong shelf prices", "price_complaint", "en", E, store="CAI-03")
c("CAI 06 security guard shouted at me", "staff_behavior", "en", E, store="CAI-06")
c("فرع أسيوط الجديدة مش بيقبل النقط", "loyalty_points", "egy", E, store="UPE-05")
c("The Heliopolis store didn't apply the discount", "price_complaint", "en", E, store="CAI-03")
c("في فرع الغردقة اتعاملت بطريقة وحشة جدا", "staff_behavior", "egy", E)
c("فرع مصر الجديده الأسعار فيه غير الفاتورة", "price_complaint", "egy", E, store="CAI-03")
c("Sheikh Zayed branch put the wrong item in my bag", "wrong_item", "en", E, store="GIZ-01")
c("el far3 elly f Sidi Gaber mafihoush points redemption", "loyalty_points", "franco", E, store="ALX-02")
c("موبايل سامسونغ وقف بعد يومين", "product_quality", "egy", E, prod="Samsung")
c("الباور بانك شاومى بيفصل لوحده", "product_quality", "egy", E, prod="Xiaomi")
c("سماعة فولت اكس صوتها واطي جدا", "product_quality", "egy", E, prod="VoltX")
c("خلاط تورنيدو ريحته وحشة وهو شغال", "product_quality", "egy", E, prod="Tornado", amb=1)
c("لبن جهينه جالي مفتوح", "product_quality", "egy", E, prod="Juhayna")
c("بيرسيل الجديد مش بينضف خالص", "product_quality", "egy", E, prod="Persil")
c("تلاجة كريازى صوتها عالي جدا", "product_quality", "egy", E, prod="Kiriazi")
c("Ariel detergent I bought from you smells bad", "product_quality", "en", E)
c("the iPhone charger I bought stopped working", "product_quality", "en", E)
c("طلبت مكواة فريش وموبايل سامسونج والمكواة ما وصلتش", "delivery_delay", "egy", E, prod="Fresh")
c("Is the Kiriazi fridge available in the Mansoura 2 branch?", "inquiry", "en", E, prod="Kiriazi", store="DLT-06")
c("عايز أعرف مواعيد فرع العبور", "inquiry", "egy", E, store="CAI-11")
c("el kashier f far3 Port Said kan bey3amelny wa7esh", "staff_behavior", "franco", E, store="CNL-02")

U = "urgency_edge"
c("دي رابع مرة أشتكي والطلب لسه ما وصلش", "delivery_delay", "egy", U, urg=3)
c("للمرة الثالثة أتواصل معكم بخصوص استرداد المبلغ", "refund_request", "msa", U, urg=3)
c("this is my 4th complaint about the app logging me out", "app_issue", "en", U, urg=3)
c("ريحة حريق طالعة من الشاحن VoltX وهو بيشحن", "product_quality", "egy", U, prod="VoltX", urg=3)
c("الباور بانك شاومي طلع منه دخان", "product_quality", "egy", U, prod="Xiaomi", urg=3)
c("the iron sparks every time I plug it in", "product_quality", "en", U, urg=3)
c("جالي تسمم من لبن جهينة اللي اشتريته امبارح", "product_quality", "egy", U, prod="Juhayna", urg=3)
c("عايز استرد 1000 جنيه اتخصموا مرتين من الكارت", "refund_request", "egy", U, urg=3)
c("I was charged 999 EGP twice, please fix it", "refund_request", "en", U, urg=2)
c("اتخصم مني ألفين جنيه مرتين على نفس الأوردر", "refund_request", "egy", U, urg=3)
c("اتخصم 5000 جنيه مرتين على الفيزا", "refund_request", "egy", U, urg=3)
c("الطلب متأخر 20 يوم", "delivery_delay", "egy", U, urg=2)
c("expired yogurt on the shelf at the Banha branch", "product_quality", "en", U, store="DLT-05", urg=3)
c("el zabady elly 3al raf f far3 Zagazig khlsan salahyto", "product_quality", "franco", U, store="DLT-03", urg=3)
c("انتهت صلاحية زيت Orvana Select قبل موعدها المكتوب", "product_quality", "msa", U, prod="Orvana Select", urg=3)
c("اللبن جهينة طعمه حامض ومش طبيعي", "product_quality", "egy", U, prod="Juhayna", amb=1)
c("كل مرة أكلمكم محدش بيرد والأوردر متأخر", "delivery_delay", "egy", U, amb=1)
c("دي تالت مرة أسأل: فرع شبرا بيفتح امتى؟", "inquiry", "egy", U, store="CAI-05", urg=3, amb=1)

S = "sentiment_edge"
c("شكرا جدا على التأخير الجميل ده 👏👏 أسبوعين بالظبط", "delivery_delay", "egy", S)
c("great job, my order is only two weeks late 🙄", "delivery_delay", "en", S)
c("لو سمحت بعد إذن حضرتك الطلب متأخر شوية", "delivery_delay", "egy", S)
c("Not a big deal but the coupon didn't work in the app", "app_issue", "en", S)
c("برافو عليكم بجد، تالت مرة الأبلكيشن يقفل وأنا بدفع", "app_issue", "egy", S, urg=3)
c("شكرا على الخدمة الممتازة في فرع سموحة", "inquiry", "egy", S, store="ALX-01", sent="positive", amb=1)
c("thank you, the delivery guy was very polite today", "inquiry", "en", S, sent="positive", amb=1)

O = "out_of_scope"
c("السلام عليكم", "inquiry", "egy", O)
c("test test", "inquiry", "en", O)
c("هو كارفور فاتح النهارده؟", "inquiry", "egy", O)
c("عايز أشتغل عندكم، أقدم فين؟", "inquiry", "egy", O)
c("اكسب 5000 جنيه في اليوم من البيت اضغط على اللينك", "inquiry", "egy", O, amb=1)
c("ok", "inquiry", "en", O)

I = "incident"
c("بعد آخر تحديث للأبلكيشن الدفع بالفيزا مش بيكمل والفلوس بتتخصم", "app_issue", "egy", I, amb=1, acc="app_issue|refund_request")
c("since version 5.3 the app crashes on startup", "app_issue", "en", I)
c("el website wa2e3 men embare7 w mesh 3aref atlob", "app_issue", "franco", I)
c("Fawry payment failed in the app but the money was deducted", "app_issue", "en", I, amb=1, acc="app_issue|refund_request")
c("الأبلكيشن الجديد مش بيظهر المنتجات خالص", "app_issue", "egy", I)

N = "style_noise"
c("الطلب فين", "delivery_delay", "egy", N)
c("refund??", "refund_request", "en", N)
c("APP NOT WORKING!!!!!", "app_issue", "en", N)
c("انا عميل من 10 سنين وكنت بحب الخدمة جدا وبشتري كل حاجة منكم بس للأسف المرة دي زيت Orvana Select جالي منتهي الصلاحية", "product_quality", "egy", N, prod="Orvana Select", urg=3)
c("ايوه الو يعني انا كنت طالب حاجات من الابلكيشن والحاجات جت ناقصه حاجتين وانا دافع تمنهم", "wrong_item", "voice", N)
c("😡😡 الكاشير 😡😡 قليل الادب 😡😡", "staff_behavior", "egy", N)
c("plz el order bta3y fen??? 3 ayam!!!", "delivery_delay", "franco", N)
c("Hi, I hope you are doing well. I wanted to let you know that my Dove shampoo arrived with a broken cap and leaked everywhere. Thanks", "product_quality", "en", N, prod="Dove")


# ---------------------------------------------------------------- more coverage for thin slices
# MSA
c("أطلب منكم توضيح سبب تأخر شحنتي التي كان من المفترض أن تصل الأسبوع الماضي", "delivery_delay", "msa", P)
c("تجاوز الطلب موعد التوصيل المحدد بثلاثة أيام دون أي تواصل من جانبكم", "delivery_delay", "msa", P)
c("وصلتني سلعة لا تطابق الطلب، فقد طلبت خلاطًا فوصلتني مكواة", "wrong_item", "msa", P)
c("الطرد ناقص منتجين رغم احتسابهما في الفاتورة", "wrong_item", "msa", P)
c("لم يُرد إليّ ثمن المنتج المرتجع حتى تاريخه", "refund_request", "msa", P)
c("خُصم المبلغ من بطاقتي مرتين، وأطلب استرداد المبلغ المكرر وقدره 800 جنيه", "refund_request", "msa", U, urg=2)
c("توقف جهاز سامسونج عن العمل بعد يومين من شرائه", "product_quality", "msa", P, prod="Samsung")
c("تنبعث رائحة احتراق من الشاحن عند توصيله بالكهرباء", "product_quality", "msa", U, urg=3)
c("تم احتساب سعر أعلى من السعر المعلن في العرض الترويجي", "price_complaint", "msa", P)
c("ارتفعت أسعار المنتجات بشكل غير مبرر خلال الشهر الماضي", "price_complaint", "msa", P)
c("يتوقف التطبيق عن الاستجابة عند محاولة إتمام عملية الدفع", "app_issue", "msa", P)
c("لا تصلني رسالة رمز التحقق عند محاولة الدخول إلى حسابي", "app_issue", "msa", P)
c("تعرضت لمعاملة سيئة من موظف الأمن في فرع التجمع", "staff_behavior", "msa", P, store="CAI-04")
c("رفض الموظف استبدال المنتج دون إبداء أي سبب", "staff_behavior", "msa", P)
c("لا أستطيع استخدام نقاطي عند الشراء من الفرع", "loyalty_points", "msa", P)
c("اختفت نقاط برنامج الولاء من حسابي فجأة", "loyalty_points", "msa", P)
c("هل يتوفر التوصيل إلى محافظة الإسكندرية؟", "inquiry", "msa", P)
c("ما هي سياسة الاسترجاع للأجهزة الكهربائية؟", "inquiry", "msa", P)
c("تأخر الطلب أسبوعًا، بالإضافة إلى أن أحد المنتجات كان تالفًا", "delivery_delay", "msa", M, sec="product_quality")
c("هذه هي المرة الرابعة التي أشتكي فيها من نفس المشكلة في التطبيق", "app_issue", "msa", U, urg=3)
# mixed Arabic/English
c("الـ order اتأخر والـ customer service مش بيرد", "delivery_delay", "mixed", P)
c("عملت return للـ Samsung phone ومستني الـ refund", "refund_request", "mixed", P, prod="Samsung")
c("الـ app بيعمل crash كل ما أفتح الـ cart", "app_issue", "mixed", P)
c("الـ cashier في فرع الدقي was very rude", "staff_behavior", "mixed", E, store="GIZ-02")
c("الـ offer مكتوب 20% بس الـ system عند الكاشير ماطبقهوش", "price_complaint", "mixed", P)
c("جالي item غلط في الـ delivery", "wrong_item", "mixed", P)
c("الـ points بتاعة Orvana Plus مش بتتحسب صح", "loyalty_points", "mixed", P)
c("الـ charger بتاع VoltX بيسخن جامد", "product_quality", "mixed", U, prod="VoltX", urg=3)
c("ممكن أعرف الـ working hours بتاعة فرع ستانلي؟", "inquiry", "mixed", P, store="ALX-06")
c("الـ refund اتعمل بس الـ amount ناقص", "refund_request", "mixed", P)
c("الـ delivery man رمى الشنطة على الباب", "staff_behavior", "mixed", P)
c("الـ Molto كان expired", "product_quality", "mixed", U, prod="Edita", urg=3)
# voice-to-text style (no punctuation, fillers)
c("الو ايوه انا بكلمكم تاني بخصوص الاوردر اللي ما وصلش بقالي خمس ايام مستني", "delivery_delay", "voice", P, amb=1)
c("بص يا باشا انا اشتريت التلاجه من عندكم وهي مش بتبرد خالص وعايز حد يجي يشوفها", "product_quality", "voice", P)
c("يعني ايه الفلوس تتخصم مني وانا ماخدتش حاجه انا عايز فلوسي", "refund_request", "voice", P)
c("الموظفه في الفرع كلمتني بطريقه مش كويسه خالص وانا ست كبيره", "staff_behavior", "voice", P)
c("انا مش عارف ادخل على الحساب بتاعي في البرنامج من امبارح", "app_issue", "voice", P)
c("النقط اللي جمعتها راحت فين محدش فاهم حاجه", "loyalty_points", "voice", P)
c("عايز اسال على العروض اللي عندكم الاسبوع ده", "inquiry", "voice", P)
c("اللي حصل ان الكاشير حاسبني على حاجه انا ما اخدتهاش", "price_complaint", "voice", P, amb=1, acc="price_complaint|refund_request")
c("الواد بتاع الدليفري قالي هجيلك بعد ساعه ومن ساعتها ماجاش", "delivery_delay", "voice", P)
c("ريحه الزيت غريبه وفيه حاجه عايمه فيه", "product_quality", "voice", U, amb=1)
# Franco
c("el delivery et2akhar 3 ayam w mafeesh 7ad by2ool 7aga", "delivery_delay", "franco", P)
c("gali el talaga Kiriazi maksoura", "product_quality", "franco", P, prod="Kiriazi")
c("msh 3aref asta5dem el coupon fel app", "app_issue", "franco", P)
c("el kashier 7asabny ghalat w zawed 50 geneh", "price_complaint", "franco", P)
c("3ayez felousy, el order etlagha mn 10 ayam", "refund_request", "franco", P)
c("el mandoob kan bey3ayatly fel telephone", "staff_behavior", "franco", P)
c("el points mesh bt7sab 3la el moshtarayat", "loyalty_points", "franco", P)
c("far3 Mokattam byfta7 emta?", "inquiry", "franco", P, store="CAI-09")
c("el power bank Xiaomi beyskhan gamed", "product_quality", "franco", U, prod="Xiaomi", urg=3)
c("el app 3amal debit marteen 2000 geneh", "refund_request", "franco", U, urg=3, amb=1, acc="refund_request|app_issue")
c("el order lesa mawsalsh w mafish had bayrod", "delivery_delay", "franco", P)
c("el shampoo Dove gali maftoo7", "product_quality", "franco", P, prod="Dove")
c("msh la2y el order fel history bta3 el app", "app_issue", "franco", P)
c("el se3r fel app gher el se3r fel far3", "price_complaint", "franco", P)
c("law sama7t fe far3 f Assiut?", "inquiry", "franco", P, store="UPE-01", amb=1)
# more multi-issue connectors
c("اتخصم مني مرتين 1200 جنيه وكمان الأبلكيشن بيقفل", "refund_request", "egy", M, sec="app_issue", urg=3)
c("el kashier za3a2 w kaman ma rdeesh ydeeny el points", "staff_behavior", "franco", M, sec="loyalty_points")
c("My Persil arrived opened and also the delivery was 3 days late", "product_quality", "en", M, sec="delivery_delay", prod="Persil")
c("النقط اختفت ومش بس كده الأسعار زادت", "loyalty_points", "egy", M, sec="price_complaint")
c("The app doesn't load; in addition, my last refund never arrived", "app_issue", "en", M, sec="refund_request")
c("الطلب جه ناقص وبرده المندوب اتأخر ساعتين", "wrong_item", "egy", M, sec="delivery_delay")
c("الكاشير ما طبقش العرض وبعدين زعقلي لما اعترضت", "price_complaint", "egy", M, sec="staff_behavior")
# more entities
c("فرع الهرم الكاشير اتخانق معايا", "staff_behavior", "egy", E, store="GIZ-03")
c("ALX5 had expired Juhayna milk on the shelf", "product_quality", "en", E, prod="Juhayna", store="ALX-05", urg=3)
c("فرع ميت غمر السعر على الرف غلط", "price_complaint", "egy", E, store="DLT-09")
c("Mallawi branch didn't accept my points", "loyalty_points", "en", E, store="UPE-06")
c("فرع الاسماعيلية الموظف اتعامل معايا وحش", "staff_behavior", "egy", E, store="CNL-01")
c("زيت اورفانا سيليكت منتهي الصلاحية", "product_quality", "egy", E, prod="Orvana Select", urg=3)
c("شاحن فولتكس باظ بعد يومين", "product_quality", "egy", E, prod="VoltX")
c("مولتو بالشوكولاتة جالي مكسور ومفتوح", "product_quality", "egy", E, prod="Edita")
# more out-of-scope / sentiment
c("صباح الخير", "inquiry", "egy", O)
c("?", "inquiry", "en", O)
c("Can I speak to a manager please?", "inquiry", "en", O)
c("شكرا ليكم المشكلة اتحلت", "inquiry", "egy", S, sent="positive", amb=1)
c("you guys are the worst", "inquiry", "en", S, amb=1, acc="inquiry|staff_behavior")
c("انتو أحسن سوبر ماركت في مصر ❤️", "inquiry", "egy", S, sent="positive", amb=1)


def lang_of(text):
    import re
    if re.search(r"[ء-ي]", text):
        return "mixed" if re.search(r"[A-Za-z]{3,}", text) else "egy"
    return "franco" if re.search(r"[a-z][2357]|[2357][a-z]|\b(?:el|w|men|fel|ana|law|mesh|talabt|gali|bta3)\b", text, re.I) else "en"


def main():
    base = pd.DataFrame(CASES)
    ch = pd.read_csv(HERE.parent / "fixtures" / "challenge_set.csv", dtype=str).fillna("")
    ch["urgency"] = ch.urgency.astype(int)
    ch["lang"] = ch.message_text.map(lang_of)
    ch["phenomenon"] = [M if s else (U if u == 3 else P) for s, u in zip(ch.secondary_category, ch.urgency)]
    ch["ambiguous"], ch["acceptable_categories"] = 0, ch.category
    base = pd.concat([base, ch.drop(columns=["ticket_id"])], ignore_index=True)
    base = base.drop_duplicates("message_text").reset_index(drop=True)
    base.insert(0, "ticket_id", [f"S{i:04d}" for i in range(len(base))])

    # stratified 60/40 split, frozen once written
    rng = np.random.default_rng(2026)
    split = pd.Series("dev", index=base.index)
    for _, g in base.groupby(["lang", "phenomenon"]):
        idx = rng.permutation(g.index.to_numpy())
        split[idx[: int(round(0.4 * len(idx)))]] = "holdout"
    base["split"] = split

    out = HERE / "base.csv"
    if out.exists() and "--force" not in sys.argv:
        old = pd.read_csv(out, dtype=str)
        if not old[old.split == "holdout"].equals(base[base.split == "holdout"].astype(str)):
            sys.exit("refusing to change the frozen holdout (pass --force only before any evaluation)")
    base.to_csv(out, index=False)
    hold = base[base.split == "holdout"].to_csv(index=False).encode("utf-8")
    (HERE / "HOLDOUT_SHA256.txt").write_text(hashlib.sha256(hold).hexdigest() + "\n")
    print(base.groupby(["split"]).size().to_dict(), base.lang.value_counts().to_dict(), base.phenomenon.value_counts().to_dict(),
          "ambiguous:", int(base.ambiguous.sum()))


if __name__ == "__main__":
    main()
