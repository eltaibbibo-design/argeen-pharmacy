import os
import re
import json
import difflib
import unicodedata
from functools import lru_cache

from flask import Flask, render_template, request, jsonify

try:
    from dotenv import load_dotenv
    # اقرأ متغيرات البيئة محلياً؛ في الاستضافة تُضبط من لوحة الخدمة.
    _env_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env.txt")
    if os.path.exists(_env_txt):
        load_dotenv(_env_txt, override=True)
    else:
        load_dotenv()
except Exception:
    pass

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "Argeen_pharmacy_intents.json")
MEDICINES_FILE = os.path.join(BASE_DIR, "medicines_intents_ready.json")

PHARMACY_NAME = "صيدلية أرقين"
ADDRESS = "عطبرة – الحي الشرقي – مربع 8"
OPENING_HOURS = "24 ساعة"
PRICE_NOTE = "الأسعار أدناه حسب قائمة الأسعار المرفقة، وقد تتغير ويجب تأكيد السعر والتوفر قبل الطلب."

# -----------------------------------------------------------------------------
# AI / NLP SETTINGS
# -----------------------------------------------------------------------------
# يعمل النظام محلياً أولاً، ثم يستخدم OpenAI فقط عندما لا يجد جواباً محلياً.
# لا تضع مفتاح OpenAI داخل الكود أو GitHub؛ استخدم OPENAI_API_KEY في بيئة التشغيل.
AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}")
        return {}


@lru_cache(maxsize=4)
def load_intents_file():
    return load_json(INTENTS_FILE)


@lru_cache(maxsize=4)
def load_medicines_file():
    return load_json(MEDICINES_FILE)


def normalize_text(text):
    """Arabic/English normalization for robust matching."""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text).lower()

    # Arabic tashkeel + tatweel
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("ـ", "")

    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ؤ": "و", "ئ": "ي",
        "پ": "ب", "ڤ": "ف", "گ": "ك", "چ": "ج",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)

    # Arabic/English punctuation -> spaces
    text = re.sub(r"[^\w\s\u0600-\u06FF.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_medicine(text):
    text = normalize_text(text)
    # Remove common Arabic prefixes when they are attached to medicine names.
    words = text.split()
    cleaned = []
    for word in words:
        original = word
        if len(word) > 3 and word.startswith("بال"):
            word = word[2:]
        if len(word) > 3 and word.startswith("لل"):
            word = word[2:]
        if len(word) > 3 and word.startswith("ال"):
            word = word[2:]
        if len(word) > 3 and word.startswith("و"):
            candidate = word[1:]
            if candidate:
                word = candidate
        cleaned.append(word or original)
    return " ".join(cleaned)


def tokens(text):
    return set(normalize_text(text).split())


ALIASES = {
    "بنادول": ["panadol", "بنادول", "البنادول", "بالبنادول", "وبنادول", "فبنادول"],
    "بروفين": ["brufen", "بروفين", "بورفين", "بروفن", "ابروفين", "ايبوبروفين", "ibuprofen"],
}


def value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "، ".join(str(x) for x in value if x is not None)
    if isinstance(value, dict):
        return "، ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def get_medicines():
    """Read ALL medicine records, including top-level 'intents'."""
    data = load_medicines_file()
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    for key in ("medicines", "data", "items", "intents"):
        value = data.get(key)
        if isinstance(value, list):
            # medicines_intents_ready.json stores medicine records under intents.
            if key != "intents" or any(isinstance(x, dict) and "medicine_name" in x for x in value):
                return value
    return []


def get_intents():
    data = load_intents_file()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("intents", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def get_medicine_name(item):
    if not isinstance(item, dict):
        return ""
    for key in ("medicine_name", "medicine", "name", "name_ar", "drug_name"):
        if item.get(key):
            return value_to_text(item[key])
    return ""


def get_active_ingredient(item):
    if not isinstance(item, dict):
        return ""
    for key in ("active_ingredient", "activeIngredient", "ingredient", "generic_name"):
        if item.get(key):
            return value_to_text(item[key])
    return ""


def get_price(item):
    if not isinstance(item, dict):
        return None
    for key in ("retail_price_sdg", "price_sdg", "retail_price", "price", "سعر"):
        if key in item and item[key] not in (None, ""):
            value = item[key]
            if isinstance(value, dict):
                # Prefer common strip/pack values.
                for sub in ("strip_10_sdg", "strip_15_sdg", "sdg", "value"):
                    if sub in value:
                        return value[sub]
                return value
            return value
    return None


def get_indications(item):
    if not isinstance(item, dict):
        return ""
    for key in ("indications", "uses", "usage", "دواعي_الاستعمال"):
        if item.get(key):
            return value_to_text(item[key])
    return ""


def get_alternatives(item):
    if not isinstance(item, dict):
        return []
    for key in ("possible_alternatives_from_same_file", "alternatives", "بدائل"):
        value = item.get(key)
        if value:
            if isinstance(value, list):
                return [value_to_text(x) for x in value]
            return [value_to_text(value)]
    return []


def get_unit(item):
    if not isinstance(item, dict):
        return ""
    return value_to_text(item.get("unit", ""))


def medicine_alias_match(query, name):
    q = normalize_medicine(query)
    n = normalize_medicine(name)
    if not q or not n:
        return False
    for alias_group in ALIASES.values():
        q_alias = any(normalize_medicine(a) == q for a in alias_group)
        n_alias = any(normalize_medicine(a) == n for a in alias_group)
        if q_alias and n_alias:
            return True
    return False


def medicine_score(query, item):
    q = normalize_medicine(query)
    name = normalize_medicine(get_medicine_name(item))
    active = normalize_medicine(get_active_ingredient(item))
    if not q or not name:
        return 0.0

    if q == name:
        return 1.0
    if medicine_alias_match(query, get_medicine_name(item)):
        return 0.99
    if q in name or name in q:
        return 0.92

    qt = tokens(q)
    nt = tokens(name)
    score = 0.0
    if qt and nt:
        score = max(score, len(qt & nt) / max(1, len(qt | nt)))

    # Compare full strings and token order.
    score = max(score, difflib.SequenceMatcher(None, q, name).ratio())

    if active and (q in active or active in q):
        score = max(score, 0.90)

    return score


def search_medicine(query, threshold=0.52):
    medicines = get_medicines()
    if not medicines:
        return None

    # Exact / alias first.
    q = normalize_medicine(query)
    for item in medicines:
        name = get_medicine_name(item)
        if q == normalize_medicine(name) or medicine_alias_match(query, name):
            return item

    best = None
    best_score = 0.0
    for item in medicines:
        score = medicine_score(query, item)
        if score > best_score:
            best_score = score
            best = item

    return best if best_score >= threshold else None


def search_by_active_ingredient(query):
    q = normalize_medicine(query)
    best, best_score = None, 0.0
    for item in get_medicines():
        active = normalize_medicine(get_active_ingredient(item))
        if not active:
            continue
        if q in active or active in q:
            return item
        score = difflib.SequenceMatcher(None, q, active).ratio()
        if score > best_score:
            best_score, best = score, item
    return best if best_score >= 0.72 else None


def format_price(price):
    if price is None:
        return "غير مسجل"
    if isinstance(price, dict):
        parts = []
        for k, v in price.items():
            parts.append(f"{k}: {v} جنيه سوداني")
        return "، ".join(parts)
    try:
        number = float(str(price).replace(",", ""))
        if number.is_integer():
            return f"{int(number):,} جنيه سوداني"
        return f"{number:,.2f} جنيه سوداني"
    except Exception:
        return f"{price} جنيه سوداني"


# -----------------------------------------------------------------------------
# Intent / NLP classifier
# -----------------------------------------------------------------------------

def has_any(q, words):
    q = normalize_text(q)
    return any(w in q for w in words)


def is_greeting(q):
    qn = normalize_text(q)
    greetings = [
        "السلام عليكم", "سلام عليكم", "السلام عليكم ورحمة الله وبركاته",
        "سلام", "مرحبا", "مرحبا بك", "اهلا", "اهلا وسهلا", "هلا", "يا هلا",
        "هاي", "hello", "hi", "hey"
    ]
    return any(qn == normalize_text(x) or qn.startswith(normalize_text(x) + " ") for x in greetings)


def is_price_question(q):
    return has_any(q, ["سعر", "بكم", "كم سعر", "التجزئة", "تكلف", "price", "cost"])


def is_indication_question(q):
    return has_any(q, ["استخدام", "استعمال", "دواعي", "يستخدم في", "فايد", "فائدة", "شنو بعالج", "what is it used", "uses"])


def is_alternative_question(q):
    return has_any(q, ["بديل", "بدائل", "مشابه", "بديل لي", "alternative", "substitute"])


def is_active_question(q):
    return has_any(q, ["المادة الفعالة", "الماده الفعاله", "active ingredient", "مكون", "مم يتكون"])


def is_side_effect_question(q):
    return has_any(q, ["اعراض جانبية", "اثار جانبية", "آثار جانبية", "اعراضه", "side effect", "adverse"])


def is_interaction_question(q):
    return has_any(q, ["تداخل", "يتعارض", "مع دواء", "التداخلات", "interaction", "interact"])


def is_pregnancy_question(q):
    return has_any(q, ["حامل", "الحمل", "الرضاعة", "مرضع", "pregnan", "breastfeed"])


def is_children_question(q):
    return has_any(q, ["طفل", "اطفال", "أطفال", "رضيع", "الاطفال", "للطفل", "للأطفال", "للصغار", "صغير", "صغار", "بيبي", "البيبي", "ينفع للاطفال", "مناسب للاطفال", "مناسب للطفل", "للولد", "للبنت", "عمر الطفل", "children", "child", "baby", "kids"])


def is_missed_dose_question(q):
    return has_any(q, ["نسيت الجرعة", "نسيت جرع", "جرعة فاتت", "missed dose"])


def is_storage_question(q):
    return has_any(q, ["حفظ", "تخزين", "درجة حرارة", "يحفظ", "storage", "store"])


def is_prescription_question(q):
    return has_any(q, ["روشتة", "وصفة", "وصفه", "prescription", "صرف"])


def is_contraindication_question(q):
    return has_any(q, ["موانع", "ممنوع", "ما ينفع", "لا يستخدم", "contraindication"])


def is_allergy_question(q):
    return has_any(q, ["حساسية", "حساس", "allergy", "allergic"])


def is_expiry_question(q):
    return has_any(q, ["انتهاء", "منتهي", "الصلاحية", "expir", "expiry"])


def is_emergency_question(q):
    return has_any(q, ["طوارئ", "طارئ", "ضيق نفس شديد", "تورم شديد", "اغماء", "نزيف", "emergency"])


def classify_intent(q):
    checks = [
        ("emergency", is_emergency_question),
        ("price", is_price_question),
        ("alternative", is_alternative_question),
        ("indications", is_indication_question),
        ("active_ingredient", is_active_question),
        ("side_effects", is_side_effect_question),
        ("interactions", is_interaction_question),
        ("pregnancy", is_pregnancy_question),
        ("children", is_children_question),
        ("missed_dose", is_missed_dose_question),
        ("storage", is_storage_question),
        ("prescription", is_prescription_question),
        ("contraindications", is_contraindication_question),
        ("allergy", is_allergy_question),
        ("expiry", is_expiry_question),
    ]
    for name, fn in checks:
        if fn(q):
            return name
    return "general"


def extract_medicine_from_question(q):
    """Find the medicine whose name/alias is mentioned anywhere in the question."""
    medicines = get_medicines()
    nq = normalize_text(q)

    # Exact names first, longest first.
    ordered = sorted(medicines, key=lambda x: len(normalize_text(get_medicine_name(x))), reverse=True)
    for item in ordered:
        name = get_medicine_name(item)
        nn = normalize_text(name)
        if nn and nn in nq:
            return item

    # Alias groups. For generic names such as Brufen/Ibuprofen, prefer
    # a medicine whose active ingredient matches the alias instead of a
    # potentially wrong fuzzy brand-name match.
    alias_to_active = {
        "بروفين": ["ibuprofen", "ايبوبروفين"],
        "بنادول": ["paracetamol", "acetaminophen", "باراسيتامول"],
    }
    for canonical, aliases in ALIASES.items():
        if any(normalize_text(alias) in nq for alias in aliases):
            for active_alias in alias_to_active.get(canonical, []):
                found = search_by_active_ingredient(active_alias)
                if found:
                    return found
            found = search_medicine(canonical, threshold=0.45)
            if found:
                return found

    # Remove common question words and try fuzzy chunks.
    qwords = normalize_medicine(q).split()
    stop = {
        "شنو", "ما", "هو", "هي", "سعر", "بكم", "كم", "دواء", "الدواء", "ممكن", "عايز",
        "اريد", "أريد", "محتاج", "عندي", "هل", "من", "في", "الصيدليه", "الصيدلية",
        "استخدام", "استعمال", "بديل", "بدائل", "الماده", "المادة", "الفعاله", "الفعالة",
        "what", "is", "the", "price", "of", "for", "medicine", "drug", "how", "much",
        "can", "you", "tell", "me", "used", "use", "alternative"
    }
    candidates = [w for w in qwords if w not in stop and len(w) >= 3]
    if not candidates:
        return None
    return search_medicine(" ".join(candidates), threshold=0.55)


# -----------------------------------------------------------------------------
# Responses
# -----------------------------------------------------------------------------

def medicine_information(item, intent):
    name = get_medicine_name(item) or "الدواء"
    active = get_active_ingredient(item)
    price = get_price(item)
    indications = get_indications(item)
    alternatives = get_alternatives(item)
    unit = get_unit(item)

    if intent == "price":
        return f"💊 {name}\n💰 سعر التجزئة المسجل: {format_price(price)}\n{PRICE_NOTE}"

    if intent == "indications":
        return f"💊 {name}\n📌 دواعي الاستعمال: {indications or 'غير مسجلة في البيانات المتاحة.'}\n\n⚠️ لا تستخدم الدواء بناءً على الدردشة وحدها، خصوصاً المضادات الحيوية والأدوية الموصوفة."

    if intent == "alternative":
        if alternatives:
            alt = "، ".join(alternatives)
        else:
            alt = "لا يوجد بديل مطابق بالمادة الفعالة مسجل داخل قائمة البيانات."
        return f"💊 {name}\n🔁 البدائل المسجلة في نفس الملف: {alt}\n\n⚠️ اختيار البديل يعتمد على المادة الفعالة والجرعة والحالة الصحية."

    if intent == "active_ingredient":
        return f"💊 {name}\n🧪 المادة الفعالة: {active or 'غير مسجلة في البيانات.'}"

    if intent == "side_effects":
        return f"💊 {name}\nالأعراض الجانبية تختلف حسب الدواء والجرعة. البيانات الحالية لا تحتوي على قائمة موثقة خاصة بهذا الدواء للأعراض الجانبية. إذا ظهر ضيق نفس، تورم بالوجه/اللسان، إغماء أو تفاعل شديد، اطلب مساعدة طبية عاجلة."

    if intent == "interactions":
        return f"💊 {name}\nلا أستطيع تأكيد تداخل دوائي محدد من ملف الأسعار وحده. اذكر اسم الدواء الآخر أو الأدوية التي تستخدمها لأبحث عنها في البيانات المتاحة، واستشر الصيدلي/الطبيب قبل الجمع بين الأدوية."

    if intent == "pregnancy":
        return f"💊 {name}\nبالنسبة للحمل أو الرضاعة، لا يمكن اعتبار البيانات الحالية كافية للحكم على الأمان. يجب سؤال الطبيب أو الصيدلي قبل الاستخدام."

    if intent == "children":
        return f"💊 {name}\nجرعات الأطفال تعتمد على العمر والوزن والتركيز والحالة. لا تستخدم جرعة للأطفال من الدردشة وحدها؛ راجع الطبيب أو الصيدلي."

    if intent == "missed_dose":
        return f"💊 {name}\nإذا نسيت جرعة فلا تضاعف الجرعة تلقائياً. التصرف الصحيح يعتمد على نوع الدواء ووقت تذكر الجرعة؛ راجع النشرة أو الصيدلي."

    if intent == "storage":
        return f"💊 {name}\nشروط الحفظ الدقيقة غير متوفرة في بيانات هذا الملف. اتبع تعليمات العبوة والنشرة، وتجنب الحرارة والرطوبة والضوء إذا نصت التعليمات على ذلك."

    if intent == "prescription":
        return f"💊 {name}\nمتطلبات الوصفة/الصرف لا يمكن تأكيدها من ملف الأسعار وحده. بعض الأدوية تحتاج وصفة حسب نوعها والأنظمة المحلية."

    if intent == "contraindications":
        return f"💊 {name}\nموانع الاستعمال التفصيلية غير مكتملة في ملف الأسعار. لا تستخدم الدواء إذا كان لديك تحسس معروف منه أو من مكوناته، واستشر الصيدلي/الطبيب للحالات الخاصة."

    if intent == "allergy":
        return f"💊 {name}\nإذا كان لديك تحسس معروف من المادة الفعالة أو مكونات الدواء فلا تستخدمه دون استشارة مختص. عند حدوث تورم بالوجه/اللسان أو صعوبة تنفس اطلب الطوارئ فوراً."

    if intent == "expiry":
        return f"💊 {name}\nتاريخ الانتهاء لا يظهر ضمن بيانات قائمة الأسعار الحالية. افحص تاريخ EXP المطبوع على العبوة ولا تستخدم دواءً منتهي الصلاحية."

    # General medicine response
    pieces = [f"💊 {name}"]
    if active:
        pieces.append(f"🧪 المادة الفعالة: {active}")
    if unit:
        pieces.append(f"📦 الوحدة: {unit}")
    if price is not None:
        pieces.append(f"💰 السعر المسجل: {format_price(price)}")
    if indications:
        pieces.append(f"📌 الاستخدام: {indications}")
    if alternatives:
        pieces.append(f"🔁 بدائل من نفس الملف: {', '.join(alternatives)}")
    pieces.append(f"ℹ️ {PRICE_NOTE}")
    return "\n".join(pieces)


def search_intents(question):
    qn = normalize_text(question)
    best = None
    best_score = 0.0
    for item in get_intents():
        if not isinstance(item, dict):
            continue
        patterns = item.get("patterns", item.get("questions", item.get("questions_ar", [])))
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns or []:
            p = normalize_text(pattern)
            if not p:
                continue
            if qn == p:
                return item
            score = difflib.SequenceMatcher(None, qn, p).ratio()
            qt, pt = tokens(qn), tokens(p)
            if qt and pt:
                score = max(score, len(qt & pt) / len(qt | pt))
            if score > best_score:
                best_score, best = score, item
    return best if best_score >= 0.62 else None


def get_intent_response(item):
    if not isinstance(item, dict):
        return None
    responses = item.get("responses", item.get("response"))
    if isinstance(responses, list):
        return responses[0] if responses else None
    return responses


def is_thanks(q):
    qn = normalize_text(q)
    phrases = [
        "شكرا", "شكرا لك", "شكراً", "شكراً لك", "مشكور", "مشكورة",
        "تسلم", "تسلمي", "يعطيك العافية", "يعطيك العافيه", "جزاك الله خير",
        "thank you", "thanks", "thank u", "much appreciated"
    ]
    return any(p in qn for p in [normalize_text(x) for x in phrases])


def is_farewell(q):
    qn = normalize_text(q)
    phrases = [
        "مع السلامة", "مع السلامه", "السلامة", "في امان الله", "في أمان الله",
        "باي", "باي باي", "اشوفك لاحقا", "اشوفك لاحقاً", "الى اللقاء", "إلى اللقاء",
        "تصبح على خير", "تصبحي على خير", "goodbye", "bye", "see you", "see you later"
    ]
    return any(p in qn for p in [normalize_text(x) for x in phrases])


def is_how_are_you(q):
    qn = normalize_text(q)
    phrases = ["كيف حالك", "كيفك", "عامل شنو", "كيف الامور", "how are you", "how are u"]
    return any(p in qn for p in [normalize_text(x) for x in phrases])


def is_about_bot(q):
    qn = normalize_text(q)
    phrases = ["من انت", "من انت؟", "شنو انت", "ما هو هذا", "عن البوت", "what are you", "who are you"]
    return any(p in qn for p in [normalize_text(x) for x in phrases])


def pharmacy_general_response(q):
    qn = normalize_text(q)
    if has_any(qn, ["عنوان", "الموقع", "وين الصيدلية", "وينكم", "مكانكم", "address", "location"]):
        return f"📍 عنوان {PHARMACY_NAME}: {ADDRESS}"
    if has_any(qn, ["مواعيد", "فاتحين", "دوام", "24", "ساعات", "hours", "open"]):
        return f"🕐 {PHARMACY_NAME} تعمل {OPENING_HOURS}."
    if has_any(qn, ["خدمات", "بتقدموا", "ماذا تقدم", "services"]):
        return "💙 خدمات صيدلية أرقين تشمل الاستفسار عن الأدوية والأسعار ودواعي الاستعمال والبدائل، والمساعدة في طلب الدواء والتوصيل حسب المتاح."
    if has_any(qn, ["طلب", "اطلب", "شراء", "احجز", "order", "buy"]):
        return "🛒 يمكنني مساعدتك في معرفة بيانات الدواء وسعره المسجل. اكتب اسم الدواء وطلبك، ثم يتم تأكيد التوفر والسعر قبل إتمام الطلب."
    if has_any(qn, ["توصيل", "التوصيل", "delivery", "مندوب"]):
        return "🚚 خدمة التوصيل متاحة حسب المنطقة والتنسيق. اذكر اسم الدواء والمنطقة وسنوضح لك خطوات الطلب."
    return None


def local_ai_fallback(question):
    """Local NLP fallback. Return None when it has no real answer so the external AI can run."""
    intent_item = search_intents(question)
    if intent_item:
        response = get_intent_response(intent_item)
        if response:
            return response

    qn = normalize_text(question)
    if is_emergency_question(qn):
        return "🚨 إذا كانت الحالة طارئة مثل صعوبة شديدة في التنفس، فقدان الوعي، نزيف شديد أو تورم سريع بالوجه/اللسان، توجه للطوارئ فوراً ولا تعتمد على الدردشة."

    return None


def external_ai_answer(question):
    """OpenAI fallback for general/external questions after local pharmacy logic."""
    if not AI_ENABLED or not OPENAI_API_KEY:
        if not OPENAI_API_KEY:
            print("[OpenAI] OFF: ضع OPENAI_API_KEY في متغيرات البيئة")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0, max_retries=2)

        instructions = (
            "أنت المساعد الذكي لموقع صيدلية أرقين في عطبرة – الحي الشرقي – مربع 8، وتعمل الصيدلية 24 ساعة. "
            "أجب بالعربية البسيطة، ويمكنك استخدام أسلوب سوداني مهذب عند ملاءمته. "
            "هذه الطبقة مخصصة للأسئلة العامة والخارجية والمحادثة التي لا يغطيها نظام الصيدلية المحلي. "
            "إذا كان السؤال عن سعر أو توفر أو بيانات دواء داخل الصيدلية، لا تخترع أي معلومة؛ اذكر أن بيانات الصيدلية المحلية هي المرجع. "
            "لا تشخّص الأمراض ولا تعطي جرعات شخصية خطرة. في أسئلة الأطفال والحمل والأعراض الخطيرة، قدّم إرشاداً عاماً آمناً ووجّه للطبيب أو الصيدلي عند الحاجة. "
            "كن مختصراً ومفيداً، ويمكنك الرد على الشكر والوداع والتحية بلطف. "
        )
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=question,
            max_output_tokens=700,
        )
        text = (getattr(response, "output_text", "") or "").strip()
        if text:
            print(f"[OpenAI] SUCCESS: {OPENAI_MODEL}")
            return text
    except Exception as e:
        print(f"[OpenAI] ERROR: {type(e).__name__}: {e}")
    return None


def handle_medicine_question(question):
    """Handle medicine-related questions using the local dataset first."""
    intent = classify_intent(question)
    medicine = extract_medicine_from_question(question)

    if medicine:
        return medicine_information(medicine, intent)

    if intent != "general":
        active_match = search_by_active_ingredient(question)
        if active_match:
            return medicine_information(active_match, intent)

    return None


def get_bot_response(message):
    message = (message or "").strip()
    if not message:
        return "اكتب سؤالك من فضلك 😊"

    if is_greeting(message):
        return (
            f"مرحباً بك في {PHARMACY_NAME} 💙\n"
            "نقدم خدمات صيدلانية متكاملة على مدار 24 ساعة.\n"
            f"📍 {ADDRESS}\n"
            "لسنا الوحيدون لكننا الأفضل.\n"
            "كيف يمكنني مساعدتك اليوم؟"
        )

    if is_thanks(message):
        return "العفو يا غالي 💙 سعدنا بخدمتك. لو عندك أي سؤال عن دواء أو الصيدلية أنا حاضر."

    if is_farewell(message):
        return "مع السلامة 🌷 وشكراً لزيارتك صيدلية أرقين. نتمنى لك الصحة والعافية."

    if is_how_are_you(message):
        return "بخير والحمد لله 😊 وأنا جاهز أساعدك في أي استفسار عن الصيدلية أو الأدوية."

    if is_about_bot(message):
        return "أنا المساعد الذكي لصيدلية أرقين 🤖💊 أساعدك في معلومات الأدوية والأسعار والتوفر والخدمات، ويمكنني أيضاً الإجابة عن الأسئلة العامة."

    general = pharmacy_general_response(message)
    if general:
        return general

    medicine_answer = handle_medicine_question(message)
    if medicine_answer:
        return medicine_answer

    # If a medicine is not found but the question is clearly medicine-related,
    # try the dataset's prepared intents before the optional LLM.
    local = local_ai_fallback(message)
    if local:
        return local

    ai = external_ai_answer(message)
    if ai:
        return ai

    return "لم أفهم السؤال بصورة كافية. اكتب اسم الدواء أو السؤال بطريقة أخرى، وسأحاول مساعدتك."


# -----------------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


def read_message_from_request():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        return data.get("message") or data.get("msg") or ""
    return request.form.get("message") or request.form.get("msg") or ""


@app.route("/chat", methods=["POST"])
def chat():
    try:
        message = read_message_from_request()
        return jsonify({"response": get_bot_response(message)})
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        return jsonify({"response": "حدث خطأ داخل السيرفر. راجع Terminal.", "error": str(e)}), 500


# Compatibility with older frontend code.
@app.route("/get", methods=["POST"])
def get_response():
    try:
        message = read_message_from_request()
        return jsonify({"response": get_bot_response(message)})
    except Exception as e:
        print(f"[GET ERROR] {type(e).__name__}: {e}")
        return jsonify({"response": "حدث خطأ داخل السيرفر. راجع Terminal.", "error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    medicines = get_medicines()
    intents = get_intents()
    return jsonify({
        "status": "ok",
        "pharmacy": PHARMACY_NAME,
        "medicines_count": len(medicines),
        "intents_count": len(intents),
        "medicines_file_exists": os.path.exists(MEDICINES_FILE),
        "intents_file_exists": os.path.exists(INTENTS_FILE),
        "ai_local_nlp": True,
        "external_ai_enabled": bool(AI_ENABLED and OPENAI_API_KEY),
        "external_ai_provider": "OpenAI",
        "external_ai_model": OPENAI_MODEL,
    })


if __name__ == "__main__":
    medicines = get_medicines()
    intents = get_intents()
    print("=" * 60)
    print(f"{PHARMACY_NAME} - AI Pharmacy Chatbot")
    print(f"Medicines loaded: {len(medicines)}")
    print(f"Intent records loaded: {len(intents)}")
    print(f"Medicines file: {MEDICINES_FILE}")
    print(f"Intents file: {INTENTS_FILE}")
    print(f"Local AI/NLP: ON")
    print(f"External AI (OpenAI): {'ON' if AI_ENABLED and OPENAI_API_KEY else 'OFF'}")
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
