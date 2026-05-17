# app.py — Incredible India Travel Recommender
# pip install flask pandas scikit-learn anthropic requests fpdf2 google-genai python-dotenv

from flask import Flask, render_template, request, jsonify, send_file
from recommender import recommend
import os, io, random, anthropic, requests
from fpdf import FPDF
# ── API KEYS ──────────────────────────
import pathlib
_env_path = pathlib.Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip()  # override, not setdefault

app = Flask(__name__)

# Load .env file manually as fallback
def _load_env():
    import pathlib
    env_path = pathlib.Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
WEATHER_API_KEY   = os.environ.get("WEATHER_API_KEY", None)
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", None)
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", None)
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY") or os.environ.get("UNSPLASH_KEY")
UNSPLASH_CACHE = {}

# ── Startup check ─────────────────────────────────────────────
print("=" * 50)
print(f"GROQ_API_KEY loaded: {bool(GROQ_API_KEY)}")
print(f"WEATHER_API_KEY loaded: {bool(WEATHER_API_KEY)}")
import pathlib as _pl
_ep = _pl.Path(__file__).parent / ".env"
print(f".env file found: {_ep.exists()} at {_ep}")
print("=" * 50)

COORDS = {
    "Goa":(15.2993,74.1240),"Manali":(32.2396,77.1887),
    "Kerala Backwaters":(9.4981,76.3388),"Rajasthan - Jaipur":(26.9124,75.7873),
    "Varanasi":(25.3176,82.9739),"Darjeeling":(27.0360,88.2627),
    "Leh Ladakh":(34.1526,77.5771),"Andaman Islands":(11.7401,92.6586),
    "Mysuru":(12.2958,76.6394),"Rishikesh":(30.0869,78.2676),
    "Coorg":(12.3375,75.8069),"Hampi":(15.3350,76.4600),
    "Udaipur":(24.5854,73.7125),"Munnar":(10.0889,77.0595),
    "Spiti Valley":(32.2461,78.0338),"Ooty":(11.4102,76.6950),
    "Amritsar":(31.6340,74.8723),"Ranthambore":(26.0173,76.5026),
    "Pondicherry":(11.9416,79.8083),"Agra":(27.1767,78.0081),
    "Shimla":(31.1048,77.1734),"Jodhpur":(26.2389,73.0243),
    "Alleppey":(9.4981,76.3388),"Kaziranga":(26.5775,93.1711),
    "Khajuraho":(24.8318,79.9199),"Mahabalipuram":(12.6269,80.1927),
    "Ziro Valley":(27.5330,93.8280),"Mount Abu":(24.5926,72.7156),
    "Puri":(19.8135,85.8312),"Pachmarhi":(22.4675,78.4340),
    "Tawang":(27.5861,91.8594),"Varkala":(8.7379,76.7163),
    "Chopta":(30.3928,79.2041),"Pushkar":(26.4899,74.5511),
    "Kovalam":(8.4004,76.9784),"Bikaner":(28.0229,73.3119),
    "Mawsynram":(25.2971,91.5826),"Gandikota":(14.8167,78.5167),
    "Jaisalmer":(26.9157,70.9083),"Mcleod Ganj":(32.2427,76.3234),
    "Majuli":(26.9500,94.1667),"Mahabaleshwar":(17.9237,73.6586),
    "Chikmagalur":(13.3161,75.7720),"Ellora":(20.0258,75.1780),
    "Kumarakom":(9.6167,76.4167),"Cherrapunji":(25.2800,91.7200),
    "Orchha":(25.3510,78.6413),"Nainital":(29.3803,79.4636),
    "Auroville":(12.0057,79.8107),"Dhanushkodi":(9.1667,79.4167),
    "Sandakphu":(27.1077,88.0000),"Munsiyari":(30.0653,80.2370),
    "Tarkarli":(16.0167,73.4667),"Lansdowne":(29.8417,78.6866),
    "Lonavala":(18.7537,73.4061),"Jawai":(25.0167,73.2833),
    "Bhedaghat":(23.1490,79.8095),"Chettinad":(10.1333,78.8333),
    "Pelling":(27.3000,88.2167),"Gurez Valley":(34.6333,74.8500),
    "Dzukou Valley":(25.5167,94.0500),"Shillong":(25.5788,91.8933),
    "Mathura Vrindavan":(27.4924,77.6737),"Konark":(19.8876,86.0948),
    "Binsar":(29.7167,79.7500),"Amer":(26.9855,75.8513),
    "Murudeshwar":(14.0944,74.4817),"Badami":(15.9167,75.6833),
    "Lakshadweep":(10.5667,72.6417),"Kanha":(22.3356,80.6119),
    "Bandhavgarh":(23.7167,81.0167),"Kutch":(23.7337,69.8597),
    "Diu":(20.7144,70.9874),"Nubra Valley":(34.6500,77.5500),
    "Araku Valley":(18.3273,82.8794),"Tirthan Valley":(31.6667,77.3833),
    "Landour":(30.4500,78.0667),"Mukteshwar":(29.4756,79.6456),
    "Champaner":(22.4876,73.5400),"Kolad":(18.2500,73.2500),
    "Chail":(30.9667,77.2000),"Haflong":(25.1667,93.0167),
    "Panchgani":(17.9238,73.7998),"Lepakshi":(13.7994,77.6071),
    "Mandu":(22.3556,75.3956),"Bhandardara":(19.5333,73.7500),
    "Kheerganga":(32.2167,77.3000),"Rann of Kutch":(23.7337,69.8597),
    "Marayoor":(10.2667,77.1500),"Kausani":(29.8333,79.6000),
    "Naggar":(32.1167,77.1667),"Spiti - Key Monastery":(32.3000,78.0000),
    "Patan":(23.8494,72.1266),"Mudumalai":(11.5667,76.6333),
    "Bir Billing":(32.0439,76.7194),"Muzhappilangad Beach":(11.8333,75.5167),
    "Kalo Dungar":(23.9333,69.4000),"Kabini":(11.9167,76.4000),
    "Chandra Taal":(32.4833,77.6167),"Hemis":(33.9167,77.7000),
    "Nohkalikai":(25.2667,91.5833),"Coonoor":(11.3530,76.7959),
    "Sattal":(29.4167,79.5500),"Cotigao":(15.0833,74.1167),
    "Rewalsar Lake":(31.6394,76.8261),"Yana Rocks":(14.8333,74.7833),
    "Dibang Valley":(28.6333,95.7167),"Bhimashankar":(19.0667,73.5333),
    "Tadoba":(20.2167,79.5167),"Bhutan Border - Phuentsholing":(26.8536,89.3882),
    "Pangot":(29.4333,79.4667),"Bhimtal":(29.3500,79.5667),
    "Chakrata":(30.7000,77.8667),"Dalhousie":(32.5388,75.9794),
    "Khimsar":(27.0333,73.9667),"Magnetic Hill":(34.1700,77.4000),
    "Kasol":(32.0103,77.3148),"Parashar Lake":(31.8167,76.9833),
    "Morni Hills":(30.6833,77.1000),"Daringbadi":(20.0167,84.0167),
    "Deomali":(27.3833,95.8500),"Lohit Valley":(27.8333,96.1667),
    "Longwa Village":(26.5167,95.2833),"Dzukou Valley Trek":(25.5167,94.0500),
    "Loktak Lake":(24.5167,93.7833),"Keibul Lamjao":(24.4833,93.8667),
    "Phawngpui":(22.5833,93.2167),"Vantawng Falls":(23.2333,92.7833),
    "Unakoti":(24.3167,92.0167),"Neermahal":(23.5000,91.4167),
    "Tezpur":(26.6338,92.7947),"Majhuli Satra":(26.9500,94.1667),
    "Bomdila":(27.2667,92.4167),"Along":(28.1667,94.8000),
    "Mechuka":(28.6000,94.1500),"Pakke":(27.0000,93.0000),
    "Walong":(28.1333,97.0333),"Lambasingi":(17.9167,82.6167),
    "Horsley Hills":(13.6500,78.3833),"Belum Caves":(15.1500,78.8667),
    "Bhongir Fort":(17.5167,78.8833),"Kurnool Caves":(15.8281,78.0373),
    "Bheemunipatnam":(17.8833,83.4500),"Arvalem":(15.6167,73.9000),
    "Cabo de Rama":(14.9167,73.9500),"Divar Island":(15.4833,73.9333),
    "Salim Ali Bird Sanctuary":(15.5000,73.8333),"Bhimgad":(15.0667,74.5167),
    "Kudremukh":(13.1833,75.2333),"Agumbe":(13.5000,75.1000),
    "Mullayanagiri":(13.3969,75.7337),"Bandipur":(11.6667,76.6333),
    "Nagarhole":(12.1667,76.1167),"Hogenakkal":(12.1000,77.7833),
    "Yercaud":(11.7745,78.2072),"Kodaikanal":(10.2381,77.4892),
    "Topslip":(10.4167,76.9833),"Kolukkumalai":(10.0833,77.2167),
    "Parambikulam":(10.3500,76.7333),"Thenmala":(8.9667,77.0667),
    "Silent Valley":(11.0833,76.4667),"Athirappilly":(10.2833,76.5833),
    "Wayanad":(11.6854,76.1320),"Nelliampathy":(10.5500,76.6167),
    "Idukki":(9.9667,76.9667),"Kumily":(9.5981,77.1731),
    "Vagamon":(9.6833,76.9000),"Kudle Beach":(14.4167,74.3167),
    "Om Beach":(14.4000,74.3167),"Murdeshwar":(14.0944,74.4817),
    "Yana":(14.8333,74.7833),"Jog Falls":(14.2167,74.8000),
    "Sirsi":(14.6167,74.8333),"Shravanabelagola":(12.8583,76.4854),
    "Belur Halebidu":(13.1667,75.8667),"Aihole Pattadakal":(15.9333,75.8667),
    "Ankola":(14.6667,74.3000),"Mulki":(13.0833,74.7833),
    "Udupi":(13.3409,74.7421),"Thekkady":(9.6000,77.1667),
    "Kovalam - Lighthouse":(8.4004,76.9784),"Veli Tourist Village":(8.5167,76.8833),
    "Munroe Island":(9.0000,76.5833),"Marari Beach":(9.5833,76.2833),
    "Kannur":(11.8745,75.3704),"Bekal":(12.3833,75.0500),
    "Kasaragod":(12.4996,74.9869),"Kasol Kheerganga Circuit":(32.0103,77.3148),
    "Triund":(32.2833,76.9167),"Pabbar Valley":(31.1667,77.7500),
    "Sarchu":(32.8667,77.6167),"Batal":(32.3667,77.5333),
    "Pin Valley":(31.9833,78.0500),"Kinnaur":(31.5833,78.2667),
    "Sangla":(31.4167,78.2333),"Chitkul":(31.3500,78.4500),
    "Malana":(32.0833,77.3167),"Barot Valley":(31.8667,76.9167),
    "Rann Utsav Camp":(23.7337,69.8597),"Sangla Valley":(31.4167,78.2333),
    "Chadar Trek":(33.9167,77.0167),"Dooars":(26.7000,89.4500),
    "Mirik":(26.8833,88.1833),"Munger":(25.3786,86.4733),
    "Bidar":(17.9167,77.5167),"Hamta Pass Trek":(32.2500,77.2667),
    "Valley of Flowers":(30.7167,79.6000),"Chopta Tungnath":(30.3928,79.2041),
    "Almora":(29.5971,79.6591),"Chikhalda":(21.5833,77.3167),
    "Pench":(21.7667,79.2833),"Jim Corbett":(29.5300,78.7747),
    "Satpura":(22.5833,78.4167),"Periyar":(9.5000,77.1167),
    "Valparai":(10.3667,76.9500),"Yelagiri":(12.5833,78.6333),
    "Pattadakal":(15.9500,75.8167),
}

# ── PAGES ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── RECOMMEND ────────────────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    prefs = {k: data.get(k, "") for k in
             ["budget","climate","activity_type","best_season","vibe","region"]}
    top_n   = int(data.get("top_n", 5))
    results = recommend(prefs, top_n=top_n)
    rows    = results.to_dict(orient="records")
    import math
    for r in rows:
        c = COORDS.get(r["name"], (20.5937, 78.9629))
        r["lat"] = c[0]; r["lng"] = c[1]
        # Sanitize NaN/Inf values so JSON stays valid
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
    return jsonify(rows)

# ── WEATHER ──────────────────────────────────────────────────
@app.route("/weather", methods=["POST"])
def weather():
    name = request.get_json().get("name", "")
    if not WEATHER_API_KEY:
        conds = ["Sunny ☀️","Partly Cloudy 🌤️","Clear Sky 🌙","Hazy 🌫️","Breezy 💨"]
        return jsonify({"temp":random.randint(18,34),"feels_like":random.randint(16,36),
                        "condition":random.choice(conds),"humidity":random.randint(40,85),
                        "wind":round(random.uniform(5,20),1),"mock":True})
    import requests as req
    coords = COORDS.get(name)
    if not coords: return jsonify({"error":"Not found"}), 404
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={coords[0]}&lon={coords[1]}&appid={WEATHER_API_KEY}&units=metric")
        r = req.get(url, timeout=5).json()
        return jsonify({"temp":round(r["main"]["temp"]),"feels_like":round(r["main"]["feels_like"]),
                        "condition":r["weather"][0]["main"],"humidity":r["main"]["humidity"],
                        "wind":r["wind"]["speed"],"mock":False})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── AI DESCRIPTION (Groq llama) ──────────────────────────────
@app.route("/destination-photo", methods=["POST"])
def destination_photo():
    data = request.get_json() or {}
    destination = (data.get("destination") or "").strip()
    state = (data.get("state") or "").strip()
    activity = (data.get("activity_type") or "").strip()

    if not destination:
        return jsonify({"url": None}), 400

    cache_key = f"{destination}|{state}|{activity}".lower()
    if cache_key in UNSPLASH_CACHE:
        return jsonify(UNSPLASH_CACHE[cache_key])

    if not UNSPLASH_ACCESS_KEY:
        return jsonify({"url": None, "source": "fallback"})

    queries = [
        f"{destination} {state} India travel",
        f"{destination} {state} India",
        f"{destination} India landmark",
        f"{destination} India tourism",
        f"{activity} destination India" if activity else "",
    ]
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
        "Accept-Version": "v1",
    }

    try:
        for query in [q for q in queries if q]:
            resp = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "orientation": "landscape", "per_page": 10},
                headers=headers,
                timeout=8,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                continue

            photo = results[0]
            result = {
                "url": photo.get("urls", {}).get("regular") or photo.get("urls", {}).get("small"),
                "photographer": (photo.get("user") or {}).get("name", "Unsplash"),
                "source": "unsplash",
                "link": ((photo.get("links") or {}).get("html") or "https://unsplash.com") + "?utm_source=travel_recommender&utm_medium=referral",
            }
            UNSPLASH_CACHE[cache_key] = result
            return jsonify(result)
    except Exception as e:
        return jsonify({"url": None, "source": "fallback", "error": str(e)})

    return jsonify({"url": None, "source": "fallback"})

@app.route("/ai-description", methods=["POST"])
def ai_description():
    if not GROQ_API_KEY:
        return jsonify({"description": None})
    d = request.get_json()
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (f"Write a vivid 2-sentence travel teaser for {d['destination']}, {d['state']}, India. "
                  f"Traveller wants {d['vibe']} vibe, {d['activity']} activity during {d['season']}. "
                  f"Be poetic and specific. Max 55 words.")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.8,
        )
        return jsonify({"description": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"description": None, "error": str(e)})

# ── CHATBOT (Groq llama — fast & free) ──────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json()
    messages = data.get("messages", [])

    SYSTEM_PROMPT = """You are Yatra AI, an expert travel assistant for India.
You know everything about Indian destinations, culture, food, travel tips, best seasons, budgets, and itineraries.
Keep responses concise (max 3-4 sentences), warm, and helpful.
Use occasional Hindi/Hinglish words naturally (like 'bilkul', 'bahut achha', 'zaroor').
Always end with a helpful tip or question to keep the conversation going.
Never make up facts — if unsure, say so."""

    if not GROQ_API_KEY:
        demo_replies = [
            "Namaste! I am Yatra AI — your India travel guide! Set GROQ_API_KEY in .env to enable live chat. 🙏",
            "Goa is perfect for beaches, nightlife, and seafood! November to February is the best time. 🏖️",
            "For Ladakh, June to September is ideal when roads are open. Acclimatize properly! 🏔️",
            "Kerala backwaters are magical from October to March. Try a houseboat stay! 🌴",
        ]
        return jsonify({"reply": random.choice(demo_replies)})

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in messages:
            groq_messages.append({"role": msg["role"], "content": msg["content"]})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            max_tokens=300,
            temperature=0.7,
        )
        return jsonify({"reply": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"reply": f"Sorry, something went wrong: {str(e)}"})

# ── ITINERARY (Groq llama) ───────────────────────────────────
@app.route("/itinerary", methods=["POST"])
def itinerary():
    d = request.get_json()
    dest=d.get("destination",""); state=d.get("state","")
    days=d.get("days",3); vibe=d.get("vibe","nature"); activity=d.get("activity","cultural")
    if not GROQ_API_KEY:
        mock={"days":[]}
        for i in range(1,int(days)+1):
            mock["days"].append({"day":i,"title":f"Day {i} in {dest}",
                "morning":"Explore local area and have breakfast at a local café",
                "afternoon":"Visit the main attraction and nearby sights",
                "evening":"Sunset views and local dinner experience",
                "tip":"Carry water and wear comfortable shoes"})
        return jsonify(mock)
    import json, re
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (f"Create a {days}-day itinerary for {dest}, {state}, India. "
                  f"Vibe: {vibe}, Activity: {activity}. "
                  f'Return ONLY valid JSON (no markdown, no explanation): {{"days":[{{"day":1,"title":"...","morning":"...","afternoon":"...","evening":"...","tip":"..."}}]}}')
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()
        return jsonify(json.loads(raw))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── PDF REPORT ───────────────────────────────────────────────
@app.route("/pdf-report", methods=["POST"])
def pdf_report():
    data  = request.get_json()
    dests = data.get("destinations", [])
    prefs = data.get("prefs", {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(7, 11, 20)
    pdf.rect(0, 0, 210, 44, "F")
    pdf.set_xy(0, 8)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(147, 197, 253)
    pdf.cell(210, 12, "INCREDIBLE INDIA", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 140, 200)
    pdf.cell(210, 6, "AI-Powered Travel Recommendation Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(16)

    # Preferences box
    pdf.set_fill_color(230, 240, 255)
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(0.4)
    pdf.rect(10, pdf.get_y(), 190, 22, "FD")
    pdf.set_xy(14, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(20, 40, 100)
    pdf.cell(0, 5, "Your Search Preferences", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(14, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(50, 70, 130)
    labels = {"budget":"Budget","climate":"Climate","activity_type":"Activity",
              "best_season":"Season","vibe":"Vibe","region":"Region"}
    pref_txt = "  |  ".join(f"{labels.get(k,k).title()}: {str(v).title()}"
                             for k,v in prefs.items() if k != "top_n" and v)
    pdf.cell(0, 5, pref_txt[:100], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Section title
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 30, 80)
    pdf.cell(0, 8, f"Top {len(dests)} Recommended Destinations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    for i, dest in enumerate(dests):
        y = pdf.get_y()
        if y > 240:
            pdf.add_page()
            y = pdf.get_y()

        # Card bg
        pdf.set_fill_color(240, 245, 255)
        pdf.set_draw_color(59, 130, 246)
        pdf.set_line_width(0.3)
        pdf.rect(10, y, 190, 55, "FD")

        # Rank circle
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_xy(12, y + 3)
        pdf.cell(14, 8, f"#{i+1}", fill=True, align="C")

        # Name
        pdf.set_xy(30, y + 3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(15, 25, 80)
        name = dest.get("name", "")
        pdf.cell(110, 8, name[:30], ln=False)

        # Match %
        pdf.set_xy(155, y + 3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(59, 130, 246)
        pdf.cell(44, 8, f"{dest.get('match_percent', 0)}% Match", align="R")

        # Location
        pdf.set_xy(30, y + 13)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 100, 160)
        state = dest.get("state", "")
        region = dest.get("region", "").title()
        pdf.cell(0, 5, f"  {state}  |  {region} India", new_x="LMARGIN", new_y="NEXT")

        # Description
        pdf.set_xy(12, y + 20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(60, 80, 140)
        desc = (dest.get("description") or "")[:115]
        if len(dest.get("description","")) > 115:
            desc += "..."
        pdf.multi_cell(186, 4.5, desc)

        # Tags
        tag_y = y + 38
        pdf.set_xy(12, tag_y)
        tags = [
            ("Activity", dest.get("activity_type","").title()),
            ("Budget",   dest.get("budget","").title()),
            ("Season",   dest.get("best_season","").title()),
            ("Rating",   f"{dest.get('rating',0)}/5"),
            ("Crowd",    dest.get("crowd_level","") or "—"),
        ]
        for lbl, val in tags:
            pdf.set_fill_color(59, 130, 246)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 6)
            pdf.cell(16, 5, lbl.upper(), fill=True, align="C")
            pdf.set_fill_color(210, 225, 255)
            pdf.set_text_color(20, 40, 100)
            pdf.set_font("Helvetica", "", 7)
            pdf.cell(18, 5, str(val)[:10], fill=True, align="C")
            pdf.cell(2, 5, "")

        # Airport & Best Month extras
        pdf.set_xy(12, tag_y + 7)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(80, 100, 160)
        extras = []
        if dest.get("nearest_airport") and dest.get("nearest_airport") != "—":
            extras.append(f"Airport: {dest['nearest_airport']}")
        if dest.get("best_month"):
            extras.append(f"Best Month: {dest['best_month']}")
        if extras:
            pdf.cell(0, 5, "  " + "   |   ".join(extras))
        pdf.ln(21)

    # Footer
    pdf.ln(4)
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100, 130, 200)
    pdf.cell(0, 5,
        "Generated by Incredible India AI  |  Content-Based Filtering + Cosine Similarity  |  217 Destinations",
        align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name="incredible_india_recommendations.pdf")


if __name__ == "__main__":
    app.run(debug=True)
