# recommender.py — v8 FIXED (Exact Match Scoring — no LabelEncoder cosine bug)

import pandas as pd
import numpy as np

RETURN_COLS = [
    "name","hindi_name","state","region","activity_type","vibe","best_season",
    "rating","match_percent","budget","description",
    "daily_cost_min","daily_cost_max","famous_for","crowd_level",
    "best_food","entry_fee","stay_type","travel_mode","trek_difficulty",
    "wildlife","unesco","honeymoon_suitable","family_suitable",
    "solo_suitable","adventure_level","spiritual_level","photography_score",
    "nearest_airport","nearest_railway","best_month","local_transport",
    "avg_temp_summer","avg_temp_winter","internet_connectivity",
    "hidden_gem_score","instagram_worthy","night_life","shopping_rating",
    "kid_friendly_score","accessibility_score","language"
]

# -------------------- LOAD DATA --------------------

def load_data():
    df = pd.read_csv("destinations.csv")

    num_defaults = {
        'hidden_gem_score': 5, 'instagram_worthy': 7,
        'kid_friendly_score': 6, 'accessibility_score': 5,
        'photography_score': 7, 'rating': 4.0,
        'daily_cost_min': 1000, 'daily_cost_max': 5000,
    }
    for col, val in num_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    str_defaults = {
        'adventure_level': 'low', 'spiritual_level': 'none',
        'trek_difficulty': 'none', 'wildlife': '-', 'night_life': 'low',
        'nearest_railway': '-', 'nearest_airport': '-',
        'internet_connectivity': 'medium', 'crowd_level': 'medium',
        'best_food': '-', 'entry_fee': 'free', 'stay_type': 'hotel',
        'travel_mode': 'bus|train', 'local_transport': 'auto|bus',
        'best_month': '-', 'language': 'hindi',
    }
    for col, val in str_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    df = df.fillna('')

    for col in ['vibe','activity_type','budget','climate','best_season','region']:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    return df

# -------------------- EXACT MATCH SCORING --------------------
# Root cause fix: LabelEncoder + cosine similarity gives WRONG results for
# nominal/categorical data. 'beach'=1, 'nature'=3 — numeric distance is
# meaningless for categories. Replaced with explicit weighted exact matching.

def score_destinations(df, user_prefs):
    WEIGHTS = {
        'vibe':          0.35,
        'activity_type': 0.25,
        'best_season':   0.18,
        'budget':        0.12,
        'climate':       0.10,
    }

    scores = np.zeros(len(df))

    for field, weight in WEIGHTS.items():
        user_val = user_prefs.get(field, '').lower().strip()
        if not user_val or user_val in ('any', 'all', ''):
            scores += weight * 0.5
            continue
        match = (df[field].str.lower().str.strip() == user_val).astype(float)
        scores += weight * match

    return scores

# -------------------- MAIN FUNCTION --------------------

def recommend(user_prefs, top_n=5):
    df = load_data()
    user_prefs = {k: v.lower().strip() if isinstance(v, str) else v
                  for k, v in user_prefs.items()}

    # Region hard filter
    region = user_prefs.get("region", "all")
    if region and region not in ("all", ""):
        filt = df[df["region"].str.lower() == region]
        if not filt.empty:
            df = filt.reset_index(drop=True)

    df = df.copy().reset_index(drop=True)
    df["match_score"] = score_destinations(df, user_prefs)

    df["final_score"] = (
        df["match_score"]               * 0.80 +
        (df["rating"] / 5.0)            * 0.12 +
        (df["hidden_gem_score"] / 10.0) * 0.05 +
        (df["photography_score"] / 10.0)* 0.03
    )

    max_score = df["final_score"].max()
    if max_score > 0:
        df["match_percent"] = ((df["final_score"] / max_score) * 100).round(1)
    else:
        df["match_percent"] = 0.0

    df_sorted = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    selected = []
    region_count = {}

    for _, row in df_sorted.iterrows():
        if len(selected) >= top_n:
            break
        reg = str(row.get("region", "")).lower()
        if region_count.get(reg, 0) < 2:
            selected.append(row)
            region_count[reg] = region_count.get(reg, 0) + 1

    if len(selected) < top_n:
        used_names = {r["name"] for r in selected}
        for _, row in df_sorted.iterrows():
            if len(selected) >= top_n:
                break
            if row["name"] not in used_names:
                selected.append(row)

    result = pd.DataFrame(selected)
    cols = [c for c in RETURN_COLS if c in result.columns]
    return result[cols]
