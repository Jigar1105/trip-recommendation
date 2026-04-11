from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_session import Session
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import re
import os
import json
import time
import hashlib
import logging
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pymysql
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# ==========================================
# LOGGING SETUP (production-ready)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('bharat_yatra.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ==========================================
# REDIS CACHING SETUP
# ==========================================
try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected!")
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.warning("Redis not available — falling back to in-memory cache.")

# In-memory fallback cache (agar Redis na ho)
memory_cache = {}

def cache_get(key):
    """Cache se value lo — Redis ya memory se"""
    if REDIS_AVAILABLE:
        try:
            val = redis_client.get(key)
            return json.loads(val) if val else None
        except Exception:
            pass
    return memory_cache.get(key)

def cache_set(key, value, ttl=86400):
    """Cache mein value save karo — TTL seconds mein (default 24 hours)"""
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(key, ttl, json.dumps(value))
            return
        except Exception:
            pass
    memory_cache[key] = value

def make_cache_key(prefix, **kwargs):
    """Consistent cache key banao"""
    content = prefix + json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

# ==========================================
# GROQ API SETUP
# ==========================================
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq API ready!")
else:
    groq_client = None
    logger.warning("GROQ_API_KEY not set.")

# ==========================================
# UNSPLASH SETUP
# ==========================================
UNSPLASH_KEY = os.environ.get('UNSPLASH_ACCESS_KEY', '')

# ==========================================
# AUTO-CREATE DATABASE
# ==========================================
def create_database_if_not_exists():
    try:
        conn = pymysql.connect(host='localhost', user='root', password='')
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS bharat_yatra")
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database 'bharat_yatra' ready!")
    except Exception as e:
        logger.error(f"Error creating database: {e}")

create_database_if_not_exists()

# ==========================================
# FLASK + DB SETUP
# ==========================================
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback_dev_key_change_in_production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/bharat_yatra'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'        # ← ADD KARO
app.config['SESSION_FILE_DIR'] = './flask_session'  # ← ADD KARO

db = SQLAlchemy(app)
Session(app)  # ← ADD KARO
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Pehle login karo!"
login_manager.login_message_category = "error"

# ==========================================
# MODELS
# ==========================================
class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100))
    email    = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    # Relationships
    searches      = db.relationship('SearchHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    saved_places  = db.relationship('SavedPlace',    backref='user', lazy=True, cascade='all, delete-orphan')
    itineraries   = db.relationship('Itinerary',     backref='user', lazy=True, cascade='all, delete-orphan')


class SearchHistory(db.Model):
    """User ki past searches"""
    __tablename__ = 'search_history'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    state      = db.Column(db.String(100))
    budget     = db.Column(db.Float)
    interests  = db.Column(db.String(200))
    results_count = db.Column(db.Integer, default=0)
    searched_at   = db.Column(db.DateTime, default=datetime.utcnow)


class SavedPlace(db.Model):
    """User ke saved/bookmarked places"""
    __tablename__ = 'saved_places'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    place_name  = db.Column(db.String(200), nullable=False)
    state       = db.Column(db.String(100))
    place_type  = db.Column(db.String(100))
    best_time   = db.Column(db.String(100))
    ideal_for   = db.Column(db.String(100))
    trip_cost   = db.Column(db.String(100))
    stay_duration = db.Column(db.String(100))
    max_budget  = db.Column(db.Integer, default=0)
    score       = db.Column(db.Float, default=0.0)
    image_url   = db.Column(db.String(500), default='')
    saved_at    = db.Column(db.DateTime, default=datetime.utcnow)

    # Ek user ek place ek baar hi save kar sake
    __table_args__ = (db.UniqueConstraint('user_id', 'place_name', name='uq_user_place'),)


class Itinerary(db.Model):
    """AI-generated itineraries"""
    __tablename__ = 'itineraries'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title        = db.Column(db.String(300))
    place_name   = db.Column(db.String(200))
    state        = db.Column(db.String(100))
    days         = db.Column(db.Integer, default=3)
    budget       = db.Column(db.Float, default=0)
    travel_style = db.Column(db.String(100), default='balanced')
    itinerary_json = db.Column(db.Text)   # Full AI response JSON
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ==========================================
# DATA LOADING & CLEANING
# ==========================================
 # ==========================================
# DATA LOADING & CLEANING  —  M1 + M6 FIXED
# ==========================================
def load_and_clean():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'cleaned_travel_data_unique.csv')
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        df['Type']            = df['Type'].fillna('general').str.lower().str.strip()
        df['Best Visit Time'] = df['Best Visit Time'].fillna('Year-round')
        df['State']           = df['State'].fillna('India').str.upper().str.strip()
        df['Ideal For']       = df['Ideal For'].fillna('all').str.lower().str.strip()
        df['Place Name']      = df['Place Name'].fillna('').str.strip()

        if 'City' not in df.columns:
            df['City'] = ''
        else:
            df['City'] = df['City'].fillna('').str.strip()

        if 'Stay Duration' not in df.columns:
            df['Stay Duration'] = '2-3 days'
        else:
            df['Stay Duration'] = df['Stay Duration'].fillna('2-3 days')

        # ──────────────────────────────────────────────────────────────
        # BUG M1 FIX: clean_cost — last number nahi, max meaningful number
        # "Rs. 8,000 - Rs. 12,000 for 3 days" → pehle: 3, ab: 12000
        # Logic: 100 se chhote numbers = days/nights/persons, ignore karo
        # ──────────────────────────────────────────────────────────────
        def clean_cost(val):
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            if not nums:
                return 0
            # 100 se chhote numbers = duration/quantity nahi, cost nahi
            big_nums = [int(n) for n in nums if int(n) >= 100]
            if big_nums:
                return max(big_nums)   # ← max budget lo, last nahi
            # Agar koi bhi number 100+ nahi (e.g. "5 days") → 0
            return 0

        df['max_budget'] = df['Trip Cost'].apply(clean_cost)

        # ──────────────────────────────────────────────────────────────
        # BUG M6 FIX: Content field — important fields ko zyada weight do
        # Type aur Ideal For ko repeat karke TF-IDF mein boost karo
        # Pehle: sab 1x weight, ab: Type=3x, Ideal For=2x
        # ──────────────────────────────────────────────────────────────
        df['content'] = (
            (df['Type'] + ' ') * 3 +               # Type — 3x weight (most important)
            df['Place Name'].str.lower() + ' ' +
            (df['Ideal For'] + ' ') * 2 +           # Ideal For — 2x weight
            df['Best Visit Time'].str.lower()        # Best time — 1x weight
        )

        logger.info(f"Data loaded: {len(df)} places, "
                    f"max_budget range: {df['max_budget'].min()} - {df['max_budget'].max()}")
        return df

    except FileNotFoundError:
        logger.error("cleaned_travel_data_unique.csv not found!")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


df_master = load_and_clean()


# ==========================================
# BUG M2 + M5 FIX: TF-IDF pre-compute at startup
# Pehle: har request pe fit_transform() → slow, poor vocabulary
# Ab: ek baar startup pe fit karo, har request pe sirf transform
# ==========================================
_vectorizer   = None   # global TF-IDF vectorizer
_full_matrix  = None   # global TF-IDF matrix (full dataset)

def _build_tfidf():
    """Startup pe ek baar — poori dataset pe vectorizer fit karo."""
    global _vectorizer, _full_matrix
    if df_master.empty:
        logger.warning("df_master empty — TF-IDF not built.")
        return
    try:
        _vectorizer  = TfidfVectorizer(stop_words='english', min_df=1)
        _full_matrix = _vectorizer.fit_transform(df_master['content'])
        logger.info(f"TF-IDF built: {_full_matrix.shape[0]} docs, "
                    f"{_full_matrix.shape[1]} features")
    except Exception as e:
        logger.error(f"TF-IDF build error: {e}")

_build_tfidf()   # ← app start hote hi ek baar run


# ==========================================
# KEYWORD EXPANSION  —  unchanged, working fine
# ==========================================
def expand_keywords(user_type):
    synonyms = {
        'trekking' : 'adventure mountain hiking climbing',
        'beach'    : 'sea water coastal ocean sand',
        'nature'   : 'greenery forest hills scenic waterfall lake',
        'spiritual': 'temple religious divine holy shrine',
        'history'  : 'fort monument ancient heritage palace museum',
        'wildlife' : 'animal safari tiger bird sanctuary national park',
        'hill'     : 'mountain valley cold snow mist',
        'desert'   : 'sand dune rajasthan hot dry camel',
        'waterfall': 'falls river stream nature scenic',
        'city'     : 'urban metro culture food market',
    }
    expanded = str(user_type).lower().strip()
    for key, val in synonyms.items():
        if key in expanded:
            expanded += ' ' + val
    return expanded


# ==========================================
# RECOMMENDATION ENGINE  —  M2 + M3 + M4 + M5 FIXED
# ==========================================
MIN_SCORE = 0.05   # M4 FIX: 5% minimum similarity — pure garbage results block

def get_recommendations(state, budget, interests):
    # Vectorizer ready check
    if df_master.empty or _vectorizer is None or _full_matrix is None:
        logger.error("Model not ready — df_master or TF-IDF missing.")
        return pd.DataFrame()

    df = df_master.copy()

    # ── Hard filters (budget + state) ──────────────────────────────
    mask = df['max_budget'] <= float(budget)

    # max_budget = 0 wale places (invalid cost data) ko exclude karo
    # Warna Rs. 0 budget = sab mein match ho jaayega
    mask = mask & (df['max_budget'] > 0)

    if state != 'All India':
        mask = mask & (df['State'].str.upper() == state.strip().upper())

    filtered = df[mask].copy()

    if filtered.empty:
        logger.info(f"No places found: state={state}, budget={budget}")
        return pd.DataFrame()

    # Original index positions save karo (full matrix ke liye)
    original_indices = filtered.index.tolist()
    filtered = filtered.reset_index(drop=True)

    # ── Query vector build ─────────────────────────────────────────
    expanded_query = expand_keywords(interests)

    # M2+M5 FIX: Pre-built vectorizer se sirf transform — fit nahi
    query_vec = _vectorizer.transform([expanded_query])

    # ── M3 FIX: Zero vector check ──────────────────────────────────
    # Agar user ne sirf stop words likhe (the, a, is) ya unknown words
    # → query_vec all-zeros → cosine undefined → random results
    if query_vec.nnz == 0:
        logger.warning(f"Zero query vector for interests='{interests}' — "
                       f"falling back to budget sort")
        # Fallback: budget ke andar best (highest budget = premium) places
        return filtered.assign(Score=0.1).sort_values(
            'max_budget', ascending=False
        ).head(10)

    # M2+M5 FIX: Full matrix mein se sirf filtered rows ka slice lo
    filtered_matrix = _full_matrix[original_indices]

    # ── KNN search ────────────────────────────────────────────────
    n_neighbors = min(len(filtered), 15)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(filtered_matrix)
    distances, indices = knn.kneighbors(query_vec)

    # ── Score assign ──────────────────────────────────────────────
    scores = np.zeros(len(filtered))
    for i, idx in enumerate(indices[0]):
        scores[idx] = 1 - distances[0][i]

    filtered['Score'] = scores

    # ── M4 FIX: Meaningful threshold — 0.05 minimum ───────────────
    results = filtered[filtered['Score'] >= MIN_SCORE].sort_values(
        by=['Score', 'Place Name'], ascending=[False, True]
    )

    # Agar threshold pe kuch nahi mila → top 5 by score (no empty results)
    if results.empty:
        logger.info(f"No results above threshold — returning top 5 by score")
        results = filtered.nlargest(5, 'Score').sort_values(
            by=['Score', 'Place Name'], ascending=[False, True]
        )

    logger.info(f"Recommendations: {len(results)} places for "
                f"state={state}, budget={budget}, interests={interests}")
    return results

# ==========================================
# ROUTE: Place Info (with Redis cache)
# ==========================================
@app.route('/api/place-info', methods=['POST'])
@login_required
def place_info():
    if not groq_client:
        return jsonify({'error': 'Groq API key not configured'}), 503

    data       = request.get_json()
    place_name = data.get('place_name', '').strip()
    state      = data.get('state', '').strip()
    place_type = data.get('type', '').strip()
    best_time  = data.get('best_time', '').strip()
    ideal_for  = data.get('ideal_for', '').strip()

    if not place_name:
        return jsonify({'error': 'Place name required'}), 400

    # ✅ Redis cache check
    cache_key = make_cache_key('place_info', place=place_name, state=state)
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"Cache HIT: {place_name}")
        return jsonify({'success': True, 'data': cached, 'cached': True})

    logger.info(f"Cache MISS: {place_name} — calling Groq API")

    prompt = f"""
You are an expert Indian travel guide. Provide detailed, engaging travel information about this place.

Place: {place_name}
State: {state}, India
Type: {place_type}
Best Visit Time: {best_time}
Ideal For: {ideal_for}

Return ONLY a valid JSON object with these exact keys (no markdown, no extra text):
{{
  "famous_for": "2-3 sentences about why this place is famous",
  "why_visit": "2-3 sentences about why someone should visit",
  "top_experiences": ["experience 1", "experience 2", "experience 3", "experience 4"],
  "local_tips": ["tip 1", "tip 2", "tip 3"],
  "best_season_reason": "1-2 sentences about best visit time reason",
  "nearby_attractions": ["nearby place 1", "nearby place 2", "nearby place 3"],
  "food_to_try": ["local dish 1", "local dish 2", "local dish 3"],
  "image_keywords": "3-4 specific keywords for image search of this place"
}}
"""

    try:
        time.sleep(0.5)
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        raw = response.choices[0].message.content
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()

        info = json.loads(raw)

        # ✅ Cache mein save karo (24 hours)
        cache_set(cache_key, info, ttl=86400)

        return jsonify({'success': True, 'data': info, 'cached': False})

    except Exception as e:
        logger.error(f"place_info error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==========================================
# ROUTE: Place Image (with cache)
# ==========================================
@app.route('/api/place-image', methods=['POST'])
@login_required
def place_image():
    import requests as req
    data  = request.get_json()
    query = data.get('query', '') + ' India travel'

    cache_key = make_cache_key('place_image', query=query)
    cached = cache_get(cache_key)
    if cached:
        return jsonify({'success': True, 'image_url': cached, 'cached': True})

    if not UNSPLASH_KEY:
        return jsonify({'success': False, 'image_url': ''})

    try:
        resp   = req.get(
            'https://api.unsplash.com/search/photos',
            params={'query': query, 'per_page': 1, 'orientation': 'landscape'},
            headers={'Authorization': f'Client-ID {UNSPLASH_KEY}'},
            timeout=5
        )
        result = resp.json()
        if result.get('results'):
            url = result['results'][0]['urls']['regular']
            cache_set(cache_key, url, ttl=604800)  # 7 days cache
            return jsonify({'success': True, 'image_url': url, 'cached': False})
        return jsonify({'success': False, 'image_url': ''})
    except Exception as e:
        logger.error(f"place_image error: {str(e)}")
        return jsonify({'success': False, 'image_url': ''})


# ==========================================
# ROUTE: Save Place
# ==========================================
@app.route('/api/save-place', methods=['POST'])
@login_required
def save_place():
    data = request.get_json()
    place_name = data.get('place_name', '').strip()

    if not place_name:
        return jsonify({'error': 'Place name required'}), 400

    # Already saved check
    existing = SavedPlace.query.filter_by(
        user_id=current_user.id,
        place_name=place_name
    ).first()

    if existing:
        # Toggle — already saved toh unsave karo
        db.session.delete(existing)
        db.session.commit()
        return jsonify({'success': True, 'action': 'unsaved', 'message': f'{place_name} removed from saved places'})

    # Naya save
    new_save = SavedPlace(
        user_id      = current_user.id,
        place_name   = place_name,
        state        = data.get('state', ''),
        place_type   = data.get('type', ''),
        best_time    = data.get('best_time', ''),
        ideal_for    = data.get('ideal_for', ''),
        trip_cost    = data.get('trip_cost', ''),
        stay_duration= data.get('stay_duration', ''),
        max_budget   = int(data.get('max_budget', 0)),
        score        = float(data.get('score', 0.0)),
        image_url    = data.get('image_url', ''),
    )
    db.session.add(new_save)
    db.session.commit()
    logger.info(f"User {current_user.id} saved place: {place_name}")
    return jsonify({'success': True, 'action': 'saved', 'message': f'{place_name} saved!'})


# ==========================================
# ROUTE: Check saved status (bulk)
# ==========================================
@app.route('/api/saved-status', methods=['POST'])
@login_required
def saved_status():
    data        = request.get_json()
    place_names = data.get('places', [])
    saved = SavedPlace.query.filter(
        SavedPlace.user_id == current_user.id,
        SavedPlace.place_name.in_(place_names)
    ).all()
    saved_set = {s.place_name for s in saved}
    return jsonify({'saved': list(saved_set)})

# ==========================================
# ROUTE: Generate Enhanced Itinerary
# ==========================================
@app.route('/api/generate-itinerary', methods=['POST'])
@login_required
def generate_itinerary():
    if not groq_client:
        return jsonify({'error': 'Groq API key not configured'}), 503

    data         = request.get_json()
    place_name   = data.get('place_name', '').strip()
    state        = data.get('state', '').strip()
    place_type   = data.get('type', '').strip()
    days         = int(data.get('days', 3))
    budget       = float(data.get('budget', 10000))
    travel_style = data.get('travel_style', 'balanced')
    ideal_for    = data.get('ideal_for', 'all')
    source_city  = data.get('source_city', '').strip()   # NEW: starting city

    if not place_name:
        return jsonify({'error': 'Place name required'}), 400
    if not source_city:
        return jsonify({'error': 'Source city required'}), 400

    # Cache check
    cache_key = make_cache_key('itinerary_v2',
        place=place_name, days=days,
        budget=budget, style=travel_style,
        source=source_city
    )
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"Itinerary cache HIT: {place_name}")
        return jsonify({'success': True, 'data': cached, 'cached': True})

    # ---------- HOTEL STYLE GUIDE ----------
    hotel_guide = {
        'budget' : 'budget guesthouses, hostels, dharamshalas under ₹1,500/night',
        'balanced': 'mid-range 3-star hotels ₹2,000–₹5,000/night',
        'luxury' : 'premium 4-5 star resorts and boutique hotels ₹7,000+/night',
    }.get(travel_style, 'mid-range hotels')

    prompt = f"""
You are an elite Indian travel planner. Create a hyper-detailed, professional trip itinerary.

TRIP DETAILS:
- Starting City: {source_city}
- Destination: {place_name}, {state}, India
- Place Type: {place_type}
- Duration: {days} days
- Total Budget: Rs.{budget:,.0f}
- Travel Style: {travel_style} ({hotel_guide})
- Ideal For: {ideal_for}

Return ONLY a valid JSON object (absolutely no markdown, no extra text, no backticks):
{{
  "trip_title": "Catchy trip title with destination name",
  "overview": "2-3 engaging sentences about why this trip is special",
  "source_city": "{source_city}",
  "destination": "{place_name}",
  "state": "{state}",
  "duration_days": {days},
  "travel_style": "{travel_style}",
  "total_budget_range": "Rs.X,XXX - Rs.X,XXX for {days} days per person",

  "transport": {{
    "outward": [
      {{
        "mode": "Flight",
        "icon": "✈️",
        "operator": "IndiGo / Air India / SpiceJet (example)",
        "duration": "Xh Xm",
        "price_range": "Rs.X,XXX - Rs.X,XXX per person",
        "class": "Economy",
        "frequency": "X flights daily",
        "booking_tip": "Book 2-3 weeks in advance on MakeMyTrip or airline website",
        "recommended_for": "luxury / balanced"
      }},
      {{
        "mode": "Train",
        "icon": "🚂",
        "operator": "Specific train name and number if known",
        "duration": "Xh Xm",
        "price_range": "Rs.XXX - Rs.X,XXX per person (Sleeper to AC 2T)",
        "class": "Sleeper / 3AC / 2AC",
        "frequency": "X trains daily",
        "booking_tip": "Book on IRCTC app, tatkal available",
        "recommended_for": "budget / balanced"
      }},
      {{
        "mode": "Bus",
        "icon": "🚌",
        "operator": "State SRTC or private operator name",
        "duration": "Xh Xm",
        "price_range": "Rs.XXX - Rs.X,XXX per person",
        "class": "Sleeper / Volvo AC",
        "frequency": "Multiple daily",
        "booking_tip": "Book on RedBus or AbhiBus",
        "recommended_for": "budget"
      }},
      {{
        "mode": "Private Car",
        "icon": "🚗",
        "operator": "Self-drive or cab aggregator (Ola/Uber outstation)",
        "duration": "Xh Xm",
        "price_range": "Rs.X,XXX - Rs.X,XXX total (fuel + toll)",
        "class": "Hatchback / Sedan / SUV",
        "frequency": "Anytime",
        "booking_tip": "Book Ola/Uber outstation or local taxi, carry cash for tolls",
        "recommended_for": "balanced / luxury / groups"
      }}
    ],
    "return": [
      {{
        "mode": "Flight",
        "icon": "✈️",
        "operator": "IndiGo / Air India / SpiceJet (example)",
        "duration": "Xh Xm",
        "price_range": "Rs.X,XXX - Rs.X,XXX per person",
        "class": "Economy",
        "booking_tip": "Book return together with outward for discounts"
      }},
      {{
        "mode": "Train",
        "icon": "🚂",
        "operator": "Specific train name",
        "duration": "Xh Xm",
        "price_range": "Rs.XXX - Rs.X,XXX per person",
        "class": "Sleeper / 3AC / 2AC",
        "booking_tip": "Book on IRCTC, check return availability early"
      }},
      {{
        "mode": "Bus",
        "icon": "🚌",
        "operator": "State SRTC or private operator",
        "duration": "Xh Xm",
        "price_range": "Rs.XXX - Rs.X,XXX per person",
        "booking_tip": "Return buses available daily"
      }},
      {{
        "mode": "Private Car",
        "icon": "🚗",
        "operator": "Local taxi or cab",
        "duration": "Xh Xm",
        "price_range": "Rs.X,XXX - Rs.X,XXX total",
        "booking_tip": "Hire local driver at destination for return, often cheaper"
      }}
    ]
  }},

  "hotels": [
    {{
      "name": "Hotel Name 1",
      "area": "Locality/Area name",
      "stars": 4,
      "price_per_night": "Rs.X,XXX",
      "style_match": "{travel_style}",
      "highlights": ["Swimming pool", "Free breakfast", "Free parking"],
      "rating": "4.3",
      "why_choose": "One sentence why this hotel is great",
      "booking_platforms": ["MakeMyTrip", "Booking.com", "Hotel website"]
    }},
    {{
      "name": "Hotel Name 2",
      "area": "Locality/Area name",
      "stars": 4,
      "price_per_night": "Rs.X,XXX",
      "style_match": "{travel_style}",
      "highlights": ["Sea view", "Restaurant", "Spa"],
      "rating": "4.1",
      "why_choose": "One sentence why this hotel is great",
      "booking_platforms": ["Goibibo", "Agoda"]
    }},
    {{
      "name": "Hotel Name 3",
      "area": "Locality/Area name",
      "stars": 3,
      "price_per_night": "Rs.X,XXX",
      "style_match": "{travel_style}",
      "highlights": ["Central location", "AC rooms", "WiFi"],
      "rating": "3.9",
      "why_choose": "One sentence why this hotel is great",
      "booking_platforms": ["MakeMyTrip", "OYO"]
    }},
    {{
      "name": "Hotel Name 4",
      "area": "Locality/Area name",
      "stars": 3,
      "price_per_night": "Rs.X,XXX",
      "style_match": "{travel_style}",
      "highlights": ["Budget-friendly", "Clean rooms", "Good reviews"],
      "rating": "3.7",
      "why_choose": "One sentence why this hotel is great",
      "booking_platforms": ["OYO", "Zostel"]
    }},
    {{
      "name": "Hotel Name 5",
      "area": "Locality/Area name",
      "stars": 5,
      "price_per_night": "Rs.X,XXX",
      "style_match": "{travel_style}",
      "highlights": ["Luxury amenities", "Fine dining", "Concierge"],
      "rating": "4.7",
      "why_choose": "One sentence why this hotel is great",
      "booking_platforms": ["Taj Hotels", "MakeMyTrip"]
    }}
  ],

  "days": [
    {{
      "day": 1,
      "title": "Day 1: Arrival & First Impressions",
      "theme": "Arrival, settle in, light exploration",
      "schedule": [
        {{
          "time": "06:00 AM",
          "type": "transport",
          "activity": "Depart from {source_city}",
          "details": "Head to airport/railway station. Reach 1 hour early for domestic flights.",
          "cost": "Rs.XXX (airport/station transfer)",
          "icon": "🚕"
        }},
        {{
          "time": "10:00 AM",
          "type": "arrival",
          "activity": "Arrive at {place_name}",
          "details": "Collect luggage, hire local transport to hotel",
          "cost": "Rs.XXX (local transfer)",
          "icon": "📍"
        }},
        {{
          "time": "11:00 AM",
          "type": "hotel",
          "activity": "Hotel check-in & freshen up",
          "details": "Early check-in may be available for extra charge, else store luggage",
          "cost": "Included in hotel",
          "icon": "🏨"
        }},
        {{
          "time": "12:30 PM",
          "type": "restaurant",
          "activity": "Lunch at Famous Local Restaurant",
          "place": "Restaurant name with area",
          "cuisine": "Local cuisine type",
          "must_try": ["Dish 1", "Dish 2"],
          "cost": "Rs.XXX per person",
          "icon": "🍽️"
        }},
        {{
          "time": "02:00 PM",
          "type": "activity",
          "activity": "Visit first attraction name",
          "place": "Exact place name",
          "details": "What to do/see there, tips to know",
          "duration": "2 hours",
          "cost": "Rs.XXX entry + auto",
          "icon": "🏛️"
        }},
        {{
          "time": "05:00 PM",
          "type": "activity",
          "activity": "Evening activity / viewpoint",
          "place": "Place name",
          "details": "What makes this special at this time",
          "duration": "1.5 hours",
          "cost": "Rs.XXX",
          "icon": "🌅"
        }},
        {{
          "time": "07:30 PM",
          "type": "restaurant",
          "activity": "Dinner at popular local spot",
          "place": "Restaurant name",
          "cuisine": "Cuisine type",
          "must_try": ["Dish 1", "Dish 2"],
          "cost": "Rs.XXX per person",
          "icon": "🌙"
        }},
        {{
          "time": "09:30 PM",
          "type": "rest",
          "activity": "Return to hotel, rest",
          "details": "Good night's sleep for next day",
          "cost": "Free",
          "icon": "😴"
        }}
      ],
      "day_total_cost": "Rs.X,XXX approx",
      "insider_tip": "One golden tip for this day"
    }}
  ],

  "packing_list": {{
    "essentials": ["Aadhar card/Passport", "Travel insurance", "Cash + Cards"],
    "clothing": ["Item 1 specific to destination climate", "Item 2"],
    "gear": ["Item relevant to activities"],
    "medicines": ["Basic first aid", "ORS packets", "Any personal medication"]
  }},

  "budget_breakdown": {{
    "transport_one_way": "Rs.X,XXX - Rs.X,XXX per person",
    "transport_return": "Rs.X,XXX - Rs.X,XXX per person",
    "accommodation_total": "Rs.X,XXX - Rs.X,XXX for {days} nights",
    "food_total": "Rs.X,XXX - Rs.X,XXX for {days} days",
    "activities_total": "Rs.X,XXX - Rs.X,XXX",
    "local_transport": "Rs.X,XXX",
    "miscellaneous": "Rs.XXX - Rs.X,XXX",
    "grand_total": "Rs.X,XXX - Rs.X,XXX per person"
  }},

  "emergency_contacts": [
    {{"name": "Police", "number": "100"}},
    {{"name": "Ambulance", "number": "108"}},
    {{"name": "Tourist Helpline", "number": "1363"}},
    {{"name": "Women Helpline", "number": "1091"}}
  ],

  "best_time_reminder": "One line about when to visit and why",
  "getting_there_summary": "2 sentences on best way to reach from {source_city}"
}}

STRICT RULES:
1. Generate EXACTLY {days} day objects in the days array, with each day having 6-8 schedule items
2. RESTAURANTS — THIS IS CRITICAL: Always suggest REAL, FAMOUS, HIGHLY-RATED restaurants, dhabas, 
   and cafes that are actually well-known in {place_name}, {state}. 
   Examples of the QUALITY expected:
   - Goa → Britto's, Fisherman's Wharf, Vinayak Family Restaurant, A Reverie
   - Jaipur → Laxmi Mishtan Bhandar (LMB), Suvarna Mahal, Peacock Rooftop Restaurant
   - Mumbai → Leopold Cafe, Britannia & Co., Trishna, Bademiya
   - Manali → Johnson's Cafe, Drifter's Inn, Cafe 1947
   DO NOT use generic names like "Famous Local Restaurant" or "Popular Eatery" — use ACTUAL place names.
   If you are not certain of a specific restaurant, describe it precisely: 
   e.g. "a popular rooftop cafe near Mall Road known for momos and maggi"
3. Each restaurant entry MUST include:
   - Actual place name (not placeholder)
   - Exact area/locality  
   - 2 signature dishes to try
   - Approximate cost per person in Rs.
4. All prices in Indian Rupees (Rs.) only
5. Each day must have morning, afternoon and evening covered with times
6. Transport prices should reflect real approximate Indian market rates
7. Hotels should match the {travel_style} style exactly
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        raw = response.choices[0].message.content
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()

        itinerary_data = json.loads(raw)

        # Cache karo (24 hours)
        cache_set(cache_key, itinerary_data, ttl=86400)

        # DB mein save karo
        new_itinerary = Itinerary(
            user_id        = current_user.id,
            title          = itinerary_data.get('trip_title', f'{place_name} Trip'),
            place_name     = place_name,
            state          = state,
            days           = days,
            budget         = budget,
            travel_style   = travel_style,
            itinerary_json = json.dumps(itinerary_data),
        )
        db.session.add(new_itinerary)
        db.session.commit()

        logger.info(f"Enhanced itinerary generated for {place_name} by user {current_user.id}")
        return jsonify({'success': True, 'data': itinerary_data, 'itinerary_id': new_itinerary.id})

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}\nRaw: {raw[:500]}")
        return jsonify({'error': 'AI response format error, please try again'}), 500
    except Exception as e:
        logger.error(f"generate_itinerary error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==========================================
# ROUTE: Export Itinerary as Professional PDF
# ==========================================
@app.route('/itinerary/<int:itin_id>/export-pdf')
@login_required
def export_itinerary_pdf(itin_id):
    """Generate and download a professional PDF itinerary"""
    from pdf_generator import generate_itinerary_pdf
    import io

    itin = Itinerary.query.filter_by(id=itin_id, user_id=current_user.id).first_or_404()
    data = json.loads(itin.itinerary_json)

    # Generate PDF in memory
    pdf_buffer = io.BytesIO()
    generate_itinerary_pdf(data, pdf_buffer)
    pdf_buffer.seek(0)

    filename = f"BharatYatra_{itin.place_name.replace(' ', '_')}_{itin.days}Days.pdf"

    from flask import send_file
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

# ==========================================
# ROUTE: History & Saved Places Page
# ==========================================
@app.route('/history')
@login_required
def history():
    saved   = SavedPlace.query.filter_by(user_id=current_user.id)\
                .order_by(SavedPlace.saved_at.desc()).all()
    searches = SearchHistory.query.filter_by(user_id=current_user.id)\
                .order_by(SearchHistory.searched_at.desc()).limit(20).all()
    itins   = Itinerary.query.filter_by(user_id=current_user.id)\
                .order_by(Itinerary.created_at.desc()).limit(10).all()

    return render_template('history.html',
        name       = current_user.name,
        saved      = saved,
        searches   = searches,
        itineraries= itins
    )


# ==========================================
# ROUTE: Delete Saved Place
# ==========================================
@app.route('/api/delete-saved/<int:save_id>', methods=['DELETE'])
@login_required
def delete_saved(save_id):
    item = SavedPlace.query.filter_by(id=save_id, user_id=current_user.id).first()
    if not item:
        return jsonify({'error': 'Not found'}), 404
    db.session.delete(item)
    db.session.commit()
    return jsonify({'success': True})


# ==========================================
# ROUTE: Get Itinerary by ID
# ==========================================
@app.route('/itinerary/<int:itin_id>')
@login_required
def view_itinerary(itin_id):
    itin = Itinerary.query.filter_by(id=itin_id, user_id=current_user.id).first_or_404()
    data = json.loads(itin.itinerary_json)
    return render_template('itinerary.html',
        name      = current_user.name,
        itinerary = data,
        meta      = itin
    )


# ==========================================
# AUTH ROUTES
# ==========================================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not name or not email or not password:
            flash('Sabhi fields bharna zaroori hai.', 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Yeh email pehle se registered hai.', 'error')
            return redirect(url_for('signup'))

        new_user = User(name=name, email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash('Account bana! Ab login karo.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email aur password dono chahiye.', 'error')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Email ya password galat hai.', 'error')
            return redirect(url_for('login'))

        login_user(user)
        return redirect(url_for('index'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Successfully logout ho gaye!', 'success')
    return redirect(url_for('login'))


# ==========================================
# MAIN ROUTE
# ==========================================
# ==========================================
# MAIN ROUTE
# ==========================================
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    from flask import session

    states = []
    if not df_master.empty and 'State' in df_master.columns:
        states = sorted(df_master['State'].unique())

    if request.method == 'POST':
        state     = request.form.get('state', 'All India').strip() or 'All India'
        interests = request.form.get('interests', 'nature').strip() or 'nature'
        try:
            budget = float(request.form.get('budget', 5000))
        except (ValueError, TypeError):
            budget = 5000.0

        results_df = get_recommendations(state, budget, interests)

        if not results_df.empty:
            results = results_df.to_dict('records')

            history_entry = SearchHistory(
                user_id       = current_user.id,
                state         = state,
                budget        = budget,
                interests     = interests,
                results_count = len(results)
            )
            db.session.add(history_entry)
            db.session.commit()

            # Sirf zaroori fields session mein save karo
            session['search_results'] = [
                {
                    'Place Name'    : r['Place Name'],
                    'State'         : r['State'],
                    'City'          : r.get('City', ''),
                    'Type'          : r['Type'],
                    'Best Visit Time': r['Best Visit Time'],
                    'Ideal For'     : r['Ideal For'],
                    'Trip Cost'     : r.get('Trip Cost', ''),
                    'Stay Duration' : r.get('Stay Duration', '2-3 days'),
                    'max_budget'    : int(r['max_budget']),
                    'Score'         : float(r['Score']),
                }
                for r in results
            ]
        else:
            session.pop('search_results', None)
            flash('Koi result nahi mila. Budget ya filters adjust karo.', 'error')

        return redirect(url_for('index'))  # ← BLINK FIX

    # GET request
    results = session.pop('search_results', None)

    return render_template('index.html',
        states  = states,
        results = results,
        name    = current_user.name
    )

# ==========================================
# CREATE TABLES AND RUN
# ==========================================
from app import db, app
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
