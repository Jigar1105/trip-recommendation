from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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

db = SQLAlchemy(app)
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
def load_and_clean():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'cleaned_travel_data_unique.csv')
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        df['Type']            = df['Type'].fillna('general').str.lower()
        df['Best Visit Time'] = df['Best Visit Time'].fillna('Year-round')
        df['State']           = df['State'].fillna('India').str.upper()
        df['Ideal For']       = df['Ideal For'].fillna('all').str.lower()
        df['Place Name']      = df['Place Name'].fillna('').str.strip()

        if 'City' not in df.columns:
            df['City'] = ''
        else:
            df['City'] = df['City'].fillna('').str.strip()

        if 'Stay Duration' not in df.columns:
            df['Stay Duration'] = '2-3 days'
        else:
            df['Stay Duration'] = df['Stay Duration'].fillna('2-3 days')

        def clean_cost(val):
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            return int(nums[-1]) if nums else 0

        df['max_budget'] = df['Trip Cost'].apply(clean_cost)
        df['content'] = (
            df['Type'] + " " +
            df['Place Name'].str.lower() + " " +
            df['Best Visit Time'].str.lower() + " " +
            df['Ideal For']
        )
        return df
    except FileNotFoundError:
        logger.error("final_travel_data_1000.csv not found!")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


df_master = load_and_clean()


# ==========================================
# KEYWORD EXPANSION
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
            expanded += " " + val
    return expanded


# ==========================================
# RECOMMENDATION ENGINE
# ==========================================
def get_recommendations(state, budget, interests):
    if df_master.empty:
        return pd.DataFrame()

    df   = df_master.copy()
    mask = df['max_budget'] <= float(budget)
    if state != "All India":
        mask = mask & (df['State'].str.upper() == state.strip().upper())

    filtered = df[mask].copy()
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.reset_index(drop=True)
    expanded_query = expand_keywords(interests)

    vec       = TfidfVectorizer(stop_words='english')
    matrix    = vec.fit_transform(filtered['content'])
    query_vec = vec.transform([expanded_query])

    n_neighbors = min(len(filtered), 15)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(matrix)
    distances, indices = knn.kneighbors(query_vec)

    scores = np.zeros(len(filtered))
    for i, idx in enumerate(indices[0]):
        scores[idx] = 1 - distances[0][i]

    filtered['Score'] = scores
    return filtered[filtered['Score'] > 0].sort_values(
        by=['Score', 'Place Name'], ascending=[False, True]
    )


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
# ROUTE: Generate Itinerary
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
    travel_style = data.get('travel_style', 'balanced')  # budget / balanced / luxury
    ideal_for    = data.get('ideal_for', 'all')

    if not place_name:
        return jsonify({'error': 'Place name required'}), 400

    # Cache check
    cache_key = make_cache_key('itinerary',
        place=place_name, days=days,
        budget=budget, style=travel_style
    )
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"Itinerary cache HIT: {place_name}")
        return jsonify({'success': True, 'data': cached, 'cached': True})

    prompt = f"""
You are an expert Indian travel planner. Create a detailed day-by-day itinerary.

Destination: {place_name}, {state}, India
Type: {place_type}
Duration: {days} days
Total Budget: ₹{budget:,.0f}
Travel Style: {travel_style} (budget=cheap stays/dhabas, balanced=mid-range, luxury=premium)
Ideal For: {ideal_for}

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "trip_title": "Catchy trip title",
  "overview": "2-3 sentences about this trip",
  "total_estimated_cost": "₹X,XXX - ₹X,XXX",
  "best_time_reminder": "One line reminder about when to visit",
  "days": [
    {{
      "day": 1,
      "title": "Day title (e.g. Arrival & Exploration)",
      "morning": {{
        "activity": "What to do",
        "place": "Specific place name",
        "tip": "Quick tip",
        "cost": "₹XXX approx"
      }},
      "afternoon": {{
        "activity": "What to do",
        "place": "Specific place name",
        "tip": "Quick tip",
        "cost": "₹XXX approx"
      }},
      "evening": {{
        "activity": "What to do",
        "place": "Specific place name",
        "tip": "Quick tip",
        "cost": "₹XXX approx"
      }},
      "stay": "Hotel/stay recommendation with price range",
      "food": ["breakfast suggestion", "lunch suggestion", "dinner suggestion"],
      "day_budget": "₹X,XXX approx"
    }}
  ],
  "packing_tips": ["tip 1", "tip 2", "tip 3"],
  "important_contacts": ["Local police: 100", "Tourist helpline: 1363"],
  "getting_there": "How to reach {place_name} from major cities"
}}

Generate exactly {days} day objects in the days array.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
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

        logger.info(f"Itinerary generated for {place_name} by user {current_user.id}")
        return jsonify({'success': True, 'data': itinerary_data, 'itinerary_id': new_itinerary.id})

    except Exception as e:
        logger.error(f"generate_itinerary error: {str(e)}")
        return jsonify({'error': str(e)}), 500


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
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    states = []
    if not df_master.empty and 'State' in df_master.columns:
        states = sorted(df_master['State'].unique())

    results = None

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

            # ✅ Search history save karo
            history_entry = SearchHistory(
                user_id       = current_user.id,
                state         = state,
                budget        = budget,
                interests     = interests,
                results_count = len(results)
            )
            db.session.add(history_entry)
            db.session.commit()
        else:
            flash('Koi result nahi mila. Budget ya filters adjust karo.', 'error')

    return render_template('index.html',
        states  = states,
        results = results,
        name    = current_user.name
    )


# ==========================================
# CREATE TABLES AND RUN
# ==========================================
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
