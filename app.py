from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np  # <-- NEW: Numpy added for array operations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors  # <-- NEW: KNN Algorithm imported
from sklearn.metrics.pairwise import cosine_similarity
# from dataset_helper import generate_budget_graph

import pymysql
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# ==========================================
# 🛑 AUTO-CREATE DATABASE LOGIC (MySQL) 🛑
# ==========================================
def create_database_if_not_exists():
    try:
        conn = pymysql.connect(host='localhost', user='root', password='')
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS bharat_yatra")
        conn.commit()
        cursor.close()
        conn.close()
        print("Database 'bharat_yatra' is ready!")
    except Exception as e:
        print(f"Error creating database: {e}")

create_database_if_not_exists()

# ==========================================
# 🛑 FLASK-SQLALCHEMY SETUP 🛑
# ==========================================
app.config['SECRET_KEY'] = 'super_secret_key_123' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/bharat_yatra' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- 1. DATA LOADING & CLEANING (UPDATED FOR ML) ---
def load_and_clean():
    try:
        df = pd.read_csv('final_travel_data_1000.csv')
        df.columns = df.columns.str.strip()
        df['Type'] = df['Type'].fillna('general').str.lower()
        df['Best Visit Time'] = df['Best Visit Time'].fillna('year-round').str.lower()
        df['State'] = df['State'].fillna('India').str.upper()
        df['Ideal For'] = df['Ideal For'].fillna('all').str.lower()
        
        # Budget cleaning - Extracting max budget
        def clean_cost(val):
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            return int(nums[-1]) if nums else 0

        df['max_budget'] = df['Trip Cost'].apply(clean_cost)
        
        # NEW: Super-Content string for better ML pattern matching
        df['content'] = (
            df['Type'] + " " + 
            df['Place Name'].fillna('').str.lower() + " " + 
            df['Best Visit Time'] + " " + 
            df['Ideal For']
        )
        return df
    except FileNotFoundError:
        print("Error: final_travel_data_1000.csv file not found!")
        return pd.DataFrame()

df_master = load_and_clean()

# --- NEW: KEYWORD EXPANSION LOGIC ---
def expand_keywords(user_type):
    """ Nearest Word Logic to prevent 'Not Found' """
    synonyms = {
        'trekking': 'adventure mountain hiking climbing',
        'beach': 'sea water coastal ocean sand',
        'nature': 'greenery forest hills scenic waterfall lake',
        'spiritual': 'temple religious divine holy shrine',
        'history': 'fort monument ancient heritage palace museum',
        'wildlife': 'animal safari tiger bird sanctuary national park'
    }
    
    expanded = str(user_type).lower()
    for key, val in synonyms.items():
        if key in expanded:
            expanded += " " + val
    return expanded


# --- 2. RECOMMENDATION LOGIC (UPDATED WITH KNN & NO LIMIT) ---
def get_recommendations(month, state, budget, interests):
    if df_master.empty: return pd.DataFrame()
    
    df = df_master.copy()
    
    # Primary Filters (Hard Rules: Budget & State)
    mask = (df['max_budget'] <= float(budget))
    if state != "All India":
        mask = mask & (df['State'].str.lower() == state.lower())
        
    filtered = df[mask].copy()
    if filtered.empty: 
        return pd.DataFrame() 

    filtered = filtered.reset_index(drop=True)

    # ML Logic: 1. Expand User Query
    expanded_query = expand_keywords(interests) + " " + month.lower()

    # ML Logic: 2. Vectorization
    vec = TfidfVectorizer(stop_words='english')
    matrix = vec.fit_transform(filtered['content'])
    query_vec = vec.transform([expanded_query])
    
    # ML Logic: 3. KNN Model
    n_neighbors = min(len(filtered), 10) # Find up to 10 closest geometric points
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(matrix)
    distances, indices = knn.kneighbors(query_vec)
    
    # ML Logic: 4. Score Calculation
    scores = np.zeros(len(filtered))
    for i, idx in enumerate(indices[0]):
        scores[idx] = 1 - distances[0][i]

    # Assign score (HTML is using 'Score' column)
    filtered['Score'] = scores

    # Strict Sorting (No Randomness) & ALL MATCHES (Score > 0)
    # Notice: .head(12) is REMOVED so it shows all matching items
    final_results = filtered[filtered['Score'] > 0].sort_values(
        by=['Score', 'Place Name'], ascending=[False, True]
    )
    
    return final_results


# ==========================================
# AUTHENTICATION ROUTES (Unchanged)
# ==========================================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists. Please login.', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(name=name, email=email, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.', 'error')
            return redirect(url_for('login'))

        login_user(user)
        return redirect(url_for('index'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required 
def logout():
    logout_user()
    return redirect(url_for('login'))


# --- 3. MAIN ROUTE (Unchanged) ---
@app.route('/', methods=['GET', 'POST'])
@login_required 
def index():
    states = []
    if not df_master.empty and 'State' in df_master.columns:
        states = sorted(df_master['State'].unique())
        
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    results = None
    graph = None

    if request.method == 'POST':
        month = request.form.get('month')
        if not month: month = "year-round"
        
        state = request.form.get('state')
        if not state: state = "All India"
        
        budget_input = request.form.get('budget')
        budget = budget_input if budget_input and str(budget_input).strip() != "" else 5000
        
        interests = request.form.get('interests')
        if not interests: interests = "nature"
        
        results_df = get_recommendations(month, state, budget, interests)
        
        if not results_df.empty:
            results = results_df.to_dict('records') # HTML format untouched
            graph = None
    return render_template('index.html', states=states, months=months, results=results, graph=graph, name=current_user.name)

# ==========================================
# CREATE TABLES AND RUN App
# ==========================================
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)