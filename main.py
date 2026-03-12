import json
import os
import hashlib
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from typing import List, Dict, Tuple

# ML
try:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: Install ML: pip install numpy scikit-learn")

import sqlite3

# ==============================
# Configuration
# ==============================
DB_FILE = "melodia.db"
LANGS = ["English", "Spanish", "Hindi", "Korean", "Japanese", "French", "German"]
GENRES = ["Pop", "Rock", "Hip Hop", "Jazz", "Classical", "Electronic", "R&B", "Country", "Indie", "Dance", "Alternative", "Funk"]
MOODS = ["Happy", "Sad", "Energetic", "Chill", "Romantic", "Melancholic", "Intense", "Peaceful"]

# ==============================
# Database Functions
# ==============================
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    try:
        conn = db()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_email ON users(email)")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_token ON tokens(token)")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            genre TEXT DEFAULT 'Pop',
            language TEXT DEFAULT 'English',
            mood TEXT DEFAULT 'Chill',
            bpm INTEGER DEFAULT 110,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_genre ON songs(genre)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_language ON songs(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood ON songs(mood)")

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✓ Database initialized: {DB_FILE}")

    except Exception as e:
        print(f"Database error: {e}")

def load_initial_songs():
    try:
        conn = db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM songs")
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"✓ Database has {count} songs already")
            cursor.close()
            conn.close()
            return
        
        cursor.executemany("""
        INSERT INTO songs(title, artist, genre, language, mood, bpm)
        VALUES (?, ?, ?, ?, ?, ?)
        """, DEFAULT_SONGS)
        
        conn.commit()
        print(f"✓ Loaded {len(DEFAULT_SONGS)} default songs")
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error loading songs: {e}")

def get_all_songs() -> List[Dict]:
    try:
        conn = db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM songs ORDER BY artist, title")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        songs = [{
            "id": r[0],
            "title": r[1],
            "artist": r[2],
            "genre": r[3],
            "language": r[4],
            "mood": r[5],
            "bpm": r[6]
        } for r in rows]
        
        return songs
    except Exception as e:
        print(f"Error fetching songs: {e}")
        return []

def get_unique_artists() -> List[str]:
    try:
        conn = db()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT artist FROM songs ORDER BY artist COLLATE NOCASE")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows if row[0]]
    except Exception as e:
        print(f"Error fetching artists: {e}")
        return []

# ==============================
# User Authentication
# ==============================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    try:
        conn = db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE email = ?", (email.lower().strip(),))
        if cursor.fetchone():
            return False, "Email already exists"
        
        hashed = hash_password(password)
        cursor.execute("INSERT INTO users(name, email, password) VALUES(?, ?, ?)",
                      (name.strip(), email.lower().strip(), hashed))
        conn.commit()
        cursor.close()
        conn.close()
        return True, "Account created"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def authenticate(email: str, password: str) -> Tuple[bool, str, Dict]:
    try:
        conn = db()
        cursor = conn.cursor()
        
        hashed = hash_password(password)
        cursor.execute("SELECT id, name, email FROM users WHERE email=? AND password=?",
                      (email.lower().strip(), hashed))
        row = cursor.fetchone()
        
        if not row:
            return False, "Invalid credentials", {}
        
        user = {"id": row[0], "name": row[1], "email": row[2]}
        token = secrets.token_urlsafe(32)
        
        cursor.execute("INSERT INTO tokens(user_id, token) VALUES(?, ?)",
                      (user["id"], token))
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, token, user
        
    except Exception as e:
        return False, f"Error: {str(e)}", {}

def profile_from_token(token: str) -> Dict:
    try:
        conn = db()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT u.id, u.name, u.email FROM tokens t 
        JOIN users u ON u.id=t.user_id WHERE t.token=?
        """, (token,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            return {"id": row[0], "name": row[1], "email": row[2]}
        return {}
        
    except Exception as e:
        return {}

# ==============================
# Recommendation Engine
# ==============================
def mood_vector(mood: str) -> Tuple[float, float]:
    moods = {
        "happy": (0.7, 0.9), "energetic": (0.9, 0.7), "chill": (0.3, 0.6),
        "romantic": (0.5, 0.8), "melancholic": (0.3, 0.2), "intense": (0.95, 0.4),
        "peaceful": (0.2, 0.8), "sad": (0.2, 0.1)
    }
    return moods.get(mood.lower(), (0.5, 0.5))

class Recommender:
    def __init__(self):
        self.songs = []
        self.matrix = None
        self.nn = None
        self.scaler = None
        self._fitted = False

    def load(self):
        if self.songs:
            return
        
        self.songs = get_all_songs()
        if not self.songs:
            print("No songs in database")
            return
        
        self._fit()

    def _fit(self):
        if self._fitted or not ML_AVAILABLE or not self.songs:
            return
        
        try:
            feats = []
            for t in self.songs:
                e, v = mood_vector(t.get("mood", "Chill"))
                bpm_norm = t.get("bpm", 110) / 220.0
                genre_vec = [1.0 if t.get("genre") == g else 0.0 for g in GENRES]
                lang_vec = [1.0 if t.get("language") == l else 0.0 for l in LANGS]
                feats.append([bpm_norm, e, v] + genre_vec + lang_vec)
            
            X = np.array(feats, dtype=float)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.matrix = X_scaled
            
            n_neighbors = min(50, len(self.songs))
            self.nn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
            self.nn.fit(self.matrix)
            self._fitted = True
            print(f"ML model fitted on {len(self.songs)} songs")
        except Exception as e:
            print(f"ML error: {e}")

    def recommend(self, languages: List[str], genres: List[str], moods: List[str], artists: List[str]) -> List[Dict]:
        self.load()
        
        if not self.songs:
            return []
        
        # First, filter candidates based on all criteria
        candidates = []
        for i, t in enumerate(self.songs):
            # If artist is selected, ALWAYS filter by artist (don't relax this)
            if artists:
                artist_name = t.get("artist", "")
                if not any(a.lower() in artist_name.lower() for a in artists):
                    continue
            # Apply other filters
            if languages and t.get("language") not in languages:
                continue
            if genres and t.get("genre") not in genres:
                continue
            if moods and t.get("mood") not in moods:
                continue
            candidates.append((i, t))
        
        # If no candidates match filters, relax non-artist constraints only
        use_all_songs = False
        if not candidates and not artists:
            # Only relax filters if NO artist was selected
            for i, t in enumerate(self.songs):
                if languages and t.get("language") not in languages:
                    continue
                if genres and t.get("genre") not in genres:
                    continue
                if moods and t.get("mood") not in moods:
                    continue
                candidates.append((i, t))
        
        # If still no candidates, use all songs only if no artist was selected
        if not candidates and not artists:
            candidates = [(i, t) for i, t in enumerate(self.songs)]
            use_all_songs = True
        
        # If artist was selected but still no results, return empty (don't show other artists)
        if not candidates and artists:
            return []
        
        # Calculate scores for each candidate
        scored_songs = []
        for idx, song in candidates:
            score = 0.0
            song_artist = song.get("artist", "").lower()
            song_genre = song.get("genre", "")
            song_language = song.get("language", "")
            song_mood = song.get("mood", "")
            
            # Artist match: Highest priority (50% of score)
            artist_match = False
            exact_artist_match = False
            if artists:
                artist_match = any(a.lower() in song_artist for a in artists)
                exact_artist_match = any(a.lower() == song_artist for a in artists)
                if exact_artist_match:
                    score += 0.5  # Exact match gets full artist points
                elif artist_match:
                    score += 0.45  # Partial match (artist name contains)
                elif use_all_songs:
                    score -= 0.3  # Penalty for non-matching artist when using all songs
            else:
                # If no artist selected, give neutral base
                score += 0.25
            
            # Genre match: Important (20% of score)
            if genres:
                if song_genre in genres:
                    score += 0.20
                elif use_all_songs:
                    score -= 0.15  # Penalty for non-matching genre
            else:
                score += 0.10  # Neutral if no genre selected
            
            # Language match: Important (15% of score)
            if languages:
                if song_language in languages:
                    score += 0.15
                elif use_all_songs:
                    score -= 0.10  # Penalty for non-matching language
            else:
                score += 0.075  # Neutral if no language selected
            
            # Mood match: Important (15% of score)
            if moods:
                if song_mood in moods:
                    score += 0.15
                else:
                    # Partial mood similarity - check if mood is similar
                    song_mood_vec = mood_vector(song_mood)
                    max_similarity = 0.0
                    for mood in moods:
                        mood_vec = mood_vector(mood)
                        # Euclidean distance normalized
                        dist = ((song_mood_vec[0] - mood_vec[0])**2 + (song_mood_vec[1] - mood_vec[1])**2)**0.5
                        similarity = max(0.0, 1.0 - dist / 1.5)
                        max_similarity = max(max_similarity, similarity)
                    if max_similarity > 0.4:
                        score += 0.05 * max_similarity  # Partial credit for similar moods
                    elif use_all_songs:
                        score -= 0.10  # Penalty for non-matching mood
            else:
                score += 0.075  # Neutral if no mood selected
            
            # ML-based similarity score if available (0.1 points bonus)
            ml_bonus = 0.0
            if ML_AVAILABLE and self.nn is not None and self.scaler is not None:
                try:
                    # Get song's feature vector
                    e, v = mood_vector(song_mood)
                    bpm_norm = song.get("bpm", 110) / 220.0
                    genre_vec = [1.0 if song_genre == g else 0.0 for g in GENRES]
                    lang_vec = [1.0 if song_language == l else 0.0 for l in LANGS]
                    song_features = np.array([[bpm_norm, e, v] + genre_vec + lang_vec], dtype=float)
                    song_features_scaled = self.scaler.transform(song_features)
                    
                    # Build query vector from user preferences
                    if moods:
                        mood_vecs = [mood_vector(m) for m in moods]
                        e_query = sum(v[0] for v in mood_vecs) / len(mood_vecs)
                        v_query = sum(v[1] for v in mood_vecs) / len(mood_vecs)
                    else:
                        e_query, v_query = 0.5, 0.5
                    
                    # Use average BPM of selected preferences or default
                    avg_bpm = 110.0
                    if moods:
                        # Estimate BPM from moods
                        bpm_map = {"Energetic": 130, "Happy": 120, "Romantic": 75, "Chill": 85, 
                                  "Melancholic": 70, "Intense": 140, "Peaceful": 65, "Sad": 65}
                        bpm_values = [bpm_map.get(m, 110) for m in moods if m in bpm_map]
                        if bpm_values:
                            avg_bpm = sum(bpm_values) / len(bpm_values)
                    bpm_norm_query = avg_bpm / 220.0
                    
                    genre_vec_query = [1.0 if g in genres else 0.0 for g in GENRES] if genres else [0.0] * len(GENRES)
                    lang_vec_query = [1.0 if l in languages else 0.0 for l in LANGS] if languages else [0.0] * len(LANGS)
                    
                    query_features = np.array([[bpm_norm_query, e_query, v_query] + genre_vec_query + lang_vec_query], dtype=float)
                    query_features_scaled = self.scaler.transform(query_features)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(song_features_scaled, query_features_scaled)[0][0]
                    ml_bonus = max(0.0, similarity * 0.1)
                except Exception as e:
                    pass  # If ML fails, continue without bonus
            
            score += ml_bonus
            
            # Ensure score is non-negative
            score = max(0.0, score)
            
            # Calculate match percentage based on actual match quality
            # Count how many filters are active
            active_filters = 0
            if artists: active_filters += 1
            if genres: active_filters += 1
            if languages: active_filters += 1
            if moods: active_filters += 1
            
            # Calculate base percentage from score
            # Perfect match with all filters would score around 1.0
            # Scale to percentage: score * 100, but cap meaningful range
            if active_filters > 0:
                # When filters are active, scale based on filter match quality
                percentage_score = min(100, score * 100)
            else:
                # When no filters, use ML similarity or base score
                percentage_score = min(100, score * 80)  # Slightly lower base
            
            scored_songs.append({
                **song,
                "score": float(score),
                "percentage": float(percentage_score)
            })
        
        # Sort by score (descending)
        scored_songs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 5-7 songs (prefer 5, but allow up to 7 if scores are very close)
        if len(scored_songs) == 0:
            return []
        
        # Take top 5, but include more if scores are very similar (within 5% of 5th place)
        top_count = min(5, len(scored_songs))
        if len(scored_songs) > 5:
            fifth_score = scored_songs[4]["score"]
            # Include songs with score within 0.05 of the 5th place (up to 7 total)
            for i in range(5, min(7, len(scored_songs))):
                if scored_songs[i]["score"] >= fifth_score - 0.05:
                    top_count = i + 1
                else:
                    break
        
        top_songs = scored_songs[:top_count]
        
        # Ensure percentages are reasonable (at least 8% for displayed songs)
        # This prevents very low percentages that don't make sense visually
        for song in top_songs:
            if song["percentage"] < 8:
                song["percentage"] = max(8, song["percentage"] * 1.2)
        
        return top_songs

recommender = Recommender()

# ==============================
# HTTP Server
# ==============================
class App(BaseHTTPRequestHandler):
    def _set(self, status=200, content_type="text/html"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/project.html"):
                with open("project.html", "rb") as f:
                    self._set(200, "text/html; charset=utf-8")
                    self.wfile.write(f.read())
            elif path == "/view.css":
                with open("view.css", "rb") as f:
                    self._set(200, "text/css; charset=utf-8")
                    self.wfile.write(f.read())
            else:
                self._set(404)
        except FileNotFoundError:
            self._set(404)
        except Exception as e:
            print(f"Error: {e}")
            self._set(500)

    def _json(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode('utf-8'))
        except:
            return {}

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/signup":
                payload = self._json()
                ok, msg = create_user(payload.get("name", ""), payload.get("email", ""), payload.get("password", ""))
                
                if not ok:
                    self._set(400, "application/json")
                    self.wfile.write(json.dumps({"success": False, "error": msg}).encode())
                    return
                
                ok, token, prof = authenticate(payload.get("email", ""), payload.get("password", ""))
                self._set(200, "application/json")
                self.wfile.write(json.dumps({
                    "success": True,
                    "message": msg,
                    "token": token,
                    "profile": prof
                }).encode())

            elif path == "/api/login":
                payload = self._json()
                ok, token, prof = authenticate(payload.get("email", ""), payload.get("password", ""))
                
                if not ok:
                    self._set(401, "application/json")
                    self.wfile.write(json.dumps({"success": False, "error": token}).encode())
                    return
                
                self._set(200, "application/json")
                self.wfile.write(json.dumps({
                    "success": True,
                    "token": token,
                    "profile": prof
                }).encode())

            elif path == "/api/me":
                payload = self._json()
                prof = profile_from_token(payload.get("token", ""))
                
                if not prof:
                    self._set(401, "application/json")
                    self.wfile.write(json.dumps({"success": False}).encode())
                    return
                
                self._set(200, "application/json")
                self.wfile.write(json.dumps({"success": True, "profile": prof}).encode())

            elif path == "/api/search":
                payload = self._json()
                # Allow public search without authentication. This project offers
                # anonymous access to search and artist suggestions so users can
                # use the recommendation UI without logging in.
                
                query = payload.get("query", "").lower().strip()
                
                if len(query) < 2:
                    self._set(200, "application/json")
                    self.wfile.write(json.dumps({"success": True, "songs": [], "artists": []}).encode())
                    return
                
                recommender.load()
                songs = []
                artists_set = set()
                
                # Search for matching artists
                for song in recommender.songs:
                    artist = song.get("artist", "").lower()
                    if query in artist and artist not in artists_set:
                        # Get the original artist name (not lowercase)
                        artists_set.add(song.get("artist", ""))
                
                # If we found matching artists, return their songs
                if artists_set:
                    for song in recommender.songs:
                        song_artist = song.get("artist", "")
                        if song_artist in artists_set:
                            songs.append({"title": song["title"], "artist": song["artist"]})
                else:
                    # Only search by song title if no artist matches
                    for song in recommender.songs:
                        if query in song.get("title", "").lower():
                            songs.append({"title": song["title"], "artist": song["artist"]})
                
                # Sort artists and return matching artists
                artists = [{"name": a} for a in sorted(artists_set, key=str.lower)]
                
                self._set(200, "application/json")
                self.wfile.write(json.dumps({
                    "success": True,
                    "songs": songs[:50],  # Return matching artist's songs
                    "artists": artists  # Return only matching artists
                }).encode())


            elif path == "/api/artists":
                payload = self._json()
                # Allow public artists listing without authentication
                
                artists = get_unique_artists()
                self._set(200, "application/json")
                self.wfile.write(json.dumps({"success": True, "artists": artists}).encode())

            elif path == "/api/recommend":
                payload = self._json()
                # Recommendations are accessible publicly (no login required)
                
                recs = recommender.recommend(
                    payload.get("languages", []),
                    payload.get("genres", []),
                    payload.get("moods", []),
                    payload.get("artists", [])
                )
                
                self._set(200, "application/json")
                self.wfile.write(json.dumps({"success": True, "recommendations": recs}).encode())

            else:
                self._set(404, "application/json")
                self.wfile.write(json.dumps({"error": "Not found"}).encode())

        except Exception as e:
            print(f"Error: {e}")
            self._set(500, "application/json")
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

def run(host="127.0.0.1", port=8000):
    init_db()
    load_initial_songs()
    
    srv = HTTPServer((host, port), App)
    print(f"\n✓ Server running at http://{host}:{port}/project.html\n")
    
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        srv.server_close()

# ==============================
# 🎵 ADD YOUR SONGS HERE 🎵
# ==============================
DEFAULT_SONGS = [
    # Arijit Singh - Famous Indian Singer
    ("Tum Hi Ho", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Ae Dil Hai Mushkil", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Tera Fitoor", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Raabta", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Naina", "Arijit Singh", "Pop", "Hindi", "Melancholic", 65),
    ("Channa Mereya", "Arijit Singh", "Pop", "Hindi", "Sad", 68),
    ("Gerua", "Arijit Singh", "Pop", "Hindi", "Romantic", 74),
    ("Enna Sona", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Kalank Title Track", "Arijit Singh", "Pop", "Hindi", "Melancholic", 66),
    ("First Class", "Arijit Singh", "Pop", "Hindi", "Energetic", 110),
    ("Ae Watan", "Arijit Singh", "Pop", "Hindi", "Energetic", 120),
    ("Ve Maahi", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Thodi Der", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Roke Na Ruke Naina", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Phir Bhi Tumko Chaahunga", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Khamoshiyan", "Arijit Singh", "Pop", "Hindi", "Melancholic", 65),
    ("Hamari Adhuri Kahani", "Arijit Singh", "Pop", "Hindi", "Sad", 64),
    ("Samjhawan", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Kabira", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Main Rang Sharbaton Ka", "Arijit Singh", "Pop", "Hindi", "Happy", 95),
    ("Soch Na Sake", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Dil Diyan Gallan", "Arijit Singh", "Pop", "Hindi", "Romantic", 74),
    ("Nazm Nazm", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Yeh Kaisi Jagah", "Arijit Singh", "Pop", "Hindi", "Peaceful", 65),
    ("Tera Ban Jaunga", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Sanam Re", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Bolna", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Ik Vaari Aa", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Sooraj Dooba Hain", "Arijit Singh", "Pop", "Hindi", "Happy", 105),
    ("Nashe Si Chadh Gayi", "Arijit Singh", "Pop", "Hindi", "Energetic", 115),
    ("Zaalima", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Phir Le Aaya Dil", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Shayad", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Hawayein", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Agar Tum Saath Ho", "Arijit Singh", "Pop", "Hindi", "Romantic", 66),
    ("Kaise Hua", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Humnava Mere", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Dil Ibaadat", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Duaa", "Arijit Singh", "Pop", "Hindi", "Peaceful", 65),
    ("Raabta", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Muskurane", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Bekhayali", "Arijit Singh", "Pop", "Hindi", "Sad", 65),
    ("Tujhe Kitna Chahne Lage", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Main Dhoondne Ko Zamaane Mein", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Aayat", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Tum Hi Aana", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    
    # Shreya Ghoshal - Famous Indian Singer
    ("Teri Ore", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 72),
    ("Piyu Bole", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Agar Tum Mil Jao", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 68),
    ("Barso Re", "Shreya Ghoshal", "Pop", "Hindi", "Happy", 100),
    ("Chikni Chameli", "Shreya Ghoshal", "Dance", "Hindi", "Energetic", 120),
    ("Deewani Mastani", "Shreya Ghoshal", "Pop", "Hindi", "Energetic", 110),
    ("Nagada Sang Dhol", "Shreya Ghoshal", "Pop", "Hindi", "Happy", 105),
    ("Dola Re Dola", "Shreya Ghoshal", "Pop", "Hindi", "Energetic", 115),
    ("Silsila Ye Chaahat Ka", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Yeh Haseen Vadiyan", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 68),
    ("Sunn Raha Hai", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Chaar Kadam", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 72),
    ("Mohe Rang Do Laal", "Shreya Ghoshal", "Pop", "Hindi", "Happy", 95),
    ("Jiya Re", "Shreya Ghoshal", "Pop", "Hindi", "Energetic", 110),
    ("Bairi Piya", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Saans", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 68),
    ("Aap Ki Nazron Ne Samjha", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Samjhawan", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 68),
    ("Balam Pichkari", "Shreya Ghoshal", "Pop", "Hindi", "Happy", 105),
    ("Manwa Laage", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 72),
    ("Radha", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Naino Ne Baandhi", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 68),
    ("Sawan Aaya Hai", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Do Dhaari Talwaar", "Shreya Ghoshal", "Pop", "Hindi", "Energetic", 115),
    ("Pinga", "Shreya Ghoshal", "Pop", "Hindi", "Energetic", 120),
    ("Jag Ghoomeya", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 72),
    ("Man Mohana", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Yeh Ishq Hai", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 68),
    ("Pal Pal Dil Ke Paas", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Aashiq Banaya Aapne", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 72),
    ("Pehli Nazar Mein", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Teri Meri", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 68),
    ("Agar Tum Saath Ho", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 66),
    ("Naina", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 70),
    ("Tum Se Hi", "Shreya Ghoshal", "Pop", "Hindi", "Romantic", 72),
    
    # Sonu Nigam - Famous Indian Singer
    ("Kal Ho Naa Ho", "Sonu Nigam", "Pop", "Hindi", "Romantic", 72),
    ("Abhi Mujh Mein Kahin", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Suraj Hua Maddham", "Sonu Nigam", "Pop", "Hindi", "Romantic", 68),
    ("Saathiya", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Sandese Aate Hain", "Sonu Nigam", "Pop", "Hindi", "Peaceful", 65),
    ("Tum Paas Aaye", "Sonu Nigam", "Pop", "Hindi", "Romantic", 72),
    ("Yeh Taara Woh Taara", "Sonu Nigam", "Pop", "Hindi", "Happy", 95),
    ("Do Pal", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Main Hoon Na", "Sonu Nigam", "Pop", "Hindi", "Energetic", 110),
    ("Koi Mil Gaya", "Sonu Nigam", "Pop", "Hindi", "Happy", 100),
    ("Dil Dooba", "Sonu Nigam", "Pop", "Hindi", "Happy", 105),
    ("Chhup Chhup Ke", "Sonu Nigam", "Pop", "Hindi", "Romantic", 72),
    ("Ishq Bina", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Aankhon Mein Teri", "Sonu Nigam", "Pop", "Hindi", "Romantic", 68),
    ("Tu Hi Re", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Satrangi Re", "Sonu Nigam", "Pop", "Hindi", "Romantic", 72),
    ("Kabhi Alvida Naa Kehna", "Sonu Nigam", "Pop", "Hindi", "Romantic", 68),
    ("Yeh Dil Deewana", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Mere Rang Mein", "Sonu Nigam", "Pop", "Hindi", "Romantic", 72),
    ("Chalte Chalte", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Dheere Dheere", "Sonu Nigam", "Pop", "Hindi", "Romantic", 68),
    ("Aye Udi Udi Udi", "Sonu Nigam", "Pop", "Hindi", "Happy", 100),
    ("Dil Ne Kaha", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Chaand Sifarish", "Sonu Nigam", "Pop", "Hindi", "Romantic", 72),
    ("Piya Basanti", "Sonu Nigam", "Pop", "Hindi", "Romantic", 70),
    ("Mujhe Tumse Mohabbat Hai", "Sonu Nigam", "Pop", "Hindi", "Romantic", 68),
    
    # Atif Aslam - Famous Pakistani/Indian Singer
    ("Pehli Nazar Mein", "Atif Aslam", "Pop", "Hindi", "Romantic", 72),
    ("Tera Hone Laga Hoon", "Atif Aslam", "Pop", "Hindi", "Romantic", 70),
    ("Tu Jaane Na", "Atif Aslam", "Pop", "Hindi", "Romantic", 68),
    ("Jeene Laga Hoon", "Atif Aslam", "Pop", "Hindi", "Romantic", 72),
    ("Tere Sang Yaara", "Atif Aslam", "Pop", "Hindi", "Romantic", 70),
    ("Baarishein", "Atif Aslam", "Pop", "Hindi", "Romantic", 68),
    ("Dil Diyan Gallan", "Atif Aslam", "Pop", "Hindi", "Romantic", 74),
    ("Tera Ban Jaunga", "Atif Aslam", "Pop", "Hindi", "Romantic", 72),
    ("Channa Mereya", "Atif Aslam", "Pop", "Hindi", "Sad", 68),
    ("Dil Meri Na Sune", "Atif Aslam", "Pop", "Hindi", "Romantic", 70),
    ("Aadat", "Atif Aslam", "Pop", "Hindi", "Romantic", 72),
    ("Woh Lamhe", "Atif Aslam", "Pop", "Hindi", "Romantic", 68),
    ("Tere Liye", "Atif Aslam", "Pop", "Hindi", "Romantic", 70),
    ("Kuch Is Tarah", "Atif Aslam", "Pop", "Hindi", "Romantic", 72),
    ("Main Rang Sharbaton Ka", "Atif Aslam", "Pop", "Hindi", "Happy", 95),
    ("Be Intehaan", "Atif Aslam", "Pop", "Hindi", "Romantic", 70),
    ("Piya O Re Piya", "Atif Aslam", "Pop", "Hindi", "Romantic", 72),
    ("Tera Hone Laga Hoon", "Atif Aslam", "Pop", "Hindi", "Romantic", 70),
    ("Tum Hi Ho", "Atif Aslam", "Pop", "Hindi", "Romantic", 68),
    ("Doorie", "Atif Aslam", "Pop", "Hindi", "Sad", 70),
    ("Hum Kis Gali Ja Rahe Hain", "Atif Aslam", "Pop", "Hindi", "Romantic", 72),
    ("Yeh Kaisi Jagah", "Atif Aslam", "Pop", "Hindi", "Peaceful", 65),
    ("Tere Bin", "Atif Aslam", "Pop", "Hindi", "Romantic", 70),
    ("Main Agar", "Atif Aslam", "Pop", "Hindi", "Romantic", 68),
    ("Kun Faya Kun", "Atif Aslam", "Pop", "Hindi", "Peaceful", 65),
    
    # Jubin Nautiyal - Famous Indian Singer
    ("Tum Hi Aana", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 72),
    ("Humdard", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 70),
    ("Kaabil Hoon", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 68),
    ("Tum Hi Aana", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 72),
    ("Humnava Mere", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 68),
    ("Raataan Lambiyan", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 70),
    ("Dil Galti Kar Baitha Hai", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 72),
    ("Tumse Pyaar Karke", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 70),
    ("Tera Yaar Hoon Main", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 68),
    ("Raataan Lambiyan", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 70),
    ("Tere Naal", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 72),
    ("Chaleya", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 70),
    ("Meri Zindagi Hai Tu", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 68),
    ("Akh Lad Jaave", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 72),
    ("Raat Ke Nashe", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 70),
    ("Meri Aashiqui", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 68),
    ("Tum Hi Aana", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 72),
    ("Dil Galti Kar Baitha Hai", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 70),
    ("Tera Ban Jaunga", "Jubin Nautiyal", "Pop", "Hindi", "Romantic", 72),
    ("Tum Hi Ho Bandhu", "Jubin Nautiyal", "Pop", "Hindi", "Happy", 95),
    
    # Neha Kakkar - Famous Indian Singer
    ("Dilbar", "Neha Kakkar", "Pop", "Hindi", "Energetic", 120),
    ("Mile Ho Tum", "Neha Kakkar", "Pop", "Hindi", "Happy", 105),
    ("Kar Gayi Chull", "Neha Kakkar", "Pop", "Hindi", "Energetic", 115),
    ("London Thumakda", "Neha Kakkar", "Pop", "Hindi", "Energetic", 120),
    ("Nachdi Phira", "Neha Kakkar", "Pop", "Hindi", "Energetic", 125),
    ("Proper Patola", "Neha Kakkar", "Pop", "Hindi", "Energetic", 130),
    ("Garmi", "Neha Kakkar", "Pop", "Hindi", "Energetic", 125),
    ("Aankh Marey", "Neha Kakkar", "Pop", "Hindi", "Energetic", 120),
    ("Kamariya", "Neha Kakkar", "Pop", "Hindi", "Energetic", 115),
    ("O Saki Saki", "Neha Kakkar", "Pop", "Hindi", "Energetic", 125),
    ("Dilbar", "Neha Kakkar", "Pop", "Hindi", "Energetic", 120),
    ("Ek Toh Kum Zindagani", "Neha Kakkar", "Pop", "Hindi", "Energetic", 115),
    ("Hookah Bar", "Neha Kakkar", "Pop", "Hindi", "Energetic", 120),
    ("Naagin", "Neha Kakkar", "Pop", "Hindi", "Energetic", 130),
    ("Dilbar", "Neha Kakkar", "Pop", "Hindi", "Energetic", 120),
    ("Sakhiyaan", "Neha Kakkar", "Pop", "Hindi", "Energetic", 125),
    ("Mile Ho Tum", "Neha Kakkar", "Pop", "Hindi", "Happy", 105),
    ("Kar Gayi Chull", "Neha Kakkar", "Pop", "Hindi", "Energetic", 115),
    
    # Darshan Raval - Famous Indian Singer
    ("Tera Zikr", "Darshan Raval", "Pop", "Hindi", "Romantic", 70),
    ("Chogada", "Darshan Raval", "Pop", "Hindi", "Happy", 105),
    ("Kamariya", "Darshan Raval", "Pop", "Hindi", "Energetic", 115),
    ("Tera Ban Jaunga", "Darshan Raval", "Pop", "Hindi", "Romantic", 72),
    ("Ek Tarfa", "Darshan Raval", "Pop", "Hindi", "Romantic", 68),
    ("Sanu Ik Pal", "Darshan Raval", "Pop", "Hindi", "Romantic", 70),
    ("Tera Zikr", "Darshan Raval", "Pop", "Hindi", "Romantic", 72),
    ("Baarish", "Darshan Raval", "Pop", "Hindi", "Romantic", 68),
    ("Tera Zikr", "Darshan Raval", "Pop", "Hindi", "Romantic", 70),
    ("Chogada", "Darshan Raval", "Pop", "Hindi", "Happy", 105),
    ("Tera Ban Jaunga", "Darshan Raval", "Pop", "Hindi", "Romantic", 72),
    ("Ek Tarfa", "Darshan Raval", "Pop", "Hindi", "Romantic", 68),
    
    # A.R. Rahman - Famous Indian Composer/Singer
    ("Jai Ho", "A.R. Rahman", "Pop", "Hindi", "Energetic", 120),
    ("Tum Hi Ho", "A.R. Rahman", "Pop", "Hindi", "Romantic", 70),
    ("Kun Faya Kun", "A.R. Rahman", "Pop", "Hindi", "Peaceful", 65),
    ("Maula Mere Maula", "A.R. Rahman", "Pop", "Hindi", "Peaceful", 60),
    ("Tere Bina", "A.R. Rahman", "Pop", "Hindi", "Romantic", 68),
    ("Maa Tujhe Salaam", "A.R. Rahman", "Pop", "Hindi", "Energetic", 110),
    ("Rang De Basanti", "A.R. Rahman", "Pop", "Hindi", "Energetic", 115),
    ("Roja", "A.R. Rahman", "Pop", "Hindi", "Romantic", 70),
    ("Dil Se Re", "A.R. Rahman", "Pop", "Hindi", "Energetic", 120),
    ("Chaiyya Chaiyya", "A.R. Rahman", "Pop", "Hindi", "Happy", 100),
    ("Taal Se Taal", "A.R. Rahman", "Pop", "Hindi", "Energetic", 115),
    ("Jashn-E-Bahaara", "A.R. Rahman", "Pop", "Hindi", "Romantic", 72),
    ("Nadaan Parinde", "A.R. Rahman", "Pop", "Hindi", "Romantic", 68),
    ("Kabira", "A.R. Rahman", "Pop", "Hindi", "Romantic", 70),
    ("Patakha Guddi", "A.R. Rahman", "Pop", "Hindi", "Happy", 105),
    ("Heer", "A.R. Rahman", "Pop", "Hindi", "Romantic", 70),
    ("Sooha Saha", "A.R. Rahman", "Pop", "Hindi", "Romantic", 68),
    ("Nadaan Parinde", "A.R. Rahman", "Pop", "Hindi", "Romantic", 70),
    ("Ae Ajnabi", "A.R. Rahman", "Pop", "Hindi", "Romantic", 72),
    ("Piya Haji Ali", "A.R. Rahman", "Pop", "Hindi", "Peaceful", 65),
    
    # Mohit Chauhan - Famous Indian Singer
    ("Tum Se Hi", "Mohit Chauhan", "Pop", "Hindi", "Romantic", 70),
    ("Kun Faya Kun", "Mohit Chauhan", "Pop", "Hindi", "Peaceful", 65),
    ("Masakali", "Mohit Chauhan", "Pop", "Hindi", "Happy", 100),
    ("Tum Ho", "Mohit Chauhan", "Pop", "Hindi", "Romantic", 72),
    ("Sadda Haq", "Mohit Chauhan", "Pop", "Hindi", "Energetic", 115),
    ("Aao Na", "Mohit Chauhan", "Pop", "Hindi", "Romantic", 70),
    ("Dooba Dooba", "Mohit Chauhan", "Pop", "Hindi", "Happy", 105),
    ("Pehli Baar Mohabbat", "Mohit Chauhan", "Pop", "Hindi", "Romantic", 68),
    ("Aahun Aahun", "Mohit Chauhan", "Pop", "Hindi", "Romantic", 70),
    ("Saadda Haq", "Mohit Chauhan", "Pop", "Hindi", "Energetic", 115),
    ("Tum Se Hi", "Mohit Chauhan", "Pop", "Hindi", "Romantic", 72),
    ("Aao Na", "Mohit Chauhan", "Pop", "Hindi", "Romantic", 70),
    ("Kun Faya Kun", "Mohit Chauhan", "Pop", "Hindi", "Peaceful", 65),
    ("Masakali", "Mohit Chauhan", "Pop", "Hindi", "Happy", 100),
    
    # Kumar Sanu - Famous Indian Singer
    ("Tujhe Dekha To", "Kumar Sanu", "Pop", "Hindi", "Romantic", 70),
    ("Do Dil Mil Rahe Hain", "Kumar Sanu", "Pop", "Hindi", "Romantic", 72),
    ("Tere Naam", "Kumar Sanu", "Pop", "Hindi", "Romantic", 68),
    ("Aankhon Mein Teri", "Kumar Sanu", "Pop", "Hindi", "Romantic", 70),
    ("Pehla Nasha", "Kumar Sanu", "Pop", "Hindi", "Romantic", 72),
    ("Dilwale Dulhania Le Jayenge", "Kumar Sanu", "Pop", "Hindi", "Romantic", 70),
    ("Mera Dil Bhi Kitna Pagal Hai", "Kumar Sanu", "Pop", "Hindi", "Romantic", 68),
    ("Ek Ladki Ko Dekha", "Kumar Sanu", "Pop", "Hindi", "Romantic", 70),
    ("Jab Koi Baat Bigad Jaaye", "Kumar Sanu", "Pop", "Hindi", "Romantic", 72),
    ("Churake Dil Mera", "Kumar Sanu", "Pop", "Hindi", "Romantic", 70),
    ("Tujhe Dekha To", "Kumar Sanu", "Pop", "Hindi", "Romantic", 68),
    ("Saajan", "Kumar Sanu", "Pop", "Hindi", "Romantic", 70),
    ("Aisi Deewangi", "Kumar Sanu", "Pop", "Hindi", "Romantic", 72),
    ("Mera Dil Bhi Kitna Pagal Hai", "Kumar Sanu", "Pop", "Hindi", "Romantic", 68),
    ("Tere Naam", "Kumar Sanu", "Pop", "Hindi", "Romantic", 70),
    
    # Udit Narayan - Famous Indian Singer
    ("Mitwa", "Udit Narayan", "Pop", "Hindi", "Romantic", 72),
    ("Pehla Nasha", "Udit Narayan", "Pop", "Hindi", "Romantic", 70),
    ("Tujhe Dekha To", "Udit Narayan", "Pop", "Hindi", "Romantic", 72),
    ("Kuch Kuch Hota Hai", "Udit Narayan", "Pop", "Hindi", "Romantic", 70),
    ("Tum Paas Aaye", "Udit Narayan", "Pop", "Hindi", "Romantic", 68),
    ("Ae Mere Humsafar", "Udit Narayan", "Pop", "Hindi", "Romantic", 70),
    ("Chaand Taare", "Udit Narayan", "Pop", "Hindi", "Romantic", 72),
    ("Hum Aapke Hain Koun", "Udit Narayan", "Pop", "Hindi", "Happy", 100),
    ("Dil To Pagal Hai", "Udit Narayan", "Pop", "Hindi", "Romantic", 70),
    ("Pardesi Pardesi", "Udit Narayan", "Pop", "Hindi", "Romantic", 68),
    ("Chalte Chalte", "Udit Narayan", "Pop", "Hindi", "Romantic", 70),
    ("Tere Naam", "Udit Narayan", "Pop", "Hindi", "Romantic", 72),
    ("Mere Rang Mein", "Udit Narayan", "Pop", "Hindi", "Romantic", 70),
    ("Jab Koi Baat", "Udit Narayan", "Pop", "Hindi", "Romantic", 68),
    ("Aankhon Mein Teri", "Udit Narayan", "Pop", "Hindi", "Romantic", 70),
    
    # Alka Yagnik - Famous Indian Singer
    ("Chura Ke Dil Mera", "Alka Yagnik", "Pop", "Hindi", "Romantic", 70),
    ("Dilbar", "Alka Yagnik", "Pop", "Hindi", "Romantic", 72),
    ("Aankh Marey", "Alka Yagnik", "Pop", "Hindi", "Romantic", 68),
    ("Chamma Chamma", "Alka Yagnik", "Pop", "Hindi", "Happy", 105),
    ("Taal Se Taal", "Alka Yagnik", "Pop", "Hindi", "Energetic", 115),
    ("Chaiyya Chaiyya", "Alka Yagnik", "Pop", "Hindi", "Happy", 100),
    ("Dil To Pagal Hai", "Alka Yagnik", "Pop", "Hindi", "Romantic", 70),
    ("Hum Aapke Hain Koun", "Alka Yagnik", "Pop", "Hindi", "Happy", 100),
    ("Kuch Kuch Hota Hai", "Alka Yagnik", "Pop", "Hindi", "Romantic", 72),
    ("Tum Paas Aaye", "Alka Yagnik", "Pop", "Hindi", "Romantic", 68),
    ("Ae Mere Humsafar", "Alka Yagnik", "Pop", "Hindi", "Romantic", 70),
    ("Chaand Taare", "Alka Yagnik", "Pop", "Hindi", "Romantic", 72),
    ("Pardesi Pardesi", "Alka Yagnik", "Pop", "Hindi", "Romantic", 68),
    ("Chalte Chalte", "Alka Yagnik", "Pop", "Hindi", "Romantic", 70),
    ("Mere Rang Mein", "Alka Yagnik", "Pop", "Hindi", "Romantic", 72),
    
    # Sunidhi Chauhan - Famous Indian Singer
    ("Beedi", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 125),
    ("Desi Girl", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 120),
    ("Kamli", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 115),
    ("Dhoom Machale", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 130),
    ("Ainvayi Ainvayi", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 125),
    ("Sadi Gali", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 120),
    ("Ishq Sufiyana", "Sunidhi Chauhan", "Pop", "Hindi", "Romantic", 70),
    ("Sajda", "Sunidhi Chauhan", "Pop", "Hindi", "Romantic", 68),
    ("Halkat Jawani", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 125),
    ("Kamli", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 115),
    ("Dhoom Machale", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 130),
    ("Beedi", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 125),
    ("Desi Girl", "Sunidhi Chauhan", "Pop", "Hindi", "Energetic", 120),
    
    # Kishore Kumar - Legendary Indian Singer
    ("Ek Ladki Bheegi Bhaagi Si", "Kishore Kumar", "Pop", "Hindi", "Happy", 100),
    ("Yeh Shaam Mastani", "Kishore Kumar", "Pop", "Hindi", "Romantic", 70),
    ("Pal Pal Dil Ke Paas", "Kishore Kumar", "Pop", "Hindi", "Romantic", 68),
    ("Mere Sapno Ki Rani", "Kishore Kumar", "Pop", "Hindi", "Romantic", 72),
    ("Roop Tera Mastana", "Kishore Kumar", "Pop", "Hindi", "Romantic", 70),
    ("Chalte Chalte", "Kishore Kumar", "Pop", "Hindi", "Romantic", 68),
    ("Yeh Jo Mohabbat Hai", "Kishore Kumar", "Pop", "Hindi", "Romantic", 70),
    ("Aane Se Uske Aaye Bahar", "Kishore Kumar", "Pop", "Hindi", "Romantic", 72),
    ("Koi Humdum Na Raha", "Kishore Kumar", "Pop", "Hindi", "Romantic", 68),
    ("Chhedo Na Meri Zulfein", "Kishore Kumar", "Pop", "Hindi", "Romantic", 70),
    ("O Mere Dil Ke Chain", "Kishore Kumar", "Pop", "Hindi", "Romantic", 72),
    ("Neele Neele Ambar Par", "Kishore Kumar", "Pop", "Hindi", "Romantic", 70),
    ("Mere Mehboob Qayamat Hogi", "Kishore Kumar", "Pop", "Hindi", "Romantic", 68),
    ("Zindagi Ek Safar Hai Suhana", "Kishore Kumar", "Pop", "Hindi", "Happy", 100),
    ("Musafir Hoon Yaaron", "Kishore Kumar", "Pop", "Hindi", "Happy", 95),
    
    # Lata Mangeshkar - Legendary Indian Singer
    ("Ajeeb Dastaan Hai Yeh", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 68),
    ("Lag Jaa Gale", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 70),
    ("Aap Ki Nazron Ne Samjha", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 72),
    ("Tere Bina Zindagi Se", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 68),
    ("Aaja Re Pardesi", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 70),
    ("Tujhe Dekha To", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 72),
    ("Pyar Kiya To Darna Kya", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 70),
    ("Naina Barse Rimjhim", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 68),
    ("Aapki Ankhon Mein", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 70),
    ("Mere Khwabon Mein", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 72),
    ("Kuch Kuch Hota Hai", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 70),
    ("Hum Aapke Hain Koun", "Lata Mangeshkar", "Pop", "Hindi", "Happy", 100),
    ("Dil To Pagal Hai", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 68),
    ("Chalte Chalte", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 70),
    ("Tere Bina", "Lata Mangeshkar", "Pop", "Hindi", "Romantic", 72),
    
    # Other Famous Indian Singers
    ("Mann Mera", "Gajendra Verma", "Pop", "Hindi", "Romantic", 70),
    ("Tum Hi Ho Bandhu", "Sourav Joshi", "Pop", "Hindi", "Happy", 95),
    ("Baarish", "Ash King", "Pop", "Hindi", "Romantic", 68),
    ("Tere Sang Yaara", "Rupesh Kumar Ram", "Pop", "Hindi", "Romantic", 72),
    ("Mann Meri Jaan", "King", "Pop", "Hindi", "Romantic", 70),
    ("Tu Aake Dekhle", "King", "Pop", "Hindi", "Romantic", 72),
    ("Ek Toh Kum Zindagani", "Asees Kaur", "Pop", "Hindi", "Energetic", 115),
    ("Makhna", "Asees Kaur", "Pop", "Hindi", "Energetic", 120),
    ("Raataan Lambiyan", "Asees Kaur", "Pop", "Hindi", "Romantic", 70),
    ("Dil Galti Kar Baitha Hai", "Asees Kaur", "Pop", "Hindi", "Romantic", 72),
    ("Baarish Ban Jaana", "Stebin Ben", "Pop", "Hindi", "Romantic", 70),
    ("Tera Ban Jaunga", "Tulsi Kumar", "Pop", "Hindi", "Romantic", 72),
    ("Dil Diyan Gallan", "Tulsi Kumar", "Pop", "Hindi", "Romantic", 74),
    ("Tum Hi Aana", "Tulsi Kumar", "Pop", "Hindi", "Romantic", 72),
    ("Raataan Lambiyan", "Tulsi Kumar", "Pop", "Hindi", "Romantic", 70),
    ("Tere Naal", "Parmish Verma", "Pop", "Hindi", "Romantic", 72),
    ("Saudebaazi", "Parmish Verma", "Pop", "Hindi", "Romantic", 70),
    ("Illegal Weapon", "Garry Sandhu", "Pop", "Hindi", "Energetic", 125),
    ("Soch", "Hardy Sandhu", "Pop", "Hindi", "Romantic", 70),
    ("Backbone", "Hardy Sandhu", "Pop", "Hindi", "Energetic", 115),
    ("Naah", "Harrdy Sandhu", "Pop", "Hindi", "Romantic", 72),
    ("Bijlee Bijlee", "Harrdy Sandhu", "Pop", "Hindi", "Energetic", 130),
    ("Kya Baat Ay", "Harrdy Sandhu", "Pop", "Hindi", "Happy", 105),
    ("Same Beef", "Bohemia", "Hip Hop", "Hindi", "Energetic", 120),
    ("Desi Hip Hop", "Bohemia", "Hip Hop", "Hindi", "Energetic", 115),
    ("Ek Glassy", "Yo Yo Honey Singh", "Hip Hop", "Hindi", "Energetic", 125),
    ("Blue Eyes", "Yo Yo Honey Singh", "Hip Hop", "Hindi", "Energetic", 120),
    ("Desi Kalakaar", "Yo Yo Honey Singh", "Hip Hop", "Hindi", "Energetic", 125),
    ("Brown Rang", "Yo Yo Honey Singh", "Hip Hop", "Hindi", "Energetic", 130),
    ("Love Dose", "Yo Yo Honey Singh", "Hip Hop", "Hindi", "Energetic", 125),
    ("Lungi Dance", "Yo Yo Honey Singh", "Pop", "Hindi", "Energetic", 130),
    ("Main Tera Boyfriend", "Arijit Singh", "Pop", "Hindi", "Happy", 105),
    ("Channa Ve", "Amit Trivedi", "Pop", "Hindi", "Romantic", 70),
    ("Ik Vaari Aa", "Pritam", "Pop", "Hindi", "Romantic", 72),
    ("Agar Tum Saath Ho", "Alka Yagnik", "Pop", "Hindi", "Romantic", 66),
    ("Channa Mereya", "Arijit Singh", "Pop", "Hindi", "Sad", 68),
    ("Nashe Si Chadh Gayi", "Arijit Singh", "Pop", "Hindi", "Energetic", 115),
    ("Raabta", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Ae Dil Hai Mushkil", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Tum Hi Ho", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Gerua", "Arijit Singh", "Pop", "Hindi", "Romantic", 74),
    ("Channa Mereya", "Arijit Singh", "Pop", "Hindi", "Sad", 68),
    ("Enna Sona", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Kalank Title Track", "Arijit Singh", "Pop", "Hindi", "Melancholic", 66),
    ("First Class", "Arijit Singh", "Pop", "Hindi", "Energetic", 110),
    ("Ve Maahi", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Thodi Der", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Roke Na Ruke Naina", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Phir Bhi Tumko Chaahunga", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Khamoshiyan", "Arijit Singh", "Pop", "Hindi", "Melancholic", 65),
    ("Hamari Adhuri Kahani", "Arijit Singh", "Pop", "Hindi", "Sad", 64),
    ("Samjhawan", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Kabira", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Main Rang Sharbaton Ka", "Arijit Singh", "Pop", "Hindi", "Happy", 95),
    ("Soch Na Sake", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Dil Diyan Gallan", "Arijit Singh", "Pop", "Hindi", "Romantic", 74),
    ("Nazm Nazm", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Yeh Kaisi Jagah", "Arijit Singh", "Pop", "Hindi", "Peaceful", 65),
    ("Tera Ban Jaunga", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Sanam Re", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Bolna", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Ik Vaari Aa", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Sooraj Dooba Hain", "Arijit Singh", "Pop", "Hindi", "Happy", 105),
    ("Nashe Si Chadh Gayi", "Arijit Singh", "Pop", "Hindi", "Energetic", 115),
    ("Zaalima", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Phir Le Aaya Dil", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Shayad", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Hawayein", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Agar Tum Saath Ho", "Arijit Singh", "Pop", "Hindi", "Romantic", 66),
    ("Kaise Hua", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Humnava Mere", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Dil Ibaadat", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Duaa", "Arijit Singh", "Pop", "Hindi", "Peaceful", 65),
    ("Raabta", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Muskurane", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    ("Bekhayali", "Arijit Singh", "Pop", "Hindi", "Sad", 65),
    ("Tujhe Kitna Chahne Lage", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Main Dhoondne Ko Zamaane Mein", "Arijit Singh", "Pop", "Hindi", "Romantic", 68),
    ("Aayat", "Arijit Singh", "Pop", "Hindi", "Romantic", 70),
    ("Tum Hi Aana", "Arijit Singh", "Pop", "Hindi", "Romantic", 72),
    
    # International Popular Songs
    ("Levitating", "Dua Lipa", "Dance", "English", "Happy", 103),
    ("Shape of You", "Ed Sheeran", "Pop", "English", "Romantic", 96),
    ("Uptown Funk", "Mark Ronson ft. Bruno Mars", "Funk", "English", "Energetic", 115),
    ("Radioactive", "Imagine Dragons", "Alternative", "English", "Intense", 136),
    ("Closer", "The Chainsmokers ft. Halsey", "Electronic", "English", "Chill", 95),
    ("Despacito", "Luis Fonsi", "Pop", "Spanish", "Energetic", 89),
    ("Spring Day", "BTS", "Pop", "Korean", "Melancholic", 108),
    ("Blinding Lights", "The Weeknd", "Pop", "English", "Energetic", 171),
    ("Bad Guy", "Billie Eilish", "Electronic", "English", "Intense", 135),
    ("Hotel California", "Eagles", "Rock", "English", "Chill", 75),
    ("Thinking Out Loud", "Ed Sheeran", "Pop", "English", "Romantic", 79),
    ("Believer", "Imagine Dragons", "Alternative", "English", "Intense", 125),
    ("Faded", "Alan Walker", "Electronic", "English", "Melancholic", 90),
    ("Watermelon Sugar", "Harry Styles", "Pop", "English", "Happy", 95),
    ("Perfect", "Ed Sheeran", "Pop", "English", "Romantic", 95),
    ("Circles", "Post Malone", "Pop", "English", "Chill", 90),
    ("Memories", "Maroon 5", "Pop", "English", "Melancholic", 90),
    ("Heat Waves", "Glass Animals", "Alternative", "English", "Chill", 80),
    ("Peaches", "Justin Bieber ft. Daniel Caesar", "Pop", "English", "Happy", 90),
    ("Viva La Vida", "Coldplay", "Alternative", "English", "Energetic", 138),
    ("Don't Stop Believin'", "Journey", "Rock", "English", "Happy", 119),
    ("Bohemian Rhapsody", "Queen", "Rock", "English", "Intense", 72),
    ("Stairway to Heaven", "Led Zeppelin", "Rock", "English", "Peaceful", 63),
    ("As It Was", "Harry Styles", "Pop", "English", "Chill", 173),
    ("Stay", "The Kid LAROI & Justin Bieber", "Pop", "English", "Melancholic", 170),
    ("Good 4 U", "Olivia Rodrigo", "Pop", "English", "Energetic", 140),
    ("Levitating", "Dua Lipa", "Dance", "English", "Happy", 103),
    ("Save Your Tears", "The Weeknd", "Pop", "English", "Chill", 118),
    ("Shivers", "Ed Sheeran", "Pop", "English", "Happy", 81),
    ("Industry Baby", "Lil Nas X & Jack Harlow", "Hip Hop", "English", "Energetic", 150),
    ("Heat Waves", "Glass Animals", "Alternative", "English", "Chill", 80),
    ("All Too Well", "Taylor Swift", "Pop", "English", "Melancholic", 90),
    ("Enemy", "Imagine Dragons", "Alternative", "English", "Intense", 77),
    ("Cold Heart", "Elton John & Dua Lipa", "Pop", "English", "Happy", 113),
    ("Stay", "The Kid LAROI & Justin Bieber", "Pop", "English", "Melancholic", 170),
    ("Easy On Me", "Adele", "Pop", "English", "Melancholic", 76),
    ("Flowers", "Miley Cyrus", "Pop", "English", "Happy", 118),
    ("Kill Bill", "SZA", "R&B", "English", "Energetic", 134),
    ("Unholy", "Sam Smith & Kim Petras", "Pop", "English", "Energetic", 111),
    ("I'm Good", "David Guetta & Bebe Rexha", "Electronic", "English", "Energetic", 128),
    ("Calm Down", "Rema & Selena Gomez", "Pop", "English", "Chill", 108),
    ("Made You Look", "Meghan Trainor", "Pop", "English", "Happy", 116),
    ("Anti-Hero", "Taylor Swift", "Pop", "English", "Melancholic", 97),
    ("Lavender Haze", "Taylor Swift", "Pop", "English", "Chill", 95),
    ("About Damn Time", "Lizzo", "Pop", "English", "Happy", 110),
    ("As It Was", "Harry Styles", "Pop", "English", "Chill", 173),
    ("Running Up That Hill", "Kate Bush", "Pop", "English", "Intense", 108),
    ("Sunroof", "Nicky Youre & dazy", "Pop", "English", "Happy", 158),
    ("First Class", "Jack Harlow", "Hip Hop", "English", "Energetic", 150),
    ("Break My Soul", "Beyoncé", "R&B", "English", "Energetic", 118),
    ("Glimpse of Us", "Joji", "Pop", "English", "Melancholic", 85),
    ("Late Night Talking", "Harry Styles", "Pop", "English", "Happy", 103),
    ("Numb", "Linkin Park", "Rock", "English", "Intense", 110),
    ("In The End", "Linkin Park", "Rock", "English", "Intense", 105),
    ("Somewhere I Belong", "Linkin Park", "Rock", "English", "Melancholic", 100),
    ("New Divide", "Linkin Park", "Rock", "English", "Intense", 120),
    ("Castle of Glass", "Linkin Park", "Rock", "English", "Melancholic", 95),
    ("What I've Done", "Linkin Park", "Rock", "English", "Intense", 115),
    ("Burn It Down", "Linkin Park", "Rock", "English", "Energetic", 125),
    ("Faint", "Linkin Park", "Rock", "English", "Intense", 130),
    ("Breaking The Habit", "Linkin Park", "Rock", "English", "Melancholic", 100),
    ("Numb", "Linkin Park", "Rock", "English", "Intense", 110),
    ("In The End", "Linkin Park", "Rock", "English", "Intense", 105),
    ("Somewhere I Belong", "Linkin Park", "Rock", "English", "Melancholic", 100),
    ("New Divide", "Linkin Park", "Rock", "English", "Intense", 120),
    ("Castle of Glass", "Linkin Park", "Rock", "English", "Melancholic", 95),
    ("What I've Done", "Linkin Park", "Rock", "English", "Intense", 115),
    ("Burn It Down", "Linkin Park", "Rock", "English", "Energetic", 125),
    ("Faint", "Linkin Park", "Rock", "English", "Intense", 130),
    ("Breaking The Habit", "Linkin Park", "Rock", "English", "Melancholic", 100),
]

if __name__ == "__main__":
    run()


