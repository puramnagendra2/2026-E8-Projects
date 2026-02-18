from fastapi import FastAPI, Request, Form, Depends, Response, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
import os
import re
import secrets
import hashlib
import json
import pickle
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load movie data (must include movie_id, title, poster_path)
movies = pickle.load(open("movie_list.pkl", "rb"))
similarity = pickle.load(open("model.pkl", "rb"))

# Session management
active_sessions = {}

# Database setup
DATABASE_PATH = "movie_recommender.db"


@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                movie_title TEXT,
                searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """
        )
        conn.commit()


init_db()

# Templates setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize user data storage
USER_FILE = "users.json"
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

# Helper Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Use TMDB Images API directly
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def fetch_poster(movie_id):
    movie = movies[movies["movie_id"] == movie_id].iloc[0]
    poster_path = (
        movie["poster_path"] if "poster_path" in movie and movie["poster_path"] else None
    )
    if poster_path:
        return f"{IMAGE_BASE_URL}{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Image"


def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1]
    )
    names, ids = [], []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        ids.append(movie_id)
        names.append(movies.iloc[i[0]].title)
    return names, ids


def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)


def save_user_history(username, movie):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            cursor.execute(
                "INSERT INTO user_history (user_id, movie_title) VALUES (?, ?)",
                (user["id"], movie),
            )
            conn.commit()


def get_user_history(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT movie_title, searched_at 
            FROM user_history 
            JOIN users ON user_history.user_id = users.id 
            WHERE username = ? 
            ORDER BY searched_at DESC
        """,
            (username,),
        )
        return cursor.fetchall()


# Utility functions
def validate_email(email: str):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def create_session(username: str):
    session_token = secrets.token_hex(16)
    expiry = datetime.now() + timedelta(hours=1)
    active_sessions[session_token] = {"username": username, "expiry": expiry}
    return session_token


def validate_session(session_token: str):
    if session_token not in active_sessions:
        return False

    session = active_sessions[session_token]
    if datetime.now() > session["expiry"]:
        del active_sessions[session_token]
        return False

    return True


def get_current_user(session_token: str = Cookie(None)):
    if session_token and validate_session(session_token):
        return active_sessions[session_token]["username"]
    return None


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, session_token: str = Cookie(None)):
    username = get_current_user(session_token)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "username": username, "movies": movies["title"].values.tolist()},
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, session_token: str = Cookie(None)):
    username = get_current_user(session_token)
    return templates.TemplateResponse(
        "about.html", {"request": request, "username": username}
    )


@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup", response_class=HTMLResponse)
async def signup_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    # Input validation
    if len(username) < 4:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Username must be at least 4 characters long"},
            status_code=400,
        )

    if len(password) < 8:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Password must be at least 8 characters long"},
            status_code=400,
        )

    if not validate_email(email):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Invalid email format"},
            status_code=400,
        )

    if password != confirm_password:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Passwords do not match"},
            status_code=400,
        )

    # Hash password
    hashed_password = hash_password(password)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Check if username or email already exists
            cursor.execute(
                "SELECT 1 FROM users WHERE username = ? OR email = ? LIMIT 1",
                (username, email),
            )
            if cursor.fetchone():
                return templates.TemplateResponse(
                    "signup.html",
                    {"request": request, "error": "Username or email already exists"},
                    status_code=400,
                )

            # Create new user
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed_password),
            )
            conn.commit()

        logger.info(f"New user registered: {username}")
        return RedirectResponse(url="/login", status_code=303)

    except sqlite3.Error as e:
        logger.error(f"Database error during registration: {str(e)}")
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Registration failed. Please try again."},
            status_code=500,
        )


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

        if not user or user["password"] != hash_password(password):
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Invalid username or password"},
                status_code=401,
            )

        # Create session and set cookie
        session_token = create_session(username)
        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        return response

    except sqlite3.Error as e:
        logger.error(f"Database error during login: {str(e)}")
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Login failed. Please try again."},
            status_code=500,
        )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, session_token: str = Cookie(None)):
    username = get_current_user(session_token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "username": username, "movies": movies["title"].values.tolist()},
    )


@app.post("/recommend", response_class=HTMLResponse)
async def get_recommendations(request: Request, movie: str = Form(...), session_token: str = Cookie(None)):
    username = get_current_user(session_token)
    if not username:
        return RedirectResponse(url="/login", status_code=303)

    try:
        # Save user search history
        save_user_history(username, movie)

        # Get recommendations
        names, ids = recommend(movie)
        posters = [fetch_poster(movie_id) for movie_id in ids]

        # Get user history
        history = get_user_history(username)

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "username": username,
                "movies": movies["title"].values.tolist(),
                "selected_movie": movie,
                "recommendations": zip(names, posters),
                "history": history,
                "has_recommendations": True,
            },
        )

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "username": username,
                "movies": movies["title"].values.tolist(),
                "error": f"Error getting recommendations: {str(e)}",
            },
        )


@app.get("/logout", response_class=HTMLResponse)
async def logout(request: Request, session_token: str = Cookie(None)):
    if session_token in active_sessions:
        del active_sessions[session_token]

    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("session_token")
    return response


@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request, session_token: str = Cookie(None)):
    username = get_current_user(session_token)
    return templates.TemplateResponse(
        "contact.html", {"request": request, "username": username}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
