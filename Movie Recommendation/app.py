# Required Libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import requests
import json
import os
import hashlib
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Initialize user data storage
USER_FILE = "users.json"
if not os.path.exists(USER_FILE):
    with open(USER_FILE, 'w') as f:
        json.dump({}, f)

# Load movie data
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('model.pkl', 'rb'))

# Helper Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except:
        pass
    return None

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    names, ids = [], []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        ids.append(movie_id)
        names.append(movies.iloc[i[0]].title)
    return names, ids

def load_users():
    with open(USER_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def save_user_input(username, movie):
    users = load_users()
    if username in users:
        users[username].setdefault("history", []).append(movie)
        save_users(users)

def display_user_history(username):
    users = load_users()
    return users.get(username, {}).get("history", [])

# Streamlit App Navigation
st.set_page_config(page_icon="🎦", page_title="Movie Recommender", layout="wide")
# st.sidebar.title("Navigation")
st.title("🎦 Movie Recommendation System Using Content-Based Filtering")
selected = option_menu(
    menu_title=None,
    options=["Home", "Login", "SignUp", "Metrics Dashboard"],
    icons=["house", "box-arrow-in-right", "person-plus", "bar-chart"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Global user session
if 'user' not in st.session_state:
    st.session_state.user = None

# Logout option
if st.session_state.user:
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.success("Logged out successfully")

# Home Page
if selected == "Home":
    st.header('Movie Recommender System')
    if st.session_state.user:
        movie_list = movies['title'].values
        selected_movie = st.selectbox("Type or select a movie", movie_list)
        if st.button('Show Recommendation'):
            save_user_input(st.session_state.user, selected_movie)
            names, ids = recommend(selected_movie)

            st.markdown("### Recommended Movies")
            cols = st.columns(5)
            for col, name, movie_id in zip(cols, names, ids):
                poster = fetch_poster(movie_id)
                if poster:
                    col.image(poster, use_container_width=True)
                col.markdown(f"**{name}**")

        st.markdown("---")
        st.subheader("Your Watch History")
        history = display_user_history(st.session_state.user)
        st.write(history if history else "No history yet.")
    else:
        st.warning("Please signup or login to use the recommendation system.")

# Login Page
elif selected == "Login":
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = load_users()
        if username in users and users[username]['password'] == hash_password(password):
            st.success("Logged in successfully")
            st.session_state.user = username
        else:
            st.error("Invalid username or password")

# Signup Page
elif selected == "SignUp":
    st.header("Sign Up")
    new_user = st.text_input("Choose a Username")
    new_pass = st.text_input("Choose a Password", type="password")
    if st.button("Sign Up"):
        users = load_users()
        if new_user in users:
            st.error("Username already exists")
        else:
            users[new_user] = {"password": hash_password(new_pass), "history": []}
            save_users(users)
            st.success("Signup successful! Please login.")

# Dashboard Page
elif selected == "Metrics Dashboard":
    st.header("Metrics Dashboard")
    base_dir = "plots"
    images_list = os.listdir(base_dir)
    for i in images_list:
        st.image(os.path.join(base_dir, i))
        st.divider()