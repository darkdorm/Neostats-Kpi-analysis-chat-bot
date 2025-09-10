import streamlit as st
from sqlalchemy import text
from db_test import get_engine, init_users_table
import bcrypt
import requests
import urllib.parse
from config.config import CONFIG
from auth_utils import set_logged_in  # <-- helper

# Initialize DB
init_users_table()
engine = get_engine()


# ---------------- Signup/Login ----------------
def signup(username, email, password):
    """Register a new user"""
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO users (username, email, password_hash) VALUES (:u, :e, :p)"),
                {"u": username, "e": email, "p": password_hash}
            )
        st.success("Signup successful! Please login.")
        return True
    except Exception as e:
        st.error(f"Signup failed: {str(e)}")
        return False


def login(identifier, password):
    """Login with username OR email"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT username, password_hash FROM users WHERE username=:id OR email=:id"),
            {"id": identifier}
        ).fetchone()

        if result:
            stored_username, stored_hash = result
            stored_hash = stored_hash.encode()
            if bcrypt.checkpw(password.encode(), stored_hash):
                set_logged_in(stored_username, "local")  # ✅ fixed order
                st.experimental_rerun()
                return True

    st.error("Invalid username/email or password")
    return False


# ---------------- Streamlit UI ----------------
def auth_ui():
    st.title("Multi-Auth Demo: Username/Password + OAuth")
    auth_method = st.radio("Select Authentication Method", ["Username/Password", "GitHub", "Google"])

    # ---------------- Username/Password ----------------
    if auth_method == "Username/Password":
        action = st.selectbox("Action", ["Login", "Signup"])
        identifier = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")

        if action == "Signup":
            username = st.text_input("Choose a Username")
            email = st.text_input("Email")
            if st.button("Signup"):
                signup(username, email, password)
        else:
            if st.button("Login"):
                login(identifier, password)

    # ---------------- GitHub OAuth ----------------
    elif auth_method == "GitHub":
        github_auth_url = (
            f"https://github.com/login/oauth/authorize?"
            f"client_id={CONFIG['GITHUB_CLIENT_ID']}&redirect_uri={CONFIG['GITHUB_REDIRECT_URI']}&scope=read:user"
        )
        st.markdown(f"[Login with GitHub]({github_auth_url})")

        github_params = st.query_params
        if "code" in github_params:
            code = github_params["code"]
            token_resp = requests.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": CONFIG["GITHUB_CLIENT_ID"],
                    "client_secret": CONFIG["GITHUB_CLIENT_SECRET"],
                    "code": code,
                    "redirect_uri": CONFIG["GITHUB_REDIRECT_URI"]
                },
                headers={"Accept": "application/json"}
            )
            access_token = token_resp.json().get("access_token")
            if access_token:
                set_logged_in("GitHub_User", "github")  # ✅ fixed order
                st.experimental_rerun()
            else:
                st.error("GitHub login failed.")

    # ---------------- Google OAuth ----------------
    elif auth_method == "Google":
        google_params = {
            "client_id": CONFIG["GOOGLE_CLIENT_ID"],
            "redirect_uri": CONFIG["GOOGLE_REDIRECT_URI"],
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "consent"
        }
        google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(google_params)
        st.markdown(f"[Login with Google]({google_auth_url})")

        google_params_resp = st.query_params
        if "code" in google_params_resp:
            code = google_params_resp["code"]
            token_resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": CONFIG["GOOGLE_CLIENT_ID"],
                    "client_secret": CONFIG["GOOGLE_CLIENT_SECRET"],
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": CONFIG["GOOGLE_REDIRECT_URI"]
                }
            )
            access_token = token_resp.json().get("access_token")
            if access_token:
                set_logged_in("Google_User", "google")  # ✅ fixed order
                st.experimental_rerun()
            else:
                st.error("Google login failed.")
