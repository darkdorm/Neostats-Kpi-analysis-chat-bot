import streamlit as st

# ----------------- Session State Initialization -----------------
def init_auth_state():
    """Ensure all authentication-related session state keys exist."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "github_logged_in" not in st.session_state:
        st.session_state.github_logged_in = False
    if "google_logged_in" not in st.session_state:
        st.session_state.google_logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None


# ----------------- Set Login State -----------------
def set_logged_in(method: str, username: str = None):
    """
    Mark the user as logged in using the given method.
    method: "username", "github", or "google"
    """
    init_auth_state()  # make sure keys exist
    if method == "username":
        st.session_state.logged_in = True
        st.session_state.github_logged_in = False
        st.session_state.google_logged_in = False
    elif method == "github":
        st.session_state.github_logged_in = True
        st.session_state.logged_in = False
        st.session_state.google_logged_in = False
    elif method == "google":
        st.session_state.google_logged_in = True
        st.session_state.github_logged_in = False
        st.session_state.logged_in = False

    if username:
        st.session_state.username = username


# ----------------- Check if user is authenticated -----------------
def is_authenticated() -> bool:
    """
    Returns True if user is logged in (via username/password, Google, or GitHub).
    """
    init_auth_state()
    return (
        st.session_state.logged_in
        or st.session_state.github_logged_in
        or st.session_state.google_logged_in
    )
