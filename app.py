import streamlit as st
from utils.helpers import inject_custom_css
from config import settings

# Must be the first streamlit call
st.set_page_config(
    page_title="EdgeExplain - Offline AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Global Premium Theme
inject_custom_css()

# Initialize global session state keys
if "df" not in st.session_state:
    st.session_state.df = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "semantic_types" not in st.session_state:
    st.session_state.semantic_types = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "domain" not in st.session_state:
    st.session_state.domain = None
if "dataset_type" not in st.session_state:
    st.session_state.dataset_type = None
if "recs" not in st.session_state:
    st.session_state.recs = None
if "insights" not in st.session_state:
    st.session_state.insights = None

# Sidebar branding header
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px; margin-bottom: 20px; 
                background: linear-gradient(135deg, rgba(26, 86, 219, 0.2), rgba(126, 58, 242, 0.2)); 
                border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;'>
        <span style='font-size: 2.2rem;'>🧠</span>
        <h2 style='margin: 5px 0 0 0; font-family: Outfit, sans-serif; font-size: 1.5rem; color: #F8FAFC;'>EdgeExplain</h2>
        <span style='font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em;'>Offline AI Scientist</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Modern Streamlit Page Navigation
try:
    pg = st.navigation([
        st.Page("pages/1_Home.py", title="Home Page", icon="🏠"),
        st.Page("pages/2_Upload.py", title="Data Ingestion", icon="📤"),
        st.Page("pages/3_Dataset_Overview.py", title="Dataset Intelligence", icon="📊")
    ])
    pg.run()
except Exception as e:
    st.error(f"Navigation router initialization failed: {e}")
