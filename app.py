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
if "memory" not in st.session_state:
    st.session_state.memory = None

# Configurable Settings State keys
if "outlier_threshold" not in st.session_state:
    st.session_state.outlier_threshold = settings.DEFAULT_OUTLIER_THRESHOLD
if "correlation_threshold" not in st.session_state:
    st.session_state.correlation_threshold = settings.DEFAULT_CORRELATION_THRESHOLD
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "theme" not in st.session_state:
    st.session_state.theme = settings.DEFAULT_THEME

# AI Assistant Session state initializations
if "ai_fallback_mode" not in st.session_state:
    st.session_state.ai_fallback_mode = True
if "ai_selected_model" not in st.session_state:
    st.session_state.ai_selected_model = None
if "ai_max_context" not in st.session_state:
    st.session_state.ai_max_context = 2048
if "ai_temperature" not in st.session_state:
    st.session_state.ai_temperature = 0.7
if "ai_persona" not in st.session_state:
    st.session_state.ai_persona = "Data Analyst"
if "ai_streaming" not in st.session_state:
    st.session_state.ai_streaming = True

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

# Modern Streamlit Page Navigation - Grouped BI Structure
try:
    pages = {
        "Dashboard": [
            st.Page("pages/1_Home.py", title="Dashboard Summary", icon="🏠"),
            st.Page("pages/2_Upload.py", title="Data Ingestion", icon="📤")
        ],
        "Core Intelligence": [
            st.Page("pages/3_Analytics.py", title="Analytics Explorer", icon="📊"),
            st.Page("pages/4_AI_Insights.py", title="AI Insights", icon="🧠"),
            st.Page("pages/5_Visualizations.py", title="Visualizations", icon="📈"),
            st.Page("pages/9_Edge_Intelligence.py", title="Edge Intelligence", icon="💡"),
            st.Page("pages/10_AI_Assistant.py", title="AI Chat Assistant", icon="💬"),
            st.Page("pages/11_Explainability.py", title="XAI Explainability", icon="🧬"),
            st.Page("pages/12_Dataset_Comparison.py", title="Dataset Comparison", icon="⚖️"),
            st.Page("pages/13_Model_Recommendations.py", title="AutoML Pipelines", icon="🤖"),
            st.Page("pages/14_AI_Evaluation.py", title="AI Chat Evaluation", icon="🎯"),
            st.Page("pages/3_Dataset_Overview.py", title="Legacy All-In-One", icon="🛡️")
        ],
        "Management": [
            st.Page("pages/6_Reports.py", title="Export Center", icon="💾"),
            st.Page("pages/7_Settings.py", title="Analysis Settings", icon="⚙️"),
            st.Page("pages/8_About.py", title="About Platform", icon="ℹ️")
        ]
    }
    pg = st.navigation(pages)
    pg.run()
except Exception as e:
    st.error(f"Navigation router initialization failed: {e}")
