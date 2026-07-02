import streamlit as st
from utils.helpers import inject_custom_css, format_bytes

# Re-inject CSS for layout consistency
inject_custom_css()

# Title banner
st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Executive BI Dashboard
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Offline Data Science and Structural Intelligence Summary.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# EMPTY STATE: ONBOARDING PANEL
# ----------------------------------------------------
if st.session_state.df is None:
    st.markdown(
        """
        <div class="glass-card">
            <h2 style='color:#38BDF8; font-family: Outfit, sans-serif;'>👋 Welcome to EdgeExplain</h2>
            <p style='color:#E2E8F0; font-size:1.05rem; line-height:1.6;'>
                EdgeExplain is an offline-first, private workspace designed to analyze, score, and recommendation-tune 
                your datasets without sending any bytes to external servers. Follow these quick steps to launch:
            </p>
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 25px 0;'>
                <div style='background: rgba(15,23,42,0.4); border: 1px solid #334155; border-radius: 8px; padding: 20px;'>
                    <span style='font-size: 2rem;'>📂</span>
                    <h4 style='margin: 10px 0 5px 0;'>1. Supported Formats</h4>
                    <p style='font-size: 0.85rem; color: #94A3B8;'>Ingest standard tables in CSV, Excel, Parquet, or JSON format. Up to 100MB.</p>
                </div>
                <div style='background: rgba(15,23,42,0.4); border: 1px solid #334155; border-radius: 8px; padding: 20px;'>
                    <span style='font-size: 2rem;'>📤</span>
                    <h4 style='margin: 10px 0 5px 0;'>2. Local Ingestion</h4>
                    <p style='font-size: 0.85rem; color: #94A3B8;'>Upload your file. The parser detects datatypes and maps variable relationships.</p>
                </div>
                <div style='background: rgba(15,23,42,0.4); border: 1px solid #334155; border-radius: 8px; padding: 20px;'>
                    <span style='font-size: 2rem;'>🧠</span>
                    <h4 style='margin: 10px 0 5px 0;'>3. Core Evaluations</h4>
                    <p style='font-size: 0.85rem; color: #94A3B8;'>Review quality scores, machine learning readiness profiles, and cleaning tips.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Ingestion quick action redirect
    st.markdown("### Get Started Now")
    if st.button("📤 Load Dataset File", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Upload.py")
        
    st.markdown("---")
    st.markdown("#### 🔄 Processing Pipeline Workflow")
    st.markdown(
        """
        ```
        Dataset Ingested ➔ Custom Profiler ➔ Custom Statistics ➔ Dependency Matrix ➔ Cognitive Reasoning ➔ Preprocessing Plan
        ```
        """
    )
    st.stop()

# ----------------------------------------------------
# ACTIVE STATE: BI EXECUTIVE SUMMARY DASHBOARD
# ----------------------------------------------------
df = st.session_state.df
filename = st.session_state.filename
metrics = st.session_state.metrics
domain_info = st.session_state.domain
task_info = st.session_state.dataset_type
insights = st.session_state.insights

rows, cols = df.shape
quality_score = metrics["quality_score"]
readiness_score = metrics["ml_readiness"]["score"]

# Calculate Risk parameters
risk_level = "Low"
risk_color = "#10B981"
if quality_score < 70 or len([i for i in insights if i["severity"] == "High"]) > 2:
    risk_level = "High"
    risk_color = "#EF4444"
elif quality_score < 85 or len([i for i in insights if i["severity"] == "High"]) > 0:
    risk_level = "Medium"
    risk_color = "#F59E0B"

# Render 10 Distinct KPI cards
st.markdown(
    f"""
    <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 25px;'>
        <div class="stat-card" style='border-color: rgba(56, 189, 248, 0.4);'>
            <div class="stat-label">Total Observations</div>
            <div class="stat-val" style='color: #38BDF8;'>{rows:,}</div>
        </div>
        <div class="stat-card" style='border-color: rgba(56, 189, 248, 0.4);'>
            <div class="stat-label">Dimensions</div>
            <div class="stat-val" style='color: #38BDF8;'>{cols}</div>
        </div>
        <div class="stat-card" style='border-color: rgba(56, 189, 248, 0.4);'>
            <div class="stat-label">Memory Size</div>
            <div class="stat-val" style='color: #38BDF8; font-size:1.3rem; padding-top:6px;'>{metrics['basic_metrics']['memory_readable']}</div>
        </div>
        <div class="stat-card" style='border-color: rgba(245, 158, 11, 0.4);'>
            <div class="stat-label">Missing Cells</div>
            <div class="stat-val" style='color: #F59E0B;'>{metrics['missing_data']['total_missing']:,}</div>
        </div>
        <div class="stat-card" style='border-color: rgba(245, 158, 11, 0.4);'>
            <div class="stat-label">Duplicate Rows</div>
            <div class="stat-val" style='color: #F59E0B;'>{metrics['basic_metrics']['duplicate_rows']:,}</div>
        </div>
        <div class="stat-card" style='border-color: rgba(16, 185, 129, 0.4);'>
            <div class="stat-label">Quality Score</div>
            <div class="stat-val" style='color: #10B981;'>{quality_score:.1f}%</div>
        </div>
        <div class="stat-card" style='border-color: rgba(129, 140, 248, 0.4);'>
            <div class="stat-label">ML Readiness</div>
            <div class="stat-val" style='color: #818CF8;'>{readiness_score:.1f}%</div>
        </div>
        <div class="stat-card" style='border-color: rgba(236, 72, 153, 0.4);'>
            <div class="stat-label">Dataset Focus</div>
            <div class="stat-val" style='color: #EC4899; font-size: 1.15rem; padding-top:8px;'>{task_info['type']}</div>
        </div>
        <div class="stat-card" style='border-color: rgba(139, 92, 246, 0.4);'>
            <div class="stat-label">Source Domain</div>
            <div class="stat-val" style='color: #8B5CF6; font-size: 1.15rem; padding-top:8px;'>{domain_info['domain'].replace(" Dataset", "")}</div>
        </div>
        <div class="stat-card" style='border-color: {risk_color}66;'>
            <div class="stat-label">Overall Risk</div>
            <div class="stat-val" style='color: {risk_color};'>{risk_level}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Visual Gauges
st.markdown("### 📊 Dataset Health Gauges")
g_col1, g_col2, g_col3 = st.columns(3)

with g_col1:
    st.markdown(f"**Data Quality Completeness** ({quality_score:.1f}%)")
    st.progress(quality_score / 100.0)
with g_col2:
    st.markdown(f"**ML Modeling Readiness** ({readiness_score:.1f}%)")
    st.progress(readiness_score / 100.0)
with g_col3:
    # Scale risk: Low = 100% health, Med = 50% health, High = 10% health
    risk_val = 0.95 if risk_level == "Low" else 0.50 if risk_level == "Medium" else 0.20
    st.markdown(f"**Security & Risk Health** (Level: {risk_level})")
    st.progress(risk_val)

st.markdown("---")

# Quick Actions Panel & Insights Summary Grid
c_act, c_ins = st.columns([0.8, 1.2])

with c_act:
    st.markdown("### ⚡ Quick Actions")
    
    # Render quick action buttons that trigger switch_page
    if st.button("📤 Upload New Dataset", use_container_width=True):
        st.switch_page("pages/2_Upload.py")
        
    if st.button("📊 Explore Core Analytics", use_container_width=True, type="secondary"):
        st.switch_page("pages/3_Analytics.py")
        
    if st.button("🧠 Review Cognitive AI Insights", use_container_width=True, type="secondary"):
        st.switch_page("pages/4_AI_Insights.py")
        
    if st.button("📈 Open Visualization Studio", use_container_width=True, type="secondary"):
        st.switch_page("pages/5_Visualizations.py")
        
    if st.button("💡 Explore Edge Intelligence Graph", use_container_width=True, type="secondary"):
        st.switch_page("pages/9_Edge_Intelligence.py")
        
    if st.button("💾 Open Download & Export Center", use_container_width=True, type="secondary"):
        st.switch_page("pages/6_Reports.py")

with c_ins:
    st.markdown("### 💡 High-Severity Quality Warnings")
    # Pull top 3 high or medium severity insights
    critical_findings = [i for i in insights if i["severity"] in ["High", "Medium"]]
    if not critical_findings:
        st.success("🎉 No high or medium severity risks found in the dataset structure.")
    else:
        for idx, cf in enumerate(critical_findings[:3]):
            s_color = "#F87171" if cf["severity"] == "High" else "#FBBF24"
            st.markdown(
                f"""
                <div style='padding: 12px; margin-bottom: 10px; border-left: 4px solid {s_color}; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;'>
                    <strong style='color: #F8FAFC;'>{cf['title']}</strong>
                    <p style='margin: 3px 0 0 0; font-size: 0.85rem; color: #94A3B8;'>{cf['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
