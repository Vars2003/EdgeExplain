import streamlit as st
from utils.helpers import inject_custom_css

# Re-inject css for sub-page rendering consistency
inject_custom_css()

# Header layout with gradient accent
st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Welcome to EdgeExplain
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Your Offline AI Data Scientist Platform — Secure, Private, and Local.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Grid (2 columns: Intro & Workflow)
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h3>🛡️ Edge AI: Pure Local Autonomy</h3>
            <p style='color: #E2E8F0; line-height: 1.6;'>
                Traditional data tools rely on cloud APIs, exposing sensitive business files and proprietary schemas 
                to external servers. <strong>EdgeExplain</strong> operates with complete data sovereignty:
            </p>
            <ul style='color: #E2E8F0; line-height: 1.6; margin-left: 20px;'>
                <li><strong>Zero Network Calls</strong>: Every analysis, check, and statistical model runs locally.</li>
                <li><strong>No API Keys Needed</strong>: Operates completely offline, safe from server outages and network delays.</li>
                <li><strong>Maximum Security</strong>: Your data never leaves your device. Compliant with strict data privacy guidelines.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="glass-card">
            <h3>🚀 Core Capabilities</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;'>
                <div style='padding: 10px; border-left: 3px solid #38BDF8; background: rgba(56,189,248,0.05);'>
                    <strong style='color:#38BDF8;'>Semantic Parser</strong><br>
                    <span style='font-size:0.85rem; color:#94A3B8;'>Deduces variable categories beyond plain raw data types.</span>
                </div>
                <div style='padding: 10px; border-left: 3px solid #818CF8; background: rgba(129,140,248,0.05);'>
                    <strong style='color:#818CF8;'>Custom Statistics</strong><br>
                    <span style='font-size:0.85rem; color:#94A3B8;'>Calculates skewness, kurtosis, and quartiles from scratch.</span>
                </div>
                <div style='padding: 10px; border-left: 3px solid #34D399; background: rgba(52,211,153,0.05);'>
                    <strong style='color:#34D399;'>Dependency Graph</strong><br>
                    <span style='font-size:0.85rem; color:#94A3B8;'>Analyzes relationships via Cramer's V, ANOVA, and Mutual Info.</span>
                </div>
                <div style='padding: 10px; border-left: 3px solid #FBBF24; background: rgba(251,191,36,0.05);'>
                    <strong style='color:#FBBF24;'>ML Matcher</strong><br>
                    <span style='font-size:0.85rem; color:#94A3B8;'>Ranks offline algorithms matching your dataset volume and targets.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="glass-card" style='height: 100%;'>
            <h3>📋 Platform Pipeline Workflow</h3>
            <div style='position: relative; padding-left: 20px; border-left: 2px dashed #334155; margin: 15px 0 15px 10px;'>
                <div style='margin-bottom: 20px;'>
                    <span style='position: absolute; left: -9px; background: #38BDF8; width: 16px; height: 16px; border-radius: 50%; display: inline-block;'></span>
                    <strong style='color: #F8FAFC;'>1. Data Ingestion</strong>
                    <p style='font-size: 0.85rem; color: #94A3B8; margin: 3px 0 0 0;'>Loads CSV, Excel, Parquet, or JSON with auto-type validation.</p>
                </div>
                <div style='margin-bottom: 20px;'>
                    <span style='position: absolute; left: -9px; background: #818CF8; width: 16px; height: 16px; border-radius: 50%; display: inline-block;'></span>
                    <strong style='color: #F8FAFC;'>2. Structural Analytics</strong>
                    <p style='font-size: 0.85rem; color: #94A3B8; margin: 3px 0 0 0;'>Profiles duplicates, calculates outliers, and generates quality scores.</p>
                </div>
                <div style='margin-bottom: 20px;'>
                    <span style='position: absolute; left: -9px; background: #34D399; width: 16px; height: 16px; border-radius: 50%; display: inline-block;'></span>
                    <strong style='color: #F8FAFC;'>3. Cognitive Layer</strong>
                    <p style='font-size: 0.85rem; color: #94A3B8; margin: 3px 0 0 0;'>Deduces business domains and reasons about structural anomalies.</p>
                </div>
                <div>
                    <span style='position: absolute; left: -9px; background: #FBBF24; width: 16px; height: 16px; border-radius: 50%; display: inline-block;'></span>
                    <strong style='color: #F8FAFC;'>4. Decisions & Modeling</strong>
                    <p style='font-size: 0.85rem; color: #94A3B8; margin: 3px 0 0 0;'>Suggests robust scaling options and matches algorithm suitability.</p>
                </div>
            </div>
            <hr style='border-color: #334155; margin: 15px 0;'>
            <h4>📂 Supported File Formats</h4>
            <span class="tag tag-blue">CSV (.csv)</span>
            <span class="tag tag-purple">Excel (.xlsx, .xls)</span>
            <span class="tag tag-green">Parquet (.parquet)</span>
            <span class="tag tag-orange">JSON (.json)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer Privacy Guarantee Card
st.markdown(
    """
    <div style='margin-top: 20px; padding: 20px; border-radius: 12px; background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2);'>
        <h4 style='color: #EF4444; margin: 0;'>🔒 Security & Privacy Notice</h4>
        <p style='color: #F8FAFC; font-size: 0.9rem; margin: 8px 0 0 0; line-height: 1.5;'>
            This system runs <strong>100% locally</strong>. Data is kept in-memory and cached within Streamlit session states. 
            There are no cookies tracking your behavior, no external script CDNs enabled, and no telemetry pings.
            Your local server logs are located in the <code>logs/</code> directory for complete audit transparency.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
