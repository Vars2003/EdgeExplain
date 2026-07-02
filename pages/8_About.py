import streamlit as st
from utils.helpers import inject_custom_css

# Re-inject CSS for visual consistency
inject_custom_css()

st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 30px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            About Platform
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Technical briefing, architectural maps, and offline edge software dependencies.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col_about, col_tech = st.columns([1, 1])

with col_about:
    st.subheader("🧠 EdgeExplain Overview")
    st.markdown(
        """
        **EdgeExplain** is a complete, private data intelligence terminal. Unlike common cloud dashboards 
        that transfer database records across network sockets, EdgeExplain operates entirely within the host CPU 
        boundaries, utilizing compiled numeric libraries.
        
        ### 🔄 Decoupled Layer Architecture
        The codebase is partitioned into two independent execution layers:
        
        1. **Analytics Layer**: Implements vectorized, thread-safe mathematical checks (outlier thresholds, 
           skewness definitions, Cramer's V correlations, and Plotly graph elements) using NumPy and Pandas.
        2. **AI Layer**: Ingests numerical and categorical metadata to run rule-matching heuristics, domain 
           semantic dictionaries, and logic justification trees. It operates as the grounding engine for offline LLMs.
        
        ### 🗺️ Data Science Pipeline
        ```
        Dataset Ingestion
           ↓
        Custom Profiling (Data Quality, Duplicates, Outlier counts)
           ↓
        Multi-Type Dependencies (Pearson, Cramer's V, ANOVA, Mutual Info)
           ↓
        Semantic Parsing (continuous, discrete, nominal, binary, coordinates)
           ↓
        Reasoning Engine (Confidence, Justifications, Explanations)
           ↓
        Recommendations (Imputation, Scaling, Redundancy selectors)
           ↓
        Algorithm Selection (Algorithm compatibility scoring)
        ```
        """
    )

with col_tech:
    st.subheader("🛠️ Technology Stack")
    st.markdown(
        """
        - **Frontend Framework**: [Streamlit v1.35+](https://streamlit.io/) (Modular navigation)
        - **Vector Processing**: [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/)
        - **Scientific Math**: [SciPy Stats](https://scipy.org/) (ANOVA, Chi2, Cramer's V)
        - **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Mutual Information, Decision Trees)
        - **Interactive Visuals**: [Plotly Express](https://plotly.com/) (Gauges, Network Graphs)
        - **Fast File Ingestion**: [PyArrow](https://arrow.apache.org/) & [OpenPyXL](https://openpyxl.readthedocs.io/)
        - **Local Database**: [DuckDB](https://duckdb.org/)
        
        ### 🔒 Security Sovereignty
        This terminal enforces strict data isolation:
        - **No network trackers** or remote telemetry calls.
        - **No external JavaScript libraries** fetched from CDNs.
        - All logs written locally to `logs/edgeexplain.log` for audit audits.
        
        ---
        **Version**: 1.0.0  
        **Licensing**: Proprietary Edge Deployment  
        **Author**: EdgeExplain Software Architect Group
        """
    )
