import streamlit as st
import pandas as pd
from utils.helpers import inject_custom_css
from ai.conversation import ConversationManager
from ai.llm import OfflineLLMEngine

# Re-inject CSS for visual consistency
inject_custom_css()

# Initialize session history
ConversationManager.initialize_session()

# Title banner
st.markdown(
    """
    <div style='padding: 20px 0; border-bottom: 1px solid #334155; margin-bottom: 25px;'>
        <h1 style='font-family: Outfit, sans-serif; font-size: 2.5rem; margin: 0; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Offline AI Assistant
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Ask natural language questions about your dataset structure, schema, and recommended pipelines.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Validate that a dataset is loaded and memory is cached
if st.session_state.df is None or st.session_state.memory is None:
    st.warning("⚠️ No dataset or intelligence context loaded yet. Please go to the Ingestion page in the sidebar to load a database file.")
    st.stop()

df = st.session_state.df
filename = st.session_state.filename
memory = st.session_state.memory
context = memory["context"]

# ----------------------------------------------------
# DUAL-COLUMN LAYOUT
# ----------------------------------------------------
col_chat, col_diag = st.columns([1.7, 1.3])

# --- RIGHT COLUMN: DIAGNOSTIC PANEL & METRICS ---
with col_diag:
    st.markdown("### 📊 Assistant Metrics & Status")
    
    # 1. Model Status Card
    prov_name = "Ollama" if not st.session_state.ai_fallback_mode else "Fallback Mode"
    model_name = st.session_state.ai_selected_model if not st.session_state.ai_fallback_mode else "GroundedRuleEngine"
    
    st.markdown(
        f"""
        <div class="glass-card" style='padding: 15px !important; margin-bottom: 15px;'>
            <strong style='color:#38BDF8; font-size:1.05rem;'>🖥️ AI Runtime Engine</strong>
            <p style='margin:6px 0 0 0; font-size:0.9rem;'>- Active Provider: <strong>{prov_name}</strong></p>
            <p style='margin:3px 0 0 0; font-size:0.9rem;'>- Selected Model: <strong>{model_name or "None"}</strong></p>
            <p style='margin:3px 0 0 0; font-size:0.9rem;'>- System Persona: <strong>{st.session_state.ai_persona}</strong></p>
            <p style='margin:3px 0 0 0; font-size:0.9rem;'>- Memory ID: <strong style='font-family:monospace; font-size:0.8rem;'>{memory['metadata']['dataset_id'][:12]}...</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 2. Performance Metrics
    # Retrieve performance stats from last assistant response if exists
    last_msg = next((msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "assistant"), None)
    metrics = last_msg.get("metrics", {}) if last_msg else {}
    citations = last_msg.get("citations", []) if last_msg else []
    
    st.markdown("#### ⚡ Inference Diagnostics")
    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            label="Response Latency",
            value=f"{metrics.get('response_time_ms', 0)} ms" if metrics else "0 ms",
            help="Total time taken to analyze context and generate the response."
        )
    with m2:
        st.metric(
            label="Fallback Triggered",
            value="Yes" if metrics.get("fallback_used", False) else "No",
            help="Indicates if rule-based response was fired instead of local LLM connection."
        )
        
    st.metric(
        label="Context Nodes Inspected",
        value=f"{metrics.get('retrieved_context_count', 0)} nodes",
        help="Number of Knowledge Graph features and rules matched by retriever."
    )
    
    # 3. Grounded References Accordion
    st.markdown("#### 📖 Grounding Citations")
    if citations:
        for cit in citations:
            st.markdown(f"- ✔️ *{cit}*")
    else:
        st.info("Ask a question to inspect references.")
        
    # 4. Clear chat trigger
    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        ConversationManager.clear_history()
        st.rerun()

# --- LEFT COLUMN: CHAT WINDOW ---
with col_chat:
    st.markdown("### 💬 Chat Interface")
    
    # Render chat logs
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Suggested Questions Buttons
    st.markdown("#### 💡 Suggested Questions")
    s_col1, s_col2 = st.columns(2)
    
    suggested_q = None
    with s_col1:
        if st.button("📋 What is this dataset about?", key="sq_about", use_container_width=True):
            suggested_q = "What is this dataset about?"
        if st.button("⚠️ Which columns contain missing values?", key="sq_missing", use_container_width=True):
            suggested_q = "Which columns contain missing values?"
        if st.button("🛠️ What preprocessing is recommended?", key="sq_prepro", use_container_width=True):
            suggested_q = "What preprocessing is recommended?"
            
    with s_col2:
        if st.button("⚖️ Why is normalization recommended?", key="sq_norm", use_container_width=True):
            suggested_q = "Why is normalization recommended?"
        if st.button("🤖 Which algorithm should I use?", key="sq_algo", use_container_width=True):
            suggested_q = "Which algorithm should I use?"
        if st.button("📈 Explain the data quality score.", key="sq_quality", use_container_width=True):
            suggested_q = "Explain the data quality score."

    # Handle suggested question clicks or chat input
    user_query = st.chat_input("Ask a question about the dataset...")
    if suggested_q:
        user_query = suggested_q
        
    if user_query:
        # Render User message
        with st.chat_message("user"):
            st.markdown(user_query)
            
        ConversationManager.add_message("user", user_query)
        
        # Render Assistant streaming response container
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            # Call coordinator LLM engine generator
            stream_gen = OfflineLLMEngine.answer_question(
                question=user_query,
                memory_obj=memory,
                persona=st.session_state.ai_persona,
                stream=st.session_state.ai_streaming
            )
            
            full_text = ""
            metrics_payload = {}
            citations_payload = []
            
            for event in stream_gen:
                if event["type"] == "chunk":
                    full_text += event["text"]
                    response_container.markdown(full_text + "▌")
                elif event["type"] == "final":
                    response_container.markdown(event["content"])
                    metrics_payload = event["metrics"]
                    citations_payload = event["citations"]
                    
            ConversationManager.add_message("assistant", full_text, metrics_payload, citations_payload)
            st.rerun()
