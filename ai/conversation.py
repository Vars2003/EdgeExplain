import json
import streamlit as st
from typing import List, Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger("ai.conversation")

class ConversationManager:
    """
    Manages short-term conversation logs in Streamlit session state.
    Provides history trimming, JSON exports, and hooks for long-term persistence.
    """

    @staticmethod
    def initialize_session() -> None:
        """
        Initializes Streamlit session states for chat logs.
        """
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            logger.info("Initialized fresh chat history list in session state.")

    @staticmethod
    def add_message(role: str, content: str, metrics: Optional[Dict[str, Any]] = None, citations: Optional[List[str]] = None) -> None:
        """
        Appends a message to the active session state history list.
        """
        ConversationManager.initialize_session()
        st.session_state.chat_history.append({
            "role": role,
            "content": content,
            "metrics": metrics or {},
            "citations": citations or []
        })
        logger.info(f"Added message from role '{role}' to chat history.")

    @staticmethod
    def get_messages(limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the last 'limit' messages from the short-term cache.
        """
        ConversationManager.initialize_session()
        return st.session_state.chat_history[-limit:]

    @staticmethod
    def clear_history() -> None:
        """
        Resets conversation records.
        """
        st.session_state.chat_history = []
        logger.info("Cleared conversation logs.")

    @staticmethod
    def export_history_as_json() -> str:
        """
        Serializes chat history to a JSON string for download.
        """
        ConversationManager.initialize_session()
        return json.dumps(st.session_state.chat_history, indent=2)

    # --- Future Long-Term Memory Stub ---
    
    @staticmethod
    def persist_to_long_term_storage(dataset_id: str) -> bool:
        """
        Interface hook prepared for saving chat logs to disk database in future phases.
        """
        logger.info(f"Persisting conversation checkpoint (Stub) for dataset: {dataset_id}")
        return True
