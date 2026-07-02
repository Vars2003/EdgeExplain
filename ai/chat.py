from typing import List, Dict, Any
from utils.logger import get_logger
from ai.llm import LocalLLMConnector

logger = get_logger("ai.chat")

class LocalAIChatSession:
    """
    Manages conversational memory loops and history buffers offline,
    passing conversation context to the local LLM engine.
    """

    def __init__(self, llm_connector: LocalLLMConnector):
        self.llm = llm_connector
        self.history: List[Dict[str, str]] = []
        logger.info("Local AI chat session initialized.")

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()
        logger.info("Chat history cleared.")

    def submit_user_query(self, query: str, data_context_str: str) -> str:
        """
        Appends user query, packages it with local dataset context, and queries the LLM.
        """
        self.add_message("user", query)
        
        # Package prompt with RAG grounding context
        system_prompt = (
            "You are EdgeExplain's Offline AI Data Scientist. You reason about "
            "dataset statistics, feature dependencies, and data quality issues. "
            "Answer user questions accurately and keep answers strictly grounded in "
            "the provided dataset context. Do not make up facts."
        )
        
        prompt = f"Dataset Grounding Context:\n{data_context_str}\n\nUser Question: {query}"
        
        # Run inference
        response = self.llm.generate_response(prompt, system_prompt)
        self.add_message("assistant", response)
        
        return response
