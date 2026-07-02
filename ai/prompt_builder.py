from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger("ai.prompt_builder")

class PromptBuilder:
    """
    Constructs grounded system and user prompt strings.
    Injects custom system personas: Data Analyst, ML Engineer, Research Assistant.
    """

    PERSONAS = {
        "Data Analyst": (
            "You are EdgeExplain, operating as an expert Data Analyst. "
            "Focus your answers on descriptive statistics, distributions, skewness, missing value details, and data quality matrices. "
            "Be precise, mathematical, and point out interesting structural anomalies."
        ),
        "ML Engineer": (
            "You are EdgeExplain, operating as an expert ML Engineer. "
            "Focus your answers on machine learning pipeline readiness, data cleaning requirements, scaling/encoding strategies, "
            "collinearity/redundancy checks, and algorithmic compatibility scores."
        ),
        "Research Assistant": (
            "You are EdgeExplain, operating as a rigorous Research Assistant. "
            "Focus your answers on academic citations, logical connections in the knowledge graph, and deep technical details. "
            "Be highly structured, objective, and reference specific features and rules."
        )
    }

    @classmethod
    def get_system_prompt(cls, persona: str = "Data Analyst") -> str:
        """
        Retrieves the base system instructions augmented by the chosen persona guidelines.
        """
        persona_instructions = cls.PERSONAS.get(persona, cls.PERSONAS["Data Analyst"])
        
        system_prompt = f"""{persona_instructions}

CORE PRINCIPLES:
1. Answer ONLY using the supplied grounded context.
2. Never invent or fabricate information. 
3. If the context does not contain the answer, explicitly state that the information is unavailable.
4. Do NOT reference any external or web sources.
5. All calculations and recommendations must rest strictly on the context metrics provided.
"""
        return system_prompt

    @staticmethod
    def build_user_prompt(question: str, context_str: str) -> str:
        """
        Formats user prompts containing query and retrieval contexts.
        """
        user_prompt = f"""Grounded Context Facts:
---------------------
{context_str}
---------------------

User Question:
{question}

Formulate your response:
"""
        return user_prompt
