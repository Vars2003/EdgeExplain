from typing import Dict, Any
from utils.logger import get_logger
from ai.llm import LocalLLMConnector

logger = get_logger("ai.explainer")

class LocalDatasetExplainer:
    """
    Leverages local inference engines to summarize dataset observations, quality metrics,
    and preprocessing paths into structured reports.
    """

    def __init__(self, llm_connector: LocalLLMConnector):
        self.llm = llm_connector
        logger.info("Local dataset explainer helper initialized.")

    def compile_markdown_report(self, dataset_name: str, metrics: Dict[str, Any], recommendations: Dict[str, Any]) -> str:
        """
        Creates a prompt containing dataset diagnostics and sends it to the LLM
        to produce a detailed descriptive markdown report.
        """
        logger.info(f"Preparing narrative report prompt for dataset: {dataset_name}")
        
        prompt = (
            f"Generate a professional, detailed markdown report analyzing the dataset '{dataset_name}'.\n"
            f"Include an Executive Summary, Data Quality Review, and Preprocessing Recommendations.\n\n"
            f"Data Profile Summary:\n{str(metrics)}\n\n"
            f"Recommendations:\n{str(recommendations)}"
        )
        
        if not self.llm.is_loaded:
            # Return a default formatted markdown report if the LLM is not active
            return self._generate_static_fallback_report(dataset_name, metrics, recommendations)
            
        return self.llm.generate_response(prompt)

    def _generate_static_fallback_report(self, name: str, metrics: Dict[str, Any], recommendations: Dict[str, Any]) -> str:
        """
        Returns a beautifully formatted static report based directly on engine output.
        """
        return f"""# Data Scientist Report: {name}

## 1. Executive Summary
- **Dataset Name**: {name}
- **Observations**: The Analytics engine has successfully completed local diagnostics.
- **Overall Data Quality Score**: {metrics.get('quality_score', 'N/A')}%
- **ML Readiness Index**: {metrics.get('ml_readiness', {}).get('score', 'N/A')}%

## 2. Structural Findings
- Total Observations: {metrics.get('basic_metrics', {}).get('rows', 0)}
- Dimension Columns: {metrics.get('basic_metrics', {}).get('columns', 0)}
- Duplicate Count: {metrics.get('basic_metrics', {}).get('duplicate_rows', 0)}

## 3. Preprocessing Roadmap
This roadmap was compiled by EdgeExplain's rules-based recommendation systems:
- *Please load an offline GGUF LLM model into `ai/llm.py` to enable customized conversational summaries of these files.*
"""
