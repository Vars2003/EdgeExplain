import json
import os
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("intelligence.exporter")

class IntelligenceExporter:
    """
    Saves compiled Memory Object JSON payloads and Markdown briefings
    locally to the reports folder.
    """

    @staticmethod
    def export_memory_json(memory_obj: Dict[str, Any], output_path: str) -> bool:
        """
        Saves the memory object as a raw JSON file.
        """
        try:
            logger.info(f"Exporting Memory JSON to: {output_path}")
            # Ensure folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(memory_obj, f, indent=2)
            logger.info("Memory JSON exported successfully.")
            return True
        except Exception as e:
            logger.exception(f"Failed to export memory JSON: {e}")
            return False

    @staticmethod
    def export_markdown_briefing(memory_obj: Dict[str, Any], output_path: str) -> bool:
        """
        Saves all generated summarizer markdown cards as a single text file.
        """
        try:
            logger.info(f"Exporting Markdown Briefing to: {output_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            metadata = memory_obj["metadata"]
            context = memory_obj["context"]
            summaries = memory_obj["summaries"]
            
            briefing_text = f"""# EdgeExplain AI Briefing: {context['dataset_name']}

- **Report Version**: {metadata['version']}
- **Report Date**: {metadata['created_at']}
- **Dataset ID**: {metadata['dataset_id']}
- **Application Engine Version**: {metadata['application_version']}

---

{summaries['executive']}

---

{summaries['technical']}

---

{summaries['ml']}

---

{summaries['quality']}

---
Report compiled locally by EdgeExplain Offline AI.
"""
            with open(output_report_path := output_path, "w", encoding="utf-8") as f:
                f.write(briefing_text)
            logger.info("Markdown Briefing exported successfully.")
            return True
        except Exception as e:
            logger.exception(f"Failed to export Markdown briefing: {e}")
            return False
