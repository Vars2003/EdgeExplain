import os
import pandas as pd
from typing import Dict, Any
from plugins import BasePlugin
from reports.pdf_generator import ReportGenerator
from utils.logger import get_logger

logger = get_logger("plugins.report")

class ProfessionalReportPlugin(BasePlugin):
    """
    Plugin coordinating executive, technical, and AI intelligence reports.
    """

    def __init__(self):
        super().__init__("Professional Report Compiler", ["pandas", "reportlab"])

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        filename = memory_obj.get("context", {}).get("dataset_name", "dataset")
        clean_name = filename.split(".")[0]
        
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Compile paths
        exec_pdf = os.path.join(reports_dir, f"exec_report_{clean_name}.pdf")
        tech_pdf = os.path.join(reports_dir, f"tech_report_{clean_name}.pdf")
        ai_pdf = os.path.join(reports_dir, f"ai_report_{clean_name}.pdf")
        
        exec_html = os.path.join(reports_dir, f"exec_report_{clean_name}.html")
        tech_html = os.path.join(reports_dir, f"tech_report_{clean_name}.html")
        ai_html = os.path.join(reports_dir, f"ai_report_{clean_name}.html")
        
        # Generate PDFs (or Fallback text files)
        s_epdf = ReportGenerator.generate_pdf_report(memory_obj, "executive", exec_pdf)
        s_tpdf = ReportGenerator.generate_pdf_report(memory_obj, "technical", tech_pdf)
        s_apdf = ReportGenerator.generate_pdf_report(memory_obj, "ai", ai_pdf)
        
        # Generate HTMLs
        s_ehtml = ReportGenerator.generate_html_report(memory_obj, "executive")
        with open(exec_html, "w", encoding="utf-8") as f:
            f.write(s_ehtml)
            
        s_thtml = ReportGenerator.generate_html_report(memory_obj, "technical")
        with open(tech_html, "w", encoding="utf-8") as f:
            f.write(s_thtml)
            
        s_ahtml = ReportGenerator.generate_html_report(memory_obj, "ai")
        with open(ai_html, "w", encoding="utf-8") as f:
            f.write(s_ahtml)
            
        result = {
            "executive_pdf_path": exec_pdf,
            "technical_pdf_path": tech_pdf,
            "ai_pdf_path": ai_pdf,
            "executive_html_path": exec_html,
            "technical_html_path": tech_html,
            "ai_html_path": ai_html
        }
        
        return {
            "result": result,
            "confidence": 100,
            "evidence": [f"Exported PDF & HTML report packages for executive, technical, and AI categories in reports/ directory."]
        }
