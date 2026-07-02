from typing import Dict, Any, List
import pandas as pd
from utils.logger import get_logger

logger = get_logger("intelligence.summarizer")

class DatasetSummarizer:
    """
    Renders natural-language Markdown summaries (Executive, Technical, ML, Quality)
    using deterministic formatting templates grounded in dataset context.
    """

    @staticmethod
    def generate_executive_summary(ctx: Dict[str, Any]) -> str:
        return f"""### 📈 Executive Summary

The dataset **'{ctx['dataset_name']}'** has been successfully parsed and evaluated locally. 
This is a **{ctx['dataset_type']}** dataset estimated to originate from the **{ctx['domain']}** domain. 

- **Data Integrity Health**: The overall quality score is **{ctx['quality_overview']['quality_score']:.1f}%**.
- **ML Suitability**: The dataset registers an ML Readiness index of **{ctx['quality_overview']['ml_readiness_score']:.1f}%**.
- **Core Recommendation**: Based on the targets, we advise establishing a **{ctx['recommended_ml_task']}** pipeline.
"""

    @staticmethod
    def generate_technical_summary(ctx: Dict[str, Any]) -> str:
        return f"""### ⚙️ Technical Summary

A deep structural profile reveals the following database dimensions:
- **Observation Count**: `{ctx['row_count']:,}` rows.
- **Feature Dimension**: `{ctx['column_count']}` columns.
- **In-Memory Size**: `{ctx['memory_readable']}`.
- **Cardinality Profiles**: The schema is mapped into semantic classes. 
  You can inspect specific column categories and unique ratios in the *Schema Mapping* tab.
"""

    @staticmethod
    def generate_ml_summary(ctx: Dict[str, Any], algo_recs: List[Dict[str, Any]]) -> str:
        top_algos = [a["algorithm"] for a in algo_recs[:2]]
        algo_bullets = "\n".join([f"- **{a['algorithm']}** (Compatibility: {a['compatibility_score']}%) — *{a['reasoning']}*" for a in algo_recs[:3]])
        
        target_clause = f"Candidate target columns identified: `{', '.join(ctx['candidate_targets'])}`." if ctx['candidate_targets'] else "No standard target candidate column was auto-inferred (unsupervised analysis suggested)."
        
        return f"""### 🤖 Machine Learning Modeling Summary

- **Proposed Pipeline Task**: `{ctx['recommended_ml_task']}`
- **Target Configurations**: {target_clause}
- **Top Algorithm Options**:
{algo_bullets}
"""

    @staticmethod
    def generate_data_quality_summary(ctx: Dict[str, Any]) -> str:
        risks = ctx.get("key_risks", [])
        if risks:
            risk_bullets = "\n".join([f"- **[{r['severity']} Risk] {r['risk']}**: *{r['detail']}*" for r in risks])
        else:
            risk_bullets = "- *No major data health vulnerabilities or quality concerns detected.*"
            
        return f"""### 🔬 Data Quality Diagnostics Summary

- **Missing Cells**: `{ctx['quality_overview']['missing_cells_pct']:.2f}%` of cells are null.
- **Row Duplication**: `{ctx['quality_overview']['duplicate_rows_count']:,}` duplicate rows detected.
- **Identified Structural Risks**:
{risk_bullets}
"""

    @classmethod
    def compile_all_summaries(cls, ctx: Dict[str, Any], algo_recs: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Assembles all four summary briefings.
        """
        logger.info("Compiling all summaries.")
        return {
            "executive": cls.generate_executive_summary(ctx),
            "technical": cls.generate_technical_summary(ctx),
            "ml": cls.generate_ml_summary(ctx, algo_recs),
            "quality": cls.generate_data_quality_summary(ctx)
        }
