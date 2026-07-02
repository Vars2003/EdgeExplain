"""
Structured prompt templates for offline local LLM tabular reasoning.
These templates frame data diagnostics for the local inference model.
"""

SYSTEM_PROMPT = (
    "You are EdgeExplain's Offline AI Data Scientist. Your goal is to help users "
    "understand their datasets, diagnose issues, and select proper machine learning pipelines. "
    "Keep responses professional, grounded, and concise. Never make up facts."
)

EXPLAIN_PREPROCESSING_TEMPLATE = """
Context:
The user has a tabular dataset with the following characteristics:
- Column Name: {column}
- Semantic Type: {semantic_type}
- Statistics: {statistics}
- Recommended Preprocessing: {preprocessing_type}
- Suggested Fix: {suggested_fix}
- Reasoning: {reasoning}

Question:
Why should I perform {preprocessing_type} on the column '{column}'?

Instruction:
Explain the purpose of this preprocessing step in simple terms. Mention the statistical reasons (such as skewness, outliers, scale differences) based on the context above. Explain the mathematical formula if relevant.
"""

ALGORITHM_SUITABILITY_TEMPLATE = """
Context:
The dataset is classified as a {dataset_type} task.
- Shape: {rows} rows, {cols} columns
- Missing cells: {missing_pct}% of total cells
- Categorical features: {cat_count} columns
- Flagged outliers: {outlier_count} instances
- Top recommended algorithms: {top_algos}

Question:
Which machine learning algorithms are most suitable for this dataset, and why?

Instruction:
Provide a clear comparison of the suggested algorithms based on the dataset details. Mention pros and cons of tree-based vs. linear models for this specific size and feature combination.
"""
