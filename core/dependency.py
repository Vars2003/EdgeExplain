import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Tuple
from utils.logger import get_logger

logger = get_logger("core.dependency")

def get_clean_pairs(s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Returns aligned, non-null pairs from two pandas Series.
    """
    mask = s1.notna() & s2.notna()
    return s1[mask], s2[mask]

def calculate_pearson(s1: pd.Series, s2: pd.Series) -> float:
    c1, c2 = get_clean_pairs(s1, s2)
    if len(c1) < 2:
        return 0.0
    r, _ = stats.pearsonr(c1, c2)
    return float(r) if np.isfinite(r) else 0.0

def calculate_spearman(s1: pd.Series, s2: pd.Series) -> float:
    c1, c2 = get_clean_pairs(s1, s2)
    if len(c1) < 2:
        return 0.0
    r, _ = stats.spearmanr(c1, c2)
    return float(r) if np.isfinite(r) else 0.0

def calculate_chi_square_cramers_v(s1: pd.Series, s2: pd.Series) -> Tuple[float, float]:
    """
    Calculates Chi-Square statistic and Cramer's V for categorical features.
    """
    c1, c2 = get_clean_pairs(s1, s2)
    if len(c1) < 2:
        return 0.0, 0.0
        
    contingency_table = pd.crosstab(c1, c2)
    if contingency_table.size == 0 or contingency_table.shape[0] <= 1 or contingency_table.shape[1] <= 1:
        return 0.0, 0.0
        
    try:
        chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
        n = len(c1)
        r, c = contingency_table.shape
        min_dim = min(r - 1, c - 1)
        
        if min_dim == 0:
            cramers_v = 0.0
        else:
            cramers_v = np.sqrt((chi2 / n) / min_dim)
            
        return float(chi2), float(cramers_v) if np.isfinite(cramers_v) else 0.0
    except Exception as e:
        logger.debug(f"Chi-square calculation error: {e}")
        return 0.0, 0.0

def calculate_anova(numerical_col: pd.Series, categorical_col: pd.Series) -> Tuple[float, float]:
    """
    Performs One-Way ANOVA. Returns F-statistic and p-value.
    """
    num_c, cat_c = get_clean_pairs(numerical_col, categorical_col)
    if len(num_c) < 3:
        return 0.0, 1.0
        
    groups = []
    unique_categories = cat_c.unique()
    
    if len(unique_categories) <= 1:
        return 0.0, 1.0
        
    for cat in unique_categories:
        group_data = num_c[cat_c == cat].to_numpy()
        if len(group_data) > 0:
            groups.append(group_data)
            
    if len(groups) <= 1:
        return 0.0, 1.0
        
    try:
        f_stat, p_val = stats.f_oneway(*groups)
        return (float(f_stat) if np.isfinite(f_stat) else 0.0), (float(p_val) if np.isfinite(p_val) else 1.0)
    except Exception as e:
        logger.debug(f"ANOVA calculation error: {e}")
        return 0.0, 1.0

def calculate_mutual_information(s1: pd.Series, s2: pd.Series, is_target_numeric: bool) -> float:
    """
    Estimates mutual information. Uses label encoding for strings.
    """
    c1, c2 = get_clean_pairs(s1, s2)
    if len(c1) < 5:
        return 0.0
        
    # Standardize to numeric representation
    x = c1.to_numpy().reshape(-1, 1)
    if not pd.api.types.is_numeric_dtype(c1):
        x = LabelEncoder().fit_transform(c1.astype(str)).reshape(-1, 1)
        
    y = c2.to_numpy()
    if not pd.api.types.is_numeric_dtype(c2):
        y = LabelEncoder().fit_transform(c2.astype(str))
        
    try:
        if is_target_numeric:
            mi = mutual_info_regression(x, y, random_state=42)[0]
        else:
            mi = mutual_info_classif(x, y, random_state=42)[0]
        return float(mi) if np.isfinite(mi) else 0.0
    except Exception as e:
        logger.debug(f"Mutual Information error: {e}")
        return 0.0

def compute_dependency_matrix(df: pd.DataFrame, semantic_types: Dict[str, str]) -> Dict[str, Any]:
    """
    Computes a generalized dependency matrix containing scores scaled between 0 and 1.
    """
    logger.info("Computing generalized feature dependency matrix.")
    columns = df.columns.tolist()
    n = len(columns)
    matrix = np.zeros((n, n))
    methods = {}
    
    for i in range(n):
        col_i = columns[i]
        type_i = semantic_types.get(col_i, "nominal")
        
        for j in range(i, n):
            col_j = columns[j]
            type_j = semantic_types.get(col_j, "nominal")
            
            if i == j:
                matrix[i, j] = 1.0
                methods[f"{col_i} <-> {col_j}"] = "Self-correlation"
                continue
                
            # Determine relationship type
            is_i_num = pd.api.types.is_numeric_dtype(df[col_i])
            is_j_num = pd.api.types.is_numeric_dtype(df[col_j])
            
            val = 0.0
            method = "None"
            
            if is_i_num and is_j_num:
                # Numerical-Numerical -> Pearson
                val = abs(calculate_pearson(df[col_i], df[col_j]))
                method = "Pearson Correlation (linear)"
            elif not is_i_num and not is_j_num:
                # Categorical-Categorical -> Cramer's V
                _, val = calculate_chi_square_cramers_v(df[col_i], df[col_j])
                method = "Cramer's V (association)"
            else:
                # Numeric-Categorical -> ANOVA
                num_col = col_i if is_i_num else col_j
                cat_col = col_j if is_i_num else col_i
                
                # ANOVA F-stat can be large; we map the p-value or mutual info.
                # Let's use Mutual Information for mixed-type dependencies since it is normalized.
                val = calculate_mutual_information(df[num_col], df[cat_col], is_target_numeric=True)
                # Cap and scale MI score to 0-1 range for dependency matrix
                val = min(1.0, val / 2.0)  
                method = "Mutual Information (non-linear mixed)"
                
            matrix[i, j] = val
            matrix[j, i] = val
            methods[f"{col_i} <-> {col_j}"] = method
            methods[f"{col_j} <-> {col_i}"] = method
            
    return {
        "columns": columns,
        "matrix": matrix.tolist(),
        "methods": methods
    }

def generate_correlation_network(df: pd.DataFrame, semantic_types: Dict[str, str], threshold: float = 0.3) -> Dict[str, Any]:
    """
    Generates nodes and edges for a correlation network diagram.
    """
    dep_data = compute_dependency_matrix(df, semantic_types)
    cols = dep_data["columns"]
    matrix = np.array(dep_data["matrix"])
    methods = dep_data["methods"]
    
    nodes = []
    edges = []
    
    # Add nodes
    for i, col in enumerate(cols):
        nodes.append({
            "id": col,
            "label": col,
            "type": semantic_types.get(col, "unknown")
        })
        
    # Add edges above threshold
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            weight = matrix[i, j]
            if weight >= threshold:
                edges.append({
                    "source": cols[i],
                    "target": cols[j],
                    "weight": float(weight),
                    "method": methods.get(f"{cols[i]} <-> {cols[j]}", "Association")
                })
                
    return {
        "nodes": nodes,
        "edges": edges
    }
