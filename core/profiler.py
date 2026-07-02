import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from config import settings
from utils.logger import get_logger
from utils.helpers import format_bytes

logger = get_logger("core.profiler")

class DatasetProfiler:
    """
    Core profiling engine that evaluates data quality, calculates summary statistics,
    and checks machine learning readiness without cloud APIs.
    """

    @staticmethod
    def get_basic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gathers fundamental dimensions and memory footprint of the dataset.
        """
        rows, cols = df.shape
        mem_usage = df.memory_usage(deep=True).sum()
        
        # Identify constant columns (only 1 unique value)
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        
        # Calculate duplicate rows
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = (duplicate_count / rows) * 100 if rows > 0 else 0.0
        
        return {
            "rows": rows,
            "columns": cols,
            "memory_bytes": mem_usage,
            "memory_readable": format_bytes(mem_usage),
            "constant_columns": constant_cols,
            "constant_columns_count": len(constant_cols),
            "duplicate_rows": duplicate_count,
            "duplicate_pct": duplicate_pct
        }

    @staticmethod
    def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes missing count and ratio per column.
        """
        rows = len(df)
        missing_counts = df.isna().sum().to_dict()
        missing_pcts = {col: (count / rows * 100 if rows > 0 else 0.0) for col, count in missing_counts.items()}
        
        total_cells = df.size
        total_missing = sum(missing_counts.values())
        completeness_pct = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0.0
        
        return {
            "columns": missing_counts,
            "percentages": missing_pcts,
            "total_missing": total_missing,
            "completeness_pct": completeness_pct
        }

    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identifies outliers in numeric columns using the Tukey Interquartile Range (IQR) method.
        """
        outlier_counts = {}
        outlier_indices = {}
        total_outliers = 0
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) < 4:
                    outlier_counts[col] = 0
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_counts[col] = len(outliers)
                total_outliers += len(outliers)
                
        return {
            "columns": outlier_counts,
            "total_outliers": total_outliers
        }

    @classmethod
    def calculate_data_quality_score(cls, df: pd.DataFrame, basic_metrics: Dict[str, Any], missing_data: Dict[str, Any], outlier_data: Dict[str, Any]) -> float:
        """
        Derives an overall percentage score indicating the clean health of the dataset.
        """
        rows, cols = df.shape
        if rows == 0 or cols == 0:
            return 0.0
            
        score = 100.0
        
        # Deduct for missing values (linear penalty based on missing cell percentage)
        completeness = missing_data["completeness_pct"]
        score -= (100.0 - completeness)
        
        # Deduct for duplicates (up to 10 points max)
        dup_pct = basic_metrics["duplicate_pct"]
        score -= min(10.0, dup_pct)
        
        # Deduct for constant columns (5 points per constant column, up to 20 points max)
        const_count = basic_metrics["constant_columns_count"]
        score -= min(20.0, const_count * 5.0)
        
        # Deduct for outliers (ratio of outliers to total cells, up to 10 points max)
        total_cells = df.size
        outlier_count = outlier_data["total_outliers"]
        outlier_ratio = (outlier_count / total_cells) * 100 if total_cells > 0 else 0
        score -= min(10.0, outlier_ratio * 2.0)
        
        return max(0.0, min(100.0, score))

    @staticmethod
    def calculate_ml_readiness_score(df: pd.DataFrame, missing_data: Dict[str, Any], basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates dataset compliance with ML training rules to yield a readiness index.
        """
        rows, cols = df.shape
        if rows == 0 or cols == 0:
            return {"score": 0.0, "checks": {}}
            
        checks = {}
        scores = []
        
        # 1. Missing Values check
        # Ideally, no columns should have > 20% missing values
        missing_pcts = missing_data["percentages"]
        columns_with_high_missing = [col for col, pct in missing_pcts.items() if pct > 20.0]
        missing_score = max(0.0, 100.0 - (len(columns_with_high_missing) / cols * 100))
        checks["missing_values"] = {
            "status": "PASS" if len(columns_with_high_missing) == 0 else "WARNING",
            "score": missing_score,
            "detail": f"{len(columns_with_high_missing)} columns have >20% missing values."
        }
        scores.append(missing_score)
        
        # 2. Duplicate rows check
        dup_pct = basic_metrics["duplicate_pct"]
        dup_score = max(0.0, 100.0 - dup_pct)
        checks["duplicates"] = {
            "status": "PASS" if dup_pct < 5.0 else "WARNING",
            "score": dup_score,
            "detail": f"{dup_pct:.2f}% duplicate rows detected."
        }
        scores.append(dup_score)
        
        # 3. Categorical encoding check
        # Flag if encoding is required for object dtypes
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        encoding_score = 100.0
        if len(cat_cols) > 0:
            encoding_score = 60.0  # Needs encoding
        checks["encoding"] = {
            "status": "PASS" if len(cat_cols) == 0 else "INFO",
            "score": encoding_score,
            "detail": f"{len(cat_cols)} categorical columns need encoding."
        }
        scores.append(encoding_score)
        
        # 4. Feature scaling check
        # Flag if numeric scales differ by more than an order of magnitude
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaling_score = 100.0
        if len(numeric_cols) > 1:
            stds = [df[col].std() for col in numeric_cols if df[col].std() > 0]
            if len(stds) > 1:
                ratio = max(stds) / min(stds)
                if ratio > 10.0:
                    scaling_score = 70.0
        checks["scaling"] = {
            "status": "PASS" if scaling_score == 100.0 else "INFO",
            "score": scaling_score,
            "detail": "Scale differences exceed 10x. Standardization recommended." if scaling_score < 100.0 else "Scale ranges are comparable."
        }
        scores.append(scaling_score)
        
        # 5. Target column check
        # Try to infer if a target exists (a column that is not an identifier, named target, label, class, etc.)
        target_candidates = [col for col in df.columns if any(t in col.lower() for t in ["target", "label", "class", "clicked", "bought", "price", "churn", "fraud", "output"])]
        target_score = 100.0 if len(target_candidates) > 0 else 50.0
        checks["target_availability"] = {
            "status": "PASS" if len(target_candidates) > 0 else "WARNING",
            "score": target_score,
            "detail": f"Target column candidate inferred: {', '.join(target_candidates)}" if len(target_candidates) > 0 else "No standard target keyword (target, class, price, churn) found."
        }
        scores.append(target_score)
        
        # 6. Feature Redundancy (High Correlation)
        redundant_count = 0
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            redundant_count = len([col for col in upper_tri.columns if any(upper_tri[col] > 0.90)])
        redundancy_score = max(0.0, 100.0 - (redundant_count / max(1, len(numeric_cols)) * 100))
        checks["redundancy"] = {
            "status": "PASS" if redundant_count == 0 else "WARNING",
            "score": redundancy_score,
            "detail": f"{redundant_count} highly correlated features (>0.90) flagged."
        }
        scores.append(redundancy_score)
        
        overall_score = float(np.mean(scores))
        
        return {
            "score": overall_score,
            "checks": checks
        }

    @staticmethod
    def generate_ydata_report(df: pd.DataFrame, output_path: str) -> bool:
        """
        Generates a ydata-profiling HTML report on demand. Safe from ImportError.
        """
        try:
            from ydata_profiling import ProfileReport
            logger.info("Initializing optional ydata-profiling ProfileReport.")
            profile = ProfileReport(df, title="EdgeExplain Dataset Report", explorative=True, dark_mode=True)
            profile.to_file(output_path)
            logger.info(f"YData Report successfully generated at {output_path}")
            return True
        except ImportError:
            logger.warning("ydata-profiling not installed. Reverting to custom profiling engine.")
            return False
        except Exception as e:
            logger.exception(f"Failed to generate ydata-profiling report: {e}")
            return False
