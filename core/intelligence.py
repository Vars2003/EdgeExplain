import numpy as np
import pandas as pd
import re
from typing import Dict, Any, List, Tuple
from config import settings
from utils.logger import get_logger

logger = get_logger("core.intelligence")

class DataIntelligenceEngine:
    """
    Infers semantic metadata, dataset classifications, feature roles, 
    and source domains completely offline using rule heuristics.
    """

    @staticmethod
    def detect_feature_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Classifies each column into a semantic category with evidence and confidence.
        """
        logger.info("Running semantic feature type detection.")
        feature_types = {}
        rows = len(df)
        
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            cardinality = non_null.nunique()
            unique_ratio = (cardinality / rows) if rows > 0 else 0.0
            col_lower = col.lower()
            
            # Default values
            assigned_type = "nominal"
            confidence = 50
            evidence = [f"Data type: {series.dtype}", f"Cardinality: {cardinality} ({unique_ratio * 100:.1f}%)"]
            explanation = ""
            
            # Helper flags
            is_numeric = pd.api.types.is_numeric_dtype(series)
            
            # 1. Check for DateTime
            is_date_like_name = any(word in col_lower for word in ["date", "time", "timestamp", "year", "month", "day", "epoch"])
            is_date_dtype = pd.api.types.is_datetime64_any_dtype(series)
            
            can_parse_date = False
            if not is_date_dtype and not is_numeric and len(non_null) > 0:
                try:
                    # Test parse on a sample of 10 rows
                    sample = non_null.head(10).astype(str)
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() / len(sample) > 0.8:
                        can_parse_date = True
                except Exception:
                    pass

            if is_date_dtype:
                assigned_type = "datetime"
                confidence = 99
                evidence.append("Pandas natively parsed column as datetime")
                explanation = "Recognized as a datetime sequence by the compiler."
            elif can_parse_date:
                assigned_type = "datetime"
                confidence = 90
                evidence.append("Successfully parsed a sample of entries as timestamps")
                explanation = "String column elements conform to standard ISO/date formats."
            elif is_date_like_name and not is_numeric:
                assigned_type = "datetime"
                confidence = 70
                evidence.append(f"Header contains time keyword: '{col}'")
                explanation = "Inferred as date/time marker based on column heading."
                
            # 2. Check for Identifier
            elif (unique_ratio > 0.98 or (unique_ratio == 1.0 and cardinality > 10)) and any(word in col_lower for word in ["id", "key", "uuid", "code", "num", "pk"]):
                assigned_type = "identifier"
                confidence = 99 if unique_ratio == 1.0 else 85
                evidence.append(f"Unique ratio: {unique_ratio:.2f}")
                evidence.append(f"Identifier keyword in header: '{col}'")
                explanation = "Unique primary keys or record tags which carry no training significance."
                
            # 3. Check for Boolean
            elif pd.api.types.is_bool_dtype(series) or (cardinality == 2 and set(non_null.unique()).issubset({0, 1, True, False, "0", "1", "True", "False", "true", "false"})):
                assigned_type = "boolean"
                confidence = 99
                evidence.append(f"Unique values: {list(non_null.unique())}")
                explanation = "Two-state logical boolean flag."
                
            # 4. Check for Binary (exactly two unique states)
            elif cardinality == 2:
                assigned_type = "binary"
                confidence = 98
                evidence.append(f"Unique outcomes: {list(non_null.unique())}")
                explanation = "Variable contains exactly two categorical states."
                
            # 5. Check for Geographic Coordinates
            elif is_numeric and any(word in col_lower for word in ["latitude", "longitude", "lat", "lon", "gps", "coord"]):
                assigned_type = "geographic_coordinate"
                confidence = 95
                evidence.append("Column is numeric")
                evidence.append(f"Header contains spatial keyword: '{col}'")
                explanation = "Represents spatial coordinate mappings (Lat/Lon)."
                
            # 6. Check for Numerical columns
            elif is_numeric:
                # Continuous vs Discrete
                # If cardinality is low, it behaves as discrete/categorical
                if cardinality <= 15:
                    assigned_type = "discrete_numerical"
                    confidence = 85
                    evidence.append("Column is numeric")
                    explanation = "Discrete numeric count representation with a small, bounded range."
                else:
                    assigned_type = "continuous_numerical"
                    confidence = 95
                    evidence.append("Column is numeric")
                    explanation = "Continuous real-valued measurements."
                    
            # 7. Check for Text (high-cardinality strings)
            elif unique_ratio > 0.30 and len(non_null) > 0 and non_null.astype(str).str.len().mean() > 50:
                assigned_type = "text"
                confidence = 90
                avg_len = non_null.astype(str).str.len().mean()
                evidence.append(f"Average string length: {avg_len:.1f} characters")
                explanation = "Unstructured free text paragraphs suitable for NLP."
                
            # 8. Check for Ordinal Categorical
            elif any(word in col_lower for word in ["grade", "rating", "rank", "level", "stage"]):
                assigned_type = "ordinal"
                confidence = 80
                evidence.append(f"Ordinal keyword in header: '{col}'")
                explanation = "Ordered categories indicating sequential level or ranking."
                
            # 9. Fallback Nominal Categorical
            else:
                assigned_type = "nominal"
                confidence = 75
                explanation = "Standard categorical variable with no inherent ordering."
                
            feature_types[col] = {
                "type": assigned_type,
                "confidence": confidence,
                "evidence": evidence,
                "explanation": explanation
            }
            
        return feature_types

    @staticmethod
    def detect_dataset_type(df: pd.DataFrame, feature_types: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determines the dataset's primary operational focus (Classification, Regression, Time Series, etc.).
        """
        rows, cols = df.shape
        if rows == 0 or cols == 0:
            return {
                "type": "Empty Dataset",
                "confidence": 100,
                "evidence": ["Rows: 0", "Columns: 0"],
                "explanation": "No data present."
            }
            
        # Count types
        counts = {}
        for col, info in feature_types.items():
            t = info["type"]
            counts[t] = counts.get(t, 0) + 1
            
        has_time = counts.get("datetime", 0) > 0
        has_geo = counts.get("geographic_coordinate", 0) >= 2
        has_text = counts.get("text", 0) > (cols * 0.4)
        
        # Look for target column candidates
        target_col = None
        for col in df.columns:
            if any(t in col.lower() for t in ["target", "label", "class", "clicked", "price", "churn", "fraud"]):
                target_col = col
                break
                
        # Heuristics
        if has_time and (counts.get("continuous_numerical", 0) > 0 or counts.get("discrete_numerical", 0) > 0):
            return {
                "type": "Time Series",
                "confidence": 95 if df.index.name == "Timestamp" or any("time" in str(col).lower() for col in df.columns) else 80,
                "evidence": [f"DateTime columns found: {counts.get('datetime', 0)}"],
                "explanation": "Dataset represents continuous observations mapped chronologically."
            }
        elif has_geo:
            return {
                "type": "Geospatial Dataset",
                "confidence": 90,
                "evidence": [f"Geographic Coordinate columns found: {counts.get('geographic_coordinate', 0)}"],
                "explanation": "Primary focus involves coordinate points and spatial mapping."
            }
        elif has_text:
            return {
                "type": "Text Dataset",
                "confidence": 85,
                "evidence": [f"High-cardinality text columns: {counts.get('text', 0)}"],
                "explanation": "Contains mainly natural language text, requiring NLP pipelines."
            }
        elif target_col:
            target_type = feature_types[target_col]["type"]
            if target_type in ["nominal", "binary", "boolean", "discrete_numerical"]:
                return {
                    "type": "Classification",
                    "confidence": 90,
                    "evidence": [f"Target candidate found: '{target_col}'", f"Target feature class: {target_type}"],
                    "explanation": "Configured for predicting categorical labels or discrete classes."
                }
            elif target_type == "continuous_numerical":
                return {
                    "type": "Regression",
                    "confidence": 90,
                    "evidence": [f"Target candidate found: '{target_col}'", f"Target feature class: continuous numeric"],
                    "explanation": "Configured for forecasting continuous real-valued numeric targets."
                }
                
        # Default mixed tabular
        return {
            "type": "Mixed Dataset",
            "confidence": 75,
            "evidence": [f"Numerical features: {counts.get('continuous_numerical', 0) + counts.get('discrete_numerical', 0)}", f"Categorical features: {counts.get('nominal', 0)}"],
            "explanation": "General multi-type tabular data with mixed numeric and category variables."
        }

    @staticmethod
    def infer_source_domain(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimates the business or scientific industry domain of the dataset.
        """
        logger.info("Running dataset source domain inference.")
        columns = [str(col).lower() for col in df.columns]
        
        domain_scores = {}
        domain_matches = {}
        
        # Check vocabularies
        for domain, vocab in settings.DOMAIN_VOCABULARIES.items():
            matches = []
            for col in columns:
                # Check direct match or substring match
                for word in vocab:
                    if re.search(r'\b' + re.escape(word) + r'\b', col) or word in col:
                        matches.append(col)
                        break # Move to next column
            if matches:
                domain_matches[domain] = list(set(matches))
                domain_scores[domain] = len(matches)
                
        if not domain_scores:
            return {
                "domain": "Generic Tabular",
                "confidence": 50,
                "evidence": ["No columns matched domain-specific keywords"],
                "explanation": "Unable to map columns to specialized domains. Treated as a generic database."
            }
            
        # Select highest domain
        best_domain = max(domain_scores, key=domain_scores.get)
        raw_score = domain_scores[best_domain]
        total_score = sum(domain_scores.values())
        
        # Confidence calculation
        confidence = min(99, int((raw_score / total_score * 80) + min(20, raw_score * 5)))
        
        # Prepare evidence
        matched_columns = domain_matches[best_domain]
        evidence = matched_columns
        
        explanation = f"Medical terminology strongly matches healthcare vocabulary." if best_domain == "Healthcare" else f"Database headers match vocabulary for {best_domain} applications."
        
        return {
            "domain": f"{best_domain} Dataset",
            "confidence": confidence,
            "evidence": evidence,
            "explanation": explanation
        }
