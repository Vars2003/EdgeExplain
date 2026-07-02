import pandas as pd
import numpy as np
from typing import List, Union, Any, Dict
from utils.logger import get_logger

logger = get_logger("core.cleaner")

class DataCleaner:
    """
    Routines for dataset cleaning, duplicate removal, imputations, and basic encodings.
    """

    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df_cleaned = df.drop_duplicates()
        after = len(df_cleaned)
        logger.info(f"Dropped duplicates. Rows reduced from {before} to {after}.")
        return df_cleaned

    @staticmethod
    def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            df_cleaned = df.drop(columns=constant_cols)
            logger.info(f"Dropped constant columns: {constant_cols}")
            return df_cleaned
        return df

    @staticmethod
    def impute_missing(df: pd.DataFrame, column: str, strategy: str = "mean", fill_value: Any = None) -> pd.DataFrame:
        """
        Fills missing values in a specified column using a designated strategy.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
            
        df_cleaned = df.copy()
        
        # If column has no missing values, do nothing
        if not df_cleaned[column].isna().any():
            return df_cleaned
            
        logger.info(f"Imputing missing values in '{column}' using '{strategy}' strategy.")
        
        if strategy == "mean":
            val = df_cleaned[column].mean()
            df_cleaned[column] = df_cleaned[column].fillna(val)
        elif strategy == "median":
            val = df_cleaned[column].median()
            df_cleaned[column] = df_cleaned[column].fillna(val)
        elif strategy == "mode":
            mode_series = df_cleaned[column].mode()
            val = mode_series.iloc[0] if not mode_series.empty else None
            if val is not None:
                df_cleaned[column] = df_cleaned[column].fillna(val)
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value must be provided for 'constant' strategy.")
            df_cleaned[column] = df_cleaned[column].fillna(fill_value)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
            
        return df_cleaned

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
        """
        Filters out rows where the specified column value lies beyond Tukey's IQR boundaries.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
            
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Outlier removal only applies to numeric columns. '{column}' is not numeric.")
            
        df_cleaned = df.copy()
        series = df_cleaned[column].dropna()
        if len(series) < 4:
            return df_cleaned
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        before = len(df_cleaned)
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
        after = len(df_cleaned)
        
        logger.info(f"Removed {before - after} outliers from '{column}'. Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")
        return df_cleaned

    @staticmethod
    def encode_categorical(df: pd.DataFrame, column: str, strategy: str = "onehot") -> pd.DataFrame:
        """
        Performs one-hot or ordinal encoding on a categorical column.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found.")
            
        df_cleaned = df.copy()
        logger.info(f"Encoding categorical column '{column}' using '{strategy}'.")
        
        if strategy == "onehot":
            encoded = pd.get_dummies(df_cleaned[column], prefix=column, drop_first=True, dtype=int)
            df_cleaned = pd.concat([df_cleaned.drop(columns=[column]), encoded], axis=1)
        elif strategy == "ordinal":
            codes, _ = pd.factorize(df_cleaned[column])
            df_cleaned[column] = codes
        else:
            raise ValueError(f"Encoding strategy '{strategy}' not supported.")
            
        return df_cleaned
