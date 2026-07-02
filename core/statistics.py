import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
from utils.logger import get_logger

logger = get_logger("core.statistics")

def clean_input(data: Any) -> np.ndarray:
    """
    Converts input into a clean 1D numpy array, dropping NaNs and non-numeric values.
    """
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data.flatten()
    else:
        arr = np.array(data, dtype=float).flatten()
    
    # Keep only finite numeric values
    arr = arr[np.isfinite(arr)]
    return arr

def calculate_mean(data: Any) -> float:
    arr = clean_input(data)
    if len(arr) == 0:
        return float('nan')
    return float(np.sum(arr) / len(arr))

def calculate_median(data: Any) -> float:
    arr = clean_input(data)
    if len(arr) == 0:
        return float('nan')
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if n % 2 == 1:
        return float(sorted_arr[n // 2])
    else:
        return float((sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0)

def calculate_mode(data: Any) -> str:
    """
    Calculates the mode of a dataset.
    If multiple modes exist, returns them comma-separated (e.g. "Male, Female").
    If no mode exists, returns "No unique mode".
    Every output is returned as a string to guarantee uniform data types in DataFrames.
    """
    if isinstance(data, pd.Series):
        series = data.dropna()
    else:
        series = pd.Series(data).dropna()
        
    if len(series) == 0:
        return "No unique mode"
        
    vals, counts = np.unique(series.to_numpy(), return_counts=True)
    if len(counts) == 0:
        return "No unique mode"
        
    max_count = np.max(counts)
    if max_count == 1 and len(vals) > 1:
        return "No unique mode"
        
    modes = vals[counts == max_count].tolist()
    if len(modes) == 0:
        return "No unique mode"
    elif len(modes) == 1:
        val = modes[0]
        if isinstance(val, (float, np.floating)) and val.is_integer():
            return str(int(val))
        return str(val)
    else:
        formatted_modes = []
        for val in modes:
            if isinstance(val, (float, np.floating)) and val.is_integer():
                formatted_modes.append(str(int(val)))
            else:
                formatted_modes.append(str(val))
        return ", ".join(formatted_modes)

def calculate_variance(data: Any, ddof: int = 1) -> float:
    """
    Calculates sample variance (ddof=1) or population variance (ddof=0).
    """
    arr = clean_input(data)
    n = len(arr)
    if n <= ddof:
        return float('nan')
    mean = calculate_mean(arr)
    sq_diff = np.sum((arr - mean) ** 2)
    return float(sq_diff / (n - ddof))

def calculate_std_dev(data: Any, ddof: int = 1) -> float:
    var = calculate_variance(data, ddof)
    return float('nan') if np.isnan(var) else float(np.sqrt(var))

def calculate_min(data: Any) -> float:
    arr = clean_input(data)
    if len(arr) == 0:
        return float('nan')
    return float(np.min(arr))

def calculate_max(data: Any) -> float:
    arr = clean_input(data)
    if len(arr) == 0:
        return float('nan')
    return float(np.max(arr))

def calculate_quartiles(data: Any) -> Dict[str, float]:
    """
    Returns Q1 (25th), Q2 (50th/Median), and Q3 (75th) percentiles.
    """
    arr = clean_input(data)
    if len(arr) == 0:
        return {"q1": float('nan'), "q2": float('nan'), "q3": float('nan')}
    sorted_arr = np.sort(arr)
    # Using linear interpolation for percentiles matching pandas
    return {
        "q1": float(np.percentile(sorted_arr, 25)),
        "q2": float(np.percentile(sorted_arr, 50)),
        "q3": float(np.percentile(sorted_arr, 75))
    }

def calculate_skewness(data: Any) -> float:
    """
    Calculates Fisher-Pearson standardized skewness coefficient from scratch.
    """
    arr = clean_input(data)
    n = len(arr)
    if n < 3:
        return float('nan')
        
    mean = calculate_mean(arr)
    diff = arr - mean
    
    # 2nd and 3rd central moments
    m2 = np.sum(diff ** 2) / n
    m3 = np.sum(diff ** 3) / n
    
    if m2 == 0:
        return 0.0
        
    # Unbiased skewness calculation
    skew = m3 / (m2 ** 1.5)
    # Correct for bias in small samples
    unbiased_skew = (np.sqrt(n * (n - 1)) / (n - 2)) * skew
    return float(unbiased_skew)

def calculate_kurtosis(data: Any) -> float:
    """
    Calculates Fisher's excess kurtosis (excess over 3) from scratch.
    """
    arr = clean_input(data)
    n = len(arr)
    if n < 4:
        return float('nan')
        
    mean = calculate_mean(arr)
    diff = arr - mean
    
    m2 = np.sum(diff ** 2) / n
    m4 = np.sum(diff ** 4) / n
    
    if m2 == 0:
        return 0.0
        
    kurt = m4 / (m2 ** 2) - 3.0
    # Correct for bias
    bias_correction = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * kurt + 6)
    return float(bias_correction)

def summarize_column(data: Any) -> Dict[str, Any]:
    """
    Compiles all statistics for a single numeric feature.
    """
    arr = clean_input(data)
    if len(arr) == 0:
        return {}
        
    quartiles = calculate_quartiles(arr)
    return {
        "count": len(arr),
        "mean": calculate_mean(arr),
        "median": calculate_median(arr),
        "mode": calculate_mode(arr),
        "variance": calculate_variance(arr),
        "std_dev": calculate_std_dev(arr),
        "min": calculate_min(arr),
        "max": calculate_max(arr),
        "q1": quartiles["q1"],
        "q2": quartiles["q2"],
        "q3": quartiles["q3"],
        "iqr": quartiles["q3"] - quartiles["q1"],
        "skewness": calculate_skewness(arr),
        "kurtosis": calculate_kurtosis(arr)
    }

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Generates statistics for all numeric columns in a pandas DataFrame.
    """
    logger.info("Calculating custom statistics summary for dataframe columns.")
    summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col] = summarize_column(df[col])
    return summary
