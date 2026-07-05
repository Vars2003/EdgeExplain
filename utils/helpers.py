import os
import time
from typing import Callable, Any
from config import settings
from utils.logger import get_logger

logger = get_logger("utils.helpers")

def format_bytes(size: float) -> str:
    """
    Formats a byte size into a human-readable string (e.g. KB, MB, GB).
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def get_file_extension(filename: str) -> str:
    """
    Returns the file extension in lowercase, including the dot (e.g., '.csv').
    """
    _, ext = os.path.splitext(filename)
    return ext.lower()

def is_supported_file(filename: str) -> bool:
    """
    Checks if the given filename is supported based on configured formats.
    """
    ext = get_file_extension(filename)
    for format_name, extensions in settings.SUPPORTED_FORMATS.items():
        if ext in extensions:
            return True
    return False

def get_format_type(filename: str) -> str:
    """
    Identifies the format type name (e.g., 'csv', 'parquet') based on extension.
    """
    ext = get_file_extension(filename)
    for format_name, extensions in settings.SUPPORTED_FORMATS.items():
        if ext in extensions:
            return format_name
    raise ValueError(f"Unsupported file extension: {ext}")

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that logs the execution time of a function.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"Function {func.__name__} took {elapsed:.4f} seconds to execute.")
        return result
    return wrapper

def inject_custom_css() -> None:
    """
    Injects custom premium slate-dark glassmorphism styling into the Streamlit session.
    """
    import streamlit as st
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0F172A !important;
    }
    
    /* Header styles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #334155 !important;
    }
    
    /* Glassmorphic cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.6)) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        text-align: center !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
    }
    .stat-label {
        font-size: 0.85rem !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 4px !important;
    }
    .stat-val {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #38BDF8 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Custom tag styles */
    .tag {
        display: inline-block !important;
        padding: 4px 10px !important;
        border-radius: 9999px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        margin-right: 6px !important;
    }
    .tag-blue { background: rgba(59, 130, 246, 0.2) !important; color: #60A5FA !important; border: 1px solid rgba(59, 130, 246, 0.3) !important; }
    .tag-purple { background: rgba(139, 92, 246, 0.2) !important; color: #A78BFA !important; border: 1px solid rgba(139, 92, 246, 0.3) !important; }
    .tag-green { background: rgba(16, 185, 129, 0.2) !important; color: #34D399 !important; border: 1px solid rgba(16, 185, 129, 0.3) !important; }
    .tag-orange { background: rgba(245, 158, 11, 0.2) !important; color: #FBBF24 !important; border: 1px solid rgba(245, 158, 11, 0.3) !important; }
    .tag-red { background: rgba(239, 68, 68, 0.2) !important; color: #F87171 !important; border: 1px solid rgba(239, 68, 68, 0.3) !important; }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def normalize_dataframe_for_rendering(df_to_render: Any) -> Any:
    """
    Ensures that any column in the DataFrame does not contain Python lists, 
    tuples, sets, dicts, or numpy arrays, which would cause PyArrow serialization errors 
    when rendered in Streamlit.
    """
    import pandas as pd
    import numpy as np
    import json

    if not isinstance(df_to_render, pd.DataFrame):
        return df_to_render

    df_clean = df_to_render.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            def clean_val(x):
                if isinstance(x, (list, tuple, set)):
                    return ", ".join(map(str, x))
                elif isinstance(x, dict):
                    try:
                        return json.dumps(x)
                    except Exception:
                        return str(x)
                elif isinstance(x, np.ndarray):
                    return ", ".join(map(str, x.tolist()))
                elif x is None or pd.isna(x):
                    return x
                elif not isinstance(x, (str, int, float, bool)):
                    return str(x)
                return x
            
            df_clean[col] = df_clean[col].apply(clean_val)
    return df_clean


