import os
import pandas as pd
from typing import Tuple
from config import settings
from utils.logger import get_logger
from utils.helpers import get_file_extension, get_format_type, format_bytes

logger = get_logger("core.loader")

class DataLoader:
    """
    Data Loader module responsible for loading datasets (CSV, Excel, JSON, Parquet)
    and validating file parameters.
    """
    
    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        """
        Detects file type, validates dimensions/size, and reads it into a pandas DataFrame.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The dataset file at '{file_path}' does not exist.")
            
        file_size = os.path.getsize(file_path)
        logger.info(f"Loading dataset: {file_path} (Size: {format_bytes(file_size)})")
        
        # File size check
        if file_size > settings.MAX_UPLOAD_SIZE:
            msg = f"File size ({format_bytes(file_size)}) exceeds maximum limit of {format_bytes(settings.MAX_UPLOAD_SIZE)}."
            logger.warning(msg)
            raise ValueError(msg)
            
        # Detect extension and load
        try:
            format_type = get_format_type(file_path)
        except ValueError as e:
            logger.error(str(e))
            raise ValueError(f"Unsupported file format. Please upload one of: {', '.join(settings.SUPPORTED_FORMATS.keys())}")
            
        try:
            if format_type == "csv":
                df = pd.read_csv(file_path)
            elif format_type == "excel":
                # Requires openpyxl
                df = pd.read_excel(file_path)
            elif format_type == "json":
                df = pd.read_json(file_path)
            elif format_type == "parquet":
                # Requires pyarrow/fastparquet
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Reader not implemented for format: {format_type}")
                
            # Log success metrics
            rows, cols = df.shape
            logger.info(f"Successfully loaded dataset. Shape: {rows} rows, {cols} columns.")
            return df
            
        except Exception as e:
            logger.exception(f"Error reading dataset at {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to read the dataset file. Error detail: {str(e)}")
