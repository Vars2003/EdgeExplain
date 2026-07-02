import logging
import sys
from config import settings

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance writing to both logs/edgeexplain.log and stdout.
    """
    logger = logging.getLogger(name)
    
    # If the logger is already configured, return it
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    
    # Formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(settings.LOG_FILE_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If writing to file fails (e.g. permission issues), output to stderr
        print(f"Warning: Failed to create file handler for logger: {e}", file=sys.stderr)
        
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger
