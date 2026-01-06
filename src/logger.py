"""
Simple logging configuration for the application.
"""
import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a simple logger with console output.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)
    
    # Console handler with simple formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Simple format: timestamp - level - logger name - message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

