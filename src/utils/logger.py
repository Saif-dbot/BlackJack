"""Structured logging for training."""

import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str, log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any]) -> None:
    """Log metrics in JSON format.
    
    Args:
        logger: Logger instance
        metrics: Metrics dictionary
    """
    logger.info(f"Metrics: {json.dumps(metrics)}")
