"""Shared logging configuration for the support-copilot project."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger for support-copilot modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
