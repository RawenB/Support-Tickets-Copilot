"""Helpers for deterministic seeding across model and data operations."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and the standard random module."""
    random.seed(seed)
    np.random.seed(seed)
