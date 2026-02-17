"""Utility functions: seed setting, logging, early stopping, device selection."""

import copy
import logging
import os
import random
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return CUDA device if available, else CPU."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required for get_device()")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both file and console."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lg_cotrain")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(output_dir, "experiment.log"))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


class EarlyStopping:
    """Early stopping based on a monitored metric (higher is better).

    Works with any model that supports state_dict()/load_state_dict(),
    or with a simple dict for testing without torch.
    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0
        self.best_state_dict = None

    def step(self, score: float, model) -> bool:
        """Update with new score. Returns True if training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model):
        """Restore the model to the best checkpoint."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
