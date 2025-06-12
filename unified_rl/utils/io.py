"""Utility functions for I/O operations and reproducibility."""

import json
import random
from pathlib import Path
from typing import Any, Dict

import gym
import numpy as np
import torch
from omegaconf import DictConfig


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dictionary data to a JSON file.

    Args:
        data: Dictionary to save
        path: Path to save the JSON file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_yaml(cfg: DictConfig, path: str) -> None:
    """Save Hydra configuration to a YAML file.

    Args:
        cfg: Hydra configuration object
        path: Path to save the YAML file
    """
    from omegaconf import OmegaConf

    OmegaConf.save(config=cfg, f=path)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gym.spaces.prng.seed(seed)
