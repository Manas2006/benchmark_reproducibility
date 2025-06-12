"""Base trainer class and algorithm adapters."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
from datasets import Dataset
from omegaconf import DictConfig


class Trainer(ABC):
    """Abstract base class for RL trainers."""

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        dataset: Dataset,
        reward_fn: Callable[[List[str]], List[float]],
    ):
        """Initialize the trainer.

        Args:
            cfg: Configuration object
            model: Language model
            dataset: Training dataset
            reward_fn: Function that computes rewards for texts
        """
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.reward_fn = reward_fn

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run training loop.

        Returns:
            Dictionary of training metrics
        """
        pass
