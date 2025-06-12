"""Reward model registry and factory functions."""

import importlib
from typing import Callable, List

import torch
from omegaconf import DictConfig
from transformers import pipeline


def get_reward_fn(cfg: DictConfig) -> Callable[[List[str]], List[float]]:
    """Get a reward function based on configuration.

    Args:
        cfg: Configuration containing reward model settings

    Returns:
        Function that takes a list of texts and returns rewards

    Raises:
        ValueError: If reward model configuration is invalid
    """
    if not cfg.reward_model.type:
        raise ValueError("reward_model.type must be set")

    if cfg.reward_model.type == "pipeline":
        if not cfg.reward_model.name:
            raise ValueError("reward_model.name must be set for pipeline type")
        return get_pipeline_reward_fn(cfg)

    if cfg.reward_model.type == "custom":
        if not cfg.reward_model.custom_path:
            raise ValueError("reward_model.custom_path must be set for custom type")
        return get_custom_reward_fn(cfg)

    raise ValueError(f"Invalid reward_model.type: {cfg.reward_model.type}")


def get_pipeline_reward_fn(cfg: DictConfig) -> Callable[[List[str]], List[float]]:
    """Get a reward function from a Hugging Face pipeline.

    Args:
        cfg: Configuration containing reward model settings

    Returns:
        Function that takes a list of texts and returns rewards
    """
    # Initialize pipeline
    pipe = pipeline(
        "text-classification",
        model=cfg.reward_model.name,
        device=cfg.reward_model.device,
    )

    def reward_fn(texts: List[str]) -> List[float]:
        """Get rewards for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of reward values
        """
        results = pipe(texts)
        return [result["score"] for result in results]

    return reward_fn


def get_custom_reward_fn(cfg: DictConfig) -> Callable[[List[str]], List[float]]:
    """Get a custom reward function from a module.

    Args:
        cfg: Configuration containing reward model settings

    Returns:
        Function that takes a list of texts and returns rewards
    """
    # Parse module and function names
    module_path, func_name = cfg.reward_model.custom_path.split(":")

    # Import module and get function
    module = importlib.import_module(module_path)
    reward_fn = getattr(module, func_name)

    return reward_fn
