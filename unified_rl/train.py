"""Main training script for unified RLHF."""

import os
from typing import Dict

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from algorithms import trl, openr1, verl
from reward.registry import get_reward_fn
from utils.io import save_json, save_yaml, seed_everything


# Map library names to their respective modules
ADAPTERS = {"trl": trl, "openr1": openr1, "verl": verl}


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function.

    This script supports training with different RL libraries:
    - TRL (Transformer Reinforcement Learning)
    - Open-R1
    - Verl

    You can specify which library to use in two ways:
    1. Command line: python train.py library=trl
    2. Config file: Set library: trl in default.yaml

    Args:
        cfg: Hydra configuration object. Must include:
            - library: One of ["trl", "openr1", "verl"]
            - dataset.name: Dataset name or path
            - reward_model.type: Type of reward model
            - reward_model.name: Name of reward model

    Raises:
        RuntimeError: If required configuration is missing
        ValueError: If unsupported library is specified
    """
    # Set random seeds
    seed_everything(cfg.seed)

    # Validate and load dataset
    if not cfg.dataset.name:
        raise RuntimeError("You must set dataset.name for offline/text RL.")

    dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
        data_files=getattr(cfg.dataset, "data_files", None),
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_name,
        device_map=cfg.model.device_map,
        load_in_8bit=cfg.model.load_in_8bit,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    # Validate and get reward function
    if not cfg.reward_model.type or not cfg.reward_model.name:
        raise RuntimeError("You must specify reward_model.type and reward_model.name.")
    reward_fn = get_reward_fn(cfg)

    # Get trainer adapter
    if cfg.library not in ADAPTERS:
        raise ValueError(
            f"Unsupported library: {cfg.library}. "
            f"Must be one of {list(ADAPTERS.keys())}"
        )
    adapter = ADAPTERS[cfg.library]

    # Build and run trainer
    trainer = adapter.build_trainer(cfg, model, dataset, reward_fn)
    metrics = trainer.train()

    # Save results
    save_yaml(cfg, "config_used.yaml")
    save_json(metrics, "metrics.json")

    if cfg.verbose:
        print("\nTraining results:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
