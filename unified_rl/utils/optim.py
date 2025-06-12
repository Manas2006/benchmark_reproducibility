"""Optimizer and scheduler utilities."""

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.optim import Adam, AdamW, SGD
from transformers import get_scheduler


def get_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Create an optimizer based on configuration.

    Args:
        model: Model whose parameters to optimize
        cfg: Configuration containing optimizer settings

    Returns:
        Configured optimizer

    Raises:
        ValueError: If optimizer type is not supported
    """
    if not hasattr(cfg.optimizer, "args"):
        raise ValueError("optimizer.args must be set in configuration")

    args = cfg.optimizer.args

    if cfg.optimizer.type == "AdamW":
        return AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
            eps=args.eps,
        )

    if cfg.optimizer.type == "Adam":
        return Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
            eps=args.eps,
        )

    if cfg.optimizer.type == "SGD":
        return SGD(
            model.parameters(),
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
            weight_decay=args.weight_decay,
            nesterov=getattr(args, "nesterov", False),
        )

    if cfg.optimizer.type == "Lion":
        try:
            from lion_pytorch import Lion
        except ImportError:
            raise ImportError(
                "Lion optimizer requires lion-pytorch package. "
                "Install with: pip install lion-pytorch"
            )
        return Lion(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=getattr(args, "beta1", 0.9),
            beta2=getattr(args, "beta2", 0.99),
        )

    raise ValueError(f"Unsupported optimizer type: {cfg.optimizer.type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    num_training_steps: Optional[int] = None,
) -> Any:
    """Create a scheduler based on configuration.

    Args:
        optimizer: Optimizer to schedule
        cfg: Configuration containing scheduler settings
        num_training_steps: Total number of training steps (optional)

    Returns:
        Configured scheduler

    Raises:
        ValueError: If scheduler type is not supported
    """
    if not hasattr(cfg.scheduler, "args"):
        raise ValueError("scheduler.args must be set in configuration")

    args = cfg.scheduler.args

    if cfg.scheduler.type == "linear":
        return get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(args.warmup_ratio * args.total_steps),
            num_training_steps=args.total_steps,
        )

    if cfg.scheduler.type == "cosine":
        return get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=int(args.warmup_ratio * args.total_steps),
            num_training_steps=args.total_steps,
            num_cycles=getattr(args, "num_cycles", 1.0),
            min_lr=getattr(args, "min_lr", 0.0),
        )

    if cfg.scheduler.type == "constant":
        return get_scheduler(
            name="constant",
            optimizer=optimizer,
        )

    if cfg.scheduler.type == "constant_with_warmup":
        return get_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=getattr(args, "num_warmup_steps", 0),
        )

    raise ValueError(f"Unsupported scheduler type: {cfg.scheduler.type}")
