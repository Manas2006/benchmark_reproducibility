"""TRL (Transformer Reinforcement Learning) adapter."""

from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from trl import (
    PPOTrainer,
    PPOConfig,
    DPOTrainer,
    DPOConfig,
    GRPOTrainer,
    GRPOConfig,
    TrlParser,
)
from trl.script_utils import ScriptArguments, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import Trainer


class TRLTrainer(Trainer):
    """TRL trainer adapter."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataset: Dataset,
        reward_fn: Callable[[List[str]], List[float]],
    ):
        """Initialize the TRL trainer.

        Args:
            cfg: Configuration object
            model: Language model
            dataset: Training dataset
            reward_fn: Function that computes rewards for texts
        """
        super().__init__(cfg, model, dataset, reward_fn)

        # Parse arguments using TRL's parser
        if cfg.algorithm.name == "ppo":
            parser = TrlParser((ScriptArguments, PPOConfig, ModelConfig))
            script_args, training_args, model_args = parser.parse_args_and_config()

            self.trainer = PPOTrainer(
                model=model,
                config=training_args,
                dataset=dataset,
                tokenizer=None,  # Will be loaded by trainer
                reward_fn=reward_fn,
            )
        elif cfg.algorithm.name == "dpo":
            parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
            script_args, training_args, model_args = parser.parse_args_and_config()

            self.trainer = DPOTrainer(
                model=model,
                config=training_args,
                dataset=dataset,
                tokenizer=None,  # Will be loaded by trainer
            )
        elif cfg.algorithm.name == "grpo":
            parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
            script_args, training_args, model_args = parser.parse_args_and_config()

            self.trainer = GRPOTrainer(
                model=model,
                config=training_args,
                dataset=dataset,
                tokenizer=None,  # Will be loaded by trainer
                reward_fn=reward_fn,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")

    def train(self) -> Dict[str, Any]:
        """Run training using TRL trainer.

        Returns:
            Dictionary of training metrics
        """
        # Run training
        train_result = self.trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.dataset)

        # Log and save metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        # Save model
        self.trainer.save_model(self.cfg.training.output_dir)

        # Evaluate if enabled
        if self.cfg.training.do_eval:
            eval_metrics = self.trainer.evaluate()
            eval_metrics["eval_samples"] = len(self.dataset)
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)

        return metrics


def build_trainer(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset: Dataset,
    reward_fn: Callable[[List[str]], List[float]],
) -> Trainer:
    """Build a TRL trainer.

    Args:
        cfg: Configuration object
        model: Language model
        dataset: Training dataset
        reward_fn: Function that computes rewards for texts

    Returns:
        Configured TRL trainer
    """
    return TRLTrainer(cfg, model, dataset, reward_fn)
