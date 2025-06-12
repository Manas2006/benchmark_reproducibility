"""Open-R1 adapter for RLHF training."""

from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from omegaconf import DictConfig
from trl import GRPOTrainer, TrlParser
from open_r1.configs import GRPOScriptArguments, GRPOConfig, ModelConfig

from . import Trainer


class OpenR1Trainer(Trainer):
    """Open-R1 trainer adapter."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataset: Dataset,
        reward_fn: Callable[[List[str]], List[float]],
    ):
        """Initialize the Open-R1 trainer.

        Args:
            cfg: Configuration object
            model: Language model
            dataset: Training dataset
            reward_fn: Function that computes rewards for texts
        """
        super().__init__(cfg, model, dataset, reward_fn)

        # Parse arguments using Open-R1's parser
        parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config()

        # Format dataset into conversations
        def make_conversation(
            example, prompt_column: str = script_args.dataset_prompt_column
        ):
            prompt = []
            if training_args.system_prompt is not None:
                prompt.append(
                    {"role": "system", "content": training_args.system_prompt}
                )
            if prompt_column not in example:
                raise ValueError(
                    f"Dataset Question Field Error: {prompt_column} is not supported."
                )
            prompt.append({"role": "user", "content": example[prompt_column]})
            return {"prompt": prompt}

        self.dataset = self.dataset.map(make_conversation)
        for split in self.dataset:
            if "messages" in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].remove_columns("messages")

        # Initialize GRPO trainer
        self.trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=training_args,
            train_dataset=self.dataset[script_args.dataset_train_split],
            eval_dataset=(
                self.dataset[script_args.dataset_test_split]
                if training_args.eval_strategy != "no"
                else None
            ),
            peft_config=None,  # TODO: Add PEFT support if needed
            callbacks=None,  # TODO: Add callback support if needed
        )

    def train(self) -> Dict[str, Any]:
        """Run training using Open-R1's GRPO trainer.

        Returns:
            Dictionary of training metrics
        """
        # Run training
        train_result = self.trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.dataset[self.cfg.dataset.train_split])

        # Log and save metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        # Save model
        self.trainer.model.generation_config.eos_token_id = (
            self.trainer.tokenizer.eos_token_id
        )
        self.trainer.save_model(self.cfg.training.output_dir)

        # Evaluate if enabled
        if self.cfg.training.do_eval:
            eval_metrics = self.trainer.evaluate()
            eval_metrics["eval_samples"] = len(
                self.dataset[self.cfg.dataset.test_split]
            )
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)

        return metrics


def build_trainer(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset: Dataset,
    reward_fn: Callable[[List[str]], List[float]],
) -> Trainer:
    """Build an Open-R1 trainer.

    Args:
        cfg: Configuration object
        model: Language model
        dataset: Training dataset
        reward_fn: Function that computes rewards for texts

    Returns:
        Configured Open-R1 trainer
    """
    return OpenR1Trainer(cfg, model, dataset, reward_fn)
