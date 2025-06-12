"""Verl adapter for RLHF training."""

from typing import Any, Callable, Dict, List

import ray
import torch
from datasets import Dataset
from omegaconf import DictConfig
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils import hf_processor, hf_tokenizer
from verl.configs import VerlConfig, RayConfig
from verl.workers.fsdp_workers import CriticWorker

from . import Trainer


class VerlTrainer(Trainer):
    """Verl trainer adapter."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataset: Dataset,
        reward_fn: Callable[[List[str]], List[float]],
    ):
        """Initialize the Verl trainer.

        Args:
            cfg: Configuration object
            model: Language model
            dataset: Training dataset
            reward_fn: Function that computes rewards for texts
        """
        super().__init__(cfg, model, dataset, reward_fn)

        # Create Verl config
        verl_config = VerlConfig(
            model_name=cfg.model.pretrained_name,
            dataset_name=cfg.dataset.name,
            prompt_column=cfg.dataset.prompt_column,
            batch_size=cfg.training.batch_size,
            max_length=cfg.dataset.max_length,
            output_dir=cfg.training.output_dir,
            learning_rate=cfg.training.learning_rate,
            max_grad_norm=cfg.training.max_grad_norm,
            target_kl=cfg.algorithm.target_kl,
            epochs=cfg.training.num_train_epochs,
            gamma=cfg.algorithm.gamma,
            seed=cfg.training.seed,
            do_eval=cfg.training.do_eval,
            system_prompt=cfg.training.system_prompt,
        )

        # Create Ray config
        ray_config = RayConfig(
            num_workers=cfg.ray.num_workers,
            num_gpus=cfg.ray.num_gpus,
            num_cpus=cfg.ray.num_cpus,
            memory=cfg.ray.memory,
        )

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=ray_config.num_cpus,
                num_gpus=ray_config.num_gpus,
                memory=ray_config.memory,
            )

        # Get actor rollout class
        actor_rollout_cls = ray.remote(verl_config.actor_rollout_ref)

        # Map roles to worker classes
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Configure resource pools
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [1] * ray_config.num_workers,  # 1 GPU per worker
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Initialize resource pool manager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        # Create RLHF dataset
        rlhf_dataset = RLHFDataset(
            dataset=dataset,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_length=verl_config.max_length,
            prompt_column=verl_config.prompt_column,
            system_prompt=verl_config.system_prompt,
        )

        # Initialize trainer
        self.trainer = RayPPOTrainer(
            model=model,
            dataset=rlhf_dataset,
            reward_fn=reward_fn,
            config=verl_config,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
        )

    def train(self) -> Dict[str, Any]:
        """Run training using Verl's Ray PPO trainer.

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
    """Build a Verl trainer.

    Args:
        cfg: Configuration object
        model: Language model
        dataset: Training dataset
        reward_fn: Function that computes rewards for texts

    Returns:
        Configured Verl trainer
    """
    return VerlTrainer(cfg, model, dataset, reward_fn)
