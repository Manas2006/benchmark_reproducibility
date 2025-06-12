"""Transformer-based policy network for RL."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class PolicyValueNet(nn.Module):
    """Transformer-based policy and value network.

    This network consists of:
    1. A shared transformer encoder
    2. A policy head that outputs action logits
    3. A value head that outputs state values
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the network.

        Args:
            cfg: Configuration containing model parameters
        """
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(
            cfg.env.observation_space.shape[0], cfg.model.hidden_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.hidden_size,
            nhead=cfg.model.num_heads,
            dim_feedforward=cfg.model.hidden_size * 4,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.model.num_layers
        )

        # Policy head
        self.policy_head = nn.Linear(cfg.model.hidden_size, cfg.env.action_space.n)

        # Value head
        self.value_head = nn.Linear(cfg.model.hidden_size, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, obs_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (action_logits, value)
        """
        # Project input
        x = self.input_proj(x)

        # Transformer encoding
        x = self.transformer(x, mask=mask)

        # Get final hidden state
        x = x[:, -1]  # (batch_size, hidden_size)

        # Policy and value heads
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)

        return logits, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate for a state.

        Args:
            x: Input tensor of shape (batch_size, seq_len, obs_dim)

        Returns:
            Value estimate
        """
        _, value = self.forward(x)
        return value

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action probabilities, value, log probs, and entropy.

        Args:
            x: Input tensor
            action: Optional action to evaluate

        Returns:
            Tuple of (action_probs, value, log_probs, entropy)
        """
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)

        if action is None:
            action = torch.multinomial(probs, num_samples=1)

        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        return probs, value, action_log_probs, entropy
