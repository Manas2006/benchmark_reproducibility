import torch
import torch.nn.functional as F
import numpy as np


class ProbabilityRecorder:
    """
    Records probabilities and all occurrences of a full multi-token sequence match.
    Compatible with vLLM's logits_processors hook.
    """

    def __init__(self, gold_token_ids: list[int], tokenizer):
        self.gold_token_ids = gold_token_ids
        self.tokenizer = tokenizer

        # Tracks our progress through the gold sequence (e.g., 0, 1, 2...)
        self.target_token_index = 0
        # A temporary buffer to hold the steps of a potential match
        self.current_match_steps = []
        # The final list that stores all completed matches
        self.successful_matches = []

        # Per-step probability of the current target token
        self.probs = []
        
        # Per-step probability of the chosen token
        self.chosen_token_probs = []
        
        # Per-step entropy of the next token distribution
        self.entropies = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=-1)
        
        # Get the token the model actually chose
        chosen_id = torch.argmax(logits).item()
        
        # Always record the probability of the chosen token
        self.chosen_token_probs.append(probabilities[chosen_id].item())
        
        # Always calculate and record the entropy of the probability distribution
        # Use numerically stable entropy calculation
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        log_probs = torch.log(probabilities + eps)
        entropy = -torch.sum(probabilities * log_probs).item()
        self.entropies.append(entropy)
        
        # If we don't have a gold sequence, we can't track correct token probability
        if not self.gold_token_ids:
            # Still append a placeholder for consistency in list lengths
            self.probs.append(0.0)  # Or use NaN if preferred
            return logits

        # Determine the target ID based on our progress
        current_target_id = self.gold_token_ids[self.target_token_index]

        # Record the probability of the gold token at this step
        self.probs.append(probabilities[current_target_id].item())

        current_step_number = len(token_ids)

        # State Machine Logic for Matching
        if chosen_id == current_target_id:
            # MATCH: The model is correctly following the sequence.
            self.current_match_steps.append(current_step_number)
            self.target_token_index += 1

            # Check if the full sequence has just been completed
            if self.target_token_index == len(self.gold_token_ids):
                # SUCCESS! We found a full match.
                self.successful_matches.append(self.current_match_steps)
                # Reset to look for the next potential match.
                self.target_token_index = 0
                self.current_match_steps = []
        else:
            # MISMATCH: The model broke the sequence. Reset everything.
            self.target_token_index = 0
            self.current_match_steps = []

            # Edge Case: The token that broke the sequence might be the start of a new one.
            if chosen_id == self.gold_token_ids[0]:
                self.target_token_index = 1
                self.current_match_steps.append(current_step_number)

        return logits

