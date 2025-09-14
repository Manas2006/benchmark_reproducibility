"""
Deterministic checks for CoT evaluation.

This package contains pure functions for checking various aspects of
Chain-of-Thought reasoning without requiring LLM calls.
"""

from .arithmetic import check_step_equations
from .nli import nli_label, nli_probs
from .coverage import number_coverage
from .redundancy import redundancy_pairs
from .heuristics import wrong_but_right, self_repair_markers, shortcut_signature

__all__ = [
    "check_step_equations",
    "nli_label",
    "nli_probs", 
    "number_coverage",
    "redundancy_pairs",
    "wrong_but_right",
    "self_repair_markers",
    "shortcut_signature"
]
