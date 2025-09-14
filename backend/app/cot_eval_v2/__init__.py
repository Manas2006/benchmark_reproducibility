"""
CoT Evaluation v2 - Phase 1: Deterministic checks and flags

This package provides modular code checks for Chain-of-Thought reasoning
without requiring LLM calls. It focuses on arithmetic, coherence, coverage,
redundancy, and faithfulness heuristics.
"""

from .flags import Flag, FlagCollector
from .evaluator import PillarsEvaluator
from .scoring import rule_scores, fuse_with_judge, compare_score_methods
from .judge import Judge, MockJudge

__version__ = "2.0.0"
__all__ = ["Flag", "FlagCollector", "PillarsEvaluator", "rule_scores", "fuse_with_judge", "compare_score_methods", "Judge", "MockJudge"]
