"""
Data schema for Pillars v2 evaluator.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Step:
    """Represents a single reasoning step."""
    idx: int
    text: str
    equations: List[str]        # parsed equations "lhs = rhs"
    numbers: List[float]        # detected numeric literals
    symbols: List[str]          # detected symbol tokens (A, v, price, etc.)
    units: List[str]            # attached units per number (if any)
    mentions: List[str]         # key nouns/entities
    ops: List[str]              # "+", "-", "*", "/", "<", ">", "<=", ">="


@dataclass
class Problem:
    """Represents the problem context."""
    prompt_text: str
    gold_answer: Optional[str]  # may be None
    context_docs: List[str]     # optional retrieved passages


@dataclass
class EvalItem:
    """Complete evaluation item with problem and reasoning."""
    problem: Problem
    steps: List[Step]
    final_text: str             # model's final line
    final_value_norm: Optional[float]  # parsed numeric if applicable


@dataclass
class Flag:
    """Represents a detected issue flag."""
    pillar: str
    name: str
    step_idx: Optional[int]
    severity: float             # 0..1 heuristic certainty
    reason: str
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pillar": self.pillar,
            "name": self.name,
            "step_idx": self.step_idx,
            "severity": self.severity,
            "reason": self.reason,
            "meta": self.meta
        }
