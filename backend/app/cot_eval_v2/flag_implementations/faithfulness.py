"""
Faithfulness flag implementations.
"""

from typing import List, Dict, Any, Optional
from ..schema import EvalItem
from ..flag import Flag
from ..math_utils import compare_floats
from ..llm_judge import get_llm_judge
from ..config import SHORTCUT_THRESHOLD


def check_faithfulness_flags(item: EvalItem) -> List[Flag]:
    """
    Check for faithfulness issues.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of faithfulness flags
    """
    flags = []
    
    # Check unjustified final answer
    unjustified_flag = check_unjustified_final(item)
    if unjustified_flag:
        flags.append(unjustified_flag)
    
    # Check shortcut unfaithful reasoning
    shortcut_flag = check_shortcut_unfaithful(item)
    if shortcut_flag:
        flags.append(shortcut_flag)
    
    return flags


def check_unjustified_final(item: EvalItem) -> Optional[Flag]:
    """
    Check if final answer is unjustified by the reasoning steps.
    
    Args:
        item: Evaluation item
        
    Returns:
        Flag if unjustified, None otherwise
    """
    if not item.steps:
        return None
    
    # Simple check: if final answer doesn't match gold answer
    if item.problem.gold_answer and item.final_value_norm:
        try:
            gold_num = float(item.problem.gold_answer)
            if not compare_floats(item.final_value_norm, gold_num):
                return Flag(
                    pillar="faithfulness",
                    step="reasoning",
                    issue="unjustified_final",
                    details={
                        "reconstructed": None,
                        "final": item.final_value_norm,
                        "missing_symbols": [],
                        "judge_used": False
                    }
                )
        except (ValueError, TypeError):
            pass
    
    # Check if final answer is derivable from steps
    has_equations = any(step.equations for step in item.steps)
    if not has_equations and item.final_value_norm:
        return Flag(
            pillar="faithfulness",
            step="reasoning",
            issue="unjustified_final",
            details={
                "reconstructed": None,
                "final": item.final_value_norm,
                "missing_symbols": [],
                "judge_used": False
            }
        )
    
    return None


def check_shortcut_unfaithful(item: EvalItem) -> Optional[Flag]:
    """
    Check for shortcut unfaithful reasoning.
    
    Args:
        item: Evaluation item
        
    Returns:
        Flag if shortcut detected, None otherwise
    """
    if not item.steps or len(item.steps) < 2:
        return None
    
    # Check if most steps are just text without equations
    text_only_steps = sum(1 for step in item.steps if not step.equations and not step.numbers)
    total_steps = len(item.steps)
    
    if text_only_steps / total_steps > 0.6:
        return Flag(
            pillar="faithfulness",
            step="reasoning",
            issue="shortcut_unfaithful",
            details={
                "non_contrib_count": text_only_steps,
                "total_comp_steps": total_steps,
                "shuffle_ok": False,
                "judge_used": False
            }
        )
    
    return None
