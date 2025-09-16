"""
Utility flag implementations.
"""

from typing import List, Dict, Any, Optional
from ..schema import EvalItem
from ..flag import Flag
from ..nlp_utils import get_nlp_utils
from ..llm_judge import get_llm_judge
from ..config import REDUNDANCY_SIM, OFFTOPIC_JACCARD


def check_utility_flags(item: EvalItem) -> List[Flag]:
    """
    Check for utility issues.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of utility flags
    """
    flags = []
    
    # Check redundant steps
    redundant_flags = check_redundant_step(item)
    flags.extend(redundant_flags)
    
    # Check off-topic steps
    offtopic_flags = check_off_topic(item)
    flags.extend(offtopic_flags)
    
    return flags


def check_redundant_step(item: EvalItem) -> List[Flag]:
    """
    Check for redundant steps that make no progress.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of redundant step flags
    """
    flags = []
    
    if len(item.steps) < 2:
        return flags
    
    # Get NLP utils
    nlp = get_nlp_utils()
    
    # Check each step for redundancy
    for i, step in enumerate(item.steps):
        # Check similarity to previous context
        prev_context = " ".join(item.steps[j].text for j in range(i))
        similarity = nlp.similarity(step.text, prev_context)
        
        if similarity >= REDUNDANCY_SIM:
            flags.append(Flag(
                pillar="utility",
                step=f"step={step.idx + 1}",
                issue="redundant_step",
                details={
                    "sim": similarity,
                    "used_later": False,
                    "delta": 0.0
                }
            ))
    
    return flags


def check_off_topic(item: EvalItem) -> List[Flag]:
    """
    Check for off-topic steps.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of off-topic flags
    """
    flags = []
    
    # Get NLP utils
    nlp = get_nlp_utils()
    
    # Check each step for topicality
    for step in item.steps:
        # Skip very short steps (likely parsing artifacts)
        if len(step.text.strip()) < 15:
            continue
            
        # Check if step is off-topic relative to problem
        is_offtopic = nlp.is_off_topic(step.text, item.problem.prompt_text)
        
        if is_offtopic:
            flags.append(Flag(
                pillar="utility",
                step=f"step={step.idx + 1}",
                issue="off_topic",
                details={
                    "jaccard": nlp.similarity(step.text, item.problem.prompt_text),
                    "used_later": False,
                    "judge_used": False
                }
            ))
    
    return flags
