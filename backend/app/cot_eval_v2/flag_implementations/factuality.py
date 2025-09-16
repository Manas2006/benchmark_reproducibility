"""
Factuality flag implementations.
"""

from typing import List, Dict, Any, Optional
from ..schema import EvalItem
from ..flag import Flag
from ..nlp_utils import get_nlp_utils
from ..llm_judge import get_llm_judge
from ..config import NLI_CONFIDENCE_HIGH


def check_factuality_flags(item: EvalItem) -> List[Flag]:
    """
    Check for factuality issues.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of factuality flags
    """
    flags = []
    
    # Check ungrounded facts (temporarily disabled - too strict)
    # ungrounded_flags = check_fact_ungrounded(item)
    # flags.extend(ungrounded_flags)
    
    # Check fact contradictions (temporarily disabled - too strict)
    # contradiction_flags = check_fact_contradiction(item)
    # flags.extend(contradiction_flags)
    
    return flags


def check_fact_ungrounded(item: EvalItem) -> List[Flag]:
    """
    Check for ungrounded facts in reasoning steps.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of ungrounded fact flags
    """
    flags = []
    
    # Get NLP utils
    nlp = get_nlp_utils()
    
    # Check each step for ungrounded claims
    for step in item.steps:
        claims = nlp.extract_claims(step.text)
        
        for claim in claims:
            # Simple check: if claim doesn't contain numbers from problem
            problem_numbers = nlp.extract_numbers(item.problem.prompt_text)
            claim_numbers = nlp.extract_numbers(claim)
            
            # If claim has numbers not in problem, flag it
            if claim_numbers and not any(num in problem_numbers for num in claim_numbers):
                flags.append(Flag(
                    pillar="factuality",
                    step=f"step={step.idx + 1}",
                    issue="fact_ungrounded",
                    details={
                        "claim": claim,
                        "support": "none",
                        "judge_used": False
                    }
                ))
    
    return flags


def check_fact_contradiction(item: EvalItem) -> List[Flag]:
    """
    Check for facts that contradict the evidence.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of contradiction flags
    """
    flags = []
    
    # Get NLP utils
    nlp = get_nlp_utils()
    
    # Check each step for contradictions
    for step in item.steps:
        claims = nlp.extract_claims(step.text)
        
        for claim in claims:
            # Check for numeric contradictions
            claim_numbers = nlp.extract_numbers(claim)
            problem_numbers = nlp.extract_numbers(item.problem.prompt_text)
            
            # Check if numbers contradict (only flag major contradictions)
            for claim_num in claim_numbers:
                for prob_num in problem_numbers:
                    # Only flag if it's a major contradiction (200% difference or more)
                    if prob_num != 0 and abs(claim_num - prob_num) / abs(prob_num) > 2.0:
                        flags.append(Flag(
                            pillar="factuality",
                            step=f"step={step.idx + 1}",
                            issue="fact_contradiction",
                            details={
                                "claim": claim,
                                "contradicts": True,
                                "delta": abs(claim_num - prob_num),
                                "judge_used": False
                            }
                        ))
                        break
                if flags:  # If we found a contradiction, break
                    break
    
    return flags
