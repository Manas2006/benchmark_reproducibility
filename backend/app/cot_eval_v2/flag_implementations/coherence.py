"""
Coherence flag implementations.
"""

from typing import List, Dict, Any, Optional, Set
from ..schema import EvalItem
from ..flag import Flag
from ..nlp_utils import get_nlp_utils
from ..llm_judge import get_llm_judge


def check_coherence_flags(item: EvalItem) -> List[Flag]:
    """
    Check for coherence issues.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of coherence flags
    """
    flags = []
    
    # Check dangling references (temporarily disabled - too strict)
    # dangling_flags = check_dangling_reference(item)
    # flags.extend(dangling_flags)
    
    # Check disordered chain (temporarily disabled - too strict)
    # disorder_flags = check_disordered_chain(item)
    # flags.extend(disorder_flags)
    
    return flags


def check_dangling_reference(item: EvalItem) -> List[Flag]:
    """
    Check for dangling references (use-before-define).
    
    Args:
        item: Evaluation item
        
    Returns:
        List of dangling reference flags
    """
    flags = []
    
    # Track defined symbols and entities
    defined_symbols = set()
    defined_entities = set()
    
    # Add symbols from problem context
    nlp = get_nlp_utils()
    problem_mentions = nlp.extract_mentions(item.problem.prompt_text)
    defined_entities.update(problem_mentions)
    
    # Check each step for dangling references
    for step in item.steps:
        # Check for undefined symbols
        undefined_symbols = []
        for symbol in step.symbols:
            if symbol not in defined_symbols:
                undefined_symbols.append(symbol)
        
        # Check for undefined entities (only flag proper nouns and technical terms)
        undefined_entities = []
        step_mentions = nlp.extract_mentions(step.text)
        for mention in step_mentions:
            # Only flag if it's a proper noun or technical term (capitalized or contains numbers)
            if (mention not in defined_entities and 
                (mention[0].isupper() or any(c.isdigit() for c in mention))):
                undefined_entities.append(mention)
        
        # Only flag if there are significant undefined references
        if undefined_symbols or (undefined_entities and len(undefined_entities) > 2):
            flags.append(Flag(
                pillar="coherence",
                step=f"step={step.idx + 1}",
                issue="dangling_reference",
                details={
                    "unknown_tokens": undefined_symbols + undefined_entities,
                    "step": step.idx
                }
            ))
        
        # Update defined symbols and entities
        defined_symbols.update(step.symbols)
        defined_entities.update(step_mentions)
    
    return flags


def check_disordered_chain(item: EvalItem) -> List[Flag]:
    """
    Check for disordered reasoning chain.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of disordered chain flags
    """
    flags = []
    
    if len(item.steps) < 2:
        return flags
    
    # Simple check: if steps don't have logical connectors
    for i, step in enumerate(item.steps):
        if i == 0:
            continue  # First step doesn't need connectors
        
        # Check if step has logical connectors
        connectors = ['so', 'therefore', 'thus', 'hence', 'then', 'next', 'also', 'additionally']
        has_connector = any(connector in step.text.lower() for connector in connectors)
        
        if not has_connector:
            flags.append(Flag(
                pillar="coherence",
                step=f"step={step.idx + 1}",
                issue="disordered_chain",
                details={
                    "no_premise": True,
                    "future_dep": False,
                    "judge_used": False
                }
            ))
    
    return flags
