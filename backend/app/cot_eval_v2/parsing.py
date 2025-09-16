"""
Text parsing utilities for extracting steps, equations, and entities.
"""

import re
from typing import List, Tuple, Optional
from .schema import Step


def split_steps(text: str) -> List[str]:
    """
    Split text into reasoning steps using multiple strategies.
    
    Args:
        text: The full reasoning text
        
    Returns:
        List of step text strings
    """
    if not text or not text.strip():
        return []
    
    # Strategy 1: Numbered lists
    numbered_pattern = r'\n\d+\.\s*'
    numbered_steps = re.split(numbered_pattern, text)
    
    if len(numbered_steps) > 1:
        steps = []
        for i, step_text in enumerate(numbered_steps):
            if step_text.strip():
                steps.append(step_text.strip())
        return steps
    
    # Strategy 2: Bullet points
    bullet_pattern = r'\n[\*\-\•]\s*'
    bullet_steps = re.split(bullet_pattern, text)
    
    if len(bullet_steps) > 1:
        steps = []
        for i, step_text in enumerate(bullet_steps):
            if step_text.strip():
                steps.append(step_text.strip())
        return steps
    
    # Strategy 3: Logical connectors (more conservative)
    connector_patterns = [
        r'\b(therefore|thus|hence|consequently|additionally|furthermore|moreover)\b',
        r'\b(first|second|third|last|finally)\b'
    ]
    
    for pattern in connector_patterns:
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        if len(parts) > 1:
            steps = []
            for i, part in enumerate(parts):
                part = part.strip()
                if part and len(part) > 20:  # Minimum length threshold
                    # Add connector back if not first part
                    if i > 0:
                        connector_match = re.search(pattern, text, flags=re.IGNORECASE)
                        if connector_match:
                            connector = connector_match.group(0)
                            part = connector + ' ' + part
                    steps.append(part)
            if len(steps) > 1:
                return steps
    
    # Fallback: single step
    return [text.strip()]


def parse_step(idx: int, text: str) -> Step:
    """
    Parse a single step text into a Step object.
    
    Args:
        idx: Step index
        text: Step text
        
    Returns:
        Step object with parsed content
    """
    equations = extract_equations(text)
    numbers = extract_numbers(text)
    symbols = extract_symbols(text)
    units = extract_units(text)
    mentions = extract_mentions(text)
    ops = extract_operations(text)
    
    return Step(
        idx=idx,
        text=text,
        equations=equations,
        numbers=numbers,
        symbols=symbols,
        units=units,
        mentions=mentions,
        ops=ops
    )


def extract_equations(text: str) -> List[str]:
    """Extract equations in the form 'lhs = rhs'."""
    # Pattern for equations: variable = expression
    pattern = r'(?P<lhs>\b[A-Za-z]\w*\b)\s*=\s*(?P<rhs>[^;\n]+)'
    matches = re.findall(pattern, text)
    return [f"{lhs} = {rhs.strip()}" for lhs, rhs in matches]


def extract_numbers(text: str) -> List[float]:
    """Extract numeric literals from text."""
    # Pattern for numbers (including decimals and scientific notation)
    pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers


def extract_symbols(text: str) -> List[str]:
    """Extract symbol tokens (variables, identifiers)."""
    # Pattern for identifiers (letters followed by alphanumeric)
    pattern = r'\b[A-Za-z]\w*\b'
    matches = re.findall(pattern, text)
    
    # Filter out common stopwords and short words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'so', 'then', 'now', 'also',
        'next', 'first', 'second', 'third', 'last', 'finally', 'therefore', 'thus', 'hence',
        'consequently', 'because', 'since', 'as', 'if', 'when', 'where', 'how', 'why', 'what',
        'which', 'who', 'whom', 'whose', 'let', 'denote', 'represent', 'means', 'equals'
    }
    
    symbols = []
    for match in matches:
        if len(match) > 1 and match.lower() not in stopwords:
            symbols.append(match)
    
    return list(set(symbols))  # Remove duplicates


def extract_units(text: str) -> List[str]:
    """Extract units from text."""
    # Pattern for units (common measurement units)
    unit_patterns = [
        r'\b(\d+(?:\.\d+)?)\s*(dollars?|cents?|pounds?|euros?|yen)\b',
        r'\b(\d+(?:\.\d+)?)\s*(feet|inches|meters?|yards?|miles?|kilometers?)\b',
        r'\b(\d+(?:\.\d+)?)\s*(hours?|minutes?|seconds?|days?|weeks?|months?|years?)\b',
        r'\b(\d+(?:\.\d+)?)\s*(pounds?|kilograms?|grams?|tons?)\b',
        r'\b(\d+(?:\.\d+)?)\s*(percent|%)\b'
    ]
    
    units = []
    for pattern in unit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) == 2:
                units.append(match[1])
    
    return units


def extract_mentions(text: str) -> List[str]:
    """Extract key noun phrases and entities."""
    # Simple noun phrase extraction
    # Pattern for noun phrases (determiner + adjective* + noun)
    pattern = r'\b(?:the|a|an)\s+(?:[a-zA-Z]+\s+)*[a-zA-Z]+\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Also extract capitalized words (proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    mentions = []
    for match in matches:
        # Clean up the mention
        mention = re.sub(r'^(the|a|an)\s+', '', match, flags=re.IGNORECASE).strip()
        if len(mention) > 2:
            mentions.append(mention)
    
    mentions.extend(proper_nouns)
    return list(set(mentions))  # Remove duplicates


def extract_operations(text: str) -> List[str]:
    """Extract mathematical and logical operations."""
    ops = []
    
    # Mathematical operations
    if '+' in text:
        ops.append('+')
    if '-' in text:
        ops.append('-')
    if '*' in text or '×' in text:
        ops.append('*')
    if '/' in text or '÷' in text:
        ops.append('/')
    if '=' in text:
        ops.append('=')
    
    # Comparison operations
    if '<=' in text or '≤' in text:
        ops.append('<=')
    elif '<' in text:
        ops.append('<')
    if '>=' in text or '≥' in text:
        ops.append('>=')
    elif '>' in text:
        ops.append('>')
    
    return ops


def extract_final_answer(text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract final answer from text.
    
    Returns:
        Tuple of (answer_text, normalized_numeric_value)
    """
    # Look for final answer patterns (order matters - more specific patterns first)
    patterns = [
        r'####\s*([^\n]+)',  # Common CoT format: #### 18
        r'\\boxed\{([^}]+)\}',  # LaTeX format: \boxed{18}
        r'\\boxed\s*([^\n]+)',  # LaTeX format without braces: \boxed 18
        r'the answer is\s*:?\s*([^\n]+)',
        r'answer\s*:?\s*([^\n]+)',
        r'final answer\s*:?\s*([^\n]+)',
        r'therefore\s*,?\s*([^\n]+)',
        r'so\s+there\s+are\s+([^\n]+)',
        r'thus\s*,?\s*([^\n]+)',
        r'hence\s*,?\s*([^\n]+)',
        r'consequently\s*,?\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_text = match.group(1).strip()
            
            # Try to extract numeric value
            numeric_value = None
            try:
                # Look for numbers in the answer
                numbers = extract_numbers(answer_text)
                if numbers:
                    numeric_value = numbers[-1]  # Take the last number found
            except (ValueError, TypeError):
                pass
            
            return answer_text, numeric_value
    
    # Fallback: look for numbers at the end
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        numbers = extract_numbers(last_line)
        if numbers:
            return last_line, numbers[-1]
    
    return None, None
