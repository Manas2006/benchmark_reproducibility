"""
Coverage checking for CoT evaluation.

This module provides functions to check whether all given numbers
in a problem are used in the reasoning steps.
"""

import re
from typing import Dict, List, Set, Optional, Any


def number_coverage(problem: str, steps: List[str]) -> Dict[str, List[str]]:
    """
    Extract numbers from problem and steps to check coverage.
    
    Args:
        problem: The original problem text
        steps: List of reasoning step texts
        
    Returns:
        Dictionary with:
        - "given": list of numbers found in problem
        - "used": list of numbers found in steps  
        - "unused": list of numbers from problem not found in steps
    """
    # Extract numbers from problem
    given_numbers = extract_numbers(problem)
    
    # Extract numbers from all steps
    used_numbers = set()
    for step in steps:
        step_numbers = extract_numbers(step)
        used_numbers.update(step_numbers)
    
    # Find unused numbers
    unused_numbers = []
    for num in given_numbers:
        if not is_number_used(num, used_numbers):
            unused_numbers.append(num)
    
    return {
        "given": given_numbers,
        "used": list(used_numbers),
        "unused": unused_numbers
    }


def extract_numbers(text: str) -> List[str]:
    """
    Extract all numbers from text using regex.
    
    Supports:
    - Integers: 42, -17
    - Decimals: 3.14, -2.5, .5, 5.
    - Scientific notation: 1e5, 2.3e-4 (optional)
    
    Excludes step numbers like "1." at the beginning of lines.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of number strings found
    """
    # Pattern for numbers: optional minus, digits, optional decimal part
    # Case-insensitive matching
    pattern = r'-?\d+\.?\d*'
    
    # Find all matches
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Filter out step numbers and empty strings
    numbers = []
    for match in matches:
        if match.strip():  # Skip empty matches
            # Skip step numbers (like "1." at start of line)
            if re.match(r'^\d+\.$', match.strip()):
                continue
                
            # Normalize the number string
            normalized = normalize_number_string(match)
            if normalized:
                numbers.append(normalized)
    
    return numbers


def normalize_number_string(num_str: str) -> Optional[str]:
    """
    Normalize a number string for comparison.
    
    Args:
        num_str: Raw number string from regex
        
    Returns:
        Normalized number string or None if invalid
    """
    if not num_str or not num_str.strip():
        return None
    
    # Remove leading/trailing whitespace
    num_str = num_str.strip()
    
    # Handle edge cases
    if num_str in ['.', '-', '-.']:
        return None
    
    # Normalize decimal points
    if num_str.startswith('.'):
        num_str = '0' + num_str
    elif num_str.endswith('.'):
        num_str = num_str[:-1]
    
    # Validate it's a proper number
    try:
        float(num_str)
        return num_str
    except ValueError:
        return None


def is_number_used(target_num: str, used_numbers: Set[str]) -> bool:
    """
    Check if a target number is used in the set of used numbers.
    
    Uses fuzzy matching to handle different representations of the same number.
    
    Args:
        target_num: The number to check for
        used_numbers: Set of numbers found in reasoning steps
        
    Returns:
        True if the number is considered used
    """
    try:
        target_val = float(target_num)
    except ValueError:
        return False
    
    for used_num in used_numbers:
        try:
            used_val = float(used_num)
            # Check if they're numerically equal (with small tolerance)
            if abs(target_val - used_val) < 1e-10:
                return True
        except ValueError:
            # If we can't convert to float, do string comparison
            if target_num == used_num:
                return True
    
    return False


def get_number_context(text: str, number: str) -> List[str]:
    """
    Get context around a number in text.
    
    Args:
        text: The text to search in
        number: The number to find context for
        
    Returns:
        List of context strings around the number
    """
    # Escape special regex characters in number
    escaped_number = re.escape(number)
    
    # Find all occurrences of the number
    pattern = rf'\b{escaped_number}\b'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    contexts = []
    for match in matches:
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        context = text[start:end].strip()
        contexts.append(context)
    
    return contexts


def analyze_number_usage(problem: str, steps: List[str]) -> Dict[str, Any]:
    """
    Comprehensive analysis of number usage in reasoning.
    
    Args:
        problem: The original problem text
        steps: List of reasoning step texts
        
    Returns:
        Detailed analysis of number usage
    """
    coverage = number_coverage(problem, steps)
    
    # Get context for unused numbers
    unused_contexts = {}
    for unused_num in coverage["unused"]:
        contexts = get_number_context(problem, unused_num)
        unused_contexts[unused_num] = contexts
    
    return {
        "coverage": coverage,
        "unused_contexts": unused_contexts,
        "coverage_rate": len(coverage["used"]) / max(1, len(coverage["given"])),
        "unused_count": len(coverage["unused"])
    }
