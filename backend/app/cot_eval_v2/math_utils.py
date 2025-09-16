"""
Mathematical utilities for safe evaluation and comparison.
"""

import re
from typing import Dict, Any, Optional
from .config import FLOAT_TOL

try:
    import sympy
    from sympy import sympify, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sympy = None
    sympify = None
    N = None


def safe_eval(expr: str, env: Dict[str, float]) -> Optional[float]:
    """
    Safely evaluate a mathematical expression using SymPy.
    
    Args:
        expr: Mathematical expression string
        env: Environment with variable values
        
    Returns:
        Evaluated result or None if evaluation fails
    """
    if not SYMPY_AVAILABLE:
        return None
    
    try:
        # Clean the expression
        expr = expr.strip()
        
        # Replace variables with their values
        for var, value in env.items():
            expr = expr.replace(var, str(value))
        
        # Parse and evaluate
        sympy_expr = sympify(expr)
        result = float(N(sympy_expr))
        
        return result
    except Exception:
        return None


def compare_floats(a: float, b: float, tol: float = FLOAT_TOL) -> bool:
    """
    Compare two floats with tolerance.
    
    Args:
        a: First number
        b: Second number
        tol: Tolerance for comparison
        
    Returns:
        True if numbers are equal within tolerance
    """
    return abs(a - b) <= tol


def is_safe_expression(expr: str) -> bool:
    """
    Check if expression is safe to evaluate.
    
    Args:
        expr: Expression string
        
    Returns:
        True if expression appears safe
    """
    # Only allow safe characters
    safe_chars = set('0123456789+-*/()=.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ ')
    
    if not all(c in safe_chars for c in expr):
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'import\s+',
        r'__\w+__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expr, re.IGNORECASE):
            return False
    
    return True


def normalize_number(text: str) -> Optional[float]:
    """
    Normalize a number from text, handling various formats.
    
    Args:
        text: Text containing a number
        
    Returns:
        Normalized float or None if no valid number found
    """
    # Remove common prefixes/suffixes
    text = text.strip()
    text = re.sub(r'^\$', '', text)  # Remove dollar sign
    text = re.sub(r'%$', '', text)   # Remove percent sign
    text = re.sub(r',', '', text)    # Remove commas
    
    # Extract number
    number_match = re.search(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
    if number_match:
        try:
            return float(number_match.group(0))
        except ValueError:
            pass
    
    return None
