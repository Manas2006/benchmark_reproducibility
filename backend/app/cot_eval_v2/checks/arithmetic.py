"""
Arithmetic checking for CoT evaluation.

This module provides functions to validate arithmetic expressions
in reasoning steps using SymPy for safe mathematical evaluation.
"""

import re
from typing import Dict, List, Any, Optional

try:
    import sympy
    from sympy import sympify, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sympy = None
    sympify = None
    N = None


def check_step_equations(step: str) -> Dict[str, Any]:
    """
    Parse and validate simple arithmetic equations in a reasoning step.
    
    Looks for patterns like "3+4=7", "7*8=56", supports parentheses and decimals.
    Uses SymPy for safe mathematical evaluation.
    
    Args:
        step: The reasoning step text to analyze
        
    Returns:
        Dictionary with:
        - "ok": number of correct equations found
        - "bad": number of incorrect equations found  
        - "examples": list of equation details with errors if any
        
    Raises:
        ImportError: If SymPy is not available
    """
    if not SYMPY_AVAILABLE:
        raise ImportError(
            "SymPy is required for arithmetic checking. "
            "Install with: pip install sympy"
        )
    
    # Pattern to match equations: expression = result
    # Look for patterns like "3 + 4 = 7" or "2 * 5 = 10"
    # This pattern captures the full left side expression
    equation_pattern = r'(\d+\s*[+\-*/]\s*\d+)\s*=\s*(\d+)'
    
    equations = re.findall(equation_pattern, step)
    
    ok_count = 0
    bad_count = 0
    examples = []
    
    for left_expr, right_expr in equations:
        try:
            # Parse both sides of the equation
            left_sympy = sympify(left_expr)
            right_sympy = sympify(right_expr)
            
            # Evaluate both sides numerically
            left_val = float(N(left_sympy))
            right_val = float(N(right_sympy))
            
            # Check if they're equal (with small tolerance for floating point)
            is_correct = abs(left_val - right_val) < 1e-10
            
            example = {
                "expr": f"{left_expr} = {right_expr}",
                "lhs": left_expr,
                "rhs": right_expr,
                "lhs_val": left_val,
                "rhs_val": right_val,
                "correct": is_correct
            }
            
            if is_correct:
                ok_count += 1
            else:
                bad_count += 1
                example["error"] = f"Expected {left_val}, got {right_val}"
            
            examples.append(example)
            
        except Exception as e:
            # If we can't parse or evaluate, count as bad
            bad_count += 1
            examples.append({
                "expr": f"{left_expr} = {right_expr}",
                "lhs": left_expr,
                "rhs": right_expr,
                "error": f"Parse error: {str(e)}",
                "correct": False
            })
    
    return {
        "ok": ok_count,
        "bad": bad_count,
        "examples": examples
    }


def extract_arithmetic_expressions(text: str) -> List[str]:
    """
    Extract all arithmetic expressions from text.
    
    Args:
        text: Text to search for arithmetic expressions
        
    Returns:
        List of arithmetic expressions found
    """
    # Pattern for arithmetic expressions (numbers, operators, parentheses)
    pattern = r'[0-9]+(?:\.[0-9]+)?(?:[+\-*/()][0-9]+(?:\.[0-9]+)?)+'
    return re.findall(pattern, text)


def is_safe_arithmetic(expression: str) -> bool:
    """
    Check if an arithmetic expression is safe to evaluate.
    
    Args:
        expression: The expression to check
        
    Returns:
        True if the expression appears safe for evaluation
    """
    # Only allow numbers, basic operators, parentheses, and spaces
    safe_chars = set('0123456789+-*/()=. ')
    return all(c in safe_chars for c in expression)


def validate_arithmetic_safely(expression: str) -> Optional[float]:
    """
    Safely validate and evaluate an arithmetic expression.
    
    Args:
        expression: The arithmetic expression to evaluate
        
    Returns:
        The numeric result if valid and safe, None otherwise
    """
    if not SYMPY_AVAILABLE or not is_safe_arithmetic(expression):
        return None
    
    try:
        sympy_expr = sympify(expression)
        return float(N(sympy_expr))
    except Exception:
        return None
