"""
Factuality flag implementations.
"""

from typing import List
from ..schema import EvalItem
from ..flag import Flag


def check_factuality_flags(item: EvalItem) -> List[Flag]:
    """
    Check for factuality issues.
    
    Args:
        item: Evaluation item
        
    Returns:
        List of factuality flags
    """
    flags = []
    
    return flags
