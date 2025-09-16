"""
Flag class definition to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Flag:
    """Represents a single evaluation flag/issue."""
    pillar: str
    step: str
    issue: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flag to dictionary for JSON serialization."""
        result = {
            "pillar": self.pillar,
            "step": self.step,
            "issue": self.issue
        }
        if self.details is not None:
            result["details"] = self.details
        return result
