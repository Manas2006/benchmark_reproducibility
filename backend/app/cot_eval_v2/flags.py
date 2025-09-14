"""
Core data structures for CoT evaluation flags and evidence collection.

This module provides the Flag dataclass and FlagCollector for tracking
issues across different evaluation pillars.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


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


class FlagCollector:
    """
    Collects and manages evaluation flags across different pillars.
    
    Supports four main pillars:
    - faithfulness: Issues with reasoning faithfulness to problem
    - utility: Issues with reasoning utility/effectiveness  
    - coherence: Issues with step-to-step coherence
    - factuality: Issues with factual accuracy
    """
    
    # Supported evaluation pillars
    PILLARS = ["faithfulness", "utility", "coherence", "factuality"]
    
    def __init__(self):
        self._flags: List[Flag] = []
    
    def add(self, pillar: str, step: str, issue: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new flag to the collector.
        
        Args:
            pillar: The evaluation pillar (must be in PILLARS)
            step: Description of which step/part this applies to
            issue: Description of the issue found
            details: Optional additional details about the issue
        """
        if pillar not in self.PILLARS:
            raise ValueError(f"Invalid pillar '{pillar}'. Must be one of: {self.PILLARS}")
        
        flag = Flag(pillar=pillar, step=step, issue=issue, details=details)
        self._flags.append(flag)
    
    def as_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return flags organized by pillar as a JSON-serializable dictionary.
        
        Returns:
            Dictionary with pillar names as keys and lists of flag dictionaries as values
        """
        result = {pillar: [] for pillar in self.PILLARS}
        
        for flag in self._flags:
            result[flag.pillar].append(flag.to_dict())
        
        return result
    
    def summarize_for_prompt(self) -> str:
        """
        Generate a human-readable summary of all flags for use in prompts.
        
        Returns:
            Multi-line string with each flag formatted as:
            - [PILLAR] step=k: issue {details}
        """
        if not self._flags:
            return "No issues detected."
        
        lines = []
        for flag in self._flags:
            line = f"- [{flag.pillar.upper()}] {flag.step}: {flag.issue}"
            if flag.details:
                details_str = ", ".join(f"{k}={v}" for k, v in flag.details.items())
                line += f" {{{details_str}}}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_flags_by_pillar(self, pillar: str) -> List[Flag]:
        """Get all flags for a specific pillar."""
        if pillar not in self.PILLARS:
            raise ValueError(f"Invalid pillar '{pillar}'. Must be one of: {self.PILLARS}")
        
        return [flag for flag in self._flags if flag.pillar == pillar]
    
    def has_flags(self) -> bool:
        """Check if any flags have been collected."""
        return len(self._flags) > 0
    
    def count_by_pillar(self) -> Dict[str, int]:
        """Get count of flags per pillar."""
        counts = {pillar: 0 for pillar in self.PILLARS}
        for flag in self._flags:
            counts[flag.pillar] += 1
        return counts
    
    def clear(self) -> None:
        """Clear all collected flags."""
        self._flags.clear()
    
    def __len__(self) -> int:
        """Return total number of flags collected."""
        return len(self._flags)
    
    def __repr__(self) -> str:
        """String representation showing flag counts by pillar."""
        counts = self.count_by_pillar()
        return f"FlagCollector({counts})"
