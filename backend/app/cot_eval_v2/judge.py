"""
LLM Judge integration for CoT evaluation v2.

This module provides GPT-4o-mini as a judge for reasoning quality,
with configurable modes and robust JSON parsing.
"""

import json
import re
from typing import Dict, Any, Optional
import warnings


class Judge:
    """
    LLM Judge for CoT evaluation using GPT-4o-mini.
    
    Supports three modes:
    - SMART: Call judge only when flags/evidence suggest issues
    - ALWAYS: Always call judge regardless of flags
    - NEVER: Never call judge (rules+DeBERTa only)
    
    Supports diagnostic mode for debugging with explanations.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", mode: str = "SMART", diagnostic: bool = False):
        """
        Initialize the judge.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            mode: Judge calling mode - SMART, ALWAYS, or NEVER
            diagnostic: If True, include explanations in output (for debugging)
        """
        self.model = model
        self.mode = mode.upper()
        self.diagnostic = diagnostic
        
        # Validate mode
        if self.mode not in ["SMART", "ALWAYS", "NEVER"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be SMART, ALWAYS, or NEVER")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            warnings.warn("OpenAI library not available. Judge will not work.")
            self.client = None
    
    def build_prompt(self, problem: str, cot: str, gold: str, flags_summary: str, evidence: Dict[str, Any]) -> str:
        """
        Build the prompt for the LLM judge.
        
        Args:
            problem: The original problem text
            cot: The chain-of-thought reasoning text
            gold: The ground truth answer
            flags_summary: Human-readable summary of flags
            evidence: Evidence dictionary from PillarsEvaluator
            
        Returns:
            Formatted prompt string
        """
        rationale_instructions = (
            "First, briefly explain your reasoning for each dimension (1–2 sentences per dimension). Then output the JSON object.\n"
            if self.diagnostic else
            "Do NOT include explanations. Output only the JSON object.\n"
        )
        
        return f"""You are an expert evaluator of mathematical and logical reasoning. 
Score the chain-of-thought (CoT) on 4 dimensions:
- Faithfulness: reasoning is consistent with itself, no hidden shortcuts or leaps
- Utility: steps are useful toward solving the problem and lead to the final answer
- Coherence: steps follow logically from each other, no contradictions
- Factuality: steps are factually correct and grounded in the problem context

Each score must be an integer from 1–5 (1 = very poor, 5 = excellent).

## Problem
{problem}

## Model Reasoning (CoT)
{cot}

## Gold Answer
{gold}

## Code-Based Evidence & Flags
{flags_summary}

Evidence JSON:
{json.dumps(evidence, indent=2)}

{rationale_instructions}

Required JSON schema:
{{
  "faithfulness": <1-5>,
  "utility": <1-5>,
  "coherence": <1-5>,
  "factuality": <1-5>
}}"""

    def score(self, problem: str, cot: str, gold: str, flags_summary: str, evidence: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """
        Get LLM judge scores for the reasoning.
        
        Args:
            problem: The original problem text
            cot: The chain-of-thought reasoning text
            gold: The ground truth answer
            flags_summary: Human-readable summary of flags
            evidence: Evidence dictionary from PillarsEvaluator
            
        Returns:
            Dictionary with scores for each pillar (1-5) or None if judge not called
        """
        # Handle NEVER mode
        if self.mode == "NEVER":
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        
        # Handle SMART mode - skip judge if no issues and correct answer
        if self.mode == "SMART":
            has_flags = flags_summary.strip() and flags_summary != "No issues detected."
            final_correct = evidence.get("final_correct", False)
            
            # Skip judge if no flags and correct answer
            if not has_flags and final_correct:
                return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        
        # Check if OpenAI client is available
        if self.client is None:
            warnings.warn("OpenAI client not available. Returning None scores.")
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        
        # Build prompt and call OpenAI
        prompt = self.build_prompt(problem, cot, gold, flags_summary, evidence)
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a careful and consistent evaluator of reasoning quality."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1000
            )
            
            raw_output = resp.choices[0].message.content
            
            # Extract JSON using guard clause
            return self._extract_json_safely(raw_output)
            
        except Exception as e:
            warnings.warn(f"OpenAI API call failed: {str(e)}")
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
    
    def _extract_json_safely(self, raw_output: str) -> Dict[str, Optional[int]]:
        """
        Extract JSON from raw LLM output with robust parsing.
        
        Args:
            raw_output: Raw text output from LLM
            
        Returns:
            Dictionary with parsed scores or None values if parsing fails
        """
        # Try direct JSON parsing first
        try:
            parsed = json.loads(raw_output)
            if self._validate_scores(parsed):
                return parsed
        except Exception:
            pass
        
        # Try to find JSON object in the text using regex
        json_pattern = r'\{[^{}]*"faithfulness"[^{}]*"utility"[^{}]*"coherence"[^{}]*"factuality"[^{}]*\}'
        matches = re.findall(json_pattern, raw_output, re.IGNORECASE | re.DOTALL)
        
        if matches:
            try:
                parsed = json.loads(matches[-1])  # Use last match
                if self._validate_scores(parsed):
                    return parsed
            except Exception:
                pass
        
        # Try more flexible JSON extraction
        try:
            # Look for any JSON object
            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_output)
            for json_str in reversed(json_objects):  # Try last match first
                try:
                    parsed = json.loads(json_str)
                    if self._validate_scores(parsed):
                        return parsed
                except Exception:
                    continue
        except Exception:
            pass
        
        # If all parsing fails, return None scores
        warnings.warn(f"Failed to parse JSON from LLM output: {raw_output[:200]}...")
        return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
    
    @staticmethod
    def validate_and_normalize_judge(d: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """
        Validate and normalize judge scores to 1-5 range.
        
        Args:
            d: Raw dictionary from judge output
            
        Returns:
            Normalized dictionary with integer scores 1-5 or None
        """
        keys = ["faithfulness", "utility", "coherence", "factuality"]
        out = {}
        
        for k in keys:
            v = d.get(k, None)
            try:
                # Try to convert to int
                v = int(v)
            except (ValueError, TypeError):
                v = None
            
            # Clamp to valid range
            if v is not None:
                v = min(5, max(1, v))
            
            out[k] = v
        
        return out
    
    def _validate_scores(self, parsed: Dict[str, Any]) -> bool:
        """
        Validate that parsed scores are in correct format.
        
        Args:
            parsed: Parsed dictionary from JSON
            
        Returns:
            True if scores are valid, False otherwise
        """
        required_keys = ["faithfulness", "utility", "coherence", "factuality"]
        
        # Check all required keys are present
        if not all(key in parsed for key in required_keys):
            return False
        
        # Check all values are integers 1-5
        for key in required_keys:
            value = parsed[key]
            if not isinstance(value, int) or value < 1 or value > 5:
                return False
        
        return True
    
    def should_call_judge(self, flags_summary: str, evidence: Dict[str, Any]) -> bool:
        """
        Determine if judge should be called based on mode and evidence.
        
        Args:
            flags_summary: Human-readable summary of flags
            evidence: Evidence dictionary
            
        Returns:
            True if judge should be called, False otherwise
        """
        if self.mode == "ALWAYS":
            return True
        elif self.mode == "NEVER":
            return False
        elif self.mode == "SMART":
            has_flags = flags_summary.strip() and flags_summary != "No issues detected."
            final_correct = evidence.get("final_correct", False)
            return has_flags or not final_correct
        else:
            return False


class MockJudge(Judge):
    """
    Mock judge for testing without OpenAI API calls.
    """
    
    def __init__(self, mock_scores: Optional[Dict[str, int]] = None, **kwargs):
        """
        Initialize mock judge.
        
        Args:
            mock_scores: Fixed scores to return (for testing)
            **kwargs: Other arguments passed to parent
        """
        super().__init__(**kwargs)
        self.mock_scores = mock_scores or {
            "faithfulness": 4,
            "utility": 4, 
            "coherence": 4,
            "factuality": 4
        }
    
    def score(self, problem: str, cot: str, gold: str, flags_summary: str, evidence: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """Return mock scores for testing."""
        if self.mode == "NEVER":
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        
        if self.mode == "SMART" and not self.should_call_judge(flags_summary, evidence):
            return {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        
        return self.mock_scores.copy()
