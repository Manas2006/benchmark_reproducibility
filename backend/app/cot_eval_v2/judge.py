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
    
    def __init__(self, model: str = "gpt-4o-mini", mode: str = "ALWAYS", diagnostic: bool = False):
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
        Build the enhanced prompt for the LLM judge using Pillars v2 flags.
        
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
Score the chain-of-thought (CoT) on 4 dimensions using the scoring criteria below.

Each score must be an integer from 1-5 (1 = very poor, 5 = excellent).

## SCORING CRITERIA

### 1. FAITHFULNESS (1-5)
**Definition:** Reasoning is internally consistent, follows logical rules, and stays focused on the problem without hidden shortcuts or leaps.

**Scoring Guidelines:**
- 5: Perfect logical consistency, no contradictions, stays completely on-topic
- 4: Minor inconsistencies or slight tangents, but overall coherent
- 3: Some logical gaps or moderate off-topic content
- 2: Significant logical flaws or frequent tangents
- 1: Major contradictions, illogical leaps, or completely off-topic

**Dock Points For:**
- Contradictory statements within the reasoning
- Logical leaps without justification
- Going off-topic or discussing irrelevant matters
- Hidden assumptions not stated explicitly
- Unjustified final answers (not derivable from steps)
- Shortcut reasoning (non-contributing steps)

### 2. UTILITY (1-5)
**Definition:** Each step meaningfully contributes to solving the problem, calculations are correct, and reasoning efficiently leads to the final answer.

**Scoring Guidelines:**
- 5: Every step is necessary and correct, efficient path to solution
- 4: Most steps useful, minor inefficiencies or small errors
- 3: Some useful steps mixed with unnecessary ones or calculation errors
- 2: Many unnecessary steps or significant calculation errors
- 1: Mostly useless steps, major calculation errors, or repetitive content

**Dock Points For:**
- Incorrect calculations or mathematical errors
- Repetitive statements that don't add value
- Unnecessary verbose explanations
- Steps that don't advance toward the solution
- Redundant reasoning or circular logic
- Off-topic steps that don't contribute

### 3. COHERENCE (1-5)
**Definition:** Steps flow smoothly from one to the next with clear logical progression and smooth transitions.

**Scoring Guidelines:**
- 5: Perfect flow, each step naturally follows from the previous
- 4: Good flow with minor awkward transitions
- 3: Some disjointed steps but overall progression
- 2: Choppy flow with unclear connections between steps
- 1: Disjointed, random steps with no clear progression

**Dock Points For:**
- Abrupt transitions between ideas
- Missing connecting logic between steps
- Disjointed or random sequence of reasoning
- Poor organization of thoughts
- Dangling references (use-before-define)
- Disordered reasoning chain

### 4. FACTUALITY (1-5)
**Definition:** Every step must be factually correct and grounded in the problem context, not hallucinated from surface-level understanding.

**Scoring Guidelines:**
- 5: All facts and statements are accurate and grounded in the problem
- 4: Mostly accurate with minor factual errors
- 3: Some factual errors or unsupported claims
- 2: Multiple factual errors or significant hallucinations
- 1: Major factual errors, hallucinations, or completely unsupported claims

**Dock Points For:**
- Hallucinated facts not present in the problem
- Incorrect interpretations of given information
- Making assumptions not supported by the problem context
- Surface-level understanding leading to wrong facts
- Stating things as facts that are actually assumptions
- Claims that contradict the problem evidence

## EVALUATION PROCESS
1. Read the problem carefully to understand the context and given information
2. Analyze each step of the CoT reasoning
3. Check each step against the four criteria above
4. Assign scores based on the specific guidelines for each dimension
5. Ensure every step is evaluated for factual accuracy and logical soundness

## Problem
{problem}

## Model Reasoning (CoT)
{cot}

## Gold Answer
{gold}

## Automated Flag Analysis
{flags_summary}

## Evidence Summary
{json.dumps(evidence, indent=2)}

## Instructions
Based on the automated flag analysis above, carefully evaluate the reasoning. The flags highlight specific issues that should influence your scoring:

- **Faithfulness flags** indicate logical inconsistencies, unjustified conclusions, or shortcut reasoning
- **Utility flags** point to redundant, off-topic, or non-contributing steps
- **Coherence flags** reveal structural problems like dangling references or disordered chains
- **Factuality flags** identify unsupported claims or contradictions with the problem

Use these flags as guidance, but apply your own judgment to determine the final scores. Consider the severity and impact of each flagged issue.

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
        # Initialize parent without calling OpenAI
        self.model = kwargs.get('model', 'gpt-4o-mini')
        self.mode = kwargs.get('mode', 'ALWAYS').upper()
        self.diagnostic = kwargs.get('diagnostic', False)
        
        # Validate mode
        if self.mode not in ["SMART", "ALWAYS", "NEVER"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be SMART, ALWAYS, or NEVER")
        
        # Don't initialize OpenAI client for mock
        self.client = None
        
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
