"""
Main evaluator orchestration for CoT evaluation v2.

This module provides the PillarsEvaluator class that coordinates
all the deterministic checks and flag collection using the new Pillars v2 system.
"""

import re
import json
from typing import List, Tuple, Dict, Any, Optional

from .schema import EvalItem, Problem, Step
from .flag import Flag
from .parsing import split_steps, parse_step, extract_final_answer
from .flag_implementations import (
    check_faithfulness_flags,
    check_factuality_flags,
    check_coherence_flags,
    check_utility_flags
)
from .flags import FlagCollector
from .scoring import rule_scores, fuse_with_judge


def split_steps_legacy(cot_text: str) -> List[str]:
    """
    Split CoT text into individual reasoning steps (legacy function for compatibility).
    
    Uses multiple strategies:
    1. Numbered lists (1., 2., etc.)
    2. Bullet points (*, -, •)
    3. Sentence splitting with punctuation retention
    
    Args:
        cot_text: The Chain-of-Thought reasoning text
        
    Returns:
        List of individual reasoning steps
    """
    if not cot_text or not cot_text.strip():
        return []
    
    # Strategy 1: Try numbered lists first
    numbered_pattern = r'\n\d+\.\s*'
    numbered_steps = re.split(numbered_pattern, cot_text)
    
    if len(numbered_steps) > 1:
        # Clean up the steps
        steps = [step.strip() for step in numbered_steps if step.strip()]
        if len(steps) > 1:  # Only use if we found multiple steps
            return steps
    
    # Strategy 2: Try bullet points
    bullet_pattern = r'\n[\*\-\•]\s*'
    bullet_steps = re.split(bullet_pattern, cot_text)
    
    if len(bullet_steps) > 1:
        steps = [step.strip() for step in bullet_steps if step.strip()]
        if len(steps) > 1:
            return steps
    
    # Strategy 3: Split by sentences ending with periods
    sentences = re.split(r'\.(?:\s|$)', cot_text)
    steps = [sent.strip() + '.' for sent in sentences if sent.strip()]
    
    # If we only have one step, try splitting by "So" and "Therefore" patterns
    if len(steps) <= 1:
        # Look for logical connectors that indicate step boundaries
        connector_patterns = [
            r'\b(so|therefore|thus|hence|consequently|then|next|finally|also|additionally|furthermore|moreover)\b',
            r'\b(now|next|then|after|before|first|second|third|last|finally)\b'
        ]
        
        # Try to split on these patterns
        for pattern in connector_patterns:
            parts = re.split(pattern, cot_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Reconstruct steps with connectors
                steps = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        # Add connector back if it's not the first part
                        if i > 0:
                            # Find the connector that was used for splitting
                            connector_match = re.search(pattern, cot_text, flags=re.IGNORECASE)
                            if connector_match:
                                connector = connector_match.group(0)
                                part = connector + ' ' + part
                        steps.append(part)
                
                if len(steps) > 1:
                    break
        
        # If still only one step, return as-is
        if len(steps) <= 1:
            return [cot_text.strip()] if cot_text.strip() else []
    
    return steps


def check_final_answer(cot_text: str, gold: Optional[str]) -> bool:
    """
    Extract and check final answer against ground truth.
    
    Args:
        cot_text: The full CoT text including final answer
        gold: Ground truth answer (can be None)
        
    Returns:
        True if final answer matches ground truth
    """
    if not gold:
        return False
    
    # Extract final answer after #### delimiter
    if '####' in cot_text:
        final_answer = cot_text.split('####')[-1].strip()
    else:
        # Fallback: try to find final answer patterns
        # Look for "The answer is", "Therefore", etc.
        # Use more specific patterns to avoid false matches
        patterns = [
            r'the answer is\s*:?\s*([^\n]+)',
            r'therefore\s*,?\s*([^\n]+)',
            r'so\s+there\s+are\s+([^\n]+)',  # More specific for "so there are X"
            r'therefore\s+there\s+are\s+([^\n]+)',  # More specific for "therefore there are X"
            r'thus\s*,?\s*([^\n]+)',
            r'hence\s*,?\s*([^\n]+)',
            r'consequently\s*,?\s*([^\n]+)',
            r'final\s+answer\s*:?\s*([^\n]+)',
            r'answer\s*:?\s*([^\n]+)'
        ]
        
        final_answer = ""
        for pattern in patterns:
            match = re.search(pattern, cot_text, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                break
        
        # If no pattern found, use last line
        if not final_answer:
            lines = cot_text.strip().split('\n')
            final_answer = lines[-1].strip() if lines else ""
    
    # Normalize both answers for comparison
    def normalize_answer(ans):
        if not ans:
            return None
        # Remove whitespace, dollar signs, commas, parentheses
        ans = str(ans).strip().replace('$', '').replace(',', '').replace('(', '').replace(')', '')
        
        # Extract numbers from phrases like "5 chickens on the farm"
        # Look for numbers at the beginning of the answer
        import re
        number_match = re.match(r'^(\d+(?:\.\d+)?)', ans)
        if number_match:
            try:
                num_val = float(number_match.group(1))
                return int(num_val) if num_val.is_integer() else num_val
            except (ValueError, AttributeError):
                pass
        
        try:
            # Try to convert to numeric value
            num_val = float(ans)
            return int(num_val) if num_val.is_integer() else num_val
        except (ValueError, AttributeError):
            # If not a number, return cleaned string in lowercase
            return ans.lower().strip()
    
    norm_final = normalize_answer(final_answer)
    norm_gold = normalize_answer(gold)
    
    # Both must be successfully parsed for comparison
    if norm_final is None or norm_gold is None:
        return False
    
    # Strict numeric comparison with tolerance for floating point errors
    if isinstance(norm_final, (int, float)) and isinstance(norm_gold, (int, float)):
        return abs(norm_final - norm_gold) < 1e-6
    else:
        # String comparison for non-numeric answers
        return str(norm_final) == str(norm_gold)


class PillarsEvaluator:
    """
    Main evaluator that orchestrates all CoT checks across evaluation pillars.
    
    This class uses the new Pillars v2 system for flag detection
    and integrates with the existing GPT scoring infrastructure.
    """
    
    def __init__(
        self, 
        nli_pipe: Optional[Any] = None, 
        encoder: Optional[Any] = None, 
        use_nli: bool = True, 
        judge: Optional[Any] = None,
        nli_model_name: Optional[str] = None
    ):
        """
        Initialize the evaluator with optional external models.
        
        Args:
            nli_pipe: Optional HuggingFace NLI pipeline (deprecated)
            encoder: Optional sentence encoder (deprecated)
            use_nli: If True and nli_pipe is None, auto-load DeBERTa MNLI (deprecated)
            judge: Optional Judge instance for LLM-based evaluation
            nli_model_name: Optional model name override (deprecated)
        """
        # Store old parameters for compatibility but use new system
        self.nli_pipe = nli_pipe
        self.encoder = encoder
        self.judge = judge
        self.use_nli = use_nli
        
        # Auto-load DeBERTa MNLI if requested and no pipe provided
        if use_nli and self.nli_pipe is None:
            try:
                from .checks.nli_factory import create_nli_pipeline
                
                model_name = nli_model_name or "microsoft/deberta-base-mnli"
                print(f"🔄 Auto-loading NLI model: {model_name}")
                
                self.nli_pipe = create_nli_pipeline(model_name)
                print(f"✅ DeBERTa MNLI loaded successfully")
            except Exception as e:
                print(f"Failed to auto-load DeBERTa MNLI: {e}")
                self.nli_pipe = None
    
    def analyze(self, problem: str, cot_text: str, gold: Optional[str] = None) -> Tuple[FlagCollector, Dict[str, Any], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Analyze CoT reasoning using Pillars v2 flag detection and GPT scoring.
        
        Args:
            problem: The original problem text
            cot_text: The Chain-of-Thought reasoning text
            gold: Ground truth answer (optional)
            
        Returns:
            Tuple of (FlagCollector, evidence_dict, rule_scores, judge_scores, fused_scores)
        """
        # Step 1: Parse reasoning into steps
        step_texts = split_steps(cot_text)
        steps = []
        
        for i, step_text in enumerate(step_texts):
            step = parse_step(i, step_text)
            steps.append(step)
        
        # Step 2: Extract final answer
        final_text, final_value = extract_final_answer(cot_text)
        
        # Step 3: Create problem and evaluation item
        problem_obj = Problem(
            prompt_text=problem,
            gold_answer=gold,
            context_docs=[]
        )
        
        item = EvalItem(
            problem=problem_obj,
            steps=steps,
            final_text=final_text or "",
            final_value_norm=final_value
        )
        
        # Step 4: Run all flag checks
        all_flags = []
        
        # Faithfulness flags
        faithfulness_flags = check_faithfulness_flags(item)
        all_flags.extend(faithfulness_flags)
        
        # Factuality flags
        factuality_flags = check_factuality_flags(item)
        all_flags.extend(factuality_flags)
        
        # Coherence flags
        coherence_flags = check_coherence_flags(item)
        all_flags.extend(coherence_flags)
        
        # Utility flags
        utility_flags = check_utility_flags(item)
        all_flags.extend(utility_flags)
        
        # Step 5: Convert to FlagCollector format
        flags = FlagCollector()
        
        for flag in all_flags:
            # Add flag to collector (flag already has the correct step format)
            flags.add(
                pillar=flag.pillar,
                step=flag.step,
                issue=flag.issue,
                details=flag.details
            )
        
        # Step 6: Check final answer correctness
        final_correct = self._check_final_answer(cot_text, gold)
        
        # Step 7: Build evidence dictionary
        evidence = {
            "final_correct": final_correct,
            "intermediate_ok_rate": 1.0,  # Pillars v2 handles this internally
            "coh_contra_cnt": len([f for f in all_flags if f.pillar == 'coherence' and f.issue == 'disordered_chain']),
            "avg_coh_margin": 0.5,  # Placeholder
            "fact_entail_rate": 1.0 - (len([f for f in all_flags if f.pillar == 'factuality']) / max(1, len(all_flags))),
            "fact_contra_cnt": len([f for f in all_flags if f.pillar == 'factuality' and f.issue == 'fact_contradiction']),
            "avg_fact_margin": 0.5,  # Placeholder
            "redund_cnt": len([f for f in all_flags if f.pillar == 'utility' and f.issue == 'redundant_step']),
            "coverage": {"given": [], "used": [], "unused": []},  # Placeholder
            "wrong_but_right": False,  # Placeholder
            "self_repair_cnt": 0,  # Placeholder
            "arith_bad_examples": []  # Placeholder
        }
        
        # Step 8: Compute rule-based scores
        rule_scores_dict = rule_scores(evidence)
        
        # Step 9: Get judge scores if judge is available
        judge_scores = {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        if self.judge is not None:
            flags_summary = flags.summarize_for_prompt()
            judge_scores = self.judge.score(problem, cot_text, gold or "", flags_summary, evidence)
        
        # Step 10: Fuse rule and judge scores
        fused_scores = fuse_with_judge(rule_scores_dict, judge_scores, evidence)
        
        return flags, evidence, rule_scores_dict, judge_scores, fused_scores
    
    def _check_final_answer(self, cot_text: str, gold: Optional[str]) -> bool:
        """
        Check if the final answer matches the gold answer.
        
        Args:
            cot_text: The Chain-of-Thought reasoning text
            gold: Ground truth answer
            
        Returns:
            True if final answer matches gold answer
        """
        if not gold:
            return True
        
        # Extract final answer from CoT text
        final_text, final_value = extract_final_answer(cot_text)
        
        if not final_text and not final_value:
            return False
        
        # Normalize answers for comparison
        def normalize_answer(ans):
            if not ans:
                return None
            # Remove common formatting
            import re
            ans = re.sub(r'[$,]', '', str(ans))  # Remove $ and commas
            ans = re.sub(r'[()]', '', ans)  # Remove parentheses
            return ans.strip()
        
        normalized_final = normalize_answer(final_text or final_value)
        normalized_gold = normalize_answer(gold)
        
        if not normalized_final or not normalized_gold:
            return False
        
        # Try numeric comparison first
        try:
            final_num = float(normalized_final)
            gold_num = float(normalized_gold)
            return abs(final_num - gold_num) < 1e-6
        except ValueError:
            pass
        
        # Fall back to string comparison
        return normalized_final.lower() == normalized_gold.lower()