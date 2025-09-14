"""
Main evaluator orchestration for CoT evaluation v2.

This module provides the PillarsEvaluator class that coordinates
all the deterministic checks and flag collection.
"""

import re
from typing import List, Tuple, Dict, Any, Optional

from .flags import FlagCollector
from .checks.arithmetic import check_step_equations
from .checks.nli import nli_label, nli_probs
from .checks.coverage import number_coverage
from .checks.redundancy import redundancy_pairs
from .checks.heuristics import wrong_but_right, self_repair_markers, shortcut_signature
from .scoring import rule_scores, fuse_with_judge


def split_steps(cot_text: str) -> List[str]:
    """
    Split CoT text into individual reasoning steps.
    
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
    
    # If we only have one step, return as-is
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
        patterns = [
            r'the answer is\s*:?\s*(.+?)(?:\n|$)',
            r'therefore\s*:?\s*(.+?)(?:\n|$)',
            r'so\s*:?\s*(.+?)(?:\n|$)',
            r'thus\s*:?\s*(.+?)(?:\n|$)',
            r'hence\s*:?\s*(.+?)(?:\n|$)',
            r'consequently\s*:?\s*(.+?)(?:\n|$)'
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
    
    This class coordinates arithmetic, coherence, coverage, redundancy, and
    faithfulness checks to produce flags and evidence for reasoning quality.
    """
    
    def __init__(self, nli_pipe: Optional[Any] = None, encoder: Optional[Any] = None, use_nli: bool = True, judge: Optional[Any] = None):
        """
        Initialize the evaluator with optional external models.
        
        Args:
            nli_pipe: Optional HuggingFace NLI pipeline for coherence checking
            encoder: Optional sentence encoder for redundancy checking
            use_nli: If True and nli_pipe is None, auto-load DeBERTa MNLI
            judge: Optional Judge instance for LLM-based evaluation
        """
        self.nli_pipe = nli_pipe
        self.encoder = encoder
        self.judge = judge
        
        # Auto-load DeBERTa MNLI if requested and no pipe provided
        if use_nli and self.nli_pipe is None:
            try:
                from transformers import pipeline
                import torch
                import os
                
                # Detect SLURM GPU allocation
                slurm_gpu_id = self._get_slurm_gpu_id()
                if slurm_gpu_id is not None:
                    device = slurm_gpu_id
                    print(f"Using SLURM GPU {device} for DeBERTa MNLI...")
                elif torch.cuda.is_available():
                    device = 0
                    print(f"Using CUDA GPU {device} for DeBERTa MNLI...")
                else:
                    device = -1
                    print(f"Using CPU for DeBERTa MNLI (no GPU available)...")
                
                self.nli_pipe = pipeline(
                    "text-classification", 
                    model="microsoft/deberta-base-mnli", 
                    device=device
                )
                print(f"✅ DeBERTa MNLI loaded successfully on device {device}")
            except Exception as e:
                print(f"Failed to auto-load DeBERTa MNLI: {e}")
                self.nli_pipe = None
    
    def _get_slurm_gpu_id(self) -> Optional[int]:
        """
        Detect SLURM GPU allocation and return the GPU ID to use.
        
        Returns:
            GPU ID (int) if SLURM GPU is allocated, None otherwise
        """
        import os
        
        # Check for SLURM environment variables
        slurm_gpu_id = os.environ.get('SLURM_LOCALID')
        if slurm_gpu_id is not None:
            try:
                return int(slurm_gpu_id)
            except ValueError:
                pass
        
        # Check for CUDA_VISIBLE_DEVICES (set by SLURM)
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices is not None and cuda_devices.strip():
            try:
                # CUDA_VISIBLE_DEVICES maps to device 0 in PyTorch
                return 0
            except ValueError:
                pass
        
        # Check for SLURM_PROCID (alternative SLURM variable)
        slurm_procid = os.environ.get('SLURM_PROCID')
        if slurm_procid is not None:
            try:
                return int(slurm_procid)
            except ValueError:
                pass
        
        return None
    
    def analyze(self, problem: str, cot_text: str, gold: Optional[str] = None) -> Tuple[FlagCollector, Dict[str, Any], Dict[str, float], Dict[str, Any], Dict[str, float]]:
        """
        Analyze CoT reasoning and return flags, evidence, rule scores, judge scores, and fused scores.
        
        Args:
            problem: The original problem text
            cot_text: The Chain-of-Thought reasoning text
            gold: Ground truth answer (optional)
            
        Returns:
            Tuple of (FlagCollector, evidence_dict, rule_scores, judge_scores, fused_scores)
        """
        flags = FlagCollector()
        
        # Step 1: Split reasoning into steps
        steps = split_steps(cot_text)
        
        # Step 2: Check final answer correctness
        final_correct = check_final_answer(cot_text, gold)
        
        # Step 3: ARITHMETIC - Check step equations
        arith_ok = 0
        arith_bad = 0
        arith_examples = []
        
        for i, step in enumerate(steps):
            try:
                arith_result = check_step_equations(step)
                arith_ok += arith_result["ok"]
                arith_bad += arith_result["bad"]
                arith_examples.extend(arith_result["examples"])
                
                # Flag utility issues for bad arithmetic
                if arith_result["bad"] > 0:
                    flags.add(
                        "utility",
                        f"step={i+1}",
                        "arithmetic_error",
                        {"bad_equations": arith_result["bad"], "examples": arith_result["examples"]}
                    )
            except Exception:
                # If arithmetic checking fails, continue without flags
                pass
        
        intermediate_ok_rate = arith_ok / max(1, arith_ok + arith_bad)
        
        # Step 4: COHERENCE - Check step-to-step logical flow using NLI
        coh_contra_cnt = 0
        coh_margins = []
        
        for i in range(1, len(steps)):
            premise = " ".join(steps[:i])
            probs = nli_probs(premise, steps[i], self.nli_pipe)
            coh_margins.append(probs["entail"] - probs["contra"])
            if probs["contra"] >= 0.80:
                coh_contra_cnt += 1
                flags.add(
                    "coherence",
                    f"step={i+1}",
                    "contradiction",
                    {"p_contra": probs["contra"]}
                )
        
        # Step 5: COVERAGE - Check number usage
        # Only check reasoning steps, not the final answer
        try:
            # Remove final answer from steps for coverage checking
            reasoning_steps = []
            for step in steps:
                if not step.strip().startswith('####') and '####' not in step:
                    reasoning_steps.append(step)
            
            cov = number_coverage(problem, reasoning_steps)
            for unused_num in cov["unused"]:
                flags.add(
                    "utility",
                    "coverage",
                    "unused_given_number",
                    {"number": unused_num, "context": problem}
                )
        except Exception:
            cov = {"given": [], "used": [], "unused": []}
        
        # Step 6: FACTUALITY - Check step grounding against problem context
        fact_entails = 0
        fact_contra_cnt = 0
        fact_margins = []
        
        for i, step in enumerate(steps, 1):
            probs = nli_probs(problem, step, self.nli_pipe)
            fact_margins.append(probs["entail"] - probs["contra"])
            if probs["entail"] >= 0.70:
                fact_entails += 1
            if probs["contra"] >= 0.80:
                fact_contra_cnt += 1
                flags.add(
                    "factuality",
                    f"step={i}",
                    "ungrounded_or_false",
                    {"p_contra": probs["contra"]}
                )
        
        # Step 7: REDUNDANCY - Check for redundant adjacent steps
        try:
            dups = redundancy_pairs(steps, self.encoder)
            for dup in dups:
                flags.add(
                    "utility",
                    f"steps={dup['i']+1}-{dup['j']+1}",
                    "redundant_adjacent_steps",
                    {"similarity": dup["sim"]}
                )
        except Exception:
            dups = []
        
        # Step 7: FAITHFULNESS HEURISTICS
        wrong_but_right_flag = wrong_but_right(final_correct, intermediate_ok_rate)
        if wrong_but_right_flag:
            flags.add(
                "faithfulness",
                "reasoning",
                "wrong_steps_but_correct_final",
                {"intermediate_ok_rate": intermediate_ok_rate}
            )
        
        # Check for self-repair markers
        self_repair_cnt = sum(1 for step in steps if self_repair_markers(step))
        if self_repair_cnt > 0:
            flags.add(
                "faithfulness",
                "reasoning",
                "self_repair_detected",
                {"count": self_repair_cnt}
            )
        
        # Check for shortcut signature
        try:
            final_answer = cot_text.split('####')[-1].strip() if '####' in cot_text else ""
            if shortcut_signature(problem, cot_text, final_answer):
                flags.add(
                    "faithfulness",
                    "reasoning",
                    "shortcut_signature",
                    {"final_answer": final_answer}
                )
        except Exception:
            pass
        
        # Build evidence dictionary
        evidence = {
            "final_correct": final_correct,
            "intermediate_ok_rate": intermediate_ok_rate,
            "coh_contra_cnt": coh_contra_cnt,
            "avg_coh_margin": sum(coh_margins) / max(1, len(coh_margins)),
            "fact_entail_rate": fact_entails / max(1, len(steps)),
            "fact_contra_cnt": fact_contra_cnt,
            "avg_fact_margin": sum(fact_margins) / max(1, len(fact_margins)),
            "redund_cnt": len(dups),
            "coverage": cov,
            "wrong_but_right": wrong_but_right_flag,
            "self_repair_cnt": self_repair_cnt,
            "arith_bad_examples": [ex for ex in arith_examples if not ex.get("correct", True)]
        }
        
        # Compute rule-based scores
        rule_scores_dict = rule_scores(evidence)
        
        # Get judge scores if judge is available
        judge_scores = {"faithfulness": None, "utility": None, "coherence": None, "factuality": None}
        if self.judge is not None:
            flags_summary = flags.summarize_for_prompt()
            judge_scores = self.judge.score(problem, cot_text, gold or "", flags_summary, evidence)
        
        # Fuse rule and judge scores
        fused_scores = fuse_with_judge(rule_scores_dict, judge_scores, evidence)
        
        return flags, evidence, rule_scores_dict, judge_scores, fused_scores
