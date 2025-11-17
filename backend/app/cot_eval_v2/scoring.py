"""
Rule-based scoring for CoT evaluation v2.

This module provides deterministic scoring functions that don't require
LLM calls, used for smoke tests and baseline evaluation.
"""

from typing import Dict, Any


def rule_scores(evidence: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute provisional 0..1 scores based on evidence using rule-based heuristics.
    
    These scores are deterministic and don't require LLM calls. They serve
    as baselines and can be used for smoke testing.
    
    Args:
        evidence: Evidence dictionary from PillarsEvaluator.analyze()
        
    Returns:
        Dictionary with rule-based scores for each pillar:
        - "faithfulness_rule": Faithfulness score (0.0-1.0)
        - "utility_rule": Utility score (0.0-1.0) 
        - "coherence_rule": Coherence score (0.0-1.0)
        - "factuality_rule": Factuality score (0.0-1.0)
    """
    
    # UTILITY SCORE - anchored on final correctness
    if evidence.get("final_correct", False):
        # Correct final answer gets base score
        utility = 0.5 * 1.0  # Base score for correct answer
        
        # Add intermediate quality component
        intermediate_ok_rate = evidence.get("intermediate_ok_rate", 0.0)
        utility += 0.3 * intermediate_ok_rate
        
        # Add coverage component (penalize unused numbers)
        coverage = evidence.get("coverage", {})
        unused_count = len(coverage.get("unused", []))
        coverage_factor = 1.0 / (1 + unused_count)
        utility += 0.2 * coverage_factor
        
        # Penalize redundancy
        redund_cnt = evidence.get("redund_cnt", 0)
        utility *= 1.0 / (1 + redund_cnt)
        
    else:
        # Wrong final answer gets much lower score
        intermediate_ok_rate = evidence.get("intermediate_ok_rate", 0.0)
        utility = 0.2 * intermediate_ok_rate
    
    # COHERENCE SCORE - penalize contradictions
    coh_contra_cnt = evidence.get("coh_contra_cnt", 0)
    coherence = max(0.0, 1.0 - 0.5 * coh_contra_cnt)
    
    # FACTUALITY SCORE - use number grounding as proxy
    coverage = evidence.get("coverage", {})
    unused_count = len(coverage.get("unused", []))
    if unused_count == 0:
        factuality = 1.0
    else:
        factuality = 1.0 / (1 + unused_count)
    
    # FAITHFULNESS SCORE - start high, penalize issues
    faithfulness = 1.0
    
    # Penalize wrong-but-right pattern
    if evidence.get("wrong_but_right", False):
        faithfulness = min(faithfulness, 0.5)
    
    # Penalize self-repair (indicates initial errors)
    self_repair_cnt = evidence.get("self_repair_cnt", 0)
    if self_repair_cnt > 0:
        faithfulness = min(faithfulness, 0.7)
    
    # Additional penalty for multiple self-repairs
    if self_repair_cnt > 1:
        faithfulness *= 0.8
    
    return {
        "faithfulness_rule": max(0.0, min(1.0, faithfulness)),
        "utility_rule": max(0.0, min(1.0, utility)),
        "coherence_rule": max(0.0, min(1.0, coherence)),
        "factuality_rule": max(0.0, min(1.0, factuality))
    }


def compute_overall_rule_score(rule_scores: Dict[str, float]) -> float:
    """
    Compute overall score from individual pillar scores.
    
    Uses equal weighting across all pillars.
    
    Args:
        rule_scores: Dictionary of pillar scores from rule_scores()
        
    Returns:
        Overall score (0.0-1.0)
    """
    pillar_scores = [
        rule_scores.get("faithfulness_rule", 0.0),
        rule_scores.get("utility_rule", 0.0),
        rule_scores.get("coherence_rule", 0.0),
        rule_scores.get("factuality_rule", 0.0)
    ]
    
    return sum(pillar_scores) / len(pillar_scores)


def get_score_interpretation(score: float) -> str:
    """
    Get human-readable interpretation of a score.
    
    Args:
        score: Score value (0.0-1.0)
        
    Returns:
        Interpretation string
    """
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.8:
        return "Good"
    elif score >= 0.6:
        return "Fair"
    elif score >= 0.4:
        return "Poor"
    else:
        return "Very Poor"


def analyze_score_distribution(scores: list) -> Dict[str, Any]:
    """
    Analyze distribution of scores across multiple samples.
    
    Args:
        scores: List of score values
        
    Returns:
        Dictionary with distribution statistics
    """
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "percentiles": {}
        }
    
    import statistics
    
    return {
        "count": len(scores),
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "percentiles": {
            "25th": statistics.quantiles(scores, n=4)[0] if len(scores) > 1 else scores[0],
            "50th": statistics.median(scores),
            "75th": statistics.quantiles(scores, n=4)[2] if len(scores) > 1 else scores[0],
            "90th": statistics.quantiles(scores, n=10)[8] if len(scores) > 9 else max(scores)
        }
    }


def fuse_with_judge(rule: Dict[str, float], judge: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, float]:
    """
    Get final scores using LLM judge scores exclusively when available, otherwise rule-based scores.
    
    IMPORTANT: When LLM judge is available, ONLY judge scores are used. Rule-based scores
    are completely ignored and not combined with judge scores.
    
    Args:
        rule: Rule-based scores from rule_scores() (0.0-1.0) - only used if judge unavailable
        judge: Judge scores from Judge.score() (1-5 or None)
        evidence: Evidence dictionary from PillarsEvaluator.analyze() (unused, kept for compatibility)
        
    Returns:
        Final scores dictionary with 0.0-1.0 values (from judge if available, otherwise from rules)
    """
    from .judge import Judge
    
    # Validate and normalize judge scores
    judge_normalized = Judge.validate_and_normalize_judge(judge)
    
    # Check if any judge scores are available
    has_judge_scores = any(judge_normalized.get(pillar) is not None 
                          for pillar in ["faithfulness", "utility", "coherence", "factuality"])
    
    fused = {}
    
    if has_judge_scores:
        # LLM judge is available - use ONLY judge scores (ignore rule scores completely)
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            judge_score = judge_normalized.get(pillar)
            if judge_score is not None:
                # Convert from 1-5 scale to 0.0-1.0 scale
                fused[pillar] = max(0.0, min(1.0, (judge_score - 1) / 4.0))
            else:
                # Judge didn't provide score for this pillar, use rule score as fallback
                fused[pillar] = rule.get(f"{pillar}_rule", 0.0)
    else:
        # No judge scores available - use rule-based scores exclusively
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            fused[pillar] = rule.get(f"{pillar}_rule", 0.0)
    
    # Compute overall score as average of pillar scores
    fused["overall"] = sum(fused[pillar] for pillar in ["faithfulness", "utility", "coherence", "factuality"]) / 4.0
    
    return fused


def compare_score_methods(rule: Dict[str, float], judge: Dict[str, Any], fused: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare different scoring methods for analysis.
    
    Args:
        rule: Rule-based scores
        judge: Judge scores (1-5 or None)
        fused: Fused scores
        
    Returns:
        Comparison analysis dictionary
    """
    comparison = {
        "rule_overall": (rule.get("faithfulness_rule", 0.0) + rule.get("utility_rule", 0.0) + 
                        rule.get("coherence_rule", 0.0) + rule.get("factuality_rule", 0.0)) / 4.0,
        "judge_overall": None,
        "fused_overall": fused.get("overall", 0.0),
        "judge_available": False,
        "score_differences": {}
    }
    
    # Calculate judge overall if available
    judge_scores = []
    for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
        judge_score = judge.get(pillar)
        if judge_score is not None:
            judge_scores.append((judge_score - 1) / 4.0)  # Normalize to 0-1
    
    if judge_scores:
        comparison["judge_overall"] = sum(judge_scores) / len(judge_scores)
        comparison["judge_available"] = True
        
        # Calculate score differences
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            rule_key = f"{pillar}_rule"
            rule_score = rule.get(rule_key, 0.0)
            fused_score = fused.get(pillar, 0.0)
            comparison["score_differences"][pillar] = fused_score - rule_score
    
    return comparison
