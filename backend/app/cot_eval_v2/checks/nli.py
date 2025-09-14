"""
Natural Language Inference (NLI) checking for CoT evaluation.

This module provides functions to check logical relationships between
reasoning steps using HuggingFace NLI models.
"""

from typing import Dict, Optional, Any, Union
import warnings


def nli_probs(premise: str, hypothesis: str, pipe: Optional[Any] = None) -> Dict[str, float]:
    """
    Run NLI model on (premise, hypothesis).
    Returns dict of entail/neutral/contra probs.
    If no pipeline is provided, return zeros.
    """
    if pipe is None:
        return {"entail": 0.0, "neutral": 0.0, "contra": 0.0}
    
    try:
        out = pipe({"text": hypothesis, "text_pair": premise}, truncation=True, top_k=None)
        scores = {d["label"].upper(): float(d["score"]) for d in out}
        return {
            "entail": scores.get("ENTAILMENT", 0.0),
            "neutral": scores.get("NEUTRAL", 0.0),
            "contra": scores.get("CONTRADICTION", 0.0),
        }
    except Exception as e:
        warnings.warn(f"NLI pipeline failed: {str(e)}")
        return {"entail": 0.0, "neutral": 0.0, "contra": 0.0}


def nli_label(premise: str, hypothesis: str, pipe: Optional[Any] = None) -> Dict[str, Union[str, float]]:
    """
    Check natural language inference relationship between premise and hypothesis.
    
    Args:
        premise: The premise text (previous reasoning context)
        hypothesis: The hypothesis text (current step to check)
        pipe: Optional HuggingFace pipeline for NLI. If None, returns UNKNOWN.
        
    Returns:
        Dictionary with:
        - "label": "ENTAILMENT" | "NEUTRAL" | "CONTRADICTION" | "UNKNOWN"
        - "score": confidence score (0.0-1.0)
    """
    if pipe is None:
        return {
            "label": "UNKNOWN",
            "score": 0.0
        }
    
    try:
        # Prepare input for HuggingFace pipeline
        # Most NLI models expect text and text_pair format
        inputs = {
            "text": hypothesis,
            "text_pair": premise
        }
        
        # Get predictions from the pipeline
        results = pipe(inputs)
        
        # Handle different output formats
        if isinstance(results, list) and len(results) > 0:
            # Multiple results - take the highest scoring one
            best_result = max(results, key=lambda x: x.get('score', 0))
            label = best_result.get('label', 'UNKNOWN')
            score = best_result.get('score', 0.0)
        elif isinstance(results, dict):
            # Single result
            label = results.get('label', 'UNKNOWN')
            score = results.get('score', 0.0)
        else:
            # Unexpected format
            return {
                "label": "UNKNOWN", 
                "score": 0.0
            }
        
        # Normalize label names to standard format
        label = label.upper()
        if label not in ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]:
            # Map common variations
            label_mapping = {
                "ENTAILS": "ENTAILMENT",
                "SUPPORTS": "ENTAILMENT", 
                "CONTRADICTS": "CONTRADICTION",
                "REFUTES": "CONTRADICTION",
                "UNRELATED": "NEUTRAL",
                "IRRELEVANT": "NEUTRAL"
            }
            label = label_mapping.get(label, "UNKNOWN")
        
        return {
            "label": label,
            "score": float(score)
        }
        
    except Exception as e:
        # Log warning but don't crash
        warnings.warn(f"NLI pipeline failed: {str(e)}")
        return {
            "label": "UNKNOWN",
            "score": 0.0
        }


def create_nli_pipeline(model_name: str = "facebook/bart-large-mnli") -> Optional[Any]:
    """
    Create a HuggingFace NLI pipeline.
    
    Args:
        model_name: Name of the NLI model to use
        
    Returns:
        Pipeline object if successful, None otherwise
    """
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )
        return pipe
    except ImportError:
        warnings.warn("transformers library not available for NLI")
        return None
    except Exception as e:
        warnings.warn(f"Failed to create NLI pipeline: {str(e)}")
        return None


def check_contradiction(premise: str, hypothesis: str, pipe: Optional[Any] = None, 
                       threshold: float = 0.8) -> bool:
    """
    Check if hypothesis contradicts premise.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text  
        pipe: Optional NLI pipeline
        threshold: Minimum score threshold for contradiction
        
    Returns:
        True if contradiction detected above threshold
    """
    result = nli_label(premise, hypothesis, pipe)
    return (result["label"] == "CONTRADICTION" and 
            result["score"] >= threshold)


def check_entailment(premise: str, hypothesis: str, pipe: Optional[Any] = None,
                    threshold: float = 0.8) -> bool:
    """
    Check if premise entails hypothesis.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        pipe: Optional NLI pipeline  
        threshold: Minimum score threshold for entailment
        
    Returns:
        True if entailment detected above threshold
    """
    result = nli_label(premise, hypothesis, pipe)
    return (result["label"] == "ENTAILMENT" and 
            result["score"] >= threshold)
