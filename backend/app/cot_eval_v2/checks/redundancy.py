"""
Redundancy checking for CoT evaluation.

This module provides functions to detect redundant or repetitive
reasoning steps using sentence embeddings.
"""

from typing import List, Dict, Optional, Any
import warnings


def redundancy_pairs(steps: List[str], encoder: Optional[Any] = None, 
                    thresh: float = 0.90) -> List[Dict[str, Any]]:
    """
    Find redundant adjacent step pairs using cosine similarity.
    
    Args:
        steps: List of reasoning step texts
        encoder: Optional sentence encoder for embeddings. If None, returns empty list.
        thresh: Similarity threshold for considering steps redundant (0.0-1.0)
        
    Returns:
        List of redundant pairs with:
        - "i": index of first step
        - "j": index of second step  
        - "sim": similarity score
    """
    if encoder is None or len(steps) < 2:
        return []
    
    try:
        # Get embeddings for all steps
        embeddings = encoder.encode(steps)
        
        # Check adjacent pairs only (i, i+1)
        redundant_pairs = []
        
        for i in range(len(steps) - 1):
            j = i + 1
            
            # Compute cosine similarity
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            
            if similarity >= thresh:
                redundant_pairs.append({
                    "i": i,
                    "j": j, 
                    "sim": float(similarity)
                })
        
        return redundant_pairs
        
    except Exception as e:
        warnings.warn(f"Redundancy checking failed: {str(e)}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0.0-1.0)
    """
    import math
    
    # Compute dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Compute magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (magnitude1 * magnitude2)


def create_sentence_encoder(model_name: str = "all-MiniLM-L6-v2") -> Optional[Any]:
    """
    Create a sentence transformer encoder.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Encoder object if successful, None otherwise
    """
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name)
        return encoder
    except ImportError:
        warnings.warn("sentence-transformers library not available")
        return None
    except Exception as e:
        warnings.warn(f"Failed to create sentence encoder: {str(e)}")
        return None


def detect_repetitive_patterns(steps: List[str], min_length: int = 3) -> List[Dict[str, Any]]:
    """
    Detect repetitive patterns in reasoning steps using simple text analysis.
    
    Args:
        steps: List of reasoning step texts
        min_length: Minimum pattern length to consider
        
    Returns:
        List of detected patterns with repetition counts
    """
    patterns = []
    
    # Look for repeated phrases
    for i, step in enumerate(steps):
        words = step.lower().split()
        
        # Check for repeated phrases of different lengths
        for length in range(min_length, len(words) + 1):
            for start in range(len(words) - length + 1):
                phrase = " ".join(words[start:start + length])
                
                # Count occurrences of this phrase in other steps
                count = 0
                for j, other_step in enumerate(steps):
                    if i != j and phrase in other_step.lower():
                        count += 1
                
                if count > 0:
                    patterns.append({
                        "phrase": phrase,
                        "length": length,
                        "repetitions": count,
                        "first_occurrence": i,
                        "step": step
                    })
    
    # Remove duplicates and sort by repetition count
    unique_patterns = []
    seen_phrases = set()
    
    for pattern in sorted(patterns, key=lambda x: x["repetitions"], reverse=True):
        if pattern["phrase"] not in seen_phrases:
            unique_patterns.append(pattern)
            seen_phrases.add(pattern["phrase"])
    
    return unique_patterns


def analyze_step_diversity(steps: List[str]) -> Dict[str, Any]:
    """
    Analyze diversity of reasoning steps.
    
    Args:
        steps: List of reasoning step texts
        
    Returns:
        Analysis of step diversity
    """
    if not steps:
        return {
            "total_steps": 0,
            "unique_steps": 0,
            "diversity_ratio": 0.0,
            "avg_length": 0.0,
            "length_variance": 0.0
        }
    
    # Count unique steps
    unique_steps = len(set(steps))
    total_steps = len(steps)
    diversity_ratio = unique_steps / total_steps if total_steps > 0 else 0.0
    
    # Analyze step lengths
    lengths = [len(step.split()) for step in steps]
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0
    
    # Calculate variance in step lengths
    if len(lengths) > 1:
        mean_length = avg_length
        variance = sum((length - mean_length) ** 2 for length in lengths) / (len(lengths) - 1)
    else:
        variance = 0.0
    
    return {
        "total_steps": total_steps,
        "unique_steps": unique_steps,
        "diversity_ratio": diversity_ratio,
        "avg_length": avg_length,
        "length_variance": variance
    }
