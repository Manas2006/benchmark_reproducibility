"""
Natural language processing utilities for text similarity and analysis.
"""

import re
from typing import List, Set, Optional
from .config import OFFTOPIC_JACCARD

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class NLPUtils:
    """NLP utilities with optional sentence transformer support."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize NLP utilities.
        
        Args:
            model_name: Sentence transformer model name (None for TF-IDF fallback)
        """
        self.model = None
        self.use_embeddings = False
        
        if model_name and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.use_embeddings = True
            except Exception:
                pass
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def get_stopwords(self) -> Set[str]:
        """Get common English stopwords."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'so', 'then', 'now', 'also',
            'next', 'first', 'second', 'third', 'last', 'finally', 'therefore', 'thus', 'hence',
            'consequently', 'because', 'since', 'as', 'if', 'when', 'where', 'how', 'why', 'what',
            'which', 'who', 'whom', 'whose', 'let', 'denote', 'represent', 'means', 'equals',
            'step', 'steps', 'answer', 'solution', 'problem', 'question', 'given', 'find', 'calculate'
        }
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> Set[str]:
        """Get n-grams from tokens."""
        if len(tokens) < n:
            return set(tokens)
        
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.add(ngram)
        
        return ngrams
    
    def jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate Jaccard similarity between two token lists.
        
        Args:
            tokens1: First token list
            tokens2: Second token list
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        import math
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Uses sentence transformers if available, otherwise TF-IDF-based Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if self.use_embeddings and self.model:
            return self._embedding_similarity(text1, text2)
        else:
            return self._tfidf_similarity(text1, text2)
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence embeddings."""
        try:
            embeddings = self.model.encode([text1, text2])
            return self.cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
        except Exception:
            return self._tfidf_similarity(text1, text2)
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using TF-IDF and Jaccard."""
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        
        # Remove stopwords
        stopwords = self.get_stopwords()
        tokens1 = [t for t in tokens1 if t not in stopwords]
        tokens2 = [t for t in tokens2 if t not in stopwords]
        
        # Use both unigrams and bigrams
        ngrams1 = self.get_ngrams(tokens1, 1) | self.get_ngrams(tokens1, 2)
        ngrams2 = self.get_ngrams(tokens2, 1) | self.get_ngrams(tokens2, 2)
        
        return self.jaccard_similarity(list(ngrams1), list(ngrams2))
    
    def is_off_topic(self, step_text: str, query_text: str) -> bool:
        """
        Check if step is off-topic relative to query.
        
        Args:
            step_text: Step text to check
            query_text: Query/problem text
            
        Returns:
            True if step appears off-topic
        """
        similarity = self.similarity(step_text, query_text)
        return similarity < OFFTOPIC_JACCARD
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numeric literals from text."""
        import re
        # Pattern for numbers (including decimals and scientific notation)
        pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract key noun phrases and entities (conservative)."""
        import re
        
        # Only extract proper nouns (capitalized words) and technical terms
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Filter out common words
        common_words = {
            'The', 'She', 'He', 'It', 'We', 'They', 'This', 'That', 'These', 'Those',
            'Let', 'Now', 'Then', 'So', 'Also', 'Next', 'First', 'Second', 'Third',
            'Last', 'Finally', 'Therefore', 'Thus', 'Hence', 'Consequently'
        }
        
        mentions = []
        for noun in proper_nouns:
            if noun not in common_words and len(noun) > 2:
                mentions.append(noun)
        
        return list(set(mentions))  # Remove duplicates
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract atomic claims from text.
        
        Args:
            text: Input text
            
        Returns:
            List of claim sentences
        """
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                claims.append(sentence)
        
        return claims


# Global instance
_nlp_utils = None

def get_nlp_utils(model_name: Optional[str] = None) -> NLPUtils:
    """Get global NLP utils instance."""
    global _nlp_utils
    if _nlp_utils is None:
        _nlp_utils = NLPUtils(model_name)
    return _nlp_utils
