"""
LLM judge for cases where deterministic code paths are insufficient.
"""

import json
import time
from typing import Dict, Any, Optional, List
from .config import GPT_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, GPT_TIMEOUT_S, DETERMINISTIC_MODE


class LLMJudge:
    """LLM judge for NLI-style and fuzzy judgments."""
    
    def __init__(self, provider: str = GPT_PROVIDER, api_key: Optional[str] = None):
        """
        Initialize LLM judge.
        
        Args:
            provider: LLM provider ("openai", "none")
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key or OPENAI_API_KEY
        self.client = None
        self.disabled = DETERMINISTIC_MODE or provider == "none"
        
        if not self.disabled and provider == "openai":
            self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            self.disabled = True
        except Exception:
            self.disabled = True
    
    def entailment(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Check if premise entails hypothesis.
        
        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            
        Returns:
            Dictionary with "result" and "confidence"
        """
        if self.disabled:
            return {"result": "unknown", "confidence": 0.0}
        
        prompt = f"""Given the premise, determine if it entails the hypothesis.

Premise: {premise}

Hypothesis: {hypothesis}

Does the premise entail the hypothesis? Respond with one of: "entails", "contradicts", or "unknown".
Also provide a confidence score from 0.0 to 1.0.

Format your response as JSON:
{{"result": "entails|contradicts|unknown", "confidence": 0.0-1.0}}"""

        try:
            response = self._call_llm(prompt)
            result = json.loads(response)
            return {
                "result": result.get("result", "unknown"),
                "confidence": float(result.get("confidence", 0.0))
            }
        except Exception:
            return {"result": "unknown", "confidence": 0.0}
    
    def fact_check(self, claim: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Check if claim is supported by evidence.
        
        Args:
            claim: Claim to check
            evidence: List of evidence strings
            
        Returns:
            Dictionary with "result" and "confidence"
        """
        if self.disabled:
            return {"result": "not_found", "confidence": 0.0}
        
        evidence_text = "\n".join(evidence)
        prompt = f"""Given the evidence, determine if the claim is supported.

Evidence:
{evidence_text}

Claim: {claim}

Is the claim supported by the evidence? Respond with one of: "supported", "refuted", or "not_found".
Also provide a confidence score from 0.0 to 1.0.

Format your response as JSON:
{{"result": "supported|refuted|not_found", "confidence": 0.0-1.0}}"""

        try:
            response = self._call_llm(prompt)
            result = json.loads(response)
            return {
                "result": result.get("result", "not_found"),
                "confidence": float(result.get("confidence", 0.0))
            }
        except Exception:
            return {"result": "not_found", "confidence": 0.0}
    
    def topic_alignment(self, step: str, query: str) -> Dict[str, Any]:
        """
        Check if step is on-topic relative to query.
        
        Args:
            step: Step text to check
            query: Query/problem text
            
        Returns:
            Dictionary with "result" and "confidence"
        """
        if self.disabled:
            return {"result": "on_topic", "confidence": 0.5}
        
        prompt = f"""Given the query, determine if the step is on-topic.

Query: {query}

Step: {step}

Is the step on-topic relative to the query? Respond with one of: "on_topic" or "off_topic".
Also provide a confidence score from 0.0 to 1.0.

Format your response as JSON:
{{"result": "on_topic|off_topic", "confidence": 0.0-1.0}}"""

        try:
            response = self._call_llm(prompt)
            result = json.loads(response)
            return {
                "result": result.get("result", "on_topic"),
                "confidence": float(result.get("confidence", 0.5))
            }
        except Exception:
            return {"result": "on_topic", "confidence": 0.5}
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        if self.disabled or not self.client:
            return '{"result": "unknown", "confidence": 0.0}'
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text relationships. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=GPT_TIMEOUT_S
            )
            
            return response.choices[0].message.content.strip()
        except Exception:
            return '{"result": "unknown", "confidence": 0.0}'


# Global instance
_llm_judge = None

def get_llm_judge(provider: str = GPT_PROVIDER, api_key: Optional[str] = None) -> LLMJudge:
    """Get global LLM judge instance."""
    global _llm_judge
    if _llm_judge is None:
        _llm_judge = LLMJudge(provider, api_key)
    return _llm_judge
