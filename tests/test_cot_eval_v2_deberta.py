"""
Unit tests for DeBERTa MNLI integration in CoT evaluation v2.

Tests the NLI-based coherence and factuality flagging functionality
using dummy NLI pipelines for deterministic testing.
"""

import pytest
import sys
import os

# Add the backend app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

from cot_eval_v2.evaluator import PillarsEvaluator
from cot_eval_v2.checks.nli import nli_probs


class DummyPipe:
    """Dummy NLI pipeline for testing."""
    
    def __call__(self, x, truncation=True, top_k=None):
        hyp = x["text"]
        if "wrong" in hyp.lower():
            return [
                {"label": "CONTRADICTION", "score": 0.9},
                {"label": "ENTAILMENT", "score": 0.05},
                {"label": "NEUTRAL", "score": 0.05},
            ]
        if "true" in hyp.lower() or "correct" in hyp.lower():
            return [
                {"label": "ENTAILMENT", "score": 0.9},
                {"label": "CONTRADICTION", "score": 0.05},
                {"label": "NEUTRAL", "score": 0.05},
            ]
        return [
            {"label": "NEUTRAL", "score": 0.9},
            {"label": "ENTAILMENT", "score": 0.05},
            {"label": "CONTRADICTION", "score": 0.05},
        ]


class TestNLIProbs:
    """Test the nli_probs function."""
    
    def test_nli_probs_with_pipe(self):
        """Test nli_probs with dummy pipeline."""
        pipe = DummyPipe()
        
        # Test contradiction
        probs = nli_probs("The sky is blue", "The sky is wrong color", pipe)
        assert probs["contra"] == 0.9
        assert probs["entail"] == 0.05
        assert probs["neutral"] == 0.05
        
        # Test entailment
        probs = nli_probs("The sky is blue", "The sky is correct color", pipe)
        assert probs["entail"] == 0.9
        assert probs["contra"] == 0.05
        assert probs["neutral"] == 0.05
        
        # Test neutral
        probs = nli_probs("The sky is blue", "The sky is visible", pipe)
        assert probs["neutral"] == 0.9
        assert probs["entail"] == 0.05
        assert probs["contra"] == 0.05
    
    def test_nli_probs_without_pipe(self):
        """Test nli_probs without pipeline."""
        probs = nli_probs("premise", "hypothesis", None)
        assert probs["entail"] == 0.0
        assert probs["neutral"] == 0.0
        assert probs["contra"] == 0.0


class TestCoherenceFlagging:
    """Test coherence flagging functionality."""
    
    def test_coherence_contradiction_detection(self):
        """Test detection of contradictions between steps."""
        ev = PillarsEvaluator(nli_pipe=DummyPipe(), use_nli=False)
        problem = "Fact: sky is blue"
        cot = "Step 1. The sky is blue.\nStep 2. The sky is wrong color.\n#### answer"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold=None)
        
        # Check coherence flags
        coh_flags = flags.get_flags_by_pillar("coherence")
        assert len(coh_flags) > 0
        assert any(f.issue == "contradiction" for f in coh_flags)
        
        # Check evidence
        assert evidence["coh_contra_cnt"] == 1
        assert evidence["avg_coh_margin"] < 0  # Negative margin indicates contradiction
    
    def test_coherence_no_contradiction(self):
        """Test when there are no contradictions."""
        ev = PillarsEvaluator(nli_pipe=DummyPipe(), use_nli=False)
        problem = "Fact: sky is blue"
        cot = "Step 1. The sky is blue.\nStep 2. The sky is correct color.\n#### answer"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold=None)
        
        # Should have no coherence flags
        coh_flags = flags.get_flags_by_pillar("coherence")
        assert len(coh_flags) == 0
        
        # Check evidence
        assert evidence["coh_contra_cnt"] == 0
        assert evidence["avg_coh_margin"] > 0  # Positive margin indicates entailment
    
    def test_coherence_without_nli_pipe(self):
        """Test coherence checking without NLI pipeline."""
        ev = PillarsEvaluator(nli_pipe=None, use_nli=False)
        problem = "Fact: sky is blue"
        cot = "Step 1. The sky is blue.\nStep 2. The sky is wrong color.\n#### answer"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold=None)
        
        # Should have no coherence flags when no NLI pipe
        coh_flags = flags.get_flags_by_pillar("coherence")
        assert len(coh_flags) == 0
        
        # Evidence should show zero counts
        assert evidence["coh_contra_cnt"] == 0
        assert evidence["avg_coh_margin"] == 0.0


class TestFactualityFlagging:
    """Test factuality flagging functionality."""
    
    def test_factuality_contradiction_detection(self):
        """Test detection of contradictions between steps and problem."""
        ev = PillarsEvaluator(nli_pipe=DummyPipe(), use_nli=False)
        problem = "Fact: 2+2=4"
        cot = "Step 1. 2+2=4 is true.\nStep 2. 2+2=5 is wrong.\n#### answer"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold=None)
        
        # Check factuality flags
        fact_flags = flags.get_flags_by_pillar("factuality")
        assert len(fact_flags) > 0
        assert any(f.issue == "ungrounded_or_false" for f in fact_flags)
        
        # Check evidence
        assert evidence["fact_contra_cnt"] == 1
        assert evidence["fact_entail_rate"] > 0  # Some steps should entail
        # Note: avg_fact_margin might be 0 if all steps have same margin
    
    def test_factuality_entailment_detection(self):
        """Test detection of entailment between steps and problem."""
        ev = PillarsEvaluator(nli_pipe=DummyPipe(), use_nli=False)
        problem = "Fact: 2+2=4"
        cot = "Step 1. 2+2=4 is true.\nStep 2. 2+2=4 is correct.\n#### answer"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold=None)
        
        # Should have no factuality flags for contradictions
        fact_flags = flags.get_flags_by_pillar("factuality")
        assert len(fact_flags) == 0
        
        # Check evidence
        assert evidence["fact_contra_cnt"] == 0
        assert evidence["fact_entail_rate"] > 0  # Steps should entail
        assert evidence["avg_fact_margin"] > 0  # Positive margin indicates entailment
    
    def test_factuality_without_nli_pipe(self):
        """Test factuality checking without NLI pipeline."""
        ev = PillarsEvaluator(nli_pipe=None, use_nli=False)
        problem = "Fact: 2+2=4"
        cot = "Step 1. 2+2=4 is true.\nStep 2. 2+2=5 is wrong.\n#### answer"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold=None)
        
        # Should have no factuality flags when no NLI pipe
        fact_flags = flags.get_flags_by_pillar("factuality")
        assert len(fact_flags) == 0
        
        # Evidence should show zero counts
        assert evidence["fact_contra_cnt"] == 0
        assert evidence["fact_entail_rate"] == 0.0
        assert evidence["avg_fact_margin"] == 0.0


class TestCombinedNLI:
    """Test combined coherence and factuality checking."""
    
    def test_both_coherence_and_factuality_flags(self):
        """Test when both coherence and factuality issues are detected."""
        ev = PillarsEvaluator(nli_pipe=DummyPipe(), use_nli=False)
        problem = "The capital of France is Paris"
        cot = "Step 1. France is a European country.\nStep 2. The capital of France is London.\nStep 3. This is wrong information.\n#### London"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold="Paris")
        
        # Check both types of flags
        coh_flags = flags.get_flags_by_pillar("coherence")
        fact_flags = flags.get_flags_by_pillar("factuality")
        
        assert len(coh_flags) > 0
        assert len(fact_flags) > 0
        
        # Check evidence
        assert evidence["coh_contra_cnt"] > 0
        assert evidence["fact_contra_cnt"] > 0
        assert evidence["fact_entail_rate"] >= 0
    
    def test_evidence_metrics_calculation(self):
        """Test that evidence metrics are calculated correctly."""
        ev = PillarsEvaluator(nli_pipe=DummyPipe(), use_nli=False)
        problem = "Test problem"
        cot = "Step 1. This is true.\nStep 2. This is wrong.\n#### answer"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = ev.analyze(problem, cot, gold=None)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "coh_contra_cnt", "avg_coh_margin",
            "fact_entail_rate", "fact_contra_cnt", "avg_fact_margin"
        ]
        
        for metric in expected_metrics:
            assert metric in evidence
            assert isinstance(evidence[metric], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
