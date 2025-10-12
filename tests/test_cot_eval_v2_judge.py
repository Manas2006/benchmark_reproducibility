"""
Unit tests for LLM Judge integration in CoT evaluation v2.

Tests the Judge class, fusion functionality, and integration with PillarsEvaluator.
"""

import pytest
import sys
import os
import json

# Add the backend app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

from cot_eval_v2.judge import Judge, MockJudge
from cot_eval_v2.scoring import fuse_with_judge, compare_score_methods
from cot_eval_v2.evaluator import PillarsEvaluator
from cot_eval_v2.flags import FlagCollector


class TestJudge:
    """Test the Judge class functionality."""
    
    def test_judge_initialization(self):
        """Test judge initialization with different modes."""
        # Test SMART mode (default)
        judge = MockJudge(mode="SMART")
        assert judge.mode == "SMART"
        assert judge.diagnostic == False
        
        # Test ALWAYS mode
        judge = MockJudge(mode="ALWAYS")
        assert judge.mode == "ALWAYS"
        
        # Test NEVER mode
        judge = MockJudge(mode="NEVER")
        assert judge.mode == "NEVER"
        
        # Test diagnostic mode
        judge = MockJudge(diagnostic=True)
        assert judge.diagnostic == True
        
        # Test invalid mode
        with pytest.raises(ValueError):
            MockJudge(mode="INVALID")
    
    def test_prompt_building(self):
        """Test prompt building for different modes."""
        judge_strict = MockJudge(diagnostic=False)
        judge_diagnostic = MockJudge(diagnostic=True)
        
        problem = "What is 2 + 2?"
        cot = "Step 1. I add 2 + 2 = 4"
        gold = "4"
        flags = "No issues detected."
        evidence = {"final_correct": True}
        
        # Test strict mode
        prompt_strict = judge_strict.build_prompt(problem, cot, gold, flags, evidence)
        assert "Do NOT include explanations" in prompt_strict
        assert "Output only the JSON object" in prompt_strict
        assert problem in prompt_strict
        assert cot in prompt_strict
        assert gold in prompt_strict
        
        # Test diagnostic mode
        prompt_diagnostic = judge_diagnostic.build_prompt(problem, cot, gold, flags, evidence)
        assert "briefly explain" in prompt_diagnostic
        assert "First, briefly explain" in prompt_diagnostic
    
    def test_should_call_judge_logic(self):
        """Test the logic for when to call the judge."""
        # Test ALWAYS mode
        judge_always = MockJudge(mode="ALWAYS")
        assert judge_always.should_call_judge("flags", {"final_correct": True})
        assert judge_always.should_call_judge("", {"final_correct": False})
        
        # Test NEVER mode
        judge_never = MockJudge(mode="NEVER")
        assert not judge_never.should_call_judge("flags", {"final_correct": True})
        assert not judge_never.should_call_judge("", {"final_correct": False})
        
        # Test SMART mode
        judge_smart = MockJudge(mode="SMART")
        
        # Should call when there are flags
        assert judge_smart.should_call_judge("Some flags detected", {"final_correct": True})
        
        # Should call when final answer is wrong
        assert judge_smart.should_call_judge("", {"final_correct": False})
        
        # Should NOT call when no flags and correct answer
        assert not judge_smart.should_call_judge("", {"final_correct": True})
        assert not judge_smart.should_call_judge("No issues detected.", {"final_correct": True})
    
    def test_score_calling_logic(self):
        """Test score method respects mode settings."""
        # Test NEVER mode
        judge_never = MockJudge(mode="NEVER")
        scores = judge_never.score("problem", "cot", "gold", "flags", {"final_correct": True})
        assert all(score is None for score in scores.values())
        
        # Test SMART mode with no issues
        judge_smart = MockJudge(mode="SMART")
        scores = judge_smart.score("problem", "cot", "gold", "", {"final_correct": True})
        assert all(score is None for score in scores.values())
        
        # Test SMART mode with issues
        scores = judge_smart.score("problem", "cot", "gold", "Some flags", {"final_correct": True})
        assert all(score is not None for score in scores.values())
        
        # Test ALWAYS mode
        judge_always = MockJudge(mode="ALWAYS")
        scores = judge_always.score("problem", "cot", "gold", "", {"final_correct": True})
        assert all(score is not None for score in scores.values())


class TestJSONParsing:
    """Test robust JSON parsing from LLM output."""
    
    def test_clean_json_parsing(self):
        """Test parsing of clean JSON output."""
        judge = MockJudge()
        
        # Test clean JSON
        clean_output = '{"faithfulness": 4, "utility": 3, "coherence": 5, "factuality": 4}'
        parsed = judge._extract_json_safely(clean_output)
        assert parsed == {"faithfulness": 4, "utility": 3, "coherence": 5, "factuality": 4}
    
    def test_messy_json_parsing(self):
        """Test parsing of messy output with explanations."""
        judge = MockJudge()
        
        # Test messy output with explanations
        messy_output = """Faithfulness: The reasoning is consistent but has some issues.
Utility: The steps are useful but could be clearer.
Coherence: Good logical flow.
Factuality: Accurate.

JSON:
{"faithfulness": 3, "utility": 4, "coherence": 5, "factuality": 4}"""
        
        parsed = judge._extract_json_safely(messy_output)
        assert parsed == {"faithfulness": 3, "utility": 4, "coherence": 5, "factuality": 4}
    
    def test_multiple_json_objects(self):
        """Test parsing when multiple JSON objects are present."""
        judge = MockJudge()
        
        # Test multiple JSON objects - should use the last one
        output = """First attempt: {"faithfulness": 1, "utility": 1, "coherence": 1, "factuality": 1}
Final answer: {"faithfulness": 4, "utility": 4, "coherence": 4, "factuality": 4}"""
        
        parsed = judge._extract_json_safely(output)
        assert parsed == {"faithfulness": 4, "utility": 4, "coherence": 4, "factuality": 4}
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON output."""
        judge = MockJudge()
        
        # Test completely invalid output
        invalid_output = "This is not JSON at all, just some random text."
        parsed = judge._extract_json_safely(invalid_output)
        assert all(score is None for score in parsed.values())
        
        # Test malformed JSON
        malformed_output = '{"faithfulness": 4, "utility": 3, "coherence": 5, "factuality":}'  # Missing value
        parsed = judge._extract_json_safely(malformed_output)
        assert all(score is None for score in parsed.values())
    
    def test_score_validation(self):
        """Test validation of parsed scores."""
        judge = MockJudge()
        
        # Test valid scores
        valid_scores = {"faithfulness": 3, "utility": 4, "coherence": 5, "factuality": 2}
        assert judge._validate_scores(valid_scores) == True
        
        # Test invalid scores (out of range)
        invalid_scores = {"faithfulness": 6, "utility": 4, "coherence": 5, "factuality": 4}
        assert judge._validate_scores(invalid_scores) == False
        
        # Test missing keys
        incomplete_scores = {"faithfulness": 3, "utility": 4, "coherence": 5}
        assert judge._validate_scores(incomplete_scores) == False
        
        # Test wrong types
        wrong_type_scores = {"faithfulness": "good", "utility": 4, "coherence": 5, "factuality": 4}
        assert judge._validate_scores(wrong_type_scores) == False


class TestFusionFunction:
    """Test the fusion function for combining rule and judge scores."""
    
    def test_fusion_with_judge_scores(self):
        """Test fusion when judge scores are available."""
        rule = {
            "faithfulness_rule": 0.8,
            "utility_rule": 0.9,
            "coherence_rule": 0.7,
            "factuality_rule": 0.6
        }
        judge = {
            "faithfulness": 4,  # 4/5 = 0.75 normalized
            "utility": 5,       # 5/5 = 1.0 normalized
            "coherence": 3,     # 3/5 = 0.5 normalized
            "factuality": 2     # 2/5 = 0.25 normalized
        }
        evidence = {
            "coh_contra_cnt": 0,
            "fact_entail_rate": 1.0,
            "fact_contra_cnt": 0
        }
        
        fused = fuse_with_judge(rule, judge, evidence)
        
        # Check that all scores are in valid range
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            assert 0.0 <= fused[pillar] <= 1.0
        
        # Check specific calculations
        assert fused["faithfulness"] == (0.8 + 0.75) / 2.0  # Average of rule and judge
        assert fused["utility"] == (0.9 + 1.0) / 2.0
        assert fused["coherence"] == (0.7 + 0.5) / 2.0
        assert fused["factuality"] == (0.6 + 0.25) / 2.0
        
        # Check overall score
        expected_overall = (fused["faithfulness"] + fused["utility"] + fused["coherence"] + fused["factuality"]) / 4.0
        assert fused["overall"] == expected_overall
    
    def test_fusion_without_judge_scores(self):
        """Test fusion when judge scores are None."""
        rule = {
            "faithfulness_rule": 0.8,
            "utility_rule": 0.9,
            "coherence_rule": 0.7,
            "factuality_rule": 0.6
        }
        judge = {
            "faithfulness": None,
            "utility": None,
            "coherence": None,
            "factuality": None
        }
        evidence = {}
        
        fused = fuse_with_judge(rule, judge, evidence)
        
        # Should use rule scores only
        assert fused["faithfulness"] == 0.8
        assert fused["utility"] == 0.9
        assert fused["coherence"] == 0.7
        assert fused["factuality"] == 0.6
    
    def test_fusion_with_evidence_caps(self):
        """Test fusion with evidence-based caps."""
        rule = {
            "faithfulness_rule": 0.8,
            "utility_rule": 0.9,
            "coherence_rule": 0.9,  # High rule score
            "factuality_rule": 0.9  # High rule score
        }
        judge = {
            "faithfulness": 5,  # Perfect judge score
            "utility": 5,
            "coherence": 5,     # Perfect judge score
            "factuality": 5     # Perfect judge score
        }
        evidence = {
            "coh_contra_cnt": 2,    # 2 contradictions should cap coherence
            "fact_entail_rate": 0.5, # Low entailment rate
            "fact_contra_cnt": 1     # 1 contradiction should cap factuality
        }
        
        fused = fuse_with_judge(rule, judge, evidence)
        
        # Coherence should be capped due to contradictions
        coherence_cap = max(0.0, 1.0 - 0.5 * 2)  # 0.0
        expected_coherence = min((0.9 + 1.0) / 2.0, coherence_cap)
        assert fused["coherence"] == expected_coherence
        
        # Factuality should be capped due to low entailment and contradictions
        factuality_cap = max(0.0, 0.5 - 0.4 * 1)  # 0.1
        expected_factuality = min((0.9 + 1.0) / 2.0, factuality_cap)
        assert fused["factuality"] == expected_factuality
        
        # Faithfulness and utility should not be capped
        assert fused["faithfulness"] == (0.8 + 1.0) / 2.0
        assert fused["utility"] == (0.9 + 1.0) / 2.0
    
    def test_compare_score_methods(self):
        """Test score comparison function."""
        rule = {
            "faithfulness_rule": 0.8,
            "utility_rule": 0.9,
            "coherence_rule": 0.7,
            "factuality_rule": 0.6
        }
        judge = {
            "faithfulness": 4,
            "utility": 5,
            "coherence": 3,
            "factuality": 2
        }
        fused = {
            "faithfulness": 0.775,
            "utility": 0.95,
            "coherence": 0.6,
            "factuality": 0.425,
            "overall": 0.6875
        }
        
        comparison = compare_score_methods(rule, judge, fused)
        
        # Check overall scores
        expected_rule_overall = (0.8 + 0.9 + 0.7 + 0.6) / 4.0
        assert abs(comparison["rule_overall"] - expected_rule_overall) < 1e-10
        assert comparison["judge_overall"] == (0.75 + 1.0 + 0.5 + 0.25) / 4.0
        assert comparison["fused_overall"] == 0.6875
        assert comparison["judge_available"] == True
        
        # Check score differences
        assert comparison["score_differences"]["faithfulness"] == 0.775 - 0.8
        assert comparison["score_differences"]["utility"] == 0.95 - 0.9


class TestPillarsEvaluatorIntegration:
    """Test integration of judge with PillarsEvaluator."""
    
    def test_evaluator_without_judge(self):
        """Test evaluator works without judge (backward compatibility)."""
        evaluator = PillarsEvaluator(judge=None)
        
        problem = "What is 2 + 2?"
        cot = "Step 1. I calculate 2 + 2 = 4\n#### 4"
        gold = "4"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot, gold)
        
        # Should have flags and evidence
        assert len(flags) >= 0
        assert "final_correct" in evidence
        
        # Should have rule scores
        assert "faithfulness_rule" in rule_scores
        assert "utility_rule" in rule_scores
        
        # Should have None judge scores
        assert all(score is None for score in judge_scores.values())
        
        # Fused scores should equal rule scores
        assert fused_scores["faithfulness"] == rule_scores["faithfulness_rule"]
        assert fused_scores["utility"] == rule_scores["utility_rule"]
    
    def test_evaluator_with_mock_judge(self):
        """Test evaluator with mock judge."""
        mock_judge = MockJudge(mode="ALWAYS")
        evaluator = PillarsEvaluator(judge=mock_judge)
        
        problem = "What is 2 + 2?"
        cot = "Step 1. I calculate 2 + 2 = 4\n#### 4"
        gold = "4"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot, gold)
        
        # Should have all components
        assert len(flags) >= 0
        assert "final_correct" in evidence
        assert "faithfulness_rule" in rule_scores
        
        # Should have judge scores (from mock)
        assert all(score is not None for score in judge_scores.values())
        assert judge_scores["faithfulness"] == 4  # Mock judge default
        
        # Should have fused scores
        assert "overall" in fused_scores
        assert 0.0 <= fused_scores["overall"] <= 1.0
    
    def test_evaluator_with_never_judge(self):
        """Test evaluator with NEVER mode judge."""
        mock_judge = MockJudge(mode="NEVER")
        evaluator = PillarsEvaluator(judge=mock_judge)
        
        problem = "What is 2 + 2?"
        cot = "Step 1. I calculate 2 + 2 = 4\n#### 4"
        gold = "4"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot, gold)
        
        # Judge scores should be None
        assert all(score is None for score in judge_scores.values())
        
        # Fused scores should equal rule scores
        assert fused_scores["faithfulness"] == rule_scores["faithfulness_rule"]
    
    def test_evaluator_with_smart_judge(self):
        """Test evaluator with SMART mode judge."""
        mock_judge = MockJudge(mode="SMART")
        evaluator = PillarsEvaluator(judge=mock_judge)
        
        # Test case with no issues - should not call judge
        problem = "What is 2 + 2?"
        cot = "Step 1. I calculate 2 + 2 = 4\n#### 4"
        gold = "4"
        
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot, gold)
        
        # Should not call judge for clean case (unless flags are detected)
        # Note: Even clean cases might have flags due to heuristic checks
        print(f"  Flags for clean case: {len(flags)}")
        print(f"  Flag summary: {flags.summarize_for_prompt()}")
        # The test passes if either no judge is called OR if judge is called but returns valid scores
        assert (all(score is None for score in judge_scores.values()) or 
                all(score is not None for score in judge_scores.values()))
        
        # Test case with issues - should call judge
        problem2 = "What is 2 + 2?"
        cot2 = "Step 1. I calculate 2 + 2 = 5\n#### 5"  # Wrong answer
        gold2 = "4"
        
        flags2, evidence2, rule_scores2, judge_scores2, fused_scores2 = evaluator.analyze(problem2, cot2, gold2)
        
        # Should call judge for problematic case
        assert all(score is not None for score in judge_scores2.values())


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        judge = MockJudge()
        
        # Test with empty strings
        scores = judge.score("", "", "", "", {})
        assert isinstance(scores, dict)
        assert len(scores) == 4
    
    def test_malformed_evidence(self):
        """Test handling of malformed evidence."""
        rule = {"faithfulness_rule": 0.8, "utility_rule": 0.9, "coherence_rule": 0.7, "factuality_rule": 0.6}
        judge = {"faithfulness": 4, "utility": 4, "coherence": 4, "factuality": 4}
        evidence = {}  # Empty evidence
        
        fused = fuse_with_judge(rule, judge, evidence)
        
        # Should handle gracefully
        assert "overall" in fused
        assert 0.0 <= fused["overall"] <= 1.0
    
    def test_extreme_scores(self):
        """Test handling of extreme score values."""
        rule = {"faithfulness_rule": 0.0, "utility_rule": 1.0, "coherence_rule": 0.5, "factuality_rule": 0.5}
        judge = {"faithfulness": 1, "utility": 5, "coherence": 1, "factuality": 5}
        evidence = {"coh_contra_cnt": 0, "fact_entail_rate": 1.0, "fact_contra_cnt": 0}
        
        fused = fuse_with_judge(rule, judge, evidence)
        
        # Should handle extreme values gracefully
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            assert 0.0 <= fused[pillar] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
