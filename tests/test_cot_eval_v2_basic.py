"""
Basic unit tests for CoT evaluation v2.

Tests the core functionality without external dependencies.
"""

import pytest
import sys
import os

# Add the backend app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

from cot_eval_v2.flags import Flag, FlagCollector
from cot_eval_v2.evaluator import PillarsEvaluator, split_steps, check_final_answer
from cot_eval_v2.scoring import rule_scores, compute_overall_rule_score
from cot_eval_v2.checks.arithmetic import check_step_equations
from cot_eval_v2.checks.coverage import number_coverage
from cot_eval_v2.checks.heuristics import wrong_but_right, self_repair_markers, shortcut_signature


class TestFlagCollector:
    """Test FlagCollector functionality."""
    
    def test_empty_collector(self):
        """Test empty flag collector."""
        collector = FlagCollector()
        assert len(collector) == 0
        assert not collector.has_flags()
        assert collector.as_dict() == {
            "faithfulness": [],
            "utility": [],
            "coherence": [],
            "factuality": []
        }
    
    def test_add_flags(self):
        """Test adding flags to collector."""
        collector = FlagCollector()
        collector.add("utility", "step=1", "arithmetic_error", {"bad_equations": 2})
        collector.add("coherence", "step=2", "contradiction", {"score": 0.9})
        
        assert len(collector) == 2
        assert collector.has_flags()
        
        flags_dict = collector.as_dict()
        assert len(flags_dict["utility"]) == 1
        assert len(flags_dict["coherence"]) == 1
        assert len(flags_dict["faithfulness"]) == 0
        assert len(flags_dict["factuality"]) == 0
    
    def test_summarize_for_prompt(self):
        """Test prompt summarization."""
        collector = FlagCollector()
        collector.add("utility", "step=1", "arithmetic_error", {"bad_equations": 2})
        collector.add("coherence", "step=2", "contradiction", {"score": 0.9})
        
        summary = collector.summarize_for_prompt()
        assert "- [UTILITY] step=1: arithmetic_error {bad_equations=2}" in summary
        assert "- [COHERENCE] step=2: contradiction {score=0.9}" in summary
    
    def test_invalid_pillar(self):
        """Test invalid pillar raises error."""
        collector = FlagCollector()
        with pytest.raises(ValueError):
            collector.add("invalid_pillar", "step=1", "test_issue")


class TestStepSplitting:
    """Test step splitting functionality."""
    
    def test_numbered_steps(self):
        """Test splitting numbered steps."""
        cot_text = """1. First, I need to calculate 3 * 2 = 6
2. Then I add 6 + 4 = 10
3. Therefore, the answer is 10"""
        
        steps = split_steps(cot_text)
        assert len(steps) == 3
        assert "First, I need to calculate 3 * 2 = 6" in steps[0]
        assert "Then I add 6 + 4 = 10" in steps[1]
        assert "Therefore, the answer is 10" in steps[2]
    
    def test_bullet_steps(self):
        """Test splitting bullet point steps."""
        cot_text = """* First, I need to calculate 3 * 2 = 6
- Then I add 6 + 4 = 10
• Therefore, the answer is 10"""
        
        steps = split_steps(cot_text)
        assert len(steps) == 3
    
    def test_sentence_splitting(self):
        """Test fallback to sentence splitting."""
        cot_text = """First, I need to calculate 3 * 2 = 6. Then I add 6 + 4 = 10. Therefore, the answer is 10."""
        
        steps = split_steps(cot_text)
        assert len(steps) == 3
    
    def test_empty_text(self):
        """Test empty text handling."""
        assert split_steps("") == []
        assert split_steps("   ") == []


class TestFinalAnswerChecking:
    """Test final answer checking functionality."""
    
    def test_boxed_answer(self):
        """Test #### delimiter answer extraction."""
        cot_text = """Let me solve this step by step.
First, I calculate 3 * 2 = 6.
Then I add 6 + 4 = 10.
#### 10"""
        
        assert check_final_answer(cot_text, "10") == True
        assert check_final_answer(cot_text, "11") == False
    
    def test_numeric_comparison(self):
        """Test numeric answer comparison."""
        cot_text = """The calculation is 3 * 2 = 6.
#### 6.0"""
        
        assert check_final_answer(cot_text, "6") == True
        assert check_final_answer(cot_text, "6.0") == True
        assert check_final_answer(cot_text, "6.00") == True
    
    def test_no_gold_truth(self):
        """Test with no ground truth."""
        cot_text = """The answer is 10."""
        assert check_final_answer(cot_text, None) == False


class TestArithmeticChecking:
    """Test arithmetic checking functionality."""
    
    def test_correct_equations(self):
        """Test correct arithmetic equations."""
        step = "First, I calculate 3 + 4 = 7 and 2 * 5 = 10."
        
        try:
            result = check_step_equations(step)
            assert result["ok"] == 2
            assert result["bad"] == 0
            assert len(result["examples"]) == 2
        except ImportError:
            pytest.skip("SymPy not available")
    
    def test_incorrect_equations(self):
        """Test incorrect arithmetic equations."""
        step = "I calculate 3 + 4 = 8 and 2 * 5 = 9."
        
        try:
            result = check_step_equations(step)
            assert result["ok"] == 0
            assert result["bad"] == 2
            assert len(result["examples"]) == 2
        except ImportError:
            pytest.skip("SymPy not available")
    
    def test_mixed_equations(self):
        """Test mixed correct and incorrect equations."""
        step = "I calculate 3 + 4 = 7 and 2 * 5 = 9."
        
        try:
            result = check_step_equations(step)
            assert result["ok"] == 1
            assert result["bad"] == 1
        except ImportError:
            pytest.skip("SymPy not available")


class TestCoverageChecking:
    """Test number coverage functionality."""
    
    def test_full_coverage(self):
        """Test when all numbers are used."""
        problem = "Alice buys 3 apples at $2 each. What's the total?"
        steps = ["First, I calculate 3 * 2 = 6", "Therefore, the total is 6"]
        
        coverage = number_coverage(problem, steps)
        assert len(coverage["given"]) > 0
        assert len(coverage["unused"]) == 0
    
    def test_unused_numbers(self):
        """Test when some numbers are unused."""
        problem = "Alice buys 3 apples at $2 each and 2 oranges at $1 each. What's the total?"
        steps = ["I calculate 3 * 2 = 6", "The total is 6"]
        
        coverage = number_coverage(problem, steps)
        assert len(coverage["unused"]) > 0  # Should have unused numbers


class TestHeuristics:
    """Test heuristic functions."""
    
    def test_wrong_but_right(self):
        """Test wrong-but-right detection."""
        assert wrong_but_right(True, 0.5) == True  # Correct final, low intermediate
        assert wrong_but_right(True, 0.8) == False  # Correct final, high intermediate
        assert wrong_but_right(False, 0.5) == False  # Wrong final
    
    def test_self_repair_markers(self):
        """Test self-repair marker detection."""
        assert self_repair_markers("Actually, let me correct that") == True
        assert self_repair_markers("I made a mistake") == True
        assert self_repair_markers("The answer is 10") == False
    
    def test_shortcut_signature(self):
        """Test shortcut signature detection."""
        problem = "What is 3 + 4? The answer is 7."
        cot_text = "Let me think about this."
        final_answer = "7"
        
        assert shortcut_signature(problem, cot_text, final_answer) == True


class TestPillarsEvaluator:
    """Test main evaluator functionality."""
    
    def test_correct_math(self):
        """Test correct mathematical reasoning."""
        problem = "Alice buys 3 apples at $2 each. What's the total?"
        cot_text = """1. First, I calculate 3 * 2 = 6
2. Therefore, the total cost is $6
#### 6"""
        
        evaluator = PillarsEvaluator(nli_pipe=None, use_nli=False)
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot_text, "6")
        
        assert evidence["final_correct"] == True
        assert evidence["intermediate_ok_rate"] > 0.5
        # Should have minimal flags for correct reasoning
        assert len(flags) < 3  # Allow some minor flags
    
    def test_wrong_intermediate_correct_final(self):
        """Test wrong intermediate but correct final answer."""
        problem = "What is 3 + 4?"
        cot_text = """1. First, I calculate 3 + 4 = 8
2. Actually, let me correct that: 3 + 4 = 7
3. Therefore, the answer is 7
#### 7"""
        
        evaluator = PillarsEvaluator(nli_pipe=None, use_nli=False)
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot_text, "7")
        
        assert evidence["final_correct"] == True
        assert evidence["self_repair_cnt"] > 0
        assert evidence["wrong_but_right"] == True
        
        # Should have faithfulness flags
        faithfulness_flags = flags.get_flags_by_pillar("faithfulness")
        assert len(faithfulness_flags) > 0
    
    def test_unused_numbers(self):
        """Test unused number detection."""
        problem = "Alice buys 3 apples at $2 each and 2 oranges at $1 each. What's the total?"
        cot_text = """1. I calculate 3 * 2 = 6
2. The total is 6
#### 6"""
        
        evaluator = PillarsEvaluator(nli_pipe=None, use_nli=False)
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot_text, "6")
        
        # Should have utility flags for unused numbers
        utility_flags = flags.get_flags_by_pillar("utility")
        unused_flags = [f for f in utility_flags if f.issue == "unused_given_number"]
        assert len(unused_flags) > 0
    
    def test_empty_cot(self):
        """Test empty CoT handling."""
        problem = "What is 2 + 2?"
        cot_text = ""
        
        evaluator = PillarsEvaluator(nli_pipe=None, use_nli=False)
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(problem, cot_text, "4")
        
        assert evidence["final_correct"] == False
        assert evidence["intermediate_ok_rate"] == 0.0
        # Empty CoT should have some flags (unused numbers, etc.)
        assert len(flags) > 0


class TestRuleScoring:
    """Test rule-based scoring functionality."""
    
    def test_correct_reasoning_scores(self):
        """Test scores for correct reasoning."""
        evidence = {
            "final_correct": True,
            "intermediate_ok_rate": 1.0,
            "coh_contra_cnt": 0,
            "redund_cnt": 0,
            "coverage": {"given": ["3", "2"], "used": ["3", "2"], "unused": []},
            "wrong_but_right": False,
            "self_repair_cnt": 0
        }
        
        scores = rule_scores(evidence)
        
        assert scores["utility_rule"] > 0.8
        assert scores["coherence_rule"] > 0.9
        assert scores["factuality_rule"] > 0.9
        assert scores["faithfulness_rule"] > 0.9
    
    def test_wrong_reasoning_scores(self):
        """Test scores for wrong reasoning."""
        evidence = {
            "final_correct": False,
            "intermediate_ok_rate": 0.3,
            "coh_contra_cnt": 1,
            "redund_cnt": 2,
            "coverage": {"given": ["3", "2"], "used": ["3"], "unused": ["2"]},
            "wrong_but_right": False,
            "self_repair_cnt": 0
        }
        
        scores = rule_scores(evidence)
        
        assert scores["utility_rule"] < 0.5
        assert scores["coherence_rule"] < 0.7
        assert scores["factuality_rule"] < 0.7
    
    def test_overall_score(self):
        """Test overall score computation."""
        scores = {
            "faithfulness_rule": 0.8,
            "utility_rule": 0.7,
            "coherence_rule": 0.9,
            "factuality_rule": 0.6
        }
        
        overall = compute_overall_rule_score(scores)
        assert overall == 0.75  # Average of the four scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
