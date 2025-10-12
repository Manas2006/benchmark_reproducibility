"""
Tests for the new four-pillar CoT evaluation pipeline migration.

Tests the complete pipeline from PillarsRunner through to API responses,
ensuring no legacy CQS dependencies remain.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

# Set up test environment
os.environ['JUDGE_GATING'] = 'NEVER'  # Disable judge for testing
# NLI has been removed from the system
os.environ['EVAL_MODE'] = 'PILLARS_ONLY'


class TestPillarsRunner:
    """Test the PillarsRunner class."""
    
    def test_pillars_runner_initialization(self):
        """Test PillarsRunner initializes correctly."""
        from backend.app.cot_eval_v2.pillars_runner import PillarsRunner
        
        runner = PillarsRunner(
            evaluator=None,
            llm_fn=None,
            gating="NEVER",
            budget=100,
            diagnostic=False
        )
        
        assert runner.gating == "NEVER"
        assert runner.budget == 100
        assert runner.judge_calls_made == 0
        assert runner.judge is None  # No LLM function provided
    
    def test_single_sample_analysis(self):
        """Test analysis of a single sample."""
        from backend.app.cot_eval_v2.pillars_runner import PillarsRunner
        
        runner = PillarsRunner(gating="NEVER", budget=100)
        
        # Test sample
        problem = "What is 2 + 2?"
        cot_text = "I need to add 2 and 2.\n2 + 2 = 4\nSo the answer is 4."
        gold = "4"
        
        result = runner.run(problem, cot_text, gold)
        
        # Check result structure
        assert hasattr(result, 'scores')
        assert hasattr(result, 'flags')
        assert hasattr(result, 'evidence')
        assert hasattr(result, 'config_snapshot')
        
        # Check scores are in valid range
        assert 0.0 <= result.scores.faithfulness <= 1.0
        assert 0.0 <= result.scores.utility <= 1.0
        assert 0.0 <= result.scores.coherence <= 1.0
        assert 0.0 <= result.scores.factuality <= 1.0
        assert 0.0 <= result.scores.overall <= 1.0
    
    def test_batch_analysis(self):
        """Test batch analysis of multiple samples."""
        from backend.app.cot_eval_v2.pillars_runner import PillarsRunner
        
        runner = PillarsRunner(gating="NEVER", budget=100)
        
        samples = [
            {
                "problem": "What is 2 + 2?",
                "cot_text": "2 + 2 = 4",
                "gold": "4"
            },
            {
                "problem": "What is 3 * 3?",
                "cot_text": "3 * 3 = 9",
                "gold": "9"
            }
        ]
        
        result = runner.run_batch(samples, "test_job")
        
        # Check response structure
        assert hasattr(result, 'job_id')
        assert hasattr(result, 'per_sample')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'analysis_method')
        
        assert result.job_id == "test_job"
        assert result.analysis_method == "pillars_v2"
        assert len(result.per_sample) == 2
        
        # Check summary statistics
        assert result.summary.total_samples == 2
        assert result.summary.judge_budget_total == 100
        assert result.summary.judge_budget_used == 0  # No judge calls in NEVER mode


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        from backend.app.cot_eval_v2.config import PillarsConfig
        
        # Test valid configuration
        with patch.dict(os.environ, {
            'EVAL_MODE': 'PILLARS_ONLY',
            'JUDGE_GATING': 'SMART',
            'JUDGE_BUDGET': '1000'
        }):
            config = PillarsConfig()
            assert config.validate() is True
    
    def test_invalid_config(self):
        """Test invalid configuration detection."""
        from backend.app.cot_eval_v2.config import PillarsConfig
        
        # Test invalid mode
        with patch.dict(os.environ, {'EVAL_MODE': 'INVALID_MODE'}):
            config = PillarsConfig()
            assert config.validate() is False
    
    def test_judge_mode_override(self):
        """Test judge mode override based on JUDGE_ENABLED."""
        from backend.app.cot_eval_v2.config import PillarsConfig
        
        # Test JUDGE_ENABLED=0 overrides gating
        with patch.dict(os.environ, {
            'JUDGE_GATING': 'SMART',
            'JUDGE_ENABLED': '0'
        }):
            config = PillarsConfig()
            assert config.get_judge_mode() == "NEVER"


class TestJudgeValidation:
    """Test judge score validation."""
    
    def test_judge_score_validation(self):
        """Test judge score validation and normalization."""
        from backend.app.cot_eval_v2.judge import Judge
        
        # Test valid scores
        valid_scores = {
            "faithfulness": 3,
            "utility": 4,
            "coherence": 2,
            "factuality": 5
        }
        
        normalized = Judge.validate_and_normalize_judge(valid_scores)
        assert normalized["faithfulness"] == 3
        assert normalized["utility"] == 4
        assert normalized["coherence"] == 2
        assert normalized["factuality"] == 5
    
    def test_judge_score_clamping(self):
        """Test score clamping to 1-5 range."""
        from backend.app.cot_eval_v2.judge import Judge
        
        # Test out-of-range scores
        invalid_scores = {
            "faithfulness": 0,  # Too low
            "utility": 6,       # Too high
            "coherence": "invalid",  # Invalid type
            "factuality": 3.5   # Float
        }
        
        normalized = Judge.validate_and_normalize_judge(invalid_scores)
        assert normalized["faithfulness"] == 1  # Clamped up
        assert normalized["utility"] == 5       # Clamped down
        assert normalized["coherence"] is None  # Invalid input
        assert normalized["factuality"] == 3    # Converted to int


class TestScoreFusion:
    """Test score fusion logic."""
    
    def test_fuse_with_judge(self):
        """Test score fusion with judge scores."""
        from backend.app.cot_eval_v2.scoring import fuse_with_judge
        
        rule_scores = {
            "faithfulness_rule": 0.8,
            "utility_rule": 0.6,
            "coherence_rule": 0.7,
            "factuality_rule": 0.9
        }
        
        judge_scores = {
            "faithfulness": 4,
            "utility": 3,
            "coherence": 5,
            "factuality": 4
        }
        
        evidence = {
            "coh_contra_cnt": 0,
            "fact_entail_rate": 1.0,
            "fact_contra_cnt": 0
        }
        
        fused = fuse_with_judge(rule_scores, judge_scores, evidence)
        
        # Check all pillars are present
        assert "faithfulness" in fused
        assert "utility" in fused
        assert "coherence" in fused
        assert "factuality" in fused
        assert "overall" in fused
        
        # Check scores are in valid range
        for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
            assert 0.0 <= fused[pillar] <= 1.0
        
        # Check overall is average of pillars
        expected_overall = sum(fused[p] for p in ["faithfulness", "utility", "coherence", "factuality"]) / 4.0
        assert abs(fused["overall"] - expected_overall) < 1e-10
    
    def test_fuse_without_judge(self):
        """Test score fusion without judge scores."""
        from backend.app.cot_eval_v2.scoring import fuse_with_judge
        
        rule_scores = {
            "faithfulness_rule": 0.8,
            "utility_rule": 0.6,
            "coherence_rule": 0.7,
            "factuality_rule": 0.9
        }
        
        judge_scores = {
            "faithfulness": None,
            "utility": None,
            "coherence": None,
            "factuality": None
        }
        
        evidence = {}
        
        fused = fuse_with_judge(rule_scores, judge_scores, evidence)
        
        # Should use rule scores directly
        assert fused["faithfulness"] == 0.8
        assert fused["utility"] == 0.6
        assert fused["coherence"] == 0.7
        assert fused["factuality"] == 0.9


class TestSchemaValidation:
    """Test Pydantic schema validation."""
    
    def test_pillars_scores_validation(self):
        """Test PillarsScores validation."""
        from backend.app.schemas import PillarsScores
        
        # Valid scores
        scores = PillarsScores(
            faithfulness=0.8,
            utility=0.6,
            coherence=0.7,
            factuality=0.9,
            overall=0.75
        )
        assert scores.faithfulness == 0.8
        assert scores.overall == 0.75
    
    def test_pillars_scores_validation_error(self):
        """Test PillarsScores validation with invalid values."""
        from backend.app.schemas import PillarsScores
        from pydantic import ValidationError
        
        # Invalid scores (out of range)
        with pytest.raises(ValidationError):
            PillarsScores(
                faithfulness=1.5,  # Too high
                utility=0.6,
                coherence=0.7,
                factuality=0.9,
                overall=0.75
            )
    
    def test_pillars_entry_validation(self):
        """Test PillarsEntry validation."""
        from backend.app.schemas import PillarsEntry, PillarsScores, PillarsFlag
        
        scores = PillarsScores(
            faithfulness=0.8,
            utility=0.6,
            coherence=0.7,
            factuality=0.9,
            overall=0.75
        )
        
        flags = [
            PillarsFlag(
                pillar="utility",
                step=1,
                issue="arithmetic_error",
                details="2 + 2 = 5 is incorrect"
            )
        ]
        
        entry = PillarsEntry(
            scores=scores,
            flags=flags,
            evidence={"final_correct": True},
            config_snapshot={"model": "test"}
        )
        
        assert entry.scores.overall == 0.75
        assert len(entry.flags) == 1
        assert entry.flags[0].issue == "arithmetic_error"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

