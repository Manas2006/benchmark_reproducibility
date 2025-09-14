#!/usr/bin/env python3
"""
Smoke test for the new four-pillar CoT evaluation pipeline.

Tests the complete pipeline end-to-end to ensure migration is successful.
"""

import os
import sys
import time

# Set up test environment
os.environ['JUDGE_GATING'] = 'NEVER'  # Disable judge for smoke test
os.environ['NLI_ENABLED'] = '0'       # Disable NLI for smoke test
os.environ['EVAL_MODE'] = 'PILLARS_ONLY'

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'app'))


def test_nli_factory():
    """Test NLI factory functionality."""
    print("🧪 Testing NLI Factory...")
    
    try:
        from cot_eval_v2.checks.nli_factory import get_device_info, resolve_device
        
        # Test device info
        info = get_device_info()
        print(f"  Device info: {info}")
        
        # Test device resolution
        device = resolve_device()
        print(f"  Resolved device: {device}")
        
        print("✅ NLI Factory test passed")
        return True
        
    except Exception as e:
        print(f"❌ NLI Factory test failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("🧪 Testing Configuration...")
    
    try:
        from cot_eval_v2.config import PillarsConfig
        
        config = PillarsConfig()
        config.print_config()
        
        if config.validate():
            print("✅ Configuration test passed")
            return True
        else:
            print("❌ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_pillars_runner():
    """Test PillarsRunner functionality."""
    print("🧪 Testing PillarsRunner...")
    
    try:
        from cot_eval_v2.pillars_runner import PillarsRunner
        
        # Initialize runner
        runner = PillarsRunner(
            evaluator=None,
            llm_fn=None,
            gating="NEVER",
            budget=100,
            diagnostic=False
        )
        
        print(f"  Runner initialized with gating: {runner.gating}")
        
        # Test single sample
        problem = "What is 2 + 2?"
        cot_text = "I need to add 2 and 2.\n2 + 2 = 4\nSo the answer is 4."
        gold = "4"
        
        result = runner.run(problem, cot_text, gold)
        
        print(f"  Single sample result:")
        print(f"    Scores: {result.scores}")
        print(f"    Flags: {len(result.flags)}")
        print(f"    Evidence keys: {list(result.evidence.keys())}")
        
        # Test batch analysis
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
        
        batch_result = runner.run_batch(samples, "smoke_test_job")
        
        print(f"  Batch result:")
        print(f"    Job ID: {batch_result.job_id}")
        print(f"    Analysis method: {batch_result.analysis_method}")
        print(f"    Total samples: {batch_result.summary.total_samples}")
        print(f"    Judge calls: {batch_result.summary.judge_budget_used}")
        
        print("✅ PillarsRunner test passed")
        return True
        
    except Exception as e:
        print(f"❌ PillarsRunner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_judge_validation():
    """Test judge score validation."""
    print("🧪 Testing Judge Validation...")
    
    try:
        from cot_eval_v2.judge import Judge
        
        # Test valid scores
        valid_scores = {
            "faithfulness": 3,
            "utility": 4,
            "coherence": 2,
            "factuality": 5
        }
        
        normalized = Judge.validate_and_normalize_judge(valid_scores)
        print(f"  Valid scores normalized: {normalized}")
        
        # Test invalid scores
        invalid_scores = {
            "faithfulness": 0,
            "utility": 6,
            "coherence": "invalid",
            "factuality": 3.5
        }
        
        normalized_invalid = Judge.validate_and_normalize_judge(invalid_scores)
        print(f"  Invalid scores normalized: {normalized_invalid}")
        
        print("✅ Judge validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Judge validation test failed: {e}")
        return False


def test_score_fusion():
    """Test score fusion logic."""
    print("🧪 Testing Score Fusion...")
    
    try:
        from cot_eval_v2.scoring import fuse_with_judge
        
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
        
        print(f"  Fused scores: {fused}")
        print(f"  Overall score: {fused['overall']}")
        
        # Test without judge scores
        fused_no_judge = fuse_with_judge(rule_scores, {k: None for k in judge_scores}, evidence)
        print(f"  Fused without judge: {fused_no_judge}")
        
        print("✅ Score fusion test passed")
        return True
        
    except Exception as e:
        print(f"❌ Score fusion test failed: {e}")
        return False


def test_schema_validation():
    """Test Pydantic schema validation."""
    print("🧪 Testing Schema Validation...")
    
    try:
        from schemas import PillarsScores, PillarsEntry, PillarsFlag
        
        # Test PillarsScores
        scores = PillarsScores(
            faithfulness=0.8,
            utility=0.6,
            coherence=0.7,
            factuality=0.9,
            overall=0.75
        )
        print(f"  PillarsScores created: {scores}")
        
        # Test PillarsFlag
        flag = PillarsFlag(
            pillar="utility",
            step="reasoning",
            issue="arithmetic_error",
            details={"error": "2 + 2 = 5 is incorrect"}
        )
        print(f"  PillarsFlag created: {flag}")
        
        # Test PillarsEntry
        entry = PillarsEntry(
            scores=scores,
            flags=[flag],
            evidence={"final_correct": True},
            config_snapshot={"model": "test"}
        )
        print(f"  PillarsEntry created with {len(entry.flags)} flags")
        
        print("✅ Schema validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("🚀 Starting Four-Pillar CoT Evaluation Smoke Tests")
    print("=" * 60)
    
    tests = [
        test_configuration,
        test_nli_factory,
        test_judge_validation,
        test_score_fusion,
        test_schema_validation,
        test_pillars_runner,
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        print()
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print(f"🏁 Smoke Test Results:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.2f}s")
    
    if failed == 0:
        print("🎉 All smoke tests passed! Migration appears successful.")
        return 0
    else:
        print("❌ Some smoke tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
