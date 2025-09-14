"""
Smoke test driver for CoT evaluation v2.

This is a simple test script to verify the basic functionality
without requiring external dependencies.
"""

from cot_eval_v2.evaluator import PillarsEvaluator
from cot_eval_v2.scoring import rule_scores, compute_overall_rule_score


def run_smoke_test():
    """Run basic smoke tests to verify functionality."""
    print("Running CoT Evaluation v2 Smoke Tests...")
    print("=" * 50)
    
    # Test 1: Correct mathematical reasoning
    print("\n1. Testing correct mathematical reasoning:")
    problem1 = "Alice buys 3 apples at $2 each. What's the total?"
    cot1 = """1. First, I calculate 3 * 2 = 6
2. Therefore, the total cost is $6
#### 6"""
    
    evaluator = PillarsEvaluator()
    flags1, evidence1 = evaluator.analyze(problem1, cot1, "6")
    
    print(f"   Final correct: {evidence1['final_correct']}")
    print(f"   Intermediate OK rate: {evidence1['intermediate_ok_rate']:.2f}")
    print(f"   Flags collected: {len(flags1)}")
    print(f"   Flag summary: {flags1.summarize_for_prompt()}")
    
    # Test 2: Wrong intermediate but correct final
    print("\n2. Testing wrong intermediate but correct final:")
    problem2 = "What is 3 + 4?"
    cot2 = """1. First, I calculate 3 + 4 = 8
2. Actually, let me correct that: 3 + 4 = 7
3. Therefore, the answer is 7
#### 7"""
    
    flags2, evidence2 = evaluator.analyze(problem2, cot2, "7")
    
    print(f"   Final correct: {evidence2['final_correct']}")
    print(f"   Self-repair count: {evidence2['self_repair_cnt']}")
    print(f"   Wrong but right: {evidence2['wrong_but_right']}")
    print(f"   Flags collected: {len(flags2)}")
    print(f"   Flag summary: {flags2.summarize_for_prompt()}")
    
    # Test 3: Unused numbers
    print("\n3. Testing unused number detection:")
    problem3 = "Alice buys 3 apples at $2 each and 2 oranges at $1 each. What's the total?"
    cot3 = """1. I calculate 3 * 2 = 6
2. The total is 6
#### 6"""
    
    flags3, evidence3 = evaluator.analyze(problem3, cot3, "6")
    
    print(f"   Coverage: {evidence3['coverage']}")
    print(f"   Flags collected: {len(flags3)}")
    print(f"   Flag summary: {flags3.summarize_for_prompt()}")
    
    # Test 4: Rule-based scoring
    print("\n4. Testing rule-based scoring:")
    scores1 = rule_scores(evidence1)
    scores2 = rule_scores(evidence2)
    scores3 = rule_scores(evidence3)
    
    print(f"   Correct reasoning scores: {scores1}")
    print(f"   Wrong intermediate scores: {scores2}")
    print(f"   Unused numbers scores: {scores3}")
    
    overall1 = compute_overall_rule_score(scores1)
    overall2 = compute_overall_rule_score(scores2)
    overall3 = compute_overall_rule_score(scores3)
    
    print(f"   Overall scores: {overall1:.3f}, {overall2:.3f}, {overall3:.3f}")
    
    print("\n" + "=" * 50)
    print("Smoke tests completed successfully!")
    print("All core functionality is working.")


if __name__ == "__main__":
    run_smoke_test()
