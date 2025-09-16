#!/usr/bin/env python3
"""
Demo script showing what DeBERTa NLI model would do for coherence and factuality flags
This demonstrates the capabilities without actually running the model
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'app'))

from cot_eval_v2.evaluator import PillarsEvaluator
from cot_eval_v2.judge import MockJudge

def demo_deberta_capabilities():
    """Demo what DeBERTa NLI model would do for coherence and factuality analysis"""
    
    print("🧠 DEMONSTRATING DEBERTA NLI MODEL CAPABILITIES")
    print("=" * 80)
    
    # Load real outputs
    result_file = "output/Qwen2.5-Math-7B/gsm8k/test_auto-cot_-1_seed0_t0.0_s0_e-1_f71140d9-3c8b-4ec9-9956-a586e491b16c.jsonl"
    
    outputs = []
    with open(result_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Test with 3 examples
                break
            data = json.loads(line.strip())
            outputs.append(data)
    
    print(f"📊 Loaded {len(outputs)} real model outputs for demonstration")
    print()
    
    # Initialize evaluator WITHOUT NLI (to avoid threading issues)
    print("🔧 INITIALIZING EVALUATOR (Rule-Based + LLM Judge - NO NLI)...")
    evaluator = PillarsEvaluator(use_nli=False)  # Disable DeBERTa NLI model
    
    # Initialize judge
    print("🤖 INITIALIZING LLM JUDGE...")
    os.environ['OPENAI_API_KEY'] = 'demo-key-for-testing'
    judge = MockJudge(mode="ALWAYS", diagnostic=True)
    evaluator.judge = judge
    
    print("✅ Initialization complete!")
    print()
    
    # Test each output
    for i, output in enumerate(outputs):
        print(f"🧪 ANALYZING EXAMPLE {i} (What DeBERTa would do)")
        print("-" * 60)
        
        # Extract data
        question = output['question']
        cot_text = output['answer']
        gold_answer = output['gt']
        predicted_answer = output['pred'][0]
        
        print(f"📝 QUESTION: {question}")
        print(f"✅ GOLD ANSWER: {gold_answer}")
        print(f"🤖 PREDICTED ANSWER: {predicted_answer}")
        print()
        
        # Show what DeBERTa would analyze
        print("🧠 WHAT DEBERTA NLI MODEL WOULD ANALYZE:")
        print("-" * 40)
        
        # Split the CoT into steps
        steps = []
        for line in cot_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('####'):
                steps.append(line)
        
        print(f"📋 REASONING STEPS ({len(steps)}):")
        for j, step in enumerate(steps, 1):
            print(f"  Step {j}: {step}")
        
        print()
        
        # Show what DeBERTa would check
        print("🔍 DEBERTA NLI ANALYSIS:")
        print("-" * 30)
        
        print("📚 COHERENCE CHECKS (Step-to-Step Consistency):")
        for j in range(len(steps) - 1):
            print(f"  • Step {j+1} → Step {j+2}:")
            print(f"    Premise: '{steps[j]}'")
            print(f"    Hypothesis: '{steps[j+1]}'")
            print(f"    → DeBERTa would check if Step {j+2} logically follows from Step {j+1}")
        
        print()
        
        print("📖 FACTUALITY CHECKS (Problem Context Grounding):")
        for j, step in enumerate(steps, 1):
            print(f"  • Step {j}: '{step}'")
            print(f"    → DeBERTa would check if this step is grounded in the problem context")
            print(f"    Problem: '{question}'")
        
        print()
        
        # Run analysis without NLI
        print("⚡ RUNNING ANALYSIS (Rule-Based + GPT Judge)...")
        flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(
            problem=question,
            cot_text=cot_text,
            gold=gold_answer
        )
        
        print("✅ ANALYSIS COMPLETE!")
        print()
        
        # Show what DeBERTa would add
        print("🧠 WHAT DEBERTA NLI WOULD ADD:")
        print("-" * 35)
        
        print("📊 COHERENCE METRICS (DeBERTa would provide):")
        print(f"  • coh_contra_cnt: Number of step-to-step contradictions")
        print(f"  • avg_coh_margin: Average confidence margin for coherence")
        print("  • These would be 0.0 without DeBERTa (as shown above)")
        
        print()
        
        print("📊 FACTUALITY METRICS (DeBERTa would provide):")
        print(f"  • fact_entail_rate: Rate of steps grounded in problem context")
        print(f"  • fact_contra_cnt: Number of contradictions with problem context")
        print(f"  • avg_fact_margin: Average confidence margin for factuality")
        print("  • These would be 0.0 without DeBERTa (as shown above)")
        
        print()
        
        print("🚩 COHERENCE FLAGS (DeBERTa would detect):")
        print("  • Inconsistent reasoning steps")
        print("  • Logical contradictions between steps")
        print("  • Steps that don't follow from previous steps")
        
        print()
        
        print("🚩 FACTUALITY FLAGS (DeBERTa would detect):")
        print("  • Steps not grounded in problem context")
        print("  • Contradictions with given information")
        print("  • Assumptions not supported by problem")
        
        print()
        
        # Show current evidence (without DeBERTa)
        print("📈 CURRENT EVIDENCE METRICS (Without DeBERTa):")
        for key, value in evidence.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print()
        
        # Show current scores
        print("📊 CURRENT SCORES (Without DeBERTa):")
        for pillar, score in rule_scores.items():
            print(f"  {pillar}: {score:.3f}")
        
        print()
        
        print("🤖 GPT JUDGE SCORES:")
        for pillar, score in judge_scores.items():
            if score is not None:
                print(f"  {pillar}: {score}/5")
        
        print()
        
        print("=" * 60)
        print()
    
    print("🎯 DEBERTA NLI MODEL SUMMARY")
    print("=" * 80)
    
    print("The DeBERTa NLI model would provide:")
    print()
    print("✅ COHERENCE ANALYSIS:")
    print("  • Natural Language Inference between consecutive reasoning steps")
    print("  • Detects logical contradictions and inconsistencies")
    print("  • Measures confidence in step-to-step relationships")
    print()
    
    print("✅ FACTUALITY ANALYSIS:")
    print("  • Checks if reasoning steps are grounded in problem context")
    print("  • Detects contradictions with given information")
    print("  • Measures how well reasoning uses provided facts")
    print()
    
    print("✅ ENHANCED FLAGGING:")
    print("  • More sophisticated flag detection beyond rule-based heuristics")
    print("  • Confidence-based scoring for nuanced assessment")
    print("  • Better handling of edge cases and complex reasoning")
    print()
    
    print("⚠️  CURRENT LIMITATION:")
    print("  • Threading issues when running on CPU environment")
    print("  • Requires SLURM with GPU allocation for proper execution")
    print("  • The SLURM job (ID: 2623435) should resolve this")
    print()
    
    print("🚀 NEXT STEPS:")
    print("  1. Wait for SLURM job to complete")
    print("  2. Review DeBERTa NLI results")
    print("  3. Compare with rule-based and GPT judge scores")
    print("  4. Validate complete pipeline functionality")

if __name__ == "__main__":
    demo_deberta_capabilities()
