#!/usr/bin/env python3
"""
Demonstration of Comprehensive CoT Analysis Integration
Shows the complete pipeline: Rule-based + DeBERTa NLI + GPT Judge
"""

import sys
import os
import json
import time

# Add backend/app to path
backend_app_dir = os.path.join(os.path.dirname(__file__), 'backend', 'app')
sys.path.insert(0, backend_app_dir)

def demo_comprehensive_analysis():
    """Demonstrate the complete comprehensive CoT analysis pipeline"""
    
    print("🧠 COMPREHENSIVE CHAIN-OF-THOUGHT ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print("This demo shows the complete evaluation framework:")
    print("• Rule-based analysis (arithmetic, coverage, etc.)")
    print("• DeBERTa NLI analysis (coherence, factuality) - CPU compatible")
    print("• GPT-4o-mini judge (subjective scoring)")
    print("• Score fusion (combining all methods)")
    print("=" * 80)
    
    try:
        # Import the comprehensive analysis
        from cot_analyzer import CoTAnalyzer
        
        # Initialize analyzer with OpenAI key
        analyzer = CoTAnalyzer(openai_api_key='demo-key-for-testing')
        
        # Sample problems for demonstration
        sample_problems = [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "answer": """Let me break this down step by step.

First, I need to find out how many eggs Janet has left after using some for breakfast and baking.

Janet's ducks lay 16 eggs per day.
She eats 3 eggs for breakfast every morning.
She uses 4 eggs for baking muffins every day.

So the number of eggs she uses each day is:
3 (for breakfast) + 4 (for baking) = 7 eggs

Therefore, the number of eggs she has left to sell is:
16 (total eggs) - 7 (eggs used) = 9 eggs

She sells these 9 eggs at the farmers' market for $2 per egg.

So her daily earnings from the farmers' market are:
9 eggs × $2 per egg = $18

Therefore, Janet makes $18 every day at the farmers' market.""",
                "gold": "18"
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": """Let me solve this step by step.

A robe takes:
- 2 bolts of blue fiber
- Half that much white fiber

Half of 2 bolts = 1 bolt of white fiber

Total bolts needed = blue fiber + white fiber
Total bolts needed = 2 + 1 = 3 bolts

Therefore, the robe takes 3 bolts in total.""",
                "gold": "3"
            },
            {
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "answer": """Let me calculate Josh's profit from flipping the house.

Initial house value: $80,000
Repair costs: $50,000
Total investment: $80,000 + $50,000 = $130,000

The repairs increased the value by 150%, which means the new value is:
$80,000 + (150% × $80,000) = $80,000 + $120,000 = $200,000

Profit = Final value - Total investment
Profit = $200,000 - $130,000 = $70,000

Therefore, Josh made a profit of $70,000.""",
                "gold": "70000"
            }
        ]
        
        print(f"\n📊 ANALYZING {len(sample_problems)} SAMPLE PROBLEMS")
        print("-" * 60)
        
        results = []
        
        for i, problem in enumerate(sample_problems):
            print(f"\n🔍 PROBLEM {i+1}:")
            print(f"Question: {problem['question'][:80]}...")
            print(f"Expected Answer: {problem['gold']}")
            
            # Run comprehensive analysis
            start_time = time.time()
            result = analyzer.analyze_answer_comprehensive(
                question=problem['question'],
                answer=problem['answer'],
                ground_truth=problem['gold']
            )
            analysis_time = time.time() - start_time
            
            results.append(result)
            
            # Display results
            if result.get('analysis_method') == 'comprehensive':
                print(f"✅ Comprehensive Analysis ({analysis_time:.2f}s)")
                
                # Overall score
                overall = result.get('fused_scores', {}).get('overall', 0.0)
                print(f"🎯 Overall Score: {overall:.3f}")
                
                # Pillar scores
                print("📈 Pillar Scores:")
                for pillar in ['faithfulness', 'utility', 'coherence', 'factuality']:
                    score = result.get('fused_scores', {}).get(pillar, 0.0)
                    print(f"   {pillar.capitalize()}: {score:.3f}")
                
                # Evidence metrics
                evidence = result.get('evidence', {})
                print("🔬 Evidence Metrics:")
                print(f"   Final Correct: {evidence.get('final_correct', False)}")
                print(f"   Arith Errors: {len(evidence.get('arith_bad_examples', []))}")
                print(f"   Coherence Contradictions: {evidence.get('coh_contra_cnt', 0)}")
                print(f"   Factuality Contradictions: {evidence.get('fact_contra_cnt', 0)}")
                print(f"   Fact Entail Rate: {evidence.get('fact_entail_rate', 0.0):.3f}")
                print(f"   Intermediate OK Rate: {evidence.get('intermediate_ok_rate', 0.0):.3f}")
                
                # Flags
                flags = result.get('flags', [])
                if flags:
                    print(f"🚩 Flags Detected ({len(flags)}):")
                    for flag in flags[:3]:  # Show first 3 flags
                        print(f"   {flag['pillar'].capitalize()}: {flag['issue']}")
                    if len(flags) > 3:
                        print(f"   ... and {len(flags) - 3} more flags")
                else:
                    print("✅ No flags detected - reasoning appears sound!")
                
            else:
                print(f"⚠️ Legacy Analysis ({analysis_time:.2f}s)")
                print(f"🎯 CQS Score: {result.get('metrics', {}).get('cqs_score', 0.0):.3f}")
            
            print("-" * 40)
        
        # Summary statistics
        comprehensive_results = [r for r in results if r.get('analysis_method') == 'comprehensive']
        if comprehensive_results:
            print(f"\n📊 SUMMARY STATISTICS")
            print("=" * 40)
            print(f"Total Problems Analyzed: {len(results)}")
            print(f"Comprehensive Analysis: {len(comprehensive_results)}")
            print(f"Legacy Analysis: {len(results) - len(comprehensive_results)}")
            
            if comprehensive_results:
                avg_overall = sum(r.get('fused_scores', {}).get('overall', 0.0) for r in comprehensive_results) / len(comprehensive_results)
                total_flags = sum(len(r.get('flags', [])) for r in comprehensive_results)
                avg_arith_errors = sum(len(r.get('evidence', {}).get('arith_bad_examples', [])) for r in comprehensive_results) / len(comprehensive_results)
                
                print(f"Average Overall Score: {avg_overall:.3f}")
                print(f"Total Flags Detected: {total_flags}")
                print(f"Average Arithmetic Errors: {avg_arith_errors:.1f}")
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("The comprehensive CoT analysis framework is working correctly.")
        print("It combines rule-based analysis, DeBERTa NLI, and GPT judge scoring.")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_comprehensive_analysis()
    sys.exit(0 if success else 1)
