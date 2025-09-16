#!/usr/bin/env python3
"""
Test the complete CoT evaluation pipeline including DeBERTa NLI model - LOCAL VERSION
This version runs locally to test DeBERTa coherence and factuality flags
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'app'))

from cot_eval_v2.evaluator import PillarsEvaluator
from cot_eval_v2.judge import MockJudge

def test_deberta_local():
    """Test complete pipeline with DeBERTa NLI model enabled - local version"""
    
    print("🧠 TESTING COMPLETE PIPELINE WITH DEBERTA NLI MODEL (LOCAL)")
    print("=" * 80)
    
    # Load real outputs
    result_file = "output/Qwen2.5-Math-7B/gsm8k/test_auto-cot_-1_seed0_t0.0_s0_e-1_f71140d9-3c8b-4ec9-9956-a586e491b16c.jsonl"
    
    outputs = []
    with open(result_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Test with 3 examples first
                break
            data = json.loads(line.strip())
            outputs.append(data)
    
    print(f"📊 Loaded {len(outputs)} real model outputs for testing")
    print()
    
    try:
        # Initialize evaluator WITH NLI (DeBERTa model)
        print("🔧 INITIALIZING EVALUATOR (Rule-Based + DeBERTa NLI + LLM Judge)...")
        evaluator = PillarsEvaluator(use_nli=True)  # Enable DeBERTa NLI model
        
        # Initialize judge
        print("🤖 INITIALIZING LLM JUDGE...")
        os.environ['OPENAI_API_KEY'] = 'demo-key-for-testing'
        judge = MockJudge(mode="ALWAYS", diagnostic=True)  # Force judge to always run
        evaluator.judge = judge
        
        print("✅ Initialization complete!")
        print()
        
        # Test each output
        results = []
        
        for i, output in enumerate(outputs):
            print(f"🧪 EVALUATING EXAMPLE {i} WITH DEBERTA NLI")
            print("-" * 60)
            
            # Extract data
            question = output['question']
            cot_text = output['answer']
            gold_answer = output['gt']
            predicted_answer = output['pred'][0]
            
            print(f"📝 QUESTION: {question}")
            print(f"✅ GOLD ANSWER: {gold_answer}")
            print(f"🤖 PREDICTED ANSWER: {predicted_answer}")
            print(f"🎯 CORRECT: {predicted_answer == gold_answer}")
            print()
            
            # Run comprehensive analysis WITH DeBERTa NLI
            print("⚡ RUNNING COMPREHENSIVE ANALYSIS (Rule + DeBERTa + GPT)...")
            flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(
                problem=question,
                cot_text=cot_text,
                gold=gold_answer
            )
            
            print("✅ ANALYSIS COMPLETE!")
            print()
            
            # Display results
            print("📊 COMPLETE ANALYSIS RESULTS")
            print("-" * 40)
            
            # 1. All flags (Rule-based + DeBERTa)
            print("🚩 ALL FLAGS (Rule-Based + DeBERTa NLI):")
            if flags.has_flags():
                for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                    pillar_flags = flags.get_flags_by_pillar(pillar)
                    if pillar_flags:
                        print(f"  {pillar.upper()}:")
                        for flag in pillar_flags:
                            print(f"    - {flag.step}: {flag.issue}")
                            if flag.details:
                                print(f"      Details: {flag.details}")
            else:
                print("  ✅ No flags detected - reasoning appears sound!")
            
            print()
            
            # 2. Evidence metrics (including DeBERTa NLI results)
            print("📈 EVIDENCE METRICS (Including DeBERTa NLI):")
            for key, value in evidence.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            
            print()
            
            # 3. Rule-based scores
            print("📊 RULE-BASED SCORES:")
            for pillar, score in rule_scores.items():
                print(f"  {pillar}: {score:.3f}")
            
            print()
            
            # 4. GPT Judge scores
            print("🤖 GPT JUDGE SCORES:")
            for pillar, score in judge_scores.items():
                if score is not None:
                    print(f"  {pillar}: {score}/5")
            
            print()
            
            # 5. Fused scores (Rule + DeBERTa + GPT)
            print("🔗 FUSED SCORES (Rule + DeBERTa + GPT):")
            for pillar, score in fused_scores.items():
                if pillar != 'overall':
                    print(f"  {pillar}: {score:.3f}")
            
            print()
            
            # 6. Overall assessment
            overall_score = fused_scores.get('overall', 0.0)
            print("🏆 OVERALL ASSESSMENT:")
            if overall_score >= 0.8:
                print(f"  Score: {overall_score:.3f} - EXCELLENT reasoning quality! 🌟")
            elif overall_score >= 0.6:
                print(f"  Score: {overall_score:.3f} - GOOD reasoning quality! 👍")
            elif overall_score >= 0.4:
                print(f"  Score: {overall_score:.3f} - FAIR reasoning quality ⚠️")
            else:
                print(f"  Score: {overall_score:.3f} - POOR reasoning quality ❌")
            
            # Store results for summary
            results.append({
                'idx': i,
                'question': question,
                'gold': gold_answer,
                'predicted': predicted_answer,
                'correct': predicted_answer == gold_answer,
                'final_correct': evidence.get('final_correct', False),
                'flags_count': len(flags) if hasattr(flags, '__len__') else 0,
                'coherence_flags': len(flags.get_flags_by_pillar('coherence')) if flags.has_flags() else 0,
                'factuality_flags': len(flags.get_flags_by_pillar('factuality')) if flags.has_flags() else 0,
                'utility_score': rule_scores.get('utility_rule', 0.0),
                'coherence_score': rule_scores.get('coherence_rule', 0.0),
                'factuality_score': rule_scores.get('factuality_rule', 0.0),
                'overall_score': overall_score,
                'arith_errors': len(evidence.get('arith_bad_examples', [])),
                'coh_contra_cnt': evidence.get('coh_contra_cnt', 0),
                'fact_contra_cnt': evidence.get('fact_contra_cnt', 0),
                'fact_entail_rate': evidence.get('fact_entail_rate', 0.0),
                'judge_called': any(judge_scores.values())
            })
            
            print()
            print("=" * 60)
            print()
        
        # Summary
        print("📊 COMPLETE PIPELINE EVALUATION SUMMARY")
        print("=" * 80)
        
        total_examples = len(results)
        correct_answers = sum(1 for r in results if r['correct'])
        examples_with_coherence_flags = sum(1 for r in results if r['coherence_flags'] > 0)
        examples_with_factuality_flags = sum(1 for r in results if r['factuality_flags'] > 0)
        examples_with_contradictions = sum(1 for r in results if r['coh_contra_cnt'] > 0 or r['fact_contra_cnt'] > 0)
        
        print(f"📈 OVERALL STATISTICS:")
        print(f"  Total examples: {total_examples}")
        print(f"  Correct answers: {correct_answers}/{total_examples} ({correct_answers/total_examples*100:.1f}%)")
        print(f"  Examples with coherence flags: {examples_with_coherence_flags}/{total_examples} ({examples_with_coherence_flags/total_examples*100:.1f}%)")
        print(f"  Examples with factuality flags: {examples_with_factuality_flags}/{total_examples} ({examples_with_factuality_flags/total_examples*100:.1f}%)")
        print(f"  Examples with contradictions: {examples_with_contradictions}/{total_examples} ({examples_with_contradictions/total_examples*100:.1f}%)")
        
        print()
        print("📋 DETAILED RESULTS WITH DEBERTA NLI:")
        for r in results:
            status = "✅" if r['correct'] else "❌"
            coherence_status = f"🧠{r['coherence_flags']}" if r['coherence_flags'] > 0 else "🧠0"
            factuality_status = f"📚{r['factuality_flags']}" if r['factuality_flags'] > 0 else "📚0"
            contradictions = f"⚠️{r['coh_contra_cnt']+r['fact_contra_cnt']}" if (r['coh_contra_cnt'] + r['fact_contra_cnt']) > 0 else "⚠️0"
            
            print(f"  {status} Example {r['idx']}: {r['gold']} vs {r['predicted']} | "
                  f"Overall: {r['overall_score']:.3f} | "
                  f"Coherence: {r['coherence_score']:.3f} | Factuality: {r['factuality_score']:.3f} | "
                  f"{coherence_status} {factuality_status} {contradictions}")
        
        print()
        print("🎉 COMPLETE PIPELINE TEST COMPLETE!")
        
        # Save results to file for analysis
        with open('deberta_local_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("💾 Results saved to deberta_local_results.json")
        
        return results
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("This might be due to threading issues with DeBERTa on CPU.")
        print("The SLURM job should work better with proper GPU allocation.")
        return None

if __name__ == "__main__":
    results = test_deberta_local()
