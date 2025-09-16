#!/bin/bash
#SBATCH --job-name=deberta-simple
#SBATCH --output=/home1/10757/manasp123/qwen-eval-ui/deberta-simple-%j.out
#SBATCH --error=/home1/10757/manasp123/qwen-eval-ui/deberta-simple-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=OTH24028

# Load conda environment
source ~/.bashrc
conda activate qwen-eval

# Change to project directory
cd /home1/10757/manasp123/qwen-eval-ui

# Run a simple DeBERTa test
echo "🧠 Starting simple DeBERTa NLI test..."
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo ""

# Test with just one example to avoid long runtime
python -c "
import sys
import os
sys.path.insert(0, 'backend/app')

from cot_eval_v2.evaluator import PillarsEvaluator
from cot_eval_v2.judge import MockJudge

print('Testing DeBERTa NLI with one example...')

# Simple test case
problem = 'Janet has 5 apples. She gives 2 to her friend. How many does she have left?'
cot_text = 'Janet starts with 5 apples. She gives 2 to her friend. So she has 5 - 2 = 3 apples left.'
gold = '3'

# Initialize evaluator WITH NLI
evaluator = PillarsEvaluator(use_nli=True)
os.environ['OPENAI_API_KEY'] = 'demo-key-for-testing'
judge = MockJudge(mode='ALWAYS', diagnostic=True)
evaluator.judge = judge

print('Running analysis...')
flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(
    problem=problem,
    cot_text=cot_text,
    gold=gold
)

print('Results:')
print(f'Coherence contradictions: {evidence.get(\"coh_contra_cnt\", 0)}')
print(f'Factuality contradictions: {evidence.get(\"fact_contra_cnt\", 0)}')
print(f'Factuality entail rate: {evidence.get(\"fact_entail_rate\", 0.0):.3f}')
print('✅ DeBERTa NLI test completed successfully!')
"

echo ""
echo "✅ Simple DeBERTa NLI test completed at $(date)"
