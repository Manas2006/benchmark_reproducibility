#!/bin/bash
#SBATCH --job-name=deberta-test
#SBATCH --output=/home1/10757/manasp123/qwen-eval-ui/deberta-test-%j.out
#SBATCH --error=/home1/10757/manasp123/qwen-eval-ui/deberta-test-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=OTH24028

# Load conda environment
source ~/.bashrc
conda activate qwen-eval

# Change to project directory
cd /home1/10757/manasp123/qwen-eval-ui

# Run the DeBERTa complete test
echo "🧠 Starting DeBERTa NLI complete pipeline test..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo ""

python test_deberta_complete.py

echo ""
echo "✅ DeBERTa NLI test completed at $(date)"
echo "Results saved to deberta_complete_results.json"
