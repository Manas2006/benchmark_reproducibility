#!/bin/bash
#SBATCH -J deberta-test              # Job name
#SBATCH -o deberta_test_%j.out       # Name of stdout output file (uses %j)
#SBATCH -e deberta_test_%j.err       # Name of stderr error file (uses %j)
#SBATCH -p gpu-a100-dev              # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks (single process for all GPUs)
#SBATCH -t 1:00:00                   # Run time (hh:mm:ss)
#SBATCH --mail-type=all              # Send email at begin and end of job
#SBATCH -A CCR24036                  # Project/Allocation name

# Fix MKL threading conflict
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "=============================="

# Navigate to the correct directory
cd /home1/10757/manasp123/qwen-eval-ui/backend/app

# Run the DeBERTa smoke test
echo "Running DeBERTa MNLI smoke test with SLURM GPU allocation..."
python -m cot_eval_v2.smoke_deberta

echo "=== Job completed ==="
