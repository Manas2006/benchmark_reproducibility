#!/bin/bash
# SLURM job script for running Lighteval with vLLM backend on TACC Lonestar6

#SBATCH -J lighteval-vllm                # Job name
#SBATCH -o /work/10757/manasp123/runs/slurm-%j.out  # Standard output file
#SBATCH -e /work/10757/manasp123/runs/slurm-%j.err  # Standard error file
#SBATCH -p gpu-a100-dev                  # Partition (queue) name
#SBATCH -N 1                             # Number of nodes
#SBATCH -n 1                             # Number of tasks
#SBATCH -c 8                             # Number of CPU cores per task
#SBATCH -t 02:00:00                      # Time limit (hh:mm:ss)
#SBATCH -A CCR24036                      # Project account

# module load cuda/12.1 (removed)        # CUDA module not available on this system
source ~/.bashrc                         # Source user bash config
conda activate benchmark-reproducibility # Activate the Conda environment

# Run the main benchmarking script with debug output
python -u benchmark_reproducibility/main.py --model Qwen/Qwen2.5-Math-1.5B --task 'lighteval|gsm8k|0|0' --output_dir /work/10757/manasp123/runs --debug --save_details 