#!/bin/bash
#SBATCH -J sober-reasoning-test
#SBATCH -A CCR24036
#SBATCH -p gpu-a100-small           # 1-node A100 queue
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -o /work/10757/manasp123/sober-reasoning/logs/sober-reasoning-test.%j.out
#SBATCH -e /work/10757/manasp123/sober-reasoning/logs/sober-reasoning-test.%j.err

module purge
module load cuda/12.2               # use a version that `module spider cuda` lists
#module load conda                   # TACC-supplied Miniconda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sober-reasoning      # must already exist; create if needed

# OPTIONAL: restrict job to 1 GPU on the node
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$PYTHONPATH:$HOME/sober-reasoning

mkdir -p $WORK/sober-reasoning/logs
mkdir -p $WORK/sober-reasoning/outputs

cd $HOME/sober-reasoning

# 🔧 NOTE: Added `-u` to force unbuffered output
srun python -u main.py \
  --model Qwen/Qwen2.5-Math-1.5B \
  --task "custom|gsm8k|0|0" \
  --temperature 0.8 \
  --top_p 0.9 \
  --seed 0 \
  --output_dir $WORK/sober-reasoning/outputs \
  --max_new_tokens 4096 \
  --max_model_length 4096 \
  --custom_tasks_directory lighteval_tasks \
  --overwrite
