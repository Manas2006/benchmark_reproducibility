# Benchmark Reproducibility

A Swiss‑Army evaluator for any LightEval/Lm‑eval-harness task — from GSM8K and MATH to ARC‑Challenge, BoolQ, SQuAD, or your own CSV.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/benchmark_reproducibility.git
   cd benchmark_reproducibility
   ```

2. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate benchmark-reproducibility
   ```

3. **(Optional) If you use pip:**
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

Run a benchmark with Lighteval (vLLM backend):
```bash
python main.py --model Qwen/Qwen2.5-Math-1.5B --task 'lighteval|gsm8k|0|0' --output_dir runs
```

- `--model`: HuggingFace hub ID or local path
- `--task`: Task in format `suite|task|few_shot|truncate_few_shots` (e.g., `lighteval|gsm8k|0|0`)
- `--output_dir`: Directory for results and logs
- `--framework`: Choose `lighteval` (default) or `lm-eval`

## Advanced Options

- `--shots`: Number of few-shot examples (default: 0)
- `--fewshot_file`: Path to JSONL file with exemplars
- `--temperature`, `--top_p`: Sampling parameters
- `--metrics`: Comma-separated list of metrics (e.g., `pass@1,pass@5`)
- `--system_prompt`: Custom system prompt for the model (see below)
- `--save_details`: Save per-sample predictions and references for inspection
- `--debug`: Enable debug output
- Many more options for logging, tracking, and backend-specific features (see `python main.py --help`)

## System Prompt Customization

A default system prompt is provided in `main.py`:
```python
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think step by step, show your reasoning, and give your final answer after '####'. "
    "For example: '...solution steps... #### 42'"
)
```
You can edit this string in `main.py` or override it via the CLI:
```bash
python main.py ... --system_prompt "Think step by step and answer after ####."
```

## Progress Reporting

- The tool shows a live progress spinner and, if possible, a percentage progress bar while running Lighteval.
- All debug and progress output is written to the `.out` SLURM file or your terminal.

## Inspecting Per-Sample Outputs

- Use `--save_details` to save detailed predictions and references.
- After the run, look for a `details_*.json` file in your results directory (e.g., `runs/results/<model>/details_*.json`).
- This file contains all model outputs and gold answers for inspection and error analysis.

## Example: SLURM Script

```bash
#!/bin/bash
#SBATCH -J lighteval-vllm
#SBATCH -o /work/10757/manasp123/runs/slurm-%j.out
#SBATCH -e /work/10757/manasp123/runs/slurm-%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -A CCR24036
source ~/.bashrc
conda activate benchmark-reproducibility
python -u benchmark_reproducibility/main.py --model Qwen/Qwen2.5-Math-1.5B --task 'lighteval|gsm8k|0|0' --output_dir /work/10757/manasp123/runs --debug --save_details
```

## Notes

- Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
- For GPU support, ensure your CUDA drivers are compatible with the PyTorch version.
- You can specify metrics to compute with `--metrics pass@1,pass@5`.
- See `examples/` for few-shot templates.
- For more options, run:
  ```bash
  python main.py --help
  ``` 