#!/usr/bin/env python3
"""benchmark_reproductability – Evaluate ANY LightEval/Lm-eval task, 0-shot or N-shot."""
import argparse, json, os, time, csv, subprocess, random, torch, yaml
from pathlib import Path
from runner import run_lighteval, run_lm_eval
from utils import ts, row, debug_log

##############################################################################
# Registry: built-in tasks that ship with LightEval  / lm-eval-harness
# If you drop a custom YAML into tasks/, just pass its path via --task
##############################################################################
DEFAULT_TASKS = {
    # math-ish helpers (kept for reference; not required – LightEval has these)
    "gsm8k":      "tasks/gsm8k.yaml",
    "math":       "tasks/math.yaml",
    "math500":    "tasks/math500.yaml",
}

# =====================
# Default system prompt for math tasks (editable by user)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think step by step, show your reasoning, and give your final answer after '####'. "
    "For example: '...solution steps... #### 42'"
)
# =====================

##############################################################################
# CLI
##############################################################################
def cli():
    """
    Parses command-line arguments for the benchmarking tool.
    Exposes all Lighteval/vLLM features as CLI options.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF hub ID or local path")
    p.add_argument("--task", required=True,
                   help="LightEval task name OR YAML path")
    p.add_argument("--shots", type=int, default=0,
                   help="few-shot count (0 = zero-shot)")
    p.add_argument("--fewshot_file", default=None,
                   help="optional JSONL with exemplars")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--framework", choices=["lighteval", "lm-eval"],
                   default="lighteval")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--metrics", default="pass@1", help="Comma-separated list of metrics to compute (e.g., pass@1,pass@5)")
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    # Advanced Lighteval/vLLM options
    p.add_argument("--system_prompt", default=None, help="System prompt for evaluation (Lighteval vllm)")
    p.add_argument("--cot_prompt", default=None, help="Chain of thought prompt for evaluation (Lighteval vllm)")
    p.add_argument("--dataset_loading_processes", type=int, default=1, help="Number of processes for dataset loading (Lighteval vllm)")
    p.add_argument("--custom_tasks", default=None, help="Path to custom tasks directory (Lighteval vllm)")
    p.add_argument("--num_fewshot_seeds", type=int, default=1, help="Number of seeds for few-shot eval (Lighteval vllm)")
    p.add_argument("--load_responses_from_details_date_id", default=None, help="Load responses from details dir (Lighteval vllm)")
    p.add_argument("--results_path_template", default=None, help="Template path for saving results (Lighteval vllm)")
    p.add_argument("--push_to_hub", action="store_true", help="Push results to HuggingFace hub (Lighteval vllm)")
    p.add_argument("--no_push_to_hub", action="store_true", help="Do not push results to HuggingFace hub (Lighteval vllm)")
    p.add_argument("--push_to_tensorboard", action="store_true", help="Push results to TensorBoard (Lighteval vllm)")
    p.add_argument("--no_push_to_tensorboard", action="store_true", help="Do not push results to TensorBoard (Lighteval vllm)")
    p.add_argument("--public_run", action="store_true", help="Push results/details to public repo (Lighteval vllm)")
    p.add_argument("--no_public_run", action="store_true", help="Do not push results/details to public repo (Lighteval vllm)")
    p.add_argument("--results_org", default=None, help="Organization to push results to (Lighteval vllm)")
    p.add_argument("--save_details", action="store_true", help="Save detailed sample-per-sample results (Lighteval vllm)")
    p.add_argument("--no_save_details", action="store_true", help="Do not save detailed sample-per-sample results (Lighteval vllm)")
    p.add_argument("--wandb", action="store_true", help="Push results to wandb (Lighteval vllm)")
    p.add_argument("--no_wandb", action="store_true", help="Do not push results to wandb (Lighteval vllm)")
    p.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (Lighteval vllm)")
    p.add_argument("--job_id", type=int, default=0, help="Optional job id for reference (Lighteval vllm)")
    return p.parse_args()

def main():
    """
    Main entry point: parses arguments, runs the selected framework, and saves results.
    Handles output to CSV and JSON, and supports debug logging.
    Automatically converts --task and --shots to the new Lighteval format if needed.
    """
    args = cli()
    # --- AUTO-CONVERT TASK FORMAT FOR LIGHTEVAL ---
    # Lighteval >=0.10.0 expects tasks as suite|task|few_shot|truncate_few_shots
    # If user provides a simple task name (e.g., 'gsm8k'), convert it automatically
    if args.framework == "lighteval" and "|" not in args.task:
        # Default to 'main' split, shots from CLI, no truncation
        args.task = f"{args.task}|main|{args.shots}|0"
        # Comment: This auto-converts legacy task names to the new required format for Lighteval >=0.10.0
    # --- SET SYSTEM PROMPT IF NOT PROVIDED ---
    if args.system_prompt is None:
        args.system_prompt = DEFAULT_SYSTEM_PROMPT
    debug = getattr(args, "debug", False)
    debug_log(f"Parsed arguments: {args}", debug)
    # Set random seeds for reproducibility
    random.seed(args.seed); torch.manual_seed(args.seed)
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Parse metrics list
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    debug_log(f"Using framework: {args.framework}", debug)
    # Run the selected evaluation framework
    if args.framework == "lighteval":
        scores, n_tok = run_lighteval(args, metrics, debug)
    else:
        scores, n_tok = run_lm_eval(args, metrics, debug)
    debug_log(f"Scores: {scores}", debug)
    # Build output row and save to CSV
    out = row(args, scores, n_tok)
    debug_log(f"Output row: {out}", debug)
    csv_path = Path(args.output_dir, "master.csv")
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, out.keys())
        if write_header:
            debug_log("Writing CSV header", debug)
            w.writeheader()
        w.writerow(out)
        debug_log(f"Wrote row to {csv_path}", debug)
    # Save results as JSON
    json_path = Path(args.output_dir, f"{args.task.replace('/','_')}_{ts()}.json")
    json_path.write_text(json.dumps(out, indent=2))
    debug_log(f"Wrote JSON to {json_path}", debug)
    print(json.dumps(out, indent=2))
    debug_log("Finished successfully.", debug)

if __name__ == "__main__":
    main() 