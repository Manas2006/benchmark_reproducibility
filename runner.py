import time
import subprocess
import json
from extraction import extract_answers
from metrics import compute_metrics
from utils import debug_log
import sys
import threading

def run_lighteval(args, metrics, debug):
    """
    Runs the Lighteval CLI with the vllm backend using the provided arguments.
    Passes all relevant CLI options, captures output, and computes metrics.
    Shows a progress spinner or percentage if possible.
    """
    debug_log(f"Running LightEval CLI with task: {args.task}, shots: {args.shots}", debug)
    cmd = [
        "lighteval", "vllm",
        f"model_name={args.model}",
        args.task,
        "--output-dir", args.output_dir,
    ]
    if args.fewshot_file:
        cmd += ["--custom_fewshot_path", args.fewshot_file]
    if args.use_chat_template:
        cmd += ["--use-chat-template"]
    if getattr(args, "system_prompt", None):
        cmd += ["--system-prompt", args.system_prompt]
    if getattr(args, "cot_prompt", None):
        cmd += ["--cot-prompt", args.cot_prompt]
    if getattr(args, "dataset_loading_processes", 1) != 1:
        cmd += ["--dataset-loading-processes", str(args.dataset_loading_processes)]
    if getattr(args, "custom_tasks", None):
        cmd += ["--custom-tasks", args.custom_tasks]
    if getattr(args, "num_fewshot_seeds", 1) != 1:
        cmd += ["--num-fewshot-seeds", str(args.num_fewshot_seeds)]
    if getattr(args, "load_responses_from_details_date_id", None):
        cmd += ["--load-responses-from-details-date-id", args.load_responses_from_details_date_id]
    if getattr(args, "results_path_template", None):
        cmd += ["--results-path-template", args.results_path_template]
    if getattr(args, "push_to_hub", False):
        cmd += ["--push-to-hub"]
    if getattr(args, "no_push_to_hub", False):
        cmd += ["--no-push-to-hub"]
    if getattr(args, "push_to_tensorboard", False):
        cmd += ["--push-to-tensorboard"]
    if getattr(args, "no_push_to_tensorboard", False):
        cmd += ["--no-push-to-tensorboard"]
    if getattr(args, "public_run", False):
        cmd += ["--public-run"]
    if getattr(args, "no_public_run", False):
        cmd += ["--no-public-run"]
    if getattr(args, "results_org", None):
        cmd += ["--results-org", args.results_org]
    if getattr(args, "save_details", False):
        cmd += ["--save-details"]
    if getattr(args, "no_save_details", False):
        cmd += ["--no-save-details"]
    if getattr(args, "wandb", False):
        cmd += ["--wandb"]
    if getattr(args, "no_wandb", False):
        cmd += ["--no-wandb"]
    if getattr(args, "max_samples", None) is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if getattr(args, "job_id", 0) != 0:
        cmd += ["--job-id", str(args.job_id)]
    debug_log(f"Running command: {' '.join(cmd)}", debug)
    t0 = time.time()

    spinner = ['|', '/', '-', '\\']
    spinner_idx = 0
    stop_spinner = False
    def spin():
        while not stop_spinner:
            sys.stdout.write(f"\r[INFO] Lighteval running... {spinner[spinner_idx % 4]}")
            sys.stdout.flush()
            time.sleep(0.2)
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    progress_found = False
    try:
        for line in proc.stdout:
            if debug:
                print(line, end='')
            if 'Processed' in line and '/' in line:
                progress_found = True
                import re
                match = re.search(r'Processed\s+(\d+)\s*/\s*(\d+)', line)
                if match:
                    current, total = int(match.group(1)), int(match.group(2))
                    percent = 100 * current / total if total else 0
                    sys.stdout.write(f"\r[INFO] Progress: {current}/{total} ({percent:.1f}%)\n")
                    sys.stdout.flush()
    finally:
        proc.wait()
        stop_spinner = True
        spinner_thread.join()
        sys.stdout.write("\n")
        sys.stdout.flush()
    wall = time.time() - t0
    if proc.returncode:
        debug_log(f"lighteval CLI error: {proc.stdout.read() if proc.stdout else ''}", debug)
        raise RuntimeError("Lighteval CLI failed. See output above.")
    import glob
    import os
    result_files = glob.glob(os.path.join(args.output_dir, f"*{args.task}*.json"))
    if not result_files:
        raise RuntimeError(f"No result JSON file found in {args.output_dir} for task {args.task}")
    with open(result_files[0], "r") as f:
        res = json.load(f)
    task_res = res["results"][args.task if args.task in res["results"] else args.task]
    predictions, references = extract_answers(task_res, args.task, debug)
    debug_log(f"Extracted predictions: {predictions[:3]}... (total {len(predictions)})", debug)
    debug_log(f"Extracted references: {references[:3]}... (total {len(references)})", debug)
    scores = compute_metrics(predictions, references, metrics, debug)
    debug_log(f"Computed scores: {scores}", debug)
    scores["walltime_s"] = wall
    return scores, res.get("total_tokens", 0)

def run_lm_eval(args, metrics, debug):
    """
    Runs the lm-eval-harness CLI with the provided arguments.
    Captures output and computes metrics.
    """
    debug_log(f"Running lm-eval-harness with task: {args.task}, shots: {args.shots}", debug)
    cmd = [
        "lm_eval",
        "--model", "hf-causal",
        "--model_args", f"pretrained={args.model},dtype=bfloat16",
        "--tasks", args.task,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--seed", str(args.seed),
        "--fewshot", str(args.shots),
    ]
    if args.fewshot_file:
        cmd += ["--custom_fewshot_path", args.fewshot_file]
    debug_log(f"Running command: {' '.join(cmd)}", debug)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode:
        debug_log(f"lm-eval-harness error: {proc.stderr}", debug)
        raise RuntimeError(proc.stderr)
    out = json.loads(proc.stdout)
    debug_log(f"lm-eval-harness raw output: {out}", debug)
    tr = out["results"][args.task]
    predictions, references = extract_answers(tr, args.task, debug)
    debug_log(f"Extracted predictions: {predictions[:3]}... (total {len(predictions)})", debug)
    debug_log(f"Extracted references: {references[:3]}... (total {len(references)})", debug)
    scores = compute_metrics(predictions, references, metrics, debug)
    debug_log(f"Computed scores: {scores}", debug)
    scores["walltime_s"] = wall
    return scores, out["total_tokens"] 