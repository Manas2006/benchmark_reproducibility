import time

# Returns a timestamp string for unique run IDs and filenames
def ts():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

# Builds a dictionary row for output, combining args, scores, and token stats
def row(args, scores, n_tok):
    return {
        "run_id": ts(),  # Unique run identifier
        "model": args.model,  # Model name or path
        "task": args.task,    # Task name
        "shots": args.shots,  # Number of few-shot examples
        "temp": args.temperature,  # Sampling temperature
        "top_p": args.top_p,      # Top-p sampling
        "seed": args.seed,        # Random seed
        **scores,                 # All computed metrics
        "tokens_per_s": n_tok / scores["walltime_s"],  # Throughput
    }

# Prints debug messages if debug mode is enabled
def debug_log(msg, debug):
    if debug:
        print(f"[DEBUG] {msg}") 