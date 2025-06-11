# Functions for computing evaluation metrics (e.g., pass@k)

def compute_metrics(predictions, references, metrics_list, debug=False):
    """
    Computes the requested metrics for the given predictions and references.
    Currently supports pass@1 and pass@5. Extend as needed.
    """
    from utils import debug_log
    debug_log(f"Computing metrics: {metrics_list}", debug)
    results = {}
    if "pass@1" in metrics_list:
        # Compute pass@1 metric
        results["pass@1"] = pass_at_k(predictions, references, k=1, debug=debug)
        debug_log(f"pass@1: {results['pass@1']}", debug)
    if "pass@5" in metrics_list:
        # Compute pass@5 metric
        results["pass@5"] = pass_at_k(predictions, references, k=5, debug=debug)
        debug_log(f"pass@5: {results['pass@5']}", debug)
    # Add more metrics as needed
    debug_log(f"All computed metrics: {results}", debug)
    return results


def pass_at_k(predictions, references, k=1, debug=False):
    """
    Computes pass@k: fraction of cases where the correct answer is in the top-k predictions.
    """
    from utils import debug_log
    debug_log(f"Calculating pass@{k} for {len(predictions)} predictions", debug)
    correct = 0
    for i, (pred_list, ref) in enumerate(zip(predictions, references)):
        # Check if any of the top-k predictions match the reference (case-insensitive, stripped)
        if any(str(p).strip().lower() == str(ref).strip().lower() for p in pred_list[:k]):
            correct += 1
        if debug and i < 3:
            debug_log(f"Example {i}: pred_list={pred_list[:k]}, ref={ref}, correct={correct}", debug)
    score = correct / len(references) if references else 0.0
    debug_log(f"pass@{k} score: {score}", debug)
    return score 