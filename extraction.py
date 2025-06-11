# Functions for extracting predictions and references from evaluation results

def extract_answers(result, task_name, debug=False):
    """
    Extracts predictions and references from the result object for a given task.
    If the task is GSM8K, uses a custom extractor; otherwise, uses generic keys.
    """
    from utils import debug_log
    debug_log(f"Extracting answers for task: {task_name}", debug)
    if task_name.lower() == "gsm8k":
        # Use GSM8K-specific extraction logic
        preds, refs = extract_gsm8k(result, debug)
    else:
        # Generic extraction for other tasks
        preds, refs = result.get("predictions", []), result.get("references", [])
    debug_log(f"Extracted {len(preds)} predictions, {len(refs)} references", debug)
    if debug:
        debug_log(f"Sample predictions: {preds[:3]}", debug)
        debug_log(f"Sample references: {refs[:3]}", debug)
    return preds, refs


def extract_gsm8k(result, debug=False):
    """
    Extracts predictions and references for the GSM8K task.
    This function can be extended for GSM8K-specific postprocessing if needed.
    """
    from utils import debug_log
    debug_log("Extracting GSM8K answers", debug)
    preds = result.get("predictions", [])
    refs = result.get("references", [])
    debug_log(f"GSM8K predictions: {preds[:3]}", debug)
    debug_log(f"GSM8K references: {refs[:3]}", debug)
    return preds, refs 