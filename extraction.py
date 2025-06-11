# Functions for extracting predictions and references from evaluation results
import re

def extract_answers(result, task_name, debug=False):
    """
    Extracts predictions and references from the result object for a given task.
    Extracts the boxed answer (i.e., content inside \\boxed{...}) from predictions for all datasets.
    """
    from utils import debug_log
    debug_log(f"Extracting answers for task: {task_name}", debug)
    preds = result.get("predictions", [])
    refs = result.get("references", [])
    # Extract boxed answer from each prediction
    def extract_boxed(pred):
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        match = re.search(r"\\\\boxed\{(.+?)\}", str(pred))
        return match.group(1).strip() if match else str(pred).strip()
    preds_extracted = [extract_boxed(p) for p in preds]
    debug_log(f"Extracted predictions: {preds_extracted[:3]}", debug)
    debug_log(f"References: {refs[:3]}", debug)
    return preds_extracted, refs


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