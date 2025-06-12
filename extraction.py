# Functions for extracting predictions and references from evaluation results
import re
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.utils.language import Language

# Define the robust metric for math extraction
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

def try_extract(configs, text):
    for config in configs:
        result = config.extract(str(text))
        if result is not None and str(result).strip() != "":
            return result
    return ""

def extract_answers(result, task_name, debug=False):
    """
    Uses lighteval's robust extraction logic to extract answers from predictions and references.
    Tries all extraction configs in order and returns the first non-empty result.
    """
    from utils import debug_log
    debug_log(f"Extracting answers for task: {task_name}", debug)
    preds = result.get("predictions", [])
    refs = result.get("references", [])
    pred_extracted = [try_extract(latex_gold_metric.pred_extraction_target, p) for p in preds]
    ref_extracted = [try_extract(latex_gold_metric.gold_extraction_target, r) for r in refs]
    debug_log(f"Extracted predictions: {pred_extracted[:3]}", debug)
    debug_log(f"Extracted references: {ref_extracted[:3]}", debug)
    return pred_extracted, ref_extracted


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