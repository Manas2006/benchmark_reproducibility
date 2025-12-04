import argparse
import os
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

# HumanEval official evaluation support
try:
    from humaneval_postprocess import (
        run_humaneval_official_evaluation,
        update_metrics_with_humaneval_scores
    )
    HUMANEVAL_OFFICIAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HumanEval official evaluation not available: {e}")
    HUMANEVAL_OFFICIAL_AVAILABLE = False


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False, eval_method="pass@k", k=1, run_humaneval_official=True):
    # Note: k parameter is kept for backwards compatibility but represents n_sampling internally
    # run_humaneval_official: whether to run official HumanEval evaluation (default True for humaneval)
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    
    # Check if this is HumanEval dataset (requires code execution evaluation)
    is_humaneval = (data_name == "humaneval")
    sample_pred_indices = []  # Track valid prediction indices for HumanEval
    
    if is_humaneval:
        # HumanEval: construct parameters for code execution evaluation
        # Each param: (prompt, generated_code, test_code, entry_point)
        params = []
        
        # Load original HumanEval data to get test cases and entry points if missing
        from data_loader import load_data
        try:
            original_data = load_data("humaneval", "test")
            original_data_dict = {ex.get('idx', i): ex for i, ex in enumerate(original_data)}
        except Exception as e:
            print(f"Warning: Could not load original HumanEval data: {e}")
            original_data_dict = {}
        
        for idx, sample in enumerate(samples):
            prompt = sample.get('prompt', '')
            test_info = sample['gt']  # This should be a dict with 'test', 'entry_point', 'canonical_solution'
            
            # Handle case where gt might be a string (from old data format)
            if isinstance(test_info, str):
                # Reconstruct dict from sample fields
                test_info = {
                    "test": sample.get('test', ''),
                    "entry_point": sample.get('entry_point', ''),
                    "canonical_solution": sample.get('canonical_solution', sample.get('gt', ''))
                }
                sample['gt'] = test_info  # Update sample for consistency
            
            test_code = test_info.get('test', '')
            entry_point = test_info.get('entry_point', '')
            
            # If test_code or entry_point are missing, try to load from original data
            if not test_code or not entry_point:
                sample_idx = sample.get('idx', idx)
                if sample_idx in original_data_dict:
                    orig_sample = original_data_dict[sample_idx]
                    if not test_code:
                        test_code = orig_sample.get('test', '')
                    if not entry_point:
                        entry_point = orig_sample.get('entry_point', '')
                    # Update sample for future use
                    test_info['test'] = test_code
                    test_info['entry_point'] = entry_point
                    sample['gt'] = test_info
            
            # For HumanEval, prefer 'code' field which contains raw model output
            # 'pred' field may have processed/stripped code that's missing spaces
            code_field = sample.get('code', [])
            if code_field and isinstance(code_field, list) and len(code_field) > 0:
                # Use code field as primary source for HumanEval
                preds = code_field
            else:
                # Fallback to pred field if code is not available
                preds = sample.get('pred', [])
                if preds is None:
                    preds = []
                if not isinstance(preds, list):
                    preds = [preds] if preds else []
            
            # Track valid predictions for this sample
            valid_pred_indices = []
            for pred_idx, pred in enumerate(preds):
                # Skip None or empty predictions
                if pred is None:
                    continue
                if isinstance(pred, str) and not pred.strip():
                    continue
                
                # Extract function body from raw model output
                extracted_code = extract_answer(pred, data_name)
                # Skip if extraction resulted in empty code
                if not extracted_code or not extracted_code.strip():
                    continue
                
                # Valid prediction - add to params
                params.append((prompt, extracted_code, test_code, entry_point))
                valid_pred_indices.append(pred_idx)
            
            # Store mapping of sample index to valid prediction indices
            sample_pred_indices.append(valid_pred_indices)
    else:
        # Standard evaluation: compare predictions with ground truth
        params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        if is_humaneval:
            # Use HumanEval-specific evaluation
            from grader import humaneval_check_process
            future = pool.map(humaneval_check_process, params, timeout=5)
        else:
            # Use standard math_equal evaluation
            future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    if is_humaneval:
        # For HumanEval, map scores back to original predictions (including skipped ones)
        for sample_idx, sample in enumerate(samples):
            valid_indices = sample_pred_indices[sample_idx]
            preds = sample.get('pred', [])
            if not isinstance(preds, list):
                preds = [preds] if preds else []
            
            # Initialize all scores as False (for skipped/empty predictions)
            sample_scores = [False] * len(preds)
            
            # Fill in scores for valid predictions
            for local_idx, valid_idx in enumerate(valid_indices):
                if idx + local_idx < len(scores):
                    sample_scores[valid_idx] = scores[idx + local_idx]
            
            sample['score'] = sample_scores
            score_mat.append(sample['score'])
            idx += len(valid_indices)
    else:
        # Standard evaluation: scores match predictions 1:1
        for sample in samples:
            sample['score'] = scores[idx: idx+len(sample['pred'])]
            assert len(sample['score']) == len(sample['pred'])
            score_mat.append(sample['score'])
            idx += len(sample['pred'])

    # Implement different evaluation methods
    if eval_method == "pass@k":
        # For pass@k, check if any of the k predictions are correct
        # k represents the number of samples generated (n_sampling)
        pass_at_k_scores = []
        for sample in samples:
            # Take first k predictions (or all if less than k available)
            k_predictions = sample['score'][:min(k, len(sample['score']))]
            # Sample passes if any of the k predictions are correct
            sample_passes = any(k_predictions)
            pass_at_k_scores.append(sample_passes)
        
        # Calculate pass@k accuracy
        pass_at_k_accuracy = sum(pass_at_k_scores) / len(pass_at_k_scores) * 100
        
        result_json = {
            "num_samples": len(samples),
            "num_scores": len(scores),
            "timeout_samples": timeout_cnt,
            "empty_samples": len([s for s in samples if not s['pred'][-1]]),
            "acc": round(pass_at_k_accuracy, 1),
            "eval_method": eval_method,
            "k": k
        }
    else:
        # Default behavior for other eval methods (original logic)
        max_len = max([len(s) for s in score_mat])

        for i, s in enumerate(score_mat):
            if len(s) < max_len:
                score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

        # output mean of each column of scores
        col_means= np.array(score_mat).mean(axis=0)
        mean_score = list(np.round(col_means * 100, decimals=1))

        result_json = {
            "num_samples": len(samples),
            "num_scores": len(scores),
            "timeout_samples": timeout_cnt,
            "empty_samples": len([s for s in samples if not s['pred'][-1]]),
            "acc": mean_score[0],
            "eval_method": eval_method,
            "k": k
        }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            
            if eval_method == "pass@k":
                # For pass@k, check if any of the first k predictions are correct
                # k represents the number of samples generated (n_sampling)
                k_predictions = sample['score'][:min(k, len(sample['score']))]
                sample_passes = any(k_predictions)
                type_scores[sample['type']].append(sample_passes)
            else:
                # Default behavior
                type_scores[sample['type']].append(sample['score'][-1])
        
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    # Run official HumanEval evaluation if applicable
    if is_humaneval and run_humaneval_official and HUMANEVAL_OFFICIAL_AVAILABLE:
        print("\n--- Running Official HumanEval Evaluation ---")
        try:
            # Determine output file path for official evaluation
            humaneval_output_file = file_path
            if not humaneval_output_file and samples:
                # If samples were passed directly, we need to save them temporarily
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                    for sample in samples:
                        f.write(json.dumps(sample) + '\n')
                    humaneval_output_file = f.name
            
            if humaneval_output_file:
                # Run official evaluation
                humaneval_results = run_humaneval_official_evaluation(
                    output_file=humaneval_output_file,
                    output_dir=os.path.dirname(humaneval_output_file) if humaneval_output_file else None
                )
                
                # Update result_json with official scores
                if "accuracy" in humaneval_results:
                    result_json["acc"] = round(humaneval_results["accuracy"], 1)
                    result_json["humaneval_official_acc"] = round(humaneval_results["accuracy"], 1)
                
                # Add pass@k values from official evaluation
                result_json["humaneval_official"] = {}
                for key, value in humaneval_results.items():
                    if key.startswith("pass@"):
                        result_json["humaneval_official"][key] = value
                
                result_json["humaneval_official"]["k"] = humaneval_results.get("k", k)
                result_json["humaneval_official"]["humaneval_samples_file"] = humaneval_results.get("humaneval_samples_file")
                
                print(f"Official HumanEval pass@1: {humaneval_results.get('pass@1', 'N/A')}")
                if k > 1:
                    print(f"Official HumanEval pass@{k}: {humaneval_results.get(f'pass@{k}', 'N/A')}")
        except Exception as e:
            print(f"Warning: Official HumanEval evaluation failed: {e}")
            result_json["humaneval_official_error"] = str(e)
    elif is_humaneval and run_humaneval_official and not HUMANEVAL_OFFICIAL_AVAILABLE:
        print("Warning: HumanEval official evaluation requested but human-eval package not installed")
        result_json["humaneval_official_error"] = "human-eval package not installed"

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
