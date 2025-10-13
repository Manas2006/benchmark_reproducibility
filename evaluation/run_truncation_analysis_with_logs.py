import argparse
import json
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import uuid

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from prob_recorder import BatchProbabilityRecorder


def _safe_first(item):
    if isinstance(item, list):
        return item[0] if item else ""
    return item


def _extract_cot_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    if '####' in raw_text:
        try:
            cot_text, _ = raw_text.split('####', 1)
            return cot_text.strip()
        except Exception:
            return raw_text
    return raw_text


def get_confidence_curve_with_logs(cot_text: str, final_answer_text: str, llm: LLM, hf_model, tokenizer, 
                                   temperature: float = 0.0, top_p: float = 1.0, 
                                   sample_idx: int = None, cot_type: str = "unknown", 
                                   original_prompt: str = "") -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Takes a single CoT and returns a list of confidence scores at each truncation step,
    along with detailed logs of input/output for each round.
    Uses 2-pass method: vLLM for generation, HF for probability scoring.
    Confidence is defined as the probability of the target answer tokens.
    
    Returns:
        Tuple of (confidence_scores, detailed_logs)
    """
    cot_text = cot_text or ""
    answer_text = str(final_answer_text) if final_answer_text is not None else ""

    cot_token_ids = tokenizer.encode(cot_text, add_special_tokens=False)
    answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if not answer_token_ids:
        return [], []

    confidence_scores: List[float] = []
    detailed_logs: List[Dict[str, Any]] = []
    truncation_percentages = np.linspace(0, 1, 11)  # 0% to 100% in 10% increments

    # Pass 1: Generate outputs with vLLM
    vllm_outputs = []
    prompts = []
    for step_idx, percent in enumerate(tqdm(truncation_percentages, desc="Pass 1: Generating", leave=False)):
        num_tokens_to_keep = int(len(cot_token_ids) * percent)
        truncated_cot_ids = cot_token_ids[:num_tokens_to_keep]
        truncated_cot_text = tokenizer.decode(truncated_cot_ids)

        # Construct the full prompt
        new_prompt = "\n\nGiven the reasoning above, what is the final answer? The final answer is(give your final answer directly):"
        prompt_text = f"{original_prompt}{truncated_cot_text}{new_prompt}"
        prompts.append(prompt_text)

    max_new_tokens = len(answer_token_ids) + 3

    # Generate all at once
    generation_start_time = datetime.now()
    vllm_outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        ),
    )
    generation_end_time = datetime.now()

    # Pass 2: Score with HF model
    device = next(hf_model.parameters()).device
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    for step_idx, (percent, vllm_out, prompt_text) in enumerate(tqdm(
        zip(truncation_percentages, vllm_outputs, prompts), 
        desc="Pass 2: Scoring", 
        total=len(vllm_outputs),
        leave=False
    )):
        num_tokens_to_keep = int(len(cot_token_ids) * percent)
        truncated_cot_ids = cot_token_ids[:num_tokens_to_keep]
        truncated_cot_text = tokenizer.decode(truncated_cot_ids)

        # Extract generated text and token IDs
        generated_text = vllm_out.outputs[0].text if vllm_out.outputs else ""
        gen_ids = vllm_out.outputs[0].token_ids if vllm_out.outputs else []
        prompt_ids = vllm_out.prompt_token_ids

        if not gen_ids:
            confidence_score = 0.0
            log_entry = {
                "step_idx": step_idx,
                "sample_idx": sample_idx,
                "cot_type": cot_type,
                "truncation_percent": float(percent),
                "num_tokens_kept": num_tokens_to_keep,
                "total_cot_tokens": len(cot_token_ids),
                "timestamp": generation_start_time.isoformat(),
                "generation_time_ms": (generation_end_time - generation_start_time).total_seconds() * 1000 / len(vllm_outputs),
                "input": {
                    "original_prompt": original_prompt,
                    "original_cot_text": cot_text,
                    "truncated_cot_text": truncated_cot_text,
                    "prompt_text": prompt_text,
                    "target_answer": answer_text,
                    "target_answer_tokens": answer_token_ids,
                    "cot_token_ids": cot_token_ids,
                    "truncated_cot_token_ids": truncated_cot_ids
                },
                "output": {
                    "generated_text": generated_text,
                    "confidence_score": confidence_score,
                    "target_token_probs": [],
                    "chosen_token_probs": [],
                    "chosen_token_ids": [],
                    "entropies": []
                },
                "generation_params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens
                }
            }
            confidence_scores.append(confidence_score)
            detailed_logs.append(log_entry)
            continue

        # Build input for HF model
        full_ids = prompt_ids + gen_ids
        L = len(full_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones((1, L), dtype=torch.long, device=device)
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)

        # Forward pass with HF model
        with torch.no_grad():
            logits = hf_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits  # [1, L, V]

        # Extract probabilities for generated tokens
        pl = len(prompt_ids)
        gl = len(gen_ids)
        start_idx = max(pl - 1, 0)
        end_idx = start_idx + gl
        logits_gen = logits[0, start_idx:end_idx, :]  # [gl, V]
        probs = torch.softmax(logits_gen.float(), dim=-1)  # [gl, V]

        # Get chosen token probabilities
        gen_ids_t = torch.tensor(gen_ids, dtype=torch.long, device=device)
        chosen_probs = probs.gather(1, gen_ids_t.unsqueeze(1)).squeeze(1)  # [gl]

        # Get target token probabilities (pointer-based)
        correct_ids = []
        correct_probs = []
        ptr = 0 if len(answer_token_ids) > 0 else -1
        exact_matches = 0

        for s in range(gl):
            chosen_id = int(gen_ids[s])
            if ptr == -1:
                correct_ids.append(None)
                correct_probs.append(None)
            else:
                target_id = int(answer_token_ids[ptr])
                correct_ids.append(target_id)
                correct_probs.append(float(probs[s, target_id].item()))

                if chosen_id == target_id:
                    exact_matches += 1
                    ptr += 1
                    if ptr >= len(answer_token_ids):
                        ptr = 0  # reset after full match
                else:
                    ptr = 0  # fallback to first token

        # Calculate entropies
        entropies = [float(torch.distributions.Categorical(probs=probs[s]).entropy().item()) for s in range(gl)]

        # Calculate confidence score: first target token probability at first step
        if correct_probs and correct_probs[0] is not None:
            confidence_score = correct_probs[0]
        else:
            # Fallback: average of all non-None target probabilities
            valid_probs = [p for p in correct_probs if p is not None]
            confidence_score = np.mean(valid_probs) if valid_probs else 0.0

        confidence_scores.append(confidence_score)

        # Create detailed log entry
        log_entry = {
            "step_idx": step_idx,
            "sample_idx": sample_idx,
            "cot_type": cot_type,
            "truncation_percent": float(percent),
            "num_tokens_kept": num_tokens_to_keep,
            "total_cot_tokens": len(cot_token_ids),
            "timestamp": generation_start_time.isoformat(),
            "generation_time_ms": (generation_end_time - generation_start_time).total_seconds() * 1000 / len(vllm_outputs),
            "input": {
                "original_prompt": original_prompt,
                "original_cot_text": cot_text,
                "truncated_cot_text": truncated_cot_text,
                "prompt_text": prompt_text,
                "target_answer": answer_text,
                "target_answer_tokens": answer_token_ids,
                "cot_token_ids": cot_token_ids,
                "truncated_cot_token_ids": truncated_cot_ids
            },
            "output": {
                "generated_text": generated_text,
                "confidence_score": float(confidence_score),
                "target_token_probs": [float(p) if p is not None else None for p in correct_probs],
                "chosen_token_probs": [float(p) for p in chosen_probs.tolist()],
                "chosen_token_ids": [int(i) for i in gen_ids],
                "entropies": entropies
            },
            "generation_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens
            }
        }
        
        detailed_logs.append(log_entry)

    return confidence_scores, detailed_logs


def _bucket_curves(records: List[Dict[str, Any]]) -> Tuple[List[Tuple[List[float], List[float]]], List[Tuple[List[float], List[float]]]]:
    correct_bucket: List[Tuple[List[float], List[float]]] = []
    incorrect_bucket: List[Tuple[List[float], List[float]]] = []

    for rec in records:
        gt_curve = rec.get("gt_curve", [])
        model_curve = rec.get("model_curve", [])
        if not gt_curve or not model_curve:
            continue
        if rec.get("originally_correct", False):
            correct_bucket.append((gt_curve, model_curve))
        else:
            incorrect_bucket.append((gt_curve, model_curve))

    return correct_bucket, incorrect_bucket


def _plot_aggregate(bucket: List[Tuple[List[float], List[float]]], title_suffix: str, dataset_name: str, model_stub: str, output_dir: str, job_id: str = None) -> str:
    if not bucket:
        return ""

    gt_curves = np.array([c[0] for c in bucket], dtype=float)
    model_curves = np.array([c[1] for c in bucket], dtype=float)

    # Ensure shapes are consistent
    min_len = min(gt_curves.shape[1] if gt_curves.ndim == 2 else 0,
                  model_curves.shape[1] if model_curves.ndim == 2 else 0)
    if min_len == 0:
        return ""
    gt_curves = gt_curves[:, :min_len]
    model_curves = model_curves[:, :min_len]

    steps = np.linspace(0, 100, min_len)  # percent
    gt_mean = np.nanmean(gt_curves, axis=0)
    gt_std = np.nanstd(gt_curves, axis=0)
    model_mean = np.nanmean(model_curves, axis=0)
    model_std = np.nanstd(model_curves, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.plot(steps, gt_mean, label="GT CoT", color="tab:blue", linewidth=2)
    plt.fill_between(steps, gt_mean - gt_std, gt_mean + gt_std, color="tab:blue", alpha=0.2)
    plt.plot(steps, model_mean, label="Model CoT", color="tab:orange", linewidth=2)
    plt.fill_between(steps, model_mean - model_std, model_mean + model_std, color="tab:orange", alpha=0.2)
    plt.xlabel("Truncation %", fontsize=12, labelpad=10)
    plt.ylabel("Confidence Score", fontsize=12, labelpad=10)
    plt.title(f"CoT Truncation Analysis - {dataset_name} - {title_suffix}", fontsize=14, pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Create unique filename with job ID if provided
    if job_id:
        filename = f"{dataset_name}_{title_suffix.replace(' ', '_').lower()}_{model_stub}_{job_id}.png"
    else:
        filename = f"{dataset_name}_{title_suffix.replace(' ', '_').lower()}_{model_stub}.png"
    
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def _save_detailed_logs(logs: List[Dict[str, Any]], output_dir: str, dataset_name: str, model_stub: str, job_id: str = None) -> str:
    """Save detailed logs to a JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with job ID and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if job_id:
        log_filename = f"{dataset_name}_truncation_detailed_logs_{model_stub}_{job_id}_{timestamp}.jsonl"
    else:
        log_filename = f"{dataset_name}_truncation_detailed_logs_{model_stub}_{timestamp}.jsonl"
    
    log_path = os.path.join(output_dir, log_filename)
    
    with open(log_path, 'w') as f:
        for log_entry in logs:
            f.write(json.dumps(log_entry) + '\n')
    
    return log_path


def _compute_records_with_logs(samples: List[Dict[str, Any]], llm: LLM, hf_model, tokenizer, 
                               temperature: float = 0.0, top_p: float = 1.0, 
                               progress_desc: str = "Samples") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    all_detailed_logs: List[Dict[str, Any]] = []

    for sample in tqdm(samples, desc=progress_desc):
        try:
            idx = sample.get("idx")
            gt_cot = sample.get("gt_cot", "")
            gt_answer = sample.get("gt", "")

            # Model fields
            model_output_text = _safe_first(sample.get("code", ""))
            model_cot = _extract_cot_text(model_output_text or "")
            model_answer = _safe_first(sample.get("pred", ""))

            # Correctness
            score = sample.get("score")
            if isinstance(score, list):
                originally_correct = any(bool(s) for s in score)
            else:
                originally_correct = bool(score)

            # Get the original prompt from the sample
            original_prompt = sample.get("prompt", "")
            
            # Get confidence curves with detailed logs (2-pass method)
            gt_curve, gt_logs = get_confidence_curve_with_logs(
                gt_cot or "", gt_answer or "", llm, hf_model, tokenizer, temperature, top_p, 
                sample_idx=idx, cot_type="gt", original_prompt=original_prompt
            )
            model_curve, model_logs = get_confidence_curve_with_logs(
                model_cot or "", model_answer or "", llm, hf_model, tokenizer, temperature, top_p, 
                sample_idx=idx, cot_type="model", original_prompt=original_prompt
            )

            # Add sample metadata to logs
            for log in gt_logs + model_logs:
                log.update({
                    "sample_metadata": {
                        "originally_correct": originally_correct,
                        "gt_answer": gt_answer,
                        "model_answer": model_answer,
                        "full_sample": sample  # Include full sample for reference
                    }
                })

            all_detailed_logs.extend(gt_logs + model_logs)

            records.append({
                "idx": idx,
                "originally_correct": originally_correct,
                "gt_curve": gt_curve,
                "model_curve": model_curve,
                "gt_cot": gt_cot,
                "model_cot": model_cot,
                "gt_answer": gt_answer,
                "model_answer": model_answer
            })
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Skip problematic sample but continue
            continue

    return records, all_detailed_logs


def run_truncation_analysis_over_samples_with_logs(samples: List[Dict[str, Any]], llm: LLM, hf_model, tokenizer, 
                                                    dataset_name: str, output_dir: str, model_name_or_path: str, 
                                                    temperature: float = 0.0, top_p: float = 1.0, job_id: str = None) -> Dict[str, Any]:
    model_stub = os.path.basename(model_name_or_path.rstrip('/'))
    output_dir = os.path.join(output_dir, "truncation_plots")
    os.makedirs(output_dir, exist_ok=True)

    print("Computing truncation analysis with detailed logging (2-pass method)...")
    records, detailed_logs = _compute_records_with_logs(samples, llm, hf_model, tokenizer, temperature, top_p)

    # Save detailed logs
    print("Saving detailed logs...")
    detailed_logs_path = _save_detailed_logs(detailed_logs, output_dir, dataset_name, model_stub, job_id)

    # Persist raw curves for further analysis
    if job_id:
        raw_path = os.path.join(output_dir, f"{dataset_name}_truncation_curves_{model_stub}_{job_id}.json")
    else:
        raw_path = os.path.join(output_dir, f"{dataset_name}_truncation_curves_{model_stub}.json")
    
    with open(raw_path, "w") as f:
        json.dump(records, f, indent=2)

    # Create summary statistics
    print("Creating summary statistics...")
    summary_stats = {
        "total_samples": len(samples),
        "processed_samples": len(records),
        "total_truncation_rounds": len(detailed_logs),
        "average_rounds_per_sample": len(detailed_logs) / len(records) if records else 0,
        "model_name": model_name_or_path,
        "dataset_name": dataset_name,
        "job_id": job_id,
        "generation_params": {
            "temperature": temperature,
            "top_p": top_p
        },
        "analysis_timestamp": datetime.now().isoformat()
    }

    if job_id:
        summary_path = os.path.join(output_dir, f"{dataset_name}_truncation_summary_{model_stub}_{job_id}.json")
    else:
        summary_path = os.path.join(output_dir, f"{dataset_name}_truncation_summary_{model_stub}.json")
    
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    correct_bucket, incorrect_bucket = _bucket_curves(records)
    correct_plot = _plot_aggregate(correct_bucket, "Correct", dataset_name, model_stub, output_dir, job_id)
    incorrect_plot = _plot_aggregate(incorrect_bucket, "Incorrect", dataset_name, model_stub, output_dir, job_id)

    print(f"Analysis complete!")
    print(f"- Processed {len(records)} samples")
    print(f"- Generated {len(detailed_logs)} detailed log entries")
    print(f"- Detailed logs saved to: {detailed_logs_path}")
    print(f"- Summary statistics saved to: {summary_path}")

    return {
        "raw_curves_path": raw_path,
        "detailed_logs_path": detailed_logs_path,
        "summary_stats_path": summary_path,
        "correct_plot": correct_plot,
        "incorrect_plot": incorrect_plot,
        "summary_stats": summary_stats
    }


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def main_cli():
    parser = argparse.ArgumentParser(description="Run CoT Truncation Analysis with detailed input/output logging (2-pass method).")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSONL results file from a completed evaluation run.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the vLLM-compatible model to use for the analysis.")
    parser.add_argument("--output_dir", default="truncation_plots", type=str)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID for unique filenames")
    args = parser.parse_args()

    # Setup vLLM model and tokenizer
    print(f"Initializing vLLM with model: {args.model_name_or_path}")
    print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
    
    try:
        llm = LLM(
            model=args.model_name_or_path,
            trust_remote_code=True,
            pipeline_parallel_size=args.pipeline_parallel_size,
            gpu_memory_utilization=0.4,  # Leave room for HF model
        )
        print("Successfully initialized vLLM")
    except RuntimeError as e:
        print(f"vLLM initialization failed: {e}")
        raise
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Setup HF model for scoring (Pass 2)
    print(f"Loading HF model for probability scoring...")
    try:
        device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,  # High-accuracy scoring
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        print(f"Successfully loaded HF model on {device_map}")
    except Exception as e:
        print(f"WARNING: Failed to load HF model on CUDA, falling back to CPU. Error: {e}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        ).eval()

    samples = _load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples from {args.input_file}")

    results = run_truncation_analysis_over_samples_with_logs(
        samples=samples,
        llm=llm,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        top_p=args.top_p,
        job_id=args.job_id,
    )
    
    print("\n" + "="*50)
    print("ANALYSIS RESULTS:")
    print("="*50)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main_cli()
