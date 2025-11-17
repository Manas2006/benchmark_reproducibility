import random
import os
import argparse
import time
import json
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from math import exp
import sys
import uuid
sys.path.insert(0, os.path.abspath("."))

try:
    # Optional Together API import; only required when using --use_together_api
    from together import Together
except Exception:
    Together = None

import torch
import gc # <<< NEW: Ensure gc is imported
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from prob_recorder import BatchProbabilityRecorder
from run_truncation_analysis_with_logs import run_truncation_analysis_over_samples_with_logs as run_truncation_analysis_over_samples
try:
    # Optional import for answer confidence calculation (ECE)
    from extract_answer_confidence import extract_answer_confidence
except ImportError:
    extract_answer_confidence = None

# --- NEW: helpers for subprocess Pass-1
import tempfile, subprocess


# <<< NEW: Helper Function to Report Memory Usage (Copied from our previous example)
def print_gpu_memory_usage(stage_name: str):
    """
    Prints the current, free, and total GPU memory usage.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot measure GPU memory.")
        return
        
    # Get the memory stats for the primary GPU device
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory, _ = torch.cuda.mem_get_info(device)
    used_memory = total_memory - free_memory

    # Convert bytes to Gigabytes (GB) for readability
    bytes_to_gb = 1 / (1024 ** 3)
    used_gb = used_memory * bytes_to_gb
    free_gb = free_memory * bytes_to_gb
    total_gb = total_memory * bytes_to_gb

    print("\n" + "="*50)
    print(f"--- {stage_name} ---")
    print(f"Used Memory : {used_gb:.2f} GB")
    print(f"Free Memory : {free_gb:.2f} GB")
    print(f"Total Memory: {total_gb:.2f} GB")
    print("="*50 + "\n")


# ============================================================
# OOM-safe streaming scorer for Pass-2 (HF, fp32 recommended)
# ============================================================

from typing import List, Dict, Any, Tuple, Optional

try:
    from tqdm import tqdm as _tqdm  # reuse tqdm alias safely
except Exception:
    _tqdm = None


# ---------- batching helpers ----------

def _build_len_metadata(vllm_outputs) -> List[Dict[str, int]]:
    """Collect prompt/gen lengths for each vLLM output entry."""
    meta = []
    for i_out, out in enumerate(vllm_outputs):
        p_ids = out.prompt_token_ids
        g_ids = out.outputs[0].token_ids if out.outputs and out.outputs[0].token_ids else []
        meta.append({
            "i_out": i_out,
            "pl": len(p_ids),
            "gl": len(g_ids),
            "L": len(p_ids) + len(g_ids),
        })
    return meta


def _form_token_budget_batches(
    meta: List[Dict[str, int]],
    max_tokens_per_batch: int = 16384,
    max_len_per_batch: Optional[int] = None,
) -> List[List[Dict[str, int]]]:
    """
    Greedy packing by 'token budget':
      batch_size * max_len_in_batch <= max_tokens_per_batch
    and an optional hard cap on per-sample sequence length:
      max_len_in_batch <= max_len_per_batch
    """
    meta = sorted(meta, key=lambda x: x["L"], reverse=True)
    batches: List[List[Dict[str, int]]] = []
    cur: List[Dict[str, int]] = []
    cur_max = 0

    for m in meta:
        L = m["L"]

        # If a hard cap is set and the sample itself exceeds it, run it alone.
        if max_len_per_batch is not None and L > max_len_per_batch:
            if cur:
                batches.append(cur)
                cur, cur_max = [], 0
            batches.append([m])
            continue

        prospective_bs = len(cur) + 1
        prospective_max = max(cur_max, L)

        # Honor the hard per-batch length cap if given
        if max_len_per_batch is not None and prospective_max > max_len_per_batch:
            if cur:
                batches.append(cur)
            cur, cur_max = [m], L
            continue

        cost = prospective_bs * prospective_max
        if cur and cost > max_tokens_per_batch:
            batches.append(cur)
            cur, cur_max = [], 0

        cur.append(m)
        cur_max = max(cur_max, L)

    if cur:
        batches.append(cur)
    return batches


def _pack_microbatch(
    batch_meta: List[Dict[str, int]],
    vllm_outputs,
    tokenizer,
    device: torch.device,
):
    """Right-pad each row to the microbatch max length; return (ids, mask, pos_ids, lens...)."""
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    B = len(batch_meta)
    max_len = max((m["L"] for m in batch_meta), default=0)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    position_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)

    prompt_lens, gen_lens, gen_ids_list = [], [], []

    for row, m in enumerate(batch_meta):
        out = vllm_outputs[m["i_out"]]
        p_ids = list(out.prompt_token_ids)
        g_ids = list(out.outputs[0].token_ids) if out.outputs and out.outputs[0].token_ids else []
        ids = p_ids + g_ids
        L = len(ids)
        if L > 0:
            input_ids[row, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_mask[row, :L] = 1
            position_ids[row, :L] = torch.arange(L, dtype=torch.long, device=device)
        prompt_lens.append(len(p_ids))
        gen_lens.append(len(g_ids))
        gen_ids_list.append(g_ids)

    return input_ids, attn_mask, position_ids, prompt_lens, gen_lens, gen_ids_list


def _pack_expert_cot_microbatch(
    expert_samples_in_batch: List[Dict[str, int]],
    vllm_outputs,
    samples: List[Dict[str, Any]],
    n_sampling: int,
    tokenizer,
    device: torch.device,
):
    """
    Pack a microbatch for expert CoT probability calculation.
    Each row contains: prompt_token_ids + expert_cot_token_ids
    
    Returns:
        input_ids, attn_mask, position_ids, prompt_lens, expert_cot_lens, expert_cot_ids_list
    """
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    B = len(expert_samples_in_batch)
    
    # Build sequences: prompt + expert_cot
    all_seqs = []
    prompt_lens = []
    expert_cot_lens = []
    expert_cot_ids_list = []
    
    for m in expert_samples_in_batch:
        i_out = m['i_out']
        
        # Get prompt token IDs from vllm_outputs
        out = vllm_outputs[i_out]
        p_ids = list(out.prompt_token_ids)
        
        # Get the corresponding sample
        sample_idx = i_out // n_sampling
        current_sample = samples[sample_idx]
        
        # Get expert CoT text and tokenize it (use gt_cot as expert)
        expert_cot_text = current_sample.get("gt_cot", "")
        e_ids = tokenizer.encode(expert_cot_text, add_special_tokens=False)
        
        all_seqs.append((p_ids, e_ids))
        prompt_lens.append(len(p_ids))
        expert_cot_lens.append(len(e_ids))
        expert_cot_ids_list.append(e_ids)
    
    # Calculate max length
    max_len = max((len(p_ids) + len(e_ids) for p_ids, e_ids in all_seqs), default=0)
    
    # Create tensors
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    position_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    
    # Fill tensors
    for row, (p_ids, e_ids) in enumerate(all_seqs):
        ids = p_ids + e_ids
        L = len(ids)
        
        if L > 0:
            input_ids[row, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_mask[row, :L] = 1
            position_ids[row, :L] = torch.arange(L, dtype=torch.long, device=device)
    
    return input_ids, attn_mask, position_ids, prompt_lens, expert_cot_lens, expert_cot_ids_list


# ---------- metrics: chosen probs + pointer-correct probs ----------

def _pointer_metrics_for_sample(
    probs_i: torch.Tensor,              # [gl, V], float on device
    gen_ids: List[int],
    gold_answer_text: str,
    tokenizer,
) -> Dict[str, Any]:
    """
    - chosen_probs: P(model's actual token at each step)
    - correct_probs: pointer semantics over gold answer tokens:
        pointer starts at t1; advance only if chosen==current gold token; else reset to t1.
        At each step, 'correct' = P(gold_token[pointer]).
    - entropies: per-step categorical entropy.
    """
    gl, V = probs_i.shape
    device = probs_i.device

    if gl == 0:
        return {
            "chosen_ids": [],
            "chosen_probs": [],
            "correct_ids": [],
            "correct_probs": [],
            "entropies": [],
            "exact_matches": 0,
        }

    gen_ids_t = torch.tensor(gen_ids, dtype=torch.long, device=device)
    chosen_probs_t = probs_i.gather(1, gen_ids_t.unsqueeze(1)).squeeze(1)  # [gl]

    ans = (gold_answer_text or "").strip()
    ans_ids = tokenizer.encode(ans, add_special_tokens=False)

    entropies = [
        float(torch.distributions.Categorical(probs=probs_i[s]).entropy().item())
        for s in range(gl)
    ]

    correct_ids: List[Optional[int]] = []
    correct_probs: List[Optional[float]] = []
    ptr = 0 if len(ans_ids) > 0 else -1
    exact_matches = 0

    for s in range(gl):
        chosen_id = int(gen_ids[s])
        if ptr == -1:
            correct_ids.append(None)
            correct_probs.append(None)
        else:
            gold_id = int(ans_ids[ptr])
            correct_ids.append(gold_id)
            correct_probs.append(float(probs_i[s, gold_id].item()))

            if chosen_id == gold_id:
                exact_matches += 1
                ptr += 1
                if ptr >= len(ans_ids):
                    ptr = 0  # reset after a full match of the answer
            else:
                ptr = 0   # fallback to first token

    return {
        "chosen_ids": [int(x) for x in gen_ids],
        "chosen_probs": [float(x) for x in chosen_probs_t.tolist()],
        "correct_ids": correct_ids,
        "correct_probs": correct_probs,
        "entropies": entropies,
        "exact_matches": int(exact_matches),
    }


# ---------- main streaming scorer ----------

def run_hf_scoring_streaming(
    hf_model,                              # AutoModelForCausalLM (fp32/bf16)
    vllm_outputs,                              # list of vLLM-like outputs (supports _MiniOut)
    samples: List[Dict[str, Any]],         # parallel to prompts; each must have 'gt'
    tokenizer,
    n_sampling: int,
    enable_path_vectors: bool,
    path_vectors_npz: str,
    *,
    max_tokens_per_batch: int = 16384,     # total token budget per microbatch (B * L_max)
    max_len_per_batch: Optional[int] = None,  # optional hard cap on per-sample L
    show_progress: bool = True,
    progress_label: str = "Pass2(HF)",
    log_monitor_json: bool = False,
    vram_guard_gb: Optional[float] = None,     # if set, CPU fallback when free VRAM < this
) -> Tuple[List[Optional[Dict[str, Any]]], bool, Dict[int, float]]:
    """
    Returns:
      - hf_results_per_output: list aligned to vllm_outputs; each item includes:
          chosen_ids, chosen_probs, correct_ids, correct_probs, entropies, exact_matches
      - wrote_npz: True if path vectors saved
      - expert_prob_results: dict mapping i_out to expert CoT probability
    """
    device = next(hf_model.parameters()).device
    use_cuda = (device.type == "cuda")

    # Build batches by token budget / optional length cap
    meta = _build_len_metadata(vllm_outputs)
    batches = _form_token_budget_batches(
        meta, max_tokens_per_batch=max_tokens_per_batch, max_len_per_batch=max_len_per_batch
    )

    hf_results_per_output: List[Optional[Dict[str, Any]]] = [None] * len(vllm_outputs)
    expert_prob_results: Dict[int, float] = {}  # <<< NEW: Store expert CoT probabilities
    pathvec_store: Dict[str, Any] = {}

    total_samples = len(vllm_outputs)
    samples_done = 0
    t0 = time.time()

    # Progress bar over samples, not batches
    pbar = None
    if show_progress and _tqdm is not None:
        pbar = _tqdm(
            total=total_samples,
            desc=progress_label,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} samples | Elapsed: {elapsed} | ETA: {remaining}",
            leave=True,
        )

    for b_idx, batch_meta in enumerate(batches):
        # Decide device per batch (optional CPU fallback if free VRAM is low)
        this_device = device
        moved_model_to_cpu = False
        free_gb = None

        if vram_guard_gb is not None and use_cuda:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = round(free_bytes / (1024**3), 2)
                if free_gb < float(vram_guard_gb):
                    # CPU fallback for this batch
                    hf_model.to("cpu")
                    this_device = torch.device("cpu")
                    moved_model_to_cpu = True
            except Exception:
                pass  # ignore mem probe failures

        # =========================================================
        # <<< START: NEW EXPERT CoT CALCULATION BLOCK >>>
        # =========================================================
        expert_samples_in_batch = []
        for m in batch_meta:
            i_out = m['i_out']
            sample_idx = i_out // n_sampling
            current_sample = samples[sample_idx]
            # Check if gt_cot exists and is non-empty (use as expert CoT)
            if current_sample.get("gt_cot") and str(current_sample.get("gt_cot")).strip():
                expert_samples_in_batch.append(m)
        
        print(f"DEBUG: Batch {b_idx} has {len(expert_samples_in_batch)} samples with valid gt_cot out of {len(batch_meta)} total samples")
        
        if expert_samples_in_batch:
            print(f"DEBUG: Processing {len(expert_samples_in_batch)} expert CoT samples in batch {b_idx}")
            try:
                # 1. Create a new micro-batch for expert CoT sequences
                expert_input_ids, expert_attn_mask, expert_pos_ids, expert_prompt_lens, expert_cot_lens, expert_cot_ids_list = \
                    _pack_expert_cot_microbatch(expert_samples_in_batch, vllm_outputs, samples, n_sampling, tokenizer, this_device)

                # 2. Run a new forward pass
                with torch.no_grad():
                    expert_logits = hf_model(
                        input_ids=expert_input_ids,
                        attention_mask=expert_attn_mask,
                        position_ids=expert_pos_ids,
                        use_cache=False,
                        output_hidden_states=False,
                    ).logits  # [NumExpertSamples, SeqLen, VocabSize]

                # 3. Calculate joint probability for each sample
                for row, m in enumerate(expert_samples_in_batch):
                    pl = expert_prompt_lens[row]
                    e_cot_l = expert_cot_lens[row]
                    
                    print(f"DEBUG: Sample {m['i_out']} - prompt_len={pl}, expert_cot_len={e_cot_l}")
                    
                    if e_cot_l == 0:
                        print(f"DEBUG: Skipping sample {m['i_out']} - expert CoT length is 0")
                        continue
                    
                    # Skip if prompt length is 0 (this shouldn't happen but could cause indexing issues)
                    if pl == 0:
                        print(f"WARNING: Skipping expert CoT calculation for sample {m['i_out']} - prompt length is 0")
                        continue
                    
                    # Isolate logits for the CoT part
                    # The logits at position i predict token i+1
                    # So for expert CoT starting at position pl, we need logits[pl-1:pl+e_cot_l-1]
                    # Fix: ensure start index is never negative
                    start_idx = max(0, pl - 1)
                    end_idx = pl + e_cot_l - 1
                    
                    # Additional safety check
                    if start_idx >= end_idx:
                        print(f"WARNING: Skipping expert CoT calculation for sample {m['i_out']} - invalid slice range [{start_idx}:{end_idx}]")
                        continue
                    
                    cot_logits = expert_logits[row, start_idx:end_idx, :]  # [e_cot_l, V]
                    cot_probs = torch.softmax(cot_logits.float(), dim=-1)  # [e_cot_l, V]
                    
                    # Get the actual expert CoT token IDs
                    correct_token_ids = torch.tensor(expert_cot_ids_list[row], dtype=torch.long, device=this_device)  # [e_cot_l]
                    
                    # Gather the probabilities of the correct tokens at each step
                    step_probs = cot_probs.gather(1, correct_token_ids.unsqueeze(1)).squeeze(1)  # [e_cot_l]
                    
                    # Calculate the joint probability by taking the product
                    # Use log-probabilities for numerical stability, then exponentiate
                    log_probs = torch.log(step_probs + 1e-10)  # Add small epsilon to avoid log(0)
                    joint_log_prob = torch.sum(log_probs)
                    joint_probability = torch.exp(joint_log_prob).item()

                    expert_prob_results[m['i_out']] = joint_probability
            
            except Exception as e:
                print(f"WARNING: Expert CoT calculation failed for batch {b_idx}: {e}")
                # Continue with normal processing even if expert CoT fails
        # =========================================================
        # <<< END: NEW EXPERT CoT CALCULATION BLOCK >>>
        # =========================================================

        # Pack batch tensors on 'this_device'
        input_ids, attn_mask, position_ids, prompt_lens, gen_lens, gen_ids_list = \
            _pack_microbatch(batch_meta, vllm_outputs, tokenizer, this_device)

        # Forward
        with torch.no_grad():
            logits_mb = hf_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=False,
            ).logits  # [B, L, V] on 'this_device'

        # Per-row processing
        B = input_ids.size(0)
        rows_this_batch = 0
        for row in range(B):
            info = batch_meta[row]
            i_out = info["i_out"]
            pl = prompt_lens[row]
            gl = gen_lens[row]

            if gl == 0:
                hf_results_per_output[i_out] = {
                    "chosen_ids": [],
                    "chosen_probs": [],
                    "correct_ids": [],
                    "correct_probs": [],
                    "entropies": [],
                    "exact_matches": 0,
                }
                continue

            start_idx = max(pl - 1, 0)
            end_idx = start_idx + gl
            l_rows = logits_mb[row, start_idx:end_idx, :]         # [gl, V]
            probs_i = torch.softmax(l_rows.float(), dim=-1)         # [gl, V]

            sample_index = i_out // n_sampling
            gold_answer_text = samples[sample_index].get("gt", "")

            metrics = _pointer_metrics_for_sample(probs_i, gen_ids_list[row], gold_answer_text, tokenizer)
            hf_results_per_output[i_out] = metrics

            if enable_path_vectors and np is not None:
                pathvec_store[f"s{i_out}"] = probs_i.detach().to("cpu").half().numpy()

            rows_this_batch += gl

        # Free batch tensors and (optionally) move model back to original device
        del logits_mb
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if moved_model_to_cpu:
            hf_model.to(device)

        # Progress + monitor
        samples_done += len(batch_meta)
        if pbar is not None:
            pbar.update(len(batch_meta))
        if log_monitor_json:
            elapsed = time.time() - t0
            monitor = {
                "type": "MONITOR_SCORING",
                "batch_index": b_idx,
                "batches_total": len(batches),
                "batch_size": len(batch_meta),
                "rows_scored_this_batch": rows_this_batch,
                "samples_done": samples_done,
                "samples_total": total_samples,
                "elapsed_sec_total": round(elapsed, 3),
            }
            if free_gb is not None:
                monitor["free_vram_gb_before_batch"] = free_gb
            print(json.dumps(monitor))

    if pbar is not None:
        pbar.close()
        print(f"[Pass2] All {total_samples} samples processed in {time.time() - t0:.1f}s")

    wrote_npz = False
    if enable_path_vectors and len(pathvec_store) > 0:
        try:
            os.makedirs(os.path.dirname(path_vectors_npz), exist_ok=True)
            np.savez_compressed(path_vectors_npz, **pathvec_store)
            wrote_npz = True
        except Exception as e:
            print(f"WARNING: failed to write path vectors npz: {e}")

    return hf_results_per_output, wrote_npz, expert_prob_results


# --- NEW: Pass-1 in a child process (so EngineCore exits and frees VRAM)
def run_pass1_in_subprocess(model_name_or_path: str, prompts: List[str], sampling_cfg: Dict[str, Any], gpu_memory_utilization: float = 0.4, max_model_len: int = 8192, args: Any = None):
    """
    sampling_cfg keys: temperature, top_p, top_k, max_tokens, stop (list[str]), stop_token_ids (list[int]|None)
    Returns list of dicts: {prompt, prompt_token_ids, generated_text, generated_token_ids}
    (Requires pass1_child_vllm.py next to this file.)
    """
    try:
        # 1. Load only the model's config file (very fast)
        print("🔧 Checking model's config for max_position_embeddings...")
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 2. Get the model's architectural limit from its config
        derived_max_len = getattr(config, "max_position_embeddings", 8192)
        print(f"✅ Model's architectural limit (max_position_embeddings): {derived_max_len}")

        # 3. Get the user's desired limit from the command line arguments
        user_max_len = getattr(args, 'max_model_len', 8192)

        # 4. Use the SMALLER of the two values to be safe
        final_max_model_len = min(derived_max_len, user_max_len)
        print(f"🎯 Using final max_model_len for vLLM: {final_max_model_len}")

    except Exception as e:
        print(f"⚠️ Warning: Could not automatically determine max_model_len from config. Falling back to default. Error: {e}")
        final_max_model_len = max_model_len # Fallback to the original value


    with tempfile.TemporaryDirectory() as td:
        in_path  = os.path.join(td, "pass1_in.json")
        out_path = os.path.join(td, "pass1_out.json")
        payload = {
            "model": model_name_or_path,
            "prompts": prompts,
            "temperature": sampling_cfg.get("temperature", 0.0),
            "top_p": sampling_cfg.get("top_p", 1.0),
            "top_k": sampling_cfg.get("top_k", 0),
            "max_tokens": sampling_cfg.get("max_tokens_per_call", 2048),
            "stop": sampling_cfg.get("stop", []),
            "stop_token_ids": sampling_cfg.get("stop_token_ids", None),
            "gpu_memory_utilization": gpu_memory_utilization,
            # Use the final, safe value determined above
            "max_model_len": final_max_model_len,
        }
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        print(f"🚀 Starting vLLM subprocess with max_model_len={final_max_model_len}, gpu_memory_utilization={gpu_memory_utilization}")
        # Spawn child. It will exit and release all VRAM.
        # Pass environment variables, but filter out invalid HF tokens
        env = os.environ.copy()
        # Only pass HF tokens if they look valid (not placeholder values)
        for token_var in ['HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN']:
            if token_var in env:
                token = env[token_var]
                if not token or token.startswith('your_token') or len(token) < 10:
                    # Remove invalid tokens
                    env.pop(token_var, None)
                    print(f"⚠️ Filtered out invalid {token_var}")
        subprocess.run([sys.executable, "pass1_child_vllm.py", in_path, out_path], check=True, env=env)
        print("✅ vLLM subprocess completed successfully")
        
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--prompt", type=str, help="Custom prompt template to use instead of prompt_type")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--job_id", type=str, help="Job ID to include in output filename")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--eval_method", type=str, default="pass@k", help="Evaluation method (pass@k, maj@k, rm@k)")
    parser.add_argument("--enable_prob_tracking", action="store_true",
                        help="Enable probability tracking of target answer tokens (requires vLLM)")
    # Single switch for full distribution (path vectors)
    parser.add_argument("--enable_path_vectors", action="store_true",
                        help="Store full per-step token distributions to an .npz file (one run/job per file).")
    parser.add_argument("--max_path_steps", type=int, default=0,
                        help="Maximum steps to record for path vectors (0 or negative = unlimited, to limit memory)")
    parser.add_argument("--run_truncation_analysis_after_eval", action="store_true",
                        help="Run CoT truncation analysis immediately after evaluation")
    # Together API integration
    parser.add_argument("--use_together_api", action="store_true", help="Use Together API for generation instead of local models")
    parser.add_argument("--together_api_key", type=str, default=None, help="Together API key (or via TOGETHER_API_KEY env var)")
    parser.add_argument("--together_logprobs", type=int, default=0, help="Return top logprobs per token (0-5); if 0, do not request logprobs")

    # --- NEW: run vLLM generation in a short-lived subprocess (frees VRAM before HF scoring)
    parser.add_argument("--pass1_subprocess", action="store_true",
                        help="Run vLLM Pass-1 in a subprocess so its VRAM is freed before Pass-2 HF scoring.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for VLLM (default: 0.9)")

    args = parser.parse_args()
    args.top_p = (1 if args.temperature == 0 else args.top_p)  # top_p must be 1 when using greedy sampling (vllm)
    
    # 🚀 AUTO-ENABLE SUBPROCESS MODE FOR PROBABILITY TRACKING
    # This automatically uses subprocess mode when probability tracking is enabled
    # to avoid memory conflicts between vLLM and HF scorer
    if args.enable_prob_tracking and args.use_vllm and not getattr(args, "pass1_subprocess", False):
        print("🔧 Auto-enabling subprocess mode for probability tracking to avoid memory conflicts")
        args.pass1_subprocess = True
    
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    # Use consistent naming logic with runner.py
    if hasattr(args, 'prompt') and args.prompt and args.prompt_type:
        prompt_type_for_file = f"{args.prompt_type}_custom"
    elif hasattr(args, 'prompt') and args.prompt:
        prompt_type_for_file = "custom"
    elif args.prompt_type:
        prompt_type_for_file = args.prompt_type
    else:
        prompt_type_for_file = "cot"
    
    out_file_prefix = f"{args.split}_{prompt_type_for_file}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add job_id to filename if provided
    if hasattr(args, 'job_id') and args.job_id:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{args.job_id}.jsonl"
    else:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # deduplicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def load_recorder_sidecar(output_dir, dataset, run_id, request_id):
    """Return dict or None."""
    req_dir = os.path.join(output_dir, dataset, run_id, "requests")
    path = os.path.join(req_dir, f"{request_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to read recorder sidecar {path}: {e}")
        return None



def setup(args):
    # load model
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if args.use_together_api:
        # Defer model loading; Together API uses HTTP
        llm = "together_api"
        tokenizer = None

    elif args.use_vllm:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

        if getattr(args, "pass1_subprocess", False):
            # Do NOT load vLLM in the parent; child will load & exit (freeing VRAM)
            llm = "vllm_subprocess"
        else:
            # In-process vLLM (original behavior)
            print(f"🚀 Loading vLLM model: {args.model_name_or_path}")
            
            # Limit max_model_len to 8192 to reduce KV cache memory requirements
            # Most math problems are much shorter than this, so this is a safe limit
            max_model_len = getattr(args, 'max_model_len', 8192)
            
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=max_model_len,  # Limit context length
                max_num_seqs=64,
            )
            print(f"✅ vLLM model loaded successfully with max_model_len={max_model_len}!")
            # <<< NEW: Print memory after loading vLLM
            print_gpu_memory_usage("Stage 2: After Loading Model with vLLM")


        # Load an HF scorer if probability tracking is enabled
        hf_scorer = None
        """
        if args.enable_prob_tracking:
            try:
                # Only try GPU if we have a decent amount of memory
                if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / (1024**3) >= 4.0:
                    device_map = {"": 0}
                    print("Loading HF scorer on GPU...")
                    hf_scorer = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map,
                        trust_remote_code=True,
                    ).eval()
                    # <<< NEW: Print memory after loading HF Scorer
                    print_gpu_memory_usage("Stage 4: After Loading HF Scorer Model (on GPU)")
                else:
                    raise RuntimeError("Insufficient GPU memory for HF scorer")
            except Exception as e:
                print(f"WARNING: failed to load HF scorer on CUDA; falling back to CPU. Error: {e}")
                hf_scorer = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True,
                ).eval()
                # <<< NEW: Print message for CPU fallback
                print("--- Stage 4: HF Scorer loaded on CPU (no GPU memory used) ---")
        """
        args._hf_scorer = hf_scorer

    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )
        args._hf_scorer = None

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
    
    # 🧹 COMPREHENSIVE GPU MEMORY CLEANUP
    print("\n" + "="*60)
    print("🎯 EVALUATION COMPLETED - STARTING FINAL GPU MEMORY CLEANUP")
    print("="*60)
    
    # Get references to models for cleanup
    cleanup_llm = None
    cleanup_hf_scorer = None
    
    if args.use_vllm and not getattr(args, "pass1_subprocess", False):
        # We have a loaded vLLM model
        cleanup_llm = llm
    
    if hasattr(args, '_hf_scorer') and args._hf_scorer is not None:
        # We have a loaded HF scorer
        cleanup_hf_scorer = args._hf_scorer
    
    # Perform comprehensive cleanup
    cleanup_gpu_memory(cleanup_llm, cleanup_hf_scorer)
    
    print("="*60)
    print("✅ EVALUATION AND CLEANUP COMPLETED SUCCESSFULLY!")
    print("="*60)


def cleanup_gpu_memory(llm=None, hf_scorer=None):
    """
    Comprehensive GPU memory cleanup function.
    This ensures all GPU memory is freed when switching models.
    """
    print("🧹 Starting comprehensive GPU memory cleanup...")
    
    # <<< NEW: Use our helper to show memory BEFORE cleanup
    print_gpu_memory_usage("Memory State Before Cleanup")
    
    # Cleanup vLLM model
    if llm is not None and hasattr(llm, 'llm_engine'):
        print("🗑️ Cleaning up vLLM model...")
        try:
            # Force cleanup of vLLM engine
            if hasattr(llm.llm_engine, 'engine_core'):
                del llm.llm_engine.engine_core
            del llm.llm_engine
        except Exception as e:
            print(f"⚠️ Warning: Error cleaning vLLM engine: {e}")
        del llm
    
    # Cleanup HF scorer
    if hf_scorer is not None:
        print("🗑️ Cleaning up HF scorer...")
        try:
            del hf_scorer
        except Exception as e:
            print(f"⚠️ Warning: Error cleaning HF scorer: {e}")
    
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
    
    # <<< NEW: Use our helper to show memory AFTER cleanup
    print_gpu_memory_usage("Memory State After Cleanup")


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    sidecar_base = os.path.join(args.output_dir, data_name, run_id, "requests")
    os.makedirs(sidecar_base, exist_ok=True)
    if getattr(args, "job_id", None):
        run_id += f"_{args.job_id}"
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # A unique run id to avoid overwrites and to key the path-vectors file
    path_vectors_npz = os.path.join(args.output_dir, data_name, f"pathvec_{run_id}.npz")

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)
            # Output structured information for monitoring
            print(f"MONITOR_PROMPT: {json.dumps({'idx': idx, 'question': example['question'], 'prompt': full_prompt})}")

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
            # Path vectors metadata goes in every sample
            "path_vectors_enabled": bool(args.enable_path_vectors),
            "path_vectors_file": None,  # filled in at end if enabled
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template and tokenizer is not None:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    elif args.apply_chat_template and tokenizer is None:
        print("WARNING: apply_chat_template requested but tokenizer unavailable; skipping chat template application.")
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    vllm_outputs = []  # Store vLLM outputs for HF scoring later
    epoch_request_maps = []
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        print(f"MONITOR_EPOCH: {json.dumps({'epoch': epoch, 'total_epochs': max_func_call, 'remaining_prompts': len(remain_prompts)})}")
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_together_api:
            if Together is None:
                raise RuntimeError("Together API SDK not installed. Please install 'together' package.")
            api_key = args.together_api_key or os.environ.get("TOGETHER_API_KEY")
            if not api_key:
                raise RuntimeError("Together API key not provided. Use --together_api_key or set TOGETHER_API_KEY env var.")
            client = Together(api_key=api_key)
            outputs = []
            # Together API: per-prompt generation to preserve ordering and allow logprobs capture
            for i, prompt_text in enumerate(prompts):
                completion = client.chat.completions.create(
                    model=args.model_name_or_path,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=args.max_tokens_per_call,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    logprobs=(args.together_logprobs if args.together_logprobs and args.together_logprobs > 0 else None),
                )
                choice = completion.choices[0]
                outputs.append(choice.message.content or "")
        
        elif args.use_vllm:
            # ----- vLLM path -----
            if isinstance(llm, str) and llm == "vllm_subprocess":
                # Pass-1 in child (EngineCore exits → frees VRAM)
                sampling_cfg = dict(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens_per_call,
                    stop=stop_words,
                    stop_token_ids=([151645, 151643] if "qwen2" in args.model_name_or_path.lower() else None),
                )
                # Use max_model_len of 8192 to reduce KV cache memory requirements
                max_model_len = getattr(args, 'max_model_len', 8192)
                child_records = run_pass1_in_subprocess(args.model_name_or_path, prompts, sampling_cfg, args.vllm_gpu_memory_utilization, max_model_len, args)

                # Re-wrap into minimal objects that mimic vLLM outputs you use later
                class _MiniOut:
                    __slots__ = ("prompt", "prompt_token_ids", "outputs", "request_id")
                    def __init__(self, rec, rid):
                        self.prompt = rec["prompt"]
                        self.prompt_token_ids = rec["prompt_token_ids"]
                        self.outputs = [type("O", (), {"text": rec["generated_text"], "token_ids": rec["generated_token_ids"]})()]
                        self.request_id = rid

                current_vllm_outputs = []
                for ridx, rec in enumerate(child_records):
                    current_vllm_outputs.append(_MiniOut(rec, ridx))

                outputs = [rec["generated_text"] for rec in child_records]

                # Map (synthesized) request_ids to (sample_idx, draw_idx, epoch)
                request_map = {}
                for prompt_idx, out in enumerate(current_vllm_outputs):
                    original_idx = current_prompts[prompt_idx][0]
                    sample_idx = original_idx // args.n_sampling
                    draw_idx = original_idx % args.n_sampling
                    request_map[str(out.request_id)] = (sample_idx, draw_idx, epoch)

                if epoch == 0:
                    vllm_outputs.extend(current_vllm_outputs)
                epoch_request_maps.append(request_map)

            else:
                # In-process vLLM (original)
                sampling_params_list = []
                num_reqs = len(prompts)
                request_map = {}

                for prompt_idx in range(num_reqs):
                    sp = SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.max_tokens_per_call,
                        n=1,
                        stop=stop_words,
                        stop_token_ids=([151645, 151643] if "qwen2" in args.model_name_or_path.lower() else None),
                    )
                    sampling_params_list.append(sp)

                current_vllm_outputs = llm.generate(prompts, sampling_params_list)

                for prompt_idx, out in enumerate(current_vllm_outputs):
                    original_idx = current_prompts[prompt_idx][0]
                    sample_idx = original_idx // args.n_sampling
                    draw_idx = original_idx % args.n_sampling
                    request_map[str(out.request_id)] = (sample_idx, draw_idx, epoch)

                if epoch == 0:
                    vllm_outputs.extend(current_vllm_outputs)
                epoch_request_maps.append(request_map)

                outputs = [out.outputs[0].text for out in current_vllm_outputs]

        else:
            # Plain HF generation (not vLLM)
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            try:
                output = output.rstrip()
                query += output
                
                # Output structured information for monitoring
                print(f"MONITOR_RESPONSE: {json.dumps({'epoch': epoch, 'prompt_idx': i, 'response': output, 'full_query': query})}")
                
                # Parse CoT structure from response (using only the answer field)
                raw = output.strip()
                
                # Debug: Log the raw output for troubleshooting
                if not raw:
                    print(f"WARNING: Empty response from model for prompt {i}")
                
                # Primary heuristic: If the delimiter #### is in raw, split on it
                if '####' in raw:
                    cot_text, ans_text = raw.split('####', 1)
                    cot_text = cot_text.strip()
                    ans_text = ans_text.strip()
                else:
                    # Fallback heuristic: Otherwise, split on the last newline
                    lines = raw.strip().splitlines()
                    if len(lines) == 0:
                        # Handle empty response
                        cot_text = ""
                        ans_text = ""
                        print(f"WARNING: Empty lines after splitting response for prompt {i}")
                    else:
                        cot_text = "\n".join(lines[:-1])
                        ans_text = lines[-1]
                
                # Convert to structured data (for monitoring only)
                cot_steps = [line.strip() for line in cot_text.split('\n') if line.strip()]
                final_answer = ans_text.strip() if ans_text else ""
                # Check if using custom prompt (disable code execution for custom prompts)
                using_custom_prompt = hasattr(args, 'prompt') and args.prompt
                
            except Exception as e:
                print(f"ERROR processing output for prompt {i}: {e}")
                print(f"Raw output: {repr(output)}")
                # Set default values to continue processing
                cot_text = ""
                ans_text = ""
                cot_steps = []
                final_answer = ""
                using_custom_prompt = hasattr(args, 'prompt') and args.prompt
            
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif not using_custom_prompt and "boxed" not in output and output.endswith("```"):
                # Only extract and execute code if not using custom prompt
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts (only if not using custom prompt)
        using_custom_prompt = hasattr(args, 'prompt') and args.prompt
        if not using_custom_prompt and remain_codes:
            remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    using_custom_prompt = hasattr(args, 'prompt') and args.prompt
    if using_custom_prompt:
        # For custom prompts, just extract the text without code execution
        results = [(code, "") for code in codes]
    else:
        # For standard prompts, execute code
        results = [
            run_execute(executor, code, args.prompt_type, data_name) for code in codes
        ]
    time_use = time.time() - start_time

    # === COMPREHENSIVE MEMORY CLEANUP: Free vLLM memory before HF scoring ===
    if args.use_vllm and not getattr(args, "pass1_subprocess", False):
        print("\n" + "="*60)
        print("🧹 vLLM GENERATION COMPLETED - CLEANING UP vLLM MEMORY")
        print("="*60)
        
        # Clean up vLLM model to free memory for HF scorer
        cleanup_gpu_memory(llm, None)
        # <<< NEW: Print memory after this intermediate cleanup
        print_gpu_memory_usage("Stage 3: After Deleting vLLM and Clearing Cache")
        
        # Set llm to None to prevent further use
        llm = None
        
        print("="*60)
        print("✅ vLLM MEMORY CLEANUP COMPLETED - READY FOR HF SCORING")
        print("="*60)

    # === PASS 2: HF scoring (microbatched & OOM-safe) ===
    hf_results_per_output = None
    expert_prob_results = {}
    if args.enable_prob_tracking and args.use_vllm:
        if args.enable_prob_tracking and args.use_vllm:
            print("\n" + "="*60)
            print("🚀 Loading Hugging Face scorer model for Pass 2...")
            print("="*60)
            hf_model = None
            try:
                if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / (1024**3) >= 4.0:
                    device_map = {"": 0}
                    print("Loading HF scorer on GPU...")
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map,
                        trust_remote_code=True,
                    ).eval()
                    print_gpu_memory_usage("Stage 4: After Loading HF Scorer Model (on GPU)")
                else:
                    raise RuntimeError("Insufficient GPU memory for HF scorer")
            except Exception as e:
                print(f"WARNING: failed to load HF scorer on CUDA; falling back to CPU. Error: {e}")
                hf_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True,
                ).eval()
        if hf_model is None:
            print("WARNING: enable_prob_tracking requested but HF scorer unavailable; skipping prob tracking.")
        else:
            # Tune this up/down based on GPU VRAM. Safe defaults:
            SCORING_MAX_TOKENS = 2048      # e.g., 2k (try 4096/8192 if you have headroom)
            MAX_LEN_PER_BATCH  = 1024      # cap long outliers (optional)
            print("Starting HF scoring with max tokens:", SCORING_MAX_TOKENS)
            hf_results_per_output, wrote_npz, expert_prob_results = run_hf_scoring_streaming(
                hf_model=hf_model,
                vllm_outputs=vllm_outputs,
                samples=samples,              # each has "gt"
                tokenizer=tokenizer,
                n_sampling=args.n_sampling,
                enable_path_vectors=bool(args.enable_path_vectors),
                path_vectors_npz=path_vectors_npz,
                max_tokens_per_batch=SCORING_MAX_TOKENS,
                max_len_per_batch=MAX_LEN_PER_BATCH,
                show_progress=True,
                progress_label=f"Pass2 Scoring [{data_name}]",
                log_monitor_json=True,
                vram_guard_gb=2.0,            # CPU-fallback a batch when free VRAM < 2 GB
            )

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        
        if using_custom_prompt:
            # For custom prompts, the code is just the model's response text
            # Extract the final answer from the response
            for j in range(len(preds)):
                # Try to extract answer from the response
                response_text = code[j]
                import re
                
                # Look for patterns like "Therefore, the final answer is: \boxed{ANSWER}"
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', response_text)
                if boxed_match:
                    preds[j] = boxed_match.group(1).strip()
                else:
                    # Look for framebox
                    framebox_match = re.search(r'\\framebox\{([^}]+)\}', response_text)
                    if framebox_match:
                        preds[j] = framebox_match.group(1).strip()
                    else:
                        # Try to extract number from patterns like "Final answer: 145 bananas" or "answer is 145"
                        # Look for patterns like "Final answer: NUMBER" or "answer is NUMBER" or "answer: NUMBER"
                        answer_patterns = [
                            r'(?:final\s+answer|answer|result)[\s:]+([0-9]+(?:\.[0-9]+)?)',
                            r'####\s*([0-9]+(?:\.[0-9]+)?)',
                            r'([0-9]+(?:\.[0-9]+)?)\s*(?:bananas?|pounds?|dollars?|pages?|minutes?|hours?|days?|etc\.?)',
                        ]
                        extracted_number = None
                        for pattern in answer_patterns:
                            match = re.search(pattern, response_text, re.IGNORECASE)
                            if match:
                                extracted_number = match.group(1)
                                break
                        
                        if extracted_number:
                            preds[j] = extracted_number
                        else:
                            # If no pattern found, try to extract the last number in the text
                            numbers = re.findall(r'([0-9]+(?:\.[0-9]+)?)', response_text)
                            if numbers:
                                preds[j] = numbers[-1]  # Use the last number found
                            else:
                                # Fallback: use the last line or whole response
                                lines = response_text.strip().split('\n')
                                if lines:
                                    preds[j] = lines[-1].strip()
                                else:
                                    preds[j] = response_text.strip()
        else:
            # For standard prompts, use the original logic
            for j in range(len(preds)):
                if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ]:
                    preds[j] = choice_answer_clean(code[j])
                elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                    # remove any non-choice char
                    preds[j] = "".join(
                        [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                    )

        # Keep the prompt in the results for Excel export
        sample.update({"code": code, "pred": preds, "report": reports})

        # === Fill probability fields from HF results (epoch_0) ===
        if args.enable_prob_tracking and args.use_vllm and hf_results_per_output is not None:
            output_index = i * args.n_sampling
            if output_index < len(hf_results_per_output) and hf_results_per_output[output_index] is not None:
                m = hf_results_per_output[output_index]
                sample["probability_log"] = {"epoch_0": m["correct_probs"]}
                sample["chosen_token_probs"] = {"epoch_0": m["chosen_probs"]}
                sample["chosen_token_ids"] = {"epoch_0": m["chosen_ids"]}
                sample["correct_token_ids"] = {"epoch_0": m["correct_ids"]}
                sample["entropies"] = {"epoch_0": m["entropies"]}
                sample["exact_match_steps"] = {"epoch_0": m["exact_matches"]}
                sample["path_vectors"] = {
                    "enabled": bool(args.enable_path_vectors),
                    "file": (path_vectors_npz if bool(args.enable_path_vectors) else None),
                    "run_id": run_id,
                }
                
                # Add expert CoT probability if available
                if output_index in expert_prob_results:
                    sample["expert_cot_probability"] = expert_prob_results[output_index]
                elif sample.get("gt_cot"):
                    # Use gt_cot as expert CoT if available but no probability calculated, set to None
                    sample["expert_cot_probability"] = None
                
                # Compute and add answer_confidence for ECE calculation
                # This is computed once during evaluation when we have the tokenizer
                if tokenizer is not None and extract_answer_confidence is not None:
                    try:
                        answer_details = extract_answer_confidence(sample, tokenizer, data_name, return_details=True)
                        if answer_details:
                            sample["answer_confidence"] = answer_details["confidence"]
                            # Also store answer token IDs for debugging/verification
                            sample["answer_token_ids"] = answer_details["answer_token_ids"]
                            sample["answer_token_indices"] = answer_details["answer_token_indices"]
                            sample["answer_text"] = answer_details["answer_text"]
                        else:
                            sample["answer_confidence"] = None
                            sample["answer_token_ids"] = None
                            sample["answer_token_indices"] = None
                            sample["answer_text"] = None
                    except Exception as e:
                        # If extraction fails, set to None (non-fatal)
                        sample["answer_confidence"] = None
                        sample["answer_token_ids"] = None
                        sample["answer_token_indices"] = None
                        sample["answer_text"] = None
                        if i < 3:  # Only print first few errors to avoid spam
                            print(f"Warning: Failed to extract answer confidence: {e}")
                else:
                    # extract_answer_confidence module not available
                    sample["answer_confidence"] = None
                    sample["answer_token_ids"] = None
                    sample["answer_token_indices"] = None
                    sample["answer_text"] = None
            else:
                print(f"Warning: HF results missing for output index {output_index}")
                sample["answer_confidence"] = None
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
        eval_method=args.eval_method,
        k=args.n_sampling,  # Use n_sampling as k
    )

    # After evaluation, if path vectors were enabled, ensure every sample points to the run's .npz file
    if args.enable_path_vectors and args.use_vllm:
        for s in all_samples:
            s["path_vectors_file"] = path_vectors_npz

    # Optionally run truncation analysis on in-memory results
    if getattr(args, 'run_truncation_analysis_after_eval', False):
        try:
            print("\n--- Main evaluation finished. Starting truncation analysis... ---")
            # Ensure tokenizer for vLLM path
            local_tokenizer = tokenizer
            if args.use_vllm and local_tokenizer is None:
                from transformers import AutoTokenizer as _AT
                local_tokenizer = _AT.from_pretrained(args.model_name_or_path, trust_remote_code=True)

            analysis_out_dir = os.path.join(args.output_dir, data_name)
            run_truncation_analysis_over_samples(
                samples=all_samples,
                llm=llm if not (isinstance(llm, str) and llm == "vllm_subprocess") else None,
                tokenizer=local_tokenizer,
                dataset_name=data_name,
                output_dir=analysis_out_dir,
                model_name_or_path=args.model_name_or_path,
            )
        except Exception as _e:
            print(f"Truncation analysis failed: {_e}")

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

        # Additionally save a separate probability-only JSONL when tracking is enabled
        if getattr(args, 'enable_prob_tracking', False) or (getattr(args, 'use_together_api', False) and getattr(args, 'together_logprobs', 0) and args.together_logprobs > 0):
            prob_only_file = out_file.replace(
                ".jsonl", f"_{args.prompt_type}_prob.jsonl"
            )
            prob_records = []
            for rec in all_samples:
                # Convert numpy arrays to lists for JSON serialization in prob file too
                model_path_vectors = {}
                raw_model_vectors = rec.get("model_path_vectors", {})
                for epoch, vectors in raw_model_vectors.items():
                    if vectors:
                        model_path_vectors[epoch] = [v.tolist() if hasattr(v, 'tolist') else v for v in vectors]
                    else:
                        model_path_vectors[epoch] = vectors
                
                gold_path_vectors = {}
                raw_gold_vectors = rec.get("gold_path_vectors", {})
                for epoch, vectors in raw_gold_vectors.items():
                    if vectors:
                        gold_path_vectors[epoch] = [v.tolist() if hasattr(v, 'tolist') else v for v in vectors]
                    else:
                        gold_path_vectors[epoch] = vectors
                
                entry = {
                    "idx": rec.get("idx"),
                    "probability_log": rec.get("probability_log", {}),
                    "exact_match_steps": rec.get("exact_match_steps", {}),
                    "chosen_token_probs": rec.get("chosen_token_probs", {}),
                    "chosen_token_ids": rec.get("chosen_token_ids", {}),
                    "correct_token_ids": rec.get("correct_token_ids", {}),
                    "entropies": rec.get("entropies", {}),
                    "model_path_vectors": model_path_vectors,
                    "gold_path_vectors": gold_path_vectors,
                    "score": rec.get("score", []),
                }
                
                # Preserve level information if it exists (for MATH dataset)
                if "level" in rec:
                    entry["level"] = rec["level"]
                
                # Preserve expert CoT probability if it exists
                if "expert_cot_probability" in rec:
                    entry["expert_cot_probability"] = rec["expert_cot_probability"]
                
                # Preserve answer_confidence and related fields if they exist (for ECE calculation)
                if "answer_confidence" in rec:
                    entry["answer_confidence"] = rec["answer_confidence"]
                if "answer_token_ids" in rec:
                    entry["answer_token_ids"] = rec["answer_token_ids"]
                if "answer_token_indices" in rec:
                    entry["answer_token_indices"] = rec["answer_token_indices"]
                if "answer_text" in rec:
                    entry["answer_text"] = rec["answer_text"]
                prob_records.append(entry)
            # Write as JSONL
            with open(prob_only_file, "w") as f:
                for entry in prob_records:
                    f.write(json.dumps(entry) + "\n")

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    # Add job configuration to metrics
    job_config = {
        "model": args.model_name_or_path,
        "dataset": data_name,
        "prompt_type": args.prompt_type,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "n_sampling": args.n_sampling,
        "max_tokens": args.max_tokens_per_call,
        "eval_method": args.eval_method,
        "k": args.n_sampling,  # k equals n_sampling
        "run_id": run_id,
    }
    # Add Together API config if used
    if getattr(args, 'use_together_api', False):
        job_config["use_together_api"] = True
        job_config["together_logprobs"] = int(getattr(args, 'together_logprobs', 0) or 0)
    
    # Add custom prompt if provided
    if hasattr(args, 'prompt') and args.prompt:
        job_config["prompt"] = args.prompt
    
    # Add job_id if provided
    if hasattr(args, 'job_id') and args.job_id:
        job_config["job_id"] = args.job_id
    
    result_json["job_configuration"] = job_config

    # Create metrics filename
    metrics_file = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(result_json, f, indent=4)
    
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    # <<< NEW: Print initial memory state
    print_gpu_memory_usage("Stage 1: Initial State (before any models are loaded)")

    setup(args)