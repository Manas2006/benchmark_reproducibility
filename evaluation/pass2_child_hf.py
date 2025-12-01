# pass2_child_hf.py
import json, sys
import torch
import gc
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Import necessary functions from math_eval (we'll need to add them here or import)
# For now, we'll include the helper functions directly
from transformers import AutoTokenizer, AutoModelForCausalLM

def _build_len_metadata(vllm_outputs_records) -> List[Dict[str, int]]:
    """Collect prompt/gen lengths for each vLLM output entry."""
    meta = []
    for i_out, rec in enumerate(vllm_outputs_records):
        p_ids = rec.get("prompt_token_ids", [])
        g_ids = rec.get("generated_token_ids", [])
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
    """Greedy packing by token budget."""
    meta = sorted(meta, key=lambda x: x["L"], reverse=True)
    batches: List[List[Dict[str, int]]] = []
    cur: List[Dict[str, int]] = []
    cur_max = 0

    for m in meta:
        L = m["L"]
        if max_len_per_batch is not None and L > max_len_per_batch:
            if cur:
                batches.append(cur)
                cur, cur_max = [], 0
            batches.append([m])
            continue

        prospective_bs = len(cur) + 1
        prospective_max = max(cur_max, L)
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
    vllm_outputs_records: List[Dict[str, Any]],
    tokenizer,
    device: torch.device,
):
    """Right-pad each row to the microbatch max length."""
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
        rec = vllm_outputs_records[m["i_out"]]
        p_ids = list(rec.get("prompt_token_ids", []))
        g_ids = list(rec.get("generated_token_ids", []))
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

def _pointer_metrics_for_sample(
    probs_i: torch.Tensor,
    gen_ids: List[int],
    gold_answer_text: str,
    tokenizer,
) -> Dict[str, Any]:
    """Compute chosen probs, correct probs, entropies."""
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
    chosen_probs_t = probs_i.gather(1, gen_ids_t.unsqueeze(1)).squeeze(1)

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
                    ptr = 0
            else:
                ptr = 0

    return {
        "chosen_ids": [int(x) for x in gen_ids],
        "chosen_probs": [float(x) for x in chosen_probs_t.tolist()],
        "correct_ids": correct_ids,
        "correct_probs": correct_probs,
        "entropies": entropies,
        "exact_matches": int(exact_matches),
    }

def main(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_path = cfg["model_path"]
    vllm_outputs_records = cfg["vllm_outputs"]
    samples = cfg["samples"]
    n_sampling = cfg["n_sampling"]
    enable_path_vectors = cfg.get("enable_path_vectors", False)
    path_vectors_npz = cfg.get("path_vectors_npz", None)
    max_tokens_per_batch = cfg.get("max_tokens_per_batch", 2048)
    max_len_per_batch = cfg.get("max_len_per_batch", 1024)
    
    print(f"🚀 Loading HF scorer model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model on GPU if available, otherwise CPU
    hf_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")
    
    try:
        if use_cuda and torch.cuda.mem_get_info()[0] / (1024**3) >= 4.0:
            print("Loading HF scorer on GPU...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
            ).eval()
        else:
            raise RuntimeError("Insufficient GPU memory or CUDA unavailable")
    except Exception as e:
        print(f"WARNING: failed to load HF scorer on CUDA; falling back to CPU. Error: {e}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        ).eval()
        device = torch.device("cpu")
        use_cuda = False
    
    if hf_model is None:
        raise RuntimeError("Failed to load HF scorer model")
    
    # Build batches
    meta = _build_len_metadata(vllm_outputs_records)
    batches = _form_token_budget_batches(
        meta, max_tokens_per_batch=max_tokens_per_batch, max_len_per_batch=max_len_per_batch
    )
    
    hf_results_per_output: List[Optional[Dict[str, Any]]] = [None] * len(vllm_outputs_records)
    expert_prob_results: Dict[int, float] = {}
    pathvec_store: Dict[str, Any] = {}
    
    total_samples = len(vllm_outputs_records)
    print(f"📊 Processing {total_samples} samples in {len(batches)} batches...")
    
    # Process batches
    with torch.no_grad():
        for b_idx, batch_meta in enumerate(batches):
            input_ids, attn_mask, position_ids, prompt_lens, gen_lens, gen_ids_list = \
                _pack_microbatch(batch_meta, vllm_outputs_records, tokenizer, device)
            
            # Move model to CPU temporarily if needed (for VRAM guard)
            moved_model_to_cpu = False
            if use_cuda:
                free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                if free_gb < 2.0:  # VRAM guard
                    hf_model.to("cpu")
                    input_ids = input_ids.to("cpu")
                    attn_mask = attn_mask.to("cpu")
                    position_ids = position_ids.to("cpu")
                    device = torch.device("cpu")
                    moved_model_to_cpu = True
                    print(f"[Batch {b_idx}] Moved to CPU due to low VRAM")
            
            logits_mb = hf_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits
            
            B = input_ids.size(0)
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
                l_rows = logits_mb[row, start_idx:end_idx, :]
                probs_i = torch.softmax(l_rows.float(), dim=-1)
                
                sample_index = i_out // n_sampling
                gold_answer_text = samples[sample_index].get("gt", "")
                
                metrics = _pointer_metrics_for_sample(probs_i, gen_ids_list[row], gold_answer_text, tokenizer)
                hf_results_per_output[i_out] = metrics
                
                if enable_path_vectors:
                    pathvec_store[f"s{i_out}"] = probs_i.detach().to("cpu").half().numpy()
            
            # Free batch tensors
            del logits_mb
            if moved_model_to_cpu:
                hf_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            if (b_idx + 1) % 10 == 0:
                print(f"  Processed {b_idx + 1}/{len(batches)} batches...")
    
    # Save path vectors if enabled
    wrote_npz = False
    if enable_path_vectors and len(pathvec_store) > 0 and path_vectors_npz:
        try:
            os.makedirs(os.path.dirname(path_vectors_npz), exist_ok=True)
            np.savez_compressed(path_vectors_npz, **pathvec_store)
            wrote_npz = True
            print(f"✅ Saved path vectors to {path_vectors_npz}")
        except Exception as e:
            print(f"⚠️ Warning: failed to write path vectors npz: {e}")
    
    # 🧹 COMPREHENSIVE CLEANUP: Ensure HF model completely releases GPU memory
    print("🧹 Cleaning up HF scorer subprocess...")
    try:
        # Move model to CPU first if on GPU
        if use_cuda and next(hf_model.parameters()).device.type == "cuda":
            hf_model.to("cpu")
            print("[Cleanup] Moved HF model to CPU")
        
        # Clear gradients and cached activations
        for param in hf_model.parameters():
            if param.grad is not None:
                param.grad = None
        hf_model.zero_grad(set_to_none=True)
        
        # Delete model
        del hf_model
        hf_model = None
        
        # Delete tokenizer
        del tokenizer
        tokenizer = None
        
        # Force garbage collection
        for _ in range(5):
            gc.collect()
        
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            for _ in range(5):
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
            # Check freed memory
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = round(free_bytes / (1024**3), 2)
            print(f"✅ HF scorer subprocess cleanup complete: {free_gb}GB free")
    except Exception as e:
        print(f"⚠️ Warning during cleanup: {e}")
        # Still try basic cleanup
        try:
            if torch.cuda.is_available():
                for _ in range(3):
                    torch.cuda.empty_cache()
            for _ in range(3):
                gc.collect()
        except:
            pass
    
    # Prepare output
    output = {
        "hf_results_per_output": hf_results_per_output,
        "expert_prob_results": expert_prob_results,
        "wrote_npz": wrote_npz,
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f)
    
    print("✅ HF scoring subprocess completed successfully")

if __name__ == "__main__":
    in_path, out_path = sys.argv[1], sys.argv[2]
    main(in_path, out_path)

