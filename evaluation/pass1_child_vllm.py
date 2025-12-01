#!/usr/bin/env python3
"""
Pass-1 vLLM subprocess worker for math_eval.py
This script is called as a subprocess to run vLLM inference,
then exit to free VRAM before Pass-2 HF scoring.
"""

import sys
import json
import os
import tempfile
import shutil

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    import transformers
except ImportError:
    print("ERROR: vllm not installed. Please install with: pip install vllm")
    sys.exit(1)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>")
        sys.exit(1)
    
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    # Read input payload
    with open(in_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    
    model_name = payload['model']
    prompts = payload['prompts']
    temperature = payload.get('temperature', 0.0)
    top_p = payload.get('top_p', 1.0)
    top_k = payload.get('top_k', 0)
    max_tokens = payload.get('max_tokens', 2048)
    stop = payload.get('stop', [])
    stop_token_ids = payload.get('stop_token_ids', None)
    gpu_memory_utilization = payload.get('gpu_memory_utilization', 0.9)
    max_model_len = payload.get('max_model_len', 4096)
    max_num_seqs = payload.get('max_num_seqs', 16)  # Lower default to reduce OOM risk
    
    print(f"🚀 Loading model: {model_name}")
    print(f"📊 GPU memory utilization: {gpu_memory_utilization}")
    print(f"📏 Max model length: {max_model_len}")
    print(f"📦 Max num sequences: {max_num_seqs}")
    print(f"🎲 Temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
    print(f"📝 Number of prompts: {len(prompts)}")
    
    # Detect if this is a Mistral model that needs enforce_eager
    # Mistral models sometimes have missing head_dim in config which breaks FlashAttention
    is_mistral = "mistral" in model_name.lower() or "mathstral" in model_name.lower()
    
    # Fix Mistral models with missing head_dim in config
    # We'll patch the config file in the cache after it's downloaded
    if is_mistral:
        print("⚠️ Detected Mistral model - checking config for missing head_dim...")
        try:
            # Load config first to ensure it's downloaded and check for missing head_dim
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Check if head_dim is missing or None
            needs_fix = False
            computed_head_dim = None
            if not hasattr(config, 'head_dim') or config.head_dim is None:
                # Compute head_dim from hidden_size and num_attention_heads
                if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                    computed_head_dim = config.hidden_size // config.num_attention_heads
                    print(f"   Found missing head_dim - computing: {config.hidden_size} / {config.num_attention_heads} = {computed_head_dim}")
                    needs_fix = True
                elif hasattr(config, 'dim') and hasattr(config, 'n_heads'):  # Alternative naming
                    computed_head_dim = config.dim // config.n_heads
                    print(f"   Found missing head_dim - computing: {config.dim} / {config.n_heads} = {computed_head_dim}")
                    needs_fix = True
            
            if needs_fix and computed_head_dim is not None:
                # Patch the config.json file in the cache
                # Find the cached config file
                try:
                    from huggingface_hub import snapshot_download, hf_hub_download
                    from pathlib import Path
                    # json is already imported at the top of the file
                    
                    # Get the cache directory where the config is stored
                    cache_dir = transformers.utils.hub.default_cache_path
                    
                    # Try to find the config file in the cache
                    # The path structure is: cache_dir/models--model_name/snapshots/hash/config.json
                    repo_id = model_name.replace('/', '--')
                    model_cache_base = Path(cache_dir) / "models--" / repo_id
                    
                    if model_cache_base.exists():
                        # Find the snapshot directory
                        snapshot_dirs = list(model_cache_base.glob("snapshots/*"))
                        if snapshot_dirs:
                            config_path = snapshot_dirs[0] / "config.json"
                            if config_path.exists():
                                # Read, patch, and write the config
                                with open(config_path, 'r') as f:
                                    config_dict = json.load(f)
                                
                                # Add head_dim if missing
                                if 'head_dim' not in config_dict or config_dict.get('head_dim') is None:
                                    config_dict['head_dim'] = computed_head_dim
                                    print(f"   Patching config.json at: {config_path}")
                                    with open(config_path, 'w') as f:
                                        json.dump(config_dict, f, indent=2)
                                    print(f"   ✅ Successfully patched head_dim = {computed_head_dim} in config.json")
                                else:
                                    print(f"   Config already has head_dim = {config_dict.get('head_dim')}")
                            else:
                                print(f"   Warning: config.json not found at {config_path}")
                        else:
                            print(f"   Warning: No snapshot directory found in {model_cache_base}")
                    else:
                        print(f"   Warning: Model cache directory not found at {model_cache_base}")
                except Exception as patch_error:
                    print(f"   Warning: Could not patch config file in cache ({patch_error})")
                    print(f"   vLLM may still fail, but we'll try anyway...")
        except Exception as e:
            print(f"   Warning: Could not check/patch config ({e}), proceeding anyway...")
    
    # Initialize vLLM with conditional enforce_eager for Mistral models
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,  # Limit batch size to reduce memory usage
        "tensor_parallel_size": 1,
    }
    
    if is_mistral:
        print("⚠️ Using eager attention for Mistral model (slower but more compatible)")
        llm_kwargs["enforce_eager"] = True
    else:
        print("✅ Using FlashAttention for optimal performance")
    
    try:
        llm = LLM(**llm_kwargs)
    except (TypeError, ValueError) as e:
        error_str = str(e)
        if "unsupported operand type" in error_str or "NoneType" in error_str or "head_dim" in error_str.lower():
            error_msg = (
                f"\n❌ Error loading vLLM model '{model_name}':\n"
                f"   The model configuration appears to be incomplete or incompatible with vLLM.\n"
                f"   \n"
                f"   Error: {error_str}\n"
                f"   \n"
                f"   This typically occurs when:\n"
                f"   1. The model config is missing required fields (e.g., head_dim, hidden_size)\n"
                f"   2. The model architecture is not fully compatible with the installed vLLM version\n"
                f"   3. The model format is corrupted or incomplete\n"
                f"   \n"
                f"   Possible solutions:\n"
                f"   - Try a different Mistral model variant (e.g., Mistral-7B-Instruct-v0.2)\n"
                f"   - Use a model that is known to work with vLLM\n"
                f"   - Check if the model repository is complete on HuggingFace\n"
                f"   - Update vLLM to the latest version: pip install --upgrade vllm\n"
                f"   - Try using enforce_eager mode (already attempted for Mistral models)\n"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        raise
    except Exception as e:
        error_str = str(e)
        is_oom = "out of memory" in error_str.lower() or "CUDA error: out of memory" in error_str
        
        if is_oom:
            error_msg = (
                f"\n❌ CUDA out of memory when loading vLLM model '{model_name}':\n"
                f"   {error_str}\n"
                f"   \n"
                f"   Current settings:\n"
                f"   - gpu_memory_utilization: {gpu_memory_utilization}\n"
                f"   - max_model_len: {max_model_len}\n"
                f"   - max_num_seqs: {max_num_seqs}\n"
                f"   \n"
                f"   Suggestions to fix:\n"
                f"   1. Lower gpu_memory_utilization (try 0.7, 0.6, or 0.5)\n"
                f"   2. Lower max_model_len (try {max_model_len // 2} or {max_model_len // 4})\n"
                f"   3. Lower max_num_seqs (try {max_num_seqs // 2} or {max_num_seqs // 4})\n"
                f"   4. Free up GPU memory from other processes\n"
                f"   5. Use a smaller model or reduce batch size\n"
                f"   6. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation\n"
            )
        else:
            error_msg = (
                f"\n❌ Error loading vLLM model '{model_name}':\n"
                f"   {error_str}\n"
                f"   \n"
                f"   Please verify:\n"
                f"   1. The model exists on HuggingFace: https://huggingface.co/{model_name}\n"
                f"   2. The model is compatible with vLLM\n"
                f"   3. You have sufficient GPU memory\n"
                f"   4. You have access to the model (if it's private)\n"
                f"   5. Try updating vLLM: pip install --upgrade vllm\n"
            )
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        max_tokens=max_tokens,
        stop=stop if stop else None,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )
    
    # Run inference
    print("🔄 Running inference...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Format results
    results = []
    for i, output in enumerate(outputs):
        result = {
            "prompt": output.prompt,
            "prompt_token_ids": output.prompt_token_ids,
            "generated_text": output.outputs[0].text if output.outputs else "",
            "generated_token_ids": output.outputs[0].token_ids if output.outputs else [],
        }
        results.append(result)
    
    # Write output
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Completed inference for {len(results)} prompts")
    print(f"💾 Results written to: {out_path}")
    
    # Explicitly free vLLM model and GPU memory before exiting
    print("🧹 Cleaning up vLLM model and GPU memory before exit...")
    try:
        # Delete the LLM object to free model memory
        del llm
        llm = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear PyTorch CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Verify memory is freed
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = round(free_bytes / (1024**3), 2)
            print(f"✅ GPU memory freed: {free_gb}GB free")
    except Exception as e:
        print(f"⚠️ Warning during cleanup: {e}")
    
    # Explicitly exit to free VRAM
    sys.exit(0)


if __name__ == "__main__":
    main()
