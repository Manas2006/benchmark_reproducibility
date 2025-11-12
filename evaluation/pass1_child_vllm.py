#!/usr/bin/env python3
"""
Pass-1 vLLM subprocess worker for math_eval.py
This script is called as a subprocess to run vLLM inference,
then exit to free VRAM before Pass-2 HF scoring.
"""

import sys
import json
import os

try:
    from vllm import LLM, SamplingParams
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
    
    print(f"🚀 Loading model: {model_name}")
    print(f"📊 GPU memory utilization: {gpu_memory_utilization}")
    print(f"📏 Max model length: {max_model_len}")
    print(f"🎲 Temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
    print(f"📝 Number of prompts: {len(prompts)}")
    
    # Detect if this is a Mistral model that needs enforce_eager
    # Mistral models sometimes have missing head_dim in config which breaks FlashAttention
    is_mistral = "mistral" in model_name.lower() or "mathstral" in model_name.lower()
    
    # Initialize vLLM with conditional enforce_eager for Mistral models
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": 1,
    }
    
    if is_mistral:
        print("⚠️ Detected Mistral model - using eager attention (slower but more compatible)")
        llm_kwargs["enforce_eager"] = True
    else:
        print("✅ Using FlashAttention for optimal performance")
    
    llm = LLM(**llm_kwargs)
    
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
    
    # Explicitly exit to free VRAM
    sys.exit(0)


if __name__ == "__main__":
    main()
