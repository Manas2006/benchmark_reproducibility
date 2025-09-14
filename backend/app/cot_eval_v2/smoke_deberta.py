"""
Smoke test for DeBERTa MNLI integration.

This script tests the real DeBERTa MNLI model on GPU (or CPU fallback)
to verify that coherence and factuality flagging works correctly.
"""

from cot_eval_v2.evaluator import PillarsEvaluator


def main():
    """Run smoke test with real DeBERTa MNLI model."""
    print("=== DeBERTa MNLI Smoke Test ===")
    print()
    
    # Test 1: Without auto-loading (explicit pipeline)
    print("1. Testing with explicit pipeline loading...")
    try:
        from transformers import pipeline
        import torch
        import os
        
        # Detect SLURM GPU allocation
        slurm_gpu_id = os.environ.get('SLURM_LOCALID')
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        
        if slurm_gpu_id is not None:
            device = int(slurm_gpu_id)
            print(f"Using SLURM GPU {device} for DeBERTa MNLI...")
        elif cuda_devices is not None and cuda_devices.strip():
            device = 0  # CUDA_VISIBLE_DEVICES maps to device 0 in PyTorch
            print(f"Using SLURM CUDA_VISIBLE_DEVICES GPU {device} for DeBERTa MNLI...")
        elif torch.cuda.is_available():
            device = 0
            print(f"Using CUDA GPU {device} for DeBERTa MNLI...")
        else:
            device = -1
            print(f"Using CPU for DeBERTa MNLI (no GPU available)...")
        
        print(f"Loading DeBERTa MNLI on device={device}...")
        nli = pipeline(
            "text-classification", 
            model="microsoft/deberta-base-mnli", 
            device=device
        )
        
        ev = PillarsEvaluator(nli_pipe=nli, use_nli=False)
        
        # Test case with factuality issue
        problem = "The capital of France is Paris."
        cot = (
            "Step 1. France is a European country.\n"
            "Step 2. The capital of France is London.\n"
            "#### London"
        )
        
        print(f"Problem: {problem}")
        print(f"CoT:\n{cot}")
        print()
        
        flags, evidence = ev.analyze(problem, cot, gold="Paris")
        
        print("Results:")
        print(f"Flags count: {len(flags)}")
        print(f"Flag summary: {flags.summarize_for_prompt()}")
        print()
        print("Evidence:")
        for key, value in evidence.items():
            if key in ["coh_contra_cnt", "avg_coh_margin", "fact_entail_rate", 
                      "fact_contra_cnt", "avg_fact_margin"]:
                print(f"  {key}: {value}")
        print()
        
        # Check if factuality flag was raised
        fact_flags = flags.get_flags_by_pillar("factuality")
        if fact_flags:
            print("✅ Factuality flag detected (expected)")
        else:
            print("❌ No factuality flag (unexpected)")
        
    except Exception as e:
        print(f"❌ Failed to load DeBERTa MNLI: {e}")
        print("This is expected if transformers/torch are not installed or model is not available")
        print("Continuing with graceful fallback test...")
        print()
    
    # Test 2: With auto-loading
    print("2. Testing with auto-loading...")
    try:
        ev_auto = PillarsEvaluator(use_nli=True)
        
        # Test case with coherence issue
        problem = "The sky is blue."
        cot = (
            "Step 1. The sky is blue.\n"
            "Step 2. The sky is actually red.\n"
            "#### red"
        )
        
        print(f"Problem: {problem}")
        print(f"CoT:\n{cot}")
        print()
        
        flags, evidence = ev_auto.analyze(problem, cot, gold="blue")
        
        print("Results:")
        print(f"Flags count: {len(flags)}")
        print(f"Flag summary: {flags.summarize_for_prompt()}")
        print()
        print("Evidence:")
        for key, value in evidence.items():
            if key in ["coh_contra_cnt", "avg_coh_margin", "fact_entail_rate", 
                      "fact_contra_cnt", "avg_fact_margin"]:
                print(f"  {key}: {value}")
        print()
        
        # Check if coherence flag was raised
        coh_flags = flags.get_flags_by_pillar("coherence")
        if coh_flags:
            print("✅ Coherence flag detected (expected)")
        else:
            print("❌ No coherence flag (unexpected)")
            
    except Exception as e:
        print(f"❌ Auto-loading failed: {e}")
    
    # Test 3: Without NLI (graceful fallback)
    print("3. Testing graceful fallback without NLI...")
    ev_no_nli = PillarsEvaluator(nli_pipe=None, use_nli=False)
    
    problem = "Test problem"
    cot = "Step 1. This is a test.\n#### answer"
    
    flags, evidence = ev_no_nli.analyze(problem, cot, gold="answer")
    
    print("Results:")
    print(f"Flags count: {len(flags)}")
    print(f"Flag summary: {flags.summarize_for_prompt()}")
    print()
    print("Evidence (NLI metrics should be zero):")
    for key, value in evidence.items():
        if key in ["coh_contra_cnt", "avg_coh_margin", "fact_entail_rate", 
                  "fact_contra_cnt", "avg_fact_margin"]:
            print(f"  {key}: {value}")
    
    # Check that NLI metrics are zero
    nli_metrics_zero = (
        evidence["coh_contra_cnt"] == 0 and
        evidence["avg_coh_margin"] == 0.0 and
        evidence["fact_entail_rate"] == 0.0 and
        evidence["fact_contra_cnt"] == 0 and
        evidence["avg_fact_margin"] == 0.0
    )
    
    if nli_metrics_zero:
        print("✅ Graceful fallback working correctly")
    else:
        print("❌ Graceful fallback not working correctly")
    
    print()
    print("=== Smoke test completed ===")


if __name__ == "__main__":
    main()
