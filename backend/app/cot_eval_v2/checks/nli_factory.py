"""
NLI Factory for DeBERTa MNLI model management with GPU/CPU fallback.

Handles device resolution respecting SLURM environment variables and provides
graceful fallback to CPU when GPU is unavailable.
"""

import os
import torch
from typing import Optional, Union
from transformers import pipeline, Pipeline


def resolve_device() -> Union[int, str]:
    """
    Resolve the best available device for NLI inference.
    
    Priority:
    1. CUDA if available and SLURM/CUDA_VISIBLE_DEVICES is set
    2. CUDA device 0 if available
    3. CPU fallback
    
    Returns:
        Device ID (int for CUDA, -1 for CPU) or device string
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU for NLI")
        return -1
    
    # Check SLURM GPU allocation
    slurm_gpus = os.getenv("SLURM_JOB_GPUS") or os.getenv("SLURM_STEP_GPUS")
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    
    if slurm_gpus:
        print(f"🎯 SLURM GPU allocation detected: {slurm_gpus}")
        # In SLURM, use device 0 (SLURM usually sets CUDA_VISIBLE_DEVICES correctly)
        return 0
    elif cuda_visible:
        print(f"🎯 CUDA_VISIBLE_DEVICES: {cuda_visible}")
        # Use the first visible device
        visible_devices = cuda_visible.split(',')
        if visible_devices and visible_devices[0] != '':
            return int(visible_devices[0])
        return 0
    else:
        # No SLURM/CUDA env vars, use device 0 if available
        device_count = torch.cuda.device_count()
        if device_count > 0:
            print(f"🎯 Using CUDA device 0 (total devices: {device_count})")
            return 0
        else:
            print("⚠️ No CUDA devices found, using CPU")
            return -1


def create_nli_pipeline(
    model_name: str = "microsoft/deberta-base-mnli",
    device: Optional[Union[int, str]] = None
) -> Pipeline:
    """
    Create NLI pipeline with automatic device resolution.
    
    Args:
        model_name: HuggingFace model identifier
        device: Optional device override (auto-detected if None)
        
    Returns:
        Configured NLI pipeline
        
    Raises:
        RuntimeError: If model loading fails
    """
    if device is None:
        device = resolve_device()
    
    try:
        print(f"🔄 Loading NLI model: {model_name}")
        print(f"🔄 Target device: {device}")
        
        # Create pipeline with explicit device mapping
        nli_pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            return_all_scores=True
        )
        
        # Verify the model loaded correctly
        test_result = nli_pipe([("This is a test", "This is also a test")])
        if test_result:
            print(f"✅ NLI model loaded successfully on device {device}")
            return nli_pipe
        else:
            raise RuntimeError("NLI model test inference failed")
            
    except Exception as e:
        print(f"❌ Failed to load NLI model {model_name}: {e}")
        if device != -1:
            print("🔄 Retrying with CPU fallback...")
            return create_nli_pipeline(model_name, device=-1)
        else:
            raise RuntimeError(f"NLI model loading failed on both GPU and CPU: {e}")


def get_device_info() -> dict:
    """Get current device information for debugging."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "slurm_job_gpus": os.getenv("SLURM_JOB_GPUS"),
        "slurm_step_gpus": os.getenv("SLURM_STEP_GPUS"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "current_device": resolve_device()
    }
    return info


if __name__ == "__main__":
    # Test the factory
    print("🔍 Device Information:")
    for key, value in get_device_info().items():
        print(f"  {key}: {value}")
    
    print("\n🧪 Testing NLI pipeline creation...")
    try:
        pipe = create_nli_pipeline()
        print("✅ NLI factory test passed")
    except Exception as e:
        print(f"❌ NLI factory test failed: {e}")
