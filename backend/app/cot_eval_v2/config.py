"""
Configuration management for the four-pillar CoT evaluation system.

Provides environment variable defaults and configuration validation.
"""

import os
from typing import Optional


class PillarsConfig:
    """Configuration for the four-pillar evaluation system."""
    
    # Evaluation mode
    EVAL_MODE: str = os.getenv("EVAL_MODE", "PILLARS_ONLY")
    
    # Judge configuration
    JUDGE_GATING: str = os.getenv("JUDGE_GATING", "SMART")  # SMART | ALWAYS | NEVER
    JUDGE_BUDGET: int = int(os.getenv("JUDGE_BUDGET", "999999"))
    JUDGE_DIAGNOSTIC: bool = os.getenv("JUDGE_DIAGNOSTIC", "0") == "1"
    JUDGE_ENABLED: bool = os.getenv("JUDGE_ENABLED", "1") == "1"
    
    # NLI model configuration
    NLI_MODEL: str = os.getenv("NLI_MODEL", "microsoft/deberta-base-mnli")
    NLI_ENABLED: bool = os.getenv("NLI_ENABLED", "1") == "1"
    
    # Safety rollback
    PILLARS_ROLLBACK: bool = os.getenv("PILLARS_ROLLBACK", "0") == "1"
    
    # SLURM configuration
    SLURM_THREADING: bool = os.getenv("SLURM_THREADING", "1") == "1"
    
    @classmethod
    def get_judge_mode(cls) -> str:
        """Get the effective judge mode based on configuration."""
        if not cls.JUDGE_ENABLED:
            return "NEVER"
        return cls.JUDGE_GATING
    
    @classmethod
    def get_nli_model_name(cls) -> Optional[str]:
        """Get the NLI model name if NLI is enabled."""
        if not cls.NLI_ENABLED:
            return None
        return cls.NLI_MODEL
    
    @classmethod
    def is_legacy_enabled(cls) -> bool:
        """Check if legacy CQS evaluation is enabled."""
        return cls.EVAL_MODE == "LEGACY_ONLY"
    
    @classmethod
    def should_use_rollback(cls) -> bool:
        """Check if rollback mode is enabled (rules+DeBERTa only, no judge)."""
        return cls.PILLARS_ROLLBACK
    
    @classmethod
    def get_slurm_threading_config(cls) -> dict:
        """Get SLURM threading configuration."""
        if cls.SLURM_THREADING:
            return {
                "OMP_NUM_THREADS": "1",
                "MKL_THREADING_LAYER": "GNU"
            }
        return {}
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        valid_modes = ["PILLARS_ONLY", "LEGACY_ONLY", "MIXED"]
        valid_gating = ["SMART", "ALWAYS", "NEVER"]
        
        if cls.EVAL_MODE not in valid_modes:
            print(f"❌ Invalid EVAL_MODE: {cls.EVAL_MODE}. Must be one of {valid_modes}")
            return False
        
        if cls.JUDGE_GATING not in valid_gating:
            print(f"❌ Invalid JUDGE_GATING: {cls.JUDGE_GATING}. Must be one of {valid_gating}")
            return False
        
        if cls.JUDGE_BUDGET < 0:
            print(f"❌ Invalid JUDGE_BUDGET: {cls.JUDGE_BUDGET}. Must be non-negative")
            return False
        
        print(f"✅ Configuration validated successfully")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration for debugging."""
        print("🔧 Pillars Configuration:")
        print(f"  EVAL_MODE: {cls.EVAL_MODE}")
        print(f"  JUDGE_GATING: {cls.JUDGE_GATING}")
        print(f"  JUDGE_BUDGET: {cls.JUDGE_BUDGET}")
        print(f"  JUDGE_DIAGNOSTIC: {cls.JUDGE_DIAGNOSTIC}")
        print(f"  JUDGE_ENABLED: {cls.JUDGE_ENABLED}")
        print(f"  NLI_MODEL: {cls.NLI_MODEL}")
        print(f"  NLI_ENABLED: {cls.NLI_ENABLED}")
        print(f"  PILLARS_ROLLBACK: {cls.PILLARS_ROLLBACK}")
        print(f"  SLURM_THREADING: {cls.SLURM_THREADING}")


# Global configuration instance
config = PillarsConfig()


def setup_slurm_environment():
    """Set up SLURM environment variables for optimal performance."""
    threading_config = config.get_slurm_threading_config()
    
    for key, value in threading_config.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"🔧 Set {key}={value}")


if __name__ == "__main__":
    # Test configuration
    config.print_config()
    config.validate()
    setup_slurm_environment()
