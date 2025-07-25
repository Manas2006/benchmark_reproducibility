from pydantic import BaseModel, Field, conint, confloat
from typing import Optional, List, Dict, Any
from enum import Enum
from .enums import Backend

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"
    READY_FOR_DOWNLOAD = "READY_FOR_DOWNLOAD"

class PathConfig(BaseModel):
    """Configuration for all system paths"""
    # Base directories
    workspace_dir: str = Field(description="Base workspace directory")
    evaluation_dir: str = Field(description="Path to evaluation directory containing math_eval.py")
    backend_dir: str = Field(description="Path to backend directory")
    
    # Python and environment
    python_path: str = Field(description="Path to Python executable")
    conda_env_path: str = Field(description="Path to conda environment")
    
    # Output and logs
    output_dir: str = Field(description="Directory for evaluation results")
    logs_dir: str = Field(description="Directory for log files")
    scripts_dir: str = Field(description="Directory for SLURM scripts")
    job_db_path: str = Field(description="Path to job database file")
    
    # SLURM configuration
    slurm_partition: str = Field(default="gpu-a100-dev", description="SLURM partition name")
    slurm_account: str = Field(default="CCR24036", description="SLURM account name")
    slurm_wall_time: str = Field(default="1:00:00", description="SLURM wall time limit")

class EvalRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-Math-1.5B"
    dataset: str = "gsm8k,math"
    prompt: Optional[str] = None
    temperature: confloat(ge=0, le=5) = 0.0
    top_p: confloat(gt=0, le=1) = 1.0
    top_k: conint(ge=0) = 0  # 0 = disabled
    n_sampling: conint(ge=1, le=32) = 1
    seed: conint(ge=0) = 42
    eval_method: str = "pass@k"  # enum later
    k: conint(ge=1, le=32) = 1
    backend: Backend = Backend.local
    path_config: Optional[PathConfig] = None  # Use default if not provided

class EvalResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    run_path: Optional[str] = None
    sbatch_path: Optional[str] = None

class EvalConfig(BaseModel):
    model: str
    dataset: str
    prompt_type: str = "tool-integrated"
    temperature: float
    top_p: float
    n_sampling: int
    seed: int
    backend: Backend

class PathConfigResponse(BaseModel):
    current_config: PathConfig
    default_config: PathConfig 