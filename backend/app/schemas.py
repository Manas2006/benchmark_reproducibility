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