from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunRequest(BaseModel):
    raw_sbatch_directives: str = Field(
        description="Raw SBATCH directives to include in the script"
    )
    local_dir: str = Field(description="Local directory to change to before running")
    output_dir: str = Field(description="Output directory for results")
    environment_settings: str = Field(
        description="Environment setup commands (cd, conda activate, etc.)"
    )

    # Lists for hyperparameters
    models: List[str] = Field(description="List of models to experiment with")
    datasets: List[str] = Field(
        description="List of datasets to test on (may include split name separated by |)"
    )
    top_ps: List[float] = Field(description="List of top_p values")
    top_ks: List[int] = Field(description="List of top_k values")
    temps: List[float] = Field(description="List of temperature values")
    max_lengths: List[int] = Field(description="List of max model lengths")
    max_new_tokens: List[int] = Field(description="List of max new tokens")
    seeds: List[int] = Field(description="List of random seeds")

    # Generation config
    prompt: str = Field(default="", description="Prompt for generation")

    # Evaluation config
    evaluation_metric: str = Field(
        default="pass@k", description="Evaluation metric: pass@k or maj@k"
    )
    at_k_value: int = Field(default=1, description="@k value for evaluation")
    evaluation_prompt: str = Field(default="", description="Prompt for evaluation")
    evaluation_tool: str = Field(
        default="rule-based", description="Evaluation tool: rule-based or llm"
    )
    extraction_method: str = Field(
        default="predefined", description="Extraction method: predefined or custom"
    )
    predefined_extractor: str = Field(
        default="boxed_answer", description="Predefined extraction function"
    )
    judge_model_type: str = Field(
        default="api", description="Judge model type: api or local"
    )
    judge_model: Optional[str] = Field(
        default=None, description="Judge model name when using LLM evaluation"
    )
    judge_api_key: Optional[str] = Field(
        default=None, description="API key for judge model"
    )
    local_llm_path: Optional[str] = Field(
        default=None, description="Path to local LLM for judging"
    )
    custom_extractor_code: Optional[str] = Field(
        default=None, description="Custom Python code for rule-based extraction"
    )

    # Optional existing results
    existing_result_path: Optional[str] = Field(
        default=None, description="Path to existing results to evaluate"
    )

    run_id: Optional[UUID] = Field(default=None, description="Unique run identifier")


class JobInfo(BaseModel):
    run_id: UUID
    slurm_job_id: Optional[str] = None
    status: JobStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    params: RunRequest
    log_buffer: List[str] = Field(default_factory=list)
    process_id: Optional[int] = None
    script_path: Optional[str] = None


class JobSummary(BaseModel):
    run_id: str
    slurm_job_id: Optional[str] = None
    status: JobStatus
    started_at: datetime
    duration: Optional[float] = None
    job_name: str
    total_experiments: int


class LogResponse(BaseModel):
    run_id: str
    logs: str
    status: JobStatus


class ResultEntry(BaseModel):
    run_id: str
    model: str
    dataset: str
    split: Optional[str] = None
    temperature: float
    top_p: float
    top_k: int
    seed: int
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class ExperimentResult(BaseModel):
    run_id: str
    model: str
    dataset: str
    split: Optional[str] = None
    temp: float
    top_p: float
    top_k: int
    seed: int
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    runtime: Optional[float] = None
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    evaluation_metric: Optional[str] = None
    at_k_value: Optional[int] = None
    evaluation_tool: Optional[str] = None
    judge_model: Optional[str] = None


class RunResponse(BaseModel):
    run_id: str
    message: str
    script_path: str
    total_experiments: int


class UnitTestRequest(BaseModel):
    model: str
    dataset: str
    split: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 42
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    prompt: str = ""
    local_dir: str = "/testing"


class UnitTestResponse(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
