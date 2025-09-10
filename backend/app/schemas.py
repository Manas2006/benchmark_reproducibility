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
    prompt: Optional[str] = Field(default=None, description="Custom prompt template to use for evaluation")
    prompt_type: Optional[str] = Field(default=None, description="Standard prompt type (e.g., 'tool-integrated', 'cot', 'pal')")
    temperature: confloat(ge=0, le=5) = 0.0
    top_p: confloat(gt=0, le=1) = 1.0
    top_k: conint(ge=0) = 0  # 0 = disabled
    max_tokens: conint(ge=1, le=8192) = 2048
    seed: conint(ge=0) = 42
    eval_method: str = "pass@k"  # enum later
    k: conint(ge=1, le=32) = 1  # This maps to n_sampling internally
    backend: Backend = Backend.local
    path_config: Optional[PathConfig] = None  # Use default if not provided
    # Probability tracking
    enable_prob_tracking: bool = Field(default=False, description="Track probabilities of target answer tokens (requires vLLM)")
    # Together API options
    use_together_api: bool = Field(default=False, description="Use Together API instead of local models")
    together_api_key: Optional[str] = Field(default=None, description="Together API key (if not in env)")
    together_logprobs: conint(ge=0, le=5) = 0
    # Optional plotting directives (frontend convenience; plotting done post-hoc)
    prob_plot_type: Optional[str] = Field(default=None, description="'aggregate' or 'single' for probability plots")
    prob_plot_sample_id: Optional[int] = Field(default=None, description="Sample idx for single plot")

class EvalResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    run_path: Optional[str] = None
    sbatch_path: Optional[str] = None

class EvalConfig(BaseModel):
    model: str
    dataset: str
    prompt: str
    temperature: float
    top_p: float
    n_sampling: int
    seed: int
    backend: Backend

class PathConfigResponse(BaseModel):
    current_config: PathConfig
    default_config: PathConfig

class CoTMetrics(BaseModel):
    """Chain-of-Thought analysis metrics with rigorous CQS scoring"""
    # Basic metrics
    reasoning_steps: int = Field(description="Number of reasoning steps identified")
    total_chars: int = Field(description="Total character count in reasoning")
    avg_words_per_step: float = Field(description="Average words per reasoning step")
    
    # CQS Component Scores (0.0-1.0 each)
    final_answer_correctness: float = Field(ge=0, le=1, description="Final answer matches ground truth (30% weight)")
    arithmetic_accuracy: float = Field(ge=0, le=1, description="Percentage of arithmetically correct steps (25% weight)")
    logical_structure_score: float = Field(ge=0, le=1, description="Quality of reasoning structure (20% weight)")
    consistency_completeness: float = Field(ge=0, le=1, description="Inter-step coherence and completeness (15% weight)")
    formatting_notation: float = Field(ge=0, le=1, description="Formatting and notation quality (10% weight)")
    
    # Overall CQS Score
    cqs_score: float = Field(ge=0, le=1, description="Overall CoT Quality Score (weighted combination)")
    
    # Legacy metrics for backward compatibility
    arithmetic_expressions: int = Field(description="Number of arithmetic expressions found")
    has_clear_structure: bool = Field(description="Whether reasoning has clear structure")
    has_final_answer: bool = Field(description="Whether final answer is present")
    uses_intermediate_calculations: bool = Field(description="Uses intermediate calculations")
    shows_work_explicitly: bool = Field(description="Shows work step by step")
    follows_logical_sequence: bool = Field(description="Follows logical sequence")
    
    # Error analysis
    error_patterns: List[str] = Field(description="Detected error patterns")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in reasoning quality")

class CoTSampleAnalysis(BaseModel):
    """CoT analysis for a single sample"""
    idx: Optional[int] = Field(description="Sample index")
    metrics: CoTMetrics = Field(description="Computed CoT metrics")
    is_correct: bool = Field(description="Whether the answer was correct")
    has_reasoning: bool = Field(description="Whether sample contains reasoning")

class CoTJobSummary(BaseModel):
    """Summary statistics for CoT analysis across a job"""
    total_samples: int = Field(description="Total number of samples")
    samples_with_reasoning: int = Field(description="Samples containing reasoning")
    avg_reasoning_steps: float = Field(description="Average reasoning steps per sample")
    avg_reasoning_length: float = Field(description="Average reasoning length in characters")
    arithmetic_accuracy_avg: float = Field(description="Average arithmetic accuracy")
    cqs_score_avg: float = Field(description="Average CoT Quality Score")
    
    # CQS Component Averages
    final_answer_correctness_avg: float = Field(description="Average final answer correctness")
    logical_structure_avg: float = Field(description="Average logical structure score")
    consistency_completeness_avg: float = Field(description="Average consistency & completeness score")
    formatting_notation_avg: float = Field(description="Average formatting & notation score")
    
    pattern_distribution: Dict[str, int] = Field(description="Distribution of reasoning patterns")
    error_pattern_frequency: Dict[str, int] = Field(description="Frequency of error patterns")
    correlation_with_correctness: Dict[str, float] = Field(description="Correlation metrics with correctness")

class CoTAnalysisResponse(BaseModel):
    """Complete CoT analysis response"""
    job_id: str = Field(description="Job identifier")
    job_summary: CoTJobSummary = Field(description="Aggregate statistics")
    per_sample_metrics: List[CoTSampleAnalysis] = Field(description="Per-sample analysis results")
    analysis_metadata: Dict[str, Any] = Field(description="Analysis metadata and version info")
    computation_time: float = Field(description="Time taken for analysis in seconds") 