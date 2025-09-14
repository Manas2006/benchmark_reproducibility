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
    
    # API configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key for CoT Analysis LLM Judge")
    
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
    """CoT analysis for a single sample - supports both legacy and comprehensive analysis"""
    idx: Optional[int] = Field(description="Sample index")
    question: Optional[str] = Field(default=None, description="Problem/question text")
    model_output: Optional[str] = Field(default=None, description="Full model output with reasoning")
    predicted_answer: Optional[str] = Field(default=None, description="Model's predicted answer")
    ground_truth: Optional[str] = Field(default=None, description="Ground truth answer")
    is_correct: bool = Field(description="Whether the answer was correct")
    
    # Analysis method and results
    analysis_method: Optional[str] = Field(default="legacy", description="Analysis method used")
    
    # Legacy analysis fields (optional)
    metrics: Optional[CoTMetrics] = Field(default=None, description="Computed CoT metrics (legacy)")
    has_reasoning: Optional[bool] = Field(default=None, description="Whether sample contains reasoning (legacy)")
    
    # Comprehensive analysis fields (optional)
    flags: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detected flags (comprehensive)")
    evidence: Optional[Dict[str, Any]] = Field(default=None, description="Evidence metrics (comprehensive)")
    rule_scores: Optional[Dict[str, float]] = Field(default=None, description="Rule-based scores (comprehensive)")
    judge_scores: Optional[Dict[str, float]] = Field(default=None, description="LLM judge scores (comprehensive)")
    fused_scores: Optional[Dict[str, float]] = Field(default=None, description="Fused scores (comprehensive)")
    overall_score: Optional[float] = Field(default=None, description="Overall quality score")
    final_correct: Optional[bool] = Field(default=None, description="Whether final answer is correct")
    utility_score: Optional[float] = Field(default=None, description="Utility score")
    coherence_score: Optional[float] = Field(default=None, description="Coherence score")
    factuality_score: Optional[float] = Field(default=None, description="Factuality score")
    faithfulness_score: Optional[float] = Field(default=None, description="Faithfulness score")
    flags_count: Optional[int] = Field(default=None, description="Number of flags detected")
    arith_errors: Optional[int] = Field(default=None, description="Number of arithmetic errors")
    coh_contra_cnt: Optional[int] = Field(default=None, description="Coherence contradictions count")
    fact_contra_cnt: Optional[int] = Field(default=None, description="Factuality contradictions count")
    fact_entail_rate: Optional[float] = Field(default=None, description="Factuality entailment rate")
    intermediate_ok_rate: Optional[float] = Field(default=None, description="Intermediate reasoning correctness rate")

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

class OpenAITestRequest(BaseModel):
    """Request to test OpenAI API key"""
    api_key: str = Field(description="OpenAI API key to test")

class OpenAITestResponse(BaseModel):
    """Response from OpenAI API key test"""
    valid: bool = Field(description="Whether the API key is valid")
    error: Optional[str] = Field(default=None, description="Error message if key is invalid")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Information about available models")


# =============================================================================
# NEW PILLARS-ONLY SCHEMA (Migration from CQS to Four-Pillar Evaluation)
# =============================================================================

class PillarsScores(BaseModel):
    """Four-pillar evaluation scores (0.0 to 1.0)"""
    faithfulness: float = Field(ge=0.0, le=1.0, description="Faithfulness score")
    utility: float = Field(ge=0.0, le=1.0, description="Utility score")
    coherence: float = Field(ge=0.0, le=1.0, description="Coherence score")
    factuality: float = Field(ge=0.0, le=1.0, description="Factuality score")
    overall: float = Field(ge=0.0, le=1.0, description="Overall fused score")


class PillarsFlag(BaseModel):
    """Individual flag detected during analysis"""
    pillar: str = Field(description="Pillar where flag was detected (faithfulness/utility/coherence/factuality)")
    step: str = Field(description="Step identifier where flag was detected")
    issue: str = Field(description="Type of issue detected")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed information about the issue")


class PillarsEntry(BaseModel):
    """Complete analysis result for a single sample"""
    scores: "PillarsScores"
    flags: List["PillarsFlag"]
    evidence: Dict[str, Any]
    rules_raw: Optional[Dict[str, float]] = None
    judge_raw: Optional[Dict[str, Any]] = None
    config_snapshot: Dict[str, Any]
    # Original input data
    problem: str = Field(description="Original problem statement")
    model_output: str = Field(description="Original model output/CoT reasoning")
    gold: Optional[str] = Field(default=None, description="Ground truth answer")


class PillarsSummary(BaseModel):
    """Aggregate statistics for the entire job"""
    # Score averages
    avg_faithfulness: float = Field(description="Average faithfulness score")
    avg_utility: float = Field(description="Average utility score")
    avg_coherence: float = Field(description="Average coherence score")
    avg_factuality: float = Field(description="Average factuality score")
    avg_overall: float = Field(description="Average overall score")
    
    # Flag statistics
    total_flags: int = Field(description="Total number of flags detected")
    flags_by_pillar: Dict[str, int] = Field(description="Flag count per pillar")
    
    # Judge statistics
    judge_call_rate: float = Field(description="Percentage of samples that used judge")
    judge_budget_used: int = Field(description="Number of judge calls made")
    judge_budget_total: int = Field(description="Total judge budget allocated")
    
    # Performance metrics
    total_samples: int = Field(description="Total number of samples analyzed")
    analysis_time: float = Field(description="Total analysis time in seconds")
    avg_time_per_sample: float = Field(description="Average time per sample in seconds")


class CoTAnalysisResponseV2(BaseModel):
    """New pillars-only CoT analysis response (replaces legacy CQS)"""
    job_id: str
    per_sample: List["PillarsEntry"]
    summary: "PillarsSummary"
    analysis_method: str = "pillars_v2"
    config: Dict[str, Any]
    timestamp: str 