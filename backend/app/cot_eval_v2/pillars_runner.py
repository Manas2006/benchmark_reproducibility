"""
PillarsRunner: Main orchestration class for the new four-pillar CoT evaluation pipeline.

This class coordinates DeBERTa NLI, deterministic checks, and GPT judge to produce
comprehensive evaluation results without any legacy CQS fallbacks.
"""

import os
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .evaluator import PillarsEvaluator
from .judge import Judge
from .scoring import fuse_with_judge
from ..schemas import PillarsEntry, PillarsScores, PillarsFlag, PillarsSummary, CoTAnalysisResponseV2


class PillarsRunner:
    """
    Main runner for the four-pillar CoT evaluation pipeline.
    
    Coordinates:
    1. DeBERTa MNLI for coherence & factuality flags
    2. Deterministic checks for utility & faithfulness
    3. GPT-4o-mini judge for subjective scoring
    4. Score fusion with evidence-based caps
    """
    
    def __init__(
        self,
        evaluator: Optional[PillarsEvaluator] = None,
        rubric_path: Optional[str] = None,
        llm_fn: Optional[Callable] = None,
        gating: str = "SMART",
        budget: int = 999999,
        diagnostic: bool = False
    ):
        """
        Initialize the pillars runner.
        
        Args:
            evaluator: PillarsEvaluator instance (auto-created if None)
            rubric_path: Path to judge rubric template
            llm_fn: LLM function for judge calls
            gating: Judge gating mode (SMART/ALWAYS/NEVER)
            budget: Maximum judge calls allowed
            diagnostic: Whether to include diagnostic information
        """
        self.evaluator = evaluator or PillarsEvaluator(use_nli=True)
        self.rubric_path = rubric_path
        self.llm_fn = llm_fn
        self.gating = gating
        self.budget = budget
        self.diagnostic = diagnostic
        
        # Initialize judge if LLM function provided and not in rollback mode
        from .config import config
        
        self.judge = None
        if llm_fn is not None and not config.should_use_rollback():
            effective_gating = config.get_judge_mode()
            self.judge = Judge(
                model="gpt-4o-mini",
                mode=effective_gating,
                diagnostic=diagnostic
            )
            self.evaluator.judge = self.judge
            print(f"🎯 Judge initialized with mode: {effective_gating}")
        elif config.should_use_rollback():
            print(f"🔄 Rollback mode enabled - using rules+DeBERTa only")
        
        # Track budget usage
        self.judge_calls_made = 0
        
        print(f"🚀 PillarsRunner initialized with gating={gating}, budget={budget}")
    
    def run(self, problem: str, cot_text: str, gold: Optional[str] = None) -> PillarsEntry:
        """
        Run complete four-pillar analysis on a single sample.
        
        Args:
            problem: The problem statement
            cot_text: Chain-of-thought reasoning text
            gold: Ground truth answer (optional)
            
        Returns:
            PillarsEntry with complete analysis results
        """
        start_time = time.time()
        
        # Run the core evaluation (DeBERTa + deterministic checks)
        flags, evidence, rule_scores, judge_scores, fused_scores = self.evaluator.analyze(
            problem=problem,
            cot_text=cot_text,
            gold=gold
        )
        
        # Convert flags to PillarsFlag objects
        flags_list = []
        if flags.has_flags():
            for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                pillar_flags = flags.get_flags_by_pillar(pillar)
                for flag in pillar_flags:
                    flags_list.append(PillarsFlag(
                        pillar=pillar,
                        step=flag.step,
                        issue=flag.issue,
                        details=flag.details
                    ))
        
        # Create configuration snapshot
        config_snapshot = {
            "nli_model": "microsoft/deberta-base-mnli",
            "judge_model": "gpt-4o-mini" if self.judge else None,
            "judge_gating": self.gating,
            "judge_diagnostic": self.diagnostic,
            "budget_remaining": self.budget - self.judge_calls_made
        }
        
        # Track judge calls if made
        if judge_scores and any(v is not None for v in judge_scores.values()):
            self.judge_calls_made += 1
        
        return PillarsEntry(
            scores=PillarsScores(
                faithfulness=fused_scores.get("faithfulness", 0.0),
                utility=fused_scores.get("utility", 0.0),
                coherence=fused_scores.get("coherence", 0.0),
                factuality=fused_scores.get("factuality", 0.0),
                overall=fused_scores.get("overall", 0.0)
            ),
            flags=flags_list,
            evidence=evidence,
            rules_raw=rule_scores if self.diagnostic else None,
            judge_raw=judge_scores if self.diagnostic else None,
            config_snapshot=config_snapshot,
            problem=problem,
            model_output=cot_text,
            gold=gold
        )
    
    def run_batch(
        self, 
        samples: List[Dict[str, Any]], 
        job_id: str
    ) -> CoTAnalysisResponseV2:
        """
        Run analysis on a batch of samples.
        
        Args:
            samples: List of samples with problem, cot_text, gold fields
            job_id: Job identifier
            
        Returns:
            Complete analysis response
        """
        start_time = time.time()
        per_sample_results = []
        
        print(f"🔄 Starting batch analysis for job {job_id} ({len(samples)} samples)")
        
        for i, sample in enumerate(samples):
            try:
                problem = sample.get("problem", "")
                cot_text = sample.get("cot_text", "")
                gold = sample.get("gold", "")
                
                result = self.run(problem, cot_text, gold)
                per_sample_results.append(result)
                
                # Progress logging
                if (i + 1) % 100 == 0 or i == len(samples) - 1:
                    print(f"📊 Processed {i + 1}/{len(samples)} samples")
                    
            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                # Create empty result for failed sample
                per_sample_results.append(PillarsEntry(
                    scores=PillarsScores(faithfulness=0.0, utility=0.0, coherence=0.0, factuality=0.0, overall=0.0),
                    flags=[],
                    evidence={},
                    config_snapshot={"error": str(e)}
                ))
        
        # Compute summary statistics
        total_time = time.time() - start_time
        summary = self._compute_summary(per_sample_results, total_time)
        
        return CoTAnalysisResponseV2(
            job_id=job_id,
            per_sample=per_sample_results,
            summary=summary,
            analysis_method="pillars_v2",
            config={
                "nli_model": "microsoft/deberta-base-mnli",
                "judge_model": "gpt-4o-mini" if self.judge else None,
                "judge_gating": self.gating,
                "budget_total": self.budget,
                "budget_used": self.judge_calls_made
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _compute_summary(self, results: List[PillarsEntry], total_time: float) -> PillarsSummary:
        """Compute aggregate statistics from per-sample results."""
        if not results:
            return PillarsSummary(
                avg_faithfulness=0.0, avg_utility=0.0, avg_coherence=0.0, avg_factuality=0.0, avg_overall=0.0,
                total_flags=0, flags_by_pillar={}, judge_call_rate=0.0, judge_budget_used=0, judge_budget_total=self.budget,
                total_samples=0, analysis_time=total_time, avg_time_per_sample=0.0
            )
        
        # Score averages
        avg_faithfulness = sum(r.scores.faithfulness for r in results) / len(results)
        avg_utility = sum(r.scores.utility for r in results) / len(results)
        avg_coherence = sum(r.scores.coherence for r in results) / len(results)
        avg_factuality = sum(r.scores.factuality for r in results) / len(results)
        avg_overall = sum(r.scores.overall for r in results) / len(results)
        
        # Flag statistics
        total_flags = sum(len(r.flags) for r in results)
        flags_by_pillar = {}
        for result in results:
            for flag in result.flags:
                flags_by_pillar[flag.pillar] = flags_by_pillar.get(flag.pillar, 0) + 1
        
        # Judge statistics
        judge_calls = sum(1 for r in results if r.judge_raw is not None)
        judge_call_rate = (judge_calls / len(results)) * 100.0 if results else 0.0
        
        return PillarsSummary(
            avg_faithfulness=avg_faithfulness,
            avg_utility=avg_utility,
            avg_coherence=avg_coherence,
            avg_factuality=avg_factuality,
            avg_overall=avg_overall,
            total_flags=total_flags,
            flags_by_pillar=flags_by_pillar,
            judge_call_rate=judge_call_rate,
            judge_budget_used=self.judge_calls_made,
            judge_budget_total=self.budget,
            total_samples=len(results),
            analysis_time=total_time,
            avg_time_per_sample=total_time / len(results) if results else 0.0
        )


def llm_fn_from_settings_or_env() -> Optional[Callable]:
    """
    Create LLM function from settings or environment variables.
    
    Returns:
        LLM function if OpenAI API key is available, None otherwise
    """
    try:
        from ..path_manager import PathManager
        
        path_manager = PathManager()
        config = path_manager.get_config()
        
        if config.openai_api_key:
            os.environ['OPENAI_API_KEY'] = config.openai_api_key
            print(f"✅ OpenAI API key loaded from settings")
            return lambda **kwargs: kwargs  # Placeholder - Judge class handles OpenAI calls
        else:
            print(f"⚠️ No OpenAI API key configured")
            return None
            
    except Exception as e:
        print(f"❌ Error loading OpenAI API key: {e}")
        return None
