"""
Chain-of-Thought Analysis Module

This module provides functionality to analyze mathematical reasoning chains from LLM outputs.
It extracts reasoning steps, computes quality metrics, and identifies patterns in CoT responses.
"""

import re
import json
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class CoTMetrics:
    """Data class for Chain-of-Thought metrics with rigorous CQS scoring"""
    # Basic metrics
    reasoning_steps: int
    total_chars: int
    avg_words_per_step: float
    
    # CQS Component Scores (0.0-1.0 each)
    final_answer_correctness: float
    arithmetic_accuracy: float
    logical_structure_score: float
    consistency_completeness: float
    formatting_notation: float
    
    # Overall CQS Score
    cqs_score: float
    
    # Legacy metrics for backward compatibility
    arithmetic_expressions: int
    has_clear_structure: bool
    has_final_answer: bool
    uses_intermediate_calculations: bool
    shows_work_explicitly: bool
    follows_logical_sequence: bool
    
    # Error analysis
    error_patterns: List[str]
    confidence_score: float


class CoTAnalyzer:
    """Chain-of-Thought reasoning analyzer"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        # Patterns for identifying mathematical expressions
        self.arithmetic_pattern = re.compile(r'[\d\.\,]+\s*[+\-*/÷×]\s*[\d\.\,]+')
        self.calculation_pattern = re.compile(r'<<([^>]+)>>')
        self.final_answer_pattern = re.compile(r'####\s*(.+)$', re.MULTILINE)
        
        # Initialize judge if API key is provided
        self.judge = None
        if openai_api_key:
            try:
                # Import here to avoid import errors if not available
                from .cot_eval_v2.judge import Judge
                # Set the API key in environment for the judge
                os.environ['OPENAI_API_KEY'] = openai_api_key
                self.judge = Judge(mode="SMART", diagnostic=False)
                print("✅ OpenAI Judge initialized for enhanced CoT analysis")
            except ImportError:
                print("⚠️ OpenAI library not available. Using rule-based analysis only.")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI Judge: {e}. Using rule-based analysis only.")
        
        # Common error patterns
        self.error_patterns = {
            'calculation_error': re.compile(r'(?:wrong|incorrect|error|mistake)'),
            'missing_step': re.compile(r'(?:skip|missing|forgot|omit)'),
            'unclear_reasoning': re.compile(r'(?:unclear|confusing|doesn\'t make sense)'),
            'wrong_approach': re.compile(r'(?:wrong approach|incorrect method)')
        }
    
    def analyze_answer(self, answer: str, ground_truth: str = None) -> CoTMetrics:
        """
        Analyze a single Chain-of-Thought answer using rigorous CQS scoring
        
        Args:
            answer: The model's full answer with reasoning
            ground_truth: Ground truth answer for accuracy assessment
            
        Returns:
            CoTMetrics object with computed CQS metrics
        """
        if not answer or not isinstance(answer, str):
            return self._empty_metrics()
        
        # Split reasoning from final answer
        reasoning_text, final_answer = self._extract_reasoning_and_answer(answer)
        
        # Extract reasoning steps
        steps = self._extract_reasoning_steps(reasoning_text)
        
        # Compute basic metrics
        basic_metrics = self._compute_basic_metrics(reasoning_text, steps)
        
        # === CQS COMPONENT SCORING ===
        
        # 1. Final Answer Correctness (30% weight)
        final_answer_correctness = self._score_final_answer(final_answer, ground_truth)
        
        # 2. Arithmetic Accuracy (25% weight)
        arithmetic_accuracy = self._score_arithmetic_accuracy(reasoning_text, steps)
        
        # 3. Logical Structure (20% weight)
        logical_structure_score = self._score_logical_structure(steps, basic_metrics['avg_words_per_step'])
        
        # 4. Consistency & Completeness (15% weight)
        consistency_completeness = self._score_consistency_completeness(steps, reasoning_text)
        
        # 5. Formatting & Notation (10% weight)
        formatting_notation = self._score_formatting_notation(reasoning_text, final_answer)
        
        # Calculate overall CQS score
        cqs_score = (
            0.30 * final_answer_correctness +
            0.25 * arithmetic_accuracy +
            0.20 * logical_structure_score +
            0.15 * consistency_completeness +
            0.10 * formatting_notation
        )
        
        # Legacy metrics for backward compatibility
        arithmetic_expressions = len(self.arithmetic_pattern.findall(reasoning_text)) + len(self.calculation_pattern.findall(reasoning_text))
        has_clear_structure = len(steps) > 1 and any('.' in step for step in steps)
        has_final_answer = bool(final_answer.strip())
        uses_intermediate_calculations = '<<' in reasoning_text or '=' in reasoning_text
        shows_work_explicitly = any(op in reasoning_text for op in ['+', '-', '*', '/', '='])
        follows_logical_sequence = '\n' in reasoning_text or '.' in reasoning_text
        
        # Error detection
        error_analysis = self._detect_errors(reasoning_text)
        
        return CoTMetrics(
            # Basic metrics
            reasoning_steps=basic_metrics['step_count'],
            total_chars=basic_metrics['char_count'],
            avg_words_per_step=basic_metrics['avg_words_per_step'],
            
            # CQS Component Scores
            final_answer_correctness=final_answer_correctness,
            arithmetic_accuracy=arithmetic_accuracy,
            logical_structure_score=logical_structure_score,
            consistency_completeness=consistency_completeness,
            formatting_notation=formatting_notation,
            
            # Overall CQS Score
            cqs_score=cqs_score,
            
            # Legacy metrics for backward compatibility
            arithmetic_expressions=arithmetic_expressions,
            has_clear_structure=has_clear_structure,
            has_final_answer=has_final_answer,
            uses_intermediate_calculations=uses_intermediate_calculations,
            shows_work_explicitly=shows_work_explicitly,
            follows_logical_sequence=follows_logical_sequence,
            
            # Error analysis
            error_patterns=error_analysis['patterns'],
            confidence_score=error_analysis['confidence']
        )
    
    def analyze_job_data(self, job_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze CoT metrics for an entire job dataset
        
        Args:
            job_data: List of job samples with answer, gt, score fields
            
        Returns:
            Comprehensive analysis results with per-sample and aggregate metrics
        """
        if not job_data:
            return {"error": "No data provided"}
        
        per_sample_metrics = []
        aggregate_stats = {
            'total_samples': 0,
            'samples_with_reasoning': 0,
            'avg_reasoning_steps': 0,
            'avg_reasoning_length': 0,
            'arithmetic_accuracy_avg': 0,
            'clarity_score_avg': 0,
            'pattern_distribution': {},
            'error_pattern_frequency': {},
            'correlation_with_correctness': {}
        }
        
        for sample in job_data:
            # Use the model's actual output (code field) + predicted answer, not the formatted answer field
            model_reasoning = sample.get('code', [''])
            model_reasoning_text = model_reasoning[0] if model_reasoning else ""
            predicted_answer = sample.get('pred', [''])
            predicted_answer_text = predicted_answer[0] if predicted_answer else ""
            
            # Construct the full model output for CoT analysis
            full_model_output = f"{model_reasoning_text}\n#### {predicted_answer_text}"
            
            gt = sample.get('gt', '')
            score = sample.get('score', [])
            
            # Analyze this sample using the model's actual output
            metrics = self.analyze_answer(full_model_output, gt)
            
            # Add sample metadata - convert dataclass to dict for Pydantic
            sample_result = {
                'idx': sample.get('idx'),
                'metrics': {
                                    # Basic metrics
                'reasoning_steps': metrics.reasoning_steps,
                'total_chars': metrics.total_chars,
                'avg_words_per_step': metrics.avg_words_per_step,
                
                # CQS Component Scores
                'final_answer_correctness': metrics.final_answer_correctness,
                'arithmetic_accuracy': metrics.arithmetic_accuracy,
                'logical_structure_score': metrics.logical_structure_score,
                'consistency_completeness': metrics.consistency_completeness,
                'formatting_notation': metrics.formatting_notation,
                
                # Overall CQS Score
                'cqs_score': metrics.cqs_score,
                
                # Legacy metrics for backward compatibility
                'arithmetic_expressions': metrics.arithmetic_expressions,
                'has_clear_structure': metrics.has_clear_structure,
                'has_final_answer': metrics.has_final_answer,
                'uses_intermediate_calculations': metrics.uses_intermediate_calculations,
                'shows_work_explicitly': metrics.shows_work_explicitly,
                'follows_logical_sequence': metrics.follows_logical_sequence,
                
                # Error analysis
                'error_patterns': metrics.error_patterns,
                'confidence_score': metrics.confidence_score
                },
                'is_correct': bool(score and score[0]) if score else False,
                'has_reasoning': len(model_reasoning_text.strip()) > 0
            }
            
            per_sample_metrics.append(sample_result)
        
        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_stats(per_sample_metrics)
        
        return {
            'job_summary': aggregate_stats,
            'per_sample_metrics': per_sample_metrics,
            'analysis_metadata': {
                'analyzer_version': '1.0',
                'metrics_computed': list(CoTMetrics.__annotations__.keys())
            }
        }
    
    def _extract_reasoning_and_answer(self, answer: str) -> Tuple[str, str]:
        """Extract reasoning text and final answer from full response"""
        parts = answer.split('####')
        reasoning = parts[0].strip() if parts else answer
        final_answer = parts[1].strip() if len(parts) > 1 else ""
        return reasoning, final_answer
    
    def _extract_reasoning_steps(self, reasoning: str) -> List[str]:
        """Extract individual reasoning steps from the reasoning text"""
        if not reasoning:
            return []
        
        # Split by common step delimiters
        steps = []
        
        # Try splitting by numbered steps (1., 2., etc.)
        numbered_steps = re.split(r'\n\d+\.\s*', reasoning)
        if len(numbered_steps) > 1:
            steps = [step.strip() for step in numbered_steps if step.strip()]
        else:
            # Try splitting by sentences ending with periods
            sentences = re.split(r'\.(?:\s|$)', reasoning)
            steps = [sent.strip() + '.' for sent in sentences if sent.strip()]
        
        return steps
    
    def _compute_basic_metrics(self, reasoning: str, steps: List[str]) -> Dict[str, Any]:
        """Compute basic quantitative metrics"""
        step_count = len(steps)
        char_count = len(reasoning)
        
        # Count words in each step
        total_words = 0
        for step in steps:
            words = len(step.split())
            total_words += words
        
        avg_words_per_step = total_words / step_count if step_count > 0 else 0
        
        return {
            'step_count': step_count,
            'char_count': char_count,
            'avg_words_per_step': avg_words_per_step
        }
    
    def _analyze_arithmetic(self, reasoning: str) -> Dict[str, Any]:
        """Analyze arithmetic expressions and calculations"""
        # Find arithmetic expressions
        expressions = self.arithmetic_pattern.findall(reasoning)
        
        # Find explicit calculations (<<...>> format)
        calculations = self.calculation_pattern.findall(reasoning)
        
        # Simple accuracy assessment (this could be enhanced)
        accuracy = 1.0  # Default to assuming correct unless we detect obvious errors
        
        return {
            'expression_count': len(expressions) + len(calculations),
            'accuracy': accuracy
        }
    
    def _assess_quality(self, reasoning: str, final_answer: str, steps: List[str]) -> Dict[str, Any]:
        """Assess the quality of the reasoning"""
        # Check for clear structure
        has_structure = len(steps) > 1 and any('.' in step for step in steps)
        
        # Check for final answer
        has_final = bool(final_answer.strip())
        
        # Simple clarity score based on various factors
        clarity_factors = []
        
        # Factor 1: Has multiple steps
        clarity_factors.append(1.0 if len(steps) > 1 else 0.5)
        
        # Factor 2: Uses mathematical notation
        clarity_factors.append(1.0 if '=' in reasoning else 0.7)
        
        # Factor 3: Shows intermediate results
        clarity_factors.append(1.0 if '<<' in reasoning else 0.8)
        
        # Factor 4: Reasonable length (not too short or too long)
        length_score = 1.0 if 50 <= len(reasoning) <= 500 else 0.6
        clarity_factors.append(length_score)
        
        clarity_score = sum(clarity_factors) / len(clarity_factors)
        
        return {
            'clear_structure': has_structure,
            'has_final_answer': has_final,
            'clarity_score': clarity_score
        }
    
    def _analyze_patterns(self, reasoning: str) -> Dict[str, bool]:
        """Analyze reasoning patterns"""
        return {
            'uses_calculations': '<<' in reasoning or '=' in reasoning,
            'shows_work': any(op in reasoning for op in ['+', '-', '*', '/', '=']),
            'logical_sequence': '\n' in reasoning or '.' in reasoning
        }
    
    def _detect_errors(self, reasoning: str) -> Dict[str, Any]:
        """Detect potential error patterns"""
        detected_patterns = []
        
        for error_type, pattern in self.error_patterns.items():
            if pattern.search(reasoning.lower()):
                detected_patterns.append(error_type)
        
        # Confidence score (higher = more confident in the reasoning)
        confidence = 0.9  # Default high confidence
        if detected_patterns:
            confidence = max(0.1, 0.9 - 0.2 * len(detected_patterns))
        
        return {
            'patterns': detected_patterns,
            'confidence': confidence
        }
    
    # === CQS SCORING METHODS ===
    
    def _score_final_answer(self, final_answer: str, ground_truth: str) -> float:
        """
        Score final answer correctness (Component 1: 30% weight)
        Returns 1.0 if correct, 0.0 if incorrect
        """
        if not ground_truth:
            return 0.0  # Changed from 0.5 to 0.0 for strict evaluation
        
        # Normalize both answers for comparison
        def normalize_answer(ans):
            if not ans:
                return None
            # Remove whitespace, dollar signs, commas, parentheses
            ans = str(ans).strip().replace('$', '').replace(',', '').replace('(', '').replace(')', '')
            
            # Try to convert to numeric value for numerical comparison
            try:
                # Handle both integer and float formats
                num_val = float(ans)
                # Return as int if it's a whole number, otherwise as float
                return int(num_val) if num_val.is_integer() else num_val
            except (ValueError, AttributeError):
                # If not a number, return the cleaned string in lowercase
                return ans.lower().strip()
        
        norm_final = normalize_answer(final_answer)
        norm_gt = normalize_answer(ground_truth)
        
        # Both must be successfully parsed for comparison
        if norm_final is None or norm_gt is None:
            return 0.0
        
        # Strict numeric comparison with tolerance for floating point errors
        if isinstance(norm_final, (int, float)) and isinstance(norm_gt, (int, float)):
            return 1.0 if abs(norm_final - norm_gt) < 1e-6 else 0.0
        else:
            # String comparison for non-numeric answers
            return 1.0 if str(norm_final) == str(norm_gt) else 0.0
    
    def _score_arithmetic_accuracy(self, reasoning: str, steps: List[str]) -> float:
        """
        Score arithmetic accuracy (Component 2: 25% weight)
        Returns percentage of arithmetically correct steps
        """
        if not steps:
            return 0.0
        
        # Find all calculations in <<>> format
        calculations = self.calculation_pattern.findall(reasoning)
        if not calculations:
            return 1.0  # No explicit calculations to verify
        
        correct_calculations = 0
        total_calculations = len(calculations)
        
        for calc in calculations:
            try:
                # Simple arithmetic evaluation for basic operations
                # Only evaluate safe expressions (numbers and basic operators)
                calc_clean = calc.strip()
                if self._is_safe_arithmetic(calc_clean):
                    # Extract expected result after =
                    if '=' in calc_clean:
                        expression, expected = calc_clean.split('=', 1)
                        expression = expression.strip()
                        expected = expected.strip()
                        
                        # Evaluate the expression
                        try:
                            actual = eval(expression)  # Safe due to _is_safe_arithmetic check
                            expected_num = float(expected)
                            if abs(actual - expected_num) < 0.01:  # Allow small floating point errors
                                correct_calculations += 1
                        except:
                            pass  # Calculation error
                    else:
                        # No expected result, assume correct
                        correct_calculations += 1
                else:
                    # Unsafe expression, assume correct (conservative)
                    correct_calculations += 1
            except:
                pass  # Skip malformed calculations
        
        return correct_calculations / total_calculations if total_calculations > 0 else 1.0
    
    def _is_safe_arithmetic(self, expression: str) -> bool:
        """Check if an arithmetic expression is safe to evaluate"""
        # Only allow numbers, basic operators, parentheses, and spaces
        safe_chars = set('0123456789+-*/()=. ')
        return all(c in safe_chars for c in expression)
    
    def _score_logical_structure(self, steps: List[str], avg_words_per_step: float) -> float:
        """
        Score logical structure (Component 3: 20% weight)
        Combines normalized step count and step brevity
        """
        # Normalized Step Count: full credit at ≥5 steps
        step_count_score = min(len(steps) / 5, 1.0)
        
        # Step Brevity: optimal at 12 tokens per step
        ideal_tokens = 12.0
        if avg_words_per_step > 0:
            brevity_score = max(0.0, 1.0 - abs(avg_words_per_step - ideal_tokens) / ideal_tokens)
        else:
            brevity_score = 0.0
        
        # Combine equally weighted
        return 0.5 * step_count_score + 0.5 * brevity_score
    
    def _score_consistency_completeness(self, steps: List[str], reasoning: str) -> float:
        """
        Score consistency and completeness (Component 4: 15% weight)
        Simplified version using word overlap and missing step detection
        """
        if len(steps) <= 1:
            return 0.5  # Single step gets average score
        
        # Inter-step coherence: measure word overlap between consecutive steps
        coherence_scores = []
        for i in range(len(steps) - 1):
            current_words = set(steps[i].lower().split())
            next_words = set(steps[i + 1].lower().split())
            
            if len(current_words) > 0 and len(next_words) > 0:
                overlap = len(current_words & next_words)
                total_unique = len(current_words | next_words)
                coherence = overlap / total_unique if total_unique > 0 else 0.0
                coherence_scores.append(coherence)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
        
        # Missing step penalty: check for common missing patterns
        missing_penalty = 0.0
        missing_indicators = ['therefore', 'thus', 'so', 'finally', 'in conclusion']
        if not any(indicator in reasoning.lower() for indicator in missing_indicators):
            missing_penalty = 0.2  # Penalty for missing logical connectors
        
        return max(0.0, avg_coherence - missing_penalty)
    
    def _score_formatting_notation(self, reasoning: str, final_answer: str) -> float:
        """
        Score formatting and notation quality (Component 5: 10% weight)
        Each check worth 0.2 points
        """
        score = 0.0
        
        # Check 1: Use of = in reasoning
        if '=' in reasoning:
            score += 0.2
        
        # Check 2: Use of << and >> around calculations
        if '<<' in reasoning and '>>' in reasoning:
            score += 0.2
        
        # Check 3: Clear answer delimiter (#### or similar)
        if '####' in (reasoning + final_answer):
            score += 0.2
        
        # Check 4: No repetition or fluff
        fluff_patterns = ['now let me think', 'let me see', 'hmm', 'well', 'ok so']
        has_fluff = any(pattern in reasoning.lower() for pattern in fluff_patterns)
        if not has_fluff:
            score += 0.2
        
        # Check 5: Consistent formatting (has line breaks or proper punctuation)
        if '\n' in reasoning or '. ' in reasoning:
            score += 0.2
        
        return score
    
    def _compute_aggregate_stats(self, per_sample_metrics: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics across all samples"""
        if not per_sample_metrics:
            return {}
        
        total_samples = len(per_sample_metrics)
        samples_with_reasoning = sum(1 for s in per_sample_metrics if s['has_reasoning'])
        
        # Compute averages
        metrics_list = [s['metrics'] for s in per_sample_metrics]
        
        avg_steps = sum(m['reasoning_steps'] for m in metrics_list) / total_samples
        avg_length = sum(m['total_chars'] for m in metrics_list) / total_samples
        avg_arithmetic_acc = sum(m['arithmetic_accuracy'] for m in metrics_list) / total_samples
        avg_cqs = sum(m['cqs_score'] for m in metrics_list) / total_samples
        
        # CQS component averages
        avg_final_answer_correct = sum(m['final_answer_correctness'] for m in metrics_list) / total_samples
        avg_logical_structure = sum(m['logical_structure_score'] for m in metrics_list) / total_samples
        avg_consistency_complete = sum(m['consistency_completeness'] for m in metrics_list) / total_samples
        avg_formatting_notation = sum(m['formatting_notation'] for m in metrics_list) / total_samples
        
        # Pattern distribution
        pattern_counts = {
            'uses_calculations': sum(1 for m in metrics_list if m['uses_intermediate_calculations']),
            'shows_work': sum(1 for m in metrics_list if m['shows_work_explicitly']),
            'logical_sequence': sum(1 for m in metrics_list if m['follows_logical_sequence'])
        }
        
        # Error pattern frequency
        error_freq = {}
        for metrics in metrics_list:
            for error in metrics['error_patterns']:
                error_freq[error] = error_freq.get(error, 0) + 1
        
        # Correlation with correctness
        correct_samples = [s for s in per_sample_metrics if s['is_correct']]
        incorrect_samples = [s for s in per_sample_metrics if not s['is_correct']]
        
        correlation = {
            'correct_avg_steps': sum(s['metrics']['reasoning_steps'] for s in correct_samples) / len(correct_samples) if correct_samples else 0,
            'incorrect_avg_steps': sum(s['metrics']['reasoning_steps'] for s in incorrect_samples) / len(incorrect_samples) if incorrect_samples else 0,
            'correct_avg_cqs': sum(s['metrics']['cqs_score'] for s in correct_samples) / len(correct_samples) if correct_samples else 0,
            'incorrect_avg_cqs': sum(s['metrics']['cqs_score'] for s in incorrect_samples) / len(incorrect_samples) if incorrect_samples else 0
        }
        
        return {
            'total_samples': total_samples,
            'samples_with_reasoning': samples_with_reasoning,
            'avg_reasoning_steps': round(avg_steps, 2),
            'avg_reasoning_length': round(avg_length, 2),
            'arithmetic_accuracy_avg': round(avg_arithmetic_acc, 3),
            'cqs_score_avg': round(avg_cqs, 3),
            
            # CQS Component Averages
            'final_answer_correctness_avg': round(avg_final_answer_correct, 3),
            'logical_structure_avg': round(avg_logical_structure, 3),
            'consistency_completeness_avg': round(avg_consistency_complete, 3),
            'formatting_notation_avg': round(avg_formatting_notation, 3),
            
            'pattern_distribution': pattern_counts,
            'error_pattern_frequency': error_freq,
            'correlation_with_correctness': correlation
        }
    
    def _empty_metrics(self) -> CoTMetrics:
        """Return empty/default metrics for invalid input"""
        return CoTMetrics(
            # Basic metrics
            reasoning_steps=0,
            total_chars=0,
            avg_words_per_step=0.0,
            
            # CQS Component Scores
            final_answer_correctness=0.0,
            arithmetic_accuracy=0.0,
            logical_structure_score=0.0,
            consistency_completeness=0.0,
            formatting_notation=0.0,
            
            # Overall CQS Score
            cqs_score=0.0,
            
            # Legacy metrics for backward compatibility
            arithmetic_expressions=0,
            has_clear_structure=False,
            has_final_answer=False,
            uses_intermediate_calculations=False,
            shows_work_explicitly=False,
            follows_logical_sequence=False,
            
            # Error analysis
            error_patterns=[],
            confidence_score=0.0
        )