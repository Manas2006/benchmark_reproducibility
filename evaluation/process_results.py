#!/usr/bin/env python3
"""
Post-processing script for evaluation results.
This script processes raw evaluation results and generates additional metrics and analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_results(input_file: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    return results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate basic metrics from results."""
    if not results:
        return {}
    
    total_samples = len(results)
    correct_samples = 0
    total_chars = 0
    total_reasoning_steps = 0
    
    for result in results:
        # Count correct answers
        score = result.get('score', [])
        if isinstance(score, list) and score:
            if score[0]:  # First score is typically the main correctness
                correct_samples += 1
        elif isinstance(score, bool):
            if score:
                correct_samples += 1
        
        # Calculate reasoning length
        code = result.get('code', [])
        if isinstance(code, list) and code:
            reasoning_text = str(code[0]) if code[0] else ""
        else:
            reasoning_text = str(code) if code else ""
        
        total_chars += len(reasoning_text)
        
        # Estimate reasoning steps (rough count of sentences/calculations)
        reasoning_steps = reasoning_text.count('.') + reasoning_text.count('=') + reasoning_text.count('+') + reasoning_text.count('-')
        total_reasoning_steps += max(1, reasoning_steps)
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    avg_chars = total_chars / total_samples if total_samples > 0 else 0
    avg_reasoning_steps = total_reasoning_steps / total_samples if total_samples > 0 else 0
    
    return {
        'total_samples': total_samples,
        'correct_samples': correct_samples,
        'accuracy': accuracy,
        'avg_reasoning_length': avg_chars,
        'avg_reasoning_steps': avg_reasoning_steps
    }


def process_results(input_file: str, model_name: str, output_file: str) -> bool:
    """Process evaluation results and generate metrics."""
    try:
        print(f"Processing results from: {input_file}")
        print(f"Model: {model_name}")
        print(f"Output file: {output_file}")
        
        # Load results
        results = load_results(input_file)
        if not results:
            print("Warning: No valid results found in input file")
            return False
        
        print(f"Loaded {len(results)} results")
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Create processed results with additional metadata
        processed_results = []
        for i, result in enumerate(results):
            processed_result = result.copy()
            
            # Add processing metadata
            processed_result['_processed'] = True
            processed_result['_model_name'] = model_name
            processed_result['_sample_id'] = i
            
            # Add individual metrics
            score = result.get('score', [])
            processed_result['_is_correct'] = False
            if isinstance(score, list) and score:
                processed_result['_is_correct'] = bool(score[0])
            elif isinstance(score, bool):
                processed_result['_is_correct'] = score
            
            # Add reasoning analysis
            code = result.get('code', [])
            if isinstance(code, list) and code:
                reasoning_text = str(code[0]) if code[0] else ""
            else:
                reasoning_text = str(code) if code else ""
            
            processed_result['_reasoning_length'] = len(reasoning_text)
            processed_result['_reasoning_steps'] = reasoning_text.count('.') + reasoning_text.count('=') + reasoning_text.count('+') + reasoning_text.count('-')
            
            processed_results.append(processed_result)
        
        # Write processed results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in processed_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Write metrics summary
        metrics_file = output_file.replace('.jsonl', '_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': model_name,
                'input_file': input_file,
                'output_file': output_file,
                'metrics': metrics,
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Processing completed successfully!")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Average reasoning length: {metrics['avg_reasoning_length']:.1f} characters")
        print(f"Average reasoning steps: {metrics['avg_reasoning_steps']:.1f}")
        print(f"Processed results saved to: {output_file}")
        print(f"Metrics saved to: {metrics_file}")
        
        return True
        
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process evaluation results')
    parser.add_argument('--input_file', required=True, help='Input JSONL file with results')
    parser.add_argument('--model_name_or_path', required=True, help='Model name or path')
    parser.add_argument('--output_file', required=True, help='Output file for processed results')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process results
    success = process_results(
        input_file=args.input_file,
        model_name=args.model_name_or_path,
        output_file=args.output_file
    )
    
    if success:
        print("Post-processing completed successfully!")
        sys.exit(0)
    else:
        print("Post-processing failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
