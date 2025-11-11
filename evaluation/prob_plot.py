"""
Probability Plotting Tool for Mathematical Evaluation Results

This script provides comprehensive plotting capabilities for analyzing model performance
on mathematical reasoning tasks (GSM8K, MATH, etc.) with probability metrics.

FEATURES:
---------
1. AGGREGATE PLOTS: Overall probability metrics across all samples
2. SINGLE SAMPLE PLOTS: Detailed analysis of individual questions
3. CORRECT/INCORRECT AGGREGATE: Separate analysis for correct vs incorrect answers
4. MATH DIFFICULTY LEVEL ANALYSIS:
    - level_single: Plot for a specific difficulty level (1-5)
    - level_aggregate: Comparison across all difficulty levels
5. PATH OF DISTRIBUTIONS: Visualize model vs gold path trajectories
6. TRUNCATION ANALYSIS INTEGRATION: Compare with truncation experiment results

NEW FEATURES (Latest Update):
------------------------------
- Expert CoT Probability: Aggregate plots now include a fourth line for the
  'expert_cot_probability' metric if it exists in the data.
- First/Last Token Bar Charts: Replaced the starting/ending token line plots with
  bar charts that show the average probability of the very first and very last
  token. These plots are now compatible with ALL datasets:
  * For MATH dataset: Groups by difficulty level (1-5) if available
  * For other datasets (GSM8K, SVAMP, etc.): Shows a single overall average bar
  * Handles datasets without difficulty levels gracefully
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def load_data(jsonl_file_path):
    """Loads data from a JSONL file into a dictionary indexed by sample 'idx'."""
    data = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    # Ensure there is an idx field
    return {item['idx']: item for item in data if 'idx' in item}


def parse_math_level(level_str):
    """Parse math level from string like 'Level 3' to integer 3."""
    if not level_str:
        return None
    try:
        if level_str.startswith('Level '):
            return int(level_str.split()[1])
        return int(level_str)
    except (ValueError, IndexError):
        return None


def normalize_and_bin_sequences(sequences, num_bins=100):
    """
    Normalize sequences to percentage-based bins and calculate averages.
    """
    if not sequences:
        return [np.nan] * num_bins
    
    bins = [[] for _ in range(num_bins)]
    
    for sequence in sequences:
        if not sequence or len(sequence) == 0:
            continue
            
        for i, value in enumerate(sequence):
            if value is not None:
                bin_index = int((i / len(sequence)) * num_bins)
                bin_index = min(bin_index, num_bins - 1)
                bins[bin_index].append(value)
    
    averages = []
    for bin_values in bins:
        if bin_values:
            averages.append(np.mean(bin_values))
        else:
            averages.append(np.nan)
    
    return averages


def group_data_by_math_level(data):
    """Group data by math difficulty level."""
    level_groups = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    has_levels = False
    for idx, sample in data.items():
        level_str = sample.get('level')
        if level_str:
            has_levels = True
            level_num = parse_math_level(level_str)
            if level_num and level_num in level_groups:
                level_groups[level_num][idx] = sample
    return level_groups, has_levels


def extract_expert_cot_probability(sample):
    """Extract expert CoT probability from sample, handling various data formats."""
    # Try different possible field names
    expert_prob = sample.get('expert_cot_probability')
    if expert_prob is not None:
        return expert_prob
    
    # If not found, return None (will be filtered out)
    return None


def get_answer_confidence(sample, tokenizer=None, data_name="math"):
    """
    Extract the final answer and calculate its mean token probability (confidence).
    
    This is useful for Expected Calibration Error (ECE) calculations.
    
    Args:
        sample: Sample dictionary from JSONL file
        tokenizer: Optional tokenizer (improves accuracy if provided)
        data_name: Dataset name (affects answer extraction logic)
    
    Returns:
        Dictionary with:
        - 'confidence': Mean probability of answer tokens (float or None)
        - 'answer_text': Extracted answer text
        - 'answer_token_indices': Indices of answer tokens
        - 'answer_token_probs': Probabilities of answer tokens
    """
    from parser import extract_answer
    
    result = {
        'confidence': None,
        'answer_text': None,
        'answer_token_indices': [],
        'answer_token_probs': []
    }
    
    # Get model output
    model_output = None
    for field in ["code", "pred", "output", "model_output"]:
        if field in sample and sample[field]:
            model_output = sample[field]
            break
    
    if not model_output:
        return result
    
    # Handle list format (multiple samples)
    if isinstance(model_output, list):
        model_output = model_output[0] if model_output else ""
    
    if isinstance(model_output, list):
        model_output = "\n".join(str(x) for x in model_output)
    else:
        model_output = str(model_output)
    
    # Extract answer using the same function as evaluation
    try:
        answer_text = extract_answer(model_output, data_name)
    except Exception:
        answer_text = None
    
    if not answer_text:
        return result
    
    result['answer_text'] = answer_text
    
    # Get token data
    chosen_token_ids_dict = sample.get('chosen_token_ids', {})
    chosen_token_probs_dict = sample.get('chosen_token_probs', {})
    
    # Get epoch_0 data (main generation)
    chosen_token_ids = chosen_token_ids_dict.get('epoch_0', [])
    chosen_token_probs = chosen_token_probs_dict.get('epoch_0', [])
    
    # Try alternative field names
    if not chosen_token_ids:
        chosen_token_ids = sample.get('generated_token_ids', [])
    if not chosen_token_probs:
        chosen_token_probs = sample.get('chosen_token_probs', [])
        if isinstance(chosen_token_probs, dict):
            chosen_token_probs = chosen_token_probs.get('epoch_0', [])
    
    if not chosen_token_ids or not chosen_token_probs:
        return result
    
    # Filter out None values and align lengths
    valid_token_ids = [tid for tid in chosen_token_ids if tid is not None]
    valid_token_probs = [prob for prob in chosen_token_probs if prob is not None]
    
    if len(valid_token_ids) != len(valid_token_probs):
        min_len = min(len(valid_token_ids), len(valid_token_probs))
        valid_token_ids = valid_token_ids[:min_len]
        valid_token_probs = valid_token_probs[:min_len]
    
    # Find answer token indices
    answer_indices = []
    answer_probs = []
    
    if tokenizer is not None:
        try:
            # Method 1: Direct token ID matching
            answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
            if answer_token_ids:
                # Search for this sequence in chosen_token_ids
                for i in range(len(valid_token_ids) - len(answer_token_ids) + 1):
                    if valid_token_ids[i:i+len(answer_token_ids)] == answer_token_ids:
                        answer_indices = list(range(i, i + len(answer_token_ids)))
                        answer_probs = valid_token_probs[i:i+len(answer_token_ids)]
                        break
        except Exception:
            pass
        
        # Method 2: If not found, try character position mapping
        if not answer_indices:
            try:
                decoded_text = ""
                token_boundaries = []
                
                for idx, token_id in enumerate(valid_token_ids):
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    start_char = len(decoded_text)
                    decoded_text += token_text
                    end_char = len(decoded_text)
                    token_boundaries.append((start_char, end_char, idx))
                
                if answer_text in decoded_text:
                    answer_start_char = decoded_text.find(answer_text)
                    answer_end_char = answer_start_char + len(answer_text)
                    
                    for start_char, end_char, token_idx in token_boundaries:
                        if not (end_char <= answer_start_char or start_char >= answer_end_char):
                            answer_indices.append(token_idx)
                            if token_idx < len(valid_token_probs):
                                answer_probs.append(valid_token_probs[token_idx])
            except Exception:
                pass
    
    # Method 3: Fallback - use last N tokens (heuristic)
    if not answer_indices:
        last_n = min(10, len(valid_token_ids))
        if last_n > 0:
            answer_indices = list(range(len(valid_token_ids) - last_n, len(valid_token_ids)))
            answer_probs = valid_token_probs[-last_n:]
    
    if not answer_indices or not answer_probs:
        return result
    
    result['answer_token_indices'] = answer_indices
    result['answer_token_probs'] = answer_probs
    
    # Calculate mean probability (confidence)
    valid_probs = [p for p in answer_probs if p is not None]
    if valid_probs:
        result['confidence'] = sum(valid_probs) / len(valid_probs)
    
    return result


def plot_average_probability_by_step(data, dataset_name, method_name, output_dir):
    """
    Plots average probabilities and entropy, now including Expert CoT Probability.
    """
    print("Generating aggregate plot of average metrics per step (percentage-based)...")

    prob_sequences, chosen_prob_sequences, entropy_sequences, expert_cot_probs = [], [], [], []
    log_lengths, num_correct, total = [], 0, 0

    for sample in data.values():
        prob_log = sample.get('probability_log', {}).get('epoch_0', [])
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        entropies_log = sample.get('entropies', {}).get('epoch_0', [])
        expert_prob = extract_expert_cot_probability(sample)

        if prob_log:
            log_lengths.append(len(prob_log))
            prob_sequences.append(prob_log)
        if chosen_probs_log:
            chosen_prob_sequences.append(chosen_probs_log)
        if entropies_log:
            entropy_sequences.append(entropies_log)
        if expert_prob is not None:
            expert_cot_probs.append(expert_prob)

        score = sample.get('score')
        is_correct = None
        if isinstance(score, list) and score:
            is_correct = bool(score[0] == 1)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        
        if is_correct is not None:
            total += 1
            if is_correct:
                num_correct += 1

    if not any([prob_sequences, chosen_prob_sequences, entropy_sequences]):
        print("No probability data found to plot.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    avg_probabilities = normalize_and_bin_sequences(prob_sequences)
    avg_chosen_probs = normalize_and_bin_sequences(chosen_prob_sequences)
    avg_entropies = normalize_and_bin_sequences(entropy_sequences)
    avg_expert_cot_prob = np.mean(expert_cot_probs) if expert_cot_probs else None
    
    # Debug information for expert CoT probability
    if expert_cot_probs:
        print(f"Expert CoT Probability: Found {len(expert_cot_probs)} samples with expert CoT data")
        print(f"Expert CoT Probability: Min={min(expert_cot_probs):.2e}, Max={max(expert_cot_probs):.2e}, Mean={avg_expert_cot_prob:.2e}")
    else:
        print("Expert CoT Probability: No expert CoT probability data found in samples")

    percentage_steps = list(range(100))
    avg_acc = (num_correct / total * 100.0) if total > 0 else None

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    lines = []
    
    if prob_sequences:
        line1, = ax1.plot(percentage_steps, avg_probabilities, marker='o', linestyle='-', color='blue', linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
        lines.append(line1)
    if chosen_prob_sequences:
        line2, = ax1.plot(percentage_steps, avg_chosen_probs, marker='s', linestyle='--', color='green', linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)
        lines.append(line2)
    if avg_expert_cot_prob is not None:
        # Format the expert CoT probability value appropriately for very small numbers
        if avg_expert_cot_prob < 1e-10:
            prob_label = f'Avg. Expert CoT Prob ({avg_expert_cot_prob:.2e})'
        else:
            prob_label = f'Avg. Expert CoT Prob ({avg_expert_cot_prob:.6f})'
        line4, = ax1.plot(percentage_steps, [avg_expert_cot_prob] * 100, linestyle='-.', color='purple', linewidth=2.5, label=prob_label, alpha=0.9)
        lines.append(line4)

    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax2 = ax1.twinx()
    if entropy_sequences:
        line3, = ax2.plot(percentage_steps, avg_entropies, marker='^', linestyle='-.', color='red', linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
        lines.append(line3)
    
    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=11)

    base_title = f'Generation Metrics vs. Completion % on {dataset_name} ({method_name})'
    if avg_acc is not None:
        base_title += f' — Avg Acc: {avg_acc:.1f}%'
    plt.title(base_title, fontsize=16)

    stats_text = f"Avg. Steps: {avg_steps:.1f}"
    if avg_acc is not None:
        stats_text += f"\nAvg Accuracy: {avg_acc:.1f}%"
    plt.figtext(0.5, 0.02, stats_text, fontsize=11, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_aggregate_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {full_path}")
    plt.close()
    return full_path


def plot_correct_or_incorrect_aggregate(data, dataset_name, method_name, output_dir, is_correct_plot):
    """
    A unified function to plot aggregates for correct or incorrect answers, now including Expert CoT Probability.
    """
    category = "CORRECT" if is_correct_plot else "INCORRECT"
    print(f"Generating aggregate plot for {category} answers only...")

    prob_sequences, chosen_prob_sequences, entropy_sequences, expert_cot_probs = [], [], [], []
    log_lengths = []
    num_in_category = 0
    num_total_samples = len(data)

    for sample in data.values():
        score = sample.get('score')
        is_correct = None
        if isinstance(score, list) and score:
            is_correct = bool(score[0] == 1)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        
        if (is_correct_plot and is_correct is not True) or (not is_correct_plot and is_correct is not False):
            continue
        
        num_in_category += 1
        
        prob_log = sample.get('probability_log', {}).get('epoch_0', [])
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        entropies_log = sample.get('entropies', {}).get('epoch_0', [])
        expert_prob = extract_expert_cot_probability(sample)

        if prob_log:
            log_lengths.append(len(prob_log))
            prob_sequences.append(prob_log)
        if chosen_probs_log:
            chosen_prob_sequences.append(chosen_probs_log)
        if entropies_log:
            entropy_sequences.append(entropies_log)
        if expert_prob is not None:
            expert_cot_probs.append(expert_prob)

    if num_in_category == 0:
        print(f"No {category.lower()} answers found.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    avg_probabilities = normalize_and_bin_sequences(prob_sequences)
    avg_chosen_probs = normalize_and_bin_sequences(chosen_prob_sequences)
    avg_entropies = normalize_and_bin_sequences(entropy_sequences)
    avg_expert_cot_prob = np.mean(expert_cot_probs) if expert_cot_probs else None
    
    # Debug information for expert CoT probability
    if expert_cot_probs:
        print(f"Expert CoT Probability: Found {len(expert_cot_probs)} samples with expert CoT data")
        print(f"Expert CoT Probability: Min={min(expert_cot_probs):.2e}, Max={max(expert_cot_probs):.2e}, Mean={avg_expert_cot_prob:.2e}")
    else:
        print("Expert CoT Probability: No expert CoT probability data found in samples")

    percentage_steps = list(range(100))
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    lines = []
    
    if prob_sequences:
        line1, = ax1.plot(percentage_steps, avg_probabilities, marker='o', linestyle='-', color='blue', linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
        lines.append(line1)
    if chosen_prob_sequences:
        line2, = ax1.plot(percentage_steps, avg_chosen_probs, marker='s', linestyle='--', color='green', linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)
        lines.append(line2)
    if avg_expert_cot_prob is not None:
        # Format the expert CoT probability value appropriately for very small numbers
        if avg_expert_cot_prob < 1e-10:
            prob_label = f'Avg. Expert CoT Prob ({avg_expert_cot_prob:.2e})'
        else:
            prob_label = f'Avg. Expert CoT Prob ({avg_expert_cot_prob:.6f})'
        line4, = ax1.plot(percentage_steps, [avg_expert_cot_prob] * 100, linestyle='-.', color='purple', linewidth=2.5, label=prob_label, alpha=0.9)
        lines.append(line4)
        
    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax2 = ax1.twinx()
    if entropy_sequences:
        line3, = ax2.plot(percentage_steps, avg_entropies, marker='^', linestyle='-.', color='red', linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
        lines.append(line3)

    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    plt.title(f'Aggregate Metrics on {dataset_name} ({method_name}) - {category} ANSWERS ONLY\n'
              f'Samples: {num_in_category}/{num_total_samples}', fontsize=16)
    
    stats_text = f"Avg. Steps: {avg_steps:.1f}\n{category.capitalize()} Samples: {num_in_category}"
    plt.figtext(1.15, 0.75, stats_text, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen' if is_correct_plot else 'lightcoral', alpha=0.7))

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_{category.lower()}_aggregate_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"{category.capitalize()} answers aggregate plot saved to {full_path}")
    plt.close()
    return full_path


def plot_first_token_prob(data, dataset_name, method_name, output_dir, correctness_filter=None):
    """
    Plots the average probability of the very first chosen token as a bar chart.
    Groups by difficulty level if available (MATH dataset), otherwise shows a single overall bar.
    Works with all datasets, not just MATH.
    
    Args:
        data: Dataset samples
        dataset_name: Name of dataset for titles
        method_name: Method name for titles
        output_dir: Where to save plots
        correctness_filter: None (all), 'correct', or 'incorrect' to filter samples
    """
    filter_str = ""
    if correctness_filter == 'correct':
        filter_str = " (Correct Answers Only)"
        print("Generating bar chart for first token probability (correct answers only)...")
    elif correctness_filter == 'incorrect':
        filter_str = " (Incorrect Answers Only)"
        print("Generating bar chart for first token probability (incorrect answers only)...")
    else:
        print("Generating bar chart for first token probability...")
    
    # Filter data by correctness if requested
    if correctness_filter:
        filtered_data = {}
        for idx, sample in data.items():
            score = sample.get('score')
            is_correct = None
            if isinstance(score, list) and score:
                is_correct = bool(score[0] == 1)
            elif isinstance(score, (int, bool)):
                is_correct = bool(score)
            
            if correctness_filter == 'correct' and is_correct is True:
                filtered_data[idx] = sample
            elif correctness_filter == 'incorrect' and is_correct is False:
                filtered_data[idx] = sample
        data = filtered_data
    
    # Check if this dataset has difficulty levels (like MATH dataset)
    level_groups, has_levels = group_data_by_math_level(data)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    if has_levels:
        # Dataset has difficulty levels (e.g., MATH dataset)
        levels = sorted(level_groups.keys())
        avg_probs, std_probs, labels, sample_counts, accuracies = [], [], [], [], []
        
        for level in levels:
            level_data = level_groups[level]
            if not level_data: continue

            first_token_probs = [s.get('chosen_token_probs', {}).get('epoch_0', [None])[0] for s in level_data.values()]
            first_token_probs = [p for p in first_token_probs if p is not None]
            
            # Calculate accuracy for this level
            num_correct = 0
            total_with_scores = 0
            for sample in level_data.values():
                score = sample.get('score')
                is_correct = None
                if isinstance(score, list) and score:
                    is_correct = bool(score[0] == 1)
                elif isinstance(score, (int, bool)):
                    is_correct = bool(score)
                
                if is_correct is not None:
                    total_with_scores += 1
                    if is_correct:
                        num_correct += 1
            
            accuracy = (num_correct / total_with_scores * 100.0) if total_with_scores > 0 else 0.0
            
            if first_token_probs:
                avg_probs.append(np.mean(first_token_probs))
                std_probs.append(np.std(first_token_probs, ddof=1) if len(first_token_probs) > 1 else 0.0)
                labels.append(f"Level {level}")
                sample_counts.append(len(first_token_probs))
                accuracies.append(accuracy)
        
        if avg_probs:  # Only plot if we have data
            # Calculate y-axis range to emphasize differences (account for std)
            lower_bounds = [a - s for a, s in zip(avg_probs, std_probs)]
            upper_bounds = [a + s for a, s in zip(avg_probs, std_probs)]
            min_prob = min(lower_bounds)
            max_prob = max(upper_bounds)
            prob_range = max_prob - min_prob if max_prob > min_prob else 0.05

            bars = ax.bar(labels, avg_probs, yerr=std_probs, capsize=6, ecolor='black',
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_xlabel("Difficulty Level", fontsize=12)
            # Start from min - 20% of range, but not below 0
            y_min = max(0, min_prob - 0.2 * prob_range - 0.05)
            # End at max + 20% of range, but not above 1.0
            y_max = min(1.0, max_prob + 0.2 * prob_range + 0.08)
            
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                offset = (y_max - y_min) * 0.02  # 2% of visible range
                # Add exact probability value on top of bar
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
                # Add accuracy below the probability value
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5*offset, f"Acc: {accuracies[i]:.1f}%", ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                # Add sample count inside the bar (near bottom)
                ax.text(bar.get_x() + bar.get_width()/2.0, y_min + offset, f"n={sample_counts[i]}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
        else:
            print("No first token probability data found for any difficulty level.")
            return None
    else:
        # Dataset doesn't have difficulty levels (e.g., GSM8K, SVAMP, etc.) or level info is missing
        first_token_probs = [s.get('chosen_token_probs', {}).get('epoch_0', [None])[0] for s in data.values()]
        first_token_probs = [p for p in first_token_probs if p is not None]
        
        if first_token_probs:
            avg_prob = np.mean(first_token_probs)
            # Check if this is MATH dataset but level info is missing
            if dataset_name.lower() == 'math':
                label = 'MATH Dataset (All Levels)'
                print("Note: MATH dataset detected but level information is missing from processed data.")
                print("Showing overall average across all difficulty levels.")
            else:
                label = 'Overall Average'
            
            std_prob = np.std(first_token_probs, ddof=1) if len(first_token_probs) > 1 else 0.0
            # For single bar, use reasonable y-axis range and include std
            y_min = max(0, (avg_prob - std_prob) - 0.10)
            y_max = min(1.0, (avg_prob + std_prob) + 0.10)
            
            bar = ax.bar([label], [avg_prob], yerr=[std_prob], capsize=6, ecolor='black', color='#1f77b4', width=0.4)
            offset = (y_max - y_min) * 0.02
            # Add exact probability value on top of bar
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, avg_prob + offset, f"{avg_prob:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
            # Add sample count inside the bar (near bottom)
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, y_min + offset, f"n={len(first_token_probs)}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
        else:
            print("No first token probability data found.")
            return None
            
    ax.set_ylabel("Average Probability of First Token", fontsize=12)
    ax.set_ylim(y_min, y_max)
    # Use automatic tick generation for better scaling
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f"First Token Chosen Probability on {dataset_name} ({method_name}){filter_str}", fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    if correctness_filter:
        output_filename = f"{dataset_name}_first_token_prob_{correctness_filter}_{method_name}.png"
    else:
        output_filename = f"{dataset_name}_first_token_prob_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"First token probability bar chart saved to {full_path}")
    plt.close()
    return full_path


def plot_last_token_prob(data, dataset_name, method_name, output_dir, correctness_filter=None):
    """
    Plots the average probability of the very last chosen token as a bar chart.
    Groups by difficulty level if available (MATH dataset), otherwise shows a single overall bar.
    Works with all datasets, not just MATH.
    
    Args:
        data: Dataset samples
        dataset_name: Name of dataset for titles
        method_name: Method name for titles
        output_dir: Where to save plots
        correctness_filter: None (all), 'correct', or 'incorrect' to filter samples
    """
    filter_str = ""
    if correctness_filter == 'correct':
        filter_str = " (Correct Answers Only)"
        print("Generating bar chart for last token probability (correct answers only)...")
    elif correctness_filter == 'incorrect':
        filter_str = " (Incorrect Answers Only)"
        print("Generating bar chart for last token probability (incorrect answers only)...")
    else:
        print("Generating bar chart for last token probability...")
    
    # Filter data by correctness if requested
    if correctness_filter:
        filtered_data = {}
        for idx, sample in data.items():
            score = sample.get('score')
            is_correct = None
            if isinstance(score, list) and score:
                is_correct = bool(score[0] == 1)
            elif isinstance(score, (int, bool)):
                is_correct = bool(score)
            
            if correctness_filter == 'correct' and is_correct is True:
                filtered_data[idx] = sample
            elif correctness_filter == 'incorrect' and is_correct is False:
                filtered_data[idx] = sample
        data = filtered_data
    
    # Check if this dataset has difficulty levels (like MATH dataset)
    level_groups, has_levels = group_data_by_math_level(data)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    if has_levels:
        # Dataset has difficulty levels (e.g., MATH dataset)
        levels = sorted(level_groups.keys())
        avg_probs, std_probs, labels, sample_counts, accuracies = [], [], [], [], []
        
        for level in levels:
            level_data = level_groups[level]
            if not level_data: continue

            last_token_probs = [s.get('chosen_token_probs', {}).get('epoch_0', [None])[-1] for s in level_data.values()]
            last_token_probs = [p for p in last_token_probs if p is not None]
            
            # Calculate accuracy for this level
            num_correct = 0
            total_with_scores = 0
            for sample in level_data.values():
                score = sample.get('score')
                is_correct = None
                if isinstance(score, list) and score:
                    is_correct = bool(score[0] == 1)
                elif isinstance(score, (int, bool)):
                    is_correct = bool(score)
                
                if is_correct is not None:
                    total_with_scores += 1
                    if is_correct:
                        num_correct += 1
            
            accuracy = (num_correct / total_with_scores * 100.0) if total_with_scores > 0 else 0.0
            
            if last_token_probs:
                avg_probs.append(np.mean(last_token_probs))
                std_probs.append(np.std(last_token_probs, ddof=1) if len(last_token_probs) > 1 else 0.0)
                labels.append(f"Level {level}")
                sample_counts.append(len(last_token_probs))
                accuracies.append(accuracy)
        
        if avg_probs:  # Only plot if we have data
            # Calculate y-axis range to emphasize differences (account for std)
            lower_bounds = [a - s for a, s in zip(avg_probs, std_probs)]
            upper_bounds = [a + s for a, s in zip(avg_probs, std_probs)]
            min_prob = min(lower_bounds)
            max_prob = max(upper_bounds)
            prob_range = max_prob - min_prob if max_prob > min_prob else 0.05

            bars = ax.bar(labels, avg_probs, yerr=std_probs, capsize=6, ecolor='black',
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_xlabel("Difficulty Level", fontsize=12)
            # Start from min - 20% of range, but not below 0
            y_min = max(0, min_prob - 0.2 * prob_range - 0.05)
            # End at max + 20% of range, but not above 1.0
            y_max = min(1.0, max_prob + 0.2 * prob_range + 0.08)
            
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                offset = (y_max - y_min) * 0.02  # 2% of visible range
                # Add exact probability value on top of bar
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
                # Add accuracy below the probability value
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5*offset, f"Acc: {accuracies[i]:.1f}%", ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                # Add sample count inside the bar (near bottom)
                ax.text(bar.get_x() + bar.get_width()/2.0, y_min + offset, f"n={sample_counts[i]}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
        else:
            print("No last token probability data found for any difficulty level.")
            return None
    else:
        # Dataset doesn't have difficulty levels (e.g., GSM8K, SVAMP, etc.) or level info is missing
        last_token_probs = [s.get('chosen_token_probs', {}).get('epoch_0', [None])[-1] for s in data.values()]
        last_token_probs = [p for p in last_token_probs if p is not None]
        
        if last_token_probs:
            avg_prob = np.mean(last_token_probs)
            # Check if this is MATH dataset but level info is missing
            if dataset_name.lower() == 'math':
                label = 'MATH Dataset (All Levels)'
                print("Note: MATH dataset detected but level information is missing from processed data.")
                print("Showing overall average across all difficulty levels.")
            else:
                label = 'Overall Average'
            
            std_prob = np.std(last_token_probs, ddof=1) if len(last_token_probs) > 1 else 0.0
            # For single bar, use reasonable y-axis range and include std
            y_min = max(0, (avg_prob - std_prob) - 0.10)
            y_max = min(1.0, (avg_prob + std_prob) + 0.10)
            
            bar = ax.bar([label], [avg_prob], yerr=[std_prob], capsize=6, ecolor='black', color='#1f77b4', width=0.4)
            offset = (y_max - y_min) * 0.02
            # Add exact probability value on top of bar
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, avg_prob + offset, f"{avg_prob:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
            # Add sample count inside the bar (near bottom)
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, y_min + offset, f"n={len(last_token_probs)}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
        else:
            print("No last token probability data found.")
            return None

    ax.set_ylabel("Average Probability of Last Token", fontsize=12)
    ax.set_ylim(y_min, y_max)
    # Use automatic tick generation for better scaling
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f"Last Token Chosen Probability on {dataset_name} ({method_name}){filter_str}", fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    if correctness_filter:
        output_filename = f"{dataset_name}_last_token_prob_{correctness_filter}_{method_name}.png"
    else:
        output_filename = f"{dataset_name}_last_token_prob_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Last token probability bar chart saved to {full_path}")
    plt.close()
    return full_path


def calculate_ece(confidences, correct, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: List of confidence values (0-1)
        correct: List of correctness (1 for correct, 0 for incorrect)
        n_bins: Number of bins for ECE calculation
    
    Returns:
        Dictionary with ECE value and bin details
    """
    if not confidences or not correct or len(confidences) != len(correct):
        return {'ece': None, 'bins': []}
    
    # Create bins
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = [
            (conf, corr) for conf, corr in zip(confidences, correct)
            if bin_lower <= conf < bin_upper
        ]
        
        if not in_bin:
            bin_data.append({
                'bin_range': f"[{bin_lower:.2f}, {bin_upper:.2f})",
                'count': 0,
                'avg_confidence': 0.0,
                'avg_accuracy': 0.0,
                'gap': 0.0,
                'weight': 0.0
            })
            continue
        
        bin_confs = [c for c, _ in in_bin]
        bin_corrects = [c for _, c in in_bin]
        
        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_corrects) / len(bin_corrects)
        gap = abs(avg_conf - avg_acc)
        weight = len(in_bin) / len(confidences)
        
        ece += weight * gap
        
        bin_data.append({
            'bin_range': f"[{bin_lower:.2f}, {bin_upper:.2f})",
            'count': len(in_bin),
            'avg_confidence': avg_conf,
            'avg_accuracy': avg_acc,
            'gap': gap,
            'weight': weight
        })
    
    return {'ece': ece, 'bins': bin_data, 'n_samples': len(confidences)}


def plot_correct_vs_incorrect(data, dataset_name, method_name, output_dir, tokenizer=None, data_name="math"):
    """
    Plots the average chosen token probability for correct vs incorrect answers.
    Uses the exact same plotting method as aggregate plots.
    Also calculates and displays ECE (Expected Calibration Error).
    """
    print("Generating correct vs incorrect probability comparison plot...")
    
    # Collect chosen token probabilities for correct and incorrect samples separately
    correct_chosen_sequences = []
    incorrect_chosen_sequences = []
    correct_log_lengths = []
    incorrect_log_lengths = []
    num_correct, num_incorrect, total = 0, 0, 0
    
    # For ECE calculation: collect confidences and correctness
    confidences = []
    correct_labels = []
    
    # Debug counters for ECE
    debug_stats = {
        'total_samples': 0,
        'samples_with_tokenizer': 0,
        'samples_with_answer_text': 0,
        'samples_with_token_data': 0,
        'samples_with_confidence': 0,
        'samples_no_model_output': 0,
        'samples_no_answer': 0,
        'samples_no_tokens': 0
    }
    
    for sample in data.values():
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        
        # Determine if sample is correct or incorrect
        score = sample.get('score')
        is_correct = None
        if isinstance(score, list) and score:
            is_correct = bool(score[0] == 1)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        
        if is_correct is not None:
            total += 1
            if is_correct:
                num_correct += 1
                if chosen_probs_log:
                    correct_chosen_sequences.append(chosen_probs_log)
                    correct_log_lengths.append(len(chosen_probs_log))
            else:
                num_incorrect += 1
                if chosen_probs_log:
                    incorrect_chosen_sequences.append(chosen_probs_log)
                    incorrect_log_lengths.append(len(chosen_probs_log))
            
            # Extract answer confidence for ECE calculation
            # First try reading from file (pre-computed during evaluation)
            answer_confidence = sample.get('answer_confidence')
            
            if answer_confidence is not None:
                # Use pre-computed confidence from file
                confidences.append(answer_confidence)
                correct_labels.append(1 if is_correct else 0)
                debug_stats['samples_with_confidence'] += 1
                debug_stats['samples_with_precomputed'] = debug_stats.get('samples_with_precomputed', 0) + 1
            elif tokenizer is not None:
                # Fallback: compute on-the-fly if not in file (for backward compatibility)
                debug_stats['samples_with_tokenizer'] += 1
                confidence_result = get_answer_confidence(sample, tokenizer=tokenizer, data_name=data_name)
                confidence = confidence_result.get('confidence')
                answer_text = confidence_result.get('answer_text')
                
                # Debug tracking
                if answer_text:
                    debug_stats['samples_with_answer_text'] += 1
                else:
                    debug_stats['samples_no_answer'] += 1
                    # Check why no answer
                    model_output = None
                    for field in ["code", "pred", "output", "model_output"]:
                        if field in sample and sample[field]:
                            model_output = sample[field]
                            break
                    if not model_output:
                        debug_stats['samples_no_model_output'] += 1
                
                # Check token data
                chosen_token_ids = sample.get('chosen_token_ids', {}).get('epoch_0', [])
                chosen_token_probs = sample.get('chosen_token_probs', {}).get('epoch_0', [])
                if not chosen_token_ids or not chosen_token_probs:
                    debug_stats['samples_no_tokens'] += 1
                else:
                    debug_stats['samples_with_token_data'] += 1
                
                if confidence is not None:
                    confidences.append(confidence)
                    correct_labels.append(1 if is_correct else 0)
                    debug_stats['samples_with_confidence'] += 1
            else:
                # No tokenizer and no pre-computed confidence
                debug_stats['samples_no_confidence'] = debug_stats.get('samples_no_confidence', 0) + 1
            
            debug_stats['total_samples'] += 1
    
    if not correct_chosen_sequences:
        print("No correct answers with chosen token probability data found.")
        return None
    if not incorrect_chosen_sequences:
        print("No incorrect answers with chosen token probability data found.")
        return None
    
    # Use the same normalization and binning as aggregate plots
    correct_avg_chosen_probs = normalize_and_bin_sequences(correct_chosen_sequences)
    incorrect_avg_chosen_probs = normalize_and_bin_sequences(incorrect_chosen_sequences)
    
    # Calculate average absolute difference
    correct_avg_chosen_probs = np.array(correct_avg_chosen_probs)
    incorrect_avg_chosen_probs = np.array(incorrect_avg_chosen_probs)
    avg_difference = np.mean(np.abs(correct_avg_chosen_probs - incorrect_avg_chosen_probs))

    # Calculate slope of the gap (correct - incorrect) over generation percentage
    # Use OLS with NaN masking for robustness
    x_full = np.arange(len(correct_avg_chosen_probs))
    valid_mask = (~np.isnan(correct_avg_chosen_probs)) & (~np.isnan(incorrect_avg_chosen_probs))
    if np.count_nonzero(valid_mask) >= 2:
        x = x_full[valid_mask].astype(float)
        gap = (correct_avg_chosen_probs - incorrect_avg_chosen_probs)[valid_mask].astype(float)
        X = np.vstack([x, np.ones_like(x)]).T
        slope_gap, _intercept = np.linalg.lstsq(X, gap, rcond=None)[0]
    else:
        slope_gap = np.nan
    
    # Calculate average steps
    correct_avg_steps = np.mean(correct_log_lengths) if correct_log_lengths else 0
    incorrect_avg_steps = np.mean(incorrect_log_lengths) if incorrect_log_lengths else 0
    
    # Use the same plotting style as aggregate plots
    percentage_steps = list(range(100))
    avg_acc = (num_correct / total * 100.0) if total > 0 else None
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    lines = []
    
    # Plot correct and incorrect chosen token probabilities using the same style as aggregate
    line1, = ax1.plot(percentage_steps, correct_avg_chosen_probs, marker='o', linestyle='-', 
                      color='green', linewidth=2, markersize=4, 
                      label=f'Correct Chosen Token Prob (n={len(correct_chosen_sequences)})', alpha=0.8)
    lines.append(line1)
    
    line2, = ax1.plot(percentage_steps, incorrect_avg_chosen_probs, marker='s', linestyle='--', 
                      color='red', linewidth=2, markersize=4, 
                      label=f'Incorrect Chosen Token Prob (n={len(incorrect_chosen_sequences)})', alpha=0.8)
    lines.append(line2)
    
    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Chosen Token Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11)
    
    base_title = f'Correct vs Incorrect Chosen Token Probability on {dataset_name} ({method_name})'
    if avg_acc is not None:
        base_title += f' — Avg Acc: {avg_acc:.1f}%'
    plt.title(base_title, fontsize=16)
    
    # Calculate ECE if we have confidences
    ece_value = None
    ece_info = ""
    ece_debug_text = ""  # Debug info to show on plot
    
    if confidences and len(confidences) > 0:
        ece_result = calculate_ece(confidences, correct_labels, n_bins=10)
        ece_value = ece_result['ece']
        if ece_value is not None:
            ece_info = f" | ECE: {ece_value:.4f}"
            print(f"✓ ECE calculated: {ece_value:.4f} (based on {len(confidences)} samples with answer confidence)")
    else:
        # Show "NA" for ECE when not calculated
        ece_info = " | ECE: NA"
        
        # Build debug text to show on plot
        precomputed_count = debug_stats.get('samples_with_precomputed', 0)
        if precomputed_count > 0:
            # Using pre-computed confidence from file
            ece_debug_text = f"\nECE Debug: Using pre-computed confidence ({precomputed_count}/{debug_stats.get('total_samples', 0)})"
        elif tokenizer is None:
            ece_debug_text = "\nECE Debug: No tokenizer & no pre-computed confidence"
        else:
            # Tokenizer exists but couldn't extract confidences
            debug_parts = []
            total = debug_stats.get('total_samples', 0)
            conf_count = debug_stats.get('samples_with_confidence', 0)
            ans_count = debug_stats.get('samples_with_answer_text', 0)
            token_count = debug_stats.get('samples_with_token_data', 0)
            
            if total > 0:
                if conf_count == 0:
                    debug_parts.append(f"Conf: 0/{total}")
                else:
                    debug_parts.append(f"Conf: {conf_count}/{total}")
                
                if ans_count < total:
                    debug_parts.append(f"Ans: {ans_count}/{total}")
                
                if token_count < total:
                    debug_parts.append(f"Tokens: {token_count}/{total}")
            
            if debug_parts:
                ece_debug_text = "\nECE Debug: " + " | ".join(debug_parts)
            else:
                ece_debug_text = "\nECE Debug: Tokenizer OK but no confidences extracted"
        
        # Also print to console for detailed debugging
        print("\n" + "="*60)
        print("ECE CALCULATION DEBUG INFO")
        print("="*60)
        if tokenizer is None:
            print("❌ Tokenizer: NOT PROVIDED")
            print("   → Solution: Provide --model_name argument to enable ECE calculation")
        else:
            print("✓ Tokenizer: LOADED")
            print(f"   Tokenizer type: {type(tokenizer).__name__}")
            print(f"\n📊 Sample Statistics:")
            print(f"   Total samples processed: {debug_stats['total_samples']}")
            print(f"   Samples with tokenizer: {debug_stats['samples_with_tokenizer']}")
            print(f"   Samples with answer text extracted: {debug_stats['samples_with_answer_text']}")
            print(f"   Samples with token data (ids + probs): {debug_stats['samples_with_token_data']}")
            print(f"   Samples with confidence calculated: {debug_stats['samples_with_confidence']}")
            print(f"\n⚠ Issues found:")
            if debug_stats['samples_no_model_output'] > 0:
                print(f"   - {debug_stats['samples_no_model_output']} samples missing model output (no 'code', 'pred', 'output', or 'model_output' field)")
            if debug_stats['samples_no_answer'] > 0:
                print(f"   - {debug_stats['samples_no_answer']} samples where answer could not be extracted from model output")
            if debug_stats['samples_no_tokens'] > 0:
                print(f"   - {debug_stats['samples_no_tokens']} samples missing token data (no 'chosen_token_ids' or 'chosen_token_probs')")
            
            if debug_stats['samples_with_confidence'] == 0:
                print(f"\n❌ Result: Could not extract answer confidence for any samples")
                print(f"   → Check that samples have 'chosen_token_probs' and 'chosen_token_ids' fields")
                print(f"   → Check that model output contains extractable answers")
            else:
                print(f"\n⚠ Result: Only {debug_stats['samples_with_confidence']}/{debug_stats['total_samples']} samples had confidence extracted")
        print("="*60 + "\n")
    
    stats_text = f"Correct Avg Steps: {correct_avg_steps:.1f} | Incorrect Avg Steps: {incorrect_avg_steps:.1f}"
    if avg_acc is not None:
        stats_text += f"\nAvg Accuracy: {avg_acc:.1f}% | Avg Abs Difference: {avg_difference:.3f} | Slope (Correct-Incorrect): {slope_gap:.4f}{ece_info}{ece_debug_text}"
    plt.figtext(0.5, 0.02, stats_text, fontsize=11, ha='center', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_correct_vs_incorrect_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Correct vs incorrect comparison plot saved to {full_path}")
    plt.close()
    return full_path


# --- (Other plotting functions like plot_single_question, plot_path_of_distributions, etc. would go here) ---
# ... (These are omitted for brevity but should be included in your final file) ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot probability data from JSONL results.")
    parser.add_argument("jsonl_file", help="Path to the input JSONL file.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset (e.g., GSM8K) for titles and filenames.")
    parser.add_argument("--method_name", required=True, help="Name of the method or model (e.g., CoT, Llama3-8B) for filenames.")
    parser.add_argument("--output_dir", default="plots", help="Directory to save the plots.")
    parser.add_argument("--plot_type",
                        choices=['aggregate', 'single', 'path_aggregate', 'path_single',
                                 'correct_aggregate', 'incorrect_aggregate',
                                 'level_single', 'level_aggregate',
                                 'first_token_prob', 'last_token_prob', 
                                 'first_token_prob_correct', 'first_token_prob_incorrect',
                                 'last_token_prob_correct', 'last_token_prob_incorrect',
                                 "starting_tokens_by_level", "ending_tokens_by_level",
                                 'correct_vs_incorrect'],
                        required=True, help="Type of plot to generate.")
    parser.add_argument("--sample_id", type=int, help="The 'idx' of the sample to plot (required for single plot types).")
    parser.add_argument("--math_level", type=str, help="Math difficulty level (1-5) for level-based plots.")
    parser.add_argument("--truncation_file", type=str, help="Path to truncation analysis JSONL file (optional).")
    parser.add_argument("--model_name", type=str, help="Model name/path for tokenizer (required for ECE calculation in correct_vs_incorrect plot).")
    parser.add_argument("--data_name", type=str, default="math", help="Dataset name for answer extraction (default: math).")

    args = parser.parse_args()

    all_data = load_data(args.jsonl_file)
    
    # Load tokenizer if model_name is provided (needed for ECE calculation)
    tokenizer = None
    if args.model_name:
        print(f"\n{'='*60}")
        print(f"TOKENIZER LOADING")
        print(f"{'='*60}")
        print(f"Model name provided: {args.model_name}")
        try:
            from transformers import AutoTokenizer
            print(f"Attempting to load tokenizer from: {args.model_name}")
            # Try loading tokenizer - runs on CPU, no GPU needed
            # First try with local files only (if cached), then try downloading if needed
            try:
                # Try local files first (faster, works offline)
                print("  → Trying local cache first (offline mode)...")
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_name, 
                    trust_remote_code=True,
                    local_files_only=True
                )
                print("  ✓ Tokenizer loaded from local cache")
            except Exception as local_error:
                # If local files not found, try downloading (requires internet)
                print(f"  → Local cache not found")
                print(f"  → Error: {str(local_error)[:100]}...")
                print(f"  → Trying to download (requires internet)...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.model_name, 
                        trust_remote_code=True,
                        local_files_only=False
                    )
                    print("  ✓ Tokenizer downloaded and loaded successfully")
                except Exception as download_error:
                    print(f"  ❌ Download also failed: {str(download_error)[:200]}")
                    raise download_error
        except Exception as e:
            print(f"  ❌ FAILED to load tokenizer")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)[:300]}")
            print(f"  ECE calculation will be skipped. Continuing without tokenizer...")
            print(f"  Note: Tokenizers don't require GPU, but may need internet access or cached files.")
            tokenizer = None
        print(f"{'='*60}\n")
    else:
        print(f"\n⚠ No model_name provided - ECE calculation will be skipped")
        print(f"  To enable ECE, provide --model_name argument\n")

    if args.plot_type == 'aggregate':
        plot_average_probability_by_step(all_data, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'correct_aggregate':
        plot_correct_or_incorrect_aggregate(all_data, args.dataset_name, args.method_name, args.output_dir, is_correct_plot=True)
    elif args.plot_type == 'incorrect_aggregate':
        plot_correct_or_incorrect_aggregate(all_data, args.dataset_name, args.method_name, args.output_dir, is_correct_plot=False)
    elif args.plot_type == 'first_token_prob' or args.plot_type == 'starting_tokens_by_level':
        plot_first_token_prob(all_data, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'first_token_prob_correct':
        plot_first_token_prob(all_data, args.dataset_name, args.method_name, args.output_dir, correctness_filter='correct')
    elif args.plot_type == 'first_token_prob_incorrect':
        plot_first_token_prob(all_data, args.dataset_name, args.method_name, args.output_dir, correctness_filter='incorrect')
    elif args.plot_type == 'last_token_prob' or args.plot_type == 'ending_tokens_by_level':
        plot_last_token_prob(all_data, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'last_token_prob_correct':
        plot_last_token_prob(all_data, args.dataset_name, args.method_name, args.output_dir, correctness_filter='correct')
    elif args.plot_type == 'last_token_prob_incorrect':
        plot_last_token_prob(all_data, args.dataset_name, args.method_name, args.output_dir, correctness_filter='incorrect')
    elif args.plot_type == 'correct_vs_incorrect':
        if tokenizer is None:
            print("Warning: No tokenizer provided. ECE calculation will be skipped.")
            print("To calculate ECE, provide --model_name argument.")
        plot_correct_vs_incorrect(all_data, args.dataset_name, args.method_name, args.output_dir, 
                                 tokenizer=tokenizer, data_name=args.data_name)
    else:
        # --- This block handles all the other plot types that were not changed ---
        if args.plot_type == 'single':
            if args.sample_id is None:
                print("Error: --sample_id is required for plot_type 'single'.")
            elif args.sample_id not in all_data:
                print(f"Error: Sample with index {args.sample_id} not found in the data.")
            else:
                # plot_single_question_probability(all_data[args.sample_id], args.dataset_name, args.method_name, args.output_dir)
                print("Single plot function call placeholder.") # Replace with your actual function
        
        # ... and so on for all other original plot types like 'path_aggregate', 'level_single', etc.
        # I have omitted the full code for these for clarity, but you should have them in your file.
        print(f"Plot type '{args.plot_type}' is handled by other functions not shown in this snippet.")