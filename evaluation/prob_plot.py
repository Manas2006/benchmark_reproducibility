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
        avg_probs, labels, sample_counts, accuracies = [], [], [], []
        
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
                labels.append(f"Level {level}")
                sample_counts.append(len(first_token_probs))
                accuracies.append(accuracy)
        
        if avg_probs:  # Only plot if we have data
            bars = ax.bar(labels, avg_probs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_xlabel("Difficulty Level", fontsize=12)
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                # Add exact probability value on top of bar
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
                # Add accuracy below the probability value
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"Acc: {accuracies[i]:.1f}%", ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                # Add sample count below the bar
                ax.text(bar.get_x() + bar.get_width()/2.0, yval - 0.05, f"n={sample_counts[i]}", ha='center', va='top', fontsize=9, color='gray')
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
            
            bar = ax.bar([label], [avg_prob], color='#1f77b4', width=0.4)
            # Add exact probability value on top of bar
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, avg_prob + 0.02, f"{avg_prob:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
            # Add sample count below the bar
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, avg_prob - 0.05, f"n={len(first_token_probs)}", ha='center', va='top', fontsize=9, color='gray')
        else:
            print("No first token probability data found.")
            return None
            
    ax.set_ylabel("Average Probability of First Token", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
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
        avg_probs, labels, sample_counts, accuracies = [], [], [], []
        
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
                labels.append(f"Level {level}")
                sample_counts.append(len(last_token_probs))
                accuracies.append(accuracy)
        
        if avg_probs:  # Only plot if we have data
            bars = ax.bar(labels, avg_probs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_xlabel("Difficulty Level", fontsize=12)
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                # Add exact probability value on top of bar
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
                # Add accuracy below the probability value
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"Acc: {accuracies[i]:.1f}%", ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                # Add sample count below the bar
                ax.text(bar.get_x() + bar.get_width()/2.0, yval - 0.05, f"n={sample_counts[i]}", ha='center', va='top', fontsize=9, color='gray')
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
            
            bar = ax.bar([label], [avg_prob], color='#1f77b4', width=0.4)
            # Add exact probability value on top of bar
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, avg_prob + 0.02, f"{avg_prob:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
            # Add sample count below the bar
            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, avg_prob - 0.05, f"n={len(last_token_probs)}", ha='center', va='top', fontsize=9, color='gray')
        else:
            print("No last token probability data found.")
            return None

    ax.set_ylabel("Average Probability of Last Token", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
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


def plot_correct_vs_incorrect(data, dataset_name, method_name, output_dir):
    """
    Plots the average chosen token probability for correct vs incorrect answers.
    Uses the exact same plotting method as aggregate plots.
    """
    print("Generating correct vs incorrect probability comparison plot...")
    
    # Collect chosen token probabilities for correct and incorrect samples separately
    correct_chosen_sequences = []
    incorrect_chosen_sequences = []
    correct_log_lengths = []
    incorrect_log_lengths = []
    num_correct, num_incorrect, total = 0, 0, 0
    
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
    
    if not correct_chosen_sequences:
        print("No correct answers with chosen token probability data found.")
        return None
    if not incorrect_chosen_sequences:
        print("No incorrect answers with chosen token probability data found.")
        return None
    
    # Use the same normalization and binning as aggregate plots
    correct_avg_chosen_probs = normalize_and_bin_sequences(correct_chosen_sequences)
    incorrect_avg_chosen_probs = normalize_and_bin_sequences(incorrect_chosen_sequences)
    
    # Calculate average difference
    correct_avg_chosen_probs = np.array(correct_avg_chosen_probs)
    incorrect_avg_chosen_probs = np.array(incorrect_avg_chosen_probs)
    avg_difference = np.mean(correct_avg_chosen_probs - incorrect_avg_chosen_probs)
    
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
    
    stats_text = f"Correct Avg Steps: {correct_avg_steps:.1f} | Incorrect Avg Steps: {incorrect_avg_steps:.1f}"
    if avg_acc is not None:
        stats_text += f"\nAvg Accuracy: {avg_acc:.1f}% | Avg Difference: {avg_difference:.3f}"
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

    args = parser.parse_args()

    all_data = load_data(args.jsonl_file)

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
        plot_correct_vs_incorrect(all_data, args.dataset_name, args.method_name, args.output_dir)
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