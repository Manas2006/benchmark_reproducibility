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
   - starting_tokens_by_level: First N tokens probability by difficulty
   - ending_tokens_by_level: Last N tokens probability by difficulty
5. PATH OF DISTRIBUTIONS: Visualize model vs gold path trajectories
6. TRUNCATION ANALYSIS INTEGRATION: Compare with truncation experiment results

NEW FEATURES (Latest Update):
------------------------------
- Starting/Ending Token Analysis: For MATH dataset, analyze first/last N tokens
  by difficulty level with color-coded lines (blue=Level 1, orange=Level 2, 
  green=Level 3, red=Level 4, purple=Level 5)
  
- Truncation Data Integration: Automatically fixes GT CoT data and adds truncation
  analysis lines to correct/incorrect aggregate plots showing how truncated CoT
  affects probability metrics

USAGE EXAMPLES:
--------------
# Generate starting tokens plot for MATH dataset
python prob_plot.py data.jsonl --dataset_name MATH --method_name auto-cot \\
    --plot_type starting_tokens_by_level --num_tokens 20

# Generate ending tokens plot for MATH dataset  
python prob_plot.py data.jsonl --dataset_name MATH --method_name auto-cot \\
    --plot_type ending_tokens_by_level --num_tokens 20

# Generate correct answers plot with truncation analysis
python prob_plot.py data.jsonl --dataset_name GSM8K --method_name auto-cot \\
    --plot_type correct_aggregate --truncation_file truncation_data.jsonl
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
    
    Args:
        sequences: List of sequences (e.g., list where each item is a list of probabilities)
        num_bins: Number of bins to create (default 100 for 0-100% completion)
    
    Returns:
        List of 100 average values, with np.nan for empty bins
    """
    if not sequences:
        return [np.nan] * num_bins
    
    # Create empty bins
    bins = [[] for _ in range(num_bins)]
    
    # Process each sequence
    for sequence in sequences:
        if not sequence or len(sequence) == 0:
            continue
            
        # For each value in the sequence, calculate which bin it belongs to
        for i, value in enumerate(sequence):
            if value is not None:
                # Calculate bin index: percentage of completion
                bin_index = int((i / len(sequence)) * num_bins)
                # Ensure we don't exceed the number of bins
                bin_index = min(bin_index, num_bins - 1)
                bins[bin_index].append(value)
    
    # Calculate average for each bin
    averages = []
    for bin_values in bins:
        if bin_values:
            averages.append(np.mean(bin_values))
        else:
            averages.append(np.nan)
    
    return averages


def filter_data_by_math_level(data, target_level):
    """Filter data to only include samples with the specified math level."""
    filtered_data = {}
    target_level_int = int(target_level)
    
    for idx, sample in data.items():
        # Check if sample has level information
        level_str = sample.get('level')
        if level_str:
            level_num = parse_math_level(level_str)
            if level_num == target_level_int:
                filtered_data[idx] = sample
    
    return filtered_data


def group_data_by_math_level(data):
    """Group data by math difficulty level."""
    level_groups = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    
    for idx, sample in data.items():
        level_str = sample.get('level')
        if level_str:
            level_num = parse_math_level(level_str)
            if level_num and level_num in level_groups:
                level_groups[level_num][idx] = sample
    
    return level_groups


def plot_math_level_single(data, dataset_name, method_name, output_dir, math_level):
    """
    Plot probability metrics for a single math difficulty level.
    Similar to aggregate plot but only for samples of a specific level.
    """
    print(f"Generating plot for Math Level {math_level}...")
    
    # Filter data by math level
    level_data = filter_data_by_math_level(data, math_level)
    
    if not level_data:
        print(f"No data found for Math Level {math_level}")
        return None
    
    print(f"Found {len(level_data)} samples for Level {math_level}")
    
    # Use the existing aggregate plotting function with filtered data
    # But modify the title and filename to indicate the level
    level_dataset_name = f"{dataset_name}_Level_{math_level}"
    
    # Collect sequences for each metric from this level
    prob_sequences = []
    chosen_prob_sequences = []
    entropy_sequences = []
    log_lengths = []  # To calculate the average number of steps
    num_correct = 0
    total = 0

    for sample in level_data.values():
        prob_log = sample.get('probability_log', {}).get('epoch_0', [])
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        entropies_log = sample.get('entropies', {}).get('epoch_0', [])
        
        if prob_log:
            log_lengths.append(len(prob_log))
            prob_sequences.append(prob_log)
        
        if chosen_probs_log:
            chosen_prob_sequences.append(chosen_probs_log)
                
        if entropies_log:
            entropy_sequences.append(entropies_log)
        
        # Accuracy collection (expects 'score' or 'is_correct')
        score = sample.get('score')
        if isinstance(score, list) and len(score) > 0:
            is_correct = bool(score[0] == 1 or score[0] == True)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        else:
            is_correct = None
        if is_correct is not None:
            total += 1
            if is_correct:
                num_correct += 1

    if not prob_sequences and not chosen_prob_sequences and not entropy_sequences:
        print(f"No probability data found for Level {math_level}.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    
    # Use normalize_and_bin_sequences to create percentage-based averages
    avg_probabilities = normalize_and_bin_sequences(prob_sequences)
    avg_chosen_probs = normalize_and_bin_sequences(chosen_prob_sequences)
    avg_entropies = normalize_and_bin_sequences(entropy_sequences)

    # Create percentage-based x-axis (0-100%)
    percentage_steps = list(range(100))

    # Compute dataset avg accuracy if available
    avg_acc = (num_correct / total * 100.0) if total > 0 else None

    # Create the plot with dual y-axis for entropy
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot probabilities on left y-axis
    if prob_sequences:
        line1 = ax1.plot(percentage_steps, avg_probabilities, marker='o', linestyle='-', color='blue', 
                        linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
    if chosen_prob_sequences:
        line2 = ax1.plot(percentage_steps, avg_chosen_probs, marker='s', linestyle='--', color='green', 
                        linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)
    
    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    if entropy_sequences:
        line3 = ax2.plot(percentage_steps, avg_entropies, marker='^', linestyle='-.', color='red', 
                        linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
    
    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends from both axes
    lines = []
    labels = []
    if prob_sequences:
        lines.extend(line1)
        labels.append('Correct Token Probability')
    if chosen_prob_sequences:
        lines.extend(line2)
        labels.append('Chosen Token Probability')
    if entropy_sequences:
        lines.extend(line3)
        labels.append('Next Token Entropy')
    
    ax1.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    # Title with dataset name and method
    base_title = f'Generation Metrics vs. Step on {dataset_name} Level {math_level} ({method_name})'
    if avg_acc is not None:
        base_title += f' — Avg Acc: {avg_acc:.1f}%'
    plt.title(base_title, fontsize=16)

    # Add statistics text below the legend
    stats_text = f"Avg. Steps: {avg_steps:.1f} | Samples: {len(level_data)}"
    if avg_acc is not None:
        stats_text += f" | Avg Acc: {avg_acc:.1f}%"
    plt.figtext(0.99, 0.85, stats_text, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_level_{math_level}_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Math Level {math_level} plot saved to {full_path}")
    plt.close()
    return full_path


def plot_math_level_aggregate(data, dataset_name, method_name, output_dir):
    """
    Generate 3 separate plots showing all 5 difficulty levels:
    1. Correct Token Probability vs Step
    2. Chosen Token Probability vs Step  
    3. Entropy vs Step
    Returns a ZIP file containing all three plots.
    """
    print("Generating Math Level Aggregate plots (3 separate plots)...")
    
    # Group data by math level
    level_groups = group_data_by_math_level(data)
    
    # Check that we have data for each level
    available_levels = []
    for level in [1, 2, 3, 4, 5]:
        if level_groups[level]:
            available_levels.append(level)
            print(f"Level {level}: {len(level_groups[level])} samples")
        else:
            print(f"Level {level}: No samples found")
    
    if not available_levels:
        print("No data found for any math levels")
        return None
    
    # Colors for different levels
    level_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}
    level_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    # Process each level to get metrics
    level_metrics = {}
    for level in available_levels:
        level_data = level_groups[level]
        
        step_probs = {}
        step_chosen_probs = {}
        step_entropies = {}
        log_lengths = []
        num_correct = 0
        total = 0
        
        for sample in level_data.values():
            prob_log = sample.get('probability_log', {}).get('epoch_0', [])
            chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
            entropies_log = sample.get('entropies', {}).get('epoch_0', [])
            
            if prob_log:
                log_lengths.append(len(prob_log))
                for step, prob in enumerate(prob_log):
                    step_probs.setdefault(step, []).append(prob)
            
            if chosen_probs_log:
                for step, prob in enumerate(chosen_probs_log):
                    step_chosen_probs.setdefault(step, []).append(prob)
                    
            if entropies_log:
                for step, entropy in enumerate(entropies_log):
                    step_entropies.setdefault(step, []).append(entropy)
            
            # Accuracy collection
            score = sample.get('score')
            if isinstance(score, list) and len(score) > 0:
                is_correct = bool(score[0] == 1 or score[0] == True)
            elif isinstance(score, (int, bool)):
                is_correct = bool(score)
            else:
                is_correct = None
            if is_correct is not None:
                total += 1
                if is_correct:
                    num_correct += 1
        
        # Collect sequences for percentage-based averaging
        prob_sequences = []
        chosen_prob_sequences = []
        entropy_sequences = []
        
        for sample in level_data.values():
            prob_log = sample.get('probability_log', {}).get('epoch_0', [])
            chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
            entropies_log = sample.get('entropies', {}).get('epoch_0', [])
            
            if prob_log:
                prob_sequences.append(prob_log)
            if chosen_probs_log:
                chosen_prob_sequences.append(chosen_probs_log)
            if entropies_log:
                entropy_sequences.append(entropies_log)
        
        # Use normalize_and_bin_sequences to create percentage-based averages
        avg_probabilities = normalize_and_bin_sequences(prob_sequences)
        avg_chosen_probs = normalize_and_bin_sequences(chosen_prob_sequences)
        avg_entropies = normalize_and_bin_sequences(entropy_sequences)
        
        # Create percentage-based x-axis (0-100%)
        percentage_steps = list(range(100))
        
        level_metrics[level] = {
            'steps': percentage_steps,
            'avg_probabilities': avg_probabilities,
            'avg_chosen_probs': avg_chosen_probs,
            'avg_entropies': avg_entropies,
            'avg_steps': np.mean(log_lengths) if log_lengths else 0,
            'avg_acc': (num_correct / total * 100.0) if total > 0 else None,
            'sample_count': len(level_data)
        }
    
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    # Plot 1: Correct Token Probability vs Step
    fig, ax = plt.subplots(figsize=(14, 8))
    for level in available_levels:
        metrics = level_metrics[level]
        if metrics['avg_probabilities'] and not all(np.isnan(metrics['avg_probabilities'])):
            ax.plot(metrics['steps'], metrics['avg_probabilities'], 
                   marker=level_markers[level], linestyle='-', color=level_colors[level], 
                   linewidth=2, markersize=4, alpha=0.8, 
                   label=f'Level {level} (n={metrics["sample_count"]})')
    
    ax.set_xlabel('Response Completion (%)', fontsize=12)
    ax.set_ylabel('Correct Token Probability', fontsize=12)
    ax.set_xlim(left=0, right=99)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11)
    plt.title(f'Correct Token Probability vs. Response Completion by Difficulty Level\n{dataset_name} ({method_name})', fontsize=16)
    
    plot1_path = os.path.join(output_dir, f"{dataset_name}_levels_correct_token_{method_name}.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plot_files.append(plot1_path)
    plt.close()
    
    # Plot 2: Chosen Token Probability vs Step
    fig, ax = plt.subplots(figsize=(14, 8))
    for level in available_levels:
        metrics = level_metrics[level]
        if metrics['avg_chosen_probs'] and not all(np.isnan(metrics['avg_chosen_probs'])):
            ax.plot(metrics['steps'], metrics['avg_chosen_probs'], 
                   marker=level_markers[level], linestyle='--', color=level_colors[level], 
                   linewidth=2, markersize=4, alpha=0.8, 
                   label=f'Level {level} (n={metrics["sample_count"]})')
    
    ax.set_xlabel('Response Completion (%)', fontsize=12)
    ax.set_ylabel('Chosen Token Probability', fontsize=12)
    ax.set_xlim(left=0, right=99)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11)
    plt.title(f'Chosen Token Probability vs. Response Completion by Difficulty Level\n{dataset_name} ({method_name})', fontsize=16)
    
    plot2_path = os.path.join(output_dir, f"{dataset_name}_levels_chosen_token_{method_name}.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plot_files.append(plot2_path)
    plt.close()
    
    # Plot 3: Entropy vs Step
    fig, ax = plt.subplots(figsize=(14, 8))
    for level in available_levels:
        metrics = level_metrics[level]
        if metrics['avg_entropies'] and not all(np.isnan(metrics['avg_entropies'])):
            ax.plot(metrics['steps'], metrics['avg_entropies'], 
                   marker=level_markers[level], linestyle='-.', color=level_colors[level], 
                   linewidth=2, markersize=4, alpha=0.8, 
                   label=f'Level {level} (n={metrics["sample_count"]})')
    
    ax.set_xlabel('Response Completion (%)', fontsize=12)
    ax.set_ylabel('Next Token Entropy (nats)', fontsize=12)
    ax.set_xlim(left=0, right=99)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11)
    plt.title(f'Next Token Entropy vs. Response Completion by Difficulty Level\n{dataset_name} ({method_name})', fontsize=16)
    
    plot3_path = os.path.join(output_dir, f"{dataset_name}_levels_entropy_{method_name}.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plot_files.append(plot3_path)
    plt.close()
    
    # Create ZIP file containing all plots
    import zipfile
    zip_path = os.path.join(output_dir, f"{dataset_name}_level_aggregate_{method_name}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for plot_file in plot_files:
            zipf.write(plot_file, os.path.basename(plot_file))
    
    print(f"Math Level Aggregate plots saved to ZIP: {zip_path}")
    print(f"Created {len(plot_files)} plots for levels: {available_levels}")
    
    # Clean up individual plot files (optional - keep them for now)
    # for plot_file in plot_files:
    #     os.remove(plot_file)
    
    return zip_path


def plot_average_probability_by_step(data, dataset_name, method_name, output_dir):
    """
    Calculates and plots the average probability of the target token, chosen token probability,
    and entropy at each generation step, averaged across all samples in the dataset.
    Uses percentage-based x-axis (0-100% completion) to normalize for different response lengths.
    Also shows avg accuracy across dataset if available.
    """
    print("Generating aggregate plot of average metrics per step (percentage-based)...")

    # Collect sequences for each metric
    prob_sequences = []
    chosen_prob_sequences = []
    entropy_sequences = []
    log_lengths = []  # To calculate the average number of steps
    num_correct = 0
    total = 0

    for sample in data.values():
        prob_log = sample.get('probability_log', {}).get('epoch_0', [])
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        entropies_log = sample.get('entropies', {}).get('epoch_0', [])
        
        if prob_log:
            log_lengths.append(len(prob_log))
            prob_sequences.append(prob_log)
        
        if chosen_probs_log:
            chosen_prob_sequences.append(chosen_probs_log)
                
        if entropies_log:
            entropy_sequences.append(entropies_log)
        
        # Accuracy collection (expects 'score' or 'is_correct')
        # In our JSONL, 'score' may be a list or single value; handle simply
        score = sample.get('score')
        if isinstance(score, list) and len(score) > 0:
            is_correct = bool(score[0] == 1 or score[0] == True)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        else:
            is_correct = None
        if is_correct is not None:
            total += 1
            if is_correct:
                num_correct += 1

    if not prob_sequences and not chosen_prob_sequences and not entropy_sequences:
        print("No probability data found to plot.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    
    # Use normalize_and_bin_sequences to create percentage-based averages
    avg_probabilities = normalize_and_bin_sequences(prob_sequences)
    avg_chosen_probs = normalize_and_bin_sequences(chosen_prob_sequences)
    avg_entropies = normalize_and_bin_sequences(entropy_sequences)

    # Create percentage-based x-axis (0-100%)
    percentage_steps = list(range(100))

    # Compute dataset avg accuracy if available
    avg_acc = (num_correct / total * 100.0) if total > 0 else None

    # Create the plot with dual y-axis for entropy
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot probabilities on left y-axis
    if prob_sequences:
        line1 = ax1.plot(percentage_steps, avg_probabilities, marker='o', linestyle='-', color='blue', 
                        linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
    if chosen_prob_sequences:
        line2 = ax1.plot(percentage_steps, avg_chosen_probs, marker='s', linestyle='--', color='green', 
                        linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)
    
    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    if entropy_sequences:
        line3 = ax2.plot(percentage_steps, avg_entropies, marker='^', linestyle='-.', color='red', 
                        linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
    
    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends from both axes
    lines = []
    labels = []
    if prob_sequences:
        lines.extend(line1)
        labels.append('Correct Token Probability')
    if chosen_prob_sequences:
        lines.extend(line2)
        labels.append('Chosen Token Probability')
    if entropy_sequences:
        lines.extend(line3)
        labels.append('Next Token Entropy')
    
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11)

    # Title with dataset name and method
    base_title = f'Generation Metrics vs. Completion % on {dataset_name} ({method_name})'
    if avg_acc is not None:
        base_title += f' — Avg Acc: {avg_acc:.1f}%'
    plt.title(base_title, fontsize=16)

    # Add statistics text below the plot in separate rows
    stats_text = f"Avg. Steps: {avg_steps:.1f}"
    if avg_acc is not None:
        stats_text += f"\nAvg Accuracy: {avg_acc:.1f}%"
    
    plt.figtext(0.5, 0.02, stats_text, fontsize=11, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_aggregate_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {full_path}")
    plt.close()
    return full_path


def plot_correct_answers_aggregate(data, dataset_name, method_name, output_dir, truncation_data=None):
    """
    Calculates and plots the average probability of the target token, chosen token probability,
    and entropy at each generation step, averaged across only the CORRECT samples in the dataset.
    Uses percentage-based x-axis (0-100% completion) to normalize for different response lengths.
    
    Args:
        truncation_data: Optional dict with truncation analysis data for model CoT originally correct
    """
    print("Generating aggregate plot of average metrics per step for CORRECT answers only (percentage-based)...")

    # Collect sequences for each metric from correct samples only
    prob_sequences = []
    chosen_prob_sequences = []
    entropy_sequences = []
    log_lengths = []
    num_correct = 0
    num_total_samples = 0

    for sample in data.values():
        num_total_samples += 1
        
        # Determine correctness first
        score = sample.get('score')
        if isinstance(score, list) and len(score) > 0:
            is_correct = bool(score[0] == 1 or score[0] == True)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        else:
            is_correct = None
        
        # Only process if this sample is correct
        if is_correct is not True:
            continue
            
        num_correct += 1
        
        prob_log = sample.get('probability_log', {}).get('epoch_0', [])
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        entropies_log = sample.get('entropies', {}).get('epoch_0', [])
        
        if prob_log:
            log_lengths.append(len(prob_log))
            prob_sequences.append(prob_log)
        
        if chosen_probs_log:
            chosen_prob_sequences.append(chosen_probs_log)
                
        if entropies_log:
            entropy_sequences.append(entropies_log)

    if not prob_sequences and not chosen_prob_sequences and not entropy_sequences:
        print("No probability data found for correct answers to plot.")
        return None

    if num_correct == 0:
        print("No correct answers found in the dataset.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    
    # Use normalize_and_bin_sequences to create percentage-based averages
    avg_probabilities = normalize_and_bin_sequences(prob_sequences)
    avg_chosen_probs = normalize_and_bin_sequences(chosen_prob_sequences)
    avg_entropies = normalize_and_bin_sequences(entropy_sequences)

    # Create percentage-based x-axis (0-100%)
    percentage_steps = list(range(100))

    # Create the plot with dual y-axis for entropy
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot probabilities on left y-axis
    if prob_sequences:
        line1 = ax1.plot(percentage_steps, avg_probabilities, marker='o', linestyle='-', color='blue', 
                        linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
    if chosen_prob_sequences:
        line2 = ax1.plot(percentage_steps, avg_chosen_probs, marker='s', linestyle='--', color='green', 
                        linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)
    
    # Add truncation data if available
    if truncation_data:
        trunc_percentages = truncation_data.get('percentages', [])
        trunc_probs = truncation_data.get('probabilities', [])
        if trunc_percentages and trunc_probs:
            line_trunc = ax1.plot(trunc_percentages, trunc_probs, marker='D', linestyle=':', color='orange', 
                            linewidth=2, markersize=4, label='Truncation - Model CoT (Orig. Correct)', alpha=0.8)
    
    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    if entropy_sequences:
        line3 = ax2.plot(percentage_steps, avg_entropies, marker='^', linestyle='-.', color='red', 
                        linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
    
    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends from both axes
    lines = []
    labels = []
    if prob_sequences:
        lines.extend(line1)
        labels.append('Correct Token Probability')
    if chosen_prob_sequences:
        lines.extend(line2)
        labels.append('Chosen Token Probability')
    if truncation_data and trunc_percentages and trunc_probs:
        lines.extend(line_trunc)
        labels.append('Truncation - Model CoT (Orig. Correct)')
    if entropy_sequences:
        lines.extend(line3)
        labels.append('Next Token Entropy')
    
    if lines:
        ax1.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.title(f'Average Probabilities and Entropy per Step on {dataset_name} ({method_name}) - CORRECT ANSWERS ONLY\n'
              f'Samples: {num_correct}/{num_total_samples} correct', fontsize=16)
    
    # Add statistics text in two rows below the legend, even more to the right
    stats_text = f"Avg. Steps: {avg_steps:.1f}\nCorrect Samples: {num_correct}"
    plt.figtext(1.15, 0.75, stats_text, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7))

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_correct_aggregate_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Correct answers aggregate plot saved to {full_path}")
    plt.close()
    return full_path


def plot_incorrect_answers_aggregate(data, dataset_name, method_name, output_dir, truncation_data=None):
    """
    Calculates and plots the average probability of the target token, chosen token probability,
    and entropy at each generation step, averaged across only the INCORRECT samples in the dataset.
    Uses percentage-based x-axis (0-100% completion) to normalize for different response lengths.
    
    Args:
        truncation_data: Optional dict with truncation analysis data for model CoT originally incorrect
    """
    print("Generating aggregate plot of average metrics per step for INCORRECT answers only (percentage-based)...")

    # Collect sequences for each metric from incorrect samples only
    prob_sequences = []
    chosen_prob_sequences = []
    entropy_sequences = []
    log_lengths = []
    num_incorrect = 0
    num_total_samples = 0

    for sample in data.values():
        num_total_samples += 1
        
        # Determine correctness first
        score = sample.get('score')
        if isinstance(score, list) and len(score) > 0:
            is_correct = bool(score[0] == 1 or score[0] == True)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        else:
            is_correct = None
        
        # Only process if this sample is incorrect
        if is_correct is not False:
            continue
            
        num_incorrect += 1
        
        prob_log = sample.get('probability_log', {}).get('epoch_0', [])
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        entropies_log = sample.get('entropies', {}).get('epoch_0', [])
        
        if prob_log:
            log_lengths.append(len(prob_log))
            prob_sequences.append(prob_log)
        
        if chosen_probs_log:
            chosen_prob_sequences.append(chosen_probs_log)
                
        if entropies_log:
            entropy_sequences.append(entropies_log)

    if not prob_sequences and not chosen_prob_sequences and not entropy_sequences:
        print("No probability data found for incorrect answers to plot.")
        return None

    if num_incorrect == 0:
        print("No incorrect answers found in the dataset.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    
    # Use normalize_and_bin_sequences to create percentage-based averages
    avg_probabilities = normalize_and_bin_sequences(prob_sequences)
    avg_chosen_probs = normalize_and_bin_sequences(chosen_prob_sequences)
    avg_entropies = normalize_and_bin_sequences(entropy_sequences)

    # Create percentage-based x-axis (0-100%)
    percentage_steps = list(range(100))

    # Create the plot with dual y-axis for entropy
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot probabilities on left y-axis
    if prob_sequences:
        line1 = ax1.plot(percentage_steps, avg_probabilities, marker='o', linestyle='-', color='blue', 
                        linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
    if chosen_prob_sequences:
        line2 = ax1.plot(percentage_steps, avg_chosen_probs, marker='s', linestyle='--', color='green', 
                        linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)
    
    # Add truncation data if available
    if truncation_data:
        trunc_percentages = truncation_data.get('percentages', [])
        trunc_probs = truncation_data.get('probabilities', [])
        if trunc_percentages and trunc_probs:
            line_trunc = ax1.plot(trunc_percentages, trunc_probs, marker='D', linestyle=':', color='purple', 
                            linewidth=2, markersize=4, label='Truncation - Model CoT (Orig. Incorrect)', alpha=0.8)
    
    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    if entropy_sequences:
        line3 = ax2.plot(percentage_steps, avg_entropies, marker='^', linestyle='-.', color='red', 
                        linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
    
    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends from both axes
    lines = []
    labels = []
    if prob_sequences:
        lines.extend(line1)
        labels.append('Correct Token Probability')
    if chosen_prob_sequences:
        lines.extend(line2)
        labels.append('Chosen Token Probability')
    if truncation_data and trunc_percentages and trunc_probs:
        lines.extend(line_trunc)
        labels.append('Truncation - Model CoT (Orig. Incorrect)')
    if entropy_sequences:
        lines.extend(line3)
        labels.append('Next Token Entropy')
    
    if lines:
        ax1.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.title(f'Average Probabilities and Entropy per Step on {dataset_name} ({method_name}) - INCORRECT ANSWERS ONLY\n'
              f'Samples: {num_incorrect}/{num_total_samples} incorrect', fontsize=16)
    
    # Add statistics text in two rows below the legend, even more to the right
    stats_text = f"Avg. Steps: {avg_steps:.1f}\nIncorrect Samples: {num_incorrect}"
    plt.figtext(1.15, 0.75, stats_text, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7))

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_incorrect_aggregate_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Incorrect answers aggregate plot saved to {full_path}")
    plt.close()
    return full_path


def plot_single_question_probability(sample, dataset_name, method_name, output_dir):
    """
    For a single question, plots its step-by-step probability log, chosen token probabilities,
    entropy values, and marks the steps where exact matches of the correct answer occurred.
    Also show whether that question is correct on the plot.
    Uses percentage-based x-axis (0-100% completion) to normalize for different response lengths.
    """
    idx = sample.get('idx', 'N/A')
    print(f"Generating single-question plot for sample index: {idx} (percentage-based)...")

    prob_log = sample.get('probability_log', {}).get('epoch_0', [])
    chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
    entropies_log = sample.get('entropies', {}).get('epoch_0', [])
    exact_matches = sample.get('exact_match_steps', {}).get('epoch_0', [])

    if not prob_log and not chosen_probs_log and not entropies_log:
        print(f"No probability data found for sample {idx}.")
        return None

    # Determine correctness if possible
    score = sample.get('score')
    if isinstance(score, list) and len(score) > 0:
        is_correct = bool(score[0] == 1 or score[0] == True)
    elif isinstance(score, (int, bool)):
        is_correct = bool(score)
    else:
        is_correct = None

    # Determine the maximum length for consistent x-axis
    max_len = max(len(prob_log) if prob_log else 0, 
                  len(chosen_probs_log) if chosen_probs_log else 0,
                  len(entropies_log) if entropies_log else 0)
    
    if max_len == 0:
        print(f"No data to plot for sample {idx}.")
        return None
    
    # Create percentage-based x-axis (0-100%)
    percentage_steps = [int((i / (max_len - 1)) * 99) for i in range(max_len)] if max_len > 1 else [0]
    
    # Create the plot with dual y-axis for entropy
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot probabilities on left y-axis
    if prob_log:
        line1 = ax1.plot(percentage_steps[:len(prob_log)], prob_log, marker='o', linestyle='-', color='blue', 
                        linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
    if chosen_probs_log:
        line2 = ax1.plot(percentage_steps[:len(chosen_probs_log)], chosen_probs_log, marker='s', linestyle='--', color='green', 
                        linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)

    # Mark exact matches on correct token probability
    if exact_matches and prob_log:
        all_match_steps = [step for match_list in exact_matches for step in match_list]
        valid_match_steps = [s for s in all_match_steps if s < len(prob_log)]
        if valid_match_steps:
            # Convert step indices to percentage positions
            match_percentages = [percentage_steps[s] for s in valid_match_steps]
            ax1.plot(
                match_percentages,
                [prob_log[s] for s in valid_match_steps],
                'o', markersize=8, color='red', label='Correct Token Generated',
                markeredgecolor='darkred', markeredgewidth=2
            )

    ax1.set_xlabel('Response Completion (%)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=99)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    if entropies_log:
        line3 = ax2.plot(percentage_steps[:len(entropies_log)], entropies_log, marker='^', linestyle='-.', color='red', 
                        linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
    
    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends from both axes
    lines = []
    labels = []
    if prob_log:
        lines.extend(line1)
        labels.append('Correct Token Probability')
    if chosen_probs_log:
        lines.extend(line2)
        labels.append('Chosen Token Probability')
    if exact_matches and prob_log and valid_match_steps:
        # Add a dummy line for the legend entry
        lines.append(plt.Line2D([0], [0], marker='o', color='red', linestyle='None', 
                               markersize=8, markeredgecolor='darkred', markeredgewidth=2))
        labels.append('Correct Token Generated')
    if entropies_log:
        lines.extend(line3)
        labels.append('Next Token Entropy')
    
    ax1.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    # Title with correctness badge
    correctness_str = ''
    if is_correct is True:
        correctness_str = ' — Correct'
    elif is_correct is False:
        correctness_str = ' — Incorrect'
    plt.title(f'Generation Metrics for ID {idx} on {dataset_name} ({method_name}){correctness_str}', fontsize=16)

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_single_{method_name}_id_{idx}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {full_path}")
    plt.close()
    return full_path


def plot_path_of_distributions_single(sample, dataset_name, method_name, output_dir):
    """
    Plot Path of Distributions for a single sample.
    Shows both model path and gold path in 2D space after dimensionality reduction.
    """
    idx = sample.get('idx', 'N/A')
    print(f"Generating Path of Distributions plot for sample index: {idx}...")
    
    model_vectors = sample.get('model_path_vectors', {}).get('epoch_0', [])
    gold_vectors = sample.get('gold_path_vectors', {}).get('epoch_0', [])
    
    if not model_vectors or not gold_vectors:
        print(f"No path vectors found for sample {idx}.")
        return None
    
    # Convert to numpy arrays (vectors are now stored as lists in JSON)
    model_vectors = np.array(model_vectors)
    gold_vectors = np.array(gold_vectors)
    
    # Ensure same length by taking minimum
    min_len = min(len(model_vectors), len(gold_vectors))
    model_vectors = model_vectors[:min_len]
    gold_vectors = gold_vectors[:min_len]
    
    if min_len < 2:
        print(f"Not enough steps for Path of Distributions plot for sample {idx}.")
        return None
    
    # Combine data for consistent dimensionality reduction
    combined_data = np.vstack([model_vectors, gold_vectors])
    
    # Choose dimensionality reduction method
    if UMAP_AVAILABLE and combined_data.shape[0] > 10:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, combined_data.shape[0]-1))
        method_name_dr = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42)
        method_name_dr = "PCA"
    
    # Fit and transform
    reduced_data = reducer.fit_transform(combined_data)
    
    # Split back into model and gold paths
    model_path_2d = reduced_data[:min_len]
    gold_path_2d = reduced_data[min_len:]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot gold path (solid green line)
    plt.plot(gold_path_2d[:, 0], gold_path_2d[:, 1], 'g-', linewidth=3, alpha=0.8, label='Gold Path (CoT)')
    plt.scatter(gold_path_2d[:, 0], gold_path_2d[:, 1], c='green', s=50, alpha=0.8, zorder=5)
    
    # Plot model path (dashed blue line)
    plt.plot(model_path_2d[:, 0], model_path_2d[:, 1], 'b--', linewidth=3, alpha=0.8, label='Model Path')
    plt.scatter(model_path_2d[:, 0], model_path_2d[:, 1], c='blue', s=50, alpha=0.8, zorder=5)
    
    # Mark start and end points
    plt.scatter(model_path_2d[0, 0], model_path_2d[0, 1], c='blue', s=150, marker='o', 
                edgecolors='black', linewidth=2, label='Start', zorder=10)
    plt.scatter(model_path_2d[-1, 0], model_path_2d[-1, 1], c='red', s=150, marker='s', 
                edgecolors='black', linewidth=2, label='End', zorder=10)
    
    # Determine correctness if possible
    score = sample.get('score')
    if isinstance(score, list) and len(score) > 0:
        is_correct = bool(score[0] == 1 or score[0] == True)
    elif isinstance(score, (int, bool)):
        is_correct = bool(score)
    else:
        is_correct = None
    
    correctness_str = ''
    if is_correct is True:
        correctness_str = ' — Correct'
    elif is_correct is False:
        correctness_str = ' — Incorrect'
    
    plt.title(f'Path of Distributions for ID {idx} on {dataset_name} ({method_name}){correctness_str}\nUsing {method_name_dr}', fontsize=16)
    plt.xlabel(f'{method_name_dr} Component 1', fontsize=12)
    plt.ylabel(f'{method_name_dr} Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_path_single_{method_name}_id_{idx}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Path of Distributions plot saved to {full_path}")
    plt.close()
    return full_path


def plot_correct_vs_incorrect_comparison(data, dataset_name, method_name, output_dir):
    """
    Plot comparison of chosen token probabilities between correct and incorrect answers.
    Shows only two lines: correct aggregate vs incorrect aggregate.
    Calculates and displays average difference score.
    
    Args:
        data: Dictionary of samples
        dataset_name: Name of the dataset
        method_name: Method name
        output_dir: Output directory
    
    Returns:
        Path to saved plot
    """
    print("Generating correct vs incorrect comparison plot...")
    
    # Separate samples into correct and incorrect
    correct_sequences = []
    incorrect_sequences = []
    correct_count = 0
    incorrect_count = 0
    
    for sample in data.values():
        # Determine correctness
        score = sample.get('score')
        if isinstance(score, list) and len(score) > 0:
            is_correct = bool(score[0] == 1 or score[0] == True)
        elif isinstance(score, (int, bool)):
            is_correct = bool(score)
        else:
            continue
        
        # Get chosen token probabilities
        chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
        if not chosen_probs_log:
            continue
        
        if is_correct:
            correct_sequences.append(chosen_probs_log)
            correct_count += 1
        else:
            incorrect_sequences.append(chosen_probs_log)
            incorrect_count += 1
    
    if not correct_sequences or not incorrect_sequences:
        print("Need both correct and incorrect samples for comparison plot.")
        return None
    
    # Normalize and bin sequences to percentage-based
    correct_avg = normalize_and_bin_sequences(correct_sequences)
    incorrect_avg = normalize_and_bin_sequences(incorrect_sequences)
    
    # Calculate average difference score across all percentage points
    differences = []
    for i in range(100):
        if not np.isnan(correct_avg[i]) and not np.isnan(incorrect_avg[i]):
            differences.append(correct_avg[i] - incorrect_avg[i])
    
    avg_difference = np.mean(differences) if differences else 0.0
    
    # Create percentage-based x-axis
    percentage_steps = list(range(100))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot correct and incorrect lines
    line1 = ax.plot(percentage_steps, correct_avg, marker='o', linestyle='-', color='green', 
                    linewidth=2.5, markersize=3, label=f'Correct Answers (n={correct_count})', alpha=0.8)
    line2 = ax.plot(percentage_steps, incorrect_avg, marker='s', linestyle='--', color='red', 
                    linewidth=2.5, markersize=3, label=f'Incorrect Answers (n={incorrect_count})', alpha=0.8)
    
    ax.set_xlabel('Response Completion (%)', fontsize=12)
    ax.set_ylabel('Chosen Token Probability', fontsize=12)
    ax.set_xlim(left=0, right=99)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # Title with dataset and method info
    plt.title(f'Correct vs Incorrect: Chosen Token Probability\n{dataset_name} ({method_name})', 
              fontsize=16, pad=20)
    
    # Add statistics box with average difference score
    stats_text = (
        f"Correct: {correct_count} samples\n"
        f"Incorrect: {incorrect_count} samples\n"
        f"Avg Difference: {avg_difference:.4f}"
    )
    plt.figtext(0.15, 0.75, stats_text, fontsize=11, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_correct_vs_incorrect_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Correct vs Incorrect comparison plot saved to {full_path}")
    print(f"Average Difference Score: {avg_difference:.4f}")
    plt.close()
    return full_path


def plot_starting_tokens_by_level(data, dataset_name, method_name, output_dir, num_tokens=20):
    """
    Plot average chosen token probability for the first N tokens, separated by difficulty levels.
    All levels shown in one plot with different colors.
    
    Args:
        data: Dictionary of samples
        dataset_name: Name of the dataset
        method_name: Method name
        output_dir: Output directory
        num_tokens: Number of starting tokens to analyze (default: 20)
    """
    print(f"Generating starting tokens plot by difficulty level (first {num_tokens} tokens)...")
    
    # Group data by math level
    level_groups = group_data_by_math_level(data)
    
    # Colors and markers for different levels
    level_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}
    level_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    # Process each level
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for level in [1, 2, 3, 4, 5]:
        level_data = level_groups.get(level, {})
        if not level_data:
            continue
        
        # Collect first N tokens from each sample
        token_probs_by_position = [[] for _ in range(num_tokens)]
        
        for sample in level_data.values():
            chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
            if chosen_probs_log:
                # Get first num_tokens
                for i in range(min(num_tokens, len(chosen_probs_log))):
                    token_probs_by_position[i].append(chosen_probs_log[i])
        
        # Calculate average for each position
        avg_probs = []
        for probs in token_probs_by_position:
            if probs:
                avg_probs.append(np.mean(probs))
            else:
                avg_probs.append(np.nan)
        
        # Plot this level
        if avg_probs and not all(np.isnan(avg_probs)):
            steps = list(range(1, len(avg_probs) + 1))
            ax.plot(steps, avg_probs, marker=level_markers[level], linestyle='-', 
                   color=level_colors[level], linewidth=3, markersize=8, alpha=1.0,
                   label=f'Level {level} (n={len(level_data)})')
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Average Chosen Token Probability', fontsize=12)
    ax.set_xlim(left=1, right=num_tokens)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11, title='Difficulty Level', title_fontsize=12)
    plt.title(f'Starting Token Probabilities by Difficulty Level\n{dataset_name} ({method_name}) - First {num_tokens} Tokens', 
              fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_starting_tokens_by_level_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Starting tokens by level plot saved to {full_path}")
    plt.close()
    return full_path


def plot_ending_tokens_by_level(data, dataset_name, method_name, output_dir, num_tokens=20):
    """
    Plot average chosen token probability for the last N tokens, separated by difficulty levels.
    All levels shown in one plot with different colors.
    
    Args:
        data: Dictionary of samples
        dataset_name: Name of the dataset
        method_name: Method name
        output_dir: Output directory
        num_tokens: Number of ending tokens to analyze (default: 20)
    """
    print(f"Generating ending tokens plot by difficulty level (last {num_tokens} tokens)...")
    
    # Group data by math level
    level_groups = group_data_by_math_level(data)
    
    # Colors and markers for different levels
    level_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}
    level_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    # Process each level
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for level in [1, 2, 3, 4, 5]:
        level_data = level_groups.get(level, {})
        if not level_data:
            continue
        
        # Collect last N tokens from each sample
        token_probs_by_position = [[] for _ in range(num_tokens)]
        
        for sample in level_data.values():
            chosen_probs_log = sample.get('chosen_token_probs', {}).get('epoch_0', [])
            if chosen_probs_log and len(chosen_probs_log) >= num_tokens:
                # Get last num_tokens
                last_tokens = chosen_probs_log[-num_tokens:]
                for i in range(len(last_tokens)):
                    token_probs_by_position[i].append(last_tokens[i])
        
        # Calculate average for each position
        avg_probs = []
        for probs in token_probs_by_position:
            if probs:
                avg_probs.append(np.mean(probs))
            else:
                avg_probs.append(np.nan)
        
        # Plot this level
        if avg_probs and not all(np.isnan(avg_probs)):
            # X-axis represents position from end (e.g., -20 to -1)
            steps = list(range(-num_tokens, 0))
            ax.plot(steps, avg_probs, marker=level_markers[level], linestyle='-', 
                   color=level_colors[level], linewidth=3, markersize=8, alpha=1.0,
                   label=f'Level {level} (n={len(level_data)})')
    
    ax.set_xlabel('Token Position from End', fontsize=12)
    ax.set_ylabel('Average Chosen Token Probability', fontsize=12)
    ax.set_xlim(left=-num_tokens, right=-1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11, title='Difficulty Level', title_fontsize=12)
    plt.title(f'Ending Token Probabilities by Difficulty Level\n{dataset_name} ({method_name}) - Last {num_tokens} Tokens', 
              fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_ending_tokens_by_level_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Ending tokens by level plot saved to {full_path}")
    plt.close()
    return full_path


def fix_gt_in_truncation_data(truncation_filepath):
    """
    Fix GT CoT data by setting 'originally_correct' to True.
    Returns the fixed data as a list of dictionaries.
    
    Args:
        truncation_filepath: Path to truncation JSONL file
    
    Returns:
        List of fixed data dictionaries
    """
    print(f"Loading and fixing truncation data from: {truncation_filepath}")
    fixed_data = []
    lines_modified = 0
    
    try:
        with open(truncation_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # Fix GT CoT data
                    if data.get('cot_type') == 'gt' and 'sample_metadata' in data:
                        data['sample_metadata']['originally_correct'] = True
                        lines_modified += 1
                    
                    fixed_data.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")
                    continue
        
        print(f"Loaded {len(fixed_data)} entries, fixed {lines_modified} GT entries")
        return fixed_data
    
    except FileNotFoundError:
        print(f"Warning: Truncation file not found: {truncation_filepath}")
        return []
    except Exception as e:
        print(f"Error loading truncation data: {e}")
        return []


def process_truncation_data_for_plotting(truncation_data, category='correct'):
    """
    Process truncation data to extract probabilities for plotting.
    
    Args:
        truncation_data: List of truncation data entries
        category: 'correct' for originally correct, 'incorrect' for originally incorrect
    
    Returns:
        Dictionary with 'percentages' and 'probabilities' lists
    """
    from collections import defaultdict
    
    data_by_percent = defaultdict(list)
    
    for entry in truncation_data:
        cot_type = entry.get('cot_type')
        truncation_percent_float = entry.get('truncation_percent')
        sample_metadata = entry.get('sample_metadata', {})
        originally_correct = sample_metadata.get('originally_correct')
        
        output_data = entry.get('output', {})
        input_data = entry.get('input', {})
        
        target_probs = output_data.get('target_token_probs')
        chosen_ids = output_data.get('chosen_token_ids')
        target_id_list = input_data.get('target_answer_tokens')
        
        # Validate data
        if not all([truncation_percent_float is not None, originally_correct is not None,
                   target_probs, chosen_ids, target_id_list]):
            continue
        
        # Filter by category
        if category == 'correct':
            if cot_type != 'model' or not originally_correct:
                continue
        elif category == 'incorrect':
            if cot_type != 'model' or originally_correct:
                continue
        else:
            continue
        
        # Extract chosen token probabilities directly
        chosen_probs = output_data.get('chosen_token_probs', [])
        if not chosen_probs:
            continue
        
        # Filter out None values and convert to float
        valid_probs = [float(prob) for prob in chosen_probs if prob is not None]
        if not valid_probs:
            continue
        
        truncation_percent = int(round(truncation_percent_float * 100))
        
        # Use the average of all valid chosen token probabilities
        prob_to_use = np.mean(valid_probs)
        data_by_percent[truncation_percent].append(prob_to_use)
    
    # Calculate averages
    sorted_percents = sorted(data_by_percent.keys())
    avg_probs = [np.mean(data_by_percent[p]) for p in sorted_percents]
    
    return {
        'percentages': sorted_percents,
        'probabilities': avg_probs
    }


def plot_path_of_distributions_aggregate(data, dataset_name, method_name, output_dir):
    """
    Plot aggregate Path of Distributions with mean paths and confidence intervals.
    """
    print("Generating aggregate Path of Distributions plot...")
    
    all_model_vectors = []
    all_gold_vectors = []
    valid_samples = 0
    
    # Collect all path vectors from all samples
    for sample in data.values():
        model_vectors = sample.get('model_path_vectors', {}).get('epoch_0', [])
        gold_vectors = sample.get('gold_path_vectors', {}).get('epoch_0', [])
        
        if model_vectors and gold_vectors:
            # Convert to numpy arrays (vectors are now stored as lists in JSON)
            model_vectors = np.array(model_vectors)
            gold_vectors = np.array(gold_vectors)
            
            # Ensure same length
            min_len = min(len(model_vectors), len(gold_vectors))
            if min_len >= 2:
                all_model_vectors.append(model_vectors[:min_len])
                all_gold_vectors.append(gold_vectors[:min_len])
                valid_samples += 1
    
    if valid_samples == 0:
        print("No valid path data found for aggregate plot.")
        return None
    
    print(f"Found {valid_samples} valid samples for aggregate plot.")
    
    # Find common length (minimum across all samples)
    common_len = min(len(vectors) for vectors in all_model_vectors + all_gold_vectors)
    if common_len < 2:
        print("Not enough common steps for aggregate plot.")
        return None
    
    # Truncate all vectors to common length
    all_model_vectors = [vectors[:common_len] for vectors in all_model_vectors]
    all_gold_vectors = [vectors[:common_len] for vectors in all_gold_vectors]
    
    # Combine all data for global dimensionality reduction
    global_data = []
    for model_vecs, gold_vecs in zip(all_model_vectors, all_gold_vectors):
        global_data.extend(model_vecs)
        global_data.extend(gold_vecs)
    
    global_data = np.array(global_data)
    
    # Choose dimensionality reduction method
    if UMAP_AVAILABLE and global_data.shape[0] > 10:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, global_data.shape[0]-1))
        method_name_dr = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42)
        method_name_dr = "PCA"
    
    # Fit reducer on global data
    reducer.fit(global_data)
    
    # Transform each sample's paths
    model_paths_2d = []
    gold_paths_2d = []
    
    for model_vecs, gold_vecs in zip(all_model_vectors, all_gold_vectors):
        combined_sample = np.vstack([model_vecs, gold_vecs])
        reduced_sample = reducer.transform(combined_sample)
        
        model_path_2d = reduced_sample[:common_len]
        gold_path_2d = reduced_sample[common_len:]
        
        model_paths_2d.append(model_path_2d)
        gold_paths_2d.append(gold_path_2d)
    
    # Calculate mean paths and confidence intervals
    model_paths_2d = np.array(model_paths_2d)  # Shape: (n_samples, n_steps, 2)
    gold_paths_2d = np.array(gold_paths_2d)
    
    model_mean = np.mean(model_paths_2d, axis=0)
    model_std = np.std(model_paths_2d, axis=0)
    
    gold_mean = np.mean(gold_paths_2d, axis=0)
    gold_std = np.std(gold_paths_2d, axis=0)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot mean paths
    plt.plot(gold_mean[:, 0], gold_mean[:, 1], 'g-', linewidth=4, alpha=0.9, label='Gold Path Mean')
    plt.plot(model_mean[:, 0], model_mean[:, 1], 'b--', linewidth=4, alpha=0.9, label='Model Path Mean')
    
    # Add confidence intervals (one standard deviation)
    for i in range(common_len):
        # Gold path confidence ellipse
        plt.scatter(gold_mean[i, 0], gold_mean[i, 1], c='green', s=100, alpha=0.8, zorder=5)
        circle_gold = plt.Circle((gold_mean[i, 0], gold_mean[i, 1]), 
                                np.mean(gold_std[i]), color='green', alpha=0.2)
        plt.gca().add_patch(circle_gold)
        
        # Model path confidence ellipse
        plt.scatter(model_mean[i, 0], model_mean[i, 1], c='blue', s=100, alpha=0.8, zorder=5)
        circle_model = plt.Circle((model_mean[i, 0], model_mean[i, 1]), 
                                 np.mean(model_std[i]), color='blue', alpha=0.2)
        plt.gca().add_patch(circle_model)
    
    # Mark start and end points
    plt.scatter(model_mean[0, 0], model_mean[0, 1], c='blue', s=200, marker='o', 
                edgecolors='black', linewidth=2, label='Start', zorder=10)
    plt.scatter(model_mean[-1, 0], model_mean[-1, 1], c='red', s=200, marker='s', 
                edgecolors='black', linewidth=2, label='End', zorder=10)
    
    plt.title(f'Aggregate Path of Distributions on {dataset_name} ({method_name})\nUsing {method_name_dr} — {valid_samples} samples', fontsize=16)
    plt.xlabel(f'{method_name_dr} Component 1', fontsize=12)
    plt.ylabel(f'{method_name_dr} Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_path_aggregate_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Aggregate Path of Distributions plot saved to {full_path}")
    plt.close()
    return full_path


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
                               'starting_tokens_by_level', 'ending_tokens_by_level',
                               'correct_vs_incorrect'], 
                       required=True, help="Type of plot to generate.")
    parser.add_argument("--sample_id", type=int, help="The 'idx' of the sample to plot (required for single plot types).")
    parser.add_argument("--math_level", type=str, help="Math difficulty level (1-5) for level-based plots.")
    parser.add_argument("--truncation_file", type=str, help="Path to truncation analysis JSONL file (optional).")
    parser.add_argument("--num_tokens", type=int, default=20, help="Number of tokens for starting/ending token plots (default: 20).")

    args = parser.parse_args()

    all_data = load_data(args.jsonl_file)
    
    # Load truncation data if provided
    truncation_correct_data = None
    truncation_incorrect_data = None
    if args.truncation_file:
        truncation_raw = fix_gt_in_truncation_data(args.truncation_file)
        if truncation_raw:
            truncation_correct_data = process_truncation_data_for_plotting(truncation_raw, category='correct')
            truncation_incorrect_data = process_truncation_data_for_plotting(truncation_raw, category='incorrect')

    if args.plot_type == 'aggregate':
        plot_average_probability_by_step(all_data, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'single':
        if args.sample_id is None:
            print("Error: --sample_id is required for plot_type 'single'.")
        elif args.sample_id not in all_data:
            print(f"Error: Sample with index {args.sample_id} not found in the data.")
        else:
            single_sample = all_data[args.sample_id]
            plot_single_question_probability(single_sample, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'path_aggregate':
        plot_path_of_distributions_aggregate(all_data, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'path_single':
        if args.sample_id is None:
            print("Error: --sample_id is required for plot_type 'path_single'.")
        elif args.sample_id not in all_data:
            print(f"Error: Sample with index {args.sample_id} not found in the data.")
        else:
            single_sample = all_data[args.sample_id]
            plot_path_of_distributions_single(single_sample, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'correct_aggregate':
        plot_correct_answers_aggregate(all_data, args.dataset_name, args.method_name, args.output_dir, truncation_correct_data)
    elif args.plot_type == 'incorrect_aggregate':
        plot_incorrect_answers_aggregate(all_data, args.dataset_name, args.method_name, args.output_dir, truncation_incorrect_data)
    elif args.plot_type == 'level_single':
        if args.math_level is None:
            print("Error: --math_level is required for plot_type 'level_single'.")
        else:
            plot_math_level_single(all_data, args.dataset_name, args.method_name, args.output_dir, args.math_level)
    elif args.plot_type == 'level_aggregate':
        plot_math_level_aggregate(all_data, args.dataset_name, args.method_name, args.output_dir)
    elif args.plot_type == 'starting_tokens_by_level':
        plot_starting_tokens_by_level(all_data, args.dataset_name, args.method_name, args.output_dir, args.num_tokens)
    elif args.plot_type == 'ending_tokens_by_level':
        plot_ending_tokens_by_level(all_data, args.dataset_name, args.method_name, args.output_dir, args.num_tokens)
    elif args.plot_type == 'correct_vs_incorrect':
        plot_correct_vs_incorrect_comparison(all_data, args.dataset_name, args.method_name, args.output_dir)

