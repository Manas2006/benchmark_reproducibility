import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


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


def plot_average_probability_by_step(data, dataset_name, method_name, output_dir):
    """
    Calculates and plots the average probability of the target token, chosen token probability,
    and entropy at each generation step, averaged across all samples in the dataset.
    Also shows avg accuracy across dataset if available.
    """
    print("Generating aggregate plot of average metrics per step...")

    # {step_number: [list of values at this step from all samples]}
    step_probs = {}
    step_chosen_probs = {}
    step_entropies = {}
    log_lengths = []  # To calculate the average number of steps
    num_correct = 0
    total = 0

    for sample in data.values():
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

    if not step_probs and not step_chosen_probs and not step_entropies:
        print("No probability data found to plot.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    
    # Get all possible steps from any of the metrics
    all_steps = set()
    all_steps.update(step_probs.keys())
    all_steps.update(step_chosen_probs.keys())
    all_steps.update(step_entropies.keys())
    steps = sorted(all_steps)
    
    # Calculate averages for each metric
    avg_probabilities = [np.mean(step_probs[step]) if step in step_probs else np.nan for step in steps]
    avg_chosen_probs = [np.mean(step_chosen_probs[step]) if step in step_chosen_probs else np.nan for step in steps]
    avg_entropies = [np.mean(step_entropies[step]) if step in step_entropies else np.nan for step in steps]

    # Compute dataset avg accuracy if available
    avg_acc = (num_correct / total * 100.0) if total > 0 else None

    # Create the plot with dual y-axis for entropy
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot probabilities on left y-axis
    if step_probs:
        line1 = ax1.plot(steps, avg_probabilities, marker='o', linestyle='-', color='blue', 
                        linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
    if step_chosen_probs:
        line2 = ax1.plot(steps, avg_chosen_probs, marker='s', linestyle='--', color='green', 
                        linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)
    
    ax1.set_xlabel('Generation Step', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    if step_entropies:
        line3 = ax2.plot(steps, avg_entropies, marker='^', linestyle='-.', color='red', 
                        linewidth=2, markersize=4, label='Next Token Entropy', alpha=0.8)
    
    ax2.set_ylabel('Entropy (nats)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends from both axes
    lines = []
    labels = []
    if step_probs:
        lines.extend(line1)
        labels.append('Correct Token Probability')
    if step_chosen_probs:
        lines.extend(line2)
        labels.append('Chosen Token Probability')
    if step_entropies:
        lines.extend(line3)
        labels.append('Next Token Entropy')
    
    ax1.legend(lines, labels, loc='upper right', fontsize=11)

    # Title with dataset name and method
    base_title = f'Generation Metrics vs. Step on {dataset_name} ({method_name})'
    if avg_acc is not None:
        base_title += f' — Avg Acc: {avg_acc:.1f}%'
    plt.title(base_title, fontsize=16)

    # Show average step count
    ax1.text(
        0.95,
        0.95,
        f"Avg. Steps: {avg_steps:.1f}",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5),
    )

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_aggregate_{method_name}.png"
    full_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {full_path}")
    plt.close()
    return full_path


def plot_single_question_probability(sample, dataset_name, method_name, output_dir):
    """
    For a single question, plots its step-by-step probability log, chosen token probabilities,
    entropy values, and marks the steps where exact matches of the correct answer occurred.
    Also show whether that question is correct on the plot.
    """
    idx = sample.get('idx', 'N/A')
    print(f"Generating single-question plot for sample index: {idx}...")

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
    
    steps = range(max_len)
    
    # Create the plot with dual y-axis for entropy
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot probabilities on left y-axis
    if prob_log:
        line1 = ax1.plot(range(len(prob_log)), prob_log, marker='o', linestyle='-', color='blue', 
                        linewidth=2, markersize=4, label='Correct Token Probability', alpha=0.8)
    if chosen_probs_log:
        line2 = ax1.plot(range(len(chosen_probs_log)), chosen_probs_log, marker='s', linestyle='--', color='green', 
                        linewidth=2, markersize=4, label='Chosen Token Probability', alpha=0.8)

    # Mark exact matches on correct token probability
    if exact_matches and prob_log:
        all_match_steps = [step for match_list in exact_matches for step in match_list]
        valid_match_steps = [s for s in all_match_steps if s < len(prob_log)]
        if valid_match_steps:
            ax1.plot(
                valid_match_steps,
                [prob_log[s] for s in valid_match_steps],
                'o', markersize=8, color='red', label='Correct Token Generated',
                markeredgecolor='darkred', markeredgewidth=2
            )

    ax1.set_xlabel('Generation Step', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=max_len-1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    if entropies_log:
        line3 = ax2.plot(range(len(entropies_log)), entropies_log, marker='^', linestyle='-.', color='red', 
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
    
    ax1.legend(lines, labels, loc='upper right', fontsize=11)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot probability data from JSONL results.")
    parser.add_argument("jsonl_file", help="Path to the input JSONL file.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset (e.g., GSM8K) for titles and filenames.")
    parser.add_argument("--method_name", required=True, help="Name of the method or model (e.g., CoT, Llama3-8B) for filenames.")
    parser.add_argument("--output_dir", default="plots", help="Directory to save the plots.")
    parser.add_argument("--plot_type", choices=['aggregate', 'single'], required=True, help="Type of plot to generate.")
    parser.add_argument("--sample_id", type=int, help="The 'idx' of the sample to plot (required for --plot_type single).")

    args = parser.parse_args()

    all_data = load_data(args.jsonl_file)

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

