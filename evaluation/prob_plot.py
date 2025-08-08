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
    Calculates and plots the average probability of the target token at each
    generation step, averaged across all samples in the dataset.
    Also shows avg accuracy across dataset if available.
    """
    print("Generating aggregate plot of average probability per step...")

    # {step_number: [list of probabilities at this step from all samples]}
    step_probs = {}
    log_lengths = []  # To calculate the average number of steps
    num_correct = 0
    total = 0

    for sample in data.values():
        prob_log = sample.get('probability_log', {}).get('epoch_0', [])
        if prob_log:
            log_lengths.append(len(prob_log))
            for step, prob in enumerate(prob_log):
                step_probs.setdefault(step, []).append(prob)
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

    if not step_probs:
        print("No probability data found to plot.")
        return None

    avg_steps = np.mean(log_lengths) if log_lengths else 0
    steps = sorted(step_probs.keys())
    avg_probabilities = [np.mean(step_probs[step]) for step in steps]

    # Compute dataset avg accuracy if available
    avg_acc = (num_correct / total * 100.0) if total > 0 else None

    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(steps, avg_probabilities, marker='.', linestyle='-', color='b')

    # Title with dataset name and method
    base_title = f'Average Probability vs. Step on {dataset_name} ({method_name})'
    if avg_acc is not None:
        base_title += f' — Avg Acc: {avg_acc:.1f}%'
    plt.title(base_title, fontsize=16)
    plt.xlabel('Generation Step', fontsize=12)
    plt.ylabel('Average Probability', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(left=0)

    # Show average step count
    plt.text(
        0.95,
        0.95,
        f"Avg. Steps: {avg_steps:.1f}",
        transform=plt.gca().transAxes,
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
    For a single question, plots its step-by-step probability log and marks
    the steps where exact matches of the correct answer occurred.
    Also show whether that question is correct on the plot.
    """
    idx = sample.get('idx', 'N/A')
    print(f"Generating single-question plot for sample index: {idx}...")

    prob_log = sample.get('probability_log', {}).get('epoch_0', [])
    exact_matches = sample.get('exact_match_steps', {}).get('epoch_0', [])

    if not prob_log:
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

    steps = range(len(prob_log))
    plt.figure(figsize=(15, 7))
    plt.plot(steps, prob_log, label='Prob. of Next Correct Token', color='dodgerblue', alpha=0.8)

    if exact_matches:
        all_match_steps = [step for match_list in exact_matches for step in match_list]
        plt.plot(
            [s for s in all_match_steps if s < len(prob_log)],
            [prob_log[s] for s in all_match_steps if s < len(prob_log)],
            'o', markersize=8, color='red', label='Correct Token Generated',
        )

    # Title with correctness badge
    correctness_str = ''
    if is_correct is True:
        correctness_str = ' — Correct'
    elif is_correct is False:
        correctness_str = ' — Incorrect'
    plt.title(f'Probability Trace for ID {idx} on {dataset_name} ({method_name}){correctness_str}', fontsize=16)
    plt.xlabel('Generation Step', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(left=0, right=len(list(steps)))
    plt.ylim(bottom=-0.05, top=1.05)

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

