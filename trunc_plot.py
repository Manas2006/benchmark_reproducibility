import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse

def plot_cot_probabilities(filepath, output_filename="cot_detailed_analysis_plot.png"):
    """
    Parses a JSONL file to plot model confidence vs. CoT percentage, split by
    whether the model's full CoT was originally correct or incorrect.

    This generates six lines to show confidence trends for:
    - GT CoT (Always originally correct)
    - Model CoT (Originally correct)
    - Model CoT (Originally incorrect)
    Each category is further split into "All Samples" and "Correct Only" plots.
    """
    # Structure: { 'category': { 'analysis_type': { truncation_percent: [prob1, prob2, ...] } } }
    # e.g., data['Model - Originally Correct']['All Samples'][50] = [...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    line_count = 0

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line_count += 1
                try:
                    entry = json.loads(line)

                    # --- Extract all necessary data fields ---
                    cot_type = entry.get("cot_type")
                    truncation_percent_float = entry.get("truncation_percent")
                    sample_metadata = entry.get("sample_metadata", {})
                    originally_correct = sample_metadata.get("originally_correct")
                    
                    output_data = entry.get("output", {})
                    input_data = entry.get("input", {})
                    
                    target_probs = output_data.get("target_token_probs")
                    chosen_ids = output_data.get("chosen_token_ids")
                    target_id_list = input_data.get("target_answer_tokens")

                    # --- Validate data integrity for this entry ---
                    if not all([cot_type, truncation_percent_float is not None, originally_correct is not None, 
                                target_probs, chosen_ids, target_id_list]):
                        print(f"Line {line_count}: Skipping due to missing essential data fields.")
                        continue
                    
                    if not target_id_list or not target_probs:
                        print(f"Line {line_count}: Skipping due to empty probability or target lists.")
                        continue

                    # --- Determine the category for the data point ---
                    category = ""
                    if cot_type == 'gt':
                        category = 'GT CoT'
                    elif cot_type == 'model' and originally_correct:
                        category = 'Model CoT - Originally Correct'
                    elif cot_type == 'model' and not originally_correct:
                        category = 'Model CoT - Originally Incorrect'
                    
                    if not category:
                        continue

                    # --- Conditional Logic to find the probability ---
                    target_id = target_id_list[0]
                    truncation_percent = int(round(truncation_percent_float * 100))

                    prob_to_use = None
                    is_correct_at_step = False

                    try:
                        search_start_index = 1 if chosen_ids and chosen_ids[0] is None else 0
                        found_idx = chosen_ids.index(target_id, search_start_index)
                        prob_idx = found_idx - search_start_index
                        
                        if prob_idx < len(target_probs):
                            prob_to_use = target_probs[prob_idx]
                            is_correct_at_step = True
                        else:
                            prob_to_use = np.mean(target_probs)
                            
                    except ValueError:
                        prob_to_use = np.mean(target_probs)

                    # Append data for the "All Samples" line
                    data[category]['All Samples'][truncation_percent].append(prob_to_use)

                    # Append data for the "Correct Only" line if applicable
                    if is_correct_at_step:
                        data[category]['Correct Only'][truncation_percent].append(prob_to_use)

                except json.JSONDecodeError:
                    print(f"Line {line_count}: Skipping invalid JSON line.")
                except Exception as e:
                    print(f"Line {line_count}: Skipping due to a data processing error ('{e}').")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return
    
    if not data:
        print("No valid data was processed. Cannot generate a plot.")
        return

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    plot_configs = {
        'GT CoT': {'color': 'darkorange'},
        'Model CoT - Originally Correct': {'color': 'royalblue'},
        'Model CoT - Originally Incorrect': {'color': 'green'}
    }

    for category, config in plot_configs.items():
        if category not in data: continue

        # Plot "All Samples" line
        all_samples_data = data[category]['All Samples']
        sorted_percents_all = sorted(all_samples_data.keys())
        if sorted_percents_all:
            avg_probs_all = [np.mean(all_samples_data.get(p, [np.nan])) for p in sorted_percents_all]
            ax.plot(sorted_percents_all, avg_probs_all, marker='o', linestyle='-', 
                    label=f"{category} (All Samples)", color=config['color'])

        # Plot "Correct Only" line
        correct_samples_data = data[category]['Correct Only']
        if correct_samples_data:
            sorted_percents_correct = sorted(correct_samples_data.keys())
            if sorted_percents_correct:
                avg_probs_correct = [np.mean(correct_samples_data.get(p, [np.nan])) for p in sorted_percents_correct]
                ax.plot(sorted_percents_correct, avg_probs_correct, marker='x', linestyle='--', 
                        label=f"{category} (Correct Only)", color=config['color'])

    # Formatting
    ax.set_title("Detailed Analysis of Correct Token Probability vs. Percentage of CoT", fontsize=16, pad=20)
    ax.set_xlabel("Percentage of CoT Kept (%)", fontsize=12)
    ax.set_ylabel("Probability of Correct Token", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CoT probabilities from a JSONL file.")
    parser.add_argument("filepath", nargs='?', default="/work/10757/cc123456/ls6/benchmark-reproducibility/mathevalUI/evaluation/outputs/DeepSeek-R1-Distill-Qwen-7B/gsm8k/truncation_analysis/truncation_plots/gsm8k_truncation_detailed_logs_DeepSeek-R1-Distill-Qwen-7B_trunc_1368e9aa-7b4d-4dbf-89e3-ca7926e88871_1758576024_20250922_171524.jsonl",
                        help="Path to the JSONL data file. Defaults to 'sample_data.jsonl'.")
    parser.add_argument("--output", default="cot_probability_plot_deepseek-r1-distill-qwen-7b_gsm8k.png",
                        help="Filename for the saved plot. Defaults to 'evaluation/outputs/cot_probability_plot.png'.")
    
    args = parser.parse_args()

    print(f"Attempting to plot data from '{args.filepath}'...")
    plot_cot_probabilities(args.filepath, args.output)