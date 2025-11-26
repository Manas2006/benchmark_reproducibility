#!/usr/bin/env python3
"""
Script to download and convert GPQA dataset to JSONL format.

GPQA is a gated dataset on HuggingFace. Before running this script:
1. Visit https://huggingface.co/datasets/Idavidrein/gpqa
2. Click 'Agree and access repository' to accept the terms
3. The script will automatically use the token from backend/path_config.json
"""

import os
import json
import sys
from pathlib import Path
from datasets import load_dataset

def get_hf_token():
    """Get HuggingFace token from path_config.json or environment."""
    # First try path_config.json
    config_path = Path(__file__).parent.parent / "backend" / "path_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'hf_token' in config:
                return config['hf_token']
        except Exception as e:
            print(f"Warning: Could not read path_config.json: {e}")
    
    # Fallback to environment variable
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

def download_gpqa():
    """Download and convert GPQA dataset to JSONL format."""
    # Create output directory
    output_dir = Path("./data/gpqa")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading GPQA dataset from HuggingFace...")
    print("Note: This dataset is gated and requires accepting terms on HuggingFace")
    print("If authentication fails, please:")
    print("1. Visit https://huggingface.co/datasets/Idavidrein/gpqa")
    print("2. Accept the terms and conditions")
    print("3. Run: huggingface-cli login")
    print("4. Then re-run this script\n")
    
    # Get HuggingFace token
    hf_token = get_hf_token()
    if not hf_token:
        print("✗ No HuggingFace token found!")
        print("Please set HF_TOKEN environment variable or add 'hf_token' to backend/path_config.json")
        return False
    
    # Set token in environment for datasets library
    os.environ['HF_TOKEN'] = hf_token
    os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
    
    try:
        # GPQA has multiple configs, use 'gpqa_main' as the default
        # Available configs: 'gpqa_extended', 'gpqa_main', 'gpqa_diamond', 'gpqa_experts'
        config_name = "gpqa_main"  # Main GPQA dataset
        print(f"Using HuggingFace token from config...")
        print(f"Loading GPQA dataset with config: {config_name}")
        dataset = load_dataset("Idavidrein/gpqa", config_name, token=hf_token)
        print(f"✓ Successfully loaded GPQA dataset")
        print(f"Dataset splits: {list(dataset.keys())}")
        
        # Process each split
        for split_name in dataset.keys():
            print(f"\nProcessing split: {split_name}")
            examples = []
            
            for idx, example in enumerate(dataset[split_name]):
                # Convert to our format
                # GPQA has: Question, Correct Answer (text), and Incorrect Answer 1-3
                question_text = example.get("Question", example.get("question", ""))
                correct_answer = example.get("Correct Answer", example.get("correct answer", ""))
                
                # Build question with options if available
                incorrect_answers = []
                for i in range(1, 4):
                    incorrect = example.get(f"Incorrect Answer {i}", example.get(f"incorrect answer {i}", ""))
                    if incorrect:
                        incorrect_answers.append(incorrect)
                
                # If we have options, format them as multiple choice
                if incorrect_answers and correct_answer:
                    # Create options list
                    all_answers = [correct_answer] + incorrect_answers
                    # Shuffle or keep order? Let's keep correct first for now
                    options_text = "\nOptions:\n"
                    labels = ["(A)", "(B)", "(C)", "(D)", "(E)"]
                    for i, ans in enumerate(all_answers[:5]):
                        options_text += f"{labels[i]} {ans}\n"
                    question_text = question_text.rstrip() + "\n" + options_text.rstrip()
                    # Store the label for the correct answer
                    correct_label = labels[0]  # Correct answer is first
                else:
                    correct_label = correct_answer
                
                converted_example = {
                    "idx": idx,
                    "question": question_text,
                    "target": correct_label if incorrect_answers else correct_answer,
                    "gt": correct_label if incorrect_answers else correct_answer,
                    "gt_cot": None,
                }
                
                # Add all other fields from the example (lowercase keys for consistency)
                for key, value in example.items():
                    key_lower = key.lower()
                    if key_lower not in ["question", "target", "gt", "gt_cot"]:
                        converted_example[key_lower] = value
                
                examples.append(converted_example)
            
            # Save to JSONL
            output_file = output_dir / f"{split_name}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            print(f"  ✓ Saved {len(examples)} examples to {output_file}")
            if len(examples) > 0:
                print(f"  First example keys: {list(examples[0].keys())}")
                print(f"  Sample question: {examples[0].get('question', '')[:100]}...")
                print(f"  Sample target: {examples[0].get('target', '')}")
        
        print(f"\n✓ GPQA dataset downloaded successfully!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "authenticated" in error_msg.lower():
            print(f"\n✗ Authentication required: {error_msg}")
            print("\nTo download GPQA:")
            print("1. Visit https://huggingface.co/datasets/Idavidrein/gpqa")
            print("2. Click 'Agree and access repository'")
            print("3. Run: huggingface-cli login")
            print("4. Re-run this script: python download_gpqa.py")
        else:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    download_gpqa()

