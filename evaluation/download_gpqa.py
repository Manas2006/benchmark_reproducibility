#!/usr/bin/env python3
"""
Script to download and convert GPQA dataset to JSONL format.

GPQA is a gated dataset on HuggingFace. Before running this script:
1. Visit https://huggingface.co/datasets/Idavidrein/gpqa
2. Click 'Agree and access repository' to accept the terms
3. Run: huggingface-cli login (or set HF_TOKEN environment variable)
4. Then run this script
"""

import os
import json
from pathlib import Path
from datasets import load_dataset

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
    
    try:
        # Try to load the dataset
        dataset = load_dataset("Idavidrein/gpqa")
        print(f"✓ Successfully loaded GPQA dataset")
        print(f"Dataset splits: {list(dataset.keys())}")
        
        # Process each split
        for split_name in dataset.keys():
            print(f"\nProcessing split: {split_name}")
            examples = []
            
            for idx, example in enumerate(dataset[split_name]):
                # Convert to our format
                # GPQA has: Question, Correct (answer), and possibly other fields
                converted_example = {
                    "idx": idx,
                    "question": example.get("Question", example.get("question", "")),
                    "target": example.get("Correct", example.get("correct", example.get("answer", ""))),
                    "gt": example.get("Correct", example.get("correct", example.get("answer", ""))),
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

