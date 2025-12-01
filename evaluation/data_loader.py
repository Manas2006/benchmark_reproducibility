import os
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl, lower_keys


def load_data(data_name, split, data_dir="./data"):
    # Check if data_name is a HuggingFace dataset URL and extract identifier
    hf_dataset_identifier = None
    data_name_sanitized = None
    if "huggingface.co/datasets/" in data_name:
        # Extract dataset identifier from URL
        # Format: https://huggingface.co/datasets/org/dataset_name
        parts = data_name.split("huggingface.co/datasets/")[-1].split("/")
        if len(parts) >= 2:
            dataset_org = parts[0]
            dataset_name = parts[1]
            hf_dataset_identifier = f"{dataset_org}/{dataset_name}"
            # Create a sanitized directory name from the URL
            data_name_sanitized = data_name.replace("https://", "").replace("http://", "").replace("/", "_")
    elif "/" in data_name and not os.path.exists(data_name) and not data_name.startswith("http"):
        # Check if data_name is a direct HuggingFace dataset identifier (format: org/dataset_name)
        # This handles cases like "HuggingFaceH4/MATH-500"
        # Exclude file paths and URLs
        parts = data_name.split("/")
        if len(parts) == 2 and not os.path.isabs(data_name):
            # Likely a HuggingFace dataset identifier
            hf_dataset_identifier = data_name
            data_name_sanitized = data_name.replace("/", "_")
    
    # Use sanitized name if it's a URL or HF identifier, otherwise use original data_name
    data_name_dir = data_name_sanitized if hf_dataset_identifier else data_name
    data_file = f"{data_dir}/{data_name_dir}/{split}.jsonl"
    
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        # Check if data_name is a HuggingFace dataset URL or identifier
        if hf_dataset_identifier:
                
                # Try to load the dataset with the specified split
                try:
                    dataset = load_dataset(hf_dataset_identifier, split=split)
                except Exception as e:
                    # If the split doesn't exist, try loading without split and use the first available
                    print(f"Warning: Could not load split '{split}' from {hf_dataset_identifier}: {e}")
                    print(f"Attempting to load the dataset without specifying split...")
                    dataset = load_dataset(hf_dataset_identifier)
                    # Use the first available split
                    if len(dataset.keys()) > 0:
                        first_split = list(dataset.keys())[0]
                        dataset = dataset[first_split]
                        print(f"Using split '{first_split}' instead")
                    else:
                        raise Exception(f"No splits available in dataset {hf_dataset_identifier}")
        elif data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "math500":
            dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "svamp":
            # evaluate on training set + test set
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets(
                [dataset, load_dataset("ChilleD/SVAMP", split="test")]
            )
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(
                lambda x: ";" not in x["answer"]
            )  # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example["type"] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", "all", split="test")
            # only keep stem subjects
            stem_subjects = [
                "abstract_algebra",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
                "machine_learning",
            ]
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x["type"] in stem_subjects)
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        elif data_name == "humaneval":
            # Load HumanEval from local JSONL file or HuggingFace
            try:
                dataset = load_dataset("openai/openai_humaneval", split=split)
            except Exception as e:
                # Fallback to local file if HuggingFace load fails
                print(f"Warning: Could not load HumanEval from HuggingFace: {e}")
                print(f"Attempting to load from local file...")
                examples = list(load_jsonl(f"{data_dir}/humaneval/{split}.jsonl"))
                dataset = Dataset.from_list(examples)
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name_dir}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples
