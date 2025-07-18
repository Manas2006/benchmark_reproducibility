import pytest
import subprocess
import tempfile
import os
import json
from pathlib import Path


def test_unit_run():
    """Test running a single experiment with a random example."""

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple test dataset
        test_data = {
            "questions": [
                {
                    "id": "test_001",
                    "question": "What is 2 + 2?",
                    "answer": "4",
                    "category": "math",
                },
                {
                    "id": "test_002",
                    "question": "What is the capital of France?",
                    "answer": "Paris",
                    "category": "geography",
                },
            ]
        }

        # Write test dataset
        dataset_path = os.path.join(temp_dir, "test_dataset.json")
        with open(dataset_path, "w") as f:
            json.dump(test_data, f)

        # Create a simple run.sh script for testing
        run_script = f"""#!/bin/bash
set -e

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --temperature)
            TEMP="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Simulate model output (this would normally call an actual model)
echo "Running model: $MODEL"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Temperature: $TEMP"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "Seed: $SEED"
echo "Max length: $MAX_LENGTH"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Prompt: $PROMPT"

# Simulate processing a question
question="What is 2 + 2?"
answer="4"
model_response="The answer is 4"

# Calculate accuracy (simplified)
accuracy = 1.0 if model_response.lower().find(answer.lower()) != -1 else 0.0

# Output result in expected format
result = {{
    "model": "$MODEL",
    "dataset": "$DATASET", 
    "split": "$SPLIT",
    "temp": float("$TEMP"),
    "top_p": float("$TOP_P"),
    "top_k": int("$TOP_K"),
    "seed": int("$SEED"),
    "max_length": int("$MAX_LENGTH"),
    "max_new_tokens": int("$MAX_NEW_TOKENS"),
    "accuracy": accuracy,
    "runtime": 1.23,
    "custom_metrics": {{
        "exact_match": accuracy,
        "question_count": 1
    }}
}}

echo "RESULT: ${{result}}"
"""

        # Write run script
        run_script_path = os.path.join(temp_dir, "run.sh")
        with open(run_script_path, "w") as f:
            f.write(run_script)

        # Make executable
        os.chmod(run_script_path, 0o755)

        # Run the test
        result = subprocess.run(
            [
                "bash",
                run_script_path,
                "--model",
                "test-model",
                "--dataset",
                "test_dataset.json",
                "--split",
                "test",
                "--temperature",
                "0.0",
                "--top_p",
                "1.0",
                "--top_k",
                "1",
                "--seed",
                "42",
                "--max_length",
                "2048",
                "--max_new_tokens",
                "512",
                "--prompt",
                "Answer the following question:",
                "--output_dir",
                temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        # Check that the script ran successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"

        # Check that output contains RESULT line
        assert "RESULT:" in result.stdout, "No RESULT line found in output"

        # Parse the result
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT:"):
                result_json = line[7:]  # Remove "RESULT:" prefix
                result_data = json.loads(result_json)

                # Verify result structure
                assert "model" in result_data
                assert "dataset" in result_data
                assert "accuracy" in result_data
                assert "runtime" in result_data
                assert "custom_metrics" in result_data

                # Verify values
                assert result_data["model"] == "test-model"
                assert result_data["dataset"] == "test_dataset.json"
                assert result_data["split"] == "test"
                assert result_data["temp"] == 0.0
                assert result_data["top_p"] == 1.0
                assert result_data["top_k"] == 1
                assert result_data["seed"] == 42
                assert result_data["max_length"] == 2048
                assert result_data["max_new_tokens"] == 512
                assert isinstance(result_data["accuracy"], (int, float))
                assert isinstance(result_data["runtime"], (int, float))
                assert isinstance(result_data["custom_metrics"], dict)

                break
        else:
            pytest.fail("No valid RESULT line found in output")


if __name__ == "__main__":
    pytest.main([__file__])
