from together import Together
import json
import os
import argparse


def generate_completion(model, prompt, max_tokens, logprobs, output_dir, task_name, temperature):

    # Get API key from environment variable or .env file
    api_key = os.environ.get("TOGETHER_API_KEY")
    
    # If not in environment, try to load from .env file
    if not api_key:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TOGETHER_API_KEY=") and not line.startswith("#"):
                            api_key = line.split("=", 1)[1].strip()
                            break
            except Exception as e:
                print(f"Warning: Could not read .env file: {e}")
    
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY not found. Please either:\n"
            "1. Set environment variable: export TOGETHER_API_KEY=your_key\n"
            "2. Create .env file with: TOGETHER_API_KEY=your_key\n"
            "3. Copy env.template to .env and edit it"
        )
    
    client = Together(api_key=api_key)

    if logprobs:

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=max_tokens,
            logprobs=logprobs,
            temperature=temperature,
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    output = json.dumps(completion.model_dump(), indent=1)


    output_file = os.path.join(output_dir, f"CoTtest_{task_name}_{model}_{max_tokens}_{logprobs}_{temperature}.json")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write(output)
    print(f"Completion saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-tput")
    parser.add_argument("--prompt", type=str, default="2+3=?")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/togetherai_api")
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--temperature", type=int, default=0)
    args = parser.parse_args()

    # If output_dir is relative, make it relative to evaluation directory
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)

    generate_completion(args.model, args.prompt, args.max_tokens, args.logprobs, args.output_dir, args.task_name, args.temperature)

if __name__ == "__main__":
    main()


"""
python evaluation/togetherapi.py --prompt "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$." --max_tokens 4096 --logprobs 0 --task_name aime24 --temperature 0
"""

