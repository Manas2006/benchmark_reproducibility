import random
import os
import argparse
import time
import json
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--prompt", type=str, help="Custom prompt template to use instead of prompt_type")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--job_id", type=str, help="Job ID to include in output filename")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    # Use consistent naming logic with runner.py
    if hasattr(args, 'prompt') and args.prompt and args.prompt_type:
        prompt_type_for_file = f"{args.prompt_type}_custom"
    elif hasattr(args, 'prompt') and args.prompt:
        prompt_type_for_file = "custom"
    elif args.prompt_type:
        prompt_type_for_file = args.prompt_type
    else:
        prompt_type_for_file = "cot"
    
    out_file_prefix = f"{args.split}_{prompt_type_for_file}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add job_id to filename if provided
    if hasattr(args, 'job_id') and args.job_id:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{args.job_id}.jsonl"
    else:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    ##available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)
            # Output structured information for monitoring
            print(f"MONITOR_PROMPT: {json.dumps({'idx': idx, 'question': example['question'], 'prompt': full_prompt})}")

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        print(f"MONITOR_EPOCH: {json.dumps({'epoch': epoch, 'total_epochs': max_func_call, 'remaining_prompts': len(remain_prompts)})}")
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )

            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id
            outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            # Output structured information for monitoring
            print(f"MONITOR_RESPONSE: {json.dumps({'epoch': epoch, 'prompt_idx': i, 'response': output, 'full_query': query})}")
            
            # Parse CoT structure from response (using only the answer field)
            raw = output.strip()
            
            # Primary heuristic: If the delimiter #### is in raw, split on it
            if '####' in raw:
                cot_text, ans_text = raw.split('####', 1)
                cot_text = cot_text.strip()
                ans_text = ans_text.strip()
            else:
                # Fallback heuristic: Otherwise, split on the last newline
                lines = raw.strip().splitlines()
                cot_text = "\n".join(lines[:-1])
                ans_text = lines[-1]
            
            # Convert to structured data
            cot_steps = [line.strip() for line in cot_text.split('\n') if line.strip()]
            final_answer = ans_text.strip()
            
            # Store CoT data for later serialization
            if not hasattr(args, '_cot_data'):
                args._cot_data = []
            
            args._cot_data.append({
                'question': samples[i]['question'] if 'question' in samples[i] else samples[i].get('input', ''),
                'model': args.model_name_or_path,
                'cot_steps': cot_steps,
                'final_answer': final_answer,
                'correct_answer': samples[i].get('gt', ''),
                'prompt_idx': i,
                'epoch': epoch
            })
            # Check if using custom prompt (disable code execution for custom prompts)
            using_custom_prompt = hasattr(args, 'prompt') and args.prompt
            
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif not using_custom_prompt and "boxed" not in output and output.endswith("```"):
                # Only extract and execute code if not using custom prompt
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts (only if not using custom prompt)
        using_custom_prompt = hasattr(args, 'prompt') and args.prompt
        if not using_custom_prompt and remain_codes:
            remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    using_custom_prompt = hasattr(args, 'prompt') and args.prompt
    if using_custom_prompt:
        # For custom prompts, just extract the text without code execution
        results = [(code, "") for code in codes]
    else:
        # For standard prompts, execute code
        results = [
            run_execute(executor, code, args.prompt_type, data_name) for code in codes
        ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        
        if using_custom_prompt:
            # For custom prompts, the code is just the model's response text
            # Extract the final answer from the response
            for j in range(len(preds)):
                # Try to extract answer from the response
                response_text = code[j]
                # Look for patterns like "Therefore, the final answer is: \boxed{ANSWER}"
                import re
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', response_text)
                if boxed_match:
                    preds[j] = boxed_match.group(1)
                else:
                    # If no boxed format, just use the last line or the whole response
                    lines = response_text.strip().split('\n')
                    if lines:
                        preds[j] = lines[-1].strip()
                    else:
                        preds[j] = response_text.strip()
        else:
            # For standard prompts, use the original logic
            for j in range(len(preds)):
                if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ]:
                    preds[j] = choice_answer_clean(code[j])
                elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                    # remove any non-choice char
                    preds[j] = "".join(
                        [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                    )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    # Add job configuration to metrics
    job_config = {
        "model": args.model_name_or_path,
        "dataset": data_name,
        "prompt_type": args.prompt_type,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "n_sampling": args.n_sampling,
        "max_tokens": args.max_tokens_per_call,
        "eval_method": "pass@k",  # Default value
        "k": 1,  # Default value
    }
    
    # Add custom prompt if provided
    if hasattr(args, 'prompt') and args.prompt:
        job_config["prompt"] = args.prompt
    
    # Add job_id if provided
    if hasattr(args, 'job_id') and args.job_id:
        job_config["job_id"] = args.job_id
    
    result_json["job_configuration"] = job_config

    # Create metrics filename with job_id if provided
    if hasattr(args, 'job_id') and args.job_id:
        metrics_file = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    else:
        metrics_file = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    
    with open(metrics_file, "w") as f:
        json.dump(result_json, f, indent=4)
    
    # Save CoT data to JSON if available
    if hasattr(args, '_cot_data') and args._cot_data:
        # Create job-specific output directory if job_id is provided
        if hasattr(args, 'job_id') and args.job_id:
            job_dir = os.path.join(args.output_dir, args.job_id)
            os.makedirs(job_dir, exist_ok=True)
            cot_file = os.path.join(job_dir, 'cot.json')
        else:
            # Use the same directory as other outputs
            cot_file = out_file.replace(".jsonl", "_cot.json")
        
        # Write CoT data to JSON file
        with open(cot_file, 'w') as f:
            json.dump(args._cot_data, f, indent=2)
        
        print(f"CoT data saved to: {cot_file}")
        print(f"CoT data contains {len(args._cot_data)} samples")
        
        # Log first sample structure for validation
        if args._cot_data:
            first_sample = args._cot_data[0]
            print(f"Sample CoT structure: {len(first_sample.get('cot_steps', []))} steps, answer: '{first_sample.get('final_answer', '')}'")
            
            # Validate required fields
            required_fields = ['question', 'model', 'cot_steps', 'final_answer', 'correct_answer']
            missing_fields = [field for field in required_fields if field not in first_sample]
            if missing_fields:
                print(f"Warning: Missing required fields in CoT data: {missing_fields}")
            else:
                print("CoT data structure validation: PASSED")
    
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
