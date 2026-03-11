"""
LLM-as-a-Judge 后处理脚本

对 open-ended 类型样本使用 Qwen3-32B 进行语义一致性评测。
非 open-ended 样本保留原始 reward 不变。

用法:
    python llm_judge.py \
        --input_json /path/to/output.json \
        --model_path /path/to/judge/model \
        --output_json /path/to/output_judged.json \
        --result_file /path/to/result-async/ModelName.json \
        --dataset_name mmvu-all \
        --num_gpus 2 \
        --max_tokens 2048 \
        --temperature 0.1
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
from vllm import LLM, SamplingParams


# ============== Judge Prompt ==============

OPEN_ENDED_PROMPT = """Judge whether the model answer is semantically consistent with the ground truth answer.
Reply with a brief reason, then output "Judgement: 1" if consistent or "Judgement: 0" if not.

Ground Truth: {answer}
Model Answer: {extracted_answer}
"""


# ============== Answer Extraction ==============

def extract_model_answer(model_output: str) -> str:
    """Extract the model's final answer from model_output.

    Priority:
    1. Text after "Therefore, the final answer is: "
    2. Fallback: remove <think>...</think> and return the rest
    """
    marker = "Therefore, the final answer is:"
    # Case-insensitive search for the marker
    idx = model_output.lower().find(marker.lower())
    if idx != -1:
        return model_output[idx + len(marker):].strip()

    # Fallback: remove think tags and return remaining text
    cleaned = re.sub(r'<think>.*?</think>', '', model_output, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned if cleaned else model_output.strip()


def extract_ground_truth(solution: str) -> str:
    """Extract ground truth from <answer>...</answer> tags in solution string."""
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', solution, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: return the solution as-is
    return solution.strip()


# ============== Judge Output Parsing ==============

def parse_judgement(judge_output: str) -> int:
    """Parse the judge's output to extract 0 or 1.

    Takes the LAST "Judgement: 0/1" match to handle cases where
    the model outputs multiple judgements during reasoning.
    """
    # Find ALL "Judgement: <digit>" matches, take the last one
    matches = list(re.finditer(r'Judgement\s*:\s*([01])', judge_output, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))

    # Fallback: check if the text ends with just "0" or "1"
    stripped = judge_output.strip()
    if stripped.endswith('1'):
        return 1
    if stripped.endswith('0'):
        return 0

    # Default to 0 (inconsistent) if parsing fails
    return 0


# ============== Incremental Save ==============

def compute_mean_acc(results):
    """Compute mean_acc using the same logic as the main eval script:
    average reward of all non-regression samples.
    """
    rewards = []
    for sample in results:
        if sample.get("problem_type") != "regression":
            rewards.append(sample.get("reward", 0.0))
    return sum(rewards) / len(rewards) if rewards else 0.0


def save_output(results, data, output_json):
    """Recompute accuracy and save results to JSON."""
    type_stats = defaultdict(lambda: {"count": 0, "correct": 0, "reward_sum": 0.0})
    for sample in results:
        pt = sample.get("problem_type", "unknown")
        reward = sample.get("reward", 0.0)
        type_stats[pt]["count"] += 1
        type_stats[pt]["reward_sum"] += reward
        if reward > 0.5:
            type_stats[pt]["correct"] += 1

    total_count = len(results)
    total_correct = sum(s["correct"] for s in type_stats.values())

    by_problem_type = {}
    for pt, stats in sorted(type_stats.items()):
        by_problem_type[pt] = {
            "count": stats["count"],
            "correct": stats["correct"],
            "accuracy": round(stats["correct"] / stats["count"], 4) if stats["count"] else 0.0,
        }

    final_acc = {
        "overall": {
            "count": total_count,
            "correct": total_correct,
            "accuracy": round(total_correct / total_count, 4) if total_count else 0.0,
        },
        "by_problem_type": by_problem_type,
    }

    output_data = {
        "results": results,
        "final_acc": final_acc,
    }
    for key in data:
        if key not in ("results", "final_acc"):
            output_data[key] = data[key]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return final_acc


def update_result_file(result_file, dataset_name, mean_acc):
    """Update mean_acc in the result summary file for the given dataset."""
    if os.path.exists(result_file):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    else:
        all_results = {}

    if dataset_name not in all_results:
        all_results[dataset_name] = {}

    all_results[dataset_name]["mean_acc"] = mean_acc

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"  Updated result file: {result_file}")
    print(f"  [{dataset_name}] mean_acc: {mean_acc:.4f}")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge post-processing for open-ended evaluation")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the *_output.json from the main evaluation pipeline")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-32B",
                        help="Path to the judge model (default: Qwen3-32B)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output path (default: <input>_judged.json)")
    parser.add_argument("--result_file", type=str, default=None,
                        help="Path to result summary JSON (e.g. result-async/ModelName.json)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset key in result file (e.g. mmvu-all, videoreasonbench)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs for tensor parallelism (default: auto-detect)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max tokens for judge output")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature for the judge model")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size for incremental inference (0 = all at once)")
    args = parser.parse_args()

    # Resolve output path
    if args.output_json is None:
        base, ext = os.path.splitext(args.input_json)
        args.output_json = f"{base}_judged{ext}"

    # Log file: same dir as output, *_judge_log.jsonl
    log_base, _ = os.path.splitext(args.output_json)
    log_path = f"{log_base}_log.jsonl"

    # Auto-detect GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {args.num_gpus} GPU(s)")

    # Load input JSON
    print(f"Loading input: {args.input_json}")
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in input JSON. Exiting.")
        return

    # Identify open-ended samples and build prompts
    open_ended_indices = []
    prompts = []
    for i, sample in enumerate(results):
        if sample.get("problem_type") == "open-ended":
            model_answer = extract_model_answer(sample.get("model_output", sample.get("output", "")))
            ground_truth = extract_ground_truth(sample.get("solution", ""))

            prompt = OPEN_ENDED_PROMPT.format(
                answer=ground_truth,
                extracted_answer=model_answer,
            )
            open_ended_indices.append(i)
            prompts.append(prompt)

    print(f"Total samples: {len(results)}")
    print(f"Open-ended samples to judge: {len(open_ended_indices)}")
    print(f"Log file: {log_path}")
    print(f"Output file: {args.output_json}")

    if not open_ended_indices:
        print("No open-ended samples found. Saving output with original results.")
        save_output(results, data, args.output_json)
        return

    # Initialize vLLM
    print(f"Loading judge model: {args.model_path}")
    print(f"Tensor parallel size: {args.num_gpus}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=32768,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Clear log file
    with open(log_path, "w") as f:
        pass

    # Determine batching
    batch_size = args.batch_size if args.batch_size > 0 else len(prompts)
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    judged_count = 0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_indices = open_ended_indices[start:end]

        print(f"\nBatch {batch_idx + 1}/{num_batches}: judging samples {start}~{end - 1} ({len(batch_prompts)} prompts)...")
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, (result_idx, prompt_text) in enumerate(zip(batch_indices, batch_prompts)):
            judge_text = outputs[j].outputs[0].text
            judgement = parse_judgement(judge_text)

            results[result_idx]["judge_output"] = judge_text
            results[result_idx]["reward"] = float(judgement)
            results[result_idx]["correct"] = (judgement == 1)
            judged_count += 1

            # Write log entry
            log_entry = {
                "index": result_idx,
                "problem_id": results[result_idx].get("problem_id"),
                "judge_prompt": prompt_text,
                "judge_output": judge_text,
                "parsed_judgement": judgement,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Incremental save after each batch
        final_acc = save_output(results, data, args.output_json)
        print(f"  Saved ({judged_count}/{len(prompts)} judged). "
              f"Current accuracy: {final_acc['overall']['accuracy']:.4f}")

    # Final save
    final_acc = save_output(results, data, args.output_json)
    mean_acc = compute_mean_acc(results)

    # Update result summary file
    if args.result_file and args.dataset_name:
        update_result_file(args.result_file, args.dataset_name, mean_acc)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  Judge Results Summary")
    print(f"{'='*60}")
    print(f"  mean_acc (non-regression): {mean_acc:.4f}")
    print(f"  Overall: {final_acc['overall']['correct']}/{final_acc['overall']['count']} = {final_acc['overall']['accuracy']:.4f}")
    by_problem_type = final_acc["by_problem_type"]
    print(f"  {'Type':<30s} {'Count':>6s} {'Correct':>8s} {'Accuracy':>10s}")
    print(f"  {'-'*58}")
    for pt, info in sorted(by_problem_type.items()):
        print(f"  {pt:<30s} {info['count']:>6d} {info['correct']:>8d} {info['accuracy']:>10.4f}")
    print(f"{'='*60}")
    print(f"  Log: {log_path}")
    print(f"  Output: {args.output_json}")
    print("Done.")


if __name__ == "__main__":
    main()
