import argparse
import json
import os
import sys
from collections import defaultdict

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from models import CosmosReason1, check_answer


def run_inference(dataset_name, local_dir, output_dir, tensor_parallel_size=8):
    # Initialize model with tensor parallelism
    model = CosmosReason1(tensor_parallel_size=tensor_parallel_size)

    # Download dataset
    snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=local_dir)

    # Load dataset
    dataset = load_dataset(dataset_name, split="test")

    results = []
    for item in tqdm(dataset, desc="Processing"):
        question = item["question"]
        choices_str = "\n".join([f"{k}: {v}" for k, v in item["index2ans"].items() if v is not None])
        prompt = f"{question}\n{choices_str}"
        video_path = os.path.join(local_dir, item["video_path"])
        answer = model.inference(prompt, video_path)
        results.append(
            {
                "category": item.get("category", ""),
                "subcategory": item.get("subcategory", ""),
                "video_path": video_path,
                "question": question,
                "choices_str": choices_str,
                "answer": answer,
                "correct_answer": item["answer"],
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate(data):
    category_results = defaultdict(lambda: [0, 0])
    subcategory_results = defaultdict(lambda: [0, 0])
    main_category_results = defaultdict(lambda: [0, 0])
    all_results = []

    for item in tqdm(data, desc="Evaluating results"):
        category = item.get("category", "unknown")
        subcategory = item.get("subcategory", "unknown")
        is_correct = check_answer(item["answer"], item["correct_answer"])

        category_results[category][0] += is_correct
        category_results[category][1] += 1
        subcategory_results[subcategory][0] += is_correct
        subcategory_results[subcategory][1] += 1
        all_results.append(is_correct)

        # Map to main categories
        if category.lower().startswith("space"):
            main_category_results["Space"][0] += is_correct
            main_category_results["Space"][1] += 1
        elif category.lower().startswith("time"):
            main_category_results["Time"][0] += is_correct
            main_category_results["Time"][1] += 1

    print("\n=== Category Results ===")
    for category, (correct, total) in category_results.items():
        print(f"{category}: {correct}/{total} = {correct/total:.2%}")
    for main_category, (correct, total) in main_category_results.items():
        print(f"{main_category}: {correct}/{total} = {correct/total:.2%}")

    print("\n=== Subcategory Results ===")
    for subcategory, (correct, total) in subcategory_results.items():
        print(f"{subcategory}: {correct}/{total} = {correct/total:.2%}")

    if all_results:
        overall_avg = sum(all_results) / len(all_results)
        print("\n=== Overall Results ===")
        print(f"Total: {sum(all_results)}/{len(all_results)} = {overall_avg:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Hugging Face dataset name, e.g. 'username/dataset-name'"
    )
    parser.add_argument("--local_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=8, help="Number of tensor parallel processes for model"
    )
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on existing results")
    args = parser.parse_args()

    if args.eval_only:
        # Only run evaluation on existing results
        results_file = f"{args.output_dir}/results.json"
        if os.path.exists(results_file):
            with open(results_file) as f:
                evaluate(json.load(f))
        else:
            print(f"No results found at {results_file}")
    else:
        # Run inference with tensor parallelism
        results = run_inference(
            args.dataset_name, args.local_dir, args.output_dir, tensor_parallel_size=args.tensor_parallel_size
        )
        # Evaluate results
        evaluate(results)
