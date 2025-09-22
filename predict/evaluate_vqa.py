import argparse
import logging
import os
import sys
from pathlib import Path

from pbench.utils import save_json
from pbench.vqa_evaluation import compute_vqa_accuracy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="VQA Evaluation for Cosmos Videos")
    parser.add_argument("--vqa_questions_dir", type=str, required=True,
                       help="Directory containing VQA question JSON files")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--prompt_file", type=str, required=True,
                       help="JSON file containing video metadata")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct",
                       help="Name of the VQA model to use")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run the model on")
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                       help="Number of tensor parallel processes for vLLM")
    parser.add_argument("--output_dir", type=str, default="./vqa_results",
                       help="Directory to save results")
    parser.add_argument("--subset", type=str, default=None,
                       help="Evaluate only a subset (e.g., 'human', 'av', 'common_sense')")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate input paths
    if not os.path.exists(args.vqa_questions_dir):
        logger.error(f"VQA questions directory not found: {args.vqa_questions_dir}")
        return

    if not os.path.exists(args.prompt_file):
        logger.error(f"Prompt file not found: {args.prompt_file}")
        return

    if not os.path.exists(args.video_dir):
        logger.error(f"Video directory not found: {args.video_dir}")
        return

    logger.info("Starting VQA Evaluation")
    logger.info(f"VQA Questions: {args.vqa_questions_dir}")
    logger.info(f"Video Directory: {args.video_dir}")
    logger.info(f"Prompt File: {args.prompt_file}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    # Run evaluation
    try:
        overall_accuracy, detailed_results, category_scores = compute_vqa_accuracy(
            vqa_questions_dir=args.vqa_questions_dir,
            video_dir=args.video_dir,
            prompt_file=args.prompt_file,
            model_name=args.model_name,
            device=args.device,
            tensor_parallel_size=args.tensor_parallel_size
        )

        # Save results
        results_summary = {
            "overall_accuracy": overall_accuracy,
            "total_videos": len(detailed_results),
            "model_name": args.model_name,
            "category_scores": category_scores,
            "evaluation_params": {
                "vqa_questions_dir": args.vqa_questions_dir,
                "video_dir": args.video_dir,
                "prompt_file": args.prompt_file
            }
        }

        # Save summary
        summary_file = os.path.join(args.output_dir, "vqa_summary.json")
        save_json(results_summary, summary_file)
        logger.info(f"Results summary saved to: {summary_file}")

        # Save detailed results
        detailed_file = os.path.join(args.output_dir, "vqa_detailed_results.json")
        save_json(detailed_results, detailed_file)
        logger.info(f"Detailed results saved to: {detailed_file}")

        # Print summary
        print("\n" + "="*60)
        print("VQA EVALUATION SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Total Videos Evaluated: {len(detailed_results)}")
        print(f"Model: {args.model_name}")
        print("="*60)

        # Print category-specific scores
        print("\nCATEGORY-SPECIFIC SCORES:")
        print("-" * 40)
        for score_name, score_value in category_scores.items():
            category_name = score_name.replace('_score', '').upper()
            print(f"{category_name:<15}: {score_value:.4f}")
        print("="*60)

        # Print per-video accuracy if not too many videos
        if len(detailed_results) <= 20:
            print("\nPer-Video Accuracy:")
            for result in detailed_results:
                print(f"  {result['video_id']}: {result['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
