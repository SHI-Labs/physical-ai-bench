import os
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
# Extract necessary functions from qwen_vl_utils to avoid direct dependency
from qwen_vl_utils import process_vision_info

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

from .utils import load_json
from .prompts.evaluation_prompt import (
    system_template_binary_v0,
    begin_user_template_binary_v0,
    user_template_binary_v0,
    video_template_fn_v0,
    output_format_fn_binary_v0
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_multimodal_input(messages, processor, cached_vision_info=None):
    """
    Prepare multimodal input for vLLM following the pattern from qwen2p5_vl_instruct_vllm.py

    Args:
        messages: List of message dictionaries containing text and media content
        processor: AutoProcessor instance for chat template formatting
        cached_vision_info: Optional cached (image_inputs, video_inputs) tuple to reuse

    Returns:
        tuple: (formatted_text, mm_data_dict) for vLLM input
    """
    # Apply chat template to format the conversation
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if cached_vision_info is not None:
        image_inputs, video_inputs = cached_vision_info
    else:
        image_inputs, video_inputs = process_vision_info(messages)

    # Format multi-modal data for vLLM
    mm_data = {}
    if image_inputs:
        mm_data["image"] = image_inputs
    if video_inputs:
        mm_data["video"] = video_inputs

    return text, mm_data

class QwenVLEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-72B-Instruct", device="cuda", tensor_parallel_size=8):
        """Initialize Qwen2.5-72B-VL model for VQA evaluation using vLLM"""
        self.device = device
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.sampling_params = None

    def load_model(self):
        """Load the specified VL model using vLLM"""
        logger.info(f"Loading model {self.model_name} using vLLM")

        # Configure vLLM parameters for optimal performance
        model_kwargs = {
            "model": self.model_name,
            "trust_remote_code": True,
            "max_model_len": 8192,
            "limit_mm_per_prompt": {"video": 1},
            "tensor_parallel_size": self.tensor_parallel_size,
        }

        self.model = LLM(**model_kwargs)
        logger.info(f"Successfully loaded model: {self.model_name} using vLLM")

        # Initialize processor and tokenizer following qwen2p5_vl_instruct_vllm.py pattern
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=4 * 28 * 28,
            max_pixels=768 * 28 * 28
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = 8192  # Match max_model_len
        self.processor.tokenizer = self.tokenizer

        # Set up sampling parameters for consistent generation
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop_token_ids=None,
            top_p=1.0,
            top_k=-1,
        )

    def extract_video_frames(self, video_path, max_frames=16):
        """Extract frames from video for VQA (kept for compatibility but not needed for vLLM)"""
        # This method is kept for compatibility but vLLM handles video processing directly
        # We'll use OpenCV as fallback
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        if frame_count > max_frames:
            indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        else:
            indices = range(frame_count)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

        cap.release()
        return frames

    def answer_questions_batch(self, video_path, qa_data, cached_vision_info=None, batch_size=16):
        """Answer all questions for a video in batch using vLLM with proper input preprocessing"""
        if self.model is None:
            self.load_model()

        if not qa_data:
            return []

        # Use imaginaire4 prompt templates for consistent formatting
        system_prompt = system_template_binary_v0
        begin_user_prompt = begin_user_template_binary_v0

        all_responses = []

        # Process questions in batches to avoid GPU memory issues
        for i in range(0, len(qa_data), batch_size):
            batch_qa = qa_data[i:i + batch_size]

            # Prepare batch inputs
            batch_inputs = []
            for qa in batch_qa:
                # Format the question using imaginaire4 template
                question_prompt = user_template_binary_v0(qa, is_reasoning=False)

                # Combine text prompts
                combined_text_prompt = f"{begin_user_prompt}\n\n{question_prompt}"

                # Create messages in the format expected by the processor
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "fps": 2.0,
                                "max_pixels": 768 * 28 * 28,
                            },
                            {"type": "text", "text": combined_text_prompt},
                        ],
                    }
                ]

                # Use the standardized preprocessing function
                text, mm_data = prepare_multimodal_input(messages, self.processor, cached_vision_info)
                batch_inputs.append({"prompt": text, "multi_modal_data": mm_data})

            # Generate responses for this batch
            outputs = self.model.generate(batch_inputs, sampling_params=self.sampling_params, use_tqdm=False)
            batch_responses = [output.outputs[0].text.strip() for output in outputs]
            all_responses.extend(batch_responses)

        return all_responses

    def evaluate_video_qa(self, video_path, qa_data):
        """Evaluate all questions for a single video using batch processing"""
        # Verify video file exists
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []

        if not qa_data:
            return []

        # Process video once to get cached vision info
        # Create a dummy message to extract vision info
        dummy_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 2.0,
                        "max_pixels": 768 * 28 * 28,
                    },
                ],
            }
        ]
        cached_vision_info = process_vision_info(dummy_messages)
        # logger.info(f"Processed vision info for video: {video_path}")

        # Get all model answers in batch
        model_answers = self.answer_questions_batch(video_path, qa_data, cached_vision_info)

        # Process results
        results = []
        for i, qa in enumerate(qa_data):
            question = qa['question']
            correct_answer = qa['answer']
            model_answer = model_answers[i]

            # Simple accuracy check - check if the model answer contains the correct choice
            is_correct = self.check_answer_accuracy(model_answer, correct_answer, qa['index2ans'])

            result = {
                'uid': qa['uid'],
                'question': question,
                'correct_answer': correct_answer,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'task': qa['task']
            }
            results.append(result)

        return results

    def evaluate_multi_seed_video_qa(self, video_info_list, qa_data):
        """Evaluate all questions for multiple seed versions of the same video"""
        if not video_info_list:
            logger.warning("No video files found")
            return []

        # Collect results from all seed versions
        # Each seed is a different video file, so each needs to be processed separately
        all_seed_results = []
        for video_info in video_info_list:
            video_path = video_info["path"]
            seed = video_info["seed"]

            logger.info(f"Evaluating seed {seed}: {video_path}")
            # The evaluate_video_qa method now processes the video once and reuses vision info
            seed_results = self.evaluate_video_qa(video_path, qa_data)

            # Add seed information to each result
            for result in seed_results:
                result["seed"] = seed

            all_seed_results.extend(seed_results)

        # Aggregate results across seeds for each question
        return self.aggregate_multi_seed_results(all_seed_results)

    def aggregate_multi_seed_results(self, all_seed_results):
        """Aggregate results from multiple seeds using majority vote and average accuracy"""
        # Group results by question uid
        question_groups = {}
        for result in all_seed_results:
            uid = result['uid']
            if uid not in question_groups:
                question_groups[uid] = []
            question_groups[uid].append(result)

        aggregated_results = []
        for uid, seed_results in question_groups.items():
            if not seed_results:
                continue

            # Use the first result as template
            template = seed_results[0]

            # Calculate accuracy across seeds
            correct_count = sum(1 for r in seed_results if r['is_correct'])
            total_seeds = len(seed_results)
            accuracy = correct_count / total_seeds if total_seeds > 0 else 0.0

            # Collect all model answers for reference
            all_answers = [r['model_answer'] for r in seed_results]

            # Create aggregated result using average accuracy instead of majority vote
            aggregated_result = {
                'uid': uid,
                'question': template['question'],
                'correct_answer': template['correct_answer'],
                'model_answers': all_answers,  # Keep all answers for debugging
                'accuracy': accuracy,
                'total_seeds': total_seeds,
                'task': template['task']
            }

            aggregated_results.append(aggregated_result)

        return aggregated_results

    def check_answer_accuracy(self, model_answer, correct_answer, index2ans):
        """Check if model answer is correct"""
        model_answer = model_answer.lower().strip()

        # Get the correct answer text
        correct_answer_text = index2ans[correct_answer].lower()

        # Check if model answer contains "yes" or "no" for binary questions
        if correct_answer_text == "yes":
            return "yes" in model_answer and "no" not in model_answer
        elif correct_answer_text == "no":
            return "no" in model_answer and "yes" not in model_answer
        else:
            # For other types of answers, check if the correct answer is in model response
            return correct_answer_text in model_answer


def find_all_video_files(video_id, video_dir):
    """Find all video files with different seeds for the given video_id"""
    # Try different suffixes that might match, including seed variations
    possible_suffixes = ["__0.mp4", "__1.mp4", "__2.mp4", "__3.mp4", "__4.mp4", ".mp4"]

    found_videos = []
    for suffix in possible_suffixes:
        video_path = os.path.join(video_dir, f"{video_id}{suffix}")
        if os.path.exists(video_path):
            # Extract seed from filename
            if "__" in suffix:
                seed = suffix.split("__")[1].split(".")[0]
            else:
                seed = "default"
            found_videos.append({"path": video_path, "seed": seed})

    return found_videos


def find_video_file(video_id, video_dir):
    """Find the first video file matching the video_id (for backward compatibility)"""
    videos = find_all_video_files(video_id, video_dir)
    return videos[0]["path"] if videos else None


def compute_vqa_accuracy(
    vqa_questions_dir,
    video_dir,
    prompt_file,
    model_name="Qwen/Qwen2.5-VL-72B-Instruct",
    device="cuda",
    tensor_parallel_size=8,
    **kwargs
):
    """
    Compute VQA accuracy using Qwen2.5-72B-VL model with optimized batch processing

    Refactored logic:
    1. Enumerate by video_id first
    2. For each video_id, enumerate all corresponding video_path/seed
    3. Call process_vision_info only once per video
    4. Batch call vllm on qa_pair dimension
    5. Record answers and report results

    Args:
        vqa_questions_dir: Directory containing VQA question files
        video_dir: Directory containing videos
        prompt_file: JSON file containing video metadata
        model_name: Name of the VQA model to use
        device: Device to run the model on
        tensor_parallel_size: Number of tensor parallel processes for vLLM

    Returns:
        overall_accuracy: Overall accuracy across all questions
        detailed_results: Detailed results for each video
        category_scores: Category-specific accuracy scores
    """

    # Load prompt file to get video_id mapping
    prompt_data = load_json(prompt_file)
    video_id_set = {item['video_id'] for item in prompt_data}

    # Initialize evaluator with vLLM backend
    evaluator = QwenVLEvaluator(model_name=model_name, device=device, tensor_parallel_size=tensor_parallel_size)

    # Load evaluator on rank 0 first to avoid conflicts
    if get_rank() == 0:
        evaluator.load_model()
        barrier()
    else:
        barrier()
        evaluator.load_model()

    # Get all VQA files and map them to video_ids
    vqa_files = [f for f in os.listdir(vqa_questions_dir) if f.endswith('.json')]
    vqa_files = distribute_list_to_rank(vqa_files)

    # Create video_id to vqa_file mapping
    video_id_to_vqa_file = {}
    for vqa_file in vqa_files:
        video_id = vqa_file.replace('.json', '')
        video_id_to_vqa_file[video_id] = vqa_file

    all_results = []
    correct_count = 0
    total_count = 0

    # Initialize category-specific counters
    category_stats = {
        'av': {'correct': 0, 'total': 0},
        'common_sense': {'correct': 0, 'total': 0},
        'human': {'correct': 0, 'total': 0},
        'industry': {'correct': 0, 'total': 0},
        'misc': {'correct': 0, 'total': 0},
        'physics': {'correct': 0, 'total': 0},
        'robot': {'correct': 0, 'total': 0}
    }

    def get_category_from_filename(filename):
        """Extract category from filename"""
        if filename.startswith('av_'):
            return 'av'
        elif filename.startswith('common_sense_'):
            return 'common_sense'
        elif filename.startswith('human_'):
            return 'human'
        elif filename.startswith('industry_'):
            return 'industry'
        elif filename.startswith('misc_'):
            return 'misc'
        elif filename.startswith('physics_'):
            return 'physics'
        elif filename.startswith('robot_'):
            return 'robot'
        else:
            assert False, f"Unknown category: {filename}"

    # Main evaluation loop: enumerate by video_id first
    for video_id, vqa_file in tqdm(video_id_to_vqa_file.items(), disable=get_rank() > 0, desc="Processing video_ids"):
        category = get_category_from_filename(vqa_file)

        # Skip if video_id not in prompt data
        if video_id not in video_id_set:
            logger.warning(f"Video ID {video_id} not found in prompt file, skipping")
            continue

        # Load QA data for this video_id
        qa_file_path = os.path.join(vqa_questions_dir, vqa_file)
        qa_data = load_json(qa_file_path)

        if not qa_data:
            logger.warning(f"No QA data found for {video_id}")
            continue

        # Find all corresponding video files (different seeds) for this video_id
        video_info_list = find_all_video_files(video_id, video_dir)
        if not video_info_list:
            logger.warning(f"No video files found for {video_id}")
            continue

        # logger.info(f"Processing video_id: {video_id} with {len(video_info_list)} seed versions")

        # Process each video_path/seed for this video_id
        all_seed_results = []
        for video_info in video_info_list:
            video_path = video_info["path"]
            seed = video_info["seed"]

            # logger.info(f"Evaluating video_id: {video_id}, seed: {seed}, path: {video_path}")

            # Use the optimized evaluate_video_qa method:
            # - Calls process_vision_info only once per video
            # - Batch processes all qa_pairs using vLLM
            seed_results = evaluator.evaluate_video_qa(video_path, qa_data)

            # Add seed information to each result
            for result in seed_results:
                result["seed"] = seed
                result["video_id"] = video_id

            all_seed_results.extend(seed_results)

        # Aggregate results across seeds for each question
        aggregated_results = evaluator.aggregate_multi_seed_results(all_seed_results)

        # Calculate video-level accuracy (average across questions within this video)
        video_accuracy = sum(r['accuracy'] for r in aggregated_results) / len(aggregated_results) if aggregated_results else 0.0

        # Update counters using video-level accuracy
        total_count += 1
        correct_count += video_accuracy
        if category in category_stats:
            category_stats[category]['total'] += 1
            category_stats[category]['correct'] += video_accuracy

        # Store results with multi-seed information
        video_paths = [v["path"] for v in video_info_list]
        seeds = [v["seed"] for v in video_info_list]
        all_results.append({
            'video_id': video_id,
            'video_paths': video_paths,  # List of all video paths
            'seeds': seeds,  # List of all seeds
            'category': category,
            'results': aggregated_results,  # Already aggregated across seeds
            'accuracy': video_accuracy,  # Use the already calculated video-level accuracy
            'total_seeds': len(video_info_list)
        })

    # Gather results from all ranks
    if get_world_size() > 1:
        all_results = gather_list_of_dict(all_results)

        # Recalculate totals from gathered results
        correct_count = 0
        total_count = 0
        # Reset category stats for recalculation
        for cat in category_stats:
            category_stats[cat] = {'correct': 0, 'total': 0}

        for video_result in all_results:
            category = video_result.get('category', 'unknown')
            video_accuracy = video_result.get('accuracy', 0.0)

            # Count each video once and use its overall accuracy
            total_count += 1
            correct_count += video_accuracy
            if category in category_stats:
                category_stats[category]['total'] += 1
                category_stats[category]['correct'] += video_accuracy

    # Calculate overall accuracy
    overall_accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Calculate category-specific scores
    category_scores = {}
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            category_scores[f"{category}_score"] = stats['correct'] / stats['total']
        else:
            category_scores[f"{category}_score"] = 0.0

    # Calculate total seeds processed
    total_seeds_processed = sum(result.get('total_seeds', 1) for result in all_results)

    logger.info(f"VQA Evaluation Complete: {correct_count}/{total_count} = {overall_accuracy:.4f}")
    logger.info(f"Total video_ids processed: {len(all_results)}")
    logger.info(f"Total seeds processed: {total_seeds_processed}")

    # Log category scores
    for category, score in category_scores.items():
        stats = category_stats[category.replace('_score', '')]
        logger.info(f"{category}: {stats['correct']}/{stats['total']} = {score:.4f}")

    return overall_accuracy, all_results, category_scores


if __name__ == "__main__":
    # Test the VQA evaluation
    vqa_questions_dir = "/lustre/fs1/portfolios/dir/projects/dir_cosmos_misc/users/fengzhez/repos/pai/predict/v1.0/vqa_clean"
    video_dir = "/lustre/fs1/portfolios/dir/projects/dir_cosmos_misc/users/fengzhez/evals/data/pai/organized-mini/andrewwan_20250818155604_Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_I2W_long_iter-10000"
    prompt_file = "/lustre/fs1/portfolios/dir/projects/dir_cosmos_misc/users/fengzhez/repos/VBench/data/cosmos_predict2_bench_full_info_mini.json"

    accuracy, results, category_scores = compute_vqa_accuracy(
        vqa_questions_dir=vqa_questions_dir,
        video_dir=video_dir,
        prompt_file=prompt_file,
        model_name="Qwen/Qwen2.5-VL-72B-Instruct"
    )

    print(f"Overall VQA Accuracy: {accuracy:.4f}")
    print("Category Scores:")
    for score_name, score_value in category_scores.items():
        print(f"  {score_name}: {score_value:.4f}")
