import json
import os
import pickle
from pathlib import Path
from typing import List

import attrs
import click
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from benchmark_pipelines.scores.transfer_bench.utils import read_video

NUM_PROMPTS_PER_VIDEO = 6

# Distributed utilities
def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def dist_init():
    """Initialize distributed processing when launched with torchrun"""
    # Set tokenizers parallelism to avoid fork warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.distributed.is_initialized():
        return

    backend = "gloo" if os.name == "nt" else "nccl"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def gather_list_of_dict(data):
    """Gather a list of dictionaries from all ranks"""
    world_size = get_world_size()
    if world_size == 1:
        return data

    # Serialize the data
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # Get sizes from all ranks
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Pad tensor to max size
    if tensor.numel() < max_size:
        padding = torch.ByteTensor(max_size - tensor.numel()).fill_(0).to("cuda")
        tensor = torch.cat([tensor, padding], dim=0)

    # Gather tensors from all ranks
    tensor_list = [torch.ByteTensor(max_size).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)

    # Deserialize gathered data
    gathered_data = []
    for i, size in enumerate(size_list):
        buffer = tensor_list[i][:size].cpu().numpy().tobytes()
        gathered_data.extend(pickle.loads(buffer))

    return gathered_data



def extract_video_id_from_filename(filename: str) -> str:
    """
    Extract video_id from filename.
    Input format: task_0599__1.mp4 (video_id__seed) or task_0599.mp4
    Output format: task_0599 (just video_id without extension)
    """
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__")[0]
    return stem

@attrs.define
class Task:
    video_id: str
    input_video_file_list: list[str]
    pred_vid_arrays_all_prompts: List[np.ndarray] = attrs.field(factory=list)

    fps: int | float | None = None
    video_shape: tuple[int, int, int] | None = None
    max_frames: int | None = 121  # 121
    diversity_score_lpips: float | None = None

    # for foreground/background mask
    gt_segmentation_pkl_file: str | None = None
    gt_seg_dicts: list = attrs.field(factory=list)
    gt_foreground: np.ndarray | None = None

    diversity_score_lpips_fg: float | None = None
    diversity_score_lpips_bg: float | None = None


def read_videos_for_task(task: Task) -> Task:
    """Read all videos for a single task"""
    assert task.input_video_file_list
    assert len(task.input_video_file_list) == NUM_PROMPTS_PER_VIDEO, (
        f"Expected {NUM_PROMPTS_PER_VIDEO} videos, got {len(task.input_video_file_list)}. "
        f"First file is {task.input_video_file_list[0]}"
    )

    for input_video_file in task.input_video_file_list:
        try:
            pred_frames, pred_fps = read_video(input_video_file, max_frames=task.max_frames)
            task.pred_vid_arrays_all_prompts.append(pred_frames)
            task.fps = pred_fps
            task.video_shape = pred_frames[0].shape
        except Exception as e:
            logger.error(f"Error reading video {input_video_file}: {e}")
            task.pred_vid_arrays_all_prompts = []
            return task

    logger.debug(f"Read videos for {task.video_id}: shape {task.video_shape}, max_frames {task.max_frames}, fps {task.fps}")
    return task


def compute_diversity_score(task: Task, lpips_model) -> Task:
    """Compute diversity scores for a single task using LPIPS"""
    if len(task.pred_vid_arrays_all_prompts) < NUM_PROMPTS_PER_VIDEO:
        logger.error(
            f"Skipping task {task.video_id} due to missing pred videos "
            f"(expected: {NUM_PROMPTS_PER_VIDEO} got {len(task.pred_vid_arrays_all_prompts)})."
        )
        return task

    def get_lpips_value(vid0: np.ndarray, vid1: np.ndarray) -> float:
        lpips_values = lpips_model(vid0, vid1)
        if lpips_values.numel() == 0:
            logger.warning("lpips tensor is empty. Cannot compute max LPIPS score.")
            return 0.0
        else:
            return float(torch.mean(lpips_values).cpu())

    assert task.pred_vid_arrays_all_prompts
    arrays = task.pred_vid_arrays_all_prompts
    scores = [
        get_lpips_value(arrays[i].copy(), arrays[j].copy())
        for i in range(len(arrays))
        for j in range(len(arrays))
        if i != j
    ]
    avg_score = float(np.mean(scores))
    task.diversity_score_lpips = avg_score

    if task.gt_foreground is not None:
        bg_scores, fg_scores = [], []
        for i in range(len(arrays)):
            for j in range(len(arrays)):
                if i == j:
                    continue

                bg_i = arrays[i].copy()
                bg_i[task.gt_foreground] = (255, 255, 255)
                bg_j = arrays[j].copy()
                bg_j[task.gt_foreground] = (255, 255, 255)
                bg_scores.append(get_lpips_value(bg_i, bg_j))

                fg_i = arrays[i].copy()
                fg_i[~task.gt_foreground] = (255, 255, 255)
                fg_j = arrays[j].copy()
                fg_j[~task.gt_foreground] = (255, 255, 255)
                fg_scores.append(get_lpips_value(fg_i, fg_j))

        task.diversity_score_lpips_fg = float(np.mean(fg_scores))
        task.diversity_score_lpips_bg = float(np.mean(bg_scores))

    return task


def unload_task_data_single_task(task: Task) -> Task:
    """Unload all input data to reduce load during results retrieval upon pipeline completion. Effectively purges
    heavier data, such as arrays, which are no longer of interest by the end of the pipeline to only retain metrics.
    Purged data is accessible by fetching Task state at earlier stages.
    """
    # Loaded video data
    task.pred_vid_arrays_all_prompts = []
    # Computed segments
    task.gt_seg_dicts = []
    task.gt_foreground = None
    return task


def process_tasks(tasks: List[Task]) -> List[Task]:
    """Process tasks for diversity score computation with distributed support"""
    rank = get_rank()
    world_size = get_world_size()

    # Initialize LPIPS model
    from models.lpips import LPIPS
    lpips_model = LPIPS(net="vgg")
    lpips_model.setup()
    logger.info(f"LPIPS model loaded on rank {rank}")

    # Distribute tasks across ranks
    tasks = tasks[rank::world_size]
    print0(f"Rank {rank} processing {len(tasks)} tasks")

    # Process tasks
    results = []
    for task in tqdm(tasks, desc=f"Rank {rank}", disable=(rank != 0)):
        # Read videos
        task = read_videos_for_task(task)
        # Compute diversity scores
        task = compute_diversity_score(task, lpips_model)
        # Unload task data to reduce memory usage
        task = unload_task_data_single_task(task)
        results.append(task)

    # Gather results from all ranks
    if world_size > 1:
        results = gather_list_of_dict(results)

    return results


@click.group()
def cli() -> None: ...


def process_outputs(outputs: List[Task]) -> dict:
    """Process outputs to create final results dictionary with metrics"""
    keys_to_remove = [
        "pred_vid_arrays_all_prompts",
        "gt_seg_dicts",
        "gt_foreground",
    ]
    per_video = [
        {
            k: v
            for k, v in attrs.asdict(res).items()
            # remove numpy arrays and other large data objects
            if k not in keys_to_remove and not isinstance(v, np.ndarray)
        }
        for res in outputs
    ]

    df = pd.DataFrame(per_video)
    metrics = ["diversity_score_lpips"]
    # Add conditional metrics if they exist
    if "diversity_score_lpips_fg" in df.columns:
        metrics.extend(["diversity_score_lpips_fg", "diversity_score_lpips_bg"])

    results = {}
    results["global"] = df[metrics].mean().to_dict()
    results["per_video"] = per_video

    return results


@cli.command()
@click.option(
    "--videos_path",
    type=str,
    required=True,
    help="Folder where video files are located",
    show_default=True,
)
@click.option(
    "--output_path",
    type=str,
    default=None,
    help="Optional output json path. If none, will write in directory.",
    show_default=True,
)
def calculate_diversity(
    videos_path: str,
    output_path: str | None,
) -> None:
    # Initialize distributed processing
    dist_init()
    print0(f"Distributed processing enabled. Rank: {get_rank()}, World size: {get_world_size()}")

    dataset_path = Path(videos_path)
    assert dataset_path.exists(), f"Could not find {dataset_path}"

    if output_path is None:
        output_path = (dataset_path / "diversity_metrics.json").as_posix()

    # Prepare tasks
    tasks = []
    video_dir = dataset_path / "videos"
    assert video_dir.exists(), f"Video directory {video_dir} not found"

    videos = sorted(video_dir.glob("*.mp4"))
    assert videos, f"No videos found in {video_dir}"

    # Group videos by video_id
    video_groups = {}
    for video in videos:
        video_id = extract_video_id_from_filename(video.name)
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(video)

    # Create tasks for each video_id
    for video_id, video_list in video_groups.items():
        if len(video_list) != NUM_PROMPTS_PER_VIDEO:
            logger.warning(f"Video {video_id} has {len(video_list)} files, expected {NUM_PROMPTS_PER_VIDEO}. Skipping.")
            continue

        # Sort videos to ensure consistent ordering
        video_list = sorted(video_list, key=lambda x: x.name)

        input_video_file_list = [v.as_posix() for v in video_list]

        task = Task(
            video_id=video_id,
            input_video_file_list=input_video_file_list,
            pred_vid_arrays_all_prompts=[],
        )
        tasks.append(task)

    print0(f"Processing {len(tasks)} video groups.")

    try:
        # Process tasks
        results = process_tasks(tasks)

        # Only rank 0 processes and writes the output
        if get_rank() == 0:
            if results:
                output = process_outputs(results)

                with open(output_path, "w") as fp:
                    json.dump(output, fp, indent=4)
                print0(f"Run completed! See results at {output_path}")
    except Exception as e:
        print0(f"Evaluation run failed: {e}")
        raise e


if __name__ == "__main__":
    cli()
