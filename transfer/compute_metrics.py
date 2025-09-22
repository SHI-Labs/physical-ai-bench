import json
import os
import pickle
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import attrs
import click
import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from benchmark_pipelines.scores.transfer_bench.utils import (
    numpy_array_to_video_bytes, read_video, safe_resize, should_compute,
    write_video)
from models import dover, grounded_sam_v2, video_depth_anything
from schemas import eff_segmentation


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


def distribute_list_to_rank(tasks):
    """Distribute tasks across ranks for data parallelism"""
    rank = get_rank()
    world_size = get_world_size()
    tasks_per_rank = len(tasks) // world_size
    remaining = len(tasks) % world_size
    if rank < remaining:
        start_idx = rank * (tasks_per_rank + 1)
        end_idx = start_idx + tasks_per_rank + 1
    else:
        start_idx = rank * tasks_per_rank + remaining
        end_idx = start_idx + tasks_per_rank
    return tasks[start_idx:end_idx]


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

METRICS = [
    "dover_tech_score",
    "blur_ssim",
    "canny_f1_score",
    "canny_precision",
    "canny_recall",
    "depth_si_rmse",
    "seg_m_iou",
    "seg_recall",
]


def extract_video_id_from_pred_filename(pred_filename: str) -> str:
    """
    Extract video_id from prediction filename.
    Input format: task_0599__1.mp4 (video_id__seed)
    Output format: task_0599 (just video_id without extension)
    """
    if "__" in pred_filename:
        return pred_filename.split("__")[0]
    else:
        return pred_filename.replace(".mp4", "")


def get_seg_pkl_path_from_video_id(gt_directory: str, video_id: str) -> str:
    return os.path.join(gt_directory, "sam2_pkls", f"{video_id}.pkl")


def get_depth_npy_path_from_video_id(gt_directory: str, video_id: str) -> str:
    return os.path.join(gt_directory, "depth_npzs", f"{video_id}.npz")


@attrs.define
class Task:
    pred_video_uuid: str
    pred_video_file: str
    gt_video_file: str
    video_caption_file: str
    pred_resized_video_file: str = ""
    force_recompute_gt_seg: bool = False
    force_recompute_gt_depth: bool = False

    fps: int = 0
    pred_fps: int = 0
    video_shape: tuple[int, int, int] | None = None
    gt_video_array: np.ndarray | None = None
    pred_video_array: np.ndarray | None = None
    pred_resized_video_array: np.ndarray | None = None
    gt_seg_dicts: list | None = None  # is actually a list of SAMv2 result dicts
    pred_seg_dicts: list | None = None

    gt_segmentation_pkl_file: str = ""
    gt_depth_npy_file: str = ""

    pred_segmentation_pkl_file: str = ""
    pred_segmentation_mp4_file: str = ""
    pred_depth_mp4_file: str = ""
    pred_blur_mp4_file: str = ""
    pred_canny_mp4_file: str = ""

    max_frames: int | None = 121
    caption: str | None = None

    # dover score
    dover_tech_score: float | None = None
    dover_tech_score_gt: float | None = None

    # canny score
    canny_f1_score: float | None = None
    canny_precision: float | None = None
    canny_recall: float | None = None

    # blur SSIM
    blur_ssim: float | None = None
    blur_mse: float | None = None

    # depth si-rMSE
    depth_si_rmse: float | None = None

    # seg metrics
    seg_m_iou: float | None = None
    seg_recall: float | None = None


def load_video_single_task(task: Task) -> Task:
    """Load ground truth and predicted videos from filesystem"""
    gt_frames, gt_fps = read_video(
        task.gt_video_file,  # expected shape: [T, H, W, 3]
        task.max_frames,
    )
    pred_frames, pred_fps = read_video(
        task.pred_video_file,
        task.max_frames,
    )

    task.video_shape = gt_frames.shape[:3]
    task.max_frames = gt_frames.shape[0]
    task.fps = int(gt_fps)
    task.pred_fps = int(pred_fps)
    task.gt_video_array = gt_frames
    task.pred_video_array = pred_frames

    return task


def load_videos(tasks: list[Task], num_workers: int = 8) -> list[Task]:
    """Load videos with multi-threading for I/O operations"""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_video_single_task, task) for task in tasks]
        results = [future.result() for future in futures]
    return results


def resize_video_single_task(task: Task) -> Task:
    """
    If pred video shape != gt video shape, resize pred video to match gt video shape.
    Keep the video array as python object for later stages
    If the shape is the same, don't do resizing, return stats
    """
    T_gt = task.max_frames
    T_pred = task.pred_video_array.shape[0]

    gt_frames = task.gt_video_array
    # Unload original pred video
    pred_frames, task.pred_video_array = task.pred_video_array, None

    if T_pred > T_gt:
        # logger.warning(f"Trimming pred num frames from {T_pred} to gt {T_gt}")
        pred_frames = pred_frames[:T_gt]

    if gt_frames.shape[:3] != pred_frames.shape[:3]:
        if gt_frames.shape[1] > pred_frames.shape[1]:
            interpolation = cv2.INTER_LINEAR  # upsample, use linear interp
        else:
            interpolation = cv2.INTER_AREA  # downsample, use area interp
        # logger.warning(f"Resizing pred {pred_frames.shape} to gt {gt_frames.shape}")
        pred_frames = safe_resize(
            pred_frames,
            gt_frames.shape[2],
            gt_frames.shape[1],
            interpolation=interpolation,
        )

        task.pred_resized_video_array = pred_frames
        write_resized_video(task)
    else:
        # logger.info("Prediction and GT video sizes match. No resizing needed.")
        task.pred_resized_video_file = task.pred_video_file
        task.pred_resized_video_array = pred_frames

    # Raw pred video no longer needed, delete
    del pred_frames

    return task


def write_resized_video(task: Task) -> None:
    """If the path is defined, exports pred_resized_video locally to a filesystem or remotely to S3."""
    # WARNING: the saved, resized video is only for visualization only. The video codec is lossy.
    # metrics are computed from the task.pred_resized_video_array directly.
    if not task.pred_resized_video_file:
        logger.warning("Saving pred_resized_video skipped, no path specified.")
    else:
        write_video(
            task.pred_resized_video_array,
            task.pred_resized_video_file,
            fps=task.pred_fps,
        )


def resize_videos(tasks: list[Task], num_workers: int = 4) -> list[Task]:
    """Process tasks with multi-threading for CPU-intensive resize operations"""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(resize_video_single_task, task) for task in tasks]
        results = [future.result() for future in futures]
    return results


def dover_single_task(task: Task, dover_model) -> Task:
    """Process single task with DOVER model"""
    assert task.pred_resized_video_array is not None
    assert task.gt_video_array is not None
    assert task.fps

    video_buffer_pred = numpy_array_to_video_bytes(task.pred_resized_video_array, fps=task.fps)
    video_buffer_gt = numpy_array_to_video_bytes(task.gt_video_array, fps=task.fps)

    results = dover_model([video_buffer_pred])
    task.dover_tech_score = float(results[0])

    results_gt = dover_model([video_buffer_gt])
    task.dover_tech_score_gt = float(results_gt[0])
    return task


def caption_single_task(task: Task) -> Task:
    """Load caption for single task"""
    # Check if caption already loaded
    if task.caption is None:
        if task.video_caption_file and Path(task.video_caption_file).exists():
            # If not, load from a caption file if possible
            with open(task.video_caption_file, "r") as fp:
                caption_json = json.load(fp)
        else:
            raise ValueError(f"No caption file found {task.video_caption_file}")

        assert len(caption_json) == 1
        task.caption = next(iter(caption_json.values()))
    return task


def load_captions(tasks: list[Task]) -> list[Task]:
    """Load captions for all tasks"""
    results = []
    for task in tasks:
        results.append(caption_single_task(task))
    return results


def canny_single_task(task: Task) -> Task:
    """
    assumes GT has canny npy available
    for predcited video, if canny is pre-computed, load it; otherwise compute it.
    Then compute F1 score.
    """
    from benchmark_pipelines.scores.transfer_bench.metrics_canny_blur_depth import \
        compute_canny_error_video_f1
    from benchmark_pipelines.scores.transfer_bench.video_to_canny_and_blur import \
        convert_rgb_mp4_to_canny_mp4

    # step 1, compute canny map for pred video
    try:
        # both are npy arrays
        gt_canny = convert_rgb_mp4_to_canny_mp4(
            task.gt_video_array,
            task.fps,
            out_fn_canny_mp4=None,
            out_fn_canny_npy=None,  # won't save anything for gt
            preset_strength="medium",
        )
        pred_canny = convert_rgb_mp4_to_canny_mp4(
            task.pred_resized_video_array,
            task.pred_fps,
            out_fn_canny_mp4=task.pred_canny_mp4_file,  # save computed pred canny for visualization
            out_fn_canny_npy=None,
            preset_strength="medium",
            force_overwrite=True,
        )

    except Exception as e:  # noqa: BLE001
        logger.exception(f"Got exception {e} when computing canny maps from video")
        return task

    # step 2, compute F1 score
    try:
        canny_f1_score, canny_precision, canny_recall = compute_canny_error_video_f1(pred_canny, gt_canny)
        task.canny_f1_score = canny_f1_score
        task.canny_precision = canny_precision
        task.canny_recall = canny_recall
    except Exception as e:  # noqa: BLE001
        logger.error(f"Got exception {e} for compute_canny_error_video_f1")
        return task
    return task


def blur_single_task(task: Task) -> Task:
    """
    assumes GT has blur npy available
    for predcited video, if blur is pre-computed, load it; otherwise compute it.
    Then compute SSIM on blurred video, pred vs gt.
    """
    from benchmark_pipelines.scores.transfer_bench.metrics_canny_blur_depth import \
        compute_blur_error_blur_video
    from benchmark_pipelines.scores.transfer_bench.video_to_canny_and_blur import \
        convert_rgb_mp4_to_blur_mp4

    # step 1, compute blur map for pred and gt video
    try:
        gt_blur = convert_rgb_mp4_to_blur_mp4(
            task.gt_video_array,
            task.fps,
            out_fn_blur_mp4=None,
            out_fn_blur_npy=None,
            blur_type="bilateral",
        )
        pred_blur = convert_rgb_mp4_to_blur_mp4(
            task.pred_resized_video_array,
            task.pred_fps,
            out_fn_blur_mp4=task.pred_blur_mp4_file,
            out_fn_blur_npy=None,
            blur_type="bilateral",
            force_overwrite=True,
        )

    except Exception as e:  # noqa: BLE001
        logger.exception(f"Got exception {e} when computing blur maps from video")
        return task

    try:
        blur_ssim = compute_blur_error_blur_video(pred_blur, gt_blur, metric_name="ssim")
        blur_mse = compute_blur_error_blur_video(pred_blur, gt_blur, metric_name="mse")
        task.blur_ssim = blur_ssim
        task.blur_mse = blur_mse
    except Exception as e:  # noqa: BLE001
        logger.error(f"Got exception {e} for compute_blur_error_blur_video")
        return task
    return task


def compute_and_save_segments(caption: str, frames: np.ndarray, fps: int, pkl_fn: Optional[str], sam_model) -> list:
    """Compute and save segmentation using SAM model"""
    buffer = numpy_array_to_video_bytes(frames, fps=fps)
    seg_list = sam_model.generate_single(buffer, caption)
    # save the SAM2-inferred segmentation as pkl
    if pkl_fn:
        with open(pkl_fn, "wb") as fp:
            pickle.dump([seg.to_dict() for seg in seg_list], fp)
    return seg_list


def sam_single_task(task: Task, sam_model) -> Task:
    """
    given gt caption and pred video, generate seg masks and save as pkl
    """
    assert task.caption

    task.pred_seg_dicts = compute_and_save_segments(
        task.caption,
        task.pred_resized_video_array,
        task.pred_fps,
        task.pred_segmentation_pkl_file,
        sam_model,
    )

    if should_compute(task.gt_segmentation_pkl_file, task.force_recompute_gt_seg):
        task.gt_seg_dicts = compute_and_save_segments(
            task.caption,
            task.gt_video_array,
            task.fps,
            task.gt_segmentation_pkl_file,
            sam_model,
        )
    else:
        with open(task.gt_segmentation_pkl_file, "rb") as fp:
            seg_list = pickle.load(fp)
            task.gt_seg_dicts = [eff_segmentation.SAMV2Detection.from_dict(seg) for seg in seg_list]

    return task


def mask_iou_single_task(task: Task, matching: str = "hungarian") -> Task:
    """Calculate mask IoU and recall for single task"""
    from benchmark_pipelines.scores.transfer_bench.segmentation_metrics import \
        calculate_mask_iou_and_recall

    try:
        gt = task.gt_seg_dicts
    except Exception as e:  # noqa: BLE001
        logger.exception(e)
        return task
    try:
        pred = task.pred_seg_dicts
    except Exception as e:  # noqa: BLE001
        logger.exception(e)
        return task

    if not gt or not pred or not task.video_shape:
        logger.error(
            f"MASK IOU EROOR:{gt}, {pred} and {task.video_shape}, "
            f"{task.pred_video_file}, {task.gt_segmentation_pkl_file}"
        )
        task.seg_m_iou, task.seg_recall = 0.0, 0.0
    else:
        task.seg_m_iou, task.seg_recall = calculate_mask_iou_and_recall(
            gt,
            pred,
            matching=matching,
            max_frames=task.max_frames,
        )
    return task


def segmentation_mp4_single_task(task: Task) -> Task:
    """Generate MP4 visualization of segmentation for single task"""
    from benchmark_pipelines.scores.transfer_bench.sam_pickle_to_mp4 import \
        sam_pkl_dict_to_mp4

    if not task.pred_segmentation_mp4_file:
        # Skip if saving segments is not needed
        # logger.warning("Skipping saving of pred segments as MP4: no path specified")
        return task

    assert task.pred_seg_dicts
    tmp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
    try:
        sam_pkl_dict_to_mp4(
            task.pred_seg_dicts,
            T=task.video_shape[0],
            H=task.video_shape[1],
            W=task.video_shape[2],
            fps=task.pred_fps,
            mp4_pth=tmp_file,
            max_frames=task.max_frames,
        )
        shutil.move(tmp_file, task.pred_segmentation_mp4_file)
    except Exception as e:  # noqa: BLE001
        logger.exception(e)
    finally:
        Path(tmp_file).unlink(missing_ok=True)
    return task


def unload_task_data_single_task(task: Task) -> Task:
    """Unload all input data to reduce load during results retrieval upon pipeline completion. Effectively purges
    heavier data, such as arrays, which are no longer of interest by the end of the pipeline to only retain metrics.
    Purged data is accessible by fetching Task state at earlier stages.
    """
    # Loaded video data
    task.gt_video_array = None
    task.pred_video_array = None
    task.pred_resized_video_array = None
    # Computed segments
    task.gt_seg_dicts = None
    task.pred_seg_dicts = None
    return task


def unload_task_data(tasks: list[Task]) -> list[Task]:
    """Unload task data for all tasks"""
    results = []
    for task in tasks:
        results.append(unload_task_data_single_task(task))
    return results


def depth_single_task(task: Task, depth_model) -> Task:
    """Process depth computation for single task"""
    from benchmark_pipelines.scores.transfer_bench.depth_to_mp4 import \
        convert_abs_depth_npy_to_mp4
    from benchmark_pipelines.scores.transfer_bench.metrics_canny_blur_depth import \
        compute_depth_error_video_sirmse

    try:
        assert (task.pred_resized_video_array is not None) and (task.gt_video_array is not None)

        pred_depth = depth_model.generate(task.pred_resized_video_array)
        pred_depth = pred_depth.astype(np.float64)  # absolute depth values in meters

        if should_compute(task.gt_depth_npy_file, task.force_recompute_gt_depth):
            gt_depth = depth_model.generate(task.gt_video_array)
            gt_depth = gt_depth.astype(np.float64)
            np.savez_compressed(task.gt_depth_npy_file, data=gt_depth)
        else:
            with np.load(task.gt_depth_npy_file) as npz_data:
                gt_depth = npz_data['data']

        if task.pred_depth_mp4_file:
            convert_abs_depth_npy_to_mp4(
                pred_depth,
                out_pth=task.pred_depth_mp4_file,
                fps=task.fps,
            )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Got exception {e} when computing depth maps from video")
        return task
    try:
        depth_si_rmse = compute_depth_error_video_sirmse(pred_depth, gt_depth)
        task.depth_si_rmse = depth_si_rmse
    except Exception as e:  # noqa: BLE001
        logger.error(f"Got exception {e} for compute_depth_error_video_sirmse")
        return task
    return task


def process_tasks_with_model(tasks: list[Task]) -> list[Task]:
    """Process tasks with outer model loop, inner data loop structure"""
    rank = get_rank()
    world_size = get_world_size()
    print0(f"Processing {len(tasks)} tasks across {world_size} ranks")

    # Distribute tasks to this rank
    tasks = distribute_list_to_rank(tasks)
    print0(f"Rank {rank} processing {len(tasks)} tasks")

    if not tasks:
        return []

    # Step 1: Load videos and captions (no models needed)
    print0(f"Rank {rank}: Loading videos with multi-threading...")
    tasks = load_videos(tasks, num_workers=8)

    print0(f"Rank {rank}: Processing captions...")
    tasks = load_captions(tasks)

    print0(f"Rank {rank}: Resizing videos with multi-threading...")
    tasks = resize_videos(tasks, num_workers=4)

    # Step 2: SAM Model Processing (load once, process all tasks)
    print0(f"Rank {rank}: Processing SAM segmentation...")
    sam_model = grounded_sam_v2.GroundedSAMV2()
    sam_model.setup()

    for i, task in enumerate(tqdm(tasks, desc="SAM segmentation", disable=(rank != 0))):
        tasks[i] = sam_single_task(task, sam_model)

    # Unload SAM model
    del sam_model
    torch.cuda.empty_cache()

    # Step 3: DOVER Model Processing (load once, process all tasks)
    print0(f"Rank {rank}: Computing DOVER scores...")
    dover_model = dover.DOVERVideoTechnicalScorer()
    dover_model.setup()

    for i, task in enumerate(tqdm(tasks, desc="DOVER scoring", disable=(rank != 0))):
        tasks[i] = dover_single_task(task, dover_model)

    # Unload DOVER model
    del dover_model
    torch.cuda.empty_cache()

    # Step 4: Depth Model Processing (load once, process all tasks)
    print0(f"Rank {rank}: Computing depth maps...")
    depth_model = video_depth_anything.VideoDepthAnything()
    depth_model.setup()

    for i, task in enumerate(tqdm(tasks, desc="Depth estimation", disable=(rank != 0))):
        tasks[i] = depth_single_task(task, depth_model)

    # Unload depth model
    del depth_model
    torch.cuda.empty_cache()

    # Step 5: Non-model processing (no GPU models needed)
    print0(f"Rank {rank}: Processing Canny edge detection...")
    for i, task in enumerate(tqdm(tasks, desc="Canny edge detection", disable=(rank != 0))):
        tasks[i] = canny_single_task(task)

    print0(f"Rank {rank}: Processing blur analysis with 8 processes...")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(blur_single_task, task) for task in tasks]
        for i, future in enumerate(tqdm(futures, desc="Blur analysis", disable=(rank != 0))):
            tasks[i] = future.result()

    print0(f"Rank {rank}: Computing mask IoU...")
    for i, task in enumerate(tqdm(tasks, desc="Mask IoU computation", disable=(rank != 0))):
        tasks[i] = mask_iou_single_task(task)

    print0(f"Rank {rank}: Generating segmentation MP4s...")
    for i, task in enumerate(tqdm(tasks, desc="Segmentation MP4 generation", disable=(rank != 0))):
        tasks[i] = segmentation_mp4_single_task(task)

    print0(f"Rank {rank}: Unloading data...")
    tasks = unload_task_data(tasks)

    return tasks


def launch_pipeline(
    tasks: list[Task],
) -> dict:
    print0(f"Processing {len(tasks)} tasks.")
    outputs = process_tasks_with_model(tasks)

    # Gather results from all ranks
    if get_world_size() > 1:
        outputs = gather_list_of_dict(outputs)

    # Only rank 0 processes and returns the final outputs
    if get_rank() == 0:
        assert outputs, "Processing failed to produce outputs"
        return process_outputs(outputs)
    else:
        return {}


def process_outputs(outputs: list) -> dict:
    keys_to_remove = [
        "gt_video_array",
        "pred_video_array",
        "pred_resized_video_array",
        "gt_seg_dicts",
        "pred_seg_dicts",
        "gt_foreground",
    ]
    per_video = [
        {
            k: v
            for k, v in attrs.asdict(res).items()
            # remove numpy arrays in the task object before serializing
            if k not in keys_to_remove and not isinstance(v, np.ndarray)
        }
        for res in outputs
    ]
    df = pd.DataFrame(per_video)
    results = {}
    results["global"] = df[METRICS].mean().to_dict()
    results["per_video"] = per_video

    return results


def prepare_tasks_from_filesystem(
    videos_path: str,
    gt_path: str,
    force_recompute_gt_seg: bool,
    force_recompute_gt_depth: bool,
) -> list[Task]:
    dataset_path = Path(videos_path)
    assert dataset_path.exists(), f"Could not find {dataset_path}"

    gt_dir = Path(gt_path)
    assert gt_dir.exists(), f"Could not find GT dir: {gt_dir}"

    tasks = []

    video_dir = dataset_path / "videos"
    assert video_dir.exists(), f"Video directory {video_dir} not found"

    resized_video_dir = dataset_path / "videos_orig_size"
    os.makedirs(resized_video_dir, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    assert videos, f"No videos found in {video_dir}"

    for video in videos:  # result video
        video_id = extract_video_id_from_pred_filename(video.name)
        gt_video_file = gt_dir / "videos" / f"{video_id}.mp4"
        video_caption_file = gt_dir / "captions" / f"{video_id}.json"

        if not gt_video_file.exists():
            logger.error(f"{gt_video_file} not available. Skipping...")
            continue

        if not video_caption_file.exists():
            logger.error(f"{video_caption_file} not available. Skipping...")
            continue

        task = Task(
            pred_video_uuid=video.stem,
            pred_video_file=video.as_posix(),
            force_recompute_gt_seg=force_recompute_gt_seg,
            force_recompute_gt_depth=force_recompute_gt_depth,
            pred_resized_video_file=(resized_video_dir / video.name).as_posix(),
            gt_video_file=gt_video_file.as_posix(),
            video_caption_file=video_caption_file.as_posix(),
        )

        # for SegIoU
        pred_video_file = Path(task.pred_video_file)
        seg_name = pred_video_file.name.replace(".mp4", ".pkl")
        # pred pkl file
        pred_pkl_seg = pred_video_file.parent.parent / "segmentation" / seg_name
        task.pred_segmentation_pkl_file = pred_pkl_seg.as_posix()  # will overwrite existing. Purposed to do this.
        task.pred_segmentation_mp4_file = pred_pkl_seg.as_posix().replace(".pkl", ".mp4")
        # gt pkl file
        task.gt_segmentation_pkl_file = get_seg_pkl_path_from_video_id(gt_path, video_id)
        task.gt_depth_npy_file = get_depth_npy_path_from_video_id(gt_path, video_id)

        # for blur, canny edge, depth etc.
        task.pred_canny_mp4_file = (pred_video_file.parent.parent / "canny" / pred_video_file.name).as_posix()
        task.pred_blur_mp4_file = (pred_video_file.parent.parent / "blur" / pred_video_file.name).as_posix()
        task.pred_depth_mp4_file = (pred_video_file.parent.parent / "depth" / pred_video_file.name).as_posix()

        assert Path(task.gt_video_file).exists(), f"GT video {task.gt_video_file} not found"
        assert Path(task.video_caption_file).exists(), f"GT caption {task.video_caption_file} not found"
        assert Path(task.pred_video_file).exists(), f"Input video {task.pred_video_file} not found"

        for file in [
            task.pred_canny_mp4_file,
            task.pred_blur_mp4_file,
            task.pred_depth_mp4_file,
            task.pred_segmentation_pkl_file,
            task.pred_resized_video_file,
        ]:
            if not Path(file).parent.exists():
                Path(file).parent.mkdir(parents=True, exist_ok=True)
        tasks.append(task)

    return tasks


@click.group()
def cli() -> None: ...


@cli.command()
@click.option(
    "--videos_path",
    type=str,
    required=True,
    help="Folder where files are located",
    show_default=True,
)
@click.option(
    "--gt_path",
    type=str,
    required=True,
    help="Ground truth directory.",
    show_default=True,
)
@click.option(
    "--output_path",
    type=str,
    default=None,
    help="Optional output json path. If none, will write in directory.",
    show_default=True,
)
@click.option(
    "--force_recompute_gt_seg/--no_force_recompute_gt_seg",
    default=False,
    help="If true, will run SAMv2 on GT videos and save the pkl to the GT data folder.",
    show_default=True,
)
@click.option(
    "--force_recompute_gt_depth/--no_force_recompute_gt_depth",
    default=False,
    help="If true, will run DepthAnything on GT videos and save the npy to the GT data folder.",
    show_default=True,
)
def calculate_metrics(
    videos_path: str,
    gt_path: str,
    output_path: Optional[str],
    force_recompute_gt_seg: bool,
    force_recompute_gt_depth: bool,
) -> None:
    # Initialize distributed processing
    dist_init()
    print0(f"Distributed processing enabled. Rank: {get_rank()}, World size: {get_world_size()}")

    if force_recompute_gt_seg:
        print0("\n\n=================\nWill recompute GT segs!")
    if force_recompute_gt_depth:
        print0("\n\n=================\nWill recompute GT depth!")

    dataset_path = Path(videos_path)
    assert dataset_path.exists(), f"Could not find {dataset_path}"
    if output_path is None:
        output_path = (dataset_path / "metrics.json").as_posix()

    tasks = prepare_tasks_from_filesystem(
        videos_path,
        gt_path,
        force_recompute_gt_seg,
        force_recompute_gt_depth,
    )

    try:
        results = launch_pipeline(tasks)
        # Only save from rank 0
        if get_rank() == 0:
            if results:
                with open(output_path, "w") as fp:
                    json.dump(results, fp, indent=4)
                print0(f"Evaluation run completed. See results at {output_path}")
    except Exception as e:
        print0("Evaluation run failed")
        raise e


if __name__ == "__main__":
    cli()
