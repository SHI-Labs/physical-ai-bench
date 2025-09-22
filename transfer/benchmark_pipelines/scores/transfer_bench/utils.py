from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import attrs
# import databases.types as db_types
import cv2
import decord
import imageio
import numpy as np
from loguru import logger
# from sqlalchemy.orm import Session, sessionmaker

from benchmark_pipelines.utils.data_model import MetricScore


@attrs.define
class Metric:
    foreground: float | int | None
    background: float | int | None = None


def read_video(video_path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    reader = decord.VideoReader(video_path)
    fps = reader.get_avg_fps()
    max_frames = len(reader) if max_frames is None else min(max_frames, len(reader))
    frame_ids = list(range(max_frames))
    batch = reader.get_batch(frame_ids)
    # Handle different types of tensors/arrays
    if hasattr(batch, 'asnumpy'):
        frames = batch.asnumpy()
    elif hasattr(batch, 'numpy'):
        frames = batch.numpy()
    else:
        frames = np.array(batch)
    return frames, fps


def write_video(frames: np.ndarray, output_path: str, fps: float = 30) -> None:
    """
    expects a sequence of [H, W, 3] or [H, W] frames
    """
    with imageio.get_writer(output_path, fps=fps, macro_block_size=8) as writer:
        for frame in frames:
            if len(frame.shape) == 2:  # single channel
                frame_3channel = frame[:, :, None].repeat(3, axis=2)
            else:
                frame_3channel = frame
            writer.append_data(frame_3channel)  # pyright: ignore[reportAttributeAccessIssue]


def get_video_stats(video_path: str) -> dict:
    reader = decord.VideoReader(video_path)
    fps = reader.get_avg_fps()
    frame_count = len(reader)
    height, width = reader[0].asnumpy().shape[0:2]
    del reader
    return {
        "fps": fps,
        "height": height,
        "width": width,
        "frame_count": frame_count,
    }


def safe_resize(
    frames: np.ndarray, target_width: int, target_height: int, interpolation: Any = cv2.INTER_NEAREST
) -> np.ndarray:
    resized_frames = []
    for frame in frames:
        # Check if frame is valid
        if frame is None or frame.size == 0:
            continue

        # Convert to supported data type
        if frame.dtype not in [np.uint8, np.float32, np.float64]:
            frame_float = frame.astype(np.float32)
        else:
            frame_float = frame

        # Resize
        resized = cv2.resize(frame_float, (target_width, target_height), interpolation=interpolation)
        resized_frames.append(resized)

    return np.array(resized_frames)


def fast_unique_uint8(array: np.ndarray) -> np.ndarray:
    assert array.dtype == np.uint8, f"Array must be uint8, got {array.dtype}"
    nums = np.zeros((256,), dtype=np.bool_)
    nums[array.ravel()] = True
    return np.where(nums)[0]


def numpy_array_to_video_bytes(video_array: np.ndarray, fps: int = 30) -> bytes:
    # Assuming video_array is numpy array with shape (T, H, W, 3)
    buffer = BytesIO()
    with imageio.get_writer(
        buffer,
        format="mp4",  # pyright: ignore[reportArgumentType]
        fps=fps,
        macro_block_size=8,
    ) as writer:
        for frame in video_array:
            writer.append_data(frame)  # pyright: ignore[reportAttributeAccessIssue]
    buffer.seek(0)
    video_bytes = buffer.read()
    return video_bytes


def should_save_or_overwrite(filename: str | Path | None, force: bool = False) -> bool:
    """Determine whether a file should be (over)written or discarded.

    Args:
        filename: Path to the file (local path, Path object, or enriched S3 URI). Can be None or empty string.
        force: If True, always return True to force overwrite regardless of file existence.

    Returns:
        True if the file should be written/overwritten, False if it should be discarded.
        Returns False if filename is None or empty string.

    Examples:
        >>> should_save_or_overwrite("/path/to/file.txt", force=True)  # Always True
        True
        >>> should_save_or_overwrite("profile@s3://bucket/file.txt")  # True if S3 file doesn't exist
        True
        >>> should_save_or_overwrite("")  # False for empty string
        False
        >>> should_save_or_overwrite(None)  # False for None
        False
    """
    # Handle None or empty string cases - return False (don't save)
    if not filename:
        return False

    # Early return for force mode
    if force:
        return True

    try:
        # Convert Path object to string for S3 URI check
        return not Path(filename).exists()
    except Exception as e:  # noqa: BLE001
        # If we can't determine file existence due to errors (permissions, network, etc.),
        # err on the side of caution and suggest saving/overwriting
        logger.warning(f"Error checking file existence for {filename}: {e}")
        return True


def should_compute(filename: str | Path | None, force: bool = False) -> bool:
    """Determine whether a file should be (re)created, potentially forcefully.

    Args:
        filename: Path to the file (local path or enriched S3 URI). Can be None or empty string.
        force: If True, always return True to force recomputation regardless of file existence.

    Returns:
        True if the file should be computed/created, False if it already exists and force=False.

    Examples:
        >>> should_compute("/path/to/file.txt", force=True)  # Always True
        True
        >>> should_compute("profile@s3://bucket/file.txt")  # True if S3 file doesn't exist
        True
        >>> should_compute("")  # True for empty string
        True
        >>> should_compute(None)  # True for None
        True
    """
    # Early return for force mode
    if force:
        return True

    # Handle None or empty string cases
    if not filename:
        return True

    try:
        return not Path(filename).exists()
    except Exception as e:  # noqa: BLE001
        # If we can't determine file existence due to errors (permissions, network, etc.),
        # err on the side of caution and suggest recomputation
        logger.warning(f"Error checking file existence for {filename}: {e}")
        return True


# def get_session(db: db_types.PostgresDB) -> Session:
#     engine = db.make_sa_engine_v2()
#     session_builder = sessionmaker(bind=engine)
#     session = session_builder()
#     return session


def prepare_metric_scores(metrics: list[str], data: dict[str, float], sep: str = "_") -> list[MetricScore]:
    """Categorise and group metric results into a list of MetricScore

    Assumes `metrics` contains elements following the format: `<category><separator><subcategory>`. If `separator` is
    not found in the metric, it defaults to `category=metric,subcategory=""`.
    """
    agg = defaultdict(dict)
    for metric_name in metrics:
        if sep in metric_name:
            metric_category, metric_subcategory = metric_name.rsplit(sep, maxsplit=1)
        else:
            metric_category, metric_subcategory = (metric_name, "")
        if metric_value := data[metric_name]:
            agg[metric_category][metric_subcategory] = metric_value
    return [
        MetricScore(metric_name=metric_category, metric_subcategory_value=metric_subcategory_value)
        for metric_category, metric_subcategory_value in agg.items()
    ]
