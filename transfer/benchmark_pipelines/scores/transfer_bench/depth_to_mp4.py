import shutil
import uuid

import numpy as np

from benchmark_pipelines.scores.transfer_bench.utils import write_video


def convert_abs_depth_npy_to_mp4(depth_npy: np.ndarray, out_pth: str, fps: int | None) -> None:
    if fps is None:
        fps = 30
    # Normalize and convert to uint8 for visualization
    assert depth_npy.ndim == 3  # T, H W
    depth = (depth_npy - depth_npy.min()) / (depth_npy.max() - depth_npy.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    tmp_file = f"/tmp/{uuid.uuid4()}.mp4"
    write_video(depth, tmp_file, fps)
    shutil.move(tmp_file, out_pth)
