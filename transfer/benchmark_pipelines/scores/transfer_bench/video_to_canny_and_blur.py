import cv2
import numpy as np

from benchmark_pipelines.scores.transfer_bench.utils import (
    read_video,
    safe_resize,
    should_save_or_overwrite,
    write_video,
)

PRESET_STRENGTH = "medium"
SCALE_FACTOR = 10


def get_canny_t(preset_strength: str) -> tuple[int, int]:
    if preset_strength == "none" or preset_strength == "very_low":
        t_lower, t_upper = 20, 50
    elif preset_strength == "low":
        t_lower, t_upper = 50, 100
    elif preset_strength == "medium":
        t_lower, t_upper = 100, 200
    elif preset_strength == "high":
        t_lower, t_upper = 200, 300
    elif preset_strength == "very_high":
        t_lower, t_upper = 300, 400
    else:
        raise ValueError(f"Unknown preset_strength requested: {preset_strength}")
    return t_lower, t_upper


def load_and_convert_rgb_mp4_to_canny_mp4(
    vid_fn: str,
    out_fn_canny_mp4: str | None,
    out_fn_canny_npy: str | None,
    preset_strength: str,
    max_frames: int | None = None,
    target_frames_shape: tuple | None = None,
    force_overwrite: bool = False,
) -> np.ndarray:
    frames, fps = read_video(vid_fn, max_frames)  # [T, H, W, 3]
    return convert_rgb_mp4_to_canny_mp4(
        frames, fps, out_fn_canny_mp4, out_fn_canny_npy, preset_strength, target_frames_shape, force_overwrite
    )


def convert_rgb_mp4_to_canny_mp4(
    vid_frames: np.ndarray,
    vid_fps: float,
    out_fn_canny_mp4: str | None,
    out_fn_canny_npy: str | None,
    preset_strength: str,
    target_frames_shape: tuple | None = None,
    force_overwrite: bool = False,
) -> np.ndarray:
    if target_frames_shape is not None and vid_frames.shape != target_frames_shape:
        H, W = target_frames_shape[1], target_frames_shape[2]
        print(f"Before canny: resizing video frames from {vid_frames.shape} to {target_frames_shape}")
        vid_frames = safe_resize(vid_frames, W, H, interpolation=cv2.INTER_LINEAR)

    t_lower, t_upper = get_canny_t(preset_strength)

    edge_maps = [cv2.Canny(img, t_lower, t_upper) for img in vid_frames]
    edge_maps = np.stack(edge_maps)

    if should_save_or_overwrite(out_fn_canny_mp4, force_overwrite):
        write_video(edge_maps, out_fn_canny_mp4, vid_fps)  # pyright: ignore[reportArgumentType]
    if should_save_or_overwrite(out_fn_canny_npy, force_overwrite):
        np.save(out_fn_canny_npy, edge_maps)  # pyright: ignore[reportArgumentType]
    return edge_maps


def apply_bilateral_filter(
    frames: np.ndarray,
    d: int = 30,
    sigma_color: float = 150,
    sigma_space: float = 100,  # these values are the default "medium" setting in our training
    iteration: int = 1,
) -> np.ndarray:
    """
    copied from i4 repo, projects/cosmos/diffusion/v1/datasets/augmentors/control_input.py
    to remove other dependency in that script
    """
    blurred_image = np.empty_like(frames)
    for i, _image_np in enumerate(frames):
        for _ in range(iteration):
            _image_np = cv2.bilateralFilter(_image_np, d, sigma_color, sigma_space)
        blurred_image[i] = _image_np
    return blurred_image


def apply_gaussian_filter(frames: np.ndarray, ksize: int = 5, sigmaX: float = 1.0) -> np.ndarray:
    blurred_image = [cv2.GaussianBlur(_image_np, (ksize, ksize), sigmaX=sigmaX) for _image_np in frames]
    blurred_image = np.stack(blurred_image)
    return blurred_image


def load_and_convert_rgb_mp4_to_blur_mp4(
    vid_fn: str,
    out_fn_blur_mp4: str | None,
    out_fn_blur_npy: str | None,
    blur_type: str = "gaussian",
    max_frames: int | None = None,
    target_frames_shape: tuple | None = None,
    force_overwrite: bool = False,
) -> np.ndarray:
    frames, fps = read_video(vid_fn, max_frames)  # [T, H, W, 3]
    return convert_rgb_mp4_to_blur_mp4(
        frames, fps, out_fn_blur_mp4, out_fn_blur_npy, blur_type, target_frames_shape, force_overwrite
    )


def convert_rgb_mp4_to_blur_mp4(
    vid_frames: np.ndarray,
    vid_fps: float,
    out_fn_blur_mp4: str | None,
    out_fn_blur_npy: str | None,
    blur_type: str = "gaussian",
    target_frames_shape: tuple | None = None,
    force_overwrite: bool = False,
) -> np.ndarray:
    if target_frames_shape is not None and vid_frames.shape != target_frames_shape:
        H, W = target_frames_shape[1], target_frames_shape[2]
        print(f"Before blur: resizing video frames from {vid_frames.shape} to {target_frames_shape}")
        vid_frames = safe_resize(vid_frames, W, H, interpolation=cv2.INTER_LINEAR)

    if blur_type == "bilateral":
        blur_maps = apply_bilateral_filter(vid_frames, d=30, sigma_color=150, sigma_space=100)
    else:
        blur_maps = apply_gaussian_filter(vid_frames, ksize=25, sigmaX=12.5)

    if should_save_or_overwrite(out_fn_blur_mp4, force_overwrite):
        write_video(
            blur_maps,
            out_fn_blur_mp4,  # pyright: ignore[reportArgumentType]
            vid_fps,
        )
    if should_save_or_overwrite(out_fn_blur_npy, force_overwrite):
        np.save(
            out_fn_blur_npy,  # pyright: ignore[reportArgumentType]
            blur_maps,
        )
    return blur_maps
