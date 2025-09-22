from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from benchmark_pipelines.scores.transfer_bench.utils import Metric, fast_unique_uint8, safe_resize


# blur
def apply_bilateral_filter(
    frames: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    iteration: int = 1,
) -> np.ndarray:
    """
    copied from projects/cosmos/diffusion/v1/datasets/augmentors/control_input.py
    to remove other dependency in that script
    """
    blurred_image = np.empty_like(frames)
    for i, _image_np in enumerate(frames):
        for _ in range(iteration):
            _image_np = cv2.bilateralFilter(_image_np, d, sigma_color, sigma_space)
        blurred_image[i] = _image_np
    return blurred_image


def apply_gaussian_filter(frames: np.ndarray, ksize: int = 5, sigmaX: float = 1.0) -> np.ndarray:
    """
    copied from projects/cosmos/diffusion/v1/datasets/augmentors/control_input.py
    to remove other dependency in that script
    """
    blurred_image = np.empty_like(frames)
    blurred_image = [cv2.GaussianBlur(_image_np, (ksize, ksize), sigmaX=sigmaX) for _image_np in frames]
    blurred_image = np.stack(blurred_image)
    return blurred_image


def compute_blur_error_blur_video(
    pred_frames: np.ndarray, gt_frames: np.ndarray, mask: Optional[np.ndarray] = None, metric_name: str = "ssim"
) -> Metric:
    """
    apply the same blur strengh to the generated video and compare L2 loss with input
    both inputs are for entire video, np.array, of shape [T,H,W,C]
    """
    T, H, W, _ = gt_frames.shape
    min_frames = gt_frames.shape[0]
    if gt_frames.shape[0] > pred_frames.shape[0]:
        min_frames = pred_frames.shape[0]
        print(
            f"WARNING: frame count mismatch {gt_frames.shape} vs "
            f"{pred_frames.shape}. Using the minimum of the two: {min_frames} frames"
        )
        gt_frames = gt_frames[:min_frames]

    if gt_frames.shape != pred_frames.shape:
        print(
            f"WARNING: Blur metric, shape mismatch, RESHAPING pred: {pred_frames.shape} to gt shape {gt_frames.shape}"
        )
        pred_frames = safe_resize(pred_frames, W, H, interpolation=cv2.INTER_LINEAR)[:min_frames]
    gt_frames[:min_frames]

    result_imgs = []
    T = gt_frames.shape[0]
    for t in range(T):
        if metric_name == "ssim":
            # ssim func supports computing rgb by feeding the channel_axis arg.
            # internally it computes ssim for each channel and then averages them
            res = ssim(
                gt_frames[t],
                pred_frames[t],
                data_range=gt_frames[t].max() - gt_frames[t].min(),
                full=True,
                channel_axis=2,
            )
            result_imgs.append(res[1].mean(axis=(2,)))  # per-pixel scores, [H, W]
        elif metric_name == "mse":
            mse = (gt_frames[t] - pred_frames[t]) ** 2
            mse = mse.mean(-1)  # [H, W]
            result_imgs.append(mse)
        else:
            raise NotImplementedError()

    result_tensor = np.stack(result_imgs)  # [T, H, W]

    metric = float(np.mean(result_tensor))

    return metric


# Canny edge metrics
def compute_canny_error_video_f1(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[Metric, Metric, Metric]:
    """
    both inputs are np.uint8 type array of shape [T,H,W]
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    _, H, W = gt.shape
    min_frames = gt.shape[0]
    if gt.shape[0] > pred.shape[0]:
        min_frames = pred.shape[0]
        print(
            f"WARNING: frame count mismatch {gt.shape} vs {pred.shape}. "
            f"Using the minimum of the two: {min_frames} frames"
        )
        gt = gt[:min_frames]

    if gt.shape != pred.shape:
        print(f"WARNING: Canny metric, shape mismatch, RESHAPING pred: {pred.shape} to gt shape {gt.shape}")
        pred = safe_resize(pred, W, H, interpolation=cv2.INTER_NEAREST)[:min_frames]

    # Flatten the arrays across all frames
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    if tuple(fast_unique_uint8(pred_flat)) != (0, 1):
        raise ValueError("Values in pred tensor should be 0 or 1")
    if tuple(fast_unique_uint8(gt_flat)) != (0, 1):
        raise ValueError("Values in gt tensor should be 0 or 1")

    pred_flat_filt = pred_flat
    gt_flat_filt = gt_flat

    true_positives = np.sum((pred_flat_filt == 1) & (gt_flat_filt == 1))
    false_positives = np.sum((pred_flat_filt == 1) & (gt_flat_filt == 0))
    false_negatives = np.sum((pred_flat_filt == 0) & (gt_flat_filt == 1))

    # Compute precision, recall, and F1 Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metric_f1 = float(f1_score)
    metric_precision = float(precision)
    metric_recall = float(recall)

    return metric_f1, metric_precision, metric_recall


# Depth metrics
def compute_depth_metrics_absrel_aux(
    gt_depth: np.ndarray, pred_depth: np.ndarray, mask: Optional[np.ndarray] = None
) -> dict:
    # Ensure inputs are numpy arrays
    gt_depth = np.asarray(gt_depth, dtype=np.float32)
    pred_depth = np.asarray(pred_depth, dtype=np.float32)

    # Create mask for valid pixels
    if mask is None:
        mask = gt_depth > 0
    else:
        mask = np.logical_and(mask, gt_depth > 0)

    # Add additional masking for pred_depth to avoid potential zeros
    mask = np.logical_and(mask, pred_depth > 0)

    # Flatten arrays for easier processing
    gt_valid = gt_depth[mask]
    pred_valid = pred_depth[mask]

    # Avoid division by zero
    if gt_valid.size == 0:
        return {"abs_rel": np.nan, "delta1": np.nan, "delta2": np.nan, "delta3": np.nan}

    # Compute Absolute Relative Error
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)

    # Compute thresholded accuracy (delta)
    threshold_1 = 1.25
    threshold_2 = 1.25**2
    threshold_3 = 1.25**3

    # Calculate ratios (both directions to handle overestimation and underestimation)
    # Use np.divide with 'out' parameter to handle division by zero
    gt_pred_ratio = np.divide(gt_valid, pred_valid, out=np.ones_like(gt_valid), where=pred_valid != 0)
    pred_gt_ratio = np.divide(pred_valid, gt_valid, out=np.ones_like(pred_valid), where=gt_valid != 0)

    ratio = np.maximum(gt_pred_ratio, pred_gt_ratio)

    # Calculate delta metrics
    delta1 = np.mean((ratio < threshold_1).astype(np.float32))
    delta2 = np.mean((ratio < threshold_2).astype(np.float32))
    delta3 = np.mean((ratio < threshold_3).astype(np.float32))

    return {"abs_rel": abs_rel, "delta1": delta1, "delta2": delta2, "delta3": delta3}


def compute_depth_error_video_absrel(gt_video: np.ndarray, pred_video: np.ndarray) -> dict:
    """
    Evaluate depth metrics for video sequences.

    WARNING: (qianlim) absrel is sensitive to outliers. Can cause very large value if a few gt pixels are close to 0.
    We do not use this in CosmosTransfer eval. Use the rMSE instead, see below.

    Args:
        gt_video: Ground truth depth video, numpy array of shape [T, H, W]
        pred_video: Predicted depth video, numpy array of shape [T, H, W]

    Returns:
        Dictionary of metrics averaged over all frames
    """
    # Verify shapes match
    assert gt_video.shape == pred_video.shape, "Ground truth and prediction shapes must match"

    # Initialize metrics storage
    frame_metrics = []

    # Process each frame
    for t in range(gt_video.shape[0]):
        metrics = compute_depth_metrics_absrel_aux(gt_video[t], pred_video[t])
        frame_metrics.append(metrics)

    # Compute average metrics across all frames
    avg_metrics = {}
    for key in frame_metrics[0].keys():
        values = [m[key] for m in frame_metrics]
        avg_metrics[key] = np.mean(values)

    return avg_metrics


def compute_depth_error_video_sirmse(
    pred_depth: np.ndarray, gt_depth: np.ndarray, mask: Optional[np.ndarray] = None, compute_in_log_space: bool = False
) -> Metric:
    """
    Compute depth estimation metrics between ground truth and predicted depth maps.
    https://www.cs.cornell.edu/projects/megadepth/paper.pdf, eq. 2.
    Robust to outliers
    Args:
        gt_depth: Ground truth depth maps, numpy array of shape [T, H, W]
    """
    # cast type to prevent overflow
    pred_depth = pred_depth.astype(np.float64)
    gt_depth = gt_depth.astype(np.float64)

    metric = {}
    for key in ["foreground", "background"]:
        _, H, W = gt_depth.shape
        min_frames = gt_depth.shape[0]
        if gt_depth.shape[0] > pred_depth.shape[0]:
            min_frames = pred_depth.shape[0]
            print(
                f"WARNING: frame count mismatch {gt_depth.shape} vs {pred_depth.shape}. "
                f"Using the minimum of the two: {min_frames} frames"
            )
            gt_depth = gt_depth[:min_frames]

        if gt_depth.shape != pred_depth.shape:
            print(
                f"WARNING: Depth metric, shape mismatch, RESHAPING pred_depth: "
                f"{pred_depth.shape} to gt shape {gt_depth.shape}"
            )
            pred_depth = safe_resize(pred_depth, W, H, interpolation=cv2.INTER_NEAREST)[:min_frames]

        if mask is None and key == "background":
            continue
        mask_valid = np.ones_like(gt_depth, dtype=np.bool_) if mask is None else mask
        if key == "background":
            mask_valid = ~mask_valid

        mask_valid = mask_valid[:min_frames]
        if mask_valid.shape != gt_depth.shape:
            raise ValueError(f"Inconsistent mask shape {mask_valid.shape} vs gt {gt_depth.shape}")

        gt_valid_mask = np.logical_and(gt_depth > 0, mask_valid)
        pred_valid_mask = np.logical_and(pred_depth > 0, mask_valid)
        valid_mask = np.logical_and(gt_valid_mask, pred_valid_mask)

        frame_si_mse = np.zeros(pred_depth.shape[0])
        frame_si_rmse = np.zeros(pred_depth.shape[0])

        for t in range(pred_depth.shape[0]):
            curr_valid = valid_mask[t]
            if np.sum(curr_valid) == 0:
                continue
            curr_pred = pred_depth[t][curr_valid]
            curr_gt = gt_depth[t][curr_valid]

            if not compute_in_log_space:
                ratio = np.median(curr_gt) / np.median(curr_pred)
                scaled_pred = curr_pred * ratio
                frame_si_mse[t] = np.mean((curr_gt - scaled_pred) ** 2)
                frame_si_rmse[t] = np.sqrt(frame_si_mse[t])
            else:
                log_pred = np.log(curr_pred)
                log_gt = np.log(curr_gt)

                log_diff = log_pred - log_gt
                mean_log_diff = np.mean(log_diff)

                term1 = np.mean(log_diff**2)
                term2 = mean_log_diff**2

                # SI-MSE and SI-RMSE
                frame_si_mse[t] = term1 - term2
                frame_si_rmse[t] = np.sqrt(term1 - term2)
        valid_frames = np.where(valid_mask.sum(axis=(1, 2)) > 0)[0]
        metric[key] = float(np.mean(frame_si_rmse)) if len(valid_frames) > 0 else 0.0

    return metric["foreground"]
