import cv2
import numpy as np
import torch
from schemas import eff_segmentation
from loguru import logger
from scipy.optimize import linear_sum_assignment

from benchmark_pipelines.scores.transfer_bench.sam_pickle_to_mp4 import sam_pkl_dict_to_instance_mask
from benchmark_pipelines.scores.transfer_bench.utils import safe_resize


def compute_iou(mask_A: np.ndarray, mask_B: np.ndarray, mask: np.ndarray) -> float:
    intersection_map = np.logical_and(mask_A, mask_B)
    union_map = np.logical_or(mask_A, mask_B)

    intersection = intersection_map[mask].sum()
    union = union_map[mask].sum()

    return intersection / union if union != 0 else 0.0


def get_iou_matrix(masks_A: list | np.ndarray, masks_B: list | np.ndarray, mask: np.ndarray) -> np.ndarray:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert all masks to tensors at once
    if isinstance(masks_A, list):
        masks_A_tensor = torch.stack([torch.from_numpy(m) for m in masks_A]).bool().to(device)
    else:
        masks_A_tensor = torch.from_numpy(masks_A).bool().to(device)

    if isinstance(masks_B, list):
        masks_B_tensor = torch.stack([torch.from_numpy(m) for m in masks_B]).bool().to(device)
    else:
        masks_B_tensor = torch.from_numpy(masks_B).bool().to(device)

    mask_tensor = torch.from_numpy(mask).bool().to(device)

    n = len(masks_A_tensor)
    m = len(masks_B_tensor)

    # Initialize IOU matrix on GPU
    iou_matrix = torch.zeros((n, m), dtype=torch.float32, device=device)

    # Calculate IOU for each pair using for loops
    for i in range(n):
        for j in range(m):
            # Calculate intersection and union for this pair
            intersection_map = masks_A_tensor[i] & masks_B_tensor[j]
            union_map = masks_A_tensor[i] | masks_B_tensor[j]

            # Apply the mask and calculate counts
            intersection = (intersection_map & mask_tensor).sum().float()
            union = (union_map & mask_tensor).sum().float()

            # Calculate IOU
            if union != 0:
                iou_matrix[i, j] = intersection / union
            else:
                iou_matrix[i, j] = 0.0

    return iou_matrix.cpu().numpy()


def calculate_mask_iou_and_recall(
    gt_masks: list[eff_segmentation.SAMV2Detection],
    res_masks: list[eff_segmentation.SAMV2Detection],
    matching: str = "max",
    threshold: float = 0.1,
    max_frames: int | None = None,
) -> tuple[float, float]:
    gt_phrase_index_dic = {}
    for i, gt_mask in enumerate(gt_masks):
        phrase = gt_mask.phrase
        if phrase not in gt_phrase_index_dic:
            gt_phrase_index_dic[phrase] = {i}
        else:
            gt_phrase_index_dic[phrase].add(i)

    gt_t, gt_h, gt_w = gt_masks[0].segmentation_mask_rle.mask_shape

    res_phrase_index_dic = {}
    for i, res_mask in enumerate(res_masks):
        phrase = res_mask.phrase
        if phrase not in res_phrase_index_dic:
            res_phrase_index_dic[phrase] = {i}
        else:
            res_phrase_index_dic[phrase].add(i)

    res_t, res_h, res_w = res_masks[0].segmentation_mask_rle.mask_shape

    gt_inst_masks = sam_pkl_dict_to_instance_mask(gt_masks, gt_t, gt_h, gt_w, max_frames)[..., 0]
    res_inst_masks = sam_pkl_dict_to_instance_mask(res_masks, res_t, res_h, res_w, max_frames)[..., 0]

    if res_inst_masks.shape[1:3] != gt_inst_masks.shape[1:3]:
        print(f"Resizing res_inst_masks {res_inst_masks.shape} to {gt_inst_masks.shape}")
        res_inst_masks = safe_resize(res_inst_masks, gt_w, gt_h, interpolation=cv2.INTER_NEAREST)

    # Handle the case when pred video has shorter duration (e.g. for baseline model VideoComposer, 16frames only)
    if res_inst_masks.shape[0] < gt_inst_masks.shape[0]:
        logger.warning(
            f"Instance masks have temporal duration: gt {gt_inst_masks.shape[0]} and {res_inst_masks.shape[0]}.\
                Truncating the gt frames to match the pred frames."
        )
        gt_inst_masks = gt_inst_masks[: res_inst_masks.shape[0]]

    updated_gt_masks = []
    for phrase in gt_phrase_index_dic.keys():
        selected_gt_mask_ids = gt_phrase_index_dic[phrase]
        union_gt_mask = np.zeros_like(gt_inst_masks)
        for id in selected_gt_mask_ids:
            # mask ids are from 0-N, but instance masks have 0 as background
            # with instances starting at 1
            union_gt_mask[gt_inst_masks == (id + 1)] = 255
        updated_gt_masks.append(union_gt_mask)

    updated_res_masks = []
    for phrase in res_phrase_index_dic.keys():
        selected_res_mask_ids = res_phrase_index_dic[phrase]
        union_res_mask = np.zeros_like(res_inst_masks)
        for id in selected_res_mask_ids:
            # mask ids are from 0-N, but instance masks have 0 as background
            # with instances starting at 1
            union_res_mask[res_inst_masks == (id + 1)] = 255
        updated_res_masks.append(union_res_mask)

    # print(f"Calculating iou on {len(updated_gt_masks)} gt masks and {len(updated_res_masks)} pred masks")

    foreground = np.ones_like(gt_inst_masks, dtype=np.bool_)

    iou_matrix = get_iou_matrix(updated_gt_masks, updated_res_masks, foreground)

    if matching == "max":
        matched_ious = np.max(iou_matrix, axis=1)
    elif matching == "hungarian":
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_ious = iou_matrix[row_ind, col_ind]
    else:
        raise NotImplementedError(f"Matching {matching} not available")

    if matched_ious.size == 0:
        return 0.0, 0.0

    ious = [x for x in matched_ious if x > threshold]
    m_iou = float(np.mean(ious)) if ious else 0.0
    recall = len(ious) / len(matched_ious) if len(matched_ious) > 0 else 0.0
    return m_iou, recall
