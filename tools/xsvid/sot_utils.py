"""Small shared utilities for XS-VID SOT training."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def crop(image, box, factor, output_size, jitter=0.0, scale_jit=0.0):
    """Crop a square around an xywh box and return normalized cxcywh supervision."""
    x, y, width, height = box
    center_x, center_y = x + width / 2, y + height / 2
    side = max(factor * math.sqrt(max(width * height, 4.0)), 16.0)
    if scale_jit > 0:
        side *= math.exp((np.random.random() * 2 - 1) * scale_jit)
    crop_x, crop_y = center_x, center_y
    if jitter > 0:
        crop_x += (np.random.random() * 2 - 1) * jitter * side
        crop_y += (np.random.random() * 2 - 1) * jitter * side
    x0, y0 = crop_x - side / 2, crop_y - side / 2
    affine = np.array(
        [[output_size / side, 0, -x0 * output_size / side], [0, output_size / side, -y0 * output_size / side]],
        dtype=np.float32,
    )
    cropped = cv2.warpAffine(image, affine, (output_size, output_size), borderValue=(114, 114, 114))
    target = np.array([(center_x - x0) / side, (center_y - y0) / side, width / side, height / side], dtype=np.float32)
    return cropped, target


def read(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None


def to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


def giou_loss(prediction, target):
    """Mean 1-GIoU plus L1 loss for normalized cxcywh boxes."""
    def to_xyxy(boxes):
        return torch.stack(
            [boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
             boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2],
            dim=1,
        )

    prediction, target = to_xyxy(prediction), to_xyxy(target)
    left_top, right_bottom = torch.maximum(prediction[:, :2], target[:, :2]), torch.minimum(prediction[:, 2:], target[:, 2:])
    intersection = (right_bottom - left_top).clamp(0).prod(dim=1)
    pred_area = (prediction[:, 2] - prediction[:, 0]).clamp(0) * (prediction[:, 3] - prediction[:, 1]).clamp(0)
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area + target_area - intersection + 1e-7
    iou = intersection / union
    enclosure_left, enclosure_right = torch.minimum(prediction[:, :2], target[:, :2]), torch.maximum(prediction[:, 2:], target[:, 2:])
    enclosure_area = (enclosure_right - enclosure_left).clamp(0).prod(dim=1) + 1e-7
    giou = iou - (enclosure_area - union) / enclosure_area
    return (1 - giou).mean() + F.l1_loss(prediction, target)
