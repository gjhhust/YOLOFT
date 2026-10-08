#!/usr/bin/env python3
"""Score XS-VID SOT predictions with the paper tracking protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    return np.concatenate((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]), axis=1)


def sequence_errors(prediction: np.ndarray, ground_truth: np.ndarray):
    ground_truth = xyxy_to_xywh(ground_truth)
    prediction[0] = ground_truth[0]
    prediction_center = prediction[:, :2] + 0.5 * (prediction[:, 2:] - 1.0)
    ground_truth_center = ground_truth[:, :2] + 0.5 * (ground_truth[:, 2:] - 1.0)
    center_error = np.linalg.norm(prediction_center - ground_truth_center, axis=1)
    normalized_error = np.linalg.norm(
        prediction_center / ground_truth[:, 2:] - ground_truth_center / ground_truth[:, 2:],
        axis=1,
    )
    left_top = np.maximum(prediction[:, :2], ground_truth[:, :2])
    right_bottom = np.minimum(
        prediction[:, :2] + prediction[:, 2:] - 1.0,
        ground_truth[:, :2] + ground_truth[:, 2:] - 1.0,
    )
    intersection = np.maximum(right_bottom - left_top + 1.0, 0).prod(axis=1)
    union = prediction[:, 2:].prod(axis=1) + ground_truth[:, 2:].prod(axis=1) - intersection
    return intersection / union, center_error, normalized_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = [
        name.strip()
        for name in (args.data_root / "sot" / "test.txt").read_text(encoding="utf-8").splitlines()
        if name.strip()
    ]
    overlap_curves = []
    precision_curves = []
    normalized_precision_curves = []
    mean_ious = []
    frames = 0
    for name in names:
        prediction_path = args.predictions / f"{name}.txt"
        ground_truth_path = args.data_root / "sot" / "sequences" / name / "groundtruth.txt"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        prediction = np.loadtxt(prediction_path, delimiter=",").reshape(-1, 4)
        ground_truth = np.loadtxt(ground_truth_path, delimiter=",").reshape(-1, 4)
        if len(prediction) != len(ground_truth):
            raise RuntimeError(f"Frame count mismatch in {name}: {len(prediction)} vs {len(ground_truth)}")
        overlaps, center_errors, normalized_errors = sequence_errors(prediction, ground_truth)
        overlap_curves.append((overlaps[:, None] > np.arange(0.0, 1.0001, 0.05)).mean(axis=0))
        precision_curves.append((center_errors[:, None] <= np.arange(0, 51)).mean(axis=0))
        normalized_precision_curves.append(
            (normalized_errors[:, None] <= np.arange(0, 51) / 100.0).mean(axis=0)
        )
        mean_ious.append(float(overlaps.mean()))
        frames += len(prediction)

    overlap_curve = np.asarray(overlap_curves).mean(axis=0)
    precision_curve = np.asarray(precision_curves).mean(axis=0)
    normalized_precision_curve = np.asarray(normalized_precision_curves).mean(axis=0)
    report = {
        "sequences": len(names),
        "frames": frames,
        "success_auc": float(overlap_curve.mean()),
        "overlap_precision_50": float(overlap_curve[10]),
        "overlap_precision_75": float(overlap_curve[15]),
        "precision_20px": float(precision_curve[20]),
        "normalized_precision_20": float(normalized_precision_curve[20]),
        "mean_iou": float(np.mean(mean_ious)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"XSVID_SOT_SCORE_OK {json.dumps(report, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
