#!/usr/bin/env python3
"""Standalone scorer for XS-VID detection predictions.

Scores a COCO-format prediction JSON against the canonical staging test.json
using the XS-VID custom COCO evaluator (from the legacy codes/yoloft tree).

The default leaves ``catIds`` unset, matching the retained XS-VID evaluator.
Canonical category-4 annotations carry ``ignore`` and ``iscrowd`` flags, so
they remain in the file while being excluded during metric accumulation.

Usage:
  python tools/xsvid/score_detection.py \
    --gt /path/to/XS-VID-v2/annotations/test.json \
    --pred /path/to/predictions.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CANONICAL_SCORED_CATIDS = None  # None = retain all IDs and honor the annotation-level ignore flags.


def md5sum(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth COCO JSON (staging test.json)")
    parser.add_argument("--pred", type=Path, required=True, help="Prediction COCO JSON")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON for metrics")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from pycocotools.coco import COCO
    from ultralytics.data.cocoeval_xs_vid import COCOeval

    print(f"GT: {args.gt} (md5={md5sum(args.gt)})", flush=True)
    print(f"PRED: {args.pred} (md5={md5sum(args.pred)})", flush=True)
    print(f"SCORED_CATIDS: {CANONICAL_SCORED_CATIDS} (annotation ignore flags honored)", flush=True)

    anno = COCO(str(args.gt))
    pred = anno.loadRes(str(args.pred))

    ev = COCOeval(anno, pred, "bbox")
    if CANONICAL_SCORED_CATIDS is not None:
        ev.params.catIds = CANONICAL_SCORED_CATIDS
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats.tolist()
    precision = ev.eval["precision"]

    def area_ap(area_label: str) -> float:
        area_index = ev.params.areaRngLbl.index(area_label)
        max_det_index = ev.params.maxDets.index(100)
        values = precision[:, :, :, area_index, max_det_index]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    metrics = {
        "AP": stats[0] if len(stats) > 0 else None,
        "AP50": stats[1] if len(stats) > 1 else None,
        "AP75": stats[2] if len(stats) > 2 else None,
        "AP_es": area_ap("0-12"),
        "AP_rs": area_ap("12-20"),
        "AP_gs": area_ap("20-32"),
        "AP_small": stats[3] if len(stats) > 3 else None,
        "AP_medium": stats[4] if len(stats) > 4 else None,
        "AP_large": stats[5] if len(stats) > 5 else None,
        "AR1": stats[6] if len(stats) > 6 else None,
        "AR10": stats[7] if len(stats) > 7 else None,
        "AR100": stats[8] if len(stats) > 8 else None,
        "AR_small": stats[9] if len(stats) > 9 else None,
        "AR_medium": stats[10] if len(stats) > 10 else None,
        "AR_large": stats[11] if len(stats) > 11 else None,
    }

    print(f"\n=== SCORES (catIds={CANONICAL_SCORED_CATIDS}) ===")
    print(f"AP    = {metrics['AP']:.4f}" if metrics["AP"] is not None else "AP    = N/A")
    print(f"AP50  = {metrics['AP50']:.4f}" if metrics["AP50"] is not None else "AP50  = N/A")
    print(f"AP75  = {metrics['AP75']:.4f}" if metrics["AP75"] is not None else "AP75  = N/A")
    print(f"AP_m  = {metrics['AP_medium']:.4f}" if metrics["AP_medium"] is not None else "AP_m  = N/A")
    print(f"AP_l  = {metrics['AP_large']:.4f}" if metrics["AP_large"] is not None else "AP_l  = N/A")

    output = args.output or args.pred.with_suffix(".score.json")
    record = {
        "gt_json": str(args.gt),
        "gt_md5": md5sum(args.gt),
        "pred_json": str(args.pred),
        "pred_md5": md5sum(args.pred),
        "scored_catIds": CANONICAL_SCORED_CATIDS,
        "metrics": metrics,
    }
    with open(output, "w") as f:
        json.dump(record, f, indent=2)
    print(f"SCORE_RECORD {output}", flush=True)


if __name__ == "__main__":
    main()
