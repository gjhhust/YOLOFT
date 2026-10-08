#!/usr/bin/env python
"""Run the XS-VID BoT-SORT tracking-by-detection reference pipeline.

The pipeline consumes precomputed detector boxes and extracts appearance
features with the supplied ReID checkpoint. A fresh tracker is created for each
video and its IDs are made globally unique for TAO evaluation.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from track_shards import (parse_shard, receipt_path, select_videos, write_receipt,
                          read_image, tracking_identity)

BOT_SORT_PARAMS = {
    "track_high_thresh": 0.1,
    "track_low_thresh": 0.001,
    "new_track_thresh": 0.2,
}


def load_gt_videos(gt_path):
    """Return {video_id: [ (image_id, file_name) sorted by frame_index ]}."""
    with open(gt_path, "r") as f:
        gt = json.load(f)
    by_video = defaultdict(list)
    for im in gt["images"]:
        by_video[im["video_id"]].append(
            (im["frame_index"], im["id"], im["file_name"])
        )
    videos = {}
    for vid, items in by_video.items():
        items.sort(key=lambda x: x[0])  # sort by frame_index
        videos[vid] = [(iid, fn) for (_, iid, fn) in items]
    return videos


def load_dets(det_path):
    """Return {image_id: np.float32 (N,6) [x1,y1,x2,y2,score,cls]} (xywh->xyxy)."""
    with open(det_path, "r") as f:
        dets = json.load(f)
    rows = defaultdict(list)
    for d in dets:
        x, y, w, h = d["bbox"]
        rows[d["image_id"]].append(
            [x, y, x + w, y + h, float(d["score"]), int(d["category_id"])]
        )
    out = {}
    for iid, lst in rows.items():
        out[iid] = np.asarray(lst, dtype=np.float32)
    return out


def build_tracker(reid_weights: Path, device: str):
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

    return create_tracker(
        tracker_type="botsort",
        tracker_config=get_tracker_config("botsort"),
        evolve_param_dict=BOT_SORT_PARAMS,
        per_class=False,
        reid_weights=reid_weights,
        device=device,
        half=False,
    )


def unique_track_ids(records):
    mapping, output = {}, []
    for record in records:
        key = (record["video_id"], record["track_id"])
        if key not in mapping:
            mapping[key] = (len(mapping), record["category_id"])
        item = record.copy()
        item["track_id"], item["category_id"] = mapping[key]
        output.append(item)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--det-json", type=Path, required=True)
    ap.add_argument("--reid-weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), help="Video-level index/count (default 0/1)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N videos (validation only; 0=all)")
    args = ap.parse_args()
    receipt_path(args.output).unlink(missing_ok=True)

    if not args.reid_weights.is_file():
        raise FileNotFoundError(args.reid_weights)
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "third_party" / "boxmot"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] GT videos from {args.annotation}")
    videos = load_gt_videos(args.annotation)
    print(f"[load] {len(videos)} videos")
    print(f"[load] dets from {args.det_json}")
    dets_by_image = load_dets(args.det_json)
    print(f"[load] dets for {len(dets_by_image)} image_ids")

    vid_ids = sorted(videos.keys())
    if args.limit > 0:
        vid_ids = vid_ids[: args.limit]
        print(f"[limit] only first {len(vid_ids)} videos: {vid_ids}")
    vid_ids = select_videos(vid_ids, args.shard)
    identity = tracking_identity("osnet-botsort", {"detections": args.det_json,
                                 "reid_checkpoint": args.reid_weights}, repo_root,
                                 Path(__file__), BOT_SORT_PARAMS)
    processed = {str(video): 0 for video in vid_ids}

    records = []
    empty = np.empty((0, 6), dtype=np.float32)

    for vi, vid in enumerate(vid_ids):
        tracker = build_tracker(args.reid_weights, args.device)
        frames = videos[vid]
        for image_id, file_name in frames:
            img = read_image(args.data_root / "images" / file_name, cv2.imread)
            dets = dets_by_image.get(image_id, empty)
            # Drop degenerate boxes (zero/negative w or h). StrongSort's
            # to_xyah does width/height, so a zero-height det yields inf/nan
            # that propagates into the cost matrix and crashes
            # linear_sum_assignment ("matrix contains invalid numeric
            # entries"). Tracker-agnostic, harmless for all trackers.
            if dets.shape[0] > 0:
                wh_ok = (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
                if not wh_ok.all():
                    dets = dets[wh_ok]
                    if dets.shape[0] == 0:
                        dets = empty
            out = tracker.update(dets, img)  # (M,8) [x1,y1,x2,y2,id,conf,cls,det_ind]
            processed[str(vid)] += 1
            out = np.asarray(out)
            if out.size == 0:
                continue
            for row in out:
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                records.append({
                    "image_id": int(image_id),
                    "video_id": int(vid),
                    "track_id": int(row[4]),
                    "category_id": int(row[6]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(row[5]),
                })
        print(f"[{vi + 1}/{len(vid_ids)}] video {vid}: {len(frames)} frames, "
              f"cumulative records={len(records)}")

    raw_path = args.output.with_name(f"{args.output.stem}_raw.json")
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"[write] raw -> {raw_path} ({len(records)} records)")

    uniq = unique_track_ids(records)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(uniq, f)
    write_receipt(args.output, args.annotation, videos, vid_ids, args.shard, len(uniq),
                  identity=identity, processed_frames=processed)
    n_tracks = len({r["track_id"] for r in uniq})
    print(f"[write] unique -> {args.output} ({len(uniq)} records, "
          f"{n_tracks} globally-unique track_ids)")


if __name__ == "__main__":
    main()
