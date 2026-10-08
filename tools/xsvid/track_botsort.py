#!/usr/bin/env python3
"""Run the bundled BoT-SORT association on a YOLOFT-omni embedding dump."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from track_shards import (parse_shard, receipt_path, select_videos, write_receipt,
                          read_image, require_omni_frames, tracking_identity)


def load_videos(annotation: Path):
    dataset = json.loads(annotation.read_text(encoding="utf-8"))
    videos = defaultdict(list)
    for image in dataset["images"]:
        videos[image["video_id"]].append((image["frame_index"], image["id"], image["file_name"]))
    return {
        video_id: [(image_id, file_name) for _, image_id, file_name in sorted(frames)]
        for video_id, frames in videos.items()
    }


def load_dump(path: Path):
    with path.open("rb") as handle:
        dump = pickle.load(handle)
    detections = {}
    embeddings = {}
    for frames in dump["videos"].values():
        for frame in frames:
            image_id = int(frame["iid"])
            boxes = frame["boxes"].astype(np.float32)
            scores = frame["scores"].astype(np.float32)
            classes = frame["cls"].astype(np.float32)
            detections[image_id] = (
                np.concatenate((boxes, scores[:, None], classes[:, None]), axis=1)
                if len(boxes)
                else np.empty((0, 6), np.float32)
            )
            embeddings[image_id] = frame["emb"].astype(np.float32)
    return detections, embeddings


def unique_track_ids(records):
    mapping = {}
    next_id = 0
    output = []
    for record in records:
        key = (record["video_id"], record["track_id"])
        if key not in mapping:
            mapping[key] = (next_id, record["category_id"])
            next_id += 1
        track_id, category_id = mapping[key]
        item = record.copy()
        item["track_id"] = track_id
        item["category_id"] = category_id
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=parse_shard, default=(0, 1), help="Video-level index/count (default 0/1)")
    args = parser.parse_args()
    receipt_path(args.output).unlink(missing_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "third_party" / "boxmot"))
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

    videos = load_videos(args.annotation)
    selected_videos = select_videos(sorted(videos), args.shard)
    detections, embeddings = load_dump(args.dump)
    require_omni_frames(videos, selected_videos, detections, embeddings)
    params = {"track_high_thresh": 0.1, "track_low_thresh": 0.001, "new_track_thresh": 0.2}
    identity = tracking_identity("yoloft-omni-botsort", {"dump": args.dump}, repo_root, Path(__file__), params)
    processed = {str(video): 0 for video in selected_videos}
    records = []
    for video_index, video_id in enumerate(selected_videos, start=1):
        tracker = create_tracker(
            tracker_type="botsort",
            tracker_config=get_tracker_config("botsort"),
            evolve_param_dict={
                "track_high_thresh": 0.1,
                "track_low_thresh": 0.001,
                "new_track_thresh": 0.2,
            },
            per_class=False,
        )
        for image_id, file_name in videos[video_id]:
            image = read_image(args.data_root / "images" / file_name, cv2.imread)
            dets = detections[image_id]
            embs = embeddings[image_id]
            if len(dets):
                valid = (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
                dets, embs = dets[valid], embs[valid]
            if len(embs) != len(dets):
                raise RuntimeError(f"Detection/embedding mismatch for image {image_id}")
            output = np.asarray(tracker.update(dets, image, embs=embs))
            processed[str(video_id)] += 1
            if not output.size:
                continue
            for row in output:
                x1, y1, x2, y2 = map(float, row[:4])
                records.append(
                    {
                        "image_id": int(image_id),
                        "video_id": int(video_id),
                        "track_id": int(row[4]),
                        "category_id": int(row[6]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(row[5]),
                    }
                )
        print(f"BOTSORT video={video_index}/{len(selected_videos)} records={len(records)}", flush=True)

    unique = unique_track_ids(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(unique), encoding="utf-8")
    write_receipt(args.output, args.annotation, videos, selected_videos, args.shard, len(unique),
                  identity=identity, processed_frames=processed)
    print(
        f"XSVID_BOTSORT_OK records={len(unique)} "
        f"tracks={len({record['track_id'] for record in unique})} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
