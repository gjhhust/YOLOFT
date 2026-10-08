#!/usr/bin/env python3
"""Build the legacy YOLOFT MOT/SOT training views from canonical XS-VID train.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--skip-sot-layout", action="store_true")
    args = parser.parse_args()

    annotation_path = args.annotation or args.data_root / "annotations" / "train.json"
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in dataset["images"]}
    videos = {video["id"]: video for video in dataset["videos"]}
    frame_rows = defaultdict(lambda: defaultdict(list))
    tracks = defaultdict(list)
    for annotation in dataset["annotations"]:
        if annotation.get("ignore") or annotation["category_id"] == 4:
            continue
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        row = [x, y, x + width, y + height, annotation["category_id"], annotation["track_id"]]
        frame_rows[image["video_id"]][str(image.get("frame_index", image.get("frame_id", 0)))].append(
            {"file_name": image["file_name"], "res": [row]}
        )
        tracks[(image["video_id"], annotation["track_id"])].append((image, annotation["bbox"]))

    mot_index = {}
    for video_id, frame_map in frame_rows.items():
        normalized = {}
        for frame_id, records in frame_map.items():
            boxes = []
            for record in records:
                boxes.extend(record["res"])
            normalized[frame_id] = {"file_name": records[0]["file_name"], "res": boxes}
        mot_index[str(video_id)] = normalized
    mot_path = args.data_root / "annotations" / "mot" / "train_omni.json"
    mot_path.parent.mkdir(parents=True, exist_ok=True)
    mot_path.write_text(json.dumps(mot_index), encoding="utf-8")

    sot_sequences = 0
    if not args.skip_sot_layout:
        sot_root = args.data_root / "sot" / "train"
        sot_root.mkdir(parents=True, exist_ok=True)
        for (video_id, track_id), rows in sorted(tracks.items()):
            rows.sort(key=lambda row: row[0].get("frame_index", row[0].get("frame_id", 0)))
            if len(rows) < 2:
                continue
            name = f"{videos[video_id].get('name', f'video_{video_id}')}_track_{track_id}"
            sequence = sot_root / name
            sequence.mkdir(exist_ok=True)
            boxes = []
            for index, (image, box) in enumerate(rows):
                source = args.data_root / "images" / image["file_name"]
                destination = sequence / f"{index:07d}{source.suffix.lower()}"
                if not destination.exists():
                    link_or_copy(source, destination)
                boxes.append(",".join(f"{value:.6f}" for value in box))
            (sequence / "groundtruth.txt").write_text("\n".join(boxes) + "\n", encoding="utf-8")
            sot_sequences += 1

    report = {
        "annotation": str(annotation_path),
        "mot_index": str(mot_path),
        "mot_videos": len(mot_index),
        "sot_train_sequences": sot_sequences,
        "sot_layout_created": not args.skip_sot_layout,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
