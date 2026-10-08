"""Video-level tracking partitions and completion receipts (standard library only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in value.split("/"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--shard must be index/count") from exc
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("--shard requires count >= 1 and 0 <= index < count")
    return index, count


def select_videos(video_ids: list, shard: tuple[int, int]) -> list:
    index, count = shard
    return video_ids[index::count]


def receipt_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".shard.json")


def annotation_sha256(annotation: Path) -> str:
    digest = hashlib.sha256()
    with annotation.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_omni_frames(videos, selected, detections, embeddings) -> None:
    for video in selected:
        for image_id, _ in videos[video]:
            if image_id not in detections or image_id not in embeddings:
                raise ValueError(f"Missing omni dump frame {image_id} in video {video}")


def read_image(path: Path, imread):
    image = imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Missing or unreadable image: {path}")
    return image


def tracking_identity(pipeline: str, inputs: dict, repo_root: Path, script: Path, params: dict) -> dict:
    source_root = repo_root / "third_party/boxmot"
    sources = {path.relative_to(source_root).as_posix(): annotation_sha256(path)
               for path in sorted(source_root.rglob("*.py"))}
    config = source_root / "boxmot/configs/trackers/botsort.yaml"
    return {"pipeline": pipeline,
            "inputs_sha256": {name: annotation_sha256(path) for name, path in inputs.items()},
            "tracker_config_sha256": annotation_sha256(config), "tracker_params": params,
            "tracker_sources_sha256": sources, "entry_sha256": annotation_sha256(script)}


def write_receipt(output: Path, annotation: Path, videos: dict, selected: list,
                  shard: tuple[int, int], records: int, *, identity: dict,
                  processed_frames: dict) -> None:
    index, count = shard
    expected_counts = {str(video): len(videos[video]) for video in selected}
    if processed_frames != expected_counts:
        raise ValueError("Incomplete processed input frame coverage")
    payload = {"format": "xsvid-track-shard-v2", "annotation_sha256": annotation_sha256(annotation),
               "prediction_sha256": annotation_sha256(output), "identity": identity,
               "shard_index": index, "shard_count": count, "records": records,
               "expected_video_ids": sorted(videos), "covered_video_ids": selected,
               "video_frame_counts": processed_frames,
               "complete_selected_videos": True}
    destination = receipt_path(output)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
