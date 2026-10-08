#!/usr/bin/env python3
"""Merge disjoint complete-video tracking shards with deterministic global IDs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from track_shards import annotation_sha256, parse_shard, receipt_path, select_videos


def merge_shards(inputs: list[Path], annotation: Path, allow_partial: bool = False):
    dataset = json.loads(annotation.read_text(encoding="utf-8"))
    image_videos = {image["id"]: image["video_id"] for image in dataset["images"]}
    frame_counts = Counter(image["video_id"] for image in dataset["images"])
    expected = sorted(frame_counts)
    digest = annotation_sha256(annotation)
    covered = set()
    identities = set()
    counts = set()
    by_video = defaultdict(list)
    sources = []
    common_identity = None
    for path in inputs:
        metadata = json.loads(receipt_path(path).read_text(encoding="utf-8"))
        records = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("format") != "xsvid-track-shard-v2" or not metadata.get("complete_selected_videos"):
            raise ValueError(f"Missing complete-video receipt: {path}")
        if metadata.get("prediction_sha256") != annotation_sha256(path):
            raise ValueError(f"Prediction SHA-256 mismatch: {path}")
        identity = metadata.get("identity")
        required = {"pipeline", "inputs_sha256", "tracker_config_sha256", "tracker_params",
                    "tracker_sources_sha256", "entry_sha256"}
        if not isinstance(identity, dict) or not required.issubset(identity) or not all(identity[key] for key in required):
            raise ValueError(f"Missing pipeline/input/config identity: {path}")
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError(f"Mixed pipeline/input/config identity: {path}")
        if metadata["annotation_sha256"] != digest or metadata["expected_video_ids"] != expected:
            raise ValueError(f"Annotation/expected-video mismatch: {path}")
        shard = parse_shard(f"{metadata['shard_index']}/{metadata['shard_count']}")
        if shard in identities:
            raise ValueError(f"Duplicate shard identity: {shard}")
        identities.add(shard)
        counts.add(shard[1])
        videos = metadata["covered_video_ids"]
        assigned = set(select_videos(expected, shard))
        if len(videos) != len(set(videos)) or not set(videos).issubset(assigned):
            raise ValueError(f"Invalid video partition: {path}")
        overlap = covered.intersection(videos)
        if overlap:
            raise ValueError(f"Overlapping video coverage: {sorted(overlap)}")
        if not isinstance(records, list) or metadata["records"] != len(records):
            raise ValueError(f"Record count mismatch: {path}")
        for video in videos:
            if metadata["video_frame_counts"].get(str(video)) != frame_counts[video]:
                raise ValueError(f"Incomplete frame coverage for video {video}: {path}")
        for record in records:
            video = record["video_id"]
            if video not in videos or image_videos.get(record["image_id"]) != video:
                raise ValueError(f"Record outside declared video/image coverage: {path}")
            if len(record["bbox"]) != 4 or not all(math.isfinite(value) for value in [*record["bbox"], record["score"]]):
                raise ValueError(f"Invalid prediction values: {path}")
            by_video[video].append(record)
        covered.update(videos)
        sources.append({"path": str(path), "shard_index": shard[0], "shard_count": shard[1],
                        "covered_video_ids": videos, "records": len(records)})
    if len(counts) != 1:
        raise ValueError("Tracking shards must use one common shard count")
    missing = sorted(set(expected) - covered)
    if missing and not allow_partial:
        raise ValueError(f"Missing video coverage: {missing}")
    mapping = {}
    output = []
    changed = 0
    # Restore single-process video order, keeping all within-video rows untouched.
    for video in sorted(by_video):
        for record in by_video[video]:
            key = (video, record["track_id"])
            if key not in mapping:
                mapping[key] = (len(mapping), record["category_id"])
            identifier, category = mapping[key]
            changed += int(category != record["category_id"])
            item = record.copy()
            item.update(track_id=identifier, category_id=category)
            output.append(item)
    report = {"annotation_sha256": digest, "identity": common_identity,
              "sources": sources, "disjoint_video_coverage": True,
              "expected_video_ids": expected, "covered_video_ids": sorted(covered),
              "missing_video_ids": missing, "complete_coverage": not missing,
              "records": len(output), "tracks": len(mapping), "category_updates": changed}
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Default OUTPUT.coverage.json")
    parser.add_argument("--allow-partial", action="store_true", help="Explicit subset audit; never a full benchmark result")
    args = parser.parse_args()
    records, report = merge_shards(args.inputs, args.annotation, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(records), encoding="utf-8")
    temporary.replace(args.output)
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".coverage.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"MERGE_TRACK_SHARDS_OK videos={len(report['covered_video_ids'])} "
          f"records={len(records)} complete={report['complete_coverage']} output={args.output}")


if __name__ == "__main__":
    main()
