#!/usr/bin/env python3
"""Portable XS-VID paper MOTChallenge scoring; no inference or source writes."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import sys
import zipfile

PROFILES = {
    "unified-raw": {"score_threshold": None},
    "botsort-tau03": {"score_threshold": 0.3},
}
GT_MEMBER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\.txt", re.ASCII)
GT_ZIP_MAX_FILES = 1024
GT_ZIP_MAX_MEMBER_BYTES = 16 * 1024 * 1024
GT_ZIP_MAX_TOTAL_BYTES = 128 * 1024 * 1024
GT_ZIP_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def read_gt_zip(path):
    """Read bounded, unique basename-only TXT members in memory, never extract."""
    files = {}
    names = set()
    total = 0
    if path.stat().st_size > GT_ZIP_MAX_ARCHIVE_BYTES:
        raise ValueError("GT ZIP archive exceeds the compressed-byte limit")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members or len(members) > GT_ZIP_MAX_FILES:
            raise ValueError("GT ZIP must contain 1..1024 TXT files")
        # Validate all entries before decompressing any content.
        for member in members:
            name = member.filename
            if member.orig_filename != name or not GT_MEMBER_PATTERN.fullmatch(name):
                raise ValueError(f"unsafe GT ZIP member name: {name!r}")
            if name.casefold() in names:
                raise ValueError(f"duplicate GT ZIP basename: {name!r}")
            names.add(name.casefold())
            kind = stat.S_IFMT(member.external_attr >> 16)
            if member.is_dir() or member.external_attr & 0x10 or kind not in (0, stat.S_IFREG):
                raise ValueError(f"GT ZIP member must be a regular file: {name!r}")
            if member.flag_bits & 1:
                raise ValueError("encrypted GT ZIP members are not supported")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("unsupported GT ZIP compression method")
            if not 0 < member.file_size <= GT_ZIP_MAX_MEMBER_BYTES:
                raise ValueError("GT ZIP member size is empty or exceeds the limit")
            total += member.file_size
            if total > GT_ZIP_MAX_TOTAL_BYTES:
                raise ValueError("GT ZIP exceeds the total uncompressed-byte limit")
        for member in members:
            with archive.open(member) as stream:
                content = stream.read(GT_ZIP_MAX_MEMBER_BYTES + 1)
            if len(content) != member.file_size or len(content) > GT_ZIP_MAX_MEMBER_BYTES:
                raise ValueError("GT ZIP member size mismatch")
            files[member.filename] = content
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return files, {"path": str(path.resolve()), "sha256": digest.hexdigest(),
                   "format": "mot15-2D basename-only ZIP", "members": len(files),
                   "uncompressed_bytes": total}


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer")
    if not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{label} must be an integer")
    return int(value)


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain four XYWH values")
    values = tuple(finite(v, "bbox") for v in value)
    if values[2] < 0 or values[3] < 0:
        raise ValueError("bbox sizes must be nonnegative")
    return values


def video_names(data):
    result = {}
    for video in data["videos"]:
        vid = integer(video["id"], "video id")
        name = video["name"]
        if (not isinstance(name, str) or not name or name in (".", "..")
                or "/" in name or "\\" in name):
            raise ValueError("video name must be a safe file stem")
        if vid in result or name in result.values():
            raise ValueError("duplicate video id/name")
        result[vid] = name
    return result


def mapping(data):
    names = video_names(data)
    images = {}
    seen_frames = set()
    for image in data["images"]:
        iid = integer(image["id"], "image id")
        vid = integer(image["video_id"], "image video_id")
        frame = integer(image["frame_index"], "frame_index")
        if vid not in names or frame < 0:
            raise ValueError("invalid image video/frame mapping")
        if iid in images or (vid, frame) in seen_frames:
            raise ValueError("duplicate image id or video/frame mapping")
        seen_frames.add((vid, frame))
        images[iid] = (vid, frame)
    return names, images


def mot_text(rows, ground_truth=False):
    # Preserve the historical Python %.2f quantization BEFORE IoU matching.
    template = ("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n" if ground_truth
                else "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1\n")
    return "".join(
        template % (row[:6] if ground_truth else row)
        for row in sorted(rows, key=lambda row: (row[0], row[1]))
    )


def convert_predictions(records, map_data, protocol):
    if protocol not in PROFILES:
        raise ValueError(f"unknown protocol: {protocol}")
    if not isinstance(records, list):
        raise ValueError("prediction JSON must be a list")
    names, images = mapping(map_data)
    threshold = PROFILES[protocol]["score_threshold"]
    by_video = defaultdict(list)
    seen = set()
    stats = {"input_records": len(records), "score_filtered": 0,
             "ignore_filtered": 0, "missing_score_defaulted": 0,
             "retained_records": 0, "retained_original_score_below_0_3": 0,
             "retained_rounded_score_below_0_3": 0}
    for record in records:
        if threshold is not None and "score" not in record:
            raise ValueError("botsort-tau03 requires score on every record")
        score = finite(record.get("score", 1.0), "score")
        stats["missing_score_defaulted"] += "score" not in record
        if score < -1:
            raise ValueError("score below -1 would be silently dropped by motmetrics")
        # The historical tau sweep filters the ORIGINAL JSON, not rounded TXT.
        if threshold is not None and score < threshold:
            stats["score_filtered"] += 1
            continue
        if integer(record["category_id"], "category_id") == 4:
            stats["ignore_filtered"] += 1
            continue
        iid = integer(record["image_id"], "image_id")
        if iid not in images:
            raise ValueError(f"unknown image_id: {iid}")
        vid, frame = images[iid]
        if "video_id" in record and integer(record["video_id"], "video_id") != vid:
            raise ValueError("prediction video_id conflicts with image mapping")
        tid = integer(record["track_id"], "track_id")
        key = (vid, frame, tid)
        if key in seen:
            raise ValueError(f"duplicate prediction video/frame/track: {key}")
        seen.add(key)
        by_video[names[vid]].append((frame + 1, tid, *box(record["bbox"]), score))
        stats["retained_records"] += 1
        stats["retained_original_score_below_0_3"] += score < 0.3
        stats["retained_rounded_score_below_0_3"] += float("%.2f" % score) < 0.3
    return {name: mot_text(rows) for name, rows in by_video.items()}, stats


def convert_gt(data):
    """Export the legacy MOT annotation schema, NOT arbitrary COCO/TAO JSON."""
    names = video_names(data)
    by_video = defaultdict(list)
    for annotation in data["annotations"]:
        if integer(annotation["category_id"], "GT category_id") == 4:
            continue
        vid = integer(annotation["video_id"], "GT video_id")
        frame = integer(annotation["frame_id"], "GT frame_id")
        tid = integer(annotation["instance_id"], "GT instance_id")
        if vid not in names or frame < 0:
            raise ValueError("invalid GT video/frame")
        by_video[names[vid]].append((frame + 1, tid, *box(annotation["bbox"]), 1.0))
    # Empty-GT videos are not exported by the historical script.
    return {name: mot_text(rows, ground_truth=True) for name, rows in by_video.items()}


def content_set_sha256(files):
    """Audit convention: sorted basename, NUL, then raw 32-byte file SHA."""
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()),
                             "sha256": hashlib.sha256(raw).hexdigest()}


def score_texts(gt_files, predictions, solver="lap", progress=False):
    import motmetrics as mm

    if not gt_files:
        raise ValueError("no GT sequences")
    if solver not in mm.lap.available_solvers:
        raise ValueError(f"solver {solver!r} unavailable; installed: {mm.lap.available_solvers}")
    previous_solver = mm.lap.default_solver
    mm.lap.default_solver = solver
    try:
        accs, names, missing = [], [], []
        gt_duplicates = {}
        for index, (filename, raw) in enumerate(sorted(gt_files.items())):
            name = Path(filename).stem
            if not raw.strip():
                raise ValueError(f"empty GT file: {filename}")
            gt = mm.io.loadtxt(io.StringIO(raw.decode("utf-8")),
                               fmt="mot15-2D", min_confidence=1)
            if gt.empty:
                raise ValueError(f"no confidence>=1 GT in {filename}")
            duplicates = int(gt.index.duplicated().sum())
            if duplicates:
                gt_duplicates[name] = duplicates
            text = predictions.get(name, "")
            if text.strip():
                pred = mm.io.loadtxt(io.StringIO(text), fmt="mot15-2D")
            else:
                pred = gt.iloc[0:0]
                missing.append(name)
            accs.append(mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=0.5))
            names.append(name)
            if progress:
                print(f"scored {index + 1}/{len(gt_files)} {name}", file=sys.stderr, flush=True)
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, names=names,
                                  metrics=mm.metrics.motchallenge_metrics + [
                                      "num_objects", "num_predictions", "idtp", "idfp", "idfn"],
                                  generate_overall=True)
        overall = summary.loc["OVERALL"]
        counts = {label: int(overall[key]) for label, key in {
            "FP": "num_false_positives", "FN": "num_misses", "IDsw": "num_switches",
            "GT": "num_objects", "Pred": "num_predictions",
            "IDTP": "idtp", "IDFP": "idfp", "IDFN": "idfn",
        }.items()}
        return {
            "overall": {"MOTA": float(overall["mota"]), "IDF1": float(overall["idf1"]),
                        "MOTA_percent": float(overall["mota"] * 100),
                        "IDF1_percent": float(overall["idf1"] * 100), **counts},
            "videos": len(names), "pred_missing": len(missing), "missing_sequences": missing,
            "gt_duplicate_frame_id_rows": gt_duplicates,
            "unscored_prediction_sequences": sorted(set(predictions) - set(names)),
            "per_video": {name: {"MOTA": float(summary.loc[name, "mota"]),
                                 "IDF1": float(summary.loc[name, "idf1"])} for name in names},
        }
    finally:
        mm.lap.default_solver = previous_solver


def evaluate(gt_files, records, map_data, protocol, solver="lap", progress=False):
    import importlib.metadata
    predictions, conversion = convert_predictions(records, map_data, protocol)
    result = score_texts(gt_files, predictions, solver, progress)
    result.update({
        "schema_version": 1,
        "protocol": {"profile": protocol, **PROFILES[protocol],
                     "filter_stage": "original JSON before %.2f MOT15 conversion",
                     "ignore_category_id": 4, "class_agnostic": True, "iou_threshold": 0.5,
                     "frame_numbering": "image.frame_index + 1",
                     "prediction_identity_field": "track_id", "gt_identity_field": "instance_id",
                     "gt_min_confidence": 1, "prediction_min_confidence": -1,
                     "bbox_score_decimal_places": 2, "aggregate": "motmetrics OVERALL",
                     "idf1_definition": "2 * IDTP / (GT + Pred), as in motmetrics 1.4.0"},
        "conversion": conversion,
        "gt_content_set_sha256": content_set_sha256(gt_files),
        "converted_prediction_content_set_sha256": content_set_sha256(
            {name + ".txt": text.encode() for name, text in predictions.items()}),
        "environment": {"python": sys.version.split()[0], "solver": solver, **{
            package: importlib.metadata.version(package)
            for package in ("motmetrics", "numpy", "pandas", "scipy", solver)}},
        "warnings": ["The two paper rows use different score-filter protocols; not a common-tau comparison.",
                     "Only GT-file sequences are scored; prediction-only videos are recorded, not added.",
                     "GT duplicate frame/instance rows are preserved, not repaired, to match historical scoring.",
                     "Dropping category 4 does not suppress spatial overlap with ignore regions."],
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    gt = parser.add_mutually_exclusive_group(required=True)
    gt.add_argument("--gt-dir", type=Path, help="Historical per-video MOT15 GT TXT directory")
    gt.add_argument("--gt-zip", type=Path, help="Model package protocols/paper_motchallenge_gt.zip; read without extraction")
    gt.add_argument("--gt-json", type=Path,
                    help="Legacy MOT JSON with annotation video_id/frame_id/instance_id")
    parser.add_argument("--mapping-json", required=True, type=Path,
                        help="TAO-style images.id/video_id/frame_index and videos.id/name mapping")
    parser.add_argument("--pred-json", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=PROFILES)
    parser.add_argument("--solver", choices=("lap", "scipy"), default="lap")
    parser.add_argument("--output", type=Path, help="Optional JSON report; otherwise stdout only")
    parser.add_argument("--progress", action="store_true", help="Per-video progress on stderr")
    args = parser.parse_args(argv)
    source_paths = [args.mapping_json, args.pred_json]
    if args.gt_json:
        source_paths.append(args.gt_json)
    elif args.gt_zip:
        source_paths.append(args.gt_zip)
    else:
        source_paths.extend(args.gt_dir.glob("*.txt"))
    if args.output and args.output.resolve() in {p.resolve() for p in source_paths}:
        parser.error("output must not overwrite an input artifact")
    try:
        records, pred_source = read_json(args.pred_json)
        map_data, map_source = read_json(args.mapping_json)
        if args.gt_zip:
            gt_files, gt_source = read_gt_zip(args.gt_zip)
        elif args.gt_json:
            gt_data, gt_source = read_json(args.gt_json)
            texts = convert_gt(gt_data)
            gt_files = {name + ".txt": text.encode() for name, text in texts.items()}
        else:
            gt_files = {p.name: p.read_bytes() for p in sorted(args.gt_dir.glob("*.txt"))}
            gt_source = {"path": str(args.gt_dir.resolve()), "format": "mot15-2D directory"}
        result = evaluate(gt_files, records, map_data, args.protocol, args.solver, args.progress)
        result["sources"] = {"predictions": pred_source, "mapping": map_source, "gt": gt_source}
        result["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        parser.exit(2, f"score_mot: {exc}\n")


if __name__ == "__main__":
    main()
