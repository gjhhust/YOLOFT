#!/usr/bin/env python3
"""Prepare model-package protocol artifacts without changing canonical dataset GT."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile

import score_mot

EXPECTED_GT = "4067c583bbba1cb1c32cfdd2d56decc4460ae0da897c1d82b15c626e03c45d57"


def minimal_mapping(data):
    score_mot.mapping(data)
    return {"videos": [{key: video[key] for key in ("id", "name")}
                       for video in sorted(data["videos"], key=lambda v: v["id"])],
            "images": [{key: image[key] for key in ("id", "video_id", "frame_index")}
                       for image in sorted(data["images"], key=lambda i: i["id"])]}


def gt_statistics(files):
    rows = duplicates = 0
    per_file = {}
    for name, raw in sorted(files.items()):
        seen = set()
        duplicate_count = row_count = 0
        for row in csv.reader(io.StringIO(raw.decode("utf-8"))):
            if not row:
                continue
            key = (int(row[0]), int(row[1]))
            duplicate_count += key in seen
            seen.add(key)
            row_count += 1
        per_file[name] = {"rows": row_count, "duplicate_frame_id_rows": duplicate_count,
                          "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        rows += row_count
        duplicates += duplicate_count
    return {"files": len(files), "rows": rows, "duplicate_frame_id_rows": duplicates,
            "content_set_sha256": score_mot.content_set_sha256(files), "members": per_file}


def build(gt_dir, legacy_mapping_json, canonical_test_json, output_dir):
    gt_files = {p.name: p.read_bytes() for p in sorted(gt_dir.glob("*.txt"))}
    stats = gt_statistics(gt_files)
    if (stats["content_set_sha256"] != EXPECTED_GT or stats["files"] != 64
            or stats["rows"] != 337009 or stats["duplicate_frame_id_rows"] != 423):
        raise ValueError("GT is not the audited 64-file, 337009-row, 423-duplicate paper artifact")
    legacy, legacy_source = score_mot.read_json(legacy_mapping_json)
    canonical, canonical_source = score_mot.read_json(canonical_test_json)
    minimal = minimal_mapping(legacy)
    canonical_minimal = minimal_mapping(canonical)
    comparison = {"videos": len(minimal["videos"]), "images": len(minimal["images"]),
                  "video_fields_equal": minimal["videos"] == canonical_minimal["videos"],
                  "image_fields_equal": minimal["images"] == canonical_minimal["images"],
                  "canonical_test_usable_as_mapping": minimal == canonical_minimal,
                  "video_fields": ["id", "name"],
                  "image_fields": ["id", "video_id", "frame_index"]}
    mapping_bytes = (json.dumps(minimal, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    archive_path = output_dir / "paper_motchallenge_gt.zip"
    mapping_path = output_dir / "paper_motchallenge_mapping.json"
    manifest_path = output_dir / "paper_motchallenge_manifest.json"
    if any(p.exists() for p in (archive_path, mapping_path, manifest_path)):
        raise ValueError("protocol artifact destination already exists; refusing overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(gt_files.items()):
            if not score_mot.GT_MEMBER_PATTERN.fullmatch(name):
                raise ValueError(f"unsafe source GT basename: {name!r}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw, compresslevel=9)
    roundtrip, zip_source = score_mot.read_gt_zip(archive_path)
    if roundtrip != gt_files:
        raise ValueError("GT ZIP byte roundtrip mismatch")
    mapping_path.write_bytes(mapping_bytes)
    manifest = {
        "schema_version": 1, "intended_location": "HF model package protocols/; not canonical dataset GT",
        "canonical_gt_modified": False, "uploaded": False,
        "sources": {"gt_dir": str(gt_dir.resolve()), "legacy_mapping": legacy_source,
                    "canonical_test": canonical_source},
        "gt": stats,
        "artifacts": {"gt_zip": {"path": archive_path.name, **{key: zip_source[key] for key in
                                      ("sha256", "members", "uncompressed_bytes")}},
                      "mapping_json": {"path": mapping_path.name,
                                       "sha256": hashlib.sha256(mapping_bytes).hexdigest()}},
        "mapping_comparison": comparison,
        "protocol_profiles": score_mot.PROFILES,
        "preservation": "Original GT TXT bytes, including all 423 duplicate rows, preserved without normalization.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--legacy-mapping-json", required=True, type=Path)
    parser.add_argument("--canonical-test-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.gt_dir, args.legacy_mapping_json, args.canonical_test_json, args.output_dir)
    print(json.dumps({"gt": {k: manifest["gt"][k] for k in
                             ("files", "rows", "duplicate_frame_id_rows", "content_set_sha256")},
                      "artifacts": manifest["artifacts"], "mapping_comparison": manifest["mapping_comparison"]},
                     indent=2))


if __name__ == "__main__":
    main()
