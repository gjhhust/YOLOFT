#!/usr/bin/env python3
"""Merge deterministic video shards produced by dump_omni_emb.py."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    videos = {}
    sources = []
    for path in args.inputs:
        with path.open("rb") as handle:
            shard = pickle.load(handle)
        overlap = set(videos).intersection(shard["videos"])
        if overlap:
            raise RuntimeError(f"Duplicate video IDs in {path}: {sorted(overlap)[:5]}")
        videos.update(shard["videos"])
        sources.append(str(path))
    merged = {"sources": sources, "videos": dict(sorted(videos.items()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(merged, handle, protocol=4)
    print(f"MERGE_OMNI_DUMPS_OK videos={len(videos)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
