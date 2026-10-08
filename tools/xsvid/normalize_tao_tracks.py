#!/usr/bin/env python3
"""Normalize TAO predictions to one category per globally unique track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    categories = {}
    changed = 0
    for record in records:
        track_id = int(record["track_id"])
        category_id = int(record["category_id"])
        if track_id not in categories:
            categories[track_id] = category_id
        elif category_id != categories[track_id]:
            record["category_id"] = categories[track_id]
            changed += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(records), encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"NORMALIZE_TAO_TRACKS_OK records={len(records)} "
        f"tracks={len(categories)} category_updates={changed} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
