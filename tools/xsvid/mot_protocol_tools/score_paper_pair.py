#!/usr/bin/env python3
"""Score two fresh XS-VID prediction artifacts into one CPU-only report."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import score_mot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-zip", required=True, type=Path)
    parser.add_argument("--mapping-json", required=True, type=Path)
    parser.add_argument("--unified-json", required=True, type=Path)
    parser.add_argument("--botsort-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solver", choices=("lap", "scipy"), default="lap")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    inputs = (args.gt_zip, args.mapping_json, args.unified_json, args.botsort_json)
    if args.output.resolve() in {p.resolve() for p in inputs}:
        parser.error("output must not overwrite an input artifact")
    if args.unified_json.resolve() == args.botsort_json.resolve():
        parser.error("paper rows require separate prediction artifacts")
    rows = {}
    # Reuse the scorer CLI so single-row and pair reports have identical semantics.
    for profile, prediction in (("unified-raw", args.unified_json),
                                ("botsort-tau03", args.botsort_json)):
        command = ["--gt-zip", str(args.gt_zip), "--mapping-json", str(args.mapping_json),
                   "--pred-json", str(prediction), "--protocol", profile,
                   "--solver", args.solver]
        if args.progress:
            command.append("--progress")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            score_mot.main(command)
        rows[profile] = json.loads(captured.getvalue())
    result = {"schema_version": 1, "evaluation_kind": "saved_prediction_scoring",
              "historical_target_asserted": False,
              "pair_scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": rows}
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
