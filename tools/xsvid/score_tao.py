#!/usr/bin/env python3
"""Score XS-VID MOT predictions with the repository's bundled TAO evaluator."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--tag", default="YOLOFT-omni")
    parser.add_argument("--track-field", default="paper_track_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tao_eval import Tao, TaoEval

    annotation = args.annotation or args.data_root / "annotations" / "test.json"
    logging.getLogger().setLevel(logging.ERROR)
    ground_truth = Tao(str(annotation), track_field=args.track_field)
    evaluator = TaoEval(ground_truth, str(args.pred))
    evaluator.run()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluator.print_results()
    last_line = output.getvalue().strip().splitlines()[-1]
    values = [float(value) for value in last_line.replace(",", " ").split()]
    overrides = sum(args.track_field in row for row in ground_truth.dataset["annotations"])
    report = {
        "ground_truth": str(annotation),
        "predictions": str(args.pred),
        "track_field": args.track_field,
        "track_field_overrides": overrides,
        "track_field_fallbacks": len(ground_truth.dataset["annotations"]) - overrides,
        "metrics_percent": {
            "mAP": values[0],
            "AP_s": values[3],
            "AP_m": values[4],
            "AP_s_star": values[6],
            "AP_m_star": values[7],
        },
        "raw_last_line": last_line,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"TAO_RAW_LASTLINE: {last_line}")
    print(
        "TAORESULT "
        f'{args.tag} {{"mAP": {values[0]:.3f}, "AP_s": {values[3]:.3f}, '
        f'"AP_m": {values[4]:.3f}, "AP_s_star": {values[6]:.3f}, '
        f'"AP_m_star": {values[7]:.3f}}}'
    )
    print(f"TAO_SCORE_RECORD {args.output}", flush=True)


if __name__ == "__main__":
    main()
