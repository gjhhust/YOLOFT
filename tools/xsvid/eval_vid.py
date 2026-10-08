#!/usr/bin/env python3
"""Run YOLOFT video detection on the full XS-VID test split."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from checkpoint_utils import load_release
from prepare_legacy_layout import NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--half", action="store_true", help="Optional FP16; release defaults to FP32")
    args = parser.parse_args()
    for name in ("config", "weights", "data", "cfg", "project"):
        setattr(args, name, getattr(args, name).resolve())

    repo_root = Path(__file__).resolve().parents[2]
    os.environ["YOLO_AUTOINSTALL"] = "false"
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from ultralytics import YOLOFT
    from ultralytics.nn.autobackend import AutoBackend

    AutoBackend.warmup = lambda self, imgsz=(1, 3, 640, 640): None
    model = YOLOFT(str(args.config))
    load_release(model.model, args.weights, detector_only=True)
    model.model.names = dict(enumerate(NAMES))
    model.val(
        data=str(args.data),
        cfg=str(args.cfg),
        batch=1,
        device=[args.device],
        imgsz=1024,
        workers=4,
        half=args.half,
        split="test",
        save_json=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    candidates = glob.glob(str(args.project / args.name / "*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON prediction file under {args.project / args.name}")
    print(f"XSVID_VID_EVAL_OK {max(candidates, key=os.path.getmtime)}", flush=True)


if __name__ == "__main__":
    main()
