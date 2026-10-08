#!/usr/bin/env python3
"""Train YOLOFT-L from the released VID initialization on the full training split."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.xsvid.training_recipe import parse_training_args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/xsvid/yoloft-l-temporal.yaml"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="yoloft_l")
    parser.add_argument("--fraction", type=float, default=1.0, help="1=full train; smaller values are startup checks")
    parser.add_argument("--dry-run", action="store_true")
    args = parse_training_args(parser, task="vid", default_recipe=root / "config/recipes/vid.yaml")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 1 or not 0 < args.fraction <= 1:
        parser.error("epochs, batch and imgsz must be positive; fraction must be in (0,1]")
    for name in ("data", "weights", "project", "config"):
        setattr(args, name, getattr(args, name).resolve())
    if args.dry_run:
        import json
        print(json.dumps({"epochs": args.epochs, "batch": args.batch, "train_split": "train",
                          "validation_split": "test", "fraction": args.fraction, "save": True,
                          "finite_checks": True, "gpu_executed": False}))
        return
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.chdir(root)
    from ultralytics import YOLOFT
    from ultralytics.models.yoloft.detect.train import DetectionTrainer
    from tools.xsvid.checkpoint_utils import load_release, require_finite_update, save_vid_export

    class ReleaseTrainer(DetectionTrainer):
        def get_dataset(self):
            training, _ = super().get_dataset()
            if not self.data.get("test"):
                raise ValueError("Full test split required for validation")
            self.data["val"] = self.data["test"]
            if training == self.data["test"]:
                raise ValueError("Training and test split must be distinct")
            self.data["train_video_length"] = [2]
            self.data["train_video_interval"] = 1
            return training, self.data["test"]

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
            load_release(model, args.weights, detector_only=True)
            return model

        def optimizer_step(self):
            require_finite_update(self.loss, self.model.parameters())
            super().optimizer_step()

    model = YOLOFT(str(args.config))
    model.train(trainer=ReleaseTrainer, data=str(args.data),
        cfg=str(root / "config/train/default.yaml"), epochs=args.epochs, batch=args.batch,
        device=args.device, project=str(args.project), name=args.name, resume=False,
        pretrained=False, fraction=args.fraction, amp=False, imgsz=args.imgsz, train_slit=[0],
        split="test", save_json=True, save=True, plots=False)
    save_vid_export(model.trainer.ema.ema, model.trainer.save_dir / "vid.pt")


if __name__ == "__main__":
    main()
