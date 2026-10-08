#!/usr/bin/env python3
"""CPU-only strict release checkpoint gate; does not install dependencies."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(ROOT))
from tools.xsvid.checkpoint_utils import load_release
from tools.xsvid.download_release import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vid", type=Path, required=True)
    parser.add_argument("--mot", type=Path, required=True)
    parser.add_argument("--sot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/xsvid/yoloft-l-temporal.yaml")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("release_manifest.json").read_text())
    from ultralytics import YOLOFT
    from ultralytics.nn.omni import VideoOmniModel
    from ultralytics.nn.sot_transt import YOLOFTSotTransT

    factories = {
        "vid": lambda: YOLOFT(str(args.config)).model,
        "mot": lambda: VideoOmniModel(str(args.config), ch=3, nc=7, verbose=False, embed_dim=256,
                                      emb_stride=4, use_pre_mstf=False, emb_src_hi=20, feat_mode="simple"),
        "sot": lambda: YOLOFTSotTransT(str(args.config), 3, 7, verbose=False,
                                     feat_mode="fuse918", embed=256, n_fusion=4),
    }
    for name, factory in factories.items():
        path = getattr(args, name)
        digest = sha256(path)
        filename = "vid.pt" if name == "vid" else f"unified_{name}.pt"
        if digest != manifest["weights"][filename]["file_sha256"]:
            raise ValueError(f"Unexpected {filename} bytes: {digest}")
        module = factory().cpu()
        initialized = 0
        if name != "vid":
            head = {key: value.clone() for key, value in module.state_dict().items()
                    if not key.startswith("model.")}
            initialized, _ = load_release(module, args.vid, detector_only=True)
            import torch
            if any(not torch.equal(module.state_dict()[key], value) for key, value in head.items()):
                raise RuntimeError("VID initialization changed a task-head tensor")
            del head
        loaded, _ = load_release(module, path, detector_only=name == "vid")
        print(json.dumps({"task": name, "sha256": digest, "loaded": loaded,
                          "device": "cpu", "strict": True, "vid_initialized": initialized,
                          "tensor_sha256": manifest["weights"][filename]["tensor_sha256"],
                          "selected": "released_state"}), flush=True)
        del module
        gc.collect()


if __name__ == "__main__":
    main()
