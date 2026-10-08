#!/usr/bin/env python3
"""Extract YOLOFT MOT embeddings for a set of Detection predictions.

The model carries temporal state within each video and resets it at video
boundaries. The generated dump is consumed by the bundled BoT-SORT stage.
"""
import os, sys, json, time, argparse, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn.functional as F

YF = Path(__file__).resolve().parent
sys.path.insert(0, str(YF)); os.chdir(YF)
os.environ["YOLO_AUTOINSTALL"] = "false"
from ultralytics.nn.omni import VideoOmniModel
from tools.xsvid.checkpoint_utils import load_release

def read_img(images_root, fn):
    im = cv2.imread(str(images_root / fn)); im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(im).permute(2, 0, 1).float().div_(255.0)


def load_dets(p, floor):
    d = json.load(open(p)); rows = defaultdict(list)
    for r in d:
        if r["score"] < floor:
            continue
        x, y, w, h = r["bbox"]
        rows[r["image_id"]].append([x, y, x + w, y + h, float(r["score"]), int(r["category_id"])])
    return {k: np.asarray(v, np.float32) for k, v in rows.items()}


def grid_emb(E, boxes, H, W, roi_grid=1):
    """Sample embeddings at box centers (roi_grid=1) or ROI-pool over a GxG inset grid
    inside each box (roi_grid=G>1, averaged) -> richer appearance than a single center point."""
    if roi_grid <= 1:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
        gx = (cx / (W - 1) * 2 - 1).clamp(-1, 1); gy = (cy / (H - 1) * 2 - 1).clamp(-1, 1)
        g = torch.stack([gx, gy], -1).view(1, -1, 1, 2).to(E.device)
        e = F.grid_sample(E, g, mode="bilinear", align_corners=True).view(E.size(1), -1).t()
        return F.normalize(e, dim=1)
    N = boxes.shape[0]; G = roi_grid
    t = torch.linspace(0.15, 0.85, G, device=boxes.device)              # inset fractions
    px = boxes[:, 0:1] + (boxes[:, 2:3] - boxes[:, 0:1]) * t.view(1, -1)   # [N,G] x
    py = boxes[:, 1:2] + (boxes[:, 3:4] - boxes[:, 1:2]) * t.view(1, -1)   # [N,G] y
    PX = px[:, :, None].expand(N, G, G); PY = py[:, None, :].expand(N, G, G)
    gx = (PX / (W - 1) * 2 - 1).clamp(-1, 1); gy = (PY / (H - 1) * 2 - 1).clamp(-1, 1)
    g = torch.stack([gx, gy], -1).reshape(1, N * G * G, 1, 2).to(E.device)
    e = F.grid_sample(E, g, mode="bilinear", align_corners=True)        # [1,D,N*G*G,1]
    e = e.view(E.size(1), N, G * G).permute(1, 2, 0).mean(1)            # [N,D] ROI-pooled
    return F.normalize(e, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="Downloaded XS-VID v2 root")
    ap.add_argument("--det-json", type=Path, required=True, help="Detection predictions in COCO JSON format")
    ap.add_argument("--annotation", type=Path, default=None, help="Defaults to annotations/test.json")
    ap.add_argument("--config", type=Path, default=Path("config/xsvid/yoloft-l-temporal.yaml"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--emb-stride", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--pre-mstf", type=int, default=0)
    ap.add_argument("--emb-src-hi", type=int, default=20, help="hi feat layer: 20=post-MSTF P3, 9=raw backbone P3")
    ap.add_argument("--feat-mode", default="simple", choices=["simple", "fuse918"])
    ap.add_argument("--det-floor", type=float, default=0.01)
    ap.add_argument("--roi-grid", type=int, default=1, help="1=center sample; G>1=ROI-pool GxG inside box")
    ap.add_argument("--videos", type=int, default=0, help="limit #videos (0=all)")
    ap.add_argument("--shard", default="0/1", help="Deterministic video shard as index/count")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    data_root = args.data_root.resolve()
    annotation = args.annotation or data_root / "annotations" / "test.json"
    config = args.config if args.config.is_absolute() else YF / args.config
    images_root = data_root / "images"

    model = VideoOmniModel(str(config), ch=3, nc=7, verbose=False, embed_dim=args.embed_dim,
                           emb_stride=args.emb_stride, use_pre_mstf=bool(args.pre_mstf),
                           emb_src_hi=args.emb_src_hi, feat_mode=args.feat_mode).cuda().eval()
    load_release(model, Path(args.ckpt))
    print(f"[dump] ckpt={os.path.basename(args.ckpt)} strict release load", flush=True)

    dets = load_dets(args.det_json, args.det_floor)
    tao = json.load(open(annotation))
    by_vid = defaultdict(list)
    for im in tao["images"]:
        by_vid[im["video_id"]].append(im)
    vids = sorted(by_vid.keys())
    shard_index, shard_count = [int(value) for value in args.shard.split("/")]
    if not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard index < shard count")
    vids = [video_id for index, video_id in enumerate(vids) if index % shard_count == shard_index]
    if args.videos:
        vids = vids[: args.videos]

    out = {"args": vars(args), "videos": {}}
    t0 = time.time(); totf = 0
    for vi, vid in enumerate(vids):
        frames = sorted(by_vid[vid], key=lambda im: im["frame_index"])
        net = [None, None, None]; vlist = []
        for im in frames:
            iid = im["id"]; fr = int(im["frame_index"]) + 1
            img = read_img(images_root, im["file_name"]).unsqueeze(0).cuda(); H, W = img.shape[-2:]
            with torch.no_grad():
                _x, net, _pm, E = model.predict(img, *net, mask=True)
                net = [n.detach() for n in net]
            db = dets.get(iid, np.zeros((0, 6), np.float32))
            if db.shape[0] == 0:
                vlist.append({"iid": int(iid), "fr": fr, "boxes": np.zeros((0, 4), np.float32),
                              "scores": np.zeros((0,), np.float32), "cls": np.zeros((0,), np.int16),
                              "emb": np.zeros((0, args.embed_dim), np.float16)})
                continue
            boxes = torch.from_numpy(db[:, :4])
            emb = grid_emb(E, boxes.cuda(), H, W, roi_grid=args.roi_grid).half().cpu().numpy()
            vlist.append({"iid": int(iid), "fr": fr, "boxes": db[:, :4].astype(np.float32),
                          "scores": db[:, 4].astype(np.float32), "cls": db[:, 5].astype(np.int16), "emb": emb})
        out["videos"][int(vid)] = vlist
        totf += len(frames)
        print(f"[dump] ({vi+1}/{len(vids)}) vid={vid} frames={len(frames)} {totf/(time.time()-t0):.1f}fps", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump(out, open(args.out, "wb"), protocol=4)
    sz = os.path.getsize(args.out) / 1e6
    print(f"DONE dump {len(vids)} vids {totf} frames {time.time()-t0:.0f}s -> {args.out} ({sz:.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
