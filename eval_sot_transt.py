#!/usr/bin/env python3
"""Evaluate the YOLOFT TransT SOT head on XS-VID sequences.

The tracker encodes the first-frame template once, then tracks subsequent
search regions online. Predictions are stored as one xywh text file per
sequence. Use ``--shard i/n`` for deterministic multi-GPU evaluation.
"""
import os, sys, argparse, json, math, time
from pathlib import Path
import numpy as np, cv2, torch

YF = Path(__file__).resolve().parent
sys.path.insert(0, str(YF)); os.chdir(YF)
os.environ["YOLO_AUTOINSTALL"] = "false"
from ultralytics.nn.sot_transt import YOLOFTSotTransT
from tools.xsvid.checkpoint_utils import load_release

Z, X, FZ, FX = 128, 256, 2.0, 4.0


def read(fn):
    im = cv2.imread(fn); return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def crop(img, cx, cy, side, out):
    side = max(side, 16.0); x0, y0 = cx - side / 2, cy - side / 2
    M = np.array([[out / side, 0, -x0 * out / side], [0, out / side, -y0 * out / side]], np.float32)
    return cv2.warpAffine(img, M, (out, out), borderValue=(114, 114, 114)), x0, y0, side


def to_t(c):
    return torch.from_numpy(c).permute(2, 0, 1).float().div_(255.0)


def main():
    global Z, X
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--param", required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="Downloaded XS-VID v2 root")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/xsvid/yoloft-l-temporal.yaml"))
    ap.add_argument("--gpu", default="0"); ap.add_argument("--win", type=float, default=0.2)
    ap.add_argument("--smooth", type=float, default=0.4); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--feat-mode", default="fuse918", choices=["fuse918"])
    ap.add_argument("--z", type=int, default=Z); ap.add_argument("--x", type=int, default=X)   # must match training crop px
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    Z, X = args.z, args.x
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    si, sn = [int(v) for v in args.shard.split("/")]
    data_root = args.data_root.resolve()
    sot_root = data_root / "sot"
    config = args.config if args.config.is_absolute() else YF / args.config
    link_manifest = json.loads((sot_root / "image_links.json").read_text(encoding="utf-8"))
    source_by_destination = {
        link["destination"]: data_root / "images" / link["image"]
        for link in link_manifest["links"]
    }

    model = YOLOFTSotTransT(str(config), 3, 7, verbose=False, feat_mode=args.feat_mode, embed=256, n_fusion=4).cuda().eval()
    load_release(model, Path(args.ckpt))
    seqs = [s.strip() for s in open(sot_root / "test.txt") if s.strip()]
    seqs = [s for i, s in enumerate(seqs) if i % sn == si]
    if args.limit > 0:
        seqs = seqs[:args.limit]
    outdir = args.out_root / args.param; outdir.mkdir(parents=True, exist_ok=True)
    print(f"[tt-eval] shard {si}/{sn}: {len(seqs)} seqs", flush=True)

    total_frames, total_time = 0, 0.0
    for qi, seq in enumerate(seqs):
        started = time.perf_counter()
        sd = sot_root / "sequences" / seq
        frames = sorted(f for f in os.listdir(sd) if f.endswith(".jpg"))
        if frames:
            frame_paths = [sd / frame for frame in frames]
        else:
            prefix = f"sequences/{seq}/"
            frame_paths = [source_by_destination[key] for key in sorted(source_by_destination) if key.startswith(prefix)]
        if not frame_paths:
            raise FileNotFoundError(f"No sequence frames found for {seq}")
        g0 = open(sd / "groundtruth.txt").readline().strip().replace(",", " ").split()
        gx1, gy1, gx2, gy2 = [float(v) for v in g0[:4]]
        x1, y1, w0, h0 = gx1, gy1, gx2 - gx1, gy2 - gy1
        cx, cy, bw, bh = x1 + w0 / 2, y1 + h0 / 2, w0, h0
        cz, _, _, _ = crop(read(str(frame_paths[0])), cx, cy, FZ * math.sqrt(max(bw * bh, 4.0)), Z)
        tenc = model.encode_template(to_t(cz).unsqueeze(0).cuda())
        lines = [f"{x1:.4f},{y1:.4f},{w0:.4f},{h0:.4f}"]; han = None
        for frame_path in frame_paths[1:]:
            side = FX * math.sqrt(max(bw * bh, 4.0))
            cxc, x0, y0, s = crop(read(str(frame_path)), cx, cy, side, X)
            cls, box, (hs, ws) = model.track(tenc, to_t(cxc).unsqueeze(0).cuda())
            fg = torch.softmax(cls[0], -1)[:, 0].view(hs, ws)          # class0 = target
            box = box[0].view(hs, ws, 4)
            if han is None or han.shape != fg.shape:
                hy = torch.hann_window(hs, periodic=False, device=fg.device)
                hx = torch.hann_window(ws, periodic=False, device=fg.device)
                han = hy[:, None] * hx[None, :]
            score = fg * (1 - args.win) + han * args.win
            idx = int(torch.argmax(score)); py, px = idx // ws, idx % ws
            r = box[py, px]
            ncx = x0 + float(r[0]) * s; ncy = y0 + float(r[1]) * s
            nbw = float(r[2]) * s; nbh = float(r[3]) * s
            nbw = min(max(nbw, 0.6 * bw), 1.5 * bw); nbh = min(max(nbh, 0.6 * bh), 1.5 * bh)
            cx, cy = ncx, ncy
            bw = (1 - args.smooth) * nbw + args.smooth * bw; bh = (1 - args.smooth) * nbh + args.smooth * bh
            bw = max(bw, 2.0); bh = max(bh, 2.0)
            lines.append(f"{cx-bw/2:.4f},{cy-bh/2:.4f},{bw:.4f},{bh:.4f}")
        elapsed = time.perf_counter() - started
        total_frames += len(frame_paths)
        total_time += elapsed
        (outdir / f"{seq}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if (qi + 1) % 30 == 0:
            print(f"[tt-eval] shard{si} {qi+1}/{len(seqs)}", flush=True)
    fps = total_frames / total_time if total_time else 0.0
    print(f"[tt-eval] shard {si}/{sn} DONE {len(seqs)} seqs {total_frames} frames {total_time:.2f}s {fps:.2f} fps", flush=True)


if __name__ == "__main__":
    main()
