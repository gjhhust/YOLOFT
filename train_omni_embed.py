#!/usr/bin/env python3
"""Fine-tune YOLOFT's MOT embedding head with a frozen Detection backbone.

The script uses ordered frame pairs from the canonical XS-VID training
annotations and carries temporal state between the two frames. It is a
standalone task-head training path and does not alter the Detection checkpoint.
"""
import os, sys, json, time, random, argparse, math
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from ultralytics.nn.omni import VideoOmniModel
from tools.xsvid.startup_checkpoint import save_startup_checkpoint
from tools.xsvid.checkpoint_utils import load_release, require_finite_update, training_metadata

def load_det_into(model, checkpoint: Path):
    return load_release(model, checkpoint, detector_only=True)


def read_img(image_root: Path, fn):
    im = cv2.imread(str(image_root / fn))                # BGR, 1024x1024
    if im is None:
        raise FileNotFoundError(image_root / fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(im).permute(2, 0, 1).float() / 255.0   # [3,H,W] 0-1


def sample_clip(vid_frames, intervals, min_box=1):
    """Pick a frame pair (t, t+step) within one video (step from intervals). Return two
    (file_name, boxes_xyxy[N,4], track_ids[N]) with >=1 shared track id; else None."""
    fids = sorted(int(f) for f in vid_frames.keys())
    if len(fids) < 2:
        return None
    for _ in range(8):
        step = random.choice(intervals)
        i0 = random.randint(0, len(fids) - 1)
        f0 = fids[i0]
        f1 = f0 + step
        if str(f1) not in vid_frames:
            f1 = f0 - step
            if str(f1) not in vid_frames:
                continue
        a, b = vid_frames[str(f0)], vid_frames[str(f1)]
        ra, rb = np.array(a["res"], np.float32), np.array(b["res"], np.float32)
        if ra.shape[0] == 0 or rb.shape[0] == 0:
            continue
        ta, tb = set(ra[:, 5].astype(int)), set(rb[:, 5].astype(int))
        if not (ta & tb):
            continue
        return (a["file_name"], ra[:, :4], ra[:, 5].astype(int)), (b["file_name"], rb[:, :4], rb[:, 5].astype(int))
    return None


def grid_sample_emb(E, boxes, H, W):
    """E:[1,D,h,w]; boxes:[N,4] xyxy in image px -> [N,D] L2-norm embeddings."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
    gx = (cx / (W - 1) * 2 - 1).clamp(-1, 1); gy = (cy / (H - 1) * 2 - 1).clamp(-1, 1)
    grid = torch.stack([gx, gy], -1).view(1, -1, 1, 2).to(E.device)
    e = F.grid_sample(E, grid, mode="bilinear", align_corners=True).view(E.size(1), -1).t()  # [N,D]
    return F.normalize(e, dim=1)


def multipos_loss(key, ref, key_ids, ref_ids, tau=0.07):
    """Multi-positive InfoNCE (QDTrack-style): key/ref [N,D] L2-norm; same track_id = positive."""
    if key.size(0) == 0 or ref.size(0) == 0:
        return key.sum() * 0.0
    sim = key @ ref.t() / tau                                  # [Nk,Nr]
    pos = (torch.as_tensor(key_ids)[:, None] == torch.as_tensor(ref_ids)[None, :]).to(sim.device)  # [Nk,Nr]
    valid = pos.any(1)
    if valid.sum() == 0:
        return sim.sum() * 0.0
    lse_all = torch.logsumexp(sim, dim=1)
    neg_inf = torch.full_like(sim, -1e4)
    lse_pos = torch.logsumexp(torch.where(pos, sim, neg_inf), dim=1)
    return (lse_all - lse_pos)[valid].mean()


def main():
    from tools.xsvid.training_recipe import parse_training_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="omniE")
    ap.add_argument("--data-root", type=Path, required=True, help="XS-VID v2 root")
    ap.add_argument("--det-ckpt", type=Path, required=True, help="YOLOFT detection checkpoint")
    ap.add_argument("--config", type=Path, default=ROOT / "config/xsvid/yoloft-l-temporal.yaml")
    ap.add_argument("--train-index", type=Path, default=None, help="Defaults to annotations/mot/train_omni.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--emb-stride", type=int, default=4, choices=[4, 8])
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--pre-mstf", type=int, default=0)
    ap.add_argument("--emb-src-hi", type=int, default=20, help="hi feat layer: 20=post-MSTF P3(default), 9=raw backbone P3(SOT lesson)")
    ap.add_argument("--feat-mode", default="simple", choices=["simple", "fuse918"], help="simple=single tap; fuse918=raw L9 + FPN-up L18 fusion")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bv", type=int, default=8, help="videos per step")
    ap.add_argument("--samples", type=int, default=8000, help="clips per epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, run only N steps")
    ap.add_argument("--seed", type=int, default=0)
    args = parse_training_args(ap, task="mot", default_recipe=ROOT / "config/recipes/unified_mot.yaml")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"

    train_index = args.train_index or args.data_root / "annotations/mot/train_omni.json"
    image_root = args.data_root / "images"
    if not train_index.is_file():
        raise FileNotFoundError(f"Missing MOT index: {train_index}. Run tools/xsvid/prepare_unified_train_data.py first.")
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    model = VideoOmniModel(str(args.config), ch=3, nc=7, verbose=True,
                           embed_dim=args.embed_dim, emb_stride=args.emb_stride,
                           use_pre_mstf=bool(args.pre_mstf), emb_src_hi=args.emb_src_hi,
                           feat_mode=args.feat_mode).to(dev)
    nk, nt = load_det_into(model, args.det_ckpt)
    # FREEZE detector; train only embedding branch (emb_*)
    emb_params = []
    for n, p in model.named_parameters():
        if n.startswith("emb_"):
            p.requires_grad_(True); emb_params.append(p)
        else:
            p.requires_grad_(False)
    model.eval()                                  # retain frozen normalization statistics
    for m in model.modules():
        pass
    print(f"[omni-train] det loaded {nk}/{nt}; trainable emb params={sum(p.numel() for p in emb_params)/1e6:.2f}M "
          f"| intervals={args.intervals} stride={args.emb_stride} dim={args.embed_dim} pre_mstf={args.pre_mstf}", flush=True)

    data = json.loads(train_index.read_text(encoding="utf-8"))
    vids = [v for v in data.values() if len(v) >= 2]
    print(f"[omni-train] {len(vids)} videos", flush=True)

    opt = torch.optim.AdamW(emb_params, lr=args.lr, weight_decay=1e-4)
    nsteps = (args.samples // args.bv) if not args.smoke else args.smoke
    total = nsteps * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"{args.tag}.pt"

    t_carry = t_zero = 0
    step = 0; t0 = time.time()
    for ep in range(args.epochs):
        for it in range(nsteps):
            # sample bv clips
            clips = []
            while len(clips) < args.bv:
                c = sample_clip(random.choice(vids), args.intervals)
                if c is not None:
                    clips.append(c)
            img0 = torch.stack([read_img(image_root, c[0][0]) for c in clips]).to(dev)   # [B,3,H,W] frame t
            img1 = torch.stack([read_img(image_root, c[1][0]) for c in clips]).to(dev)   # frame t+step
            H, W = img0.shape[-2:]
            # frame0 forward (grad on emb branch; detector frozen) -> embedding E0g + temporal state net0
            _x0, net0, _pm0, E0g = model.predict(img0, None, None, None, mask=True)
            net0d = [n.detach() for n in net0]                 # carry state (no backprop into frame0 detector)
            # frame1 forward WITH carried temporal state -> embedding E1
            _x1, net1, _pm1, E1 = model.predict(img1, *net0d, mask=True)
            t_carry += sum(int(n is not None) for n in net0)

            loss = 0.0; npairs = 0
            for bi, c in enumerate(clips):
                (_, b0, id0), (_, b1, id1) = c
                e0 = grid_sample_emb(E0g[bi:bi+1], torch.from_numpy(b0).to(dev), H, W)
                e1 = grid_sample_emb(E1[bi:bi+1], torch.from_numpy(b1).to(dev), H, W)
                l = multipos_loss(e0, e1, id0, id1, args.tau) + multipos_loss(e1, e0, id1, id0, args.tau)
                loss = loss + l; npairs += 1
            loss = loss / max(npairs, 1)
            require_finite_update(loss, [])
            opt.zero_grad(); loss.backward()
            require_finite_update(loss, emb_params)
            opt.step(); sched.step()
            step += 1
            if step % 10 == 0 or step <= 3:
                print(f"[omni-train] ep{ep} step{step}/{total} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} t_carry={t_carry} {step/(time.time()-t0):.2f}it/s", flush=True)
            if args.smoke and step >= args.smoke:
                save_startup_checkpoint(model, opt, ckpt_path, args=vars(args), epoch=ep,
                                        step=step, loss=loss.item(), task="mot",
                                        details={"temporal_state_carries": t_carry})
                print(f"[smoke] {step} steps OK, loss={loss.item():.4f}, t_carry={t_carry} (temporal active)", flush=True)
                return
        torch.save({"model": model.state_dict(), "args": training_metadata(vars(args)), "epoch": ep}, str(ckpt_path))
        print(f"[omni-train] saved {ckpt_path} (epoch {ep})", flush=True)
    print(f"DONE {args.tag}: {step} steps, {time.time()-t0:.0f}s, ckpt={ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
