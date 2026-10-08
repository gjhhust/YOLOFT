#!/usr/bin/env python3
"""Fine-tune the YOLOFT TransT SOT head on canonical XS-VID trajectories.

By default the Detection backbone is frozen and only the task head is updated.
The TransT head can be initialized from the supplied pretrained checkpoint.
"""
import os, sys, json, time, random, argparse, math
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from ultralytics.nn.sot_transt import YOLOFTSotTransT
from tools.xsvid.sot_utils import crop, read, to_tensor, giou_loss
from tools.xsvid.startup_checkpoint import save_startup_checkpoint
from tools.xsvid.checkpoint_utils import load_release, require_finite_update, training_metadata

Z, X, FZ, FX = 128, 256, 2.0, 4.0


def load_det(model, checkpoint: Path):
    return load_release(model, checkpoint, detector_only=True)[0]


def main():
    global Z, X
    from tools.xsvid.training_recipe import parse_training_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="unified_sot"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--data-root", type=Path, required=True, help="XS-VID v2 root")
    ap.add_argument("--det-ckpt", type=Path, required=True, help="YOLOFT detection checkpoint")
    ap.add_argument("--transt-pretrain", type=Path, default=None,
                    help="Portable tensor-only TransT checkpoint; defaults to transt-state-dict.pth beside --det-ckpt")
    ap.add_argument("--config", type=Path, default=ROOT / "config/xsvid/yoloft-l-temporal.yaml")
    ap.add_argument("--train-root", type=Path, default=None, help="Defaults to sot/train materialized from train.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bv", type=int, default=24); ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--gpu", default="5")
    ap.add_argument("--max-gap", type=int, default=50); ap.add_argument("--jitter", type=float, default=0.4)
    ap.add_argument("--scale-jit", type=float, default=0.25); ap.add_argument("--eos", type=float, default=0.0625)
    ap.add_argument("--no-pretrain", action="store_true"); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--unfreeze-bb", type=int, default=0, help="also fine-tune YOLOFT backbone (SOT-specialized, not unified)")
    ap.add_argument("--init-ckpt", default="", help="Strict release SOT weight initialization; not optimizer/epoch resume")
    ap.add_argument("--feat-mode", default="fuse918", choices=["fuse918"])
    ap.add_argument("--lr-bb", type=float, default=0.0, help="separate backbone LR (0=same as --lr)")
    ap.add_argument("--z", type=int, default=Z); ap.add_argument("--x", type=int, default=X)   # crop px (higher=more res on small targets)
    args = parse_training_args(ap, task="sot", default_recipe=ROOT / "config/recipes/unified_sot.yaml")
    Z, X = args.z, args.x
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(0); np.random.seed(0); torch.manual_seed(0); dev = "cuda"

    train_root = args.train_root or args.data_root / "sot/train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing SOT training layout: {train_root}. Run tools/xsvid/prepare_unified_train_data.py first.")
    if not args.no_pretrain and args.transt_pretrain is None:
        args.transt_pretrain = args.det_ckpt.parent / "transt-state-dict.pth"
    if not args.no_pretrain and not args.transt_pretrain.is_file():
        raise FileNotFoundError(f"Missing portable TransT pretrain: {args.transt_pretrain}. "
                                "See docs/PREREQUISITES.md for official download and local conversion.")
    model = YOLOFTSotTransT(str(args.config), 3, 7, verbose=False, feat_mode=args.feat_mode, embed=256, n_fusion=4).to(dev)
    nk = load_det(model, args.det_ckpt)
    npre = (0, 0) if args.no_pretrain else model.load_transt(args.transt_pretrain)
    if args.init_ckpt:
        load_release(model, Path(args.init_ckpt))
        print(f"[tt-train] strict weight initialization from {os.path.basename(args.init_ckpt)}", flush=True)
    for n, p in model.named_parameters():
        head = any(n.startswith(x) for x in ["input_proj", "featurefusion_network", "class_embed", "bbox_embed"])
        p.requires_grad_(head or bool(args.unfreeze_bb))  # --unfreeze-bb: also fine-tune the YOLOFT backbone
    if args.unfreeze_bb:
        model.train()                                    # BN updates when backbone is trained
    else:
        model.eval()
    tp = [p for p in model.parameters() if p.requires_grad]
    print(f"[tt-train] det {nk}; transt-pretrain {npre[0]}/{npre[1]}; unfreeze_bb={args.unfreeze_bb}; "
          f"trainable {sum(p.numel() for p in tp)/1e6:.2f}M", flush=True)

    seqs = []
    for s in sorted(os.listdir(train_root)):
        sd = train_root / s; gt = sd / "groundtruth.txt"
        if not os.path.isfile(gt):
            continue
        jpgs = sorted(f for f in os.listdir(sd) if f.endswith(".jpg"))
        boxes = np.loadtxt(gt, delimiter=",").reshape(-1, 4)
        if len(jpgs) >= 2 and len(jpgs) == len(boxes):
            seqs.append((sd, jpgs, boxes))
    print(f"[tt-train] {len(seqs)} sequences", flush=True)

    lr_bb = args.lr_bb if args.lr_bb > 0 else args.lr
    bb_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("model.")]   # YOLOFT backbone
    hd_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("model.")]  # SOT head
    opt = torch.optim.AdamW([{"params": hd_p, "lr": args.lr}, {"params": bb_p, "lr": lr_bb}], weight_decay=1e-4)
    print(f"[tt-train] LR head={args.lr} backbone={lr_bb} (bb params {sum(p.numel() for p in bb_p)/1e6:.1f}M)", flush=True)
    nsteps = (args.samples // args.bv) if not args.smoke else args.smoke
    total = nsteps * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, lr_bb], total_steps=total, pct_start=0.05)
    args.out_dir.mkdir(parents=True, exist_ok=True); ckpt = args.out_dir / f"{args.tag}.pt"
    was_tr = model.training; model.eval()                  # probe search-grid size (robust to any feat_mode/stride)
    with torch.no_grad():
        _, _, (_gh, _gw) = model.forward_sot(torch.zeros(1, 3, Z, Z, device=dev), torch.zeros(1, 3, X, X, device=dev))
    model.train(was_tr)
    GS = _gw; cell = (torch.arange(GS).float() + 0.5) / GS
    gy, gx = torch.meshgrid(cell, cell, indexing='ij'); gx = gx.reshape(-1).to(dev); gy = gy.reshape(-1).to(dev)
    print(f"[tt-train] search grid {_gh}x{_gw} (GS={GS})", flush=True)
    clsw = torch.tensor([1.0, args.eos], device=dev)       # [fg, bg]; bg down-weighted

    step = 0; t0 = time.time()
    for ep in range(args.epochs):
        for it in range(nsteps):
            T, S, G = [], [], []
            while len(T) < args.bv:
                sd, jpgs, boxes = random.choice(seqs)
                i = random.randint(0, len(jpgs) - 1)
                j = min(max(i + random.randint(-args.max_gap, args.max_gap), 0), len(jpgs) - 1)
                imz = read(str(sd / jpgs[i])); imx = read(str(sd / jpgs[j]))
                if imz is None or imx is None:
                    continue
                bz, bx = boxes[i], boxes[j]
                if min(bz[2], bz[3]) < 1 or min(bx[2], bx[3]) < 1:
                    continue
                cz, _ = crop(imz, bz, FZ, Z, jitter=0.05)
                cxx, gbox = crop(imx, bx, FX, X, jitter=args.jitter, scale_jit=args.scale_jit)
                T.append(to_tensor(cz)); S.append(to_tensor(cxx)); G.append(gbox)
            tmpl = torch.stack(T).to(dev); srch = torch.stack(S).to(dev); gt = torch.from_numpy(np.stack(G)).to(dev)
            cls, box, _ = model.forward_sot(tmpl, srch)        # cls[B,N,2] box[B,N,4]
            B, N = cls.shape[:2]
            dxx = (gx[None] - gt[:, 0:1]).abs(); dyy = (gy[None] - gt[:, 1:2]).abs()
            fg = (dxx < gt[:, 2:3] / 2) & (dyy < gt[:, 3:4] / 2) & (gt[:, 2:3] > 0)   # [B,N]
            tgt = torch.where(fg, 0, 1)                          # fg=class0, bg=class1
            cls_loss = F.cross_entropy(cls.reshape(-1, 2), tgt.reshape(-1), weight=clsw)
            if fg.any():
                pr = box[fg]; gtr = gt[:, None, :].expand(B, N, 4)[fg]
                reg_loss = giou_loss(pr, gtr)
            else:
                reg_loss = box.sum() * 0.0
            loss = cls_loss + 2.0 * reg_loss
            require_finite_update(loss, [])
            opt.zero_grad(); loss.backward()
            require_finite_update(loss, tp)
            opt.step(); sched.step(); step += 1
            if step % 20 == 0 or step <= 3:
                print(f"[tt-train] ep{ep} step{step}/{total} loss={loss.item():.4f} cls={cls_loss.item():.4f} "
                      f"reg={reg_loss.item():.4f} fg/img={fg.float().sum(1).mean().item():.1f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {step/(time.time()-t0):.2f}it/s", flush=True)
            if args.smoke and step >= args.smoke:
                save_startup_checkpoint(model, opt, ckpt, args=vars(args), epoch=ep,
                                        step=step, loss=loss.item(), task="sot")
                print(f"[smoke] {step} OK loss={loss.item():.4f}", flush=True); return
        torch.save({"model": model.state_dict(), "args": training_metadata(vars(args))}, str(ckpt))
        print(f"[tt-train] saved {ckpt} (ep {ep})", flush=True)
    print(f"DONE {args.tag}: {step} steps {time.time()-t0:.0f}s -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
