# omni-A: unified detect+track YOLOFT (ISOLATED — does NOT modify the VID detection code paths).
# A VideoDetectionModel subclass that builds the UNCHANGED detection model from the detection yaml
# (Detect stays model[-1], so Detection weights transfer verbatim) and adds an isolated stride-4
# instance-embedding branch for unified tracking association.
#
# Embedding branch (config-driven):
#   E_map = Embedding( C2f( cat( DySample(P3_postMSTF stride8), P2_preMSTF stride4 ) ) )   # stride-4, embed_dim
# stride-4 is the explicit fix to Unicorn's stride-8 failure on XS-VID tiny objects.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import VideoDetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, DySample, MSTFv1, MSTFv1_yolo
from ultralytics.utils import LOGGER


class Embedding(nn.Module):
    """Dense per-cell instance-embedding head. Input one feature map -> [B, embed_dim, H, W]
    (L2-normalized at sample time, not here). Mirrors Detect's conv-head style."""

    def __init__(self, ch_in, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            Conv(ch_in, embed_dim, 3),
            Conv(embed_dim, embed_dim, 3),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return self.proj(x)


class VideoOmniModel(VideoDetectionModel):
    """YOLOFT detector (unchanged) + isolated stride-4 embedding branch for omni tracking.

    Args (config knobs for ablation):
      embed_dim     : embedding dim (256 default, QDTrack)
      emb_src_hi    : layer idx of post-MSTF P3 (stride-8, 256ch) -> 20 for the XS-VID yaml
      emb_src_lo    : layer idx of pre-MSTF stride-4 backbone feat (128ch) -> 7
      emb_stride    : 4 (default; tap+upsample to stride-4) or 8 (no upsample, ablation = Unicorn-like)
      use_pre_mstf  : fuse the pre-MSTF stride-4 feat (sharper identity) ; False -> post-MSTF only
    """

    def __init__(self, cfg, ch=3, nc=None, verbose=True,
                 embed_dim=256, emb_src_hi=20, emb_src_lo=7, emb_stride=4, use_pre_mstf=True,
                 feat_mode="simple", emb_src_fpn=18):
        self._omni_ready = False                      # guard: stride-init forward in super().__init__ skips E_map
        super().__init__(cfg, ch, nc, verbose)
        self.omni = True
        self.embed_dim = embed_dim
        self.feat_mode = feat_mode
        self.emb_src_hi = emb_src_hi
        self.emb_src_lo = emb_src_lo
        self.emb_src_fpn = emb_src_fpn
        self.emb_stride = emb_stride
        self.use_pre_mstf = use_pre_mstf and (emb_stride == 4) and (feat_mode == "simple")
        if feat_mode == "fuse918":
            # SOT fuse918 ported: raw backbone P3 (L9,256ch s8) + FPN-up P3 (L18,512ch s8),
            # each DySample->stride4, concat 768, 1x1 emb_input_proj = learned channel weighting.
            self.emb_src_hi = 9
            self.save = sorted(set(list(self.save) + [9, emb_src_fpn]))
            self.emb_up = DySample(256, 2)                       # L9 stride-8 -> stride-4
            self.emb_fpn_up = DySample(512, 2)                   # L18 stride-8 -> stride-4
            self.emb_input_proj = nn.Conv2d(256 + 512, 256, kernel_size=1)
            self.emb_fuse = C2f(256, 128, n=1)
            self.emb_head = Embedding(128, embed_dim)
        else:
            # keep tapped layers in save list so y[hi]/y[lo] are available after the forward loop
            taps = [self.emb_src_hi] + ([emb_src_lo] if self.use_pre_mstf else [])
            self.save = sorted(set(list(self.save) + taps))
            ch_hi, ch_lo = 256, 128            # post-MSTF P3 (or raw L9) = 256, pre-MSTF stride-4 = 128
            if emb_stride == 4:
                self.emb_up = DySample(ch_hi, 2)                 # stride-8 -> stride-4 (channel-preserving)
                cin = ch_hi + (ch_lo if self.use_pre_mstf else 0)
                self.emb_fuse = C2f(cin, 128, n=1)
                self.emb_head = Embedding(128, embed_dim)
            else:                                                # stride-8 ablation (Unicorn-like)
                self.emb_up = None
                self.emb_fuse = C2f(ch_hi, 128, n=1)
                self.emb_head = Embedding(128, embed_dim)
        self._omni_ready = True
        if verbose:
            LOGGER.info(f"[omni] embedding branch: mode={feat_mode}, stride={emb_stride}, dim={embed_dim}, "
                        f"pre_mstf={self.use_pre_mstf}, src_hi={self.emb_src_hi}, src_fpn={emb_src_fpn}")

    def _compute_embedding(self, y):
        """Build the stride-4 (or 8) embedding map from saved features y[hi]/y[lo]."""
        if self.feat_mode == "fuse918":                         # raw P3(L9) + FPN-up P3(L18) fusion
            hi = self.emb_up(y[self.emb_src_hi])                 # L9 -> [B,256,H/4,W/4]
            fp = self.emb_fpn_up(y[self.emb_src_fpn])           # L18 -> [B,512,H/4,W/4]
            fused = self.emb_input_proj(torch.cat([hi, fp], 1)) # 768 -> 256 (learned channel weighting)
            return self.emb_head(self.emb_fuse(fused))
        hi = y[self.emb_src_hi]                                  # [B,256,H/8,W/8] post-MSTF P3
        if self.emb_stride == 4:
            hi = self.emb_up(hi)                                 # -> [B,256,H/4,W/4]
            if self.use_pre_mstf:
                lo = y[self.emb_src_lo]                          # [B,128,H/4,W/4] pre-MSTF stride-4
                hi = torch.cat([hi, lo], 1)
        fused = self.emb_fuse(hi)
        return self.emb_head(fused)                              # [B,embed_dim,H/s,W/s]

    def _predict_once(self, x, profile=False, visualize=False, embed=None, mask=False):
        # replicate VideoDetectionModel._predict_once (temporal MSTF net handling), then append E_map.
        y, net, pred_masks = [], [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if isinstance(m, (MSTFv1, MSTFv1_yolo)):
                if len(x) == 3:
                    pred_masks.append(x[2]); x = x[:2]
                net.append(x[1]); x = x[0]
            y.append(x if m.i in self.save else None)
        if not getattr(self, "_omni_ready", False):             # during super().__init__ stride-init
            return (x, net, pred_masks) if mask else (x, net)
        E = self._compute_embedding(y)
        if getattr(self, "_return_taps", False):                # e2e: also return Detect-input tap feats (for distill)
            taps = [y[j] for j in self.model[-1].f]              # Detect.f = [20,23,26]
            return x, net, E, taps
        if mask:
            return x, net, pred_masks, E
        return x, net, E

    def forward_e2e(self, x, n0=None, n1=None, n2=None):
        """End-to-end forward: returns (det_out, net, E_map, detect_tap_feats). Tap feats are the
        Detect inputs (layers 20/23/26) -> distill them against a frozen Detection branch."""
        self._return_taps = True
        try:
            out = self.predict(x, n0, n1, n2, mask=False)
        finally:
            self._return_taps = False
        return out
