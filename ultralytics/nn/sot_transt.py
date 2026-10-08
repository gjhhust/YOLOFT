# Unified YOLOFT SOT: TransT head with portable official pretrained tensor weights loaded into the
# featurefusion + cls/reg heads, on top of the YOLOFT conv backbone (stride-16). Inherits the
# data-hungry template-search matching knowledge (trained on GOT-10k/LaSOT/TrackingNet/COCO);
# only the input_proj + backbone are YOLOFT's -> fine-tune on XS-VID. Module names mirror TransT
# (featurefusion_network / class_embed / bbox_embed) so weights load by name.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import VideoDetectionModel
from ultralytics.nn.modules.block import MSTFv1, MSTFv1_yolo, DySample
from ultralytics.nn.transt_ff import FeatureFusionNetwork

def pos_sine(feat, mask, num_pos_feats=128, temperature=10000):
    """TransT PositionEmbeddingSine (normalize=True), adapted to take feat[B,C,h,w]+mask[B,h,w]."""
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    eps = 1e-6; scale = 2 * math.pi
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=feat.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x_embed[:, :, :, None] / dim_t; pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, in_d, hid, out_d, n):
        super().__init__()
        self.num_layers = n
        h = [hid] * (n - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip([in_d] + h, h + [out_d]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = F.relu(l(x)) if i < self.num_layers - 1 else l(x)
        return x


class YOLOFTSotTransT(VideoDetectionModel):
    """YOLOFT backbone plus TransT feature-fusion and task-head modules."""
    # feat_mode -> (tap layer, in-channels, bolt-on dysample?). head20 = full forward to layer 20
    # (YOLOFT head's TRAINED FPN-upsampled + MSTF P3, stride-8) instead of a raw backbone tap.
    MODES = {"bb11": (11, 512, False), "bb9": (9, 256, False), "bb11dys": (11, 512, True),
             "head20": (20, 256, False), "head23": (23, 512, False),
             "head17": (17, 512, False), "head18": (18, 512, False)}   # 17=FPN P4 s16, 18=FPN-up P3 s8 (both pre-MSTF)
    # fusion: concat same-stride taps; input_proj 1x1 conv = learned channel weighting.
    # fuse918 = raw backbone P3 (l9,256) + FPN-upsampled P3 (l18,512), both stride-8.
    FUSE = {"fuse918": ([9, 18], 256 + 512)}

    def __init__(self, cfg, ch=3, nc=None, verbose=False, feat_mode="bb11",
                 embed=256, n_fusion=4, bb_start=5):
        self._sot_ready = False
        super().__init__(cfg, ch, nc, verbose)
        self.sot = True; self.bb_start = bb_start; self.embed = embed; self.feat_mode = feat_mode
        self.fuse_taps = None; self.use_dys = False
        if feat_mode in self.FUSE:
            self.fuse_taps, feat_ch = self.FUSE[feat_mode]; self.feat_layer = max(self.fuse_taps)
        else:
            self.feat_layer, feat_ch, self.use_dys = self.MODES[feat_mode]
        self.emb_up = DySample(feat_ch, 2) if self.use_dys else None   # stride-16 -> stride-8 (channel-preserving)
        self.input_proj = nn.Conv2d(feat_ch, embed, kernel_size=1)
        self.featurefusion_network = FeatureFusionNetwork(d_model=embed, nhead=8,
                                                          num_featurefusion_layers=n_fusion, dim_feedforward=2048)
        self.class_embed = MLP(embed, embed, 2, 3)     # num_classes(1)+1
        self.bbox_embed = MLP(embed, embed, 4, 3)
        self._sot_ready = True

    def backbone_feat(self, x):
        if self.fuse_taps is not None:                          # multi-tap fusion (concat same-stride feats)
            feats = self._head_forward_multi(x, self.fuse_taps)
            return torch.cat([feats[t] for t in self.fuse_taps], 1)
        if self.feat_mode.startswith("head"):
            return self._head_forward(x, self.feat_layer)       # full forward -> trained FPN/MSTF feature
        for i in range(self.bb_start, self.feat_layer + 1):
            m = self.model[i]; x = m(x)
            if isinstance(m, (MSTFv1, MSTFv1_yolo)):
                x = x[0]
        if self.emb_up is not None:
            x = self.emb_up(x)                                   # DySample to stride-8
        return x

    def _head_forward(self, x, tap):
        """Run the full YOLOFT forward (InputData prelude + backbone + head, MSTF single-frame)
        up to `tap` and return y[tap] (the trained FPN-upsampled + MSTF feature)."""
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if isinstance(m, (MSTFv1, MSTFv1_yolo)):
                x = x[0] if isinstance(x, (list, tuple)) else x
            y.append(x if i in self.save else None)
            if i == tap:
                return x
        return x

    def _head_forward_multi(self, x, taps):
        """Run the full YOLOFT forward and collect outputs at multiple `taps` (for fusion)."""
        want = set(taps); mx = max(taps); out = {}
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if isinstance(m, (MSTFv1, MSTFv1_yolo)):
                x = x[0] if isinstance(x, (list, tuple)) else x
            y.append(x if i in self.save else None)
            if i in want:
                out[i] = x
            if i == mx:
                break
        return out

    def _enc(self, crop):
        f = self.input_proj(self.backbone_feat(crop))           # [B,embed,h,w]
        mask = torch.zeros((f.shape[0], f.shape[2], f.shape[3]), dtype=torch.bool, device=f.device)
        pos = pos_sine(f, mask, self.embed // 2)
        return f, mask, pos

    def forward_sot(self, template, search):
        ft, mt, pt = self._enc(template)
        fs, ms, ps = self._enc(search)
        hs = self.featurefusion_network(ft, mt, fs, ms, pt, ps)  # [1,B,Ns,embed]
        cls = self.class_embed(hs)[-1]                           # [B,Ns,2]
        box = self.bbox_embed(hs)[-1].sigmoid()                  # [B,Ns,4] cxcywh in [0,1]
        hs_g, ws_g = fs.shape[2], fs.shape[3]
        return cls, box, (hs_g, ws_g)

    @torch.no_grad()
    def encode_template(self, template):
        return self._enc(template)

    @torch.no_grad()
    def track(self, tmpl_enc, search):
        ft, mt, pt = tmpl_enc
        fs, ms, ps = self._enc(search)
        hs = self.featurefusion_network(ft, mt, fs, ms, pt, ps)
        cls = self.class_embed(hs)[-1]; box = self.bbox_embed(hs)[-1].sigmoid()
        return cls, box, (fs.shape[2], fs.shape[3])

    def load_transt(self, path):
        """Load tensor-only official TransT heads, skipping ResNet/input_proj.

        Accept a plain state dict or {'net': state_dict}; constructor metadata
        from the original ltr pickle must be removed in the trusted source env.
        """
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            # Old torch lacks the restricted loader; use only verified portable bytes.
            ck = torch.load(path, map_location="cpu")
        if not isinstance(ck, dict):
            raise TypeError("TransT pretrain must be a tensor-only state dictionary")
        sd = ck.get("net", ck.get("state_dict", ck))
        if not isinstance(sd, dict) or not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
            raise TypeError("TransT pretrain must contain only tensors in net/state_dict")
        prefixes = ("featurefusion_network.", "class_embed.", "bbox_embed.")
        msd = self.state_dict()
        expected = {k for k in msd if k.startswith(prefixes)}
        supplied = {k for k in sd if isinstance(k, str) and k.startswith(prefixes)}
        missing = expected - supplied
        unexpected = supplied - expected
        mismatched = {k for k in expected & supplied if msd[k].shape != sd[k].shape}
        if missing or unexpected or mismatched or not expected:
            raise ValueError(f"Incomplete TransT head initialization: missing={len(missing)} "
                             f"unexpected={len(unexpected)} shape_mismatch={len(mismatched)}")
        keep = {k: sd[k] for k in expected}
        if any(not torch.isfinite(v).all().item() for v in keep.values()):
            raise ValueError("TransT pretrain contains non-finite head tensors")
        self.load_state_dict(keep, strict=False)
        return len(keep), len(supplied)
