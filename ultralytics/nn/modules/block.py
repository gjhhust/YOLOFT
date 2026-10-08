import os
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, DCNV3_conv,ModulatedDeformConv,DeformConv, GroupConv
from .transformer import TransformerBlock
from .flow import SepConvGRU, ConvGRU

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

from .odconv import ODConv2d
class Bottleneck_ODConv(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        assert k[1][0] == k[1][1]
        self.cv2 = ODConv2d(c_, c2, k[1][0], s=1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_ODConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_ODConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
from .conv import Focus
    
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
    
class Bottleneck_DC(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        assert k[1][0] == k[1][1]
        self.cv2 = DeformConv(c_, c2, k[1][0], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DC(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DC(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class Bottleneck_MDC(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        assert k[1][0] == k[1][1]
        self.cv2 = ModulatedDeformConv(c_, c2, k[1][0], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C2f_MDC(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_MDC(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class Bottleneck_DCNV3(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        assert k[1][0] == k[1][1]
        self.cv2 = DCNV3_conv(c_, c2, k[1][0], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_DCNV3(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) if i == 0 or i % 2 == 0
            else Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for i in range(n)
        ])
        # self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

from .conv import PartialConv, LightConv
class Bottleneck_light(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = PartialConv(c1)
        assert k[1][0] == k[1][1]
        self.cv2 = DCNV3_conv(c_, c2, k[1][0], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class C2f_light(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.m = nn.ModuleList([
        #     Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) if i == 0 or i % 2 == 0
        #     else Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        #     for i in range(n)
        # ])
        self.m = nn.ModuleList(Bottleneck_light(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


from .memory_buffer import StreamBuffer

class C3_wise(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, depthwise=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        conv = DWConv if depthwise else Conv
        self.cv1 = conv(c1, c_, 1, 1)
        self.cv2 = conv(c1, c_, 1, 1)
        self.cv3 = conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    

class MSTF_STREAM(nn.Module): #
    def __init__(self, in_channels, depth=1.0,
        width=1.0, n_levels=3, depthwise=False,
        act=nn.SiLU(),):
        super(MSTF_STREAM, self).__init__()
        self.in_channels = in_channels
        assert len(in_channels)-1 == n_levels, "features number must be equal to n_levels"
        assert in_channels[0] == in_channels[1:], f"input last feature dims must equal to now feature dims, plase set InputData:[f{in_channels[1:]}]"
        in_channels = in_channels[1:]
        # # buffer
        # self.buffer = StreamBuffer("MemoryAtten", number_feature=n_levels)
        
        conv = DWConv if depthwise else Conv
        
        self.lateral_conv0 = conv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = C3_wise(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )  # cat

        self.reduce_conv1 = conv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = C3_wise(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )

        # bottom-up conv
        self.bu_conv2 = conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = C3_wise(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )

        # bottom-up conv
        self.bu_conv1 = conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = C3_wise(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )

        self.jian2 = nn.Sequential(
            conv(
                int(in_channels[0] * width),
                int(in_channels[0] * width) // 2,
                k=1,
                s=1,
                act=act,
            )  # 添加 conv 模块
        )

        self.jian1 = nn.Sequential(
            conv(
            int(in_channels[1] * width),
            int(in_channels[1] * width) // 2,
                    k=1,
                    s=1,
                    act=act,
                )
        )
        
        self.jian0 = nn.Sequential(
            conv(
            int(in_channels[2] * width),
            int(in_channels[2] * width) // 2,
                    k=1,
                    s=1,
                    act=act,
            )
        )
        
    
    def forward(self, x):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        # self.plot = True
        fmaps_old = x[0]  
        fmaps_new = x[1:]  #3,B,C,H,W The incoming scale must be from large to small
        
        if fmaps_old == [None, None, None]:
            fmaps_old = [torch.zeros_like(f, device=f.device, dtype=f.dtype) for f in fmaps_new]
        else:
            fmaps_old = [fmap.permute(0, 3, 1, 2) for fmap in fmaps_old]
            
        return fmaps_new + fmaps_old
        [fmap2, fmap1, fmap0] = fmaps_old
        [rurrent_x2, rurrent_x1, rurrent_x0] = fmaps_new

        rurrent_fpn_out0 = self.lateral_conv0(rurrent_x0)  # 1/8->1/8
        rurrent_f_out0 = F.interpolate(rurrent_fpn_out0, size=rurrent_x1.shape[2:4], mode='nearest')  # 1/8->1/4
        rurrent_f_out0 = torch.cat([rurrent_f_out0, rurrent_x1], 1)  # 1/4->1/4
        rurrent_f_out0 = self.C3_p4(rurrent_f_out0)  # 1/4->1/16

        rurrent_fpn_out1 = self.reduce_conv1(rurrent_f_out0)  # 1/16->1/16
        rurrent_f_out1 = F.interpolate(rurrent_fpn_out1, size=rurrent_x2.shape[2:4], mode='nearest')  # 256/8
        rurrent_f_out1 = torch.cat([rurrent_f_out1, rurrent_x2], 1)  # 256->512/8
        rurrent_pan_out2 = self.C3_p3(rurrent_f_out1)  # 512->256/8

        rurrent_p_out1 = self.bu_conv2(rurrent_pan_out2)  # 256->256/16
        rurrent_p_out1 = torch.cat([rurrent_p_out1, rurrent_fpn_out1], 1)  # 256->512/16
        rurrent_pan_out1 = self.C3_n3(rurrent_p_out1)  # 512->512/16

        rurrent_p_out0 = self.bu_conv1(rurrent_pan_out1)  # 512->512/32
        rurrent_p_out0 = torch.cat([rurrent_p_out0, rurrent_fpn_out0], 1)  # 512->1024/32
        rurrent_pan_out0 = self.C3_n4(rurrent_p_out0)  # 1024->1024/32

        # result_first_frame, fmaps_old = self.buffer.update_memory([rurrent_pan_out2,rurrent_pan_out1,rurrent_pan_out0], img_metas)
        
        # 融合
        pan_out2 = torch.cat([self.jian2(rurrent_pan_out2), self.jian2(fmap2)], dim=1) + rurrent_x2
        pan_out1 = torch.cat([self.jian1(rurrent_pan_out1), self.jian1(fmap1)], dim=1) + rurrent_x1
        pan_out0 = torch.cat([self.jian0(rurrent_pan_out0), self.jian0(fmap0)], dim=1) + rurrent_x0
        
        return [pan_out2, pan_out1, pan_out0]

from ultralytics.nn.modules.conv import CBAM

class MSTF_STREAM_cbam(nn.Module): #
    def __init__(self, in_channels, depth=1.0,
        width=1.0, n_levels=3, depthwise=False,
        act=nn.SiLU(),):
        super(MSTF_STREAM_cbam, self).__init__()
        self.in_channels = in_channels
        assert len(in_channels)-1 == n_levels, "features number must be equal to n_levels"
        assert in_channels[0] == in_channels[1:], f"input last feature dims must equal to now feature dims, plase set InputData:[f{in_channels[1:]}]"
        in_channels = in_channels[1:]
        # # buffer
        # self.buffer = StreamBuffer("MemoryAtten", number_feature=n_levels)
        
        conv = DWConv if depthwise else Conv
        
        self.lateral_conv0 = conv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = C3_wise(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )  # cat

        self.reduce_conv1 = conv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = C3_wise(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )

        # bottom-up conv
        self.bu_conv2 = conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = C3_wise(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )

        # bottom-up conv
        self.bu_conv1 = conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = C3_wise(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
        )

        self.jian2 = nn.Sequential(
            conv(
                int(in_channels[0] * width),
                int(in_channels[0] * width) // 2,
                k=1,
                s=1,
                act=act,
            )  # 添加 conv 模块
        )

        self.jian1 = nn.Sequential(
            conv(
            int(in_channels[1] * width),
            int(in_channels[1] * width) // 2,
                    k=1,
                    s=1,
                    act=act,
                )
        )
        
        self.jian0 = nn.Sequential(
            conv(
            int(in_channels[2] * width),
            int(in_channels[2] * width) // 2,
                    k=1,
                    s=1,
                    act=act,
            )
        )
        
    
    def forward(self, x):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        # self.plot = True
        fmaps_old = x[0]  
        fmaps_new = x[1:]  #3,B,C,H,W The incoming scale must be from large to small
        if fmaps_old == [None, None, None]:
            fmaps_old = [torch.zeros_like(f, device=f.device, dtype=f.dtype) for f in fmaps_new]
        else:
            fmaps_old = [fmap.permute(0, 3, 1, 2) for fmap in fmaps_old]
        
        [fmap2, fmap1, fmap0] = fmaps_old
        [rurrent_x2, rurrent_x1, rurrent_x0] = fmaps_new

        rurrent_fpn_out0 = self.lateral_conv0(rurrent_x0)  # 1/8->1/8
        rurrent_f_out0 = F.interpolate(rurrent_fpn_out0, size=rurrent_x1.shape[2:4], mode='nearest')  # 1/8->1/4
        rurrent_f_out0 = torch.cat([rurrent_f_out0, rurrent_x1], 1)  # 1/4->1/4
        rurrent_f_out0 = self.C3_p4(rurrent_f_out0)  # 1/4->1/16

        rurrent_fpn_out1 = self.reduce_conv1(rurrent_f_out0)  # 1/16->1/16
        rurrent_f_out1 = F.interpolate(rurrent_fpn_out1, size=rurrent_x2.shape[2:4], mode='nearest')  # 256/8
        rurrent_f_out1 = torch.cat([rurrent_f_out1, rurrent_x2], 1)  # 256->512/8
        rurrent_pan_out2 = self.C3_p3(rurrent_f_out1)  # 512->256/8

        rurrent_p_out1 = self.bu_conv2(rurrent_pan_out2)  # 256->256/16
        rurrent_p_out1 = torch.cat([rurrent_p_out1, rurrent_fpn_out1], 1)  # 256->512/16
        rurrent_pan_out1 = self.C3_n3(rurrent_p_out1)  # 512->512/16

        rurrent_p_out0 = self.bu_conv1(rurrent_pan_out1)  # 512->512/32
        rurrent_p_out0 = torch.cat([rurrent_p_out0, rurrent_fpn_out0], 1)  # 512->1024/32
        rurrent_pan_out0 = self.C3_n4(rurrent_p_out0)  # 1024->1024/32

        # result_first_frame, fmaps_old = self.buffer.update_memory([rurrent_pan_out2,rurrent_pan_out1,rurrent_pan_out0], img_metas)
        
        # 融合
        pan_out2 = torch.cat([self.jian2(rurrent_pan_out2), self.jian2(fmap2)], dim=1) + rurrent_x2
        pan_out1 = torch.cat([self.jian1(rurrent_pan_out1), self.jian1(fmap1)], dim=1) + rurrent_x1
        pan_out0 = torch.cat([self.jian0(rurrent_pan_out0), self.jian0(fmap0)], dim=1) + rurrent_x0
        
        return [pan_out2, pan_out1, pan_out0]

import math
class FlowDown(nn.Module):
    def __init__(self, c_4, c_2, c ,cout):
        super().__init__()
        self.c = cout // 2
        half_c_4 = math.ceil(c_4 / 2)
        half_c_2 = math.ceil(c_2 / 2)
        half_c = math.ceil(c / 2)

        half_c_4_1 = c_4 // 2
        half_c_2_1 = c_2 // 2
        half_c_1 = c // 2

        self.cv1 = Conv(half_c_4, self.c, 3, 2, 1)
        self.cv2 = Conv(half_c_4+half_c_2, self.c, 3, 2, 1)

        self.cv3 = Conv(self.c + half_c, self.c, 1, 1, 0)
        self.cv4 = Conv(half_c_1 + half_c_4_1 + half_c_2_1, self.c, 1, 1, 0)

    def forward(self, f_4, f_2, f):
        f_4 = torch.nn.functional.avg_pool2d(f_4, 2, 1, 0, False, True)
        f4_1, f4_2 = f_4.chunk(2, 1)
        f4_1 = self.cv1(f4_1)
        f4_2 = torch.nn.functional.max_pool2d(f4_2, 3, 2, 1)

        f_2 = torch.nn.functional.avg_pool2d(f_2, 2, 1, 1, False, True)[:,:,:f4_1.shape[2], :f4_1.shape[3]]
        f2_1, f2_2 = f_2.chunk(2, 1)
        f2_1 = self.cv2(torch.cat((f2_1, f4_1), 1))

        f2_2 = torch.nn.functional.max_pool2d(f2_2, 3, 2, 1)
        f4_2 = torch.nn.functional.max_pool2d(f4_2, 3, 2, 1)
        
        f1, f2 = f.chunk(2, 1)

        f1 = self.cv3(torch.cat((f2_1, f1), 1))
        f2 = self.cv4(torch.cat((f2_2, f4_2, f2), 1))

        return torch.cat((f1, f2), 1)
    
class ConvTranspose(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, kernel_size, stride, padding, output_padding=output_padding, bias=False)
        # self.norm = nn.GroupNorm(1, c2)  # 分组数为1，每个分组包含一个通道
        self.norm = nn.Identity()
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class FlowUp(nn.Module):
    def __init__(self, cx, cy, cout):
        super().__init__()
        self.c = cout // 2
        half_x = math.ceil(cx / 2)
        half_x_2 = cx // 2
        self.cv1 = ConvTranspose(half_x, self.c, 3, 2, 1, 1)
        self.cv2 = Conv(half_x_2 + cy, self.c, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, y):
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = self.upsample(x2)
        x2 = self.cv2(torch.cat((x2, y), 1))
        return torch.cat((x1, x2), 1)

class FlowEncoder(nn.Module):
    def __init__(self, gru, c, cout):
        super().__init__()
        self.gru = gru
        self.mask = Conv(c, cout, 1)
        self.offset = Conv(c, cout, 3)

    def forward(self, x):
        x = self.gru(x)
        return  self.mask(x), self.offset(x)
    
import warnings
try:
    from ultralytics.nn.modules.ops_dcnv3.modules import DCNv3Function, dcnv3_core_pytorch, CenterFeatureScaleModule,build_norm_layer, build_act_layer
    from torch.nn.init import xavier_uniform_, constant_
    print("using DCNv3")
except ImportError:
    print("ERROR: No DCNv3")
    
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, temporal_kernel=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if temporal_kernel not in (1, 2):
            raise ValueError("temporal_kernel must be 1 (legacy) or 2 (causal)")
        self.temporal_kernel = temporal_kernel

        self.depthwise_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(temporal_kernel, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            groups=in_channels,
            bias=bias
        )
        self.depthwise_bn = nn.BatchNorm3d(in_channels)
        self.depthwise_act = nn.SiLU()

        self.pointwise_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            stride=1,
            padding=0,
            bias=bias
        )
        self.pointwise_bn = nn.BatchNorm3d(out_channels)
        self.pointwise_act = nn.SiLU()

    def forward(self, x1, x2):
        x = torch.stack((x1, x2), dim=2) # Output shape: [b, c, 2, h, w]

        out = self.depthwise_conv(x) # Shape: [b, c, 2, h', w']
        out = self.depthwise_bn(out)
        out = self.depthwise_act(out)
        out = self.pointwise_conv(out) # Shape: [b, out_channels, 2, h', w']
        out = self.pointwise_bn(out)
        out = self.pointwise_act(out)

        return out[:, :, -1, :, :]
    
class FDCN(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(
            self,
            gru,
            channels=64,
            net_channels=32,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            act=True,
            is_encoder = True,
            ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.gru = gru
        self.is_encoder = is_encoder
        if is_encoder:
            self.encoder = Conv(channels, net_channels, k=1)
            
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                net_channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding= autopad(dw_kernel_size, (dw_kernel_size - 1) // 2, 1), 
                groups=net_channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()


    def reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input_x, input_y):
        """
        :param input                       (N, C, H, W)
        :param input_y                      (N, C//2, H, W)
        :return output                     (N, C, H, W)
        """
        if self.is_encoder:
            input_x_en = self.encoder(input_x)  #(N, C//2, H, W)
        else:
            input_x_en = input_x
        input_y = self.gru(input_y, input_x_en)  #(N, C//2, H, W)
        input = input_x.permute(0, 2, 3, 1) #(N, H, W, C)
        
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        dtype = x.dtype

        y = self.dw_conv(input_y) #(N, C, H, W)
        offset = self.offset(y)
        mask = self.mask(y).reshape(N, H, W, self.group, -1)
        
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)
        if not x.is_cuda:
            x = dcnv3_core_pytorch(
                x, offset, mask,
                self.kernel_size, self.kernel_size,
                self.stride, self.stride,
                self.pad, self.pad,
                self.dilation, self.dilation,
                self.group, self.group_channels,
                self.offset_scale)
            print("warning: now using DCNv3 pytroch version")
        else:
            x = DCNv3Function.apply(
                x, offset, mask,
                self.kernel_size, self.kernel_size,
                self.stride, self.stride,
                self.pad, self.pad,
                self.dilation, self.dilation,
                self.group, self.group_channels,
                self.offset_scale,
                256)

        x = self.output_proj(x)
        return self.act(x.permute(0, 3, 1, 2)).contiguous(), input_y.contiguous()
    
class MSTFDC(nn.Module):
    def __init__(
            self,
            in_channels,
            ):
        """
        MSTF Module
        """
        super().__init__()
        assert in_channels[0] == in_channels[1:], f"input last feature dims must equal to now feature dims, plase set InputData:[f{in_channels[1:]}]"
        in_channels = in_channels[1:]
        self.flows = nn.ModuleList([
            FDCN(SepConvGRU(hidden_dim=ic, input_dim=ic//3), ic) if i == 0 else FDCN(ConvGRU(hidden_dim=ic, input_dim=ic//3), ic)
            for i, ic in enumerate(in_channels)
        ])
        
    def forward(self, x):
        # self.plot = True
        fmaps_old = x[0]  
        fmaps_new = x[1:]  #3,B,C,H,W The incoming scale must be from large to small
        if fmaps_old == [None, None, None]:
            fmaps_old = [torch.zeros_like(f, device=f.device, dtype=f.dtype) for f in fmaps_new]
        else:
            fmaps_old = [fmap.permute(0, 3, 1, 2) for fmap in fmaps_old]
        
        fs = []
        nets = []
        for fn, fo, f_layer in zip(fmaps_new, fmaps_old, self.flows):
            f, net = f_layer(fn, fo)
            fs.append(f+fn)
            nets.append(net)
        return fs+nets
        # return fmaps_new+fmaps_old
    
class MSTFDC_T(nn.Module):
    def __init__(
            self,
            in_channels,
            ):
        """
        MSTF Module
        """
        super().__init__()
        assert in_channels[0] == in_channels[1:], f"input last feature dims must equal to now feature dims, plase set InputData:[f{in_channels[1:]}]"
        in_channels = in_channels[1:]
        self.flows = nn.ModuleList([
            nn.ModuleList(
                [Conv(ic, ic//3, k=1),
                ConvGRU(hidden_dim=ic, input_dim=ic//3)])
            for i, ic in enumerate(in_channels)
        ])
        
    def forward(self, x):
        # self.plot = True
        fmaps_old = x[0]  
        fmaps_new = x[1:]  #3,B,C,H,W The incoming scale must be from large to small
        if fmaps_old == [None, None, None]:
            fmaps_old = [torch.zeros_like(f, device=f.device, dtype=f.dtype) for f in fmaps_new]
        else:
            fmaps_old = [fmap.permute(0, 3, 1, 2) for fmap in fmaps_old]
        
        fs = []
        nets = []
        for fn, fo, f_layer in zip(fmaps_new, fmaps_old, self.flows):
            fn_ = f_layer[0](fn)
            net = f_layer[1](fo, fn_) 
            fs.append(net+fn)
            nets.append(net)
        return fs+nets

class MSTF_BLOCK(nn.Module):
    def __init__(
            self,
            in_channels,
            ):
        """
        MSTF Module
        """
        super().__init__()
        assert in_channels[0] == in_channels[1:], f"input last feature dims must equal to now feature dims, plase set InputData:[f{in_channels[1:]}]"
        in_channels = in_channels[1:]
        print("empty video moudle")
        
    def forward(self, x):
        # self.plot = True
        fmaps_old = x[0]  
        fmaps_new = x[1:]  #3,B,C,H,W The incoming scale must be from large to small
        if fmaps_old == [None, None, None]:
            fmaps_old = [torch.zeros_like(f, device=f.device, dtype=f.dtype) for f in fmaps_new]
        else:
            fmaps_old = [fmap.permute(0, 3, 1, 2) for fmap in fmaps_old]
    
        return fmaps_new+fmaps_old

# yolov8训练框架yoloft
class MSTFv1_yolo(nn.Module):
    def __init__(self, in_channels, dimension=1, mode="concat", width=0.25, nc=7):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
        in_channel = in_channels[0] + in_channels[1]
        self.net_dim = int(in_channel*width)
        if mode == "concat":
            self.forward = self.forward_concat
        elif mode == "net":
            assert self.net_dim == in_channels[-1], f"input last feature dims must equal to now feature dims, plase set InputData:[{in_channels[-1]}]->[{self.net_dim}]"
            self.block = FDCN(ConvGRU(hidden_dim=self.net_dim, input_dim=self.net_dim), 
                                        in_channel, 
                                        self.net_dim, 
                                        kernel_size = 3)
            self.forward = self.forward_mstf
        elif mode == "conv3d":
            # print(self.net_dim)
            in_channels.append(self.net_dim)
            print(in_channels)
            assert self.net_dim == in_channels[-1], f"input last feature dims must equal to now feature dims, plase set InputData:[{in_channels[-1]}]->[{self.net_dim}]"
            self.block = FDCN(DepthSeparableConv3D(in_channels=self.net_dim, out_channels=self.net_dim), 
                                        in_channel, 
                                        self.net_dim, 
                                        kernel_size = 3,
                                        is_encoder=True)
            self.forward = self.forward_mstf
        else:
            raise ValueError("MSTFv1 paramter [mode] must be in [concat, net, conv3d]")

        self.mask_aux_net = nn.Conv2d(self.net_dim, 1, 1) #only training mask aux net
        self.net = None

    def forward_concat(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        x1, x2, net_x = x
        return torch.cat([x1,x2], self.d), net_x

    def forward_mstf(self, x):
        x1, x2  = x
        net_x = self.net
        input_x = torch.cat([x1,x2], self.d)
        b, c, h, w = input_x.shape
        if net_x is None:
            net_x = torch.zeros([b, self.net_dim, h, w], device=input_x.device, dtype=input_x.dtype)
        elif net_x.shape!= (b, self.net_dim, h, w):
            net_x = torch.zeros([b, self.net_dim, h, w], device=input_x.device, dtype=input_x.dtype)

        x, net = self.block(input_x, net_x)
        x = x + input_x
        self.net = net.detach()
        
        return x

# yoloft 正式版本
class MSTFv1(nn.Module):
    def __init__(
        self,
        in_channels,
        dimension=1,
        mode="concat",
        width=0.25,
        nc=7,
        residual_init=1.0,
        first_frame_identity=False,
        temporal_kernel=2,
    ):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
        in_channel = in_channels[0] + in_channels[1]
        self.net_dim = int(in_channel*width)
        if mode == "concat":
            self.forward = self.forward_concat
        elif mode == "net":
            assert self.net_dim == in_channels[-1], f"input last feature dims must equal to now feature dims, plase set InputData:[{in_channels[-1]}]->[{self.net_dim}]"
            self.block = FDCN(ConvGRU(hidden_dim=self.net_dim, input_dim=self.net_dim), 
                                        in_channel, 
                                        self.net_dim, 
                                        kernel_size = 3)
            self.forward = self.forward_mstf
        elif mode == "conv3d":
            assert self.net_dim == in_channels[-1], f"input last feature dims must equal to now feature dims, plase set InputData:[{in_channels[-1]}]->[{self.net_dim}]"
            self.block = FDCN(DepthSeparableConv3D(in_channels=self.net_dim, out_channels=self.net_dim, temporal_kernel=temporal_kernel),
                                        in_channel, 
                                        self.net_dim, 
                                        kernel_size = 3,
                                        is_encoder=True)
            self.forward = self.forward_mstf
        else:
            raise ValueError("MSTFv1 paramter [mode] must be in [concat, net, conv3d]")

        if mode != "concat":
            self.temporal_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.temporal_enabled = mode != "concat"
        self.temporal_mode = mode
        self.first_frame_identity = bool(first_frame_identity)
        self.mask_aux_net = nn.Conv2d(self.net_dim, 1, 1) #only training mask aux net
        self.net = None

    def forward_concat(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        x1, x2, net_x = x
        return torch.cat([x1,x2], self.d), net_x

    def _temporal_fp32_requested(self):
        """Read both local and DDP-inherited precision controls."""
        return getattr(self, "temporal_fp32", False) or os.environ.get("YOLOFT_TEMPORAL_FP32") == "1"

    def forward_mstf(self, x):
        if self._temporal_fp32_requested() and torch.is_autocast_enabled():
            x1, x2, net_x = x
            with torch.autocast(device_type=x1.device.type, enabled=False):
                fp32_net = None if net_x is None else net_x.float()
                return self._forward_mstf_impl((x1.float(), x2.float(), fp32_net))
        return self._forward_mstf_impl(x)

    def _forward_mstf_impl(self, x):
        x1, x2, net_x = x
        input_x = torch.cat([x1,x2], self.d)
        b, c, h, w = input_x.shape
        first_frame = net_x is None
        temporal_enabled = getattr(self, "temporal_enabled", True)
        if first_frame:
            net_x = torch.zeros([b, self.net_dim, h, w], device=input_x.device, dtype=input_x.dtype)
        elif not temporal_enabled:
            # Preserve the history input in exported graphs while clearing its value.
            net_x = net_x * 0.0

        if not temporal_enabled:
            if self.training:
                mask_aux = torch.zeros(
                    (b, 1, h, w), device=input_x.device, dtype=input_x.dtype
                )
                return input_x, net_x, mask_aux
            return input_x, net_x

        temporal_x, net = self.block(input_x, net_x)
        first_frame_identity = getattr(self, "first_frame_identity", False)
        temporal_scale = getattr(self, "temporal_scale", None)
        scale = temporal_scale if temporal_scale is not None else 1.0
        x = input_x if first_frame and first_frame_identity else input_x + scale * temporal_x
        
        if self.training:
            mask_aux = self.mask_aux_net(net)
            # mask_aux = torch.ones_like(net)
            return x, net, mask_aux
        else:
            return x, net
    
import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.register_buffer('init_pos', self._init_pos())

    def reset_parameters(self):
        normal_init(self.offset, std=0.001)
        if hasattr(self, 'scope'):
            constant_init(self.scope, val=0.)
            
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample2XTest(nn.Module):
    def __init__(self, in_channels, upsample_method='nearest'):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_method = upsample_method

        # Define layers for learnable upsampling methods in the constructor
        if upsample_method == 'convtranspose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        elif upsample_method == 'subpixelconv':
            out_channels = in_channels // 4
            if in_channels % 4 != 0:
                raise ValueError(f"Input channels ({in_channels}) must be divisible by 4 for SubpixelConv with scale factor 2.")
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2)
            )
        elif upsample_method == 'deconv_bn_relu':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        elif upsample_method == 'espcn':
            # Placeholder for ESPCN layers - you'll need to define the architecture
            # based on your specific ESPCN implementation
            self.up = ESPCNUpsample2X(in_channels)
        elif upsample_method == 'carafe':
            # Placeholder for CARAFE layers - you'll need to define the architecture
            # based on your specific CARAFE implementation
            self.up = CARAFEUpsample2X(in_channels)
        else:
            self.up = None # For non-learnable methods, no specific layer in init

    def forward(self, x):
        # Assuming x is always in [b, c, h, w] format
        method = self.upsample_method
        if method == 'nearest':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        elif method == 'linear':
            raise ValueError("Linear interpolation is not suitable for 2D (image) upsampling. Consider 'bilinear'.")
        elif method == 'bilinear':
            return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif method == 'bicubic':
            return F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        elif method == 'trilinear':
            raise ValueError("Trilinear interpolation is for 3D data. Your input is 2D (image).")
        elif method in ['pixelshuffle', 'convtranspose', 'subpixelconv', 'deconv_bn_relu', 'espcn', 'carafe']:
            if self.up is not None:
                return self.up(x)
            else:
                raise NotImplementedError(f"Layer for upsampling method '{method}' not initialized correctly.")
        else:
            raise ValueError(f"Unsupported upsampling method: {method}")

# --- Implementations for advanced upsampling modules ---

class CARAFEUpsample2X(nn.Module):
    def __init__(self, in_channels, kernel_size=5, up_scale=2, compress_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_scale = up_scale
        self.compress_factor = compress_factor
        self.encoder = nn.Conv2d(in_channels, in_channels // compress_factor, kernel_size=1)
        self.decoder = nn.Conv2d(in_channels // compress_factor, up_scale * up_scale * kernel_size * kernel_size, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        n, c, h, w = x.size()
        kernel_pred = self.decoder(self.encoder(x))
        kernel_pred = kernel_pred.view(n, self.up_scale * self.up_scale, self.kernel_size * self.kernel_size, h, w)
        kernel_pred = F.softmax(kernel_pred, dim=2)

        output = torch.zeros(n, c, h * self.up_scale, w * self.up_scale, device=x.device)
        for i in range(h * self.up_scale):
            for j in range(w * self.up_scale):
                y = i // self.up_scale
                x_ = j // self.up_scale
                output[:, :, i, j] = torch.sum(
                    x[:, :, max(0, y - self.kernel_size // 2):min(h, y + self.kernel_size // 2 + 1),
                      max(0, x_ - self.kernel_size // 2):min(w, x_ + self.kernel_size // 2 + 1)] *
                    kernel_pred[:, (i % self.up_scale) * self.up_scale + (j % self.up_scale), :, y, x_].view(n, 1, -1),
                    dim=-1
                )
        return output

class ESPCNUpsample2X(nn.Module):
    def __init__(self, in_channels, num_features=64, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, out_channels * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        return F.pixel_shuffle(x, upscale_factor=2)
        

class MANetDCN(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))
    
class MANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            sum_val = torch.sum(path, dim=2, keepdim=True)  # 先计算路径和
            norm_out = 1 / sum_val  # 计算倒数
            # 通过分母是否为0生成mask，并将对应位置置0
            norm_out[sum_val == 0] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")


    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()


    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))

        return x
    
class Segmenter(nn.Module):
    def __init__(self, nc=10, ch=()):
        super(Segmenter, self).__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)  # output conv
    
    def forward(self, x):
        return [self.m[i](x[i]) for i in range(len(x))]
# if __name__ == '__main__':
#     x = torch.rand(2, 64, 4, 7)
#     dys = DySample(64)
#     print(dys(x).shape)
