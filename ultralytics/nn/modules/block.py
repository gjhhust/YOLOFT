# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, Conv2, DCNV3_conv,autopad,Conv_gn
from .conv import Conv_onany as flow_conv
print("now flow_conv:", flow_conv)
from .transformer import TransformerBlock
from .utils import getPatchFromFullimg,normMask,transform,DLT_solve,homo_align
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from .memory_buffer import MutiFeatureBuffer, FeatureBuffer, FlowBuffer
# __all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
#           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3')
from torch.cuda.amp import autocast, GradScaler #

class get_data(nn.Module):
    def __init__(self,mode=None,split=[0,2]):
        super(get_data, self,).__init__()
        self.mode = mode
        assert len(split) == 2, LOGGER.error(f"split must include start and stop")
        self.split = slice(split[0],split[1])
        LOGGER.info(f"{self.mode} data shape:{split}")
    def forward(self, x):
        with torch.no_grad():
            if self.mode == "split":
                # LOGGER.info(f"{self.mode} data shape:{x.shape}")
                return torch.clone(x)
            else:
                split_x = x[:,self.split,::]
                # LOGGER.info(f"{self.mode} data shape:{split_x.shape}")
                return torch.clone(split_x)
            


class printshape(nn.Module):
    def __init__(self):
        super(printshape, self).__init__()
        pass
    def forward(self, x):
        LOGGER.info(f"shape:{x.shape}")
        return x

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
   
        return x * self.se(x)



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.in_size = in_size
        self.out_size = out_size

        if semodule:
            self.se = SeModule(semodule)
        else:
            self.se = None
            LOGGER.warning(f"semodule is None")

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)

        if nolinear == "ReLU":
            self.nolinear1 = nn.ReLU(inplace=True)
            self.nolinear2 = nn.ReLU(inplace=True)
        elif nolinear == "hswish":
            self.nolinear1 = hswish()
            self.nolinear2 = hswish()
        else:
            LOGGER.error(f"Block module nolinear not include {nolinear},please change eg.(ReLU,hswish)")

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))

        if self.stride == 2:
            out = torch.nn.functional.avg_pool2d(out, 2, 1, 0, False, True)####

        # out = out
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.se != None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
 
        out = out + self.shortcut(x) if (self.in_size == self.out_size and self.stride == 1) else out
        return out

class BlockE1(nn.Module):
    '''depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(BlockE1, self).__init__()
        self.stride = stride
        self.in_size = in_size
        self.out_size = out_size

        if semodule:
            self.se = SeModule(semodule)
        else:
            self.se = None
            LOGGER.warning(f"semodule is None")

        self.conv2 = nn.Conv2d(in_size, in_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(in_size)

        if nolinear == "ReLU":
            self.nolinear2 = nn.ReLU(inplace=True)
        elif nolinear == "hswish":
            self.nolinear2 = hswish()
        else:
            LOGGER.error(f"Block module nolinear not include {nolinear},please change eg.(ReLU,hswish)")


        self.conv3 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        # out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(x)))
        if self.se != None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
 
        out = out + self.shortcut(x) if self.in_size == self.out_size else out
        return out


class MobileNetConv1(nn.Module):
    
    def __init__(self):
        super(MobileNetConv1,self).__init__()
        self._conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),####224
            nn.BatchNorm2d(24),
            hswish())
    def forward(self,x):
        return self._conv1(x)
    

class MobileNetV3BC(nn.Module):
    def __init__(
        self, 
        out_indices=[12, 15, 16],
        norm_eval=False,
        norm_eps=1e-5):
        super(MobileNetV3BC, self).__init__()


        _conv1 = MobileNetConv1()

        _block1 = nn.Sequential()
        self.blocks = nn.ModuleList((
            _conv1._conv1,                                         #                        0
            BlockE1(3, 24, 24, 24, nn.ReLU(inplace=True), None, 1),####112                  1
            Block(3, 24, 72, 36, nn.ReLU(inplace=True), None, 2),####112                    2
            Block(3, 36, 108, 36, nn.ReLU(inplace=True), None, 1),####56                    3

            Block(3, 36, 108, 36, nn.ReLU(inplace=True), None, 1),####56                    4
            Block(3, 36, 108, 36, nn.ReLU(inplace=True), None, 1),####56                    5 #

            Block(5, 36, 108, 60, nn.ReLU(inplace=True), SeModule(108), 2),####56           6
            Block(5, 60, 180, 60, nn.ReLU(inplace=True), SeModule(180), 1),####28           7
            Block(5, 60, 180, 60, nn.ReLU(inplace=True), SeModule(180), 1),####28           8 #
            Block(3, 60, 240, 80, hswish(), None, 2),####28                                 9
            Block(3, 80, 200, 80, hswish(), None, 1),####14                                 10
            Block(3, 80, 184, 80, hswish(), None, 1),####14                                 11
            Block(3, 80, 184, 80, hswish(), None, 1),####14                                 12
            Block(3, 80, 480, 112, hswish(), SeModule(480), 1),####14                       13
            Block(3, 112, 672, 112, hswish(), SeModule(672), 1),####14                      14 #
            Block(5, 112, 672, 160, hswish(), SeModule(672), 2),####14                      15
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),####7                       16
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),####7                       17 #
            Block(5, 160, 960, 160, hswish(), SeModule(960), 2),####fifth stage for fpn     18
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),####fifth stage for fpn     19
        ))
 

        # self.init_params()

        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.init_weights()
        self._freeze_stages()
        self._set_bn_param(0.1, norm_eps)

    def forward(self, x):

        outs = []
        for block_ind, block in enumerate(self.blocks):
            x = block(x)
            if block_ind in self.out_indices:
                outs.append(x)
                if len(outs) == len(self.out_indices):

                    break
        return tuple(outs)



    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            model_dict = self.state_dict()

            load_checkpoint(self, pretrained, strict=False, logger=logger)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(MobileNetV3BC, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


    def _freeze_stages(self):
        pass

    def _set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return

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
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
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
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
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

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

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
    
class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
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

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        if isinstance(x, tuple):
            x = x[0]
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2g(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((1 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.pop(1)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.pop(1)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
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
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
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

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class Bottleneck(nn.Module):
#     """Bottleneck with shortcut weight."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#         self.weight = nn.Parameter(torch.zeros(1))  # add this line, create learnable weights
#
#     def forward(self, x):
#         """'forward()' applies the YOLOv5 FPN to input data."""
#         return x * self.weight.sigmoid() + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
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


# EXPERIMENTAL ---------------------------------------------------------------------------------------------------------


class Cxa(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cxb(nn.Module):
    """BAD"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv5(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv5((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cxc(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv5(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cxc_act(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv5(self.c, self.c, k=3, act=True) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cxd(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv5(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv5((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv5(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cxe(nn.Module):
    """BAD - CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv5(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv5((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


    def forward(self, x):
        """Forward pass for C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cxf(nn.Module):
    """BAD - CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck2(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass for C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck2(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv5(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv5(nn.Module):
    def __init__(self, c1, c2, k=3, act=False, *args):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        self.cv1 = Conv(c1, c2, k, act=act)
        self.cv2 = DWConv(c2, c2, 5)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class C3f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.cv3 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv1(x), self.cv2(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class Cg1(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((1 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cg2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 3, 1)
        self.cv2 = Conv((1 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cg3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(sum(y))


class Cg4(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 3)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(sum(y))


class Cg7(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 3)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c, self.c, k=3) for _ in range(n))
        self.weight = nn.Parameter(torch.ones(n + 1) * 5)  # add this line, create learnable weights

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = [w * tensor for w, tensor in zip(self.weight.tanh(), y[1:])]
        return self.cv2(sum(y))


class Cg8(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        n = n * 2
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 3)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(sum(y))


# Conv2() experiments --------------------------------------------------------------------------------------------------

# class Cxa(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         n = n * 2
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Conv2(self.c, self.c, k=3) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#
# class Cxb(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         n = n * 2
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv2(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv2((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Conv2(self.c, self.c, k=3) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#
# class Cxc(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         n = n * 2
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv2(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv2((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Conv(self.c, self.c, k=3) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))


class Cg5(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck611(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cg6(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv2(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv2((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck611(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cg3(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv2(c1, self.c, 1, 1)
        self.cv2 = Conv2((1 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Cg4(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((1 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck22(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck611(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv2(c1, c_, k[0], 1)
        self.cv2 = Conv2(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck22(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv22(c1, c_, k[0], 1)
        self.cv2 = Conv22(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv22(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 2, s, 0, groups=g, dilation=d, bias=False)
        self.relu = nn.ReLU()
        self.cv2 = nn.Conv2d(c2, c2, 2, s, 1, groups=g, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn2(self.cv2(self.relu(self.bn1(self.cv1(x))))))

class ShareFeature(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=3, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.m = nn.ModuleList(C2f(self.c, self.c, n=1, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.m = nn.Sequential(
            C2f(1, 4, n=1, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            C2f(4, 8, n=1, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            C2f(8, 1, n=1, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    # def forward(self, x):
    #     """Forward pass through C2f layer."""
    #     y = list(self.cv1(x).chunk(2, 1))
    #     y.extend(m(y[-1]) for m in self.m)
    #     return self.cv2(torch.cat(y, 1))
    
    def forward(self,x):
        return self.m(x)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class get_Homograph_data(nn.Module):
    inplanes = 32
    output_dict = {
            "loss": 1,
            "layer1": inplanes,
            "layer2": inplanes*2,
            "layer3": inplanes*4,
            "layer4": inplanes*8
    }

    def __init__(self, inplanes, key=""):
        super(get_Homograph_data, self,).__init__()

        

        assert key in self.output_dict, LOGGER.error(f"Homograph not key:{key} in this output")
        self.key = key
    def forward(self, x):
        return x[self.key].clone()
    

class get_orige_data(nn.Module):
    output_dict = {
        "split": 0,#dict
        "backbone": 3,
        "motion": 0 #dict
    }
    def __init__(self,mode=None):
        super(get_orige_data, self,).__init__()
        self.mode = mode
        if self.mode != "split":
            assert mode in self.output_dict, LOGGER.error(f"key not in get_orige_data orige_dicts")
    def forward(self, x):
        if self.mode == "split":
            if isinstance(x, dict):
                return x
            else:
                if x.size()[1:] == (3,256,256) or x.size()[1:] == (3,640,640):#info
                    return {
                        "backbone":torch.ones_like(x).to(x.device),
                        "motion":  x.device,
                        "img_metas": [{"is_first":True,"epoch":0}],
                    }

        elif self.mode == "backbone": 
            return x[self.mode].clone()
        else: 
            return x[self.mode]


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
        self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

from .memory_buffer import StreamTensorMemory,StreamTensorMutiMemory
from .ConvGru import ConvGRU

        
class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame(include current). Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, buffer_name, c1, num_segments=3, shift_div=8):
        super().__init__()
        self.memory = StreamTensorMutiMemory(buffer_name, num_segments)
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.epoch_train = 10
        self.fused = C2f(c1*num_segments, c1, n=1, shortcut=True)

    def forward(self, x): # [N, C, H, W] -> [N, C, H, W]
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        input_tensor = x[0]
        img_metas = x[1]["img_metas"]
        bs = len(input_tensor)
        
        # 训练时前self.epoch_train不融合
        if self.training:
            epoch = x[1]["img_metas"][0]["epoch"]
            if epoch < self.epoch_train:
                return input_tensor
            
        self.memory.update(input_tensor, img_metas)
        
        fused_output_list = []
        
        for i in range(bs):
            tensor_frames = self.memory.get(i) # [T, C, H, W] T number pre frames pre video
            fused_output.append(self.shift(tensor_frames, self.num_segments, shift_div=self.shift_div)) # [1, T * C, H, W]
        fused_output = torch.stack(fused_output_list)  # [N, T * C, H, W]
        assert list(fused_output.shape[:2]) == [bs, self.outc] 
        
        return self.fused(fused_output) # [N, C, H, W]

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, H, W]
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [1 , N * C, H, W]
        # restore the original dimension
        return out.view(1, n * c, h, w)
    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        self.memory.reset_all()

        return self
    
class TemporalPool(nn.Module):
        """Temporal pool module.

        3D max pooling layer.

        Args:
            net (nn.Module): Module to make temporal pool.
            num_segments (int): Number of frame segments.
        """

        def __init__(self, net, num_segments):
            super().__init__()
            self.net = net
            self.num_segments = num_segments
            self.max_pool3d = nn.MaxPool3d(
                kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        def forward(self, x):
            """Defines the computation performed at every call."""
            # [N, C, H, W]
            n, c, h, w = x.size()
            # [N // num_segments, C, num_segments, H, W]
            x = x.view(n // self.num_segments, self.num_segments, c, h,
                        w).transpose(1, 2)
            # [N // num_segmnets, C, num_segments // 2, H, W]
            x = self.max_pool3d(x)
            # [N // 2, C, H, W]
            x = x.transpose(1, 2).contiguous().view(n // 2, c, h, w)
            return self.net(x)
        

import torch
import torch.nn as nn
import math
class Conv_ln(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = LayerNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    

class CBAM(nn.Module): #self att
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        # channel attention
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #spatial attention
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #channel attention
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)                         
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1) #（N,C,1,1)
        channel_out=channel_out*x
        #spatial attention
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out

class CHSA(nn.Module): #CNN history self att
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        # channel attention
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #spatial attention
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def channel_att(self, x):
        #channel attention
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)                         
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1) #（N,C,1,1)
        # channel_out=channel_out*x
        return channel_out

    def spatial_att(self, channel_out):
        #spatial attention
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        # out=out*channel_out
        return out
    
    def forward(self,x): #bs,t,c,h,w
        bs,t,c,h,w = x.shape
        x = x.view(t,bs,c,h,w)
        channel_mask_list = []
        for x_feature in x:
            channel_mask_list.append(self.channel_att(x_feature)) #（N,C,1,1)
        channel_mask_max, _ = torch.max(torch.stack(channel_mask_list, dim=0), dim=0)

        spatial_mask_list = []
        for x_feature in x:
            channel_out = channel_mask_max*x_feature
            spatial_mask_list.append(self.spatial_att(channel_out))
        return out

class CCBA(nn.Module): #cross conv
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CCBA, self).__init__()
        # channel attention
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #spatial attention
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x,q):
        #channel attention
        maxout=self.max_pool(q)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(q)                         
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(q.size(0),q.size(1),1,1) #（N,C,1,1)
        channel_out_x=channel_out*x
        #spatial attention
        max_out,_=torch.max(channel_out*q,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out*q,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out_x
        return out



class CCS(nn.Module): #self att
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        pass

    def forward(self,x,q):
        pass

    def upsample_mask(self, flow, mask, rate): 
        """ Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination   
            rate= 2 4 8
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate*H, rate*W)
    
class TCBA(nn.Module): #temporal Convolutional Block Attention Module
    def __init__(self,in_channel, out_channel, number_history=4, epoch_train=0, reduction=16,kernel_size=7):
        super(TCBA, self).__init__()
        self.in_channel = in_channel

        fused_channel = in_channel
        # self.norm1 = nn.BatchNorm2d(in_channel)
        #self
        self.selfatt = CBAM(in_channel*number_history,reduction,kernel_size)
        self.cv = nn.Sequential(
            Conv(in_channel*number_history , in_channel*number_history//2 , 3, act=nn.ReLU(inplace=True)),#layer norm
            Conv(in_channel*number_history//2 , fused_channel , 3, act=nn.ReLU(inplace=True)),
        )

        #gru
        self.memory_gru = ConvGRU(fused_channel)
        
        self.cv_loss=Conv(fused_channel , 1 , 1, act=nn.Sigmoid())

        # self.norm2 = LayerNorm2d(in_channel)
        self.norm1 = LayerNorm2d(fused_channel)
        self.norm2 = nn.BatchNorm2d(fused_channel)
        #cross att
        self.crossatt = CCBA(fused_channel, reduction,kernel_size)
        #output 
        self.outproj = Conv(fused_channel, out_channel, 1, act=nn.ReLU())

        self.convs1 = nn.Conv2d(in_channel, 2*in_channel, kernel_size=1)
        self.convs2 = nn.Conv2d(2*in_channel, in_channel, kernel_size=1)  

        self.epoch_train = int(epoch_train)

    def forward(self,x):
        pre = x[0] #B,T,C,H,W
        cur = x[1]  #B,C,H,W

        if pre[0] is None:
            bs,c,_,_ = cur.shape
            return cur
        # if self.training:
        #     epoch = x[-1]["img_metas"][0]["epoch"]
        #     if epoch < self.epoch_train:
        #         return cur
        if x[-1]["img_metas"][0]["epoch"] < self.epoch_train:
            # return {
            #     "output":cur,
            #     "layer1_loss":None #cul loss
            #     }
            return cur, None
        
        bs,t,c,h,w = pre.shape
        assert [bs,c,h,w] == list(cur.shape), f"error: cur_shape:{cur.shape}, pre.shape:{pre.shape}"

        cur, inp = self.convs1(cur).chunk(2, 1) 
        inp = torch.tanh(inp) #B,64,H,W
        cur = torch.relu(cur) #B,64,H,W

        #self
        pre_att = self.selfatt(pre.view(bs,t*c,h,w)) #B,T*C,H,W
        pre_att = self.cv(pre_att) #B,C,H,W

        # memory fsued
        memory_cur_list = []
        for i in range(bs):
            memory_cur_list.append(self.memory_gru(pre[i][-1], cur[i]))
        memory_cur = torch.stack(memory_cur_list) #B,C,H,W

        #cross
        fused_ = self.crossatt(self.norm1(memory_cur),self.norm2(pre_att))

        fmaps_new = x[1] + self.convs2(torch.cat([inp,fused_], 1))

        return fmaps_new, None
        # return {
        #     "output":fused_+cur,
        #     "layer1_loss":self.cv_loss(pre_att)#cul loss
        # }
    

class MutiFreatureMemory(nn.Module):
    def __init__(self, name, number_history, interval, mode="fused"): #update
        super(MutiFreatureMemory, self).__init__()
        self.buffer = MutiFeatureBuffer(name, number_history, interval)
        self.mode = mode

    def forward(self,x): 
        if self.mode == "update":
            # if torch.is_tensor(x[1]): #init
            #     return list(self.buffer.memory_list)
            self.buffer.update(x[-1].clone().detach(), x[0]["img_metas"])
            return [None]
        elif self.mode == "fused":
            if self.buffer.is_empty():
                return [None]
            return self.buffer.get_all()

        
    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        self.buffer.reset_all()
        # print("reset buffer")
        return self
    
class LayerNorm2d(nn.Module):
    def __init__(self,
                 embed_dim,
                 eps=1e-6,
                    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.parameter.Parameter(torch.ones(embed_dim))
        self.bias = nn.parameter.Parameter(torch.zeros(embed_dim))
        self.eps = eps
        self.normalized_shape = (embed_dim, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)  # N,C,H,W

        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


from ultralytics.nn.modules.flow import CorrBlock, AlternateCorrBlock, initialize_flow,SepConvGRU,  BasicUpdateBlock, SmallNetUpdateBlock,  NetUpdateBlock, SmallUpdateBlock, warp_feature, ConvGRU
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from ultralytics.nn.modules.ops.modules import MSDeformAttn
from ultralytics.nn.modules.memory_buffer import from_coords_refpoint, from_refpoint_coords

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class DeformableTransformerLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn._reset_parameters()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads//2, dropout=dropout)
        # self.self_attn._reset_parameters()
        # self.dropout2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)

        # cross attention
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos).float(),
                                reference_points.float(),
                                src.float(), src_spatial_shapes, level_start_index)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # # ffn
        tgt = self.forward_ffn(tgt)

        return tgt
    
class FmapMaskBlock(nn.Module):
    def __init__(self, input_dim=64):
        super(FmapMaskBlock, self).__init__()
        self.cv = nn.Sequential(
            Conv_ln(input_dim , input_dim//2 , 3, act=nn.ReLU(inplace=True)),#layer norm
            Conv_ln(input_dim//2 , 1 , 3, act=nn.ReLU(inplace=True)),
        )

    def forward(self, fmaps, x):
        x = F.softmax(self.cv(x), dim=1)+0.5 #B 1 H W
        fmaps[0] = x*fmaps[0]
        for i, h in enumerate(fmaps[1:],start=1):
            _,_,height,weight = h.shape
            x = F.interpolate(x, size=(height, weight), mode='bilinear', align_corners=False)
            fmaps[i] = x*fmaps[i]

        return fmaps
    
class FReLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.depthwise_conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)



    
import numpy as np
import os
class VelocityNet(nn.Module): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        '''
        input_dim  最浅层特征的dim，
        hidden_dim 记忆隐藏层和输入层
        n_levels 多尺度特征图数量
        n_points 
        iter_max 迭代次数
        '''
        super(VelocityNet, self).__init__()
        input_dim = inchannle[0] 

        if not (input_dim//2 >= hidden_dim):
            print("************warning****************")
            print(f"input_dim//2 must bigger than hidden_dim, {inchannle},{hidden_dim}") 
            print("***********************************")

        self.inchannle = inchannle

        self.hidden_dim = hidden_dim # input_dim//2
        self.iter_max = iter_max
        self.n_levels = n_levels
        self.radius = radius
        self.stride = stride
        self.epoch_train = epoch_train
        self.method = method
        self.aux_loss = aux_loss
        self.motion_flow = motion_flow
        self.cor_planes = [n_levels * (2*radiu + 1)**2 for radiu in radius]

        
        # self.convs1 = nn.ModuleList([nn.Conv2d(inchannle[i], self.hidden_dim, 1) 
        #                             for i in range(n_levels)])
        
        # self.convs2 = nn.ModuleList([nn.Conv2d(self.hidden_dim+self.hidden_dim//2, inchannle[i], 1) 
        #                             for i in range(n_levels)])  # optional act=FReLU(c2)
        self.convs0 = nn.ModuleList([Conv(inchannle[i], inchannle[i], 1) 
                                    for i in range(n_levels)])
        
        self.convs1 = nn.ModuleList([Conv(inchannle[i]//2, self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([Conv(inchannle[i]//2 + self.hidden_dim + self.hidden_dim, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)

        # buffer
        self.buffer = FlowBuffer("MemoryAtten", number_feature=n_levels)
        
        
        # flow fused
        # 16: 32+16 -> P1
        # 8: 8+P1*2 -> P2
        # 32: 32+P1//2+P2//4 -> P3
        cor_plane = self.cor_planes[1]
        self.cor_plane = cor_plane

        self.flow_fused0 = nn.Conv2d(self.cor_planes[2]+ self.cor_planes[1], cor_plane, 3, padding=1)
        self.flow_fused1 = nn.Conv2d(self.cor_planes[0]+ cor_plane, cor_plane, 3, padding=1)
        self.flow_fused2 = nn.Conv2d(self.cor_planes[2] + cor_plane + cor_plane, cor_plane, 3, padding=1)
        
        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, cor_plane = cor_plane)

        self.plot = False
    
    def upflow(self, flow, up_rate=2, mode='bilinear'):
        new_size = (up_rate * flow.shape[2], up_rate * flow.shape[3])
        return  up_rate * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


    def forward(self,x):
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W 传入的尺度必须是从大到小
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    # out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            
            # 梯度分流,维度一致
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                # out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                fmap = self.convs1[i](fmap) #hidden

                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/save/train243_flow6_2.88/save_tansor/"
                    frame_number = img_metas[0]["frame_number"]
                    np.save(save_dir + f'fmaps_new_{i}_{frame_number}.npy', x[1:][i].cpu().numpy())
                    np.save(save_dir + f'inp_{i}_{frame_number}.npy', inp[i].cpu().numpy())

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            corr_32 = corr_fn_muti[lvl](coords1[lvl])
            flow_32 = coords1[lvl] - coords0[lvl]
            corr_32To16 = F.interpolate(corr_32, scale_factor=2, mode="bilinear", align_corners=True)
            
            # 1/16
            lvl = 1
            corr_16 = corr_fn_muti[lvl](coords1[lvl])
            flow_16 = coords1[lvl] - coords0[lvl]
            corr_16_fused = self.flow_fused0(torch.cat([corr_16, corr_32To16],dim=1))
            net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused, flow_16)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_16To8 = F.interpolate(corr_16_fused, scale_factor=2, mode="bilinear", align_corners=True)

            # 1/8
            lvl = 0
            corr_8 = corr_fn_muti[lvl](coords1[lvl])
            flow_8 = coords1[lvl] - coords0[lvl]
            corr_8_fused = self.flow_fused1(torch.cat([corr_8, corr_16To8],dim=1))
            net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused, flow_8)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_8To32 = F.interpolate(corr_8_fused, scale_factor=1/4, mode="bilinear", align_corners=True)
            corr_16To32 = F.interpolate(corr_16_fused, scale_factor=1/2, mode="bilinear", align_corners=True)

            # 1/32
            lvl = 2
            corr_32_fused = self.flow_fused2(torch.cat([corr_32, corr_8To32, corr_16To32],dim=1))
            net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused, flow_32)
            coords1[lvl] = coords1[lvl] + delta_flow

            #get coords1\net_8\net_16\net_32
            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(save_dir + f'flow{i}_{frame_number}.npy', (coords1[i]-coords0[i]).cpu().numpy())
                    np.save(save_dir + f'net{i}_{frame_number}.npy', net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(save_dir + f'fmaps_new_fused_{i}_{frame_number}.npy', fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}
    
    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        self.buffer.reset_all()
        # print("reset buffer")
        return self


class VelocityNet_baseline0(VelocityNet): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        '''
        input_dim  最浅层特征的dim，
        hidden_dim 记忆隐藏层和输入层
        n_levels 多尺度特征图数量
        n_points 
        iter_max 迭代次数
        '''
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        
        # self.convs1 = nn.ModuleList([nn.Conv2d(inchannle[i], self.hidden_dim, 1) 
        #                             for i in range(n_levels)])
        
        # self.convs2 = nn.ModuleList([nn.Conv2d(self.hidden_dim+self.hidden_dim//2, inchannle[i], 1) 
        #                             for i in range(n_levels)])  # optional act=FReLU(c2)
        self.convs0 = nn.ModuleList([nn.Conv2d(inchannle[i], inchannle[i], 1) 
                                    for i in range(n_levels)])
        
        self.convs1 = nn.ModuleList([nn.Conv2d(inchannle[i]//2, self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([nn.Conv2d(inchannle[i]//2 + self.hidden_dim + self.hidden_dim, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)

        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, cor_plane = self.cor_plane)

    
class VelocityNet_baseline1(VelocityNet): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        '''
        input_dim  最浅层特征的dim，
        hidden_dim 记忆隐藏层和输入层
        n_levels 多尺度特征图数量
        n_points 
        iter_max 迭代次数
        '''
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        # self.convs1 = nn.ModuleList([nn.Conv2d(inchannle[i], self.hidden_dim, 1) 
        #                             for i in range(n_levels)])
        
        # self.convs2 = nn.ModuleList([nn.Conv2d(self.hidden_dim+self.hidden_dim//2, inchannle[i], 1) 
        #                             for i in range(n_levels)])  # optional act=FReLU(c2)

        self.convs1 = nn.ModuleList([Conv(inchannle[i], self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([Conv(self.hidden_dim + self.hidden_dim//2, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)

        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim//2, hidden_dim=self.hidden_dim//2, cor_plane = self.cor_plane)

    
    def forward(self,x):
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W 传入的尺度必须是从大到小
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            
            # 梯度分流,维度一致
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/save/train243_flow6_2.88/save_tansor/"
                    frame_number = img_metas[0]["frame_number"]
                    np.save(save_dir + f'fmaps_new_{i}_{frame_number}.npy', x[1:][i].cpu().numpy())
                    np.save(save_dir + f'inp_{i}_{frame_number}.npy', inp[i].cpu().numpy())

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            corr_32 = corr_fn_muti[lvl](coords1[lvl])
            flow_32 = coords1[lvl] - coords0[lvl]
            corr_32To16 = F.interpolate(corr_32, scale_factor=2, mode="bilinear", align_corners=True)
            
            # 1/16
            lvl = 1
            corr_16 = corr_fn_muti[lvl](coords1[lvl])
            flow_16 = coords1[lvl] - coords0[lvl]
            corr_16_fused = self.flow_fused0(torch.cat([corr_16, corr_32To16],dim=1))
            net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused, flow_16)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_16To8 = F.interpolate(corr_16_fused, scale_factor=2, mode="bilinear", align_corners=True)

            # 1/8
            lvl = 0
            corr_8 = corr_fn_muti[lvl](coords1[lvl])
            flow_8 = coords1[lvl] - coords0[lvl]
            corr_8_fused = self.flow_fused1(torch.cat([corr_8, corr_16To8],dim=1))
            net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused, flow_8)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_8To32 = F.interpolate(corr_8_fused, scale_factor=1/4, mode="bilinear", align_corners=True)
            corr_16To32 = F.interpolate(corr_16_fused, scale_factor=1/2, mode="bilinear", align_corners=True)

            # 1/32
            lvl = 2
            corr_32_fused = self.flow_fused2(torch.cat([corr_32, corr_8To32, corr_16To32],dim=1))
            net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused, flow_32)
            coords1[lvl] = coords1[lvl] + delta_flow

            #get coords1\net_8\net_16\net_32
            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(save_dir + f'flow{i}_{frame_number}.npy', (coords1[i]-coords0[i]).cpu().numpy())
                    np.save(save_dir + f'net{i}_{frame_number}.npy', net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(save_dir + f'fmaps_new_fused_{i}_{frame_number}.npy', fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}
    
class VelocityNet_baseline2(VelocityNet_baseline1): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        '''
        input_dim  最浅层特征的dim，
        hidden_dim 记忆隐藏层和输入层
        n_levels 多尺度特征图数量
        n_points 
        iter_max 迭代次数
        '''
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        
        self.convs1 = nn.ModuleList([nn.Conv2d(inchannle[i], self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([nn.Conv2d(self.hidden_dim + self.hidden_dim//2, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)

        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim//2, hidden_dim=self.hidden_dim//2, cor_plane = self.cor_plane)


class ADown(nn.Module):
    def __init__(self, cx, cy, cout):
        super().__init__()
        self.c = cout // 2
        self.cv1 = flow_conv(cx // 2, self.c, 3, 2, 1)
        self.cv2 = flow_conv(cx // 2 + cy, self.c, 1, 1, 0)

    def forward(self, x, y):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(torch.cat((x2, y), 1))
        return torch.cat((x1, x2), 1)

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

        self.cv1 = flow_conv(half_c_4, self.c, 3, 2, 1)
        self.cv2 = flow_conv(half_c_4+half_c_2, self.c, 3, 2, 1)

        self.cv3 = flow_conv(self.c + half_c, self.c, 1, 1, 0)
        self.cv4 = flow_conv(half_c_1 + half_c_4_1 + half_c_2_1, self.c, 1, 1, 0)

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
        self.cv2 = flow_conv(half_x_2 + cy, self.c, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, y):
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = self.upsample(x2)
        x2 = self.cv2(torch.cat((x2, y), 1))
        return torch.cat((x1, x2), 1)



class MSTF(nn.Module): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        '''
        input_dim  dim of the backbone features.
        hidden_dim Memorizing hidden and input layers
        n_levels Number of multi-scale feature maps
        epoch_train start fused 
        iter_max Number of iterations
        '''
        super(MSTF, self).__init__()
        input_dim = inchannle[0] 

        if not (input_dim//2 >= hidden_dim):
            print("************warning****************")
            print(f"input_dim//2 need bigger than hidden_dim, {inchannle},{hidden_dim}") 
            print("***********************************")

        self.inchannle = inchannle

        self.hidden_dim = hidden_dim # input_dim//2
        self.iter_max = iter_max
        self.n_levels = n_levels
        self.radius = radius
        self.stride = stride
        self.epoch_train = epoch_train
        self.method = method
        self.aux_loss = aux_loss
        self.motion_flow = motion_flow
        self.cor_planes = [n_levels * (2*radiu + 1)**2 for radiu in radius]


        self.convs1 = nn.ModuleList([flow_conv(inchannle[i], self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([flow_conv(self.hidden_dim + self.hidden_dim//2, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)
        
        # buffer
        self.buffer = FlowBuffer("MemoryAtten", number_feature=n_levels)

        cor_plane = self.cor_planes[1]
        self.cor_plane = cor_plane
        self.cor_plane = 2*(self.cor_plane//2) #Guaranteed to be even.

        self.flow_fused0 = FlowUp(self.cor_planes[2], self.cor_planes[1], self.cor_plane)
        self.flow_fused1 = FlowUp(self.cor_plane,self.cor_planes[0], self.cor_plane)
        
        self.flow_fused2 = FlowDown(self.cor_plane ,self.cor_plane, self.cor_planes[2], self.cor_plane)
        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim//2, hidden_dim=self.hidden_dim//2, cor_plane = self.cor_plane)

        self.plot = False
        

    def forward(self,x):
        # self.plot = True
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W The incoming scale must be from large to small
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            
            # Gradient triage, dimensional consistency
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "xxx/path/to//save_tansor/"
                    video_name = img_metas[0]["video_name"]
                    save_dir = os.path.join(save_dir, video_name)
                    
                    feature_new_dir = os.path.join(save_dir, "feature_new")
                    feature_fused_dir = os.path.join(save_dir, "feature_fused")
                    flow_dir = os.path.join(save_dir, "flows")
                    net_dir = os.path.join(save_dir, "nets")
                    os.makedirs(feature_new_dir, exist_ok=True)
                    os.makedirs(feature_fused_dir, exist_ok=True)
                    os.makedirs(flow_dir, exist_ok=True)
                    os.makedirs(net_dir, exist_ok=True)

                    frame_number = img_metas[0]["frame_number"]
                    np.save(os.path.join(feature_new_dir, f'level_{i}_{frame_number}.npy'), x[1:][i].cpu().numpy())


            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            corr_32 = corr_fn_muti[lvl](coords1[lvl])
            flow_32 = coords1[lvl] - coords0[lvl]
            
            # 1/16
            lvl = 1
            corr_16 = corr_fn_muti[lvl](coords1[lvl])
            flow_16 = coords1[lvl] - coords0[lvl]
            corr_16_fused = self.flow_fused0(corr_32, corr_16)
            net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused, flow_16)
            coords1[lvl] = coords1[lvl] + delta_flow

            # 1/8
            lvl = 0
            corr_8 = corr_fn_muti[lvl](coords1[lvl])
            flow_8 = coords1[lvl] - coords0[lvl]
            corr_8_fused = self.flow_fused1(corr_16_fused, corr_8)
            net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused, flow_8)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_8To32 = F.interpolate(corr_8_fused, scale_factor=1/4, mode="bilinear", align_corners=True)
            corr_16To32 = F.interpolate(corr_16_fused, scale_factor=1/2, mode="bilinear", align_corners=True)

            # 1/32
            lvl = 2
            corr_32_fused = self.flow_fused2(corr_8_fused, corr_16_fused, corr_32)
            net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused, flow_32)
            coords1[lvl] = coords1[lvl] + delta_flow

            #get coords1\net_8\net_16\net_32
            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(os.path.join(flow_dir, f'level_{i}_{frame_number}.npy'), (coords1[i]-coords0[i]).cpu().numpy())
                    print(torch.sum(coords1[i]-coords0[i]))
                    np.save(os.path.join(net_dir, f'level_{i}_{frame_number}.npy'), net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(os.path.join(feature_fused_dir,  f'level_{i}_{frame_number}.npy'), fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}
            

class VelocityNet_baseline3_split_dim(MSTF): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        
        self.dim1 = inchannle[0]-self.hidden_dim
        self.dim2 = self.hidden_dim

        self.convs1 = nn.ModuleList([flow_conv(inchannle[i], inchannle[0], 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([flow_conv(self.dim1 + self.dim2 + self.dim2, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)
        
        # self.flow_fused0 = flow_conv(self.cor_planes[2]+ self.cor_planes[1], self.cor_plane, 3, padding=1)
        # self.flow_fused1 = flow_conv(self.cor_planes[0]+ self.cor_plane, self.cor_plane, 3, padding=1)
        # self.flow_fused2 = flow_conv(self.cor_planes[2] + self.cor_plane + self.cor_plane, self.cor_plane, 3, padding=1)
        self.cor_plane = 2*(self.cor_plane//2) #保证是偶数
        self.flow_fused0 = FlowUp(self.cor_planes[2], self.cor_planes[1], self.cor_plane)
        self.flow_fused1 = FlowUp(self.cor_plane,self.cor_planes[0], self.cor_plane)
        
        self.flow_fused2 = FlowDown(self.cor_plane ,self.cor_plane, self.cor_planes[2], self.cor_plane)
        
        self.update_block = SmallNetUpdateBlock(input_dim=self.dim2, hidden_dim=self.dim2, cor_plane = self.cor_plane)

        

    def forward(self,x):
        # self.plot = True
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W 传入的尺度必须是从大到小
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).split((self.dim1, self.dim2), 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            
            # 梯度分流,维度一致
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).split((self.dim1, self.dim2), 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/UAVTOD_exper/baseline/baseline3/train94_36.2/save_tansor/"
                    video_name = img_metas[0]["video_name"]
                    save_dir = os.path.join(save_dir, video_name)
                    
                    feature_new_dir = os.path.join(save_dir, "feature_new")
                    feature_fused_dir = os.path.join(save_dir, "feature_fused")
                    flow_dir = os.path.join(save_dir, "flows")
                    net_dir = os.path.join(save_dir, "nets")
                    os.makedirs(feature_new_dir, exist_ok=True)
                    os.makedirs(feature_fused_dir, exist_ok=True)
                    os.makedirs(flow_dir, exist_ok=True)
                    os.makedirs(net_dir, exist_ok=True)

                    frame_number = img_metas[0]["frame_number"]
                    np.save(os.path.join(feature_new_dir, f'level_{i}_{frame_number}.npy'), x[1:][i].cpu().numpy())


            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            corr_32 = corr_fn_muti[lvl](coords1[lvl])
            flow_32 = coords1[lvl] - coords0[lvl]
            
            # 1/16
            lvl = 1
            corr_16 = corr_fn_muti[lvl](coords1[lvl])
            flow_16 = coords1[lvl] - coords0[lvl]
            corr_16_fused = self.flow_fused0(corr_32, corr_16)
            net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused, flow_16)
            coords1[lvl] = coords1[lvl] + delta_flow

            # 1/8
            lvl = 0
            corr_8 = corr_fn_muti[lvl](coords1[lvl])
            flow_8 = coords1[lvl] - coords0[lvl]
            corr_8_fused = self.flow_fused1(corr_16_fused, corr_8)
            net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused, flow_8)
            coords1[lvl] = coords1[lvl] + delta_flow
            corr_8To32 = F.interpolate(corr_8_fused, scale_factor=1/4, mode="bilinear", align_corners=True)
            corr_16To32 = F.interpolate(corr_16_fused, scale_factor=1/2, mode="bilinear", align_corners=True)

            # 1/32
            lvl = 2
            corr_32_fused = self.flow_fused2(corr_8_fused, corr_16_fused, corr_32)
            net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused, flow_32)
            coords1[lvl] = coords1[lvl] + delta_flow

            #get coords1\net_8\net_16\net_32
            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(os.path.join(flow_dir, f'level_{i}_{frame_number}.npy'), (coords1[i]-coords0[i]).cpu().numpy())
                    print(torch.sum(coords1[i]-coords0[i]))
                    np.save(os.path.join(net_dir, f'level_{i}_{frame_number}.npy'), net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(os.path.join(feature_fused_dir,  f'level_{i}_{frame_number}.npy'), fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}
            

class VelocityNet_baseline3_singal_flow(VelocityNet_baseline1): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[1,1,1], radius=[3,3,3], level_use=[0,1,2], n_levels=3, iter_max=2, method = "method1", motion_flow = True, aux_loss = False):
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        
        self.convs1 = nn.ModuleList([flow_conv(inchannle[i], self.hidden_dim, 1) 
                                    for i in range(n_levels)])
        
        self.convs2 = nn.ModuleList([flow_conv(self.hidden_dim + self.hidden_dim//2, inchannle[i], 1) 
                                    for i in range(n_levels)])  # optional act=FReLU(c2)
        
        # self.flow_fused0 = flow_conv(self.cor_planes[2]+ self.cor_planes[1], self.cor_plane, 3, padding=1)
        # self.flow_fused1 = flow_conv(self.cor_planes[0]+ self.cor_plane, self.cor_plane, 3, padding=1)
        # self.flow_fused2 = flow_conv(self.cor_planes[2] + self.cor_plane + self.cor_plane, self.cor_plane, 3, padding=1)
        self.level_use = level_use
        # self.cor_plane = 2*(self.cor_plane//2) 
        self.flow_fused0 = FlowUp(self.cor_planes[2], self.cor_planes[1], self.cor_plane)
        self.flow_fused1 = FlowUp(self.cor_plane,self.cor_planes[0], self.cor_plane)
        
        self.flow_fused2 = FlowDown(self.cor_plane ,self.cor_plane, self.cor_planes[2], self.cor_plane)
        
        self.update_block = SmallNetUpdateBlock(input_dim=self.hidden_dim//2, hidden_dim=self.hidden_dim//2, cor_plane = self.cor_plane)

    def forward(self,x):
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            
            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/save/train243_flow6_2.88/save_tansor/"
                    frame_number = img_metas[0]["frame_number"]
                    np.save(save_dir + f'fmaps_new_{i}_{frame_number}.npy', x[1:][i].cpu().numpy())
                    np.save(save_dir + f'inp_{i}_{frame_number}.npy', inp[i].cpu().numpy())

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            # 1/32
            lvl = 2
            if lvl in self.level_use:
                corr_32 = corr_fn_muti[lvl](coords1[lvl])
                flow_32 = coords1[lvl] - coords0[lvl]
            
            # 1/16
            lvl = 1
            if lvl in self.level_use:
                corr_16 = corr_fn_muti[lvl](coords1[lvl])
                flow_16 = coords1[lvl] - coords0[lvl]
                net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16, flow_16)
                coords1[lvl] = coords1[lvl] + delta_flow
            else:
                net_16 = inp[lvl]

            # 1/8
            lvl = 0
            if lvl in self.level_use:
                corr_8 = corr_fn_muti[lvl](coords1[lvl])
                flow_8 = coords1[lvl] - coords0[lvl]
                net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8, flow_8)
                coords1[lvl] = coords1[lvl] + delta_flow
            else:
                net_8 = inp[lvl]

            # 1/32
            lvl = 2
            if lvl in self.level_use:
                net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32, flow_32)
                coords1[lvl] = coords1[lvl] + delta_flow
            else:
                net_32 = inp[lvl]

            #get coords1\net_8\net_16\net_32
            net = [net_8, net_16, net_32]
            self.buffer.update_coords(coords1)
            self.buffer.update_net(net)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(save_dir + f'flow{i}_{frame_number}.npy', (coords1[i]-coords0[i]).cpu().numpy())
                    np.save(save_dir + f'net{i}_{frame_number}.npy', net[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],net[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(save_dir + f'fmaps_new_fused_{i}_{frame_number}.npy', fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}
            
class VelocityNet_baseline3_iter(MSTF): #
    def __init__(self, inchannle, hidden_dim=64, epoch_train=22, stride=[2,2,2], radius=[3,3,3], iter_max=2, n_levels=3, method = "method1", motion_flow = True, aux_loss = False):
        super().__init__(inchannle=inchannle, hidden_dim=hidden_dim, epoch_train=epoch_train, 
                                          stride=stride, radius=radius, n_levels=n_levels, iter_max=iter_max, 
                                          method=method, motion_flow=motion_flow, aux_loss=aux_loss)
        self.iter_max = iter_max

    def forward(self,x):
        with autocast(dtype=torch.float32):
            img_metas = x[0]["img_metas"]
            fmaps_new = x[1:]  #3,B,C,H,W 
            
            if fmaps_new[0].device.type == "cpu" or (img_metas[0]["epoch"] < self.epoch_train and self.training) or self.epoch_train == 100:
                if self.epoch_train == 100:
                    return x[1:], {"k":0.1, "loss":torch.tensor(0.0)}
                outs = []
                for i in range(self.n_levels):
                    out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                    # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                    # fmap = self.convs1[i](fmap) #hidden
                    out = self.convs2[i](torch.cat([out,fmap,torch.relu(fmap)], 1))
                    outs.append(out)
                return outs, {"k":0.1, "loss":torch.tensor(0.0)}

            out_list = []
            fmaps_new = []
            inp = []
            for i in range(self.n_levels):
                out, fmap = self.convs1[i](x[1:][i]).chunk(2, 1)
                # out, fmap = self.convs0[i](x[1:][i]).chunk(2, 1) #128,128
                # fmap = self.convs1[i](fmap) #hidden
                out_list.append(out)
                fmaps_new.append(fmap)
                inp.append(torch.relu(fmap))
                if not self.training and self.plot:
                    save_dir = "/data/jiahaoguo/ultralytics/runs/save/train243_flow6_2.88/save_tansor/"
                    frame_number = img_metas[0]["frame_number"]
                    np.save(save_dir + f'fmaps_new_{i}_{frame_number}.npy', x[1:][i].cpu().numpy())
                    np.save(save_dir + f'inp_{i}_{frame_number}.npy', inp[i].cpu().numpy())

            src_flatten_new, spatial_shapes, level_start_index = self.buffer.flatten(fmaps_new, True)

            result_first_frame, fmaps_old, net_old, coords0, coords1, topk_bbox = self.buffer.update_memory(src_flatten_new, img_metas, spatial_shapes, level_start_index)

            corr_fn_muti = []
            for lvl in range(self.n_levels):
                corr_fn_muti.append(AlternateCorrBlock(fmaps_new[lvl], fmaps_old, self.n_levels, self.radius[lvl], self.stride))

            for _ in range(self.iter_max):

                for lvl in range(self.n_levels):
                    coords1[lvl] = coords1[lvl].detach()

                # 1/32
                lvl = 2
                corr_32 = corr_fn_muti[lvl](coords1[lvl])
                flow_32 = coords1[lvl] - coords0[lvl]
                
                # 1/16
                lvl = 1
                corr_16 = corr_fn_muti[lvl](coords1[lvl])
                flow_16 = coords1[lvl] - coords0[lvl]
                corr_16_fused = self.flow_fused0(corr_32, corr_16)
                net_16, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_16_fused.contiguous(), flow_16)
                coords1[lvl] = coords1[lvl] + delta_flow

                # 1/8
                lvl = 0
                corr_8 = corr_fn_muti[lvl](coords1[lvl])
                flow_8 = coords1[lvl] - coords0[lvl]
                corr_8_fused = self.flow_fused1(corr_16_fused, corr_8)
                net_8, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_8_fused.contiguous(), flow_8)
                coords1[lvl] = coords1[lvl] + delta_flow

                # 1/32
                lvl = 2
                corr_32_fused = self.flow_fused2(corr_8_fused, corr_16_fused, corr_32)
                net_32, _, delta_flow = self.update_block(net_old[lvl], inp[lvl], corr_32_fused.contiguous(), flow_32)
                coords1[lvl] = coords1[lvl] + delta_flow

                inp = [net_8.contiguous(), net_16.contiguous(), net_32.contiguous()]

            #get coords1\net_8\net_16\net_32
            self.buffer.update_coords(coords1)
            self.buffer.update_net(inp)


            for i in range(self.n_levels):
                if not self.training and self.plot:
                    np.save(save_dir + f'flow{i}_{frame_number}.npy', (coords1[i]-coords0[i]).cpu().numpy())
                    np.save(save_dir + f'net{i}_{frame_number}.npy', inp[i].cpu().numpy())
                fmaps_new[i] = self.convs2[i](torch.cat([out_list[i],fmaps_new[i],inp[i]], 1)) + x[1:][i]
                if not self.training and self.plot:
                    np.save(save_dir + f'fmaps_new_fused_{i}_{frame_number}.npy', fmaps_new[i].cpu().numpy())

            return fmaps_new, {"k":0.1, "loss":torch.tensor(0.0).to(net_32.device)}

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        # assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

