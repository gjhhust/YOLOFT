import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, constant_

def _pair(x):
    if isinstance(x, int):
        return (x, x)
    return x

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

# class _DeformConv2d(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         deformable_groups=1,
#         bias=False,
#     ):
#         """
#         Modulated deformable convolution from :paper:`deformconv2`.

#         Arguments are similar to :class:`Conv2D`. Extra arguments:

#         Args:
#             deformable_groups (int): number of groups used in deformable convolution.
#             norm (nn.Module, optional): a normalization layer
#             activation (callable(Tensor) -> Tensor): a callable activation function
#         """
#         super(_DeformConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.with_bias = bias
        
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=bias) 

#         self.p_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#         nn.init.constant_(self.p_conv.weight, 0) #权重初始化为0

#         self.deformconv = DeformConv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#             groups=groups,
#             deform_groups=deformable_groups,
#             bias=bias
#         )

#     def forward(self, x):
#         # if x.numel() == 0:
#         #     output_shape = [
#         #         (i + 2 * p - (di * (k - 1) + 1)) // s + 1
#         #         for i, p, di, k, s in zip(
#         #             x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
#         #         )
#         #     ]
#         #     output_shape = [x.shape[0], self.out_channels] + output_shape
#         #     return _NewEmptyTensorOp.apply(x, output_shape)
#         offset = self.p_conv(x) # (b,2N,h,w) 学习到的偏移量 2N表示在x轴方向的偏移和在y轴方向的偏移
#         x = self.deformconv(x, offset)
#         return x
    
# import torchvision
# from mmengine.utils import digit_version
# from torchvision.ops import deform_conv2d as tv_deform_conv2d

# class DeformConv2dPack_MLU(DeformConv2d):
#     """This class is the DCN implementation of the MLU device. The MLU
#     backend support of the operator has been implemented in torchvision.
#     The mmcv registration mechanism is used for multiplexing here. The
#     torchvision implementation of DCN is called.
#     Args:
#         in_channels (int): Same as nn.Conv2d.
#         out_channels (int): Same as nn.Conv2d.
#         kernel_size (int or tuple[int]): Same as nn.Conv2d.
#         stride (int): Same as nn.Conv2d, while tuple is not supported.
#         padding (int): Same as nn.Conv2d, while tuple is not supported.
#         dilation (int): Same as nn.Conv2d, while tuple is not supported.
#         groups (int): Same as nn.Conv2d.
#         bias (bool or str): If specified as `auto`, it will be decided by
#             the norm_cfg. Bias will be set as True if norm_cfg is None,
#             otherwise False.
#         im2col_step (int): Number of samples processed by
#             im2col_cuda_kernel per call. It will work when ``batch_size``
#             > ``im2col_step``, but ``batch_size`` must be divisible by
#             ``im2col_step``. Default: 32. `New in version 1.7.2.
#             Currently not supported on MLU devices.`
#     """

#     def __init__(self, *args, **kwargs):
#         assert digit_version(torchvision.__version__) >= digit_version(
#             '0.10.0a0'), 'the version of torchvision should be >= 0.10.0'
#         super().__init__(*args, **kwargs)

#         self.conv_offset = nn.Conv2d(
#             self.in_channels,
#             self.deform_groups * 2 * self.kernel_size[0] *
#             self.kernel_size[1],
#             kernel_size=self.kernel_size,
#             stride=_pair(self.stride),
#             padding=_pair(self.padding),
#             dilation=_pair(self.dilation),
#             bias=True)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.conv_offset.weight.data.zero_()
#         self.conv_offset.bias.data.zero_()

#     def forward(self, x: Tensor) -> Tensor:  # type: ignore
#         cur_im2col_step = min(self.im2col_step, x.size(0))
#         assert (x.size(0) % cur_im2col_step
#                 ) == 0, 'batch size must be divisible by im2col_step'
#         offset = self.conv_offset(x)
#         x = x.type_as(offset)
#         weight = self.weight.type_as(x)
#         return tv_deform_conv2d(x, offset, weight, None, self.stride,
#                                 self.padding, self.dilation)

import torchvision
import math
from typing import Optional, Tuple, Union
from torchvision.ops import deform_conv2d as tv_deform_conv2d
from torch.nn.modules.utils import _pair, _single
class ModulatedDeformConv2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: Union[bool, str] = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class ModulatedDeformConv2dPack_MLU(ModulatedDeformConv2d):
    """This class is the DCNv2 implementation of the MLU device.

    The MLU backend support of the operator has been implemented
    in torchvision. The mmcv registration mechanism is used for
    multiplexing here. The torchvision implementation of DCNv2 is called.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias will be set as True if norm_cfg is None,
            otherwise False.
    """

    def __init__(self, *args, **kwargs):
        print("torchvision.__version__ >= 0.10.0a0" + '\nthe version of torchvision should be >= 0.10.0') 
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        # self.conv_out = nn.Conv2d(
        #     self.in_channels,
        #     self.out_channels,
        #     kernel_size=self.kernel_size,
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = x.type_as(offset)
        weight = self.weight.type_as(x)
        mask = mask.type_as(x)
        return tv_deform_conv2d(
            x,
            offset,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask)
        # return self.conv_out(x)
        
from thop.vision.calc_func import calculate_conv2d_flops

def count_modulated_deform_conv2d_pack_mlu(m, x, y):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.weight.shape),
        groups = m.groups,
        bias = m.bias
    )
    
    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.conv_offset.weight.shape),
        groups = m.conv_offset.groups,
        bias = m.conv_offset.bias
    )
    
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=True
    ):
        super().__init__()
        assert in_channels % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_channels, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_channels, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# 测试代码
if __name__ == "__main__":
    # 定义输入张量的形状
    input_shape = [2, 256, 224, 224]
    # 创建输入张量
    input_tensor = torch.randn(*input_shape)

    # 初始化 _ModulatedDeformConv 模块
    in_channels = 256
    out_channels = 256  # 可以根据需要修改输出通道数
    kernel_size = 3
    stride = 1
    deformable_groups = 1
    bias = False
    model = DeformableConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=autopad(kernel_size, None, 1),
        bias=bias
    ).cuda()

    # 前向传播
    output = model(input_tensor.cuda())

    # 打印输出
    print("Output:", output)
    print("Output shape:", output.shape)