# import numpy as np
# import torch
# import torch.nn as nn
# from mmcv.cnn import ConvModule
# from mmengine.model import constant_init,caffe2_xavier_init
# from ultralytics.nn.builder import CONVERTERS

# @CONVERTERS.register_module()
# class ContextAggregation(nn.Module):
#     """
#     Context Aggregation Block.

#     Args:
#         in_channels (int): Number of input channels.
#         reduction (int, optional): Channel reduction ratio. Default: 1.
#         conv_cfg (dict or None, optional): Config dict for the convolution
#             layer. Default: None.
#     """

#     def __init__(self, in_channels, reduction=1):
#         super(ContextAggregation, self).__init__()
#         self.in_channels = in_channels
#         self.reduction = reduction
#         self.inter_channels = max(in_channels // reduction, 1)

#         conv_params = dict(kernel_size=1, act_cfg=None)

#         self.a = ConvModule(in_channels, 1, **conv_params)
#         self.k = ConvModule(in_channels, 1, **conv_params)
#         self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
#         self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

#         self.init_weights()

#     def init_weights(self):
#         for m in (self.a, self.k, self.v):
#             caffe2_xavier_init(m.conv)
#         constant_init(self.m.conv, 0)

#     def forward(self, x):
#         #n, c = x.size(0)
#         n = x.size(0)
#         c = self.inter_channels
#         #n, nH, nW, c = x.shape

#         # a: [N, 1, H, W]
#         # import pdb;pdb.set_trace()
#         a = self.a(x).sigmoid()

#         # k: [N, 1, HW, 1]
#         k = self.k(x).view(n, 1, -1, 1).softmax(2)

#         # v: [N, 1, C, HW]
#         v = self.v(x).view(n, 1, c, -1)

#         # y: [N, C, 1, 1]
#         y = torch.matmul(v, k).view(n, c, 1, 1)
#         y = self.m(y) * a

#         return x + y
