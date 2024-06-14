import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models.feature_extraction import create_feature_extractor
# from .mobilenetv4 import MobileNetV4
from .mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large

class ScaleBackbone(nn.Module):
    def __init__(self, model_name='resnet50', out_channle=[1,1,1]):
        super().__init__() 
        if model_name == 'resnet50':
            self.backbone = ResNetBackbone(pretrained=True, model_name='resnet50')
        elif "efficientnet" in model_name:
            self.backbone = EfficientNetV2Backbone(model_name=model_name, pretrained=True)
        elif "MobileNetV4" in model_name:
            self.backbone = MobileNetV4(model=model_name)
        elif "MobileNetV3" in model_name:
            self.backbone = MobileNetV3(model_name=model_name)
        else:
            ValueError(f"model_name: [{model_name}] not exist")

        self.out_channle = out_channle
        assert out_channle == self.backbone.out_channle, f"backbone truth outchannle must qual {out_channle} write by you"

    def forward(self, x):
        return self.backbone(x)

import requests, os
def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file from {url}")

def load_model_from_local(model_class, local_path):
    model = model_class()
    model.load_state_dict(torch.load(local_path))
    return model

class MobileNetV3(nn.Module):
    def __init__(self, pretrained=True, model_name='MobileNetV3_Small'):
        super(MobileNetV3, self).__init__()
        # 加载预训练模型
        model_urls = {
            'MobileNetV3_Small': 'https://raw.githubusercontent.com/xiaolai-sqlai/mobilenetv3/master/450_act3_mobilenetv3_small.pth',
            'MobileNetV3_Large': 'https://raw.githubusercontent.com/xiaolai-sqlai/mobilenetv3/master/450_act3_mobilenetv3_large.pth',
        }
        if model_name == "MobileNetV3_Small":
            net = MobileNetV3_Small()
            self.return_indices = [2, 7, 10]
            self.out_channle = [24, 48, 96]
        elif model_name == "MobileNetV3_Large":
            net = MobileNetV3_Large() 
            self.return_indices = [5, 11, 14] 
            self.out_channle = [40, 112, 160]
        if pretrained:
            if not os.path.exists(os.path.basename(model_urls[model_name])):
                print(f"download: {os.path.basename(model_urls[model_name])} from {model_urls[model_name]} ...")
                download_file(model_urls[model_name], os.path.basename(model_urls[model_name]))
            net.load_state_dict(torch.load(os.path.basename(model_urls[model_name]), map_location='cpu'))

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.hs1 = net.hs1
        self.bneck = net.bneck
        

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))

        outputs = []
        for i, block in enumerate(self.bneck):
            x = block(x)
            # print(f"idx: {i}, shape: {x.shape}")
            if i in self.return_indices:
                outputs.append(x)
        return outputs 
    

class ResNetBackbone(nn.Module):
    out_channle = [512, 1024, 2048] #rs50
    def __init__(self, pretrained=True, model_name='resnet50'):
        super(ResNetBackbone, self).__init__()
        # 加载预训练模型
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }
        resnet_model = models.resnet.__dict__[model_name](pretrained=False)
        if pretrained:
            resnet_model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls[model_name], progress=True, map_location=torch.device('cpu')))

        # 特征提取层
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool

        # 提取不同stage的特征图
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def forward(self, x):
        # 初始卷积层和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage 1
        feat1 = self.layer1(x)
        
        # Stage 2
        feat2 = self.layer2(feat1)
        
        # Stage 3
        feat3 = self.layer3(feat2)
        
        # Stage 4
        feat4 = self.layer4(feat3)

        return feat2, feat3, feat4
    
class EfficientNetV2Backbone(nn.Module):
    def __init__(self, model_name='efficientnet_v2_s', pretrained=True):
        super(EfficientNetV2Backbone, self).__init__()

        # 根据模型名称选择模型和权重
        if model_name == 'efficientnet_v2_s':
            if pretrained:
                weights = EfficientNet_V2_S_Weights.DEFAULT
                self.model = efficientnet_v2_s(weights=weights)
            else:
                self.model = efficientnet_v2_s(weights=None)
            self.return_indices = [3, 5, 6]  
            self.out_channle = [64, 160, 256]
        elif model_name == 'efficientnet_v2_m':
            if pretrained:
                weights = EfficientNet_V2_M_Weights.DEFAULT
                self.model = efficientnet_v2_m(weights=weights)
            else:
                self.model = efficientnet_v2_m(weights=None)
            self.return_indices = [3, 5, 7]  
            self.out_channle = [80, 176, 512]
        elif model_name == 'efficientnet_v2_l':
            if pretrained:
                weights = EfficientNet_V2_L_Weights.DEFAULT
                self.model = efficientnet_v2_l(weights=weights)
            else:
                self.model = efficientnet_v2_l(weights=None)
            self.return_indices = [3, 5, 7]  
            self.out_channle = [96, 224, 640]
        elif model_name == "efficientnet_b0":
            if pretrained:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.model = efficientnet_b0(weights=weights)
            else:
                self.model = efficientnet_b0(weights=None)
            self.return_indices = [3, 5, 7]  
            self.out_channle = [40, 112, 320]
        elif model_name == "efficientnet_b1":
            if pretrained:
                weights = EfficientNet_B1_Weights.DEFAULT
                self.model = efficientnet_b1(weights=weights)
            else:
                self.model = efficientnet_b1(weights=None)
            self.return_indices = [3, 5, 7]  
            self.out_channle = [40, 112, 320]    

        # 初始化特征提取层
        self.blocks = nn.ModuleList()
        for i, stage in enumerate(self.model.features):
            if i > self.return_indices[-1]:
                break
            self.blocks.append(stage)

    def forward(self, x):
        features = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.return_indices:
                # print(f"idx: {idx}, shape: {x.shape}")
                features.append(x)

        return features  # 返回选定层的特征图


if __name__ == '__main__':
    # 使用预训练的ResNet50作为特征提取器
    feature_extractor = MobileNetV3(model_name = "MobileNetV3_Large")
    # 设置模型为评估模式
    # feature_extractor.eval()

    # 生成一个随机的图像张量，大小为 [1, 3, 1024, 1024]，模拟一个批量大小为1的输入
    # 这里 1 是批量大小, 3 是颜色通道数, 1024x1024 是图像尺寸
    random_image = torch.rand(2, 3, 1024, 1024)
    result = feature_extractor(random_image)
    for res in result:
        print(f"shape: [{res.shape}]")
