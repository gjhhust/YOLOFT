from thop import profile
from ultralytics import RTDETR
from ultralytics import YOLO
import torch
from ultralytics.models import YOLOFT
from ultralytics.nn.modules.block import Homograph
from ultralytics.nn.modules.utils import transformer,homo_align
from ultralytics.data.dataset import MOVEHomoDETDataset, MOVEHomoDETDataset_stream
from torch.utils.data import DataLoader
from ultralytics.cfg import cfg2dict
import numpy as np
import cv2,os
import imageio
import torch.nn.functional as F


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return

class DictWrapper:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)
    
import torch
# Load a COCO-pretrained RT-DETR-l model


# params
cfg = cfg2dict("config/train/orige_stream.yaml") #dict
cfg = DictWrapper(cfg)
data = cfg2dict("config/XS-VID.yaml")
model = YOLOFT('YoloftS_basline3_convnoany_epoch18s_epoch18_32.6(small AP best).pt')

# load data
dataset = MOVEHomoDETDataset_stream(
            img_path=data["path"]+"/"+data["test"],
            imgsz=1024,
            batch_size=1,
            augment=False,  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=False,  # rectangular batches
            cache= None,
            single_cls=False or False,
            stride=int(32),
            pad=0.5,
            prefix=f'val: ',
            use_segments=False,
            use_keypoints=False,
            classes=None,
            data=data,
            fraction= 1.0)
test_loader = DataLoader(dataset=dataset, collate_fn= dataset.collate_fn,batch_size=1, shuffle=True, num_workers=0, drop_last=False)


for i,data in enumerate(test_loader):
    batch = data
    break

if isinstance(batch['img'], dict):
    batch['img']["backbone"] = batch['img']["backbone"].float().cuda() / 255

model.model.args = cfg
flops, params = profile(model.model.cuda(), inputs=(batch, ))
print("model:")
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
