from PIL import Image
from ultralytics import RTDETR
import os
from ultralytics.models import YOLOFT
from ultralytics.models import YOLOFT
from ultralytics.data.build import build_stream_dataloader,build_movedet_dataset
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


cfg = cfg2dict("config/train/orige_stream.yaml") #dict
cfg = DictWrapper(cfg)
data = cfg2dict("/data/jiahaoguo/YOLOFT/config/XS-VID.yaml")


# Load a COCO-pretrained RT-DETR-l model
model = YOLOFT('/data/jiahaoguo/YOLOFT/yoloft/train25/weights/best.pt')  # load a custom model
# model = YOLOFT("runs/detect/train150/weights/best.pt")
model.model = model.model.cuda()
show_image_prefix = "YOLOFT"

# load data
gs = int(max(max(model.model.stride), 32))
dataset = build_movedet_dataset(cfg, os.path.join(data["path"],data["val"]), 1, data, mode="val", stride=gs)
dataloader = build_stream_dataloader(dataset, 1, 4, shuffle = False, rank=-1)
results = model(dataloader)

