import os
device = [2]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import YOLOFT,YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("runs/detect/train22/weights/last.pt") #resume
model = YOLO("/data/jiahaoguo/YOLOFT/config/yolo/yolo11.yaml").load("yolo11l.pt")
# model = YOLOFT("yolo11l.pt")
results = model.train(data="config/gaode_5.yaml",cfg="config/train/orige_stream.yaml", batch=14, epochs=30, imgsz=896, device=device,workers = 6)
