import os
device = [3]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import YOLOFT


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("runs/detect/train22/weights/last.pt") #resume
model = YOLOFT("config/yoloft/yoloft-L.yaml").load("yolov8l.pt")

results = model.train(data="config/target.yaml",cfg="config/train/orige_stream.yaml", batch=12, epochs=24, imgsz=1024, device=device,workers = 6)
