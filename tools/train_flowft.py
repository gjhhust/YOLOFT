import os
device = [3]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import FLOWFT


# Load a COCO-pretrained RT-DETR-l model
# model = FLOWFT("runs/detect/train22/weights/last.pt") #resume
model = FLOWFT("config/flownet/DCN+/flowL.yaml").load("yolov8l.pt")

results = model.train(data='config/XS-VID.yaml',cfg="config/train/orige_stream.yaml", batch=12, epochs=70, imgsz=800, device=device,workers = 6)
