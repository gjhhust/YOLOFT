import os
device = [0,1]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import YOLOFT,YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("yoloft/train113/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
model = YOLOFT("./config/yolo/yolov8.yaml").load("yolov8l_backbone.pt") #train yolov8-l

results = model.train(data="config/dataset_dev/task6_train1.yaml",cfg="config/train/orige_stream.yaml", batch=16*2, epochs=10, imgsz=896, device=device,workers = 6)
