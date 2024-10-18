import os
device = [2,3]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import YOLOFT,YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("yoloft/train113/weights/last.pt") #resume
model = YOLOFT("config/yoloft_dev/yoloftV2-s3-11-L.yaml").load("yolo11l.pt") #train yoloft-l
# model = YOLOFT("./config/yolo/yolov8.yaml").load("yolov8l.pt") #train yolov8-l

results = model.train(data="config/dataset_dev/task1_train2.yaml",cfg="config/train/orige_stream_trendloss.yaml", batch=14*2, epochs=12, imgsz=896, device=device,workers = 4)
