import os
device = [0,1]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import YOLOFT,YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("yoloft/train113/weights/last.pt") #resume
model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L-stream3-384.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLOFT("./config/yolo/yolov8.yaml").load("yolov8l.pt") #train yolov8-l

results = model.train(data="config/dataset_dev/task1_train2.yaml",cfg="config/train/orige_stream_trendloss.yaml", batch=12*2, epochs=21, imgsz=896, device=device,workers = 6)
