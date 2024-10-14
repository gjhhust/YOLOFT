import os
device = [2,3]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import YOLOFT,YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("yoloft/train113/weights/last.pt") #resume
model = YOLOFT("config/yoloft_dev/yoloftv2-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("./config/yolo/yolov8.yaml").load("yolov8l.pt") #train yolov8-l

results = model.train(data="config/dataset_dev/Train_gaode5_Test_minigaode6.yaml",cfg="config/train/orige_stream.yaml", batch=20, epochs=21, imgsz=896, device=device,workers = 6)
