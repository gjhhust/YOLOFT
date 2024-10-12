import os
device = [2]
if len(device)==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0]).lower() 
from ultralytics.models import YOLOFT,YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("runs/detect/train22/weights/last.pt") #resume
model = YOLOFT("config/yoloft_dev/yoloft-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("./config/yolo/yolov8.yaml").load("yolov8l.pt") #train yolov8-l

results = model.train(data="config/dataset_dev/Train_gaode5_Test_gaode6.yaml",cfg="config/train/orige_stream.yaml", batch=8, epochs=20, imgsz=896, device=device,workers = 6)
