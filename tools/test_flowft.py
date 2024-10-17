from ultralytics.models import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
path = "/data/jiahaoguo/YOLOFT/yoloft/train181_yolov8/weights/best.pt"
model = YOLOFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data='config/dataset_dev/Train_gaode5_Test_gaode6.yaml',cfg="config/train/orige_stream.yaml",batch=1,device=[0],imgsz=896, workers=1,save_json = True)  # no arguments needed, dataset and settings remembered