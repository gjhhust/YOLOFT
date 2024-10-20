from ultralytics.models import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
path = "yoloft/train242/weights/epoch8.pt"
model = YOLOFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data='config/dataset_dev/task1_train1.yaml',cfg="config/train/orige_stream.yaml",batch=8,device=[2],imgsz=896, workers=4,save_json = True)  # no arguments needed, dataset and settings remembered