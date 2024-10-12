from ultralytics.models import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
path = "yoloft/train94/weights/best.pt"
model = YOLOFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data='config/gaode_5.yaml',cfg="config/train/orige_stream.yaml",batch=1,device=[0],imgsz=896, workers=1,save_json = True)  # no arguments needed, dataset and settings remembered
