from ultralytics.models import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
path = "yoloft-L.pt"
model = YOLOFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data='config/XS-VID.yaml',cfg="config/train/orige_stream.yaml",batch=1,device=[1],imgsz=1024, workers=1,save_json = True)  # no arguments needed, dataset and settings remembered
