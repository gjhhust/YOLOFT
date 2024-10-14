from ultralytics.models import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
path = "/data/jiahaoguo/YOLOFT/yoloft/train117/weights/last.pt"
model = YOLOFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data='config/dataset_dev/Train_gaode5_Test_minigaode6.yaml',cfg="config/train/orige_stream.yaml",batch=1,device=[1],imgsz=896, workers=1,save_json = True)  # no arguments needed, dataset and settings remembered