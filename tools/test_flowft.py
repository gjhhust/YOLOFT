from ultralytics.models import FLOWFT

# Load a COCO-pretrained RT-DETR-l model
path = "YoloftS_basline3_convnoany_epoch18s_epoch18_32.6(small AP best).pt"
model = FLOWFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data='config/XS-VID+copy.yaml',cfg="config/train/orige_stream.yaml",batch=1,device=[3],imgsz=1024, workers=1,save_json = True)  # no arguments needed, dataset and settings remembered
