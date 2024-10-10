from pycocotools.coco import COCO  # noqa
# from pycocotools.cocoeval import COCOeval  # noqa
from ultralytics.data.cocoeval import COCOeval  # noqa

# anno_json = "/data/jiahaoguo/dataset/speed_merge/merge_test_1.json"
# pred_json = "/data/jiahaoguo/YOLOFT/yoloft/train53/predictions.json"
anno_json = "/data/jiahaoguo/dataset/gaode_all/gaode_5/annotations/coco/test.json"
pred_json = "/data/jiahaoguo/ultralytics/runs/gaode_5/train269_DCN3_24.4/predictions.json"


anno = COCO(str(anno_json))  # init annotations api
pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
eval = COCOeval(anno, pred, 'bbox')

# 仅对类别1进行评估
eval.params.catIds = [1]  # 设定类别ID为1

eval.evaluate()
eval.accumulate()
eval.summarize()
print(eval.stats)
print(pred_json)







# from pycocotools.coco import COCO  # noqa
# from ultralytics.data.cocoeval import COCOeval  # noqa

# anno_json = "/data/jiahaoguo/dataset/speed_merge/merge_test_1.json"
# pred_json = "/data/jiahaoguo/YOLOFT/yoloft/train53/predictions.json"

# # 加载 COCO 数据集的注释和预测
# anno = COCO(str(anno_json))  # 初始化注释 API
# pred = anno.loadRes(str(pred_json))  # 初始化预测 API

# # 实例化 COCOeval
# eval = COCOeval(anno, pred, 'bbox')

# # 过滤指定范围内的 images
# eval.params.imgIds = [img_id for img_id in anno.getImgIds() if 20793 <= img_id <= 31521]

# # 进行评估
# eval.evaluate()
# eval.accumulate()
# eval.summarize()

# print(eval.stats)
# print(pred_json)
