from pycocotools.coco import COCO  # noqa
# from pycocotools.cocoeval import COCOeval  # noqa
from ultralytics.data.cocoeval import COCOeval  # noqa

anno_json = "path/to/test.json"
pred_json = "path/to/predictions.json"

anno = COCO(str(anno_json))  # init annotations api
pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
eval = COCOeval(anno, pred, 'bbox')

eval.evaluate()
eval.accumulate()
eval.summarize()
print(eval.stats)
print(pred_json)
