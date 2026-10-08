# Ultralytics YOLO, AGPL-3.0 license
from ultralytics.engine.model import Model
from ultralytics.models import yoloft
from ultralytics.nn.tasks import VideoDetectionModel

class YOLOFT(Model):
    """YOLOFT-L video detection model."""
    def __init__(self, model="config/xsvid/yoloft-l-temporal.yaml", task="detect", verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        return {"detect": {"model": VideoDetectionModel,
                           "trainer": yoloft.detect.DetectionTrainer,
                           "validator": yoloft.detect.DetectionValidator,
                           "predictor": yoloft.detect.DetectionPredictor}}
