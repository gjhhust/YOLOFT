# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yoloft  # noqa
from ultralytics.nn.tasks import MOVEDetectionModel, PoseModel, SegmentationModel
import inspect
import sys
from pathlib import Path
from typing import Union

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.hub.utils import HUB_WEB_ROOT
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load,load_weight_as_name
from ultralytics.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, callbacks,
                               is_git_dir, yaml_load)
from ultralytics.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.utils.downloads import GITHUB_ASSET_STEMS
from ultralytics.utils.torch_utils import smart_inference_mode



class YOLOFT(Model):
    """
    YOLOFT (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
            'detect': {
                'model': MOVEDetectionModel,
                'trainer': yoloft.detect.DetectionTrainer,
                'validator': yoloft.detect.DetectionValidator,
                'predictor': yoloft.detect.DetectionPredictor, },
             }

    def setup_model_train(self, trainer=None, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        check_pip_update_available()
        overrides = self.overrides.copy()
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides.update(kwargs)
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task
        trainer = trainer or self.smart_load('trainer')
        self.trainer = trainer(overrides=overrides, _callbacks=self.callbacks)
        if not overrides.get('resume'):  # manually set model only if not resuming

            ## added by wulianjun
            if 'pretrain_model' in self.model.yaml:
                assert Path(self.model.yaml['pretrain_model']).suffix == '.pt'
                try:
                    _, self.ckpt = attempt_load_one_weight(self.model.yaml['pretrain_model'])
                    self.trainer.model = self.trainer.get_model(weights=self.ckpt, cfg=self.model.yaml)
                except:
                    pass
            else:
                self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        
        
        # # Update model and cfg after training
        # if RANK in (-1, 0):
        #     self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        #     self.overrides = self.model.args
        #     self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP