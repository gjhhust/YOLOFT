# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_movedet_dataset,build_stream_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import de_parallel
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend,AutoBackend_MY
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import increment_path
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from pycocotools.coco import COCO
from ultralytics.data.cocoeval import COCOeval
import matplotlib.pyplot as plt
from ultralytics.cfg import cfg2dict
import cv2

class DetectionValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()


    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            model = trainer.model
            # self.args.half = self.device.type != 'cpu'  # force FP16 val during training
            model = model.half() if self.args.half else model.float()
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
            model = de_parallel(model)
        # if self.training:
        #     self.device = trainer.device
        #     self.data = trainer.data
        #     model = trainer.ema.ema or trainer.model
        #     self.args.half = self.device.type != 'cpu'  # force FP16 val during training
        #     model = model.half() if self.args.half else model.float()
        #     self.model = model
        #     self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        #     self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        #     model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks('on_val_start')
            assert model is not None, 'Either trainer or model is needed for validation'
            model = AutoBackend_MY(model,
                                device=select_device(self.args.device, self.args.batch),
                                dnn=self.args.dnn,
                                data=self.args.data,
                                fp16=self.args.half)
            self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type == 'cpu':
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get("val"), self.args.batch)

            model.eval()
            # model.warmup(imgsz=(1 if pt else self.args.batch, self.model.model.yaml.get("ch",3), imgsz, imgsz))  # warmup

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            # if batch_i >= 10:
            #     break

            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch['img'], augment=augment)
                

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            # if ("classes_map" in self.data):
            #     preds = self.use_class_map(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks('on_val_batch_end')

        # self.get_result_coco_eval()
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}

            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / f'predictions.json'), 'w') as f:
                    print(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats, print_log=True)  # update stats
                results.update(stats)
            print('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            print('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    print(f'Saving {f.name}...')
                    # LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
                print(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats
    
    def use_class_map(self,preds):
        for i,pred in enumerate(preds):
            pred_classes = pred[:, 5].long()
            mapped_classes = torch.tensor([self.class_map.get(int(class_id), self.nc) for class_id in pred_classes], device=pred.device).float()
            valid_mask = mapped_classes.int() != self.nc
            pred[:, 5] = mapped_classes
            pred_filtered = pred[valid_mask]
            preds[i] = pred_filtered
        return preds
    
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        if isinstance(batch['img'], dict):
            batch['img']["backbone"] = batch['img']["backbone"].to(self.device, non_blocking=True)
            batch['img']["backbone"] = (batch['img']["backbone"].half() if self.args.half else batch['img']["backbone"].float()) / 255
            if "motion" in batch['img']:
                for key in batch['img']['motion'].keys():
                    batch['img']['motion'][key] = batch['img']['motion'][key].to(self.device, non_blocking=True).float()
            nb = len(batch['img']["backbone"])
        else:
            batch['img'] = batch['img'].to(self.device, non_blocking=True)
            batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
            nb = len(batch['img'])
            
        
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        
        self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = ("eval_ann_json" in self.data)  # is COCO
        if self.is_coco:
            with open(self.data["eval_ann_json"], 'r') as f:
                self.gt_cocodata = json.load(f)

        if "classes_map" in self.data:
            self.class_map = self.data["classes_map"]
            self.class_map = {int(index):value  for index, value in enumerate(self.class_map)}
        else:
            self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       )

    def update_metrics(self, preds, batch):
        """Metrics."""
        if isinstance(batch['img'], dict):
            batch_imgs = batch['img']["backbone"]
        else:
            batch_imgs = batch['img']
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1
            
            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch_imgs[si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch_imgs.shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch_imgs[si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                # labelsn = torch.cat((cls, is_moving, tbox), 1)  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1))) # (conf, pcls, tcls)
        

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot)

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)
    
    def _process_batch_moving(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 7]), x1, y1, x2, y2, conf, class, is_moving
            labels (array[M, 6]),  class,is_moving,  x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 2:], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        # bool_pre_ismoving  = torch.where(detections[:, 6] >= self.args.move, 1, 0) #is_moving预测大于阈值的置为1，否则为0
        # correct_class = (labels[:, 0:1] == detections[:, 5]) & (labels[:, 1:2] == bool_pre_ismoving)  # is_moving and class match
        correct_class = (labels[:, 0:1] == detections[:, 5])
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes and is_moving match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)
    
    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        
        if mode == "train" and self.data["style"]=="muti":
            images_dir = os.path.join(self.data["train_images_dir"])
            labels_dir = os.path.join(self.data["train_labels_dir"])
        elif mode == "val" and self.data["style"]=="muti":
            images_dir = os.path.join(self.data["val_images_dir"])
            labels_dir = os.path.join(self.data["val_labels_dir"])
        elif self.data["style"]=="one":
            images_dir = os.path.join(self.data["path"],self.data["images_dir"])
            labels_dir = os.path.join(self.data["path"],self.data["labels_dir"])
            
        return build_movedet_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs, images_dir=images_dir,
                                     labels_dir=labels_dir,)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
        #add by guojiahao
        if self.args.datasampler == "streamSampler":
            return build_stream_dataloader(dataset, batch_size, self.args.workers, shuffle = False, rank=-1)  # return dataloader
        else:
            return build_dataloader(dataset, batch_size, self.args.workers, shuffle = False, rank=-1)  # normalSampler
        
        # return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        if isinstance(batch, dict):
            batch_imgs = batch['img']["backbone"]
        else:
            batch_imgs = batch['img']
        plot_images(batch_imgs,
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names,
                    on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        if isinstance(batch, dict):
            batch_imgs = batch['img']["backbone"]
        else:
            batch_imgs = batch['img']
        plot_images(batch_imgs,
                    *output_to_target(preds, max_det=self.args.max_det),
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def from_coco_get_image_id(self,coco_data,im_file):
        if coco_data:
            for img in coco_data["images"]:
                if im_file == img["file_name"]:
                    return img["id"]
            raise "No find file name in coco data"
    
    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        if "/" in self.gt_cocodata["images"][0]["file_name"]: #relative address (computing): 
            path, file = os.path.split(filename)
            file_name = os.path.join(os.path.basename(path), file)
        else:
            file_name = os.path.basename(filename)

        image_id = self.from_coco_get_image_id(self.gt_cocodata,file_name)
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})
            
    def eval_json(self, stats, print_log=True):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = Path(self.data['eval_ann_json'])  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            if print_log:
                LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from ultralytics.data.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                # if self.is_coco:
                #     eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                # print(self.metrics.keys[-3],self.metrics.keys[-2], self.metrics.keys[-1])
                # stats[self.metrics.keys[-2]], stats[self.metrics.keys[-1]], stats[self.metrics.keys[-3]]  = eval.stats[:3]  # update  mAP50-95 mAP75  mAP50
                stats["metrics/mAP50-95(B)"] = eval.stats[0]
                stats["metrics/mAP75(B)"] = eval.stats[1]
                stats["metrics/mAP50(B)"] = eval.stats[2]
                stats["metrics/mAP(0-12)"] = eval.stats[3]
                stats["metrics/mAP(12-20)"] = eval.stats[4]
                stats["metrics/mAP(20-32)"] = eval.stats[5]
                stats["metrics/mAP(small)"] = eval.stats[6]
                stats["metrics/mAP(medium)"] = eval.stats[7]
                stats["metrics/mAP(large)"] = eval.stats[8]
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats


def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate trained YOLO model on validation dataset."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = DetectionValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
