# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import numpy as np

from ultralytics.data import build_dataloader, build_movedet_dataset,build_stream_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yoloft
from ultralytics.nn.modules import DetectMOVE
from ultralytics.nn.tasks import MOVEDetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results,plot_videos
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks, clean_url,
                               colorstr, emojis, yaml_save)
from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer)
from tqdm import tqdm
import torch
import math
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def build_dataset(self, img_path, mode='train', batch=None, rank=-1):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_movedet_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs, rank=rank)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size,rank)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        
        #add by guojiahao
        if self.args.datasampler == "streamSampler":
            return build_stream_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader
        else:
            return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # normalSampler

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        if not torch.is_tensor(batch['img']):
            batch['img']["backbone"] = batch['img']["backbone"].to(self.device, non_blocking=True).float() / 255
            if "motion" in batch['img']:
                for key in batch['img']['motion'].keys():
                    batch['img']['motion'][key] = batch['img']['motion'][key].to(self.device, non_blocking=True).float()
        else:
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch
    
    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        # fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        # fitness = metrics["metrics/mAP75(B)"]
        fitness = metrics["metrics/mAP50-95(B)"]
        print(f"now metrics/mAP50-95(B): {fitness*100}")

        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
            print(f"Best: {self.best_fitness*100}")
        return metrics, fitness
    
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs *
                       nb), 100) if self.args.warmup_epochs > 0 else -1  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases

        for epoch in range(self.start_epoch, self.epochs):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()

            if hasattr(self.train_loader.dataset, 'epoch'):
                self.train_loader.dataset.epoch = epoch
                self.train_loader.reset()

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                LOGGER.info('Closing dataloader video ramndom split')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

                if hasattr(self.train_loader.dataset, 'end_train_all_video'):
                    self.train_loader.dataset.end_train_all_video()   
                self.train_loader.reset()
                
            if epoch==0:
                # save starting parameter
                if hasattr(self.train_loader.dataset, '_set_orige_paramters'):
                    self.train_loader.dataset._set_orige_paramters(hyp=self.args)

            if epoch == self.args.train_slit and self.args.train_method == "memory-all":
                LOGGER.info('start train all')
                if hasattr(self.train_loader.dataset, '_train_all'):
                    self.train_loader.dataset._train_all(hyp=self.args,model=self.model)
                self.train_loader.reset()
            elif epoch == self.args.train_slit and self.args.train_method == "random-all":
                LOGGER.info('start train video')
                if hasattr(self.train_loader.dataset, '_train_video'):
                    self.train_loader.dataset._train_video(hyp=self.args)
                self.train_loader.reset()   
            elif epoch == 0 and self.args.train_method == "random-all":
                LOGGER.info('start train backbone')
                if hasattr(self.train_loader.dataset, '_train_backbone'):
                    self.train_loader.dataset._train_backbone(hyp=self.args)
                self.train_loader.reset()       
                

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                # if i >= 10:
                #     break
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss = self.loss*world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    if isinstance(batch["img"], dict):
                        pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img']["backbone"].shape[-1]))
                    else:
                        pbar.set_description(
                            ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                            (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                self.run_callbacks('on_train_batch_end')
            
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):
                
                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if (self.args.val and (epoch+1)%self.args.val_interval == 0)  or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.training = False
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def model_load_wo_P5(self, model, weights):
        cpk_model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = cpk_model.float().state_dict()  # checkpoint state_dict as FP32
        new_csd = dict()

        for k ,v in csd.items():
            if int(k.split('.')[1])==7 or int(k.split('.')[1])==8:
                continue

            if int(k.split('.')[1])>=9:
                new_k = ''
                for i,x in enumerate(k.split('.')):
                    new_k += x+'.' if i!=1 else str(int(x)-2)+'.'

                new_k = new_k[:-1]
            else:
                new_k = k

            new_csd[new_k] = v

        from ultralytics.utils.torch_utils import intersect_dicts
        new_csd = intersect_dicts(new_csd, model.state_dict()) 
        model.load_state_dict(new_csd, strict=False)
        LOGGER.info(f'Transferred {len(new_csd)}/{len(model.state_dict())} items from pretrained weights')


    def model_load_w_P2(self, model, weights):
        ## only for yolov8_s
        cpk_model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = cpk_model.float().state_dict()  # checkpoint state_dict as FP32
        new_csd = dict()
        
        for k ,v in csd.items():

            if int(k.split('.')[1]) >= 16 and int(k.split('.')[1])<=21:
                new_k = ''
                for i,x in enumerate(k.split('.')):
                    new_k += x+'.' if i!=1 else str(int(x)+6)+'.'
                new_k = new_k[:-1]
            else:
                new_k = k

            new_csd[new_k] = v

        model_state_dict = model.state_dict()
        for k in model_state_dict.keys():
            if 'model.28' in k:
                if 'model.28.cv2.0' in k or 'model.28.cv3.0' in k:
                    new_csd[k] = model_state_dict[k].detach().clone()
                else:
                    new_csd[k] = csd[k.replace('model.28.cv2.1','model.22.cv2.0').replace('model.28.cv2.2','model.22.cv2.1').replace('model.28.cv2.3','model.22.cv2.2'). \
                                     replace('model.28.cv3.1','model.22.cv3.0').replace('model.28.cv3.2','model.22.cv3.1').replace('model.28.cv3.3','model.22.cv3.2').replace('model.28.dfl','model.22.dfl')]


        from ultralytics.utils.torch_utils import intersect_dicts
        new_csd = intersect_dicts(new_csd, model.state_dict()) 
        model.load_state_dict(new_csd, strict=False)
        
        LOGGER.info(f'Transferred {len(new_csd)}/{len(model.state_dict())} items from pretrained weights')


                
    def get_model(self, cfg=None, weights=None, verbose=True,p2_p4=False,p2_p5=False):
        """Return a yoloft detection model."""
        model = MOVEDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            if p2_p4:
                self.model_load_wo_P5(model,weights)
                return model
            if p2_p5:
                self.model_load_w_P2(model,weights)
                return model
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        loss_mode = self.args.get("loss_mode","normal")

        if loss_mode == "wasserstein":
            self.loss_names = ['wass_loss', 'cls_loss']
        else:
            self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss']
        # import pdb;pdb.set_trace()
        try:
            task_model = self.model.model[-1]
        except:
            task_model = self.model.module.model[-1]
        if isinstance(task_model, DetectMOVE):
            self.loss_names.append("mov")

        return yoloft.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)
        
    def plot_training_video_samples(self, video_batch_list,batch,bbox_list,batch_list_idx,cls_list):
        """Plots training samples with their annotations."""
        plot_videos(video_batch_list,
                    batch_list_idx=batch_list_idx,
                    cls_list=cls_list,
                    bboxes_list=bbox_list,
                    paths=None,
                    fname=self.save_dir / f'train_batch.mp4',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)
    

def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
