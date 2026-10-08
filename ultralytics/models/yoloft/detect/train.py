# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy
from torch import nn, optim
import os
import numpy as np
import torch.nn as nn
import gc
from ultralytics.data import build_dataloader, build_yoloft_train_dataset, build_video_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yoloft
from ultralytics.nn.tasks import VideoDetectionModel
from ultralytics.nn.modules.memory_buffer import StreamBuffer_onnx 
from ultralytics.utils import LOGGER, RANK, LOCAL_RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results, plot_video
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.nn.modules import MSTF_STREAM, MSTF_STREAM_cbam
import warnings
from copy import copy

import numpy as np
import torch
from torch import distributed as dist
import time
from ultralytics.utils import (
    LOGGER,
    RANK,
    TQDM,
    __version__,
    colorstr,
)
from ultralytics.utils.torch_utils import (
    autocast
    )

class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        
        if mode == "train":
            return build_yoloft_train_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        else:
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train", just_dataloader = False):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
            
        if not just_dataloader:
            with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
                dataset = self.build_dataset(dataset_path, mode, batch_size)
        else:
            dataset = dataset_path
        
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        
        assert self.data.get('StreamVideoSampler', False), "StreamDataset must be True"
        if mode == "train":
            return build_video_dataloader(dataset, batch_size, workers, shuffle, rank, stack=True)  # return dataloader
        else:
            return build_video_dataloader(dataset, batch_size, workers, shuffle, rank, stack=False)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        
        return batch

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))
            self.save_sample_flag = False # plot data video again
            
    def preprocess_batch_video(self, batch_video):
        for i, frame in enumerate(batch_video):
            batch_video[i] = self.preprocess_batch(frame)
        return batch_video
    
    def now_length(self, epoch):
        split_index = 0
        for i in range(len(self.args.train_slit) - 1):
            if self.args.train_slit[i] <= epoch < self.args.train_slit[i + 1]:
                split_index = i
                break
        else:
            # 如果当前 epoch 大于等于最后一个分割点
            split_index = len(self.args.train_slit) - 1

        # 获取待设置的数据集长度
        target_length = self.data["train_video_length"][split_index]
        return target_length, int(self.data["split_batch_dict"][target_length])
    
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # plot batch video debug
        video_batch_list = []
        bbox_list = []
        batch_list_idx = []
        cls_list = []
        masks_list = []
        self.save_sample_flag = False
        
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        
        if isinstance(self.args.train_slit, int):
            self.args.train_slit = [self.args.train_slit]
        
        if "train_video_length" in self.data:
            assert len(self.args.train_slit) == len(self.data["train_video_length"]), "must equter"
        else:
            # 如果没有设置数据集长度，则默认每个 epoch 子视频长度为1
            self.data["train_video_length"] = [1]
            self.args.train_slit = [0]
        
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            
            # target_length, target_batch_size = self.now_length(epoch)
            # # 检查当前数据集长度是否和待设置的一样
            # if self.train_loader.dataset.length != target_length:
            #     if hasattr(self.train_loader.dataset, '_train_video'):
            #         if RANK in {-1, 0}:
            #             LOGGER.info('start train video\n')
            #         self.batch_size = target_batch_size * max(world_size, 1)
            #         dataset = self.train_loader.dataset
            #         self.train_loader = self.get_dataloader(dataset, batch_size=target_batch_size, rank=LOCAL_RANK, mode="train", just_dataloader=True)
            #         self.train_loader.dataset._train_video(hyp=self.args, length=target_length)
            #     # 重置训练加载器
            #     self.train_loader.reset()
            # LOGGER.info(f"Training epoch {epoch} with dataset length {self.train_loader.dataset.length} and batch size: {self.train_loader.batch_size}")


            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
                
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch_videos in pbar:
                self.run_callbacks("on_train_batch_start")
                # if i > 10:
                #     break
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    # batch = self.preprocess_batch(batch)
                    batch = batch_videos[0]
                    batch_videos = self.preprocess_batch_video(batch_videos)
                    # print(batch_videos[0]["im_file"])
                    self.loss, self.loss_items = self.model({"train_video":batch_videos})
     
                    # self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                        
                    if self.data.get('StreamVideoSampler', False) and self.args.plots:
                        for b_ in batch_videos:
                            save_flag = False
                            if not save_flag and not self.save_sample_flag:  
                                video_batch_list.append(b_["img"].clone().cpu())
                                bbox_list.append(b_["bboxes"].clone().cpu())
                                batch_list_idx.append(b_["batch_idx"].clone().cpu())
                                cls_list.append(b_['cls'].squeeze(-1).clone().cpu())
                                if "masks" in b_:
                                    masks_list.append(b_["gt_mask"].clone().cpu())
                                else:
                                    masks_list.append(np.zeros(0, dtype=np.uint8))
                                if len(video_batch_list) == 80: #save 50 frames
                                    save_flag = True
                            if save_flag and not self.save_sample_flag:
                                self.save_sample_flag = True
                                self.plot_training_video_samples(video_batch_list,b_,bbox_list,batch_list_idx,cls_list, masks_list)
                                del video_batch_list,bbox_list,batch_list_idx,cls_list, masks_list
                                video_batch_list = []
                                bbox_list = []
                                batch_list_idx = []
                                cls_list = []
                                masks_list = []
                                self.save_sample_flag = True #only plot once
                            
                self.run_callbacks("on_train_batch_end")
                
                if (i+1)*len(batch_videos) % 1000 <= len(batch_videos): #2500张图片后请理
                    # print('clear memory')
                    self._clear_memory()
                    del batch_videos

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if (self.args.val and (epoch+1)%self.args.val_interval == 0) or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")
        
        
    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = VideoDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yoloft.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_video_samples(self, video_batch_list,batch,bbox_list,batch_list_idx,cls_list,masks_list=[]):
        """Plots training samples with their annotations."""
        plot_video(video_batch_list,
                    batch_list_idx=batch_list_idx,
                    cls_list=cls_list,
                    bboxes_list=bbox_list,
                    masks_list = masks_list,
                    paths=None,
                    fname=self.save_dir / f'train_batch.mp4',
                    on_plot=self.on_plot)
        
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            masks=batch["masks"] if "masks" in batch else np.zeros(0, dtype=np.uint8),
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get batch size by calculating memory occupation of model."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)

    # def build_optimizer(self, model, model_teacher=None, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    #     """
    #     Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
    #     weight decay, and number of iterations.

    #     Args:
    #         model (torch.nn.Module): The model for which to build an optimizer.
    #         name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
    #             based on the number of iterations. Default: 'auto'.
    #         lr (float, optional): The learning rate for the optimizer. Default: 0.001.
    #         momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
    #         decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
    #         iterations (float, optional): The number of iterations, which determines the optimizer if
    #             name is 'auto'. Default: 1e5.

    #     Returns:
    #         (torch.optim.Optimizer): The constructed optimizer.
    #     """
    #     g = [], [], []  # optimizer parameter groups
    #     bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    #     if name == "auto":
    #         LOGGER.info(
    #             f"{colorstr('optimizer:')} 'optimizer=auto' found, "
    #             f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
    #             f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
    #         )
    #         nc = getattr(model, "nc", 10)  # number of classes
    #         lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
    #         name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
    #         self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

    #     for module_name, module in model.named_modules():
    #         for param_name, param in module.named_parameters(recurse=False):
    #             fullname = f"{module_name}.{param_name}" if module_name else param_name
    #             if "bias" in fullname:  # bias (no decay)
    #                 g[2].append(param)
    #             elif isinstance(module, bn):  # weight (no decay)
    #                 g[1].append(param)
    #             else:  # weight (with decay)
    #                 g[0].append(param)
    #     # change new_____________________
    #     if self.Distillation is not None:
    #         for v in model_teacher.modules():
    #             # print(v)
    #             if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
    #                 g[2].append(v.bias)
    #             if isinstance(v, bn):  # weight (no decay)
    #                 g[1].append(v.weight)
    #             elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
    #                 g[0].append(v.weight)

    #     optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
    #     name = {x.lower(): x for x in optimizers}.get(name.lower())
    #     if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
    #         optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    #     elif name == "RMSProp":
    #         optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    #     elif name == "SGD":
    #         optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    #     else:
    #         raise NotImplementedError(
    #             f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
    #             "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
    #         )

    #     optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    #     optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    #     LOGGER.info(
    #         f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
    #         f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
    #     )
    #     return optimizer
    
    # def build_optimizer(self, model, model_teacher=None, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    #     """
    #     Constructs an optimizer with different learning rates for temporal layers and other layers.
    #     Temporal layers specified by self.args.time_model will use the base learning rate (lr),
    #     while other layers will use lr * lr_scale (default 0.1).
    #     """
    #     # 获取时间层配置和学习率缩放比例
    #     time_layers = getattr(self.args, 'time_model', [])
    #     lr_scale = getattr(self.args, 'lr_scale', 0.1)  # 默认缩放比例为0.1
    #     time_layer_prefixes = [f'model.{x}.' for x in time_layers]
    #     if len(time_layer_prefixes) == 0:
    #         lr_scale = 1.0
    #         LOGGER.info("no time_layer_prefixes, all layer use normal lr")
    #     else:
    #         LOGGER.info(f"time_layer_prefixes: {time_layer_prefixes}, other layer use {lr_scale}*lr")

    #     # 参数分组：时间层和其他层各分为权重、BN、bias三组
    #     g_time = [[], [], []]  # 时间层的权重、BN、bias
    #     g_other = [[], [], []]  # 其他层的权重、BN、bias
    #     bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # BN层类型

    #     # 遍历模型的参数，分配到时序层或其他层组
    #     for module_name, module in model.named_modules():
    #         for param_name, param in module.named_parameters(recurse=False):
    #             fullname = f"{module_name}.{param_name}" if module_name else param_name
    #             # 判断参数所属的原组（权重、BN、bias）
    #             if "bias" in fullname:
    #                 group_idx = 2
    #             elif isinstance(module, bn):
    #                 group_idx = 1
    #             else:
    #                 group_idx = 0
    #             # 检查是否属于时间层
    #             is_time_layer = any(fullname.startswith(prefix) for prefix in time_layer_prefixes)
    #             # 分配到对应的组
    #             if is_time_layer:
    #                 g_time[group_idx].append(param)
    #             else:
    #                 g_other[group_idx].append(param)

    #     # 处理知识蒸馏中的教师模型参数（添加到其他层组）
    #     if self.Distillation is not None and model_teacher is not None:
    #         for module_name, module in model_teacher.named_modules():
    #             for param_name, param in module.named_parameters(recurse=False):
    #                 fullname = f"{module_name}.{param_name}" if module_name else param_name
    #                 # 教师模型参数默认分配到其他层组
    #                 if "bias" in fullname:
    #                     group_idx = 2
    #                 elif isinstance(module, bn):
    #                     group_idx = 1
    #                 else:
    #                     group_idx = 0
    #                 g_other[group_idx].append(param)

    #     # 构建参数组列表，排除空组
    #     param_groups = []
    #     # 时间层参数组（学习率不变）
    #     for group_idx in range(3):
    #         params = g_time[group_idx]
    #         if not params:
    #             continue
    #         wd = decay if group_idx == 0 else 0.0  # 仅权重组有decay
    #         param_groups.append({'params': params, 'weight_decay': wd, 'lr': lr})
            
    #     # 其他层参数组（学习率缩放）
    #     for group_idx in range(3):
    #         params = g_other[group_idx]
    #         if not params:
    #             continue
    #         wd = decay if group_idx == 0 else 0.0
    #         param_groups.append({'params': params, 'weight_decay': wd, 'lr': lr * lr_scale})

    #     # 根据优化器类型初始化
    #     optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
    #     name = {x.lower(): x for x in optimizers}.get(name.lower())
    #     if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
    #         optimizer = getattr(optim, name, optim.Adam)(param_groups, betas=(momentum, 0.999))
    #     elif name == "RMSProp":
    #         optimizer = optim.RMSprop(param_groups, momentum=momentum)
    #     elif name == "SGD":
    #         optimizer = optim.SGD(param_groups, momentum=momentum, nesterov=True)
    #     else:
    #         raise NotImplementedError(f"Optimizer '{name}' not supported.")

    #     # 日志输出参数组信息
    #     time_counts = [len(g) for g in g_time]
    #     other_counts = [len(g) for g in g_other]
    #     LOGGER.info(
    #         f"{colorstr('optimizer:')} {type(optimizer).__name__} configured with:\n"
    #         f"Temporal layers (lr={lr}): {sum(time_counts)} params - "
    #         f"{time_counts[0]} weights(decay={decay}), {time_counts[1]} BN, {time_counts[2]} bias\n"
    #         f"Other layers (lr={lr*lr_scale}): {sum(other_counts)} params - "
    #         f"{other_counts[0]} weights(decay={decay}), {other_counts[1]} BN, {other_counts[2]} bias"
    #     )

    #     return optimizer

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # normalize in channel diemension
            import torch.nn.functional as F
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss

class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel_s, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel_s,channel in zip(channels_s,channels_t)
        ]

    def forward(self, y_s, y_t,layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss

class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0).to(device)
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]
        self.norm1 = [
            nn.BatchNorm2d(set_channel, affine=False).to(device)
            for set_channel in channels_s
        ]

        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # change ---
            if self.distiller == 'cwd':
                s = self.align_module[idx](s)
                s = self.norm[idx](s)
            else:
                s = self.norm1[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss

class Distillation_loss:
    def __init__(self, modeln, modelL, distiller="CWDLoss"):  # model must be de-paralleled

        self.distiller = distiller
        # layers = ["2","4","6","8","12","15","18","21"]
        layers = ["6", "8", "12", "15", "18", "21"]
        # layers = ["15","18","21"]

        # channels_s=[32,64,128,256,128,64,128,256]
        # channels_t=[128,256,512,512,512,256,512,512]
        # channels_s=[128,256,128,64,128,256]
        # channels_t=[512,512,512,256,512,512]
        le = len(layers)
        channels_s = [256, 480, 256, 64, 143, 229][-le:]
        channels_t = [256, 512, 256, 128, 256, 512][-le:]
        # channels_s=[64,128,256]
        # channels_t=[256,512,512]
        self.D_loss_fn = FeatureLoss(channels_s=channels_s, channels_t=channels_t, distiller=distiller[:3])

        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []


        for mname, ml in modelL.named_modules():
            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    if name[1] in layers:
                        if "cv2" in mname:
                            self.teacher_module_pairs.append(ml)

        for mname, ml in modeln.named_modules():

            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    # print(mname)
                    if name[1] in layers:
                        if "cv2" in mname:
                            self.student_module_pairs.append(ml)

    def register_hook(self):
        self.teacher_outputs = []
        self.origin_outputs = []

        def make_layer_forward_hook(l):
            def forward_hook(m, input, output):
                l.append(output)

            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            # 为每层加入钩子，在进行Forward的时候会自动将每层的特征传送给model_outputs和origin_outputs
            self.remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(self.origin_outputs)))

    def get_loss(self):
        quant_loss = 0
        # for index, (mo, fo) in enumerate(zip(self.teacher_outputs, self.origin_outputs)):
        #     print(mo.shape,fo.shape)
        # quant_loss += self.D_loss_fn(mo, fo)
        quant_loss += self.D_loss_fn(y_t=self.teacher_outputs, y_s=self.origin_outputs)
        if self.distiller != 'cwd':
            quant_loss *= 0.3
        self.teacher_outputs.clear()
        self.origin_outputs.clear()
        return quant_loss

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
