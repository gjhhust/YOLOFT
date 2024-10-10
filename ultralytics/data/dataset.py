# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import torch.distributed as dist
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable

from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms,movedet_train_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label,verify_image_movelabel


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)
        self.get_coco_image_id()

    def from_coco_get_image_id(self,coco_data,im_file):
        if coco_data:
            for img in coco_data["images"]:
                if im_file == img["file_name"]:
                    return img["id"]
        return 0
    
    def get_coco_image_id(self):
        #val
        coco_data = None
        if "eval_ann_json" in self.data:
            with open(self.data["eval_ann_json"], 'r') as coco_file:
                coco_data = json.load(coco_file)

        for i,image_path in enumerate(self.im_files):
            # image_name = os.path.splitext(os.path.basename(image_path))[0]
            self.labels[i]["image_id"] = self.from_coco_get_image_id(coco_data,os.path.basename(image_path))

    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
        return x
    
    # def img2label_paths(self,img_paths):
    #         """Define label paths as a function of image paths."""
    #         sa, sb = f'{os.sep}{self.data["images_dir"]}', f'{os.sep}{self.data["labels_dir"]}'  # /images/, /labels/ substrings
    #         import pdb;pdb.set_trace()
    #         return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
    def img2label_paths(self,img_paths):
        sa, sb = f'{os.sep}{self.data["images_dir"]}', f'{os.sep}{self.data["labels_dir"]}'  # /images/, /labels/ substrings
        return [path.replace(sa, sb).split('.')[0]+'.txt' for path in img_paths]

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = self.img2label_paths(self.im_files)
        # import pdb;pdb.set_trace()
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = movedet_train_transforms(self, self.imgsz, hyp)
        else:
            if self.data["val_reimgsz"]: #测试时使用resize
                transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
            else:
                transforms = Compose([LetterBox(new_shape=(0, 0), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch

import os
import numpy as np
import math
from copy import deepcopy
import random
import time,json
import re
#add by guojiahao
class MOVEDETDataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.
    self.img_video_info{
         "frame_number"
        "video_name"
        "video_first_frame"
        "video_last_frame"
        "neg_idx"
        "pos_idx"
        "is_train"
        
    }
    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, *args, rank=-1, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        self.rank = rank
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)
        self.match_number = self.data["match_number"]
        self.interval = self.data["interval"]
        self.im_frame_matching(self.im_files)
        self.epoch = 0
        

    def from_coco_get_image_id(self,coco_data,im_file):
        if coco_data:
            for img in coco_data["images"]:
                if im_file == img["file_name"]:
                    return img["id"]
        return 0
    
    def video_sampler_split(self, video_image_dict, mode="all",length=100, raandom_seed=100):
        '''
        mode: all split_random split_legnth
        '''
        random.seed(raandom_seed)
        if mode=="all":
            self.length = -1
            self.sub_video_splits = []
            min_length = 0
            for video_name,video_list in video_image_dict.items():
                min_length = min(min_length, len(video_list))

            for video_name,video_list in video_image_dict.items():
                self.sub_video_splits.append(video_list)
                    
        elif mode=="split_legnth":
            min_video_lengthes = min([len(video_list)-1 for video_list in video_image_dict.values()])
            self.length = min(min_video_lengthes,length) #Video segments should not be too long
            print(f"min length video is {self.length}")
            self.sub_video_splits = []
            # Get the full division of self.interval interval for each video
            for video_name,video_list in video_image_dict.items():
                for i in range(self.interval):
                    sub_interval_list = video_list[i::self.interval]
                    sub_interval_length_list = [sub_interval_list[i:i + self.length] for i in range(0, len(sub_interval_list), self.length)]
                    sub_interval_length_list[-1] = sub_interval_list[-self.length:]
                    for sub_ in sub_interval_length_list:
                        if len(sub_) == self.length:
                            self.sub_video_splits.append(sub_)
            # import pdb;pdb.set_trace()
            if self.rank != -1: #muti gpu
                world_size = dist.get_world_size()
                if len(self.sub_video_splits) % world_size != 0:#Can't average the scores.
                    len_sub_video_splits = len(self.sub_video_splits)
                    nearest_multiple_of_3 = ((len_sub_video_splits - 1) // world_size + 1) * world_size
                    difference = nearest_multiple_of_3 - len_sub_video_splits
                    for i in range(difference):
                        self.sub_video_splits.append(list(self.sub_video_splits[i]))
        elif mode=="split_random":
            print(f"min length rate video is {length}")
            self.sub_video_splits = []
            # Get the full division of self.interval interval for each video
            for video_name,video_list in video_image_dict.items():
                video_length = len(video_list)-1
                max_rate = video_length//min(length, video_length) + 1
                split_length = random.choice([video_length//rate for rate in range(1,max_rate)])
    
                for i in range(self.interval):
                    sub_interval_list = video_list[i::self.interval]
                    sub_interval_length_list = [sub_interval_list[i:i + split_length] for i in range(0, len(sub_interval_list), split_length)]
                    sub_interval_length_list[-1] = sub_interval_list[-split_length:]
                    for sub_ in sub_interval_length_list:
                        if len(sub_) == split_length:
                            self.sub_video_splits.append(sub_)

        if self.rank != -1: #multicard
            world_size = dist.get_world_size()
        else:
            world_size = 1

        # self.sub_video_splits = self.sub_video_splits[:6] #debug
        indices = list(range(len(self.sub_video_splits)))
        random.shuffle(indices)
        self.muti_rank_indices_splits, self.muti_rank_sub_video_len, self.per_gpu_total_frames = self.split_video_frames(self.sub_video_splits, indices, world_size)
        
        if self.rank == 0 or self.rank == -1:
            print(f"\n*******************{'[Train]' if self.augment else '[Test]'}dataset split info************************")
            print(f"len sub videos is {[len(spi) for spi in self.sub_video_splits]}")
            print(f"per GPU frames len: {self.muti_rank_sub_video_len}")
            print(f"per GPU video number: {[len(sub_indexs) for sub_indexs in self.muti_rank_indices_splits]}")
            print(f"muti_rank_indices_splits: ")
            print(self.muti_rank_indices_splits)
            print(f"*************************************************")

        #init the first frame of the subvideo
        for info in self.img_video_info:
            if "seed" in info:
                del info["seed"]
            if "is_first" in info:
                info["is_first"] = []
        i = 0
        for gpu_video_index_list in self.muti_rank_indices_splits: #Storing a different seed for each video that the gpu may access means that if there are videos or images that are accessed twice with different seed
            # At the same time dataset get item will take seed in order and use the
            for index_video in gpu_video_index_list:
                sub_list = self.sub_video_splits[index_video]
                for index, frame in enumerate(sub_list):
                    if index==0:
                        self.img_video_info[frame["index"]]["is_first"].append(True)
                    else:
                        self.img_video_info[frame["index"]]["is_first"].append(False)


                    if "seed" in self.img_video_info[frame["index"]]:
                        self.img_video_info[frame["index"]]["seed"].append(i*5) #Belongs to muti videos
                    else:
                        self.img_video_info[frame["index"]]["seed"] = [i*5]  #Video Enhanced Random Seeds, One Video One Seed
                i += 1
                    

    def video_init_split(self, video_image_dict):
        if self.augment:
            self.video_sampler_split(video_image_dict, mode="split_random", length=50)
        else:
            self.video_sampler_split(video_image_dict, mode="all")
            

    def end_train_all_video(self):
        print(f"change data video split closed")
        self.video_sampler_split(self.video_image_dict, mode="all")

    def split_video_frames(self, sub_videos_list, indices, n):
        '''
        The multi-card environment allocates the training data on each GPU, and returns a list of length n. Each list represents the index of a part of the sub_videos_list, and the total length of the n sub-lists is similar.
        partitions_len is the length of each partition.
        return the shortest total number of frames as the total length of training min_total_frames, progress bar display: min_total_frames//batch_size
        '''
        # Calculate the total number of video frames for each sub-list
        total_frames = [len(video) for video in sub_videos_list]
        print(f"now dataset total frame is: {total_frames}")
        # Calculate the total number of video frames for all sublists
        total_frames_sum = sum(total_frames)
        
        # Calculate the total number of video frames each sublist should contain
        target_frames_per_partition = total_frames_sum // n
        
        # Initialization Result List
        partitions = [[] for _ in range(n)]
        partitions_len = [0 for _ in range(n)]

        current_partition = 0
        current_partition_frames = 0
        
        # Iterate through each sub-list
        for video_index in indices:
            frames_count = total_frames[video_index]
            # If the number of video frames in the current partition has exceeded the target, move to the next partition
            if current_partition_frames + frames_count > target_frames_per_partition:
                partitions[current_partition].append(video_index)
                partitions_len[current_partition] += frames_count

                current_partition += 1
                current_partition_frames = 0
                continue
            
            # Add the current sublist to the current partition
            partitions[current_partition].append(video_index)
            partitions_len[current_partition] += frames_count
            current_partition_frames += frames_count
        
        if partitions_len[current_partition] < target_frames_per_partition:
            video_index = indices[0]
            partitions[current_partition].append(video_index)
            partitions_len[current_partition] += frames_count

        return partitions, partitions_len, min(partitions_len)

    def get_index_from_sub(self, ix, iy): #ix indexes subvideo, iy indexes video frames
        return self.sub_video_splits[ix][iy]["index"]
            
        
    def im_frame_matching(self, im_files):
        # import json
        
        #val
        coco_data = None
        if "eval_ann_json" in self.data:
            with open(self.data["eval_ann_json"], 'r') as coco_file:
                coco_data = json.load(coco_file)

        # Create a dictionary that groups images by video name
        video_image_dict = {}
        img_video_info = []
        for i,image_path in enumerate(im_files):
            video_name = os.path.basename(os.path.dirname(image_path))
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            frame_num_string = image_name.split('_')[-1] # Assuming the video name is the first part of the filename separated by '_'
            # Extract numeric parts using regular expressions
            match = re.search(r'\d+', frame_num_string)
            digits = match.group()
            frame_num = int(digits)
            # print(f"string: {frame_num_string} -> {frame_num}")
            
            if video_name not in video_image_dict:
                video_image_dict[video_name] = []
            video_image_dict[video_name].append({
                "index":i,
                "frame_number":frame_num
            })
            img_video_info.append({
                "frame_number":frame_num,
                "video_name":video_name,
                "image_id":self.from_coco_get_image_id(coco_data,video_name+"/"+os.path.basename(image_path))
            })

        # Sort each video by frame number
        for key,value in video_image_dict.items():
            sorted_value = sorted(value, key=lambda x: x['frame_number'])
            video_image_dict[key] = sorted_value
        
    
        # Frame information before and after writing
        for video_name,video_list in video_image_dict.items(): 
            video_first_frame = video_list[0]["frame_number"] + self.match_number * self.interval
            video_last_frame = video_list[-1]["frame_number"] - self.match_number * self.interval
            # Generate indexes in the negative direction (excluding 0)
            neg_idxs = np.arange(-self.interval, -self.interval * self.match_number - 1, -self.interval)
            # Generate indexes in the positive direction (excluding 0)
            pos_idxs = np.arange(self.interval, self.match_number * self.interval + 1, self.interval)

            for i,frame in enumerate(video_list):
                # import pdb;pdb.set_trace()
                index = frame["index"]
                cur_frame_number = frame["frame_number"]
                # Starting and ending frame numbers for training
                img_video_info[index]["video_first_frame"] = video_first_frame
                img_video_info[index]["video_last_frame"] = video_last_frame
                neg_idx_cur = (neg_idxs + i).clip(0)
                pos_idx_cur = (pos_idxs + i).clip(0,len(video_list)-1)

                neg_idx_cur = neg_idx_cur if cur_frame_number  >= video_first_frame else None
                pos_idx_cur = pos_idx_cur if cur_frame_number <= video_last_frame else None

                #It's all there in order to train.
                if (neg_idx_cur is not None) and (pos_idx_cur is not None):
                    img_video_info[index]["is_train"] = True
                    img_video_info[index]["neg_idx"] = [video_list[idx]["index"] for idx in neg_idx_cur]
                    img_video_info[index]["pos_idx"] = [video_list[idx]["index"] for idx in pos_idx_cur]
                else:
                    img_video_info[index]["is_train"] = False
                
                img_video_info[index]["is_first"] = (cur_frame_number == video_first_frame)

        self.img_video_info = img_video_info
        self.video_image_dict = video_image_dict
        self.video_init_split(video_image_dict.copy())
        
        return True
    
    def _set_same_transform(self):
        # Get the current time as a random number seed
        # seed = int(time.time())
        seed = int(time.time())
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        #Get the reference image of a transform with the same index.
    def get_ref_img(self,index):

        _dict = deepcopy(self.labels[index])
        _dict['img'], _dict['ori_shape'], _dict['resized_shape'] = self.load_image(index, is_resize =self.data["val_reimgsz"])
        _dict['ratio_pad'] = (_dict['resized_shape'][0] / _dict['ori_shape'][0],
                            _dict['resized_shape'][1] / _dict['ori_shape'][1])
        if self.rect:
            _dict['rect_shape'] = self.batch_shapes[self.batch[index]]

        _trans_dict = self.transforms(self.update_labels_info(_dict))
        
        return _trans_dict["img"].clone()

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        orige_dict = self.get_image_and_label(index)
        
        trans_dict = self.transforms(orige_dict)
        trans_dict["img_metas"] = {
            "frame_number":0,
            "video_name":"none",
            "is_first": True,
            "epoch":self.epoch
        }
        # self.show_transforms(orige_dict,trans_dict)
        return trans_dict
    
    # def get_image_and_label(self, index):
    #     """Get and return label information from the dataset."""
    #     image_info = self.img_video_info[index]
        
    #     # if not train, random select
    #     while(not image_info["is_train"]):
    #         index = random.randint(0, len(self.img_video_info)-1)
    #         image_info = self.img_video_info[index]
            
    #     label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
    #     label.pop('shape', None)  # shape is for rect, remove it
    #     label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index, is_resize =self.data["val_reimgsz"])
        
    #     label["pos_idx"],label["neg_idx"] = image_info["pos_idx"],image_info["neg_idx"]
        
    #     label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
    #                           label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
    #     label["image_id"] = image_info["image_id"]
    #     if self.rect:
    #         label['rect_shape'] = self.batch_shapes[self.batch[index]]
    #     return self.update_labels_info(label)    
                
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index, is_resize =self.data["val_reimgsz"])
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        if self.rect:
            if self.data["val_reimgsz"]:
                label['rect_shape'] = self.batch_shapes[self.batch[index]]
            else:
                label['rect_shape'] = np.ceil(np.array(label['resized_shape']) / self.stride + 0.5).astype(int) * self.stride
        return self.update_labels_info(label)
    
    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
        return x

    def img2label_paths(self,img_paths):
            """Define label paths as a function of image paths."""
            sa, sb = f'{os.sep}{self.data["images_dir"]}', f'{os.sep}{self.data["labels_dir"]}'  # /images/, /labels/ substrings
            return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
        
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = self.img2label_paths(self.im_files)
        # import pdb;pdb.set_trace()
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = movedet_train_transforms(self, self.imgsz, hyp)
        else:
            if self.data["val_reimgsz"]: #Use resize for testing
                transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
            else:
                transforms = Compose([LetterBox(new_shape=(0, 0), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        # add Get fusion frames img_refs from 0-n Sort images by distance from negative distance to positive maximal distance
        if "img_refs" in batch[0]:
            for batch_ in batch:
                neg = torch.mean(batch_["img_refs"][0].to(torch.float32),axis=0).to(torch.uint8).unsqueeze(0)
                pos = torch.mean(batch_["img_refs"][-1].to(torch.float32),axis=0).to(torch.uint8).unsqueeze(0)
                batch_["img"] = torch.concat([batch_["img"],neg,pos],axis=0)

        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls', 'is_moving']:
                value = torch.cat(value, 0)

            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

        new_batch["img"] = {
                "backbone":new_batch["img"],
                "img_metas":new_batch["img_metas"]
        }
        # new_batch["cls"] = new_batch["is_moving"]
        return new_batch

def tensor_numpy(img):
    img = img.numpy()
    img = img.transpose(1,2,0)[::-1]
    return img


def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh

class MOVEHomoDETDataset(MOVEDETDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.
    self.img_video_info{
         "frame_number"
        "video_name"
        "video_first_frame"
        "video_last_frame"
        "neg_idx"
        "pos_idx"
        "is_train"
        
    }
    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Image Mean
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        # image standard deviation
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))
        # Height and width of the image block
        # self.patch_h = self.data["patch_h"] #为了防止得到的图像旋转出现黑边，裁原图一部分
        # self.patch_w = self.data["patch_w"]

        # Image block sampling interval
        self.rho = self.data["rho"]

    def _set_orige_paramters(self, hyp):
        self.orige_paramters = {}
        self.orige_paramters["homo"] = hyp.homo  # set mosaic ratio=0.0
        self.orige_paramters["box"] = hyp.box
        self.orige_paramters["cls"] = hyp.cls
        self.orige_paramters["dfl"] = hyp.dfl
        self.orige_paramters["perspective"] = hyp.perspective
        self.orige_paramters["degrees"] = hyp.degrees
        self.orige_paramters["mosaic"] = hyp.mosaic
    
    
    def _freeze_memory(self,model):
        freeze = ['model.11.', 'model.16.', 'model.21.']  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True 
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False



    def _train_all(self, hyp, model):
        """Sets bbox loss and builds transformations."""
        hyp.homo = self.orige_paramters["homo"]  # set mosaic ratio=0.0
        hyp.box = self.orige_paramters["box"]
        hyp.cls = self.orige_paramters["cls"]
        hyp.dfl = self.orige_paramters["dfl"]  # keep the same behavior as previous v8 close-mosaic
        hyp.perspective = self.orige_paramters["perspective"]
        hyp.degrees = self.orige_paramters["degrees"]
        hyp.mosaic = self.orige_paramters["mosaic"]
        self.transforms = self.v8build_transforms(hyp) 
        
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
          

    def _train_homonet(self, hyp):
        """Sets bbox loss and builds transformations."""
        hyp.homo = self.orige_paramters["homo"] # set mosaic ratio=0.0
        hyp.box = 0.0
        hyp.cls = 0.0
        hyp.dfl = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.perspective = self.orige_paramters["perspective"]
        hyp.degrees = self.orige_paramters["degrees"]
        self.transforms = self.build_transforms(hyp)    
        
    def _freeze_homonet(self, model):
        freeze = ['model.11.']  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False
    
    def _freeze_bboxnet(self, model):
        not_freeze = ['model.11.']  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = False  # train all layers
            if any(x in k for x in not_freeze):
                print(f'unfreezing {k}')
                v.requires_grad = True
                    
    def _train_bboxnet(self, hyp):
        """Sets bbox loss and builds transformations."""
        hyp.homo = 0.0  # set mosaic ratio=0.0
        hyp.box = self.orige_paramters["box"]
        hyp.cls = self.orige_paramters["cls"]
        hyp.dfl = self.orige_paramters["dfl"]  # keep the same behavior as previous v8 close-mosaic
        hyp.perspective = 0.0
        hyp.degrees = 0.0
        hyp.mosaic = self.orige_paramters["mosaic"]
        self.transforms = self.v8build_transforms(hyp) 
        


    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        orige_dict = self.get_image_and_label(index)
        
        self._set_same_transform()
        trans_dict = self.transforms(orige_dict.copy())
        
        trans_dict['img_ref'] = self.get_ref_img(orige_dict["neg_idx"][0])
        motion = self._homoDta_preprocess(tensor_numpy(trans_dict["img"]),tensor_numpy(trans_dict['img_ref']))
        trans_dict.update(motion)
        # self.show_transforms(orige_dict,trans_dict)
        return trans_dict
    
    
    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""

        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in ('img',"org_imgs","input_tensors","patch_indices","h4p"):
                
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)

            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

        new_batch["img"] = {
            "backbone":new_batch["img"],
            "motion":{
                "org_imgs": new_batch["org_imgs"],
                "input_tensors": new_batch["input_tensors"],
                "patch_indices": new_batch["patch_indices"],
                "h4p": new_batch["h4p"]
            }
        }
        # new_batch["cls"] = new_batch["is_moving"]
        return new_batch
    
import threading
import torch.distributed as dist

class MOVEHomoDETDataset_stream(MOVEHomoDETDataset):
    """

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_fre = [0] * len(self.img_video_info)
        for info in self.img_video_info:
            info["is_train"] = True
        
    
    def _set_samevideo_transform(self, seed):
        # Get the current time as a random number seed
        # seed = int(time.time())
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def _train_video(self, hyp):
        """Sets bbox loss and builds transformations."""
        # hyp.mosaic = 0.0
        self.transforms = self.v8build_transforms(hyp) 
        self.video_sampler_split(self.video_image_dict, mode="split_random", length=self.data["split_length"][1])

    def _train_backbone(self, hyp):
        """Sets bbox loss and builds transformations."""
        # hyp.mosaic = 1.0
        self.transforms = self.v8build_transforms(hyp) 
        self.video_sampler_split(self.video_image_dict, mode="split_legnth", length=self.data["split_length"][0])

    def _train_all(self, hyp):
        """Sets bbox loss and builds transformations."""
        # hyp.mosaic = 1.0
        self.transforms = self.v8build_transforms(hyp) 
        self.video_sampler_split(self.video_image_dict, mode="all", length=self.data["split_length"][0])
        
    def v8build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            if self.data["val_reimgsz"]: #测试时使用resize
                transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
            else:
                transforms = Compose([LetterBox(new_shape=(0, 0), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms
    
    def _show_data_use(self,epoch):
            import matplotlib.pyplot as plt
            data_frequency = list(self.data_fre)

            # bins = 500  # 设置为500个bins
            # range_min = min(data_frequency)
            # range_max = max(data_frequency)

            # plt.hist(data_frequency, bins=bins, range=(range_min, range_max), edgecolor='k')

            # plt.xlabel('use')
            # plt.ylabel('number')
            # plt.title('hist')

            # plt.savefig("/data2/guojiahao/ultralytics/show_pipeline/data_use/1.png")
            
            # data = np.array(self.data_fre)
            # print(f"1:{len(data[data==1])}")
            

            plt.figure(figsize=(20, 6))

            plt.bar(range(len(self.data_fre)), self.data_fre)

            plt.xlabel('Index')
            plt.ylabel('number')
            plt.title('Index distrbution')

            plt.xticks(range(0, len(self.data_fre), 100))  # Display one x-axis label for every 100 data points

            rank_ = dist.get_rank()
            
            plt.savefig(f"/data2/guojiahao/ultralytics/show_pipeline/data_use/{epoch}_{rank_}.png")
        
    # Take the current seed that should be used (as opposed to selecting the video that the current image belongs to)
    def select_now_seed(self, seed_list):
        seed = seed_list[0]
        del seed_list[0]
        seed_list.append(seed)
        return seed, seed_list

    
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        image_info = self.img_video_info[index]
        is_train = image_info["is_train"]
        # if not train
        # while(not is_train):
        #     index = random.randint(0, len(self.img_video_info)-1)
        #     image_info = self.img_video_info[index]
        #     is_train = image_info["is_train"]
            
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index, is_resize =self.data["val_reimgsz"])
        
        # if is_train:
        #     label["pos_idx"],label["neg_idx"] = image_info["pos_idx"],image_info["neg_idx"]
        
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        label["image_id"] = image_info["image_id"]
        label["img_metas"] = {
            "frame_number":image_info["frame_number"],
            "video_name":image_info["video_name"],
            "epoch":self.epoch
        }

        if len(image_info["seed"]) > 1:#image belongs to multiple video clips
            label["seed"], image_info["seed"] = self.select_now_seed(image_info["seed"])
        else:
            label["seed"] = image_info["seed"][0]

        if len(image_info["is_first"]) > 1:#image belongs to multiple video clips
            label["img_metas"]["is_first"], image_info["is_first"] = self.select_now_seed(image_info["is_first"])
        else:
            label["img_metas"]["is_first"] = image_info["is_first"][0]

        if self.rect:
            if self.data["val_reimgsz"]:
                label['rect_shape'] = self.batch_shapes[self.batch[index]]
            else:
                label['rect_shape'] = np.ceil(np.array(label['resized_shape']) / self.stride + 0.5).astype(int) * self.stride
        return self.update_labels_info(label)    
          
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        orige_dict = self.get_image_and_label(index)
        self._set_samevideo_transform(orige_dict["seed"]+self.epoch*10) #Same video in one epoch with consistent random seeds
        trans_dict = self.transforms(orige_dict.copy())
        
        # trans_dict['img_ref'] = self.get_ref_img(orige_dict["neg_idx"][0]) #The most recent frame
        # motion = self._homoDta_preprocess(tensor_numpy(trans_dict["img"]),tensor_numpy(trans_dict['img_ref']))
        # trans_dict.update(motion)

        
        # self.show_transforms(orige_dict,trans_dict)
        trans_dict["index"] = index
        trans_dict["img_metas"] = orige_dict["img_metas"]
        return trans_dict  
    
    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)
          
    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""

        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in ('img',"org_imgs","input_tensors","patch_indices","h4p"):
                value = torch.stack(value, 0)

            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)

            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

        new_batch["img"] = {
            "backbone":new_batch["img"],
            "img_metas":new_batch["img_metas"]
        }
        # new_batch["cls"] = new_batch["is_moving"]
        return new_batch    
        
# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.torch_transforms = classify_transforms(args.imgsz)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        return len(self.samples)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()
