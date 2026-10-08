# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
from torch.utils.data import dataloader, distributed
from ultralytics.utils import LOGGER
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOVideoDataset, YOLOMultiModalDataset, YOLOStreamDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    LoadTensorBuffer,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils import LOGGER
import torch
import random
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class StreamSampler(DistributedSampler):
    '''
    Streaming Sampling Datasets from Video
    '''
    def __init__(self, dataset, batch_size, seed=10,distributed=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.seed = seed
        if distributed:
            self.rank = dist.get_rank()  # Get the rank of the current process
            self.world_size = dist.get_world_size()  # Get the total number of processes
        else:
            self.rank = 0
            self.world_size = 1
            
        self.last_next_frame_index = [None for i in range(int(batch_size))]
        self.next_train_video_index = 0

        # random.seed(self.seed)
        # indices = list(range(len(self.dataset.sub_video_splits)))
        # random.shuffle(indices)
        
        # indices_per_rank = len(indices) // self.world_size  #  Data subset size per process
        # start_idx = self.rank * indices_per_rank
        # end_idx = (self.rank + 1) * indices_per_rank
        
        # self.indices = indices[start_idx:end_idx]  # Video subset of the current process
        # self.indices = self.dataset.muti_rank_indices_splits[self.rank]
        self.data_length = self.dataset.per_gpu_total_frames - self.dataset.per_gpu_total_frames%self.batch_size

        print(f"worksize:{self.world_size}, rank: {self.rank}")
    def __iter__(self):

        self.last_next_frame_index = [None for i in range(int(self.batch_size))]
        self.next_train_video_index = 0
        self.indices = self.dataset.muti_rank_indices_splits[self.rank]
        
        random.seed(44)
        random.shuffle(self.indices)
        
        self.data_length = self.dataset.per_gpu_total_frames - self.dataset.per_gpu_total_frames%self.batch_size
        # print(f"sampler init: {len(self.indices)}")

        i = 0
        for _ in range(self.data_length):
            if i == self.batch_size:
                i = 0
            
            if self.last_next_frame_index[i] is not None:
                ix,iy = self.last_next_frame_index[i] # Video indexing and frame indexing
                index = self.dataset.get_index_from_sub(ix,iy)
                cur_sampler_is_train = self.dataset.img_video_info[index]["is_train"]
            else: 
                cur_sampler_is_train = False
                
            if cur_sampler_is_train:                        #Current Video Continue Training
                yield index
                if iy + 1 == len(self.dataset.sub_video_splits[ix]):
                    self.last_next_frame_index[i] = None    #Video training complete.
                else:
                    self.last_next_frame_index[i][-1] += 1  #Continue training this video using the next frame
            else:                                           #Switching Video Training
                ix, iy = self.indices[self.next_train_video_index], -1
                self.next_train_video_index += 1
                if self.next_train_video_index == len(self.indices):
                    self.next_train_video_index = 0

                is_train = False
                # if not train, next
                while(not is_train):
                    iy += 1
                    index = self.dataset.get_index_from_sub(ix, iy)
                    is_train = self.dataset.img_video_info[index]["is_train"] if self.dataset.augment else True
                    if iy + 1 == len(self.dataset.sub_video_splits[ix]): 
                        self.last_next_frame_index[i] = None #Video training complete.
                    else:
                        self.last_next_frame_index[i] = [ix, iy+1] #Video Continues Training
                yield index
                
            i += 1
            
        self.last_next_frame_index = [None for i in self.last_next_frame_index]
        self.next_train_video_ix = 0
        
    def _epoch_reset(self):
        self.last_next_frame_index = [None for i in self.last_next_frame_index]
        self.next_train_video_ix = 0 

    def __len__(self):
        return self.data_length


class VideoSampler(DistributedSampler):
    '''
    Streaming Sampling Datasets from Video
    '''
    def __init__(self, dataset, batch_size, seed=10, distributed=False, shuffle=True):
        # Check if distributed mode is enabled and process group is already initialized
        self.dataset = dataset  # self.sub_videos
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.distributed = distributed
        self.shuffle = shuffle
        
        if distributed:
            self.rank = dist.get_rank()  # Get the rank of the current process
            self.world_size = dist.get_world_size()  # Get the total number of processes
        else:
            self.rank = 0
            self.world_size = 1
        
    def __iter__(self):
        # Generate indices for sub-videos (each sub-video is a list of frames)
        indices = list(self.dataset.sampler_indices[self.rank])
        
        # Shuffle if needed (for training)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
            if self.rank == 0 or self.rank == -1:
                print(f"epoch: {self.epoch}, shuffle indices: {indices[:5]}")
            
        for i in indices:
            yield i

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset.sampler_indices[self.rank])  # Number of batches

    def _epoch_reset(self):
        '''
        Reset for a new epoch: re-calculate the video frame indices and re-shuffle the dataset.
        This is necessary when the sub-video divisions change.
        '''
        # Recalculate the video frame indices if necessary (if you modified all_sub_videos)
        self.video_frame_indices = self.dataset._get_video_frame_indices()

        # Reset any other internal state as needed
        self.seed += 1  # Optionally increment the seed for the next epoch
        self.epoch += 1  # Increment the epoch
        self.set_epoch(self.epoch)  # Update the epoch in the distributed sampler

from typing import Iterator, List, Optional, Dict, Any
class StreamDistributedSampler(distributed.DistributedSampler):
    """
    分布式或单卡视频流采样器，预先生成每个 epoch 的帧索引列表。

    继承自 DistributedSampler (用于处理分布式逻辑) 或作为基础 Sampler (单卡模式)。
    根据 `distribute` 参数和 `batch_size`、`world_size` (训练模式下) 调整子视频数量，
    并将分配到的子视频的帧索引按指定格式高效展平存储。

    Args:
        dataset (YOLOStreamDataset): 数据集对象，必须包含 sub_videos (List[List[int]])
                                     和 is_training_augment (bool) 属性。
        batch_size (int): DataLoader 使用的 batch size (每批次的总帧数)。
                          在训练模式下，这用于确定要使用的子视频总数应是其倍数。
                          注意：这里 batch_size 的使用方式与 DataLoader 直接批处理帧不同，
                          它影响的是参与采样的子视频总数。
        distribute (bool, optional): 是否使用分布式采样。如果为 False，则在单卡模式下运行，
                                     忽略 num_replicas 和 rank 参数 (或使用默认值 1 和 0)。
                                     默认为 True。
        num_replicas (int, optional): 分布式训练的进程/副本总数。默认从 torch.distributed 获取 (如果 distribute=True)。
        rank (int, optional): 当前进程/副本的排名。默认从 torch.distributed 获取 (如果 distribute=True)。
        shuffle (bool, optional): 是否在每个 epoch 开始时打乱子视频顺序。默认为 True。
        seed (int, optional): 随机种子，用于打乱子视频顺序。默认为 0。
        drop_last (bool): 注意: 此参数继承自 DistributedSampler 但不直接控制最终帧批次的丢弃。
                          我们根据 batch_size 和 world_size 在初始化时进行子视频数量的调整。
                          父类的 drop_last 不会影响本采样器最终 yielded 的帧数量。
                          保留此参数主要是为了兼容父类签名。
    """
    def __init__(self,
                 dataset, # 类型提示应为 Dataset，但我们期望 YOLOStreamDataset
                 batch_size: int,
                 distributed: bool = True, # 新增参数：是否启用分布式采样
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 stack: bool = False, # stack sub-videos into a single batch
                 drop_last: bool = False): # drop_last 参数保留用于签名兼容性

        self.distribute = distributed # 存储分布式模式标志
        self.stack = stack

        # 根据 distribute 标志确定实际用于采样的 num_replicas 和 rank
        if self.distribute:
            # 如果显式提供了 num_replicas 和 rank，则使用它们
            # 否则，从 torch.distributed 获取
            if num_replicas is None:
                if not dist.is_initialized():
                    raise RuntimeError("在分布式模式下 (distribute=True)，torch.distributed 必须已初始化或必须显式提供 num_replicas 和 rank 参数。")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_initialized():
                    raise RuntimeError("在分布式模式下 (distribute=True)，torch.distributed 必须已初始化或必须显式提供 num_replicas 和 rank 参数。")
                rank = dist.get_rank()
            # 在分布式模式下，确保 num_replicas 和 rank 是非 None 的
            assert num_replicas is not None and rank is not None

        else: # 单卡模式
            # 在单卡模式下，强制 num_replicas=1, rank=0
            num_replicas = 1
            rank = 0
            if dist.is_initialized():
                 # 如果 dist 已初始化但指定了单卡模式，发出警告
                 LOGGER.warning("StreamDistributedSampler 指定 distribute=False，但在已初始化的分布式环境中运行。将使用单卡采样逻辑，忽略进程组信息。")


        # 调用父类 __init__。此时，实际的 num_replicas 和 rank 已经被确定并传递给父类。
        # 父类将使用这些值来设置其内部状态，尽管在单卡模式下这些值是 1 和 0。
        # 我们主要继承父类来获取其属性设置和 set_epoch 的基本逻辑。
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        if batch_size <= 0:
            raise ValueError(f"batch_size 必须是正数，但得到 {batch_size}")

        # 验证数据集是否包含必要的属性
        if not hasattr(dataset, 'sub_videos') or not isinstance(dataset.sub_videos, list) or \
           not hasattr(dataset, 'is_training_augment'):
             raise TypeError("数据集必须是 YOLOStreamDataset 实例或拥有 'sub_videos' (列表的列表) 和 'is_training_augment' (布尔值) 属性。")

        self.dataset = dataset # 存储数据集对象
        self.batch_size = batch_size
        self.is_training = dataset.augment # 从数据集获取训练模式

        self._num_original_sub_videos = len(dataset.sub_videos) # 调整/选择前的子视频总数
        self.flat_frame_indices: List[int] = [] # 这将存储当前周期的展平帧索引列表

        # mode_str = "分布式模式" if self.distribute else "单卡模式"
        # LOGGER.info(f"进程 {self.rank+1}/{self.num_replicas}: 初始化 StreamDistributedSampler ({mode_str}，周期 0)。"
        #             f"训练模式: {self.is_training}。批次大小 (用于修剪): {self.batch_size}。")
        # LOGGER.info(f"进程 {self.rank+1}/{self.num_replicas}: 周期 0 的总帧索引数: {len(self.flat_frame_indices)}")
        self._recalculate_indices()

    def _recalculate_indices(self) -> None:
        """
        重新计算当前周期的打乱、分布式分配和展平后的帧索引列表。
        使用 PyTorch 张量操作高效完成展平。
        由 __init__ 和 set_epoch 调用。
        """
        if self._num_original_sub_videos == 0:
            self.flat_frame_indices = []
            self._num_sub_videos_per_replica = 0
            self._effective_total_sub_videos = 0
            LOGGER.warning(f"进程 {self.rank+1}/{self.num_replicas}: 数据集没有子视频。采样器将不会生成任何索引。")
            return

        # 1. 生成并打乱原始子视频索引 [0, ..., _num_original_sub_videos - 1]
        sub_video_indices = list(range(self._num_original_sub_videos))
        if self.shuffle and self.is_training:
            g = torch.Generator()
            # 基于周期和随机种子生成，确保不同周期有不同打乱顺序，且同一周期在不同副本上打乱顺序相同
            g.manual_seed(self.seed + self.epoch)
            sub_video_indices = torch.randperm(self._num_original_sub_videos, generator=g).tolist()

        # 2. 确定所有副本将使用的子视频总数
        # 这是训练模式下应用丢弃/修剪逻辑的地方
        if self.is_training:
            # 根据要求，总子视频数必须是 (batch_size * num_replicas) 的整数倍。
            # 这里的 num_replicas 已经是根据 distribute 参数调整后的值 (分布式 >=1, 单卡=1)
            trim_denominator = self.batch_size * self.num_replicas
            if trim_denominator == 0: # 避免除以零，尽管 batch_size 和 num_replicas 应该都是正数
                total_sub_videos_to_use = 0
            else:
                total_sub_videos_to_use = (self._num_original_sub_videos // trim_denominator) * trim_denominator

            if total_sub_videos_to_use == 0 and self._num_original_sub_videos > 0:
                LOGGER.warning(
                    f"进程 {self.rank+1}/{self.num_replicas}: 在训练模式下，batch_size={self.batch_size} 和 num_replicas={self.num_replicas} (有效值)，"
                    f"修剪分母为 {trim_denominator}。原始子视频总数 ({self._num_original_sub_videos}) 少于此分母。"
                    f"所有子视频都将被舍弃，采样器将不会生成任何索引。"
                )

        else: # 测试/验证模式
            # 使用所有可用的子视频，不进行基于 batch_size 的修剪
            total_sub_videos_to_use = self._num_original_sub_videos

            # 在测试模式下，当 distribute=False 时 num_replicas=1，
            # total_sub_videos_to_use 总是能被 num_replicas 整除，因此不会触发子视频列表填充。
            # 如果 distribute=True (分布式测试)，会进行子视频列表填充以确保均匀分配。
            if self.num_replicas > 0 and total_sub_videos_to_use % self.num_replicas != 0:
                padding_size = self.num_replicas - (total_sub_videos_to_use % self.num_replicas)
                # 用列表开头的部分进行填充
                sub_video_indices_to_pad = sub_video_indices[:total_sub_videos_to_use]
                sub_video_indices = sub_video_indices_to_pad + sub_video_indices_to_pad[:padding_size]
                total_sub_videos_to_use = len(sub_video_indices)
                assert total_sub_videos_to_use % self.num_replicas == 0 # 填充后应能整除


        # 修剪/选择打乱后的子视频索引列表到确定的总数
        sub_video_indices = sub_video_indices[:total_sub_videos_to_use]
        self._effective_total_sub_videos = len(sub_video_indices) # 经过修剪或填充后的子视频总数


        # 3. 在副本之间分配已选定的子视频索引
        # 在单卡模式下 (num_replicas=1, rank=0)，start_idx_replica=0, end_idx_replica=_effective_total_sub_videos,
        # replica_sub_video_indices 将包含所有 _effective_total_sub_videos 个子视频的索引。
        # 在分布式模式下，正常分配。
        assert self.num_replicas > 0, "num_replicas 必须大于 0"
        assert self._effective_total_sub_videos % self.num_replicas == 0, \
            f"错误: 经过调整的总子视频数 ({self._effective_total_sub_videos}) 必须能被进程数 ({self.num_replicas}) 整除。"

        self._num_sub_videos_per_replica = self._effective_total_sub_videos // self.num_replicas

        start_idx_replica = self.rank * self._num_sub_videos_per_replica
        end_idx_replica = start_idx_replica + self._num_sub_videos_per_replica
        replica_sub_video_indices = sub_video_indices[start_idx_replica:end_idx_replica]

        assert len(replica_sub_video_indices) == self._num_sub_videos_per_replica, \
            f"进程 {self.rank+1}: 副本子视频数量不匹配。预期 {self._num_sub_videos_per_replica}，实际 {len(replica_sub_video_indices)}"

        # 4. 为当前副本高效展平帧索引列表
        # 使用 PyTorch 张量操作，即使子视频长度不同 (测试模式) 或相同 (训练模式)。
        # 在测试模式下，子视频长度不同，需要填充以便构建张量。
        # 在训练模式下，子视频长度相同，填充虽然代码上存在，但实际不会填充。
        self.flat_frame_indices = []
        if not replica_sub_video_indices:
            LOGGER.warning(f"进程 {self.rank+1}/{self.num_replicas}: 分发/修剪后没有子视频分配到此副本。")
            return

        if self.stack:
            self.flat_video_indices = replica_sub_video_indices
            return
        
        # 获取分配到的子视频的实际帧索引列表
        replica_frame_lists_info = [self.dataset.sub_videos[sv_idx]
                            for sv_idx in replica_sub_video_indices
                            # 再次检查索引有效性
                            if 0 <= sv_idx < len(self.dataset.sub_videos)]
        replica_frame_lists = [[info["original_idx"] for info in sub_infos] 
          for sub_infos in replica_frame_lists_info]


        if not replica_frame_lists:
            self.flat_frame_indices = []
            return

        # 找到这些子视频中的最大长度
        max_len = max(len(lst) for lst in replica_frame_lists)

        if max_len == 0:
            self.flat_frame_indices = []
            LOGGER.warning(f"进程 {self.rank+1}/{self.num_replicas}: 分配到的子视频列表的最大长度为 0.")
            return

        # 创建一个填充值，必须是一个在 dataset.im_files 索引范围之外的值
        # 在测试模式下，子视频长度不同，需要填充张量。训练模式下不需要实际填充。
        # 填充值 (-1) 在最终的 self.flat_frame_indices 列表中会被过滤掉。
        padding_value = -1 # 使用 -1 作为填充值

        # 创建一个 PyTorch 张量来存储填充后的帧索引
        # 形状为 [分配给此副本的子视频数, 这些视频中的最大长度]
        padded_tensor = torch.full((len(replica_frame_lists), max_len), padding_value, dtype=torch.long)

        # 将每个子视频的帧索引复制到张量中
        # 在训练模式下，len(frame_list) == max_len，填充不会实际发生。
        # 在测试模式下，len(frame_list) <= max_len，会发生填充。
        for i, frame_list in enumerate(replica_frame_lists):
            if frame_list: # 避免处理空列表导致切片错误
                # 确保 frame_list 中的索引在有效范围内，尽管通常是这样
                if all(0 <= idx < len(self.dataset.im_files) for idx in frame_list):
                    padded_tensor[i, :len(frame_list)] = torch.tensor(frame_list, dtype=torch.long)
                else:
                    LOGGER.error(f"进程 {self.rank+1}: 子视频 {replica_sub_video_indices[i]} (原始索引) 包含超出数据集总帧数范围的帧索引.")
                    # 可以选择在此处跳过此子视频或清空 flat_frame_indices
                    # 为了简单起见，这里只记录错误，并让填充值为 -1 保持，后续会被过滤
                    pass # 保持填充值为-1

        # 转置张量，使其形状变为 [最大长度, 分配给此副本的子视频数]
        # 这使得按时间步遍历更加自然 (遍历第一维)
        N, length = padded_tensor.shape
        padded_tensor = padded_tensor.reshape(N//self.batch_size, self.batch_size, length)
        transposed_tensor = padded_tensor.permute(0, 2, 1).reshape(-1, self.batch_size)

        # 展平张量，同时过滤掉填充值 (-1)。
        # 无论训练还是测试模式，填充值都不会出现在最终输出列表中。
        self.flat_frame_indices = transposed_tensor[transposed_tensor != padding_value].tolist()

        # 再次验证生成的索引范围 (可选但推荐)
        if self.flat_frame_indices:
            min_idx = min(self.flat_frame_indices)
            max_idx = max(self.flat_frame_indices)
            if min_idx < 0 or max_idx >= len(self.dataset.im_files):
                LOGGER.error(f"进程 {self.rank+1}: 生成的帧索引超出数据集总帧数范围。min={min_idx}, max={max_idx}, total_frames={len(self.dataset.im_files)}.")


        # 记录重新计算的结果
        mode_str = "分布式模式" if self.distribute else "单卡模式"
        LOGGER.info(f"进程 {self.rank+1}/{self.num_replicas}, 周期 {self.epoch}: 在 {mode_str} 下重新计算索引完成。"
                    f"使用了 {self._effective_total_sub_videos} 个子视频 (分配给此副本 {len(replica_sub_video_indices)} 个)。"
                    f"为此副本生成的总帧索引数: {len(self.flat_frame_indices)}。")



    def __iter__(self) -> Iterator[int]:
        """生成当前周期预先计算好的帧索引。"""
        # 在初始化时立即计算并存储索引 (用于 Epoch 0)
        # 直接返回预先计算好的列表的迭代器
        # LOGGER.info(f"Rank {self.rank}/{self.num_replicas}, Epoch {self.epoch}: Starting iteration.")
        if self.stack:
            return iter(self.flat_video_indices)
        else:
            return iter(self.flat_frame_indices)

    def __len__(self) -> int:
        """返回此副本在此周期将生成的总帧索引数量。"""
        # 长度就是预先计算好的列表的长度
        if self.stack:
            return len(self.flat_video_indices)
        else:
            return len(self.flat_frame_indices)

    def set_epoch(self, epoch: int) -> None:
        """
        设置此采样器的周期编号并重新计算索引。
        当 `shuffle=True` 时，这确保所有副本在每个周期使用不同的随机顺序
        进行子视频打乱，并且索引会根据这个新的打乱重新生成。
        """
        # 父类的 set_epoch 仅更新 self.epoch 和 self.seed。
        # 我们需要确保在 epoch 变化时重新计算索引。
        super().set_epoch(epoch) # 更新 self.epoch 和内部随机种子状态
        # LOGGER.info(f"Rank {self.rank}/{self.num_replicas}: Setting epoch to {epoch} and recalculating indices.")
        self.dataset.set_epoch(epoch) # 更新数据集的 epoch
        self._recalculate_indices() # 重新计算索引以反映新的周期打乱
        
from torch.utils.data.sampler import BatchSampler
class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()

class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    if data.get('StreamVideoSampler', False):
        dataset = YOLOStreamDataset
    else:
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
        
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0
    )
    

def build_yoloft_train_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    # dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    if data.get('StreamVideoSampler', False):
        dataset = YOLOVideoDataset
    else:
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
        
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

    
def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_video_dataloader(dataset, batch, workers, shuffle=True, rank=-1, stack=False):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = StreamDistributedSampler(dataset, batch, seed=1, stack=stack, distributed=False) if rank == -1 else StreamDistributedSampler(dataset, batch, seed=1, stack=stack, distributed=True, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )

# def build_stream_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
#     """Return an InfiniteDataLoader or DataLoader for training or validation set."""
#     batch = min(batch, len(dataset))
#     nd = torch.cuda.device_count()  # number of CUDA devices
#     nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
#     sampler = StreamSampler(dataset, batch, seed=1, distributed=False) if rank == -1 else StreamSampler(dataset, batch, seed=1, distributed=True)
#     generator = torch.Generator()
#     generator.manual_seed(6148914691236517205 + RANK)
#     return InfiniteDataLoader(
#         dataset=dataset,
#         batch_size=batch,
#         shuffle=shuffle and sampler is None,
#         num_workers=nw,
#         sampler=sampler,
#         pin_memory=PIN_MEMORY,
#         collate_fn=getattr(dataset, "collate_fn", None),
#         worker_init_fn=seed_worker,
#         generator=generator,
#     )
    
def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor, buffer = False, False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        if len(source)==2 and len(source[-1]) == 3:
            if isinstance(source[0], torch.Tensor):
                tensor = True
            else:
                from_img = True
            buffer = True
        else:
            source = autocast_list(source)  # convert all list elements to PIL or np arrays
            from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor, buffer


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor, buffer = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor, buffer)

    # Dataloader
    if tensor and buffer:
        dataset = LoadTensorBuffer(source[0], source[1])
    elif tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img and buffer:
        dataset = LoadPilAndNumpy(source[0], source[1])
        # AttributeError("error, img buffer have not achieve")
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
