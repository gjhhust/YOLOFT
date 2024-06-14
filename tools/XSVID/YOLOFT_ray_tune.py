import os
import random
import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ultralytics.models import FLOWFT

def read_yaml(path):
    """读取并返回 YAML 文件的内容"""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def model_write_yaml(data, path):
    """将数据写入 YAML 文件"""
    yaml_str = """
# Parameters
nc: {nc}  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  l: {scales_l}  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs

backbone:
  # [from, repeats, module, args]
  {backbone}

head:
  {head}
""".strip()

    # 格式化backbone和head部分
    formatted_backbone = '\n  '.join(
        f'- {item}' for item in data['backbone']
    )
    formatted_head = '\n  '.join(
        f'- {item}' for item in data['head']
    )

    # 填充模板
    yaml_str = yaml_str.format(
        nc=data['nc'],
        scales_l=data['scales']['s'] if "s" in data['scales'] else data['scales']['l'],
        backbone=formatted_backbone,
        head=formatted_head
    )

    with open(path, 'w') as file:
        file.write(yaml_str)

def write_yaml(data, path):
    """将数据写入 YAML 文件"""
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, represent_list, Dumper=NoAliasDumper)

    with open(path, 'w') as file:
        yaml.dump(data, file, Dumper=NoAliasDumper, default_flow_style=False)

import re

def update_model_params(config, updated_model_config):
    """更新模型参数"""
    for key, value in config.items():
        match = re.match(r'(\w+)-(\d+)(?:-(\d+))?-\(([^)]+)\)', key)
        if match:
            module_name, param_index, list_index, param_name = match.groups()
            param_index = int(param_index)
            if list_index is not None:
                list_index = int(list_index)
            # 遍历 backbone 和 head 进行更新
            for section in ['backbone', 'head']:
                for layer in updated_model_config.get(section, []):
                    if layer[2] == module_name:
                        if list_index is not None:
                            layer[3][param_index][list_index] = value
                        else:
                            layer[3][param_index] = value
    return updated_model_config

def update_config(config, params):
    """根据参数更新配置"""
    for key, value in params.items():
        if key in config:
            config[key] = value
    return config

import sys
import logging

def train_model(config, model_config_path, training_config_path, dataset_config_path, device):
    """实际的模型训练和评估代码"""

    log_path = os.path.join(config['save_dir'], 'log_train.log')
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode='w', format='%(message)s')

    # Redirect stdout and stderr to the log file
    sys.stdout = open(log_path, 'a')
    sys.stderr = open(log_path, 'a')

    model = FLOWFT(model_config_path).load(config["pretrain"])
    model.setup_model_train(
        data=dataset_config_path,
        cfg=training_config_path,
        save_dir=config['save_dir'],
        batch=config['batch_size'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        device=device,
        workers=config['workers']
    )
    start_epoch, epochs, val_interval = model.trainer._setup_trainer_train(world_size=1)  # one train just use one GPU, no multi-gpu code

    best_metrics = None
    for epoch in range(start_epoch, epochs):
        is_val = ((epoch + 1) % val_interval == 0)
        metrics = model.trainer.train_one_epoch(epoch, is_val=is_val)
        
        if is_val:
            current_metrics = {
                "metrics/mAP50-95(B)": metrics["metrics/mAP50-95(B)"],
                "metrics/mAP(0-12)": metrics["metrics/mAP(0-12)"]
            }
            if best_metrics is None or current_metrics["metrics/mAP50-95(B)"] > best_metrics["metrics/mAP50-95(B)"] or (current_metrics["metrics/mAP50-95(B)"] == best_metrics["metrics/mAP50-95(B)"] and (current_metrics["metrics/mAP(0-12)"] > best_metrics["metrics/mAP(0-12)"])):
                best_metrics = current_metrics

            # Report intermediate results to Ray Tune
            tune.report(mean_mAP50_95=current_metrics["metrics/mAP50-95(B)"], mean_mAP_0_12=current_metrics["metrics/mAP(0-12)"], training_iteration=epoch+1)

    end_metrics = model.trainer.end_train_val(epoch)
    tune.report(mean_mAP50_95=end_metrics["metrics/mAP50-95(B)"], mean_mAP_0_12=end_metrics["metrics/mAP(0-12)"], training_iteration=epochs+1)

    if best_metrics is None or end_metrics["metrics/mAP50-95(B)"] > best_metrics["metrics/mAP50-95(B)"] or (end_metrics["metrics/mAP50-95(B)"] == best_metrics["metrics/mAP50-95(B)"] and (end_metrics["metrics/mAP(0-12)"] > best_metrics["metrics/mAP(0-12)"])):
        best_metrics = current_metrics
    
    # Restore stdout and stderr
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    return best_metrics

# 定义训练过程
def train(config, base_train_dir, model_config_path, training_config_path, dataset_config_path):
    gpu_ids = ray.get_gpu_ids()
    device = gpu_ids if len(gpu_ids) > 0 else [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device))
    print("device:", os.environ["CUDA_VISIBLE_DEVICES"])

    # 保存更新后的配置到当前训练目录
    if not os.path.exists(base_train_dir):
        os.makedirs(base_train_dir)
    dirs = [d for d in os.listdir(base_train_dir) if os.path.isdir(os.path.join(base_train_dir, d))]
    train_dir = os.path.join(base_train_dir, f'train{len(dirs) + 1}')
    os.makedirs(train_dir, exist_ok=True)

    # 更新保存路径
    config['save_dir'] = train_dir

    # 根据参数更新配置
    updated_model_config = update_config(model_config.copy(), config)
    updated_training_config = update_config(training_config.copy(), config)
    updated_dataset_config = update_config(dataset_config.copy(), config)
    
    updated_model_config = update_model_params(config, updated_model_config)
    
    model_config_path = os.path.join(train_dir, 'model_config.yaml')
    training_config_path = os.path.join(train_dir, 'training_config.yaml')
    dataset_config_path = os.path.join(train_dir, 'dataset_config.yaml')

    model_write_yaml(updated_model_config, model_config_path)
    write_yaml(updated_training_config, training_config_path)
    write_yaml(updated_dataset_config, dataset_config_path)

    # 实际训练并获取指标
    best_metrics = train_model(config, model_config_path, training_config_path, dataset_config_path, device)
    


if __name__ == "__main__":
    # 用户输入的参数
    CUDA_VISIBLE_DEVICES = '3'  # 设置可用的 GPU
    base_train_dir = '/data/jiahaoguo/ultralytics/runs/ray_tune/tune_test'  # 根训练目录
    model_config_path = 'config/flownet/S/flowS_baseline3_start4.yaml'  # 模型配置文件路径
    training_config_path = 'config/train/orige_stream.yaml'  # 训练策略文件路径
    dataset_config_path = 'config/UAVTOD_[8,50].yaml'  # 数据集配置文件路径
    num_samples = 4  # 进行超参数搜索的样本数量
    max_t = 30  # ASHAScheduler 的最大训练epoch
    grace_period = 10  # ASHAScheduler 的宽限期，确保至少运行10个epoch
    reduction_factor = 2  # ASHAScheduler 的缩减因子

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    # 加载默认的 YAML 文件
    model_config = read_yaml(model_config_path)
    training_config = read_yaml(training_config_path)
    dataset_config = read_yaml(dataset_config_path)

    # 定义超参数搜索空间
    search_space = {
        'lr0': tune.uniform(0.005, 0.02),  # 连续浮点数  tune.randint(4, 32),  # 连续整数
        # 'hsv_h': tune.uniform(0.0, 0.1),  # (float) image HSV-Hue augmentation (fraction)
        # 'hsv_s': tune.uniform(0.5, 1.0),  # (float) image HSV-Saturation augmentation (fraction)
        # 'hsv_v': tune.uniform(0.3, 0.7),  # (float) image HSV-Value augmentation (fraction)
        # 'degrees': tune.uniform(0.0, 5.0),  # (float) image rotation (+/- deg)
        # 'translate': tune.uniform(0.0, 0.01),  # (float) image translation (+/- fraction)
        # 'scale': tune.uniform(0.0, 0.5),  # (float) image scale (+/- gain)
        # 'shear': tune.uniform(0.0, 2.0),  # (float) image shear (+/- deg)
        # 'perspective': tune.uniform(0.0, 0.0006),  # (float) image perspective (+/- fraction), range 0-0.001
        # 'flipud': tune.uniform(0.0, 0.5),  # (float) image flip up-down (probability)
        # 'fliplr': tune.uniform(0.0, 0.5),  # (float) image flip left-right (probability)
        # 'mosaic': tune.uniform(0.0, 0.0),  # (float) image mosaic (probability)
        # 'mixup': tune.uniform(0.0, 1.0),  # (float) image mixup (probability)
        # 'copy_paste': tune.uniform(0.0, 1.0),  # (float) segment copy-paste (probability)

        'split_length': tune.choice([[8, 20], [8, 50], [20, 40], [20, 80]]),  # 固定配置列表
        # dim搜索
        'VelocityNet_baseline3-0-(dim)': tune.choice([32, 64, 96, 128, 256]),
        # start_epoch搜索
        'VelocityNet_baseline3-1-(dim)': tune.choice([0,2,4,6,8]),
        # stride搜索
        'VelocityNet_baseline3-2-0-(stride1)': tune.choice([1,2]),
        'VelocityNet_baseline3-2-1-(stride2)': tune.choice([1,2,3]),
        'VelocityNet_baseline3-2-2-(stride3)': tune.choice([1,2,3]),
        # range搜索
        'VelocityNet_baseline3-3-0-(range1)': tune.choice([3,4,5]),
        'VelocityNet_baseline3-3-1-(range2)': tune.choice([3,4,5]),
        'VelocityNet_baseline3-3-2-(range3)': tune.choice([3,4,5]),

        "pretrain": "/data/jiahaoguo/ultralytics/yolov8s.pt",
        "batch_size": 22,
        'epochs': 24,  # 需要训练24个epoch
        'imgsz': 1024,  # 固定参数
        'workers': 6,  # 固定参数
    }

    # 定义调度器
    scheduler = ASHAScheduler(metric='mean_mAP50_95', mode='max', max_t=max_t, grace_period=grace_period, reduction_factor=reduction_factor)

    # 初始化 Ray
    ray.init(num_gpus=1, num_cpus=12, local_mode=False)

    # 运行超参数搜索
    analysis = tune.run(
        tune.with_parameters(train, base_train_dir=base_train_dir, model_config_path=model_config_path, training_config_path=training_config_path, dataset_config_path=dataset_config_path),
        config=search_space,
        resources_per_trial={'cpu': 6, 'gpu': 1},
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir = os.path.join(base_train_dir, "results")
    )

    # 输出最佳参数
    print("Best Parameters: ", analysis.best_config)
