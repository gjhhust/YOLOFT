import flow_viz
import cv2
import numpy as np
import os,tqdm

import torch
import torch.nn.functional as F

from ultralytics.utils.plotting import feature_visualization

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_heatmap(feature_map, save_video_size, original_image, filename=None, save=False):
    # Resize feature map
    feature_map = F.interpolate(torch.tensor(feature_map), size=save_video_size, mode="bilinear", align_corners=True).numpy()

    # Average across channels
    avg_feature_map = np.mean(feature_map[0].transpose(1, 2, 0), axis=-1)
    
    # Normalize to [0, 255]
    heatmap = np.uint8(255 * (avg_feature_map - avg_feature_map.min()) / (avg_feature_map.max() - avg_feature_map.min()))
    
    # Apply colormap
    cmap = plt.cm.get_cmap('coolwarm')
    heatmap_color = (cmap(heatmap) * 255).astype(np.uint8)
    
    # Convert to BGR format
    heatmap_color_bgr = cv2.cvtColor(heatmap_color, cv2.COLOR_RGBA2BGR)
    
    # Resize original image to match save_video_size
    # original_image_resized = cv2.resize(original_image, save_video_size)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_image, 0.8, heatmap_color_bgr, 0.5, 0)
    
    # Save or return the overlay image
    if save and filename:
        cv2.imwrite(filename, overlay)
        return heatmap_color_bgr
    else:
        return heatmap_color_bgr

def upflow(flow, new_size, mode='bilinear'):
    rate = max(new_size[0] / flow.shape[2], new_size[1] / flow.shape[3])
    return  rate * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def viz(flo, save_dir=None):
    flo = flo[0].permute(1,2,0).numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    if save_dir:
        save_path = os.path.join(save_dir, "flow.png")
        cv2.imwrite(save_path, flo[:, :, [2,1,0]])
    return flo[:, :, [2,1,0]]

import os
import numpy as np
from collections import defaultdict

def load_and_sort_files(directory):
    # 创建一个字典来存储每个 level_i 的文件列表
    files_dict = defaultdict(list)

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.startswith('level_') and filename.endswith('.npy'):
            parts = filename.split('_')
            level_i = int(parts[1])  # 提取 i
            frame_number = int(parts[2].split('.')[0])  # 提取 frame_number
            
            # 将文件名添加到对应 level_i 的列表中
            files_dict[level_i].append((frame_number, os.path.join(directory,filename)))

    # 对每个 level_i 的文件列表按照 frame_number 进行排序
    for level_i in files_dict:
        files_dict[level_i].sort()
        files_dict[level_i] = [f[1] for f in files_dict[level_i]]
    return files_dict

import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_field(velocity):
    H, W, _ = velocity.shape
    Y, X = np.mgrid[0:H, 0:W]

    # 提取速度向量的分量
    U = velocity[:, :, 1]
    V = velocity[:, :, 0]
    C = np.sin(U)
    fig, ax = plt.subplots(dpi=600)
    # 绘制箭头
    q = ax.quiver(X, Y, U, V, C)
    # 该函数绘制一个箭头标签在 (X, Y) 处， 长度为U, 详见参考资料4
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                label='Quiver key, length = 10', labelpos='E')

    plt.savefig("seepd.svg", format="svg")

# 假设您有一个名为 velocity 的速度场，其维度为 [H, W, 2]
# 这里使用随机数据进行示例
# H, W = 20, 20  # 假设图像大小为 20x20
# velocity = np.ones(H, W, 2)*1
# # 绘制速度场
# plot_velocity_field(velocity)

# 定义视频参数
width = 1024
height = 1024
fps = 25

image_dir = "/data/jiahaoguo/dataset/UAVTOD_1/images"
file_dir = "runs/UAVTOD_exper/baseline/baseline3/train94_36.2/save_tansor/UAVTOD_DJI0824Part5_0"

flow_npy_flies = load_and_sort_files(os.path.join(file_dir, "flows"))
video_name = os.path.basename(file_dir)
# for level_i in flow_npy_flies:
#     file_list = flow_npy_flies[level_i]
#     # if level_i!=2:
#     #     continue
#     # # 创建 VideoWriter 对象
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码器
#     out = cv2.VideoWriter(os.path.join(file_dir, f'{video_name}_flow_{level_i}.avi'), fourcc, fps, (width, height))
#     for file_path in tqdm.tqdm(file_list):
#         flow = np.load(file_path)
#         flow = upflow(torch.tensor(flow), (width, height))
#         flow_image = viz(flow)
#         if flow_image.shape[:2] != (height, width):
#             print(f"Error: Combined frame dimensions are incorrect: {flow_image.shape}")
#             continue
#         out.write(flow_image)
#     out.release()


feature_new_npy_flies = load_and_sort_files(os.path.join(file_dir, "feature_new"))
feature_fused_npy_flies = load_and_sort_files(os.path.join(file_dir, "feature_fused"))
for level_i in feature_new_npy_flies:
    new_list = feature_new_npy_flies[level_i]
    fused_list = feature_fused_npy_flies[level_i] 
    if level_i!=0:
        continue
    # # 创建 VideoWriter 对象
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码器
    # out = cv2.VideoWriter(os.path.join(file_dir, f'{video_name}_feature_{level_i}.avi'), fourcc, fps, (width*2, height))
    for frame_number, (new_path, fused_path) in tqdm.tqdm(enumerate(zip(new_list, fused_list)),total=len(fused_list)):
        orige_image = os.path.join(image_dir, video_name, f"{frame_number:07d}.png")
        fmaps0_old = np.load(new_path)
        fmaps0_new = np.load(fused_path)
        orige_image = cv2.imread(orige_image)
        fmaps0_old = save_heatmap(fmaps0_old, (width, height), orige_image, None, False)
        fmaps0_new = save_heatmap(fmaps0_new, (width, height), orige_image, None, False)
        combined_frame = np.concatenate([fmaps0_old, fmaps0_new], axis=1)
        if combined_frame.shape[:2] != (height, width * 2):
            print(f"Error: Combined frame dimensions are incorrect: {combined_frame.shape}")
            continue
        os.makedirs(os.path.join(file_dir, video_name+f"_feature_{level_i}"), exist_ok=True)
        cv2.imwrite(os.path.join(file_dir, video_name+f"_feature_{level_i}", f"{frame_number:07d}_f.png"), combined_frame)
        # cv2.imwrite(os.path.join(file_dir, video_name+f"_feature_{level_i}", f"{frame_number:07d}_o.png"), combined_frame)
        # out.write(combined_frame)
        
    # out.release()
