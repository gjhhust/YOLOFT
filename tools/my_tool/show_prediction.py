import json
import os
import cv2
import numpy as np
from collections import defaultdict
import tqdm
import matplotlib.pyplot as plt
from pt_seq_nms import seq_nms, seq_nms_from_list
import torch

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def seq_nms_my(video_predictions, linkage_threshold=0.5, iou_threshold=0.5):
    for video_name, frames in tqdm.tqdm(video_predictions.items(),total=len(video_predictions)):
        boxes_list = []
        scores_list = []
        classes_list = []

        for frame_number, preds in sorted(frames.items()):
            boxes = []
            scores = []
            classes = []
            for pred in preds:
                x, y, w, h = pred['bbox']
                boxes.append([x, y, x + w, y + h])
                scores.append(pred['score'])
                classes.append(pred['category_id'])

            if boxes:
                boxes_list.append(torch.tensor(boxes, dtype=torch.float))
                scores_list.append(torch.tensor([scores], dtype=torch.float))
                classes_list.append(torch.tensor([classes], dtype=torch.int))

        if boxes_list:
            updated_scores_list = seq_nms_from_list(boxes_list, scores_list, classes_list, linkage_threshold, iou_threshold)

            for frame_number, updated_scores in zip(sorted(frames.keys()), updated_scores_list):
                preds = frames[frame_number]
                for score, pred in zip(updated_scores, preds):
                    pred['score'] = score.item()

        break

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def nms(predictions, nms_threshold=0.7, area_max=2000):
    if len(predictions) == 0:
        return predictions

    # 过滤掉面积大于 area_max 的预测框
    filtered_predictions = []
    for pred in predictions:
        bbox = pred['bbox']
        area = bbox[2] * bbox[3]  # 计算面积
        if area <= area_max:
            filtered_predictions.append(pred)
    
    if len(filtered_predictions) == 0:
        return filtered_predictions

    boxes = np.array([pred['bbox'] for pred in filtered_predictions])
    scores = np.array([pred['score'] for pred in filtered_predictions])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.2, nms_threshold=nms_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        return [filtered_predictions[i] for i in indices]
    else:
        return []
    
def get_category_colors(num_categories):
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, num_categories)]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors

def draw_predictions(image, predictions, category_colors, thickness=2):
    for pred in predictions:
        category_id = pred['category_id']
        bbox = pred['bbox']
        color = category_colors[category_id]
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

def process_videos(coco_data, predictions_file, image_dir, output_dir, fps=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取类别数量并生成颜色
    num_categories = len(coco_data['categories'])
    category_colors = get_category_colors(num_categories+1)
    print("category_colors len: ",len(category_colors))
    
    predictions = read_json(predictions_file)
    # 尝试从文件中读取video_predictions
    video_predictions_path = predictions_file.replace('predictions','video_predictions')
    if os.path.exists(video_predictions_path):
        video_predictions = read_json(video_predictions_path)
    else:
        # 将预测结果按视频分组
        video_predictions = defaultdict(lambda: defaultdict(list))
        for pred in tqdm.tqdm(predictions,total=len(predictions)):
            image_id = pred['image_id']
            image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
            if image_info:
                file_name = image_info['file_name']
                video_name = os.path.dirname(file_name)
                frame_number = int(os.path.basename(file_name).split('.')[0])
                video_predictions[video_name][frame_number].append(pred)
        write_json(video_predictions, video_predictions_path)

    # seq_nms_my(video_predictions)
    # write_json(video_predictions, video_predictions_path.replace('video_predictions','video_predictions_seqnms'))

    for video_name, frames in tqdm.tqdm(video_predictions.items(),total=len(video_predictions)):
        frames = sorted(frames.items())  # 按照frame_number排序
        images = []
        
        for frame_number, preds in tqdm.tqdm(frames):
            frame_number = int(frame_number)
            image_path = os.path.join(image_dir, video_name, f'{frame_number:07d}.png')
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    preds = nms(preds)
                    image = draw_predictions(image, preds, category_colors)
                    images.append(image)
        
        if images:
            height, width, _ = images[0].shape
            output_path = os.path.join(output_dir, f'{video_name}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for img in images:
                out.write(img)
            out.release()
            # exit(0)

# 示例使用
coco_test_file = '/data/jiahaoguo/dataset/UAVTOD_1/all_fix/coco/test.json'  # 替换为实际的 JSON 文件路径
predictions_file = '/data/jiahaoguo/ultralytics/runs/detect/val13/predictions.json'  # 替换为实际的预测结果文件路径
image_dir = '/data/jiahaoguo/dataset/UAVTOD_1/images'  # 替换为图像目录路径
output_dir = '/data/jiahaoguo/ultralytics/runs/UAVTOD_exper/baseline/baseline3/train94_36.2/prediction_videos/'  # 替换为输出视频目录

coco_data = read_json(coco_test_file)
process_videos(coco_data, predictions_file, image_dir, output_dir)
