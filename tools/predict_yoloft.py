import os
from ultralytics.models import YOLOFT
from ultralytics.data.build import build_stream_dataloader,build_movedet_dataset
from torch.utils.data import DataLoader
from ultralytics.cfg import cfg2dict
import numpy as np
import cv2,os,json
import imageio
import os
import argparse
from tqdm import tqdm
import torch
import re
from PIL import Image
class DictWrapper:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

def get_imgMeta(image_path):
    # Input the image address, get the video name, frame number information and return it
    # example: input: /data/videos/video1/000032.png can get video_name:video1 frame_number:32
    # If your video images are named differently, you can modify the function
    video_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    frame_num_string = image_name.split('_')[-1] # Assuming the video name is the first part of the filename separated by '_'
    # Extract numeric parts using regular expressions
    match = re.search(r'\d+', frame_num_string)
    digits = match.group()
    frame_num = int(digits)
    img_metas = {
            "frame_number":frame_num,
            "video_name":video_name,
            "epoch":0, # dont change
        }
    return img_metas

def abspath_to_filename(abspath):
    # Input the full image address, 
    # transformed to assess the json in the images field file_name similar format, 
    # used to query the corresponding image_id
    return os.path.join(os.path.basename(os.path.dirname(abspath)), os.path.basename(abspath))


def pad_to_32_multiple(image):
    """
    Pads the input image so that both its width and height are multiples of 32.

    Parameters:
    - image: Original image (NumPy array)

    Returns:
    - padded_image: Image padded to dimensions that are multiples of 32 (NumPy array)
    - padding_info: Dictionary containing the number of pixels padded at the top, bottom, left, and right
    """
    height, width, _ = image.shape

    # Calculate the padding required to make dimensions multiples of 32
    pad_height = (32 - height % 32) % 32  # If height is already a multiple of 32, pad_height is 0
    pad_width = (32 - width % 32) % 32

    # Determine padding amounts for top, bottom, left, and right
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Pad the image with zeros (black pixels)
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Return the padded image and padding information
    padding_info = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    return padded_image, padding_info

def map_bbox_to_original(padding_info, padded_bbox):
    """
    Maps the detected bbox coordinates from the padded image back to the original image.

    Parameters:
    - padding_info: Dictionary containing the number of pixels padded at the top, bottom, left, and right
    - padded_bbox: Detected bbox on the padded image (x_min, y_min, x_max, y_max)

    Returns:
    - original_bbox: Mapped bbox coordinates on the original image (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = padded_bbox

    # Adjust coordinates using padding information to map back to the original image
    x_min_original = x_min - padding_info['left']
    y_min_original = y_min - padding_info['top']
    x_max_original = x_max - padding_info['left']
    y_max_original = y_max - padding_info['top']

    # Return the mapped bbox coordinates on the original image
    original_bbox = (x_min_original, y_min_original, x_max_original, y_max_original)
    return original_bbox


def get_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def get_first_png_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                return os.path.join(root, file)
    ValueError("no .png or .jpg files are found, return None")
    return None 

def image_paths_2_predictdatas(image_paths, eval_json=None):
    datas = []
    if eval_json:
        with open(eval_json, 'r') as f:
            coco_data = json.load(f)
        path_id_maps = {img["file_name"]:img["id"] for img in coco_data["images"]}
        
    for i, path in enumerate(image_paths):
        orige_img = cv2.imread(path)
        padding_img, padding_info = pad_to_32_multiple(orige_img)
        img = np.ascontiguousarray(padding_img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img).unsqueeze(0)
        img_metas = get_imgMeta(path)
        
        if i == 0:
            img_metas["is_first"] = True
        else:
            img_metas["is_first"] = False
            
        if eval_json:
            image_id = path_id_maps[abspath_to_filename(path)]
        else:
            image_id = 0
        datas.append({
            "im_file": [path], 
            "img": {
                "backbone":img,
                "img_metas":[img_metas],
            },
            "image_id": [image_id],
            "padding_info": padding_info
        })
    return datas 
    
def predict(args):  
    if args.mode == "one":
        video_dirs = [args.image_dir]
    else:
        video_dirs = [os.path.join(args.image_dir, video_name)  for video_name in os.listdir(args.image_dir)]
    
    print("\n!!!!!!!!!!!!!!!!!!!")
    print("Will test the applicability of the function by taking the full directory of an image to see if we can correctly get the [video name], [frame number], and the corresponding [id] number in the evaluation json.")
    input_path = get_first_png_file(video_dirs[0])
    file_name = abspath_to_filename(input_path)
    img_metas = get_imgMeta(input_path)
    print(f"input_path: {input_path}")
    print(f"file_name: {file_name} (check it exists in eval json if you need eval)")
    print(f"frame number: {img_metas['frame_number']}, video name: {img_metas['video_name']}")
    print("!!!!!!!!!!!!!!!!!!!\n")
    
    print(f"scan total video: {len(video_dirs)}")
    json_results = []
    os.makedirs(args.save_dir, exist_ok=True)
    for video_dir in tqdm(video_dirs):
        image_paths = get_image_paths(video_dir)
        predict_datas = image_paths_2_predictdatas(image_paths, args.eval_json)
        # Load a COCO-pretrained RT-DETR-l model
        model = YOLOFT(args.checkpoint)  # load a custom model
        model.model = model.model.cuda()
        results = model(predict_datas)

        first_im_array = predict_datas[0]["img"]["backbone"]
        _, layers, height, width = first_im_array.shape
        
        # 创建一个VideoWriter对象，用于保存MP4视频
        video_path = os.path.join(args.save_dir, os.path.basename(video_dir)+".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
        video_writer = cv2.VideoWriter(video_path, fourcc, 25, (width, height))
    
        for i,result in enumerate(results):
            ######################plot###################
            im_array = result.plot(line_width=1).astype(np.uint8)  # Get NumPy array in BGR format
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)  # Convert to RGB

            video_writer.write(im_rgb)
            
            ######################save pred###################
            if args.eval_json:
                for bbox_ in result.boxes:
                    bbox_xyxy = [round(float(x), 3) for x in bbox_.xyxy[0]]
                    bbox_xyxy = map_bbox_to_original(predict_datas[i]["padding_info"], bbox_xyxy)
                    bbox = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]
                    json_results.append({
                    'image_id': predict_datas[i]["image_id"][0],
                    'category_id': int(bbox_.cls[0])+1,
                    'bbox': bbox,
                    'bbox_xyxy': bbox_xyxy,
                    'score': round(float(bbox_.conf[0]), 3)})
            
        video_writer.release()
        print(f"predict video saved in: {video_path}")
    
    with open(os.path.join(args.save_dir, "predict.json"), 'w') as f:
        json.dump(json_results, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Process images for evaluation and prediction.")
    parser.add_argument('image_dir', type=str, help="Path to the directory containing images or subdirectories of images.")
    parser.add_argument('checkpoint', type=str, help="Path to the directory containing images or subdirectories of images.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory where prediction results and videos will be saved.")
    parser.add_argument('--mode', type=str, required=True, choices=['one', 'muti'], help="Mode of operation: 'one' if image_dir contains all images, 'muti' if image_dir contains subdirectories for each video.")
    parser.add_argument('--eval_json', type=str, help="Path to the evaluation JSON file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    predict(args)