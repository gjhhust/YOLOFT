# YOLOFT: [An Extremely Small Video Object Detection Benchmark](https://gjhhust.github.io/XS-VID/) Baseline

## :loudspeaker: Introduction
This is the official implementation of the baseline model for [XS-VID](https://gjhhust.github.io/XS-VID/) benchmark.

## :ferris_wheel: Dependencies
 - CUDA 11.7
 - Python 3.8
 - PyTorch 1.12.1(cu116)
 - TorchVision 0.13.1(cu116)
 - numpy 1.24.4

## :open_file_folder: Datasets
Our work is based on the large-scale extremely small video object detection benchmark **XS-VID**. Download the dataset(s) from corresponding links below.
- [Google drive]：[annotations](https://drive.google.com/file/d/1-MF_H6OnLL-6ZAHwmwTOdxIeKY9zqGO9/view?usp=sharing); [images(0-3)](https://drive.google.com/drive/folders/1EGTIWLCLUAlKfbq7KEeHqXL8PAyKHNQ_?usp=sharing); [images(4-5)](https://drive.google.com/drive/folders/1m7YL3XVDjmiiVEy_rY4gVr0tJxnn8e0Y?usp=sharing);
- [BaiduNetDisk]：[annotations and images](https://pan.baidu.com/s/1VXle03mUYpKtmp3xj6C4dA?pwd=yp5g);

Please choose a download method to download the annotations and all images. Make sure all the split archive files (e.g., `images.zip`, `images.z01`, `images.z02`, etc.) are in the same directory. Use the following command to extract them:

```bash
unzip images.zip
unzip annotations.zip
```
We have released several annotation formats to facilitate subsequent research and use, including COCO, COCOVID, YOLO

## 🛠️ Install
This repository is build on **[Ultralytics](https://github.com/ultralytics/ultralytics) 8.0.143**  which can be installed by running the following scripts. Please ensure that all dependencies have been satisfied before setting up the environment.
```
conda create --name ultr python=3.8
conda activate ultr
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/gjhhust/YOLOFT
cd YOLOFT
pip install -e .

cd ./ultralytics/nn/modules/ops_dcnv3
python setup.py build install

cd ../alt_cuda_corr_sparse
python setup.py build install
```

## :hourglass: Data preparation

If you want to use a custom video dataset for training tests, it needs to be converted to yolo format for annotation, and the dataset files are organized in the following format:

```
data_root_dir/               # Root data directory
├── test.txt                 # List of test data files, each line contains a relative path to an image file
├── train.txt                # List of training data files, each line contains a relative path to an image file
├── images/                  # Directory containing image files
│   ├── video1/              # Directory for image files of the first video
│   │   ├── 0000000.png      # First frame image file of the first video
│   │   └── 0000001.png      # Second frame image file of the first video
│   ├── video2/              # Directory for image files of the second video
│   │   └── ...              # More image files
│   └── ...                  # More video directories
└── labels/                  # Directory containing label files
    ├── video1/              # Directory for label files of the first video
    │   ├── 0000000.txt      # Label file for the first frame of the first video (matches the image file)
    │   └── 0000001.txt      # Label file for the second frame of the first video (matches the image file)
    ├── video2/              # Directory for label files of the second video
    │   └── ...              # More label files
    └── ...                  # More video directories
```


Note: The name of the image and the name of the label in yolo format must be the same, and the format is frameNumber.png, e.g. "0000001.png and 0000001.txt".


## 🚀 Training

### One training session
```
python tools/train_flowft.py
```
Multiple GPUs please change devices

### Repeat training several times for a model configuration yaml file
```
python tools/XSVID/yoloft_baseline.py
```
Parameters:
- repeats: repetition
- model_config_path: Model yaml
- pretrain_model: pre-training weight
- dataset_config_path: Dataset Profiles
- training_config_path: Training hypercamera configurations

Eventually, you will get a log file containing the results of all the repeated experiments, and you can analyze the log file to get the optimal results and the location where they are saved:
```
python tools/XSVID/analy_log.py path/to/xxxx.log
```

### Comparison test on multiple model configuration yaml files
```
python tools/XSVID/yoloft_conpresion.py
```
Parameters:
- repeats: Number of training repetitions for a single model file
- model_config_dir: Directory of model profiles to be compared

Eventually, we will get all the log files of the comparison experiments and a csv of all the experimental results in runs/logs, which can be further analyzed to get the optimal configuration of the experimental model:
```
python tools/XSVID/analy_csv.py path/to/xxxx.csv
```


## 📈 Evaluation
```
python tools/test_flowft.py
```


When **save_json=True**, evaluations in coco format will be output for training and testing, otherwise only Ultralytics' own results evaluations will be output

To evaluate the performance of other models, you can use the [eval tool](https://github.com/gjhhust/XS-VID) 

## :trophy: Result

### Result on XS-VID
| **Method**        | **Schedule** | **Backbone** | **$AP$** | **$AP_{50}$** | **$AP_{75}$** | **$AP_{eS}$** | **$AP_{rS}$** | **$AP_{gS}$** | **Inference(ms)** |
|:------------------|:-------------|:-------------|---------:|--------------:|--------------:|--------------:|--------------:|--------------:|-------------------:|
| DFF               | 1x           | R50          |      9.4 |           15  |          10.2 |           0.0 |           0.3 |           3.0 |               20.0 |
| DFF               | 1x           | x101         |      9.6 |           16.9|           9.9 |           0.0 |           0.5 |           4.5 |               25.5 |
| FGFA              | 1x           | R50          |      7.8 |           18.8|           5.0 |           1.1 |           2.0 |           6.1 |              151.0 |
| FGFA              | 1x           | x101         |     12.3 |           18.0|          14.1 |           0.2 |           1.1 |           6.4 |              181.8 |
| SELSA             | 1x           | R50          |     13.6 |           18.1|          15.5 |           0.0 |           2.2 |           8.1 |               88.5 |
| SELSA             | 1x           | x101         |     13.6 |           18.8|          15.8 |           0.0 |           1.7 |           8.3 |              110.0 |
| TROI              | 1x           | R50          |     12.3 |           16.9|          14.0 |           0.0 |           1.3 |           5.6 |              232.0 |
| TROI              | 1x           | x101         |     12.8 |           18.5|          14.7 |           0.0 |           1.3 |           7.6 |              285.7 |
| MEGA              | 1x           | R101         |      7.8 |           18.8|           5.0 |           1.1 |           2.0 |           6.1 |                nan |
| DiffusionVID      | 50e          | R101         |     10.6 |           24.3|           8.2 |           2.7 |           5.6 |           9.4 |                nan |
| TransVOD          | 50e          | R50          |     21.8 |           39.6|          21.1 |           8.8 |          13.6 |          20.5 |              136.0 |
| StreamYOLO        | 1x          | YOLOX        |     33.4 |           47.3|          37.5 |          18.7 |          26.7 |          33.6 |               47.5 |
| FCOS              | 1x           | R50          |     24.9 |           41.3|          24.8 |           7.7 |          17.3 |          22.6 |               31.8 |
| ATSS              | 1x           | R50          |     26.9 |           43.3|          26.8 |           8.4 |          19.2 |          23.9 |               34.9 |
| YOLOX-S           | 50e          | YOLOX        |     29.1 |           44.0|          30.4 |          15.0 |          20.0 |          25.6 |               24.0 |
| YOLOX-L           | 50e          | YOLOX        |     31.0 |           44.9|          33.8 |          17.4 |          21.7 |          25.6 |               37.4 |
| DyHead            | 1x           | R50          |     23.7 |           39.6|          22.7 |           7.0 |          15.9 |          20.5 |               98.0 |
| RepPoints         | 1x           | R50          |     23.7 |           41.7|          22.8 |           9.1 |          18.6 |          23.9 |               37.8 |
| Deformable-DETR   | 1x           | R50          |     21.3 |           38.0|          21.3 |          11.3 |          13.7 |          18.7 |               52.3 |
| Sparse RCNN       | 1x           | R50          |     21.0 |           34.2|          21.8 |           9.0 |          13.9 |          17.5 |               41.8 |
| Cascade RPN       | 1x           | R50          |     27.0 |           44.5|          26.6 |          13.5 |          19.4 |          22.1 |               45.3 |
| CESCE             | 15e          | nan          |     22.6 |           40.1|          21.5 |          10.3 |          16.2 |          21.3 |               31.0 |
| CFINet            | 1x           | R50          |     29.5 |           48.8|          31.0 |          16.6 |          21.8 |          25.1 |               47.1 |
| Yolov8-s          | 2x           | YOLOv8       |     30.0 |           45.3|          32.1 |          17.8 |          24.1 |          27.0 |               14.0 |
| Yolov8-L          | 2x           | YOLOv8       |     33.6 |           48.8|          36.9 |          21.3 |          27.4 |          32.7 |               26.0 |
| Yolov9-C          | 2x           | nan          |     31.6 |           47.0|          34.3 |          18.4 |          24.6 |          31.2 |               22.0 |
| YOLOFT-S    | 2x           | YOLOv8       |     32.9 |           49.2|          36.5 |          21.4 |          26.5 |          34.2 |               16.0 |
| YOLOFT-L   | 2x           | YOLOv8       |     36.4 |           52.9|          41.2 |          24.7 |          28.9 |          33.4 |               36.0 |

### Result on Visdrone2019 VID(test-dev)

| **Method** | **Schedule** | **Backbone** | **$AP$** | **$AP_{50}$** | **$AP_{75}$** | **$AP_{eS}$** | **$AP_{rS}$** | **$AP_{gS}$** | **$AP_{m}$** | **$AP_{l}$** |
|--------------------|--------------|--------------|---------|---------------|--------------|---------------|---------------|---------------|--------------|--------------|
| DFF                | 1x           | R50          | 5.8     | 12.2          | 4.9          | 0.0           | 0.2           | 1.1           | 6.9          | 12.4         |
| DFF                | 1x           | x101         | 10.3    | 20.8          | 9.1          | 0.0           | 0.1           | 3.4           | 13.6         | 21.8         |
| FGFA               | 1x           | R50          | 7.5     | 14.5          | 7.1          | 0.0           | 0.2           | 1.5           | 9.6          | 17.0         |
| FGFA               | 1x           | x101         | 13.6    | 29.2          | 10.5         | 0.0           | 0.9           | 6.3           | 17.8         | 28.5         |
| SELSA              | 1x           | R50          | 6.7     | 12.7          | 6.4          | 0.0           | 0.2           | 1.2           | 8.6          | 15.0         |
| SELSA              | 1x           | x101         | 11.8    | 23.0          | 11.1         | 0.0           | 0.5           | 2.7           | 14.3         | 30.2         |
| TROI               | 1x           | R50          | 7.9     | 15.9          | 7.0          | 0.0           | 0.2           | 1.5           | 10.3         | 16.3         |
| TROI               | 1x           | x101         | 12.0    | 23.9          | 10.4         | 0.0           | 0.1           | 4.8           | 16.6         | 24.7         |
| TransVOD           | 50e          | R50          | 9.7     | 21.1          | 8.0          | 1.0           | 3.2           | 4.9           | 11.5         | 23.8         |
| StreamYOLO         | 1x          | YOLOX        | 18.0    | 35.0          | 16.7         | 1.6           | 5.1           | 10.6          | 22.3         | 33.9         |
| FCOS               | 1x           | R50          | 12.4    | 24.6          | 11.5         | 1.3           | 3.1           | 4.8           | 13.8         | 30.6         |
| ATSS               | 1x           | R50          | 13.7    | 28.2          | 11.9         | 1.5           | 4.6           | 7.2           | 16.2         | 29.9         |
| YOLOX-S            | 50e          | YOLOX        | 7.8     | 17.0          | 6.4          | 1.6           | 3.5           | 5.6           | 10.4         | 12.8         |
| DyHead             | 1x           | R50          | 9.3     | 19.3          | 8.0          | 1.4           | 3.5           | 5.0           | 10.7         | 20.7         |
| RepPoints          | 1x           | R50          | 13.6    | 28.3          | 11.7         | 0.7           | 3.9           | 5.4           | 16.3         | 29.0         |
| Deformable-DETR    | 1x           | R50          | 9.8     | 20.2          | 8.4          | 2.5           | 3.7           | 5.1           | 11.9         | 19.5         |
| Sparse RCNN        | 1x           | R50          | 8.1     | 16.6          | 7.1          | 1.0           | 2.9           | 4.5           | 9.5           | 16.0         |
| Cascade RPN        | 1x           | R50          | 12.5    | 25.0          | 11.3         | 0.9           | 3.9           | 6.2           | 15.1          | 25.3         |
| CESCE              | 15e          | nan          | 11.0    | 23.4          | 9.3          | 1.7           | 3.5           | 4.4           | 13.0         | 23.8         |
| CFINet             | 1x           | R50          | 12.2    | 25.8          | 10.0         | 1.0           | 3.3           | 6.3           | 15.1         | 25.8         |
| Yolov8-s           | 2x           | YOLOv8       | 13.2    | 26.1          | 12.1         | 3.9           | 5.0           | 10.1          | 16.1         | 22.9         |
| Yolov8-L           | 2x           | YOLOv8       | 16.0    | 31.2          | 15.2         | 3.6           | 5.1           | 9.9           | 19.7         | 27.3         |
| Yolov9-C           | 2x           | nan          | 15.5    | 30.3          | 14.3         | 1.8           | 5.8           | 9.8           | 19.1         | 33.4         |
| YOLOFT-S     | 2x           | YOLOv8       | 14.8    | 29.4          | 13.6         | 4.4           | 6.1           | 10.8          | 16.4         | 26.2         |
| YOLOFT-L    | 2x           | YOLOv8       | 15.8    | 31.4          | 14.4         | 4.9           | 6.5           | 11.8          | 19.4         | 25.8         |

## 📚  Checkpoints

| Model    | Params (M) | FLOPs (G) | Inference (ms) | Dataset | Checkpoint |
|----------|------------|-----------|----------------|------------|------------|
| YOLOFT-L | 45.16      | 230.14    | 36             | XS-VID | [yoloft-L.pt](https://drive.usercontent.google.com/u/0/uc?id=1-SN7vTwEci0KjVKYJiX2jkhaK0rt2zWQ&export=download)|
| YOLOFT-S | 53.58      | 13.02     | 16             | XS-VID | [yoloft-S.pt](https://drive.usercontent.google.com/u/0/uc?id=1-Vnm92bicMIy8RskEmnBFf2ScTzNrMD7&export=download) |

## :e-mail: Contact
If you have any problems about this repo or XS-VID benchmark, please be free to contact us at gjh_hust@hust.edu.cn 😉
