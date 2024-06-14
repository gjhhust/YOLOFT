## install
**cuda 11.1**
```bash
conda create --name ultr python=3.8
conda activate ultr
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
git clone -b exp-rt https://github.com/ultralytics/ultralytics 
cd ultralytics
pip install -e .

cd ./ultralytics/nn/modules/ops_dcnv3
python setup.py build install

cd ../alt_cuda_corr_sparse
python setup.py build install
```
## data prepare
data_root_dir:
--test.txt
--train.txt
--images
----video1
--------video1_0.png
--------video1_1.png
----video2
--labels
----video1
--------video1_0.txt(名字和图片要要一致)
--------video1_1.txt
----video2

注意：图片名和yolo格式的标注名称必须一致，且格式为videoName_frameNumber.png，例如"sky_0001.png和sky_0001.txt"
如果是别的格式可以修改ultralytics/data/dataset.py代码454行


## train
```bash
python tools_/train_movedet_L_flow.py
```
## eval

```bash
python tools_/test_movedet_L.py
```
