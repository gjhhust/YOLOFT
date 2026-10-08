# YOLOFT-L

VID with YOLOFT; MOT and SOT with Unified YOLOFT on XS-VID.

## Results

[XS-VID, IEEE TPAMI](https://doi.org/10.1109/TPAMI.2026.3741044).

| Task | Result |
| --- | --- |
| VID AP | 29.3 |
| Unified MOT TAO mAP | 31.5 |
| Unified SOT AUC / Precision@20 | 59.63 / 81.35 |

## Setup

Python 3.10, PyTorch 2.4.0+cu118, torchvision 0.19.0+cu118. Install from the
[official cu118 wheel index](https://pytorch.org/get-started/previous-versions/#v240):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt -c config/environment/cu118.txt
python -c "import torch, torchvision; assert torch.__version__ == '2.4.0+cu118'; assert torchvision.__version__ == '0.19.0+cu118'; assert torch.version.cuda == '11.8'"
```

CUDA operator builds use a CUDA 11.8 development toolkit with `nvcc`, GCC/G++ 11
and a visible NVIDIA GPU. Set `CUDA_HOME` to the local toolkit directory:

```bash
export CUDA_HOME=/path/to/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=gcc CXX=g++ MAX_JOBS=4
nvcc --version
python -c "import torch; assert torch.cuda.is_available()"
python -m pip install setuptools==72.1.0 wheel==0.44.0
python -m pip install --no-build-isolation --no-deps ./ultralytics/nn/modules/ops_dcnv3
python -m pip install --no-build-isolation --no-deps ./ultralytics/nn/modules/alt_cuda_corr_sparse
python -c "import DCNv3, alt_cuda_sparse_corr; print('CUDA operators imported')"
```

Set the data and model paths from the checkout:

```bash
export YOLO_AUTOINSTALL=false
export DATA=/path/to/packages/XS-VID-v2
export MODEL=/path/to/packages/YOLOFT-XSVID-v2-weights
export OUT=/path/to/results
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## Download

Data: [HF](https://huggingface.co/datasets/lanlanlan23/XS-VID-v2) /
[MS](https://modelscope.cn/datasets/lanlanlanrr/XS-VID-v2).
[Models](https://huggingface.co/lanlanlan23/YOLOFT-XSVID-v2-weights):
`vid.pt`, `unified_mot.pt`, `unified_sot.pt`.

```bash
python tools/xsvid/download_release.py --hub hf --component dataset --destination /path/to/packages
# Alternative dataset mirror:
python tools/xsvid/download_release.py --hub ms --component dataset --destination /path/to/packages
python tools/xsvid/download_release.py --hub hf --component model --destination /path/to/packages
```

## VID Train/Test

The downloader prepares `legacy_yoloft.yaml`; `config/xsvid/data.yaml` is the
equivalent data configuration template.

```bash
python tools/xsvid/train_vid.py --recipe config/recipes/vid.yaml --data "$DATA/legacy_yoloft.yaml" --weights "$MODEL/vid.pt" --project "$OUT/train_vid" --device 0
python tools/xsvid/eval_vid.py --config config/xsvid/yoloft-l-temporal.yaml --weights "$MODEL/vid.pt" --data "$DATA/legacy_yoloft.yaml" --cfg config/train/default.yaml --project "$OUT" --name vid --device 0
python tools/xsvid/score_detection.py --gt "$DATA/annotations/protocols/paper_detection_test.json" --pred "$OUT/vid/predictions.json" --output "$OUT/vid/metrics.json"
```

## Unified MOT Train/Test

Prepare a writable training view from `train.json`; keep test annotations unchanged.

```bash
python tools/xsvid/prepare_unified_train_data.py --data-root "$DATA" --annotation "$DATA/annotations/train.json"
python train_omni_embed.py --recipe config/recipes/unified_mot.yaml --data-root "$DATA" --det-ckpt "$MODEL/vid.pt" --out-dir "$OUT/train_mot" --gpu 0
python dump_omni_emb.py --data-root "$DATA" --annotation "$DATA/annotations/test.json" --det-json "$OUT/vid/predictions.json" --ckpt "$MODEL/unified_mot.pt" --out "$OUT/mot/dump.pkl" --shard 0/1 --det-floor .001 --roi-grid 5 --gpu 0
python tools/xsvid/track_botsort.py --data-root "$DATA" --annotation "$DATA/annotations/test.json" --dump "$OUT/mot/dump.pkl" --output "$OUT/mot/predictions.json"
python tools/xsvid/normalize_tao_tracks.py --input "$OUT/mot/predictions.json" --output "$OUT/mot/predictions.json"
python tools/xsvid/score_tao.py --data-root "$DATA" --pred "$OUT/mot/predictions.json" --track-field paper_track_id --output "$OUT/mot/metrics.json"
```

## Unified SOT Train/Test

SOT training requires official TransT initialization. Follow
[Prerequisites](docs/PREREQUISITES.md) and set `AUX` to the local auxiliary directory.
SOT inference does not require TransT pretrained files.

```bash
python train_sot_transt.py --recipe config/recipes/unified_sot.yaml --data-root "$DATA" --det-ckpt "$MODEL/vid.pt" --transt-pretrain "$AUX/transt-state-dict.pth" --out-dir "$OUT/train_sot" --gpu 0
python eval_sot_transt.py --ckpt "$MODEL/unified_sot.pt" --param fixed --data-root "$DATA" --out-root "$OUT/sot" --feat-mode fuse918 --z 128 --x 256 --win .2 --smooth .4 --gpu 0
python tools/xsvid/score_sot.py --data-root "$DATA" --predictions "$OUT/sot/fixed" --output "$OUT/sot/metrics.json"
```

Full protocols, CPU checks, checkpoint exports and external BoTSORT+OSNet:
[commands and configuration](docs/REPRODUCTION.md).
Each task recipe supplies training defaults; explicit CLI values override them.
Use `--print-config` with the same required input/output arguments to inspect resolved settings.

## License

AGPL-3.0; retained Ultralytics, BoTSORT and CUDA-operator notices apply.
Auxiliary pretrained assets retain their original licenses. Models and media
are external downloads, not included in this source tree.
