# YOLOFT-L

Video object detection with YOLOFT and multiple-object/single-object tracking
with Unified YOLOFT on XS-VID. Use `vid.pt`, `unified_mot.pt`, and
`unified_sot.pt` with the matching source release. The model package is
`lanlanlan23/YOLOFT-XSVID-v2-weights`; file and tensor SHA-256 values are in
`tools/xsvid/release_manifest.json`.

## Results

XS-VID benchmark: [TPAMI paper](https://doi.org/10.1109/TPAMI.2026.3741044).

| Task | YOLOFT-L |
| --- | --- |
| VID AP | 29.3 |
| Unified MOT TAO mAP | 31.5 |
| Unified SOT AUC / Precision@20 | 59.63 / 81.35 |

## Download

Dataset: [Hugging Face](https://huggingface.co/datasets/lanlanlan23/XS-VID-v2)
/ [ModelScope](https://modelscope.cn/datasets/lanlanlanrr/XS-VID-v2).
Models: [YOLOFT weights](https://huggingface.co/lanlanlan23/YOLOFT-XSVID-v2-weights).

```bash
python tools/xsvid/download_release.py --hub hf --component dataset --destination /path/to/packages
# ModelScope dataset mirror:
python tools/xsvid/download_release.py --hub ms --component dataset --destination /path/to/packages
python tools/xsvid/download_release.py --hub hf --component model --destination /path/to/packages
```

Both dataset commands verify archives and prepare the data interface.
Models download separately and contain only the three task weights.
See [Prerequisites](PREREQUISITES.md) for official auxiliary downloads and local conversion.

## Environment

Use Python 3.10, PyTorch 2.4.0+cu118 and torchvision 0.19.0+cu118.
[Setup commands](QUICKSTART.md#setup) install these exact CUDA wheels and
runtime dependencies using `config/environment/cu118.txt`, then build DCNv3
and sparse flow correlation against the local CUDA 11.8 toolkit.
No entry point automatically installs dependencies.
BoTSORT uses the bundled BoxMOT implementation. Its auxiliary
appearance model is OSNet x0.25 MSMT17, not the Unified MOT embedding head.
SOT head training uses the portable official TransT tensor-only pretrain.
The original TransT Python pickle is not required.

Set dataset, model and output paths, then run from the checkout:

```bash
export DATA=/path/to/XS-VID
export MODEL=/path/to/model-package
export OUT=/path/to/results
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="$OUT/config" MPLCONFIGDIR="$OUT/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python tools/xsvid/check_release.py --vid "$MODEL/vid.pt" \
  --mot "$MODEL/unified_mot.pt" --sot "$MODEL/unified_sot.pt"
python -m unittest tools.xsvid.test_release_checkpoint tools.xsvid.test_release_protocol
```

Use a prepared full-test data YAML with chronological real video streams,
seven original class slots, and the intended evaluation annotation. Category
4 is the ignore slot, separate from foreground classes. Detection's
paper evaluation uses `annotations/protocols/paper_detection_test.json`;
canonical evaluation uses `annotations/test.json`. Keep protocols distinct.

## VID

```bash
python tools/xsvid/train_vid.py --recipe config/recipes/vid.yaml --weights "$MODEL/vid.pt" \
  --data "$DATA/legacy_yoloft.yaml" --project "$OUT/train_vid" \
  --device 0
python tools/xsvid/eval_vid.py --config config/xsvid/yoloft-l-temporal.yaml \
  --weights "$MODEL/vid.pt" --data "$DATA/legacy_yoloft.yaml" \
  --cfg config/train/default.yaml --project "$OUT" --name vid --device 0
python tools/xsvid/score_detection.py \
  --gt "$DATA/annotations/protocols/paper_detection_test.json" \
  --pred "$OUT/vid/predictions.json" --output "$OUT/vid/metrics.json"
```

Inference defaults to FP32. Optional `--half` changes precision. The official
XS-VID evaluator validates full frame coverage, class mapping and time order.
VID training initializes from release weights on the full training partition,
with the test partition for validation. Use a writable training view.

## Unified MOT

```bash
python dump_omni_emb.py --data-root "$DATA" \
  --annotation "$DATA/annotations/test.json" --det-json "$OUT/vid/predictions.json" \
  --config config/xsvid/yoloft-l-temporal.yaml --ckpt "$MODEL/unified_mot.pt" \
  --out "$OUT/mot/dump.pkl" --shard 0/1 --det-floor .001 --roi-grid 5 --gpu 0
python tools/xsvid/track_botsort.py --data-root "$DATA" \
  --annotation "$DATA/annotations/test.json" --dump "$OUT/mot/dump.pkl" \
  --output "$OUT/mot/predictions.json"
python tools/xsvid/normalize_tao_tracks.py --input "$OUT/mot/predictions.json" \
  --output "$OUT/mot/predictions.json"
python tools/xsvid/score_tao.py --data-root "$DATA" --pred "$OUT/mot/predictions.json" \
  --track-field paper_track_id --output "$OUT/mot/metrics.json"
```

`paper_track_id` is the sparse paper-compatible mapping with `track_id`
fallback. Use `--track-field track_id` for canonical IDs, with a separate output.

## Unified SOT

```bash
python eval_sot_transt.py --ckpt "$MODEL/unified_sot.pt" --param fixed \
  --data-root "$DATA" --out-root "$OUT/sot" \
  --config config/xsvid/yoloft-l-temporal.yaml --feat-mode fuse918 \
  --z 128 --x 256 --win .2 --smooth .4 --gpu 0
python tools/xsvid/score_sot.py --data-root "$DATA" \
  --predictions "$OUT/sot/fixed" --output "$OUT/sot/metrics.json"
```

## BoTSORT + OSNet

This external tracking-by-detection pipeline consumes the same VID predictions.
It has no additional YOLOFT training configuration.

```bash
python tools/xsvid/track_botsort_reid.py --data-root "$DATA" \
  --annotation "$DATA/annotations/test.json" --det-json "$OUT/vid/predictions.json" \
  --reid-weights "$AUX/osnet_x0_25_msmt17.pt" \
  --output "$OUT/tbd/predictions.json" --device cuda:0
python tools/xsvid/normalize_tao_tracks.py --input "$OUT/tbd/predictions.json" \
  --output "$OUT/tbd/predictions.json"
python tools/xsvid/score_tao.py --data-root "$DATA" --pred "$OUT/tbd/predictions.json" \
  --track-field paper_track_id --output "$OUT/tbd/metrics.json"
python tools/xsvid/mot_protocol_tools/build_paper_gt.py \
  --annotation "$DATA/annotations/test.json" --output "$OUT/protocols/paper_motchallenge_gt.zip"
python tools/xsvid/mot_protocol_tools/score_paper_pair.py \
  --unified-json "$OUT/mot/predictions.json" --botsort-json "$OUT/tbd/predictions.json" \
  --gt-zip "$OUT/protocols/paper_motchallenge_gt.zip" \
  --mapping-json "$DATA/annotations/test.json" --output "$OUT/paper_motchallenge.json"
```

MOTChallenge scores Unified predictions without a score threshold and BoTSORT
predictions at tau=.3. The GT contains 423 duplicate frame/identity rows.
MOTA/IDF1 use MOTChallenge matching; TAO mAP uses the TAO evaluator.
`build_paper_gt.py` combines public annotations with the bundled 26 KB paper
identity mapping and verifies all 64 original TXT contents plus archive SHA-256
`c00b0e091902af2add831339867f05e52ddab5dfab7ccb9e00b357100605e34b`.
The generated protocol artifact is separate from canonical dataset annotations.

## Unified Training

Prepare the Unified training view from `annotations/train.json`.
The training view and full-test evaluation use separate annotations.
SOT initialization and external BoTSORT appearance weights are listed in
[Prerequisites](PREREQUISITES.md).

```bash
python tools/xsvid/prepare_unified_train_data.py --data-root "$DATA" \
  --annotation "$DATA/annotations/train.json"
python train_omni_embed.py --recipe config/recipes/unified_mot.yaml \
  --data-root "$DATA" --det-ckpt "$MODEL/vid.pt" --out-dir "$OUT/train_mot" --gpu 0
python train_sot_transt.py --recipe config/recipes/unified_sot.yaml \
  --data-root "$DATA" --det-ckpt "$MODEL/vid.pt" \
  --transt-pretrain "$AUX/transt-state-dict.pth" \
  --out-dir "$OUT/train_sot" --gpu 0
```

The VID tensors initialize both trainers exactly. MOT embedding heads are
randomly initialized with seed 0 and the detector is frozen. SOT starts from
TransT head pretraining and fine-tunes the backbone; `--lr-bb 0` means the head
learning rate, not a frozen backbone. `--init-ckpt` initializes weights only,
not optimizer/epoch state. One-step save/reload checks use `--epochs 1 --smoke 1`
with separate output directories. Keep the model,
preprocessing, runtime environment and evaluation configuration together.

## License

YOLOFT and the retained Ultralytics runtime are AGPL-3.0. Bundled BoTSORT,
operator sources and auxiliary pretraining assets retain their respective
licenses and copyright notices. No model weights or dataset media are included
in this source tree.
