# Prerequisites

The model download contains only `vid.pt`, `unified_mot.pt`, and `unified_sot.pt`.
Unified MOT uses its own embedding head. SOT inference uses its trained head;
neither needs the following external pretrained files.

## SOT Training

Obtain `transt.pth` from the [official TransT model folder](https://drive.google.com/drive/folders/1GVQV1GoW-ttDJRRqaVAtLUtubtgLhWCE),
linked by the [authors](https://github.com/chenxin-dlut/TransT). Convert locally
before training; all 430 tensors, including the required 170 head tensors, are preserved.

```bash
export AUX=/path/to/auxiliary
git clone https://github.com/chenxin-dlut/TransT "$AUX/TransT"
python -m gdown --folder 'https://drive.google.com/drive/folders/1GVQV1GoW-ttDJRRqaVAtLUtubtgLhWCE' -O "$AUX/official"
python tools/xsvid/convert_transt.py --source "$AUX/official/transt.pth" --upstream-source "$AUX/TransT" --output "$AUX/transt-state-dict.pth"
```

The converter rejects files whose SHA-256 is not
`b4460d6fa4e3b4ab0bc119602953e3cacdfa770907e13ff1398f652145cd8fc0`
before loading the original pickle. Pass `--transt-pretrain "$AUX/transt-state-dict.pth"`
to SOT training. Do not disable this initialization.

## External BoTSORT+OSNet

Download OSNet x0.25 MSMT17 directly from the
[official model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html).
No conversion is needed.

```bash
python -m gdown 'https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF' -O "$AUX/osnet_x0_25_msmt17.pt"
printf '%s  %s\n' 6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99 "$AUX/osnet_x0_25_msmt17.pt" | sha256sum -c -
```

Pass `--reid-weights "$AUX/osnet_x0_25_msmt17.pt"` to external BoTSORT evaluation.
These auxiliary weights are not redistributed in the YOLOFT model package.
