"""Checkpoint helpers shared by the XS-VID legacy release tools."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import torch


def load_checkpoint(path: Path, *, tensor_only: bool = False):
    """Load trusted legacy payloads on new torch and pre-weights_only releases."""
    try:
        return torch.load(path, map_location="cpu", weights_only=tensor_only)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def checkpoint_state(path: Path, *, require_ema: bool = False, tensor_only: bool = False) -> dict[str, torch.Tensor]:
    """Return a plain state dictionary from supported legacy payloads."""
    checkpoint = load_checkpoint(path, tensor_only=tensor_only)
    if isinstance(checkpoint, dict):
        if checkpoint.get("task") in ("vid", "unified_mot", "unified_sot") and "model_yaml" in checkpoint:
            value = checkpoint.get("model")
            verify_release_payload(checkpoint, path)
            if require_ema and checkpoint.get("task") != "vid":
                raise ValueError("Detector initialization requires the released VID state")
        elif require_ema:
            value = checkpoint.get("ema")
            if value is None:
                raise ValueError("Release VID initialization requires EMA; no model fallback")
        else:
            value = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        value = checkpoint
    if hasattr(value, "state_dict"):
        value = value.state_dict()
    if not isinstance(value, dict) or not all(isinstance(item, torch.Tensor) for item in value.values()):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(value)!r}")
    return value


def tensor_digest(state):
    """SHA-256 of sorted key/shape/dtype/content descriptors, independent of pickle."""
    manifest = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor checkpoint value: {key}")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        manifest[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": hashlib.sha256(raw).hexdigest()}
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def verify_release_payload(checkpoint, path):
    manifest = json.loads(Path(__file__).with_name("release_manifest.json").read_text())
    task = checkpoint.get("task")
    filenames = {"vid": "vid.pt", "unified_mot": "unified_mot.pt", "unified_sot": "unified_sot.pt"}
    if task not in filenames or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Unsupported release task or state payload")
    expected = manifest["weights"][filenames[task]]
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected["file_sha256"]:
        raise ValueError("Release checkpoint file SHA-256 mismatch")
    state = checkpoint["model"]
    if len(state) != expected["tensor_count"] or tensor_digest(state) != expected["tensor_sha256"]:
        raise ValueError("Release checkpoint tensor SHA-256 mismatch")


def require_release_state(state):
    """Reject unhealthy tensors and legacy temporal kernels before loading."""
    for key, value in state.items():
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
    kernels = [value for key, value in state.items() if key.endswith("gru.depthwise_conv.weight")]
    if len(kernels) != 3 or any(value.ndim != 5 or value.shape[2] != 2 for value in kernels):
        raise ValueError("Release Conv3d checkpoint requires three temporal-kernel-2 tensors")


def load_release(module, path: Path, *, detector_only: bool = False):
    """Strict full inference, or exact EMA detector initialization with new heads.

    The tensor-only VID export is verified against pinned release identity.
    No index migration, shape filtering, or unverified VID fallback is applied.
    Only non-detector task-head keys may remain initialized during training.
    """
    state = checkpoint_state(path, require_ema=detector_only, tensor_only=True)
    require_release_state(state)
    target = module.state_dict()
    expected = {key for key in target if key.startswith("model.")} if detector_only else set(target)
    if set(state) != expected:
        raise ValueError(f"Checkpoint key mismatch: missing={sorted(expected - set(state))}, "
                         f"unexpected={sorted(set(state) - expected)}")
    for key, value in state.items():
        if value.shape != target[key].shape:
            raise ValueError(f"Checkpoint shape mismatch: {key}: {value.shape} != {target[key].shape}")
    incompatible = module.load_state_dict(state, strict=not detector_only)
    if incompatible.unexpected_keys or any(key in expected for key in incompatible.missing_keys):
        raise RuntimeError("Strict release checkpoint load failed")
    loaded = module.state_dict()
    if any(not torch.equal(loaded[key].cpu(), value.cpu()) for key, value in state.items()):
        raise RuntimeError("Checkpoint tensors changed during loading (dtype conversion or custom loader)")
    return len(state), len(state)


def require_finite_update(loss, parameters):
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError("Non-finite training loss")
    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in parameters):
        raise FloatingPointError("Non-finite training gradient")


def training_metadata(arguments):
    """Keep scalar hyperparameters portable; omit input/output paths."""
    excluded = {"data_root", "det_ckpt", "config", "out_dir", "train_root", "train_index",
                "transt_pretrain", "init_ckpt"}
    return {key: value for key, value in arguments.items() if key not in excluded
            and (isinstance(value, (str, int, float, bool)) or
                 isinstance(value, list) and all(isinstance(item, (int, float, str, bool)) for item in value))}


def save_vid_export(ema, path):
    state = {key: value.detach().cpu().clone() for key, value in ema.float().state_dict().items()}
    require_release_state(state)
    torch.save({"ema": state, "model": state}, path)


