"""Save and CPU-audit a task checkpoint before declaring startup smoke success."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import torch

from tools.xsvid.checkpoint_utils import load_checkpoint


def audit_finite(value, label: str) -> int:
    """Check nested model/optimizer state, returning the number of CPU tensors."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"CPU checkpoint audit found non-CPU tensor: {label}")
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"Non-finite checkpoint tensor: {label}")
        return 1
    if isinstance(value, dict):
        return sum(audit_finite(item, f"{label}.{key}") for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(audit_finite(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Non-finite checkpoint scalar: {label}")
    return 0


def save_startup_checkpoint(model, optimizer, checkpoint: Path, *, args: dict,
                            epoch: int, step: int, loss: float, task: str, details: dict = None) -> dict:
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    receipt = checkpoint.parent / "startup_smoke.json"
    receipt.unlink(missing_ok=True)
    if step < 1 or not math.isfinite(loss):
        raise ValueError("Startup requires completed optimizer steps and finite loss")

    temporary = None
    receipt_temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=checkpoint.parent, prefix=".startup-", delete=False) as output:
            temporary = Path(output.name)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "args": args, "epoch": epoch, "step": step, "task": task}, output)
        saved = load_checkpoint(temporary)
        if not saved.get("model") or not saved.get("optimizer", {}).get("state"):
            raise ValueError("Startup checkpoint requires model tensors and initialized optimizer state")
        model_tensors = audit_finite(saved["model"], "model")
        optimizer_tensors = audit_finite(saved["optimizer"], "optimizer")
        if not model_tensors or not optimizer_tensors:
            raise ValueError("Startup checkpoint lacks model/optimizer tensors")
        os.replace(temporary, checkpoint)
        report = {"verification": "Task startup save/reload only, not paper retraining/reproduction",
                  "task": task, "checkpoint": str(checkpoint.resolve()), "epoch": epoch,
                  "optimizer_updates": step, "loss": loss, "cpu_reload_verified": True,
                  "finite_model_optimizer": True, "model_tensors": model_tensors,
                  "optimizer_tensors": optimizer_tensors, "details": details or {}}
        with tempfile.NamedTemporaryFile(mode="w", dir=receipt.parent, prefix=".receipt-", delete=False) as output:
            receipt_temporary = Path(output.name)
            json.dump(report, output, indent=2)
            output.write("\n")
        os.replace(receipt_temporary, receipt)
        return report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if receipt_temporary is not None:
            receipt_temporary.unlink(missing_ok=True)
