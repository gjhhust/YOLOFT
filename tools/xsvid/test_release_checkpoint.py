"""Focused CPU regression tests; no data, extensions, downloads, or jobs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tools.xsvid.checkpoint_utils import load_release, require_release_state, tensor_digest, training_metadata


def detector_state():
    return {f"model.{i}.block.gru.depthwise_conv.weight": torch.ones(1, 1, 2, 1, 1)
            for i in range(3)}


class FakeModule:
    def __init__(self, head=False):
        self.state = detector_state()
        if head:
            self.state["embedding.weight"] = torch.zeros(1)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, strict):
        missing = list(set(self.state) - set(state))
        if strict and missing:
            raise RuntimeError("missing keys")
        self.state.update(state)
        return torch.nn.modules.module._IncompatibleKeys(missing, [])


class ReleaseCheckpointTests(unittest.TestCase):
    def test_ema_selection_and_random_head_preserved(self):
        model = FakeModule(head=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vid.pt"
            torch.save({"ema": detector_state(), "model": {"bad": torch.ones(1)}}, path)
            self.assertEqual(load_release(model, path, detector_only=True), (3, 3))
            self.assertEqual(model.state["embedding.weight"].item(), 0)

    def test_no_ema_fallback(self):
        with patch("tools.xsvid.checkpoint_utils.load_checkpoint", return_value={"model": detector_state()}):
            with self.assertRaisesRegex(ValueError, "requires EMA"):
                load_release(FakeModule(), Path("unused"), detector_only=True)

    def test_full_load_rejects_missing_head(self):
        with patch("tools.xsvid.checkpoint_utils.load_checkpoint", return_value={"model": detector_state()}):
            with self.assertRaisesRegex(ValueError, "key mismatch"):
                load_release(FakeModule(head=True), Path("unused"))

    def test_shape_mismatch(self):
        state = detector_state()
        state["model.0.block.gru.depthwise_conv.weight"] = torch.ones(2, 1, 2, 1, 1)
        with patch("tools.xsvid.checkpoint_utils.load_checkpoint", return_value={"model": state}):
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_release(FakeModule(), Path("unused"))

    def test_kernel1_rejected(self):
        state = detector_state()
        state["model.0.block.gru.depthwise_conv.weight"] = torch.ones(1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "kernel-2"):
            require_release_state(state)

    def test_nonfinite_rejected(self):
        state = detector_state()
        next(iter(state.values())).fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            require_release_state(state)

    def test_tensor_only_vid_export_initializes_without_ema_object(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vid.pt"
            state = detector_state()
            torch.save({"format": "yoloft-xsvid-fp32-state-dict", "task": "vid", "model_yaml": {}, "model": state}, path)
            manifest = {"weights": {"vid.pt": {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                "tensor_sha256": tensor_digest(state), "tensor_count": 3}}}
            with patch("tools.xsvid.checkpoint_utils.json.loads", return_value=manifest):
                self.assertEqual(load_release(FakeModule(head=True), path, detector_only=True), (3, 3))

    def test_tensor_digest_includes_key_shape_and_dtype(self):
        self.assertNotEqual(tensor_digest({"a": torch.ones(2)}), tensor_digest({"b": torch.ones(2)}))
        self.assertNotEqual(tensor_digest({"a": torch.ones(2)}), tensor_digest({"a": torch.ones(1, 2)}))

    def test_training_metadata_is_tensor_only_loader_safe(self):
        metadata = training_metadata({"data_root": Path("private-data"), "init_ckpt": "/private/model.pt",
                                      "epochs": 8, "intervals": [1, 2, 3], "feat_mode": "simple"})
        self.assertNotIn("data_root", metadata)
        self.assertNotIn("init_ckpt", metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trained.pt"
            torch.save({"model": detector_state(), "args": metadata}, path)
            load_release(FakeModule(), path)


if __name__ == "__main__":
    unittest.main()
