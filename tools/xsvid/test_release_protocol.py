"""Focused protocol/kernel and release download-profile checks (CPU only)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tools.xsvid import download_release as release
from ultralytics.models.yoloft.detect.protocol import XSVIDProtocolMixin
from ultralytics.nn.modules.block import DepthSeparableConv3D
from ultralytics.models.yoloft.detect.val import DetectionValidator


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        gt = {"images": [{"id": 3, "file_name": "video/a.jpg", "video_id": 9, "frame_id": 1},
                         {"id": 4, "file_name": "video/b.jpg", "video_id": 9, "frame_id": 2}],
              "categories": [{"id": 0, "name": "car"}], "annotations": []}
        (self.root / "test.json").write_text(json.dumps(gt))
        self.validator = XSVIDProtocolMixin()
        self.validator.data = {"path": str(self.root), "eval_ann_json": "test.json", "classes_map": [0]}
        self.validator.args = SimpleNamespace(save_hybrid=False, save_json=True)
        self.validator.jdict = []
        self.validator.init_paper_protocol(SimpleNamespace(names={0: "car"}))

    def test_sequence_and_duplicate_rejection(self):
        self.assertTrue(self.validator.stream_reset_required({"im_file": ["/tmp/video/a.jpg"]}))
        self.assertFalse(self.validator.stream_reset_required({"im_file": ["/tmp/video/b.jpg"]}))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validator.stream_reset_required({"im_file": ["/tmp/video/b.jpg"]})

    def test_backwards_stream_rejected(self):
        self.validator.stream_reset_required({"im_file": ["video/b.jpg"]})
        with self.assertRaisesRegex(ValueError, "Non-chronological"):
            self.validator.stream_reset_required({"im_file": ["video/a.jpg"]})

    def test_exact_full_suffix(self):
        with self.assertRaisesRegex(ValueError, "not in paper protocol"):
            self.validator.paper_image("different/a.jpg")

    def test_class_mapping_not_silently_shifted(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.validator.init_paper_protocol(SimpleNamespace(names={0: "truck"}))

    def test_nonfinite_prediction_rejected(self):
        with self.assertRaises(FloatingPointError):
            self.validator.paper_pred_to_json(torch.tensor([[0, 0, 1, 1, float("nan"), 0]]), "video/a.jpg")

    def test_temporal_kernel_uses_both_inputs(self):
        torch.manual_seed(0)
        layer = DepthSeparableConv3D(1, 1, kernel_size=1, padding=0, temporal_kernel=2).eval()
        with torch.no_grad():
            layer.depthwise_conv.weight.fill_(1)
            layer.pointwise_conv.weight.fill_(1)
        previous = torch.ones(1, 1, 2, 2, requires_grad=True)
        current = torch.ones(1, 1, 2, 2, requires_grad=True)
        result = layer(previous, current)
        self.assertEqual(result.shape, current.shape)
        result.sum().backward()
        self.assertGreater(previous.grad.abs().sum().item(), 0)
        self.assertGreater(current.grad.abs().sum().item(), 0)

    def test_validator_nms_six_candidates_not_end2end(self):
        validator = SimpleNamespace(nc=7, end2end=False, lb=[],
                                    args=SimpleNamespace(conf=.001, iou=.7, single_cls=False,
                                                         agnostic_nms=False, max_det=300))
        prediction = torch.zeros(1, 11, 6)
        prediction[:, :4, :] = 10
        prediction[:, 6, :] = .9
        result = DetectionValidator.postprocess(validator, prediction)
        self.assertEqual(result[0].shape, (1, 6))
        self.assertEqual(result[0][0, 5].item(), 2)

    def test_release_download_filter_and_sha(self):
        records = [(name, None, 1) for name in ("vid.pt", "unified_mot.pt", "unified_sot.pt", "untracked.pt")]
        with patch.object(release, "list_files", return_value=records), patch.object(release, "fetch") as fetch:
            release.snapshot("hf", "owner/model", "model", "main", self.root,
                             model_hashes=release.RELEASE_HASHES)
        self.assertEqual(fetch.call_count, 3)
        for call in fetch.call_args_list:
            self.assertEqual(call.args[2], release.RELEASE_HASHES[call.args[1].name])

    def test_release_prepare_requires_every_pinned_asset(self):
        hashes = {}
        for name in release.RELEASE_HASHES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            hashes[name] = release.sha256(path)
        pretrain = {**release.TRANST_PRETRAIN, "size_bytes": 7}
        with patch.object(release, "RELEASE_HASHES", hashes), patch.object(release, "TRANST_PRETRAIN", pretrain):
            release.prepare_model(self.root, current_release=True)
            (self.root / "vid.pt").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                release.prepare_model(self.root, current_release=True)


if __name__ == "__main__":
    unittest.main()
