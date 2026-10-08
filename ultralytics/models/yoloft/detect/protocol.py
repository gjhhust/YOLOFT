"""Official XS-VID evaluation and chronological video protocol."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics.utils.ops import xyxy2xywh


class XSVIDProtocolMixin:
    def init_paper_protocol(self, model):
        """Retain the protocol's original IDs and seven slots, including ignore."""
        if self.data.get("fake_video", False):
            raise ValueError("Paper validation requires chronological real-video streams, not fake_video resets")
        path = Path(self.data["eval_ann_json"])
        self.eval_json_path = path if path.is_absolute() else Path(self.data["path"]) / path
        with self.eval_json_path.open() as handle:
            self.gt_cocodata = json.load(handle)
        self.paper_images = {image["file_name"].replace("\\", "/"): image for image in self.gt_cocodata["images"]}
        if len(self.paper_images) != len(self.gt_cocodata["images"]):
            raise ValueError("Duplicate image names in paper ground truth")
        if len({image['id'] for image in self.paper_images.values()}) != len(self.paper_images):
            raise ValueError("Duplicate canonical image IDs in paper ground truth")
        self.image_name_map_id = {name: image["id"] for name, image in self.paper_images.items()}
        categories = {category["name"]: category["id"] for category in self.gt_cocodata["categories"]}
        names = model.names
        names = dict(enumerate(names)) if isinstance(names, list) else names
        if "classes_map" in self.data:
            self.class_map = dict(enumerate(self.data["classes_map"]))
        else:
            try:
                self.class_map = {index: categories[name] for index, name in names.items()}
            except KeyError as exc:
                raise ValueError(f"Model class name absent from paper categories: {exc}") from exc
        if len(self.class_map) != len(names) or any(value not in categories.values() for value in self.class_map.values()):
            raise ValueError("Paper category mapping does not cover all original class slots")
        if any(categories.get(name) != self.class_map[index] for index, name in names.items()):
            raise ValueError("Paper class mapping disagrees with the model's original class slots")
        if self.args.save_hybrid:
            raise ValueError("Paper evaluation forbids ground-truth injection via save_hybrid")
        self.args.save_json = True
        self.paper_seen_ids = set()
        self._stream_video = None
        self._stream_frame = None
        self._closed_stream_videos = set()

    def paper_image(self, filename):
        """Match a full protocol filename suffix, never a potentially ambiguous basename."""
        parts = str(filename).replace("\\", "/").split("/")
        for index in range(len(parts)):
            key = "/".join(parts[index:])
            if key in self.paper_images:
                return self.paper_images[key]
        raise ValueError(f"Image is not in paper protocol: {filename}")

    def stream_reset_required(self, batch):
        """Enforce chronological one-frame streaming and detect real video boundaries."""
        if "eval_ann_json" in self.data:
            image = self.paper_image(batch["im_file"][0])
            video, frame = image["video_id"], image["frame_id"]
            if image["id"] in self.paper_seen_ids:
                raise ValueError(f"Duplicate validation frame: {image['id']}")
            self.paper_seen_ids.add(image["id"])
        else:
            video = batch["sub_video_id"][0]
            frame = batch.get("frame_id", [None])[0]
        reset = video != self._stream_video or self.data.get("fake_video", False)
        if reset:
            if video in self._closed_stream_videos and not self.data.get("fake_video", False):
                raise ValueError(f"Validation video revisited after reset: {video}")
            if self._stream_video is not None:
                self._closed_stream_videos.add(self._stream_video)
        elif frame is not None and self._stream_frame is not None and frame <= self._stream_frame:
            raise ValueError(f"Non-chronological validation stream: video={video}, frame={frame}")
        self._stream_video, self._stream_frame = video, frame
        return reset

    def paper_pred_to_json(self, predn, filename):
        image = self.paper_image(filename)
        self.paper_seen_ids.add(image["id"])
        boxes = xyxy2xywh(predn[:, :4])
        boxes[:, :2] -= boxes[:, 2:] / 2
        if not torch.isfinite(predn).all():
            raise FloatingPointError(f"Non-finite validation prediction: {filename}")
        for prediction, box in zip(predn.tolist(), boxes.tolist()):
            self.jdict.append({"image_id": image["id"], "category_id": self.class_map[int(prediction[5])],
                               "bbox": [round(value, 3) for value in box], "score": round(prediction[4], 5)})

    def save_prediction_json(self, trainer=None):
        """Keep every evaluated epoch's predictions as well as the latest copy."""
        self.pred_json_path = self.save_dir / "predictions.json"
        serialized = json.dumps(self.jdict, allow_nan=False)
        self.pred_json_path.write_text(serialized)
        if trainer is not None:
            self.pred_json_path = self.save_dir / f"predictions_epoch{trainer.epoch + 1:04d}.json"
            self.pred_json_path.write_text(serialized)

    def accumulate_validation_loss(self, items):
        """Frame validation measures detection/motion, not training-only clip auxiliaries."""
        items = items.reshape(-1)
        if items.numel() > self.loss.numel():
            raise ValueError("Validation loss output exceeds the declared model.loss_names")
        if self.loss.numel() <= 4 and items.numel() != self.loss.numel():
            raise ValueError("Detection/motion validation loss cardinality differs from training loss")
        if not torch.isfinite(items).all():
            raise FloatingPointError("Non-finite validation loss")
        self.loss[:items.numel()] += items
        self.validation_frame_loss_items = items.numel()

    def evaluate_paper_json(self, stats):
        """Use the retained official XS-VID evaluator, never silently fall back to internal mAP."""
        from pycocotools.coco import COCO
        from ultralytics.data.cocoeval_xs_vid import COCOeval

        expected = {image["id"] for image in self.gt_cocodata["images"]}
        if self.paper_seen_ids != expected:
            raise ValueError(f"Incomplete paper validation: seen={len(self.paper_seen_ids)}, expected={len(expected)}")
        annotation = COCO(str(self.eval_json_path))
        if self.jdict:
            predictions = annotation.loadRes(str(self.pred_json_path))
        else:
            predictions = COCO()
            predictions.dataset = {"images": deepcopy(annotation.dataset["images"]),
                                   "categories": deepcopy(annotation.dataset["categories"]), "annotations": []}
            predictions.createIndex()
        evaluator = COCOeval(annotation, predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        def area_ap(evaluation, label):
            index = evaluation.params.areaRngLbl.index(label)
            values = evaluation.eval["precision"][:, :, :, index, evaluation.params.maxDets.index(100)]
            values = values[values > -1]
            return float(values.mean()) if values.size else -1.0

        paper = {"AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
                 "AP75": float(evaluator.stats[2]), "APtiny": area_ap(evaluator, "0-12"),
                 "AP_12_20": area_ap(evaluator, "12-20"), "AP_20_32": area_ap(evaluator, "20-32"),
                 "AP_small": area_ap(evaluator, "small"), "AP_medium": area_ap(evaluator, "medium"),
                 "AP_large": area_ap(evaluator, "large")}
        if getattr(self.args, "paper_eval_merged_fg", False):
            merged = COCOeval(annotation, predictions, "bbox")
            merged.params.useCats = 0
            merged.evaluate()
            merged.accumulate()
            merged.summarize()
            paper["mergedFG_AP"] = float(merged.stats[0])
        stats.update({f"paper/{key}": value for key, value in paper.items()})
        # Best-checkpoint fitness uses the paper metric; internal metrics remain separately named.
        stats["fitness"] = paper["AP"]
        record = {"evaluator": "ultralytics.data.cocoeval_xs_vid.COCOeval", "metrics": paper,
                  "ground_truth": str(self.eval_json_path),
                  "gt_sha256": hashlib.sha256(Path(self.eval_json_path).read_bytes()).hexdigest(),
                  "predictions": str(self.pred_json_path),
                  "pred_sha256": hashlib.sha256(self.pred_json_path.read_bytes()).hexdigest(),
                  "images": len(expected), "class_map": self.class_map,
                  "APtiny_definition": "Official XS-VID area label 0-12 (area range 0 to 144 pixels squared)",
                  "mergedFG_is_auxiliary_class_agnostic": bool(getattr(self.args, "paper_eval_merged_fg", False))}
        if self.training:
            record["frame_validation_loss_items_computed"] = getattr(self, "validation_frame_loss_items", 0)
            record["clip_only_auxiliary_validation_losses"] = "not computed; named slots remain zero, not paper metrics"
        self.pred_json_path.with_suffix(".paper_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
        return stats
