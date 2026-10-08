import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile
import stat
import struct

import build_protocol_artifacts as builder

import score_mot as scorer
import score_paper_pair as pair_scorer


def map_data():
    return {"videos": [{"id": 7, "name": "a"}, {"id": 8, "name": "b"}],
            "images": [{"id": 100, "video_id": 7, "frame_index": 0},
                       {"id": 200, "video_id": 7, "frame_index": 1},
                       {"id": 300, "video_id": 8, "frame_index": 0}]}


def prediction(image=100, tid=42, score=0.9, category=1):
    return {"image_id": image, "track_id": tid, "paper_track_id": 999,
            "bbox": [1, 1, 10, 10], "score": score, "category_id": category}


class PairScorerTests(unittest.TestCase):
    def test_pair_profiles_and_combined_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt = root / "gt.zip"
            with zipfile.ZipFile(gt, "w") as archive:
                archive.writestr("a.txt", "1,42,1.00,1.00,10.00,10.00,1,-1,-1,-1\n")
            mapping_path = root / "mapping.json"
            mapping_path.write_text(json.dumps(map_data()))
            unified = root / "unified.json"
            botsort = root / "botsort.json"
            for path in (unified, botsort):
                path.write_text(json.dumps([prediction(score=0.29999)]))
            output = root / "paper_motchallenge.json"
            with contextlib.redirect_stdout(io.StringIO()):
                pair_scorer.main(["--gt-zip", str(gt), "--mapping-json", str(mapping_path),
                                  "--unified-json", str(unified), "--botsort-json", str(botsort),
                                  "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertFalse(report["historical_target_asserted"])
            self.assertEqual(report["rows"]["unified-raw"]["overall"]["Pred"], 1)
            self.assertEqual(report["rows"]["botsort-tau03"]["overall"]["Pred"], 0)
            self.assertEqual(report["rows"]["unified-raw"]["sources"]["gt"]["sha256"],
                             report["rows"]["botsort-tau03"]["sources"]["gt"]["sha256"])

    def test_pair_rejects_input_overwrite_and_same_prediction_path(self):
        base = ["--gt-zip", "gt.zip", "--mapping-json", "mapping.json",
                "--unified-json", "unified.json", "--botsort-json", "botsort.json"]
        with contextlib.redirect_stderr(io.StringIO()):
            for path in ("gt.zip", "mapping.json", "unified.json", "botsort.json"):
                with self.subTest(path=path), self.assertRaises(SystemExit) as error:
                    pair_scorer.main(base + ["--output", path])
                self.assertEqual(error.exception.code, 2)
            with self.assertRaises(SystemExit):
                pair_scorer.main(base + ["--botsort-json", "unified.json", "--output", "report.json"])


class ConversionTests(unittest.TestCase):
    def test_no_implicit_profile(self):
        with self.assertRaises(ValueError):
            scorer.convert_predictions([], map_data(), "common-tau")

    def test_filter_before_rounding_and_inclusive_boundary(self):
        records = [prediction(score=0.29999), prediction(200, score=0.3)]
        raw, rs = scorer.convert_predictions(records, map_data(), "unified-raw")
        filtered, fs = scorer.convert_predictions(records, map_data(), "botsort-tau03")
        self.assertEqual(len(raw["a"].splitlines()), 2)
        self.assertIn("1,42,", raw["a"])
        self.assertIn(",0.30,-1", raw["a"])
        self.assertEqual(filtered["a"].splitlines()[0].split(",")[:2], ["2", "42"])
        self.assertEqual(rs["retained_original_score_below_0_3"], 1)
        self.assertEqual(rs["retained_rounded_score_below_0_3"], 0)
        self.assertEqual(fs["score_filtered"], 1)

    def test_missing_score_only_raw_default(self):
        record = prediction()
        del record["score"]
        rows, stats = scorer.convert_predictions([record], map_data(), "unified-raw")
        self.assertIn(",1.00,-1", rows["a"])
        self.assertEqual(stats["missing_score_defaulted"], 1)
        with self.assertRaises(ValueError):
            scorer.convert_predictions([record], map_data(), "botsort-tau03")

    def test_ignore_and_class_agnostic_track_field(self):
        rows, stats = scorer.convert_predictions(
            [prediction(category=4), prediction(200, category=2)], map_data(), "unified-raw")
        self.assertEqual(stats["ignore_filtered"], 1)
        self.assertEqual(rows["a"].split(",")[:2], ["2", "42"])
        self.assertNotIn("999", rows["a"])

    def test_mapping_is_not_json_position(self):
        rows, _ = scorer.convert_predictions(
            [prediction(300)], map_data(), "unified-raw")
        self.assertEqual(rows["b"].split(",")[0], "1")

    def test_invalid_inputs(self):
        variants = [dict(prediction(), image_id=404),
                    dict(prediction(), video_id=8), dict(prediction(), track_id=1.5),
                    dict(prediction(), score=float("nan")),
                    dict(prediction(), bbox=[0, 0, -1, 2])]
        for record in variants:
            with self.subTest(record=record), self.assertRaises(ValueError):
                scorer.convert_predictions([record], map_data(), "unified-raw")
        with self.assertRaises(ValueError):
            scorer.convert_predictions([prediction(), prediction()], map_data(), "unified-raw")

    def test_mapping_duplicates_and_unsafe_name(self):
        data = map_data()
        data["images"].append(data["images"][0])
        with self.assertRaises(ValueError):
            scorer.mapping(data)
        data = map_data()
        data["videos"][0]["name"] = "../a"
        with self.assertRaises(ValueError):
            scorer.mapping(data)

    def test_gt_export_schema_and_ignore(self):
        data = {"videos": map_data()["videos"], "annotations": [
            {"video_id": 7, "frame_id": 0, "instance_id": 3,
             "bbox": [1.234, 2.345, 10, 10], "category_id": 1},
            {"video_id": 8, "category_id": 4}]}
        rows = scorer.convert_gt(data)
        self.assertEqual(set(rows), {"a"})
        self.assertEqual(rows["a"], "1,3,1.23,2.35,10.00,10.00,1,-1,-1,-1\n")

    def test_digest_basename_sorted_and_path_independent(self):
        self.assertEqual(scorer.content_set_sha256({"b.txt": b"b", "a.txt": b"a"}),
                         scorer.content_set_sha256({"a.txt": b"a", "b.txt": b"b"}))

    def test_minimal_mapping_has_only_requested_fields(self):
        data = map_data()
        data["videos"][0]["extra"] = "ignore"
        data["images"][0]["bbox"] = "ignore"
        minimal = builder.minimal_mapping(data)
        self.assertEqual(set(minimal), {"videos", "images"})
        self.assertEqual(set(minimal["videos"][0]), {"id", "name"})
        self.assertEqual(set(minimal["images"][0]), {"id", "video_id", "frame_index"})
        self.assertEqual(scorer.mapping(minimal), scorer.mapping(data))


class ZipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "gt.zip"
        self.raw = b"1,1,1,1,10,10,1,-1,-1,-1\n"

    def write_zip(self, members):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, raw in members:
                    archive.writestr(name, raw)

    def test_read_keeps_bytes_and_duplicates_without_extraction(self):
        self.write_zip([("a.txt", self.raw + self.raw)])
        before = self.path.read_bytes()
        with mock.patch.object(zipfile.ZipFile, "extract", side_effect=AssertionError("no extraction")), \
                mock.patch.object(zipfile.ZipFile, "extractall", side_effect=AssertionError("no extraction")):
            files, source = scorer.read_gt_zip(self.path)
        self.assertEqual(files, {"a.txt": self.raw + self.raw})
        self.assertEqual(source["members"], 1)
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(list(self.root.iterdir()), [self.path])
        self.assertEqual(builder.gt_statistics(files)["duplicate_frame_id_rows"], 1)

    def test_unsafe_member_names_rejected(self):
        for name in ["../a.txt", "/a.txt", "dir/a.txt", "dir\\a.txt", "C:a.txt", ".a.txt",
                     "a..txt", "a.json", "a.TXT", "a.txt/", "a\n.txt", "a\x00b.txt"]:
            with self.subTest(name=name):
                self.write_zip([(name, self.raw)])
                # ZipInfo truncates NUL when constructing archives, so the
                # written safe 'a' (without .txt) still fails the name rule.
                with self.assertRaises(ValueError):
                    scorer.read_gt_zip(self.path)

    def test_duplicate_and_case_colliding_names_rejected(self):
        for names in [("a.txt", "a.txt"), ("a.txt", "A.txt")]:
            self.write_zip([(name, self.raw) for name in names])
            with self.assertRaises(ValueError):
                scorer.read_gt_zip(self.path)

    def test_symlink_and_directory_attributes_rejected(self):
        for attr in [(stat.S_IFLNK | 0o777) << 16, (stat.S_IFDIR | 0o755) << 16, 0x10]:
            member = zipfile.ZipInfo("a.txt")
            member.create_system = 3
            member.external_attr = attr
            self.write_zip([(member, self.raw)])
            with self.assertRaises(ValueError):
                scorer.read_gt_zip(self.path)

    def test_empty_zip_or_member_rejected(self):
        for members in [[], [("a.txt", b"")]]:
            self.write_zip(members)
            with self.assertRaises(ValueError):
                scorer.read_gt_zip(self.path)

    def test_bounded_members_and_decompression(self):
        self.write_zip([("a.txt", self.raw), ("b.txt", self.raw)])
        for limit in ["GT_ZIP_MAX_FILES", "GT_ZIP_MAX_MEMBER_BYTES", "GT_ZIP_MAX_TOTAL_BYTES",
                      "GT_ZIP_MAX_ARCHIVE_BYTES"]:
            with mock.patch.object(scorer, limit, 1), self.assertRaises(ValueError):
                scorer.read_gt_zip(self.path)

    def test_invalid_zip_rejected(self):
        self.path.write_bytes(b"not a ZIP")
        with self.assertRaises(zipfile.BadZipFile):
            scorer.read_gt_zip(self.path)

    def test_nul_truncated_txt_name_rejected(self):
        self.write_zip([("a.txtXevil.txt", self.raw)])
        raw = self.path.read_bytes().replace(b"a.txtXevil.txt", b"a.txt\0evil.txt")
        self.path.write_bytes(raw)
        with self.assertRaises(ValueError):
            scorer.read_gt_zip(self.path)

    def test_encrypted_flag_rejected(self):
        self.write_zip([("a.txt", self.raw)])
        raw = bytearray(self.path.read_bytes())
        central = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, 6, struct.unpack_from("<H", raw, 6)[0] | 1)
        struct.pack_into("<H", raw, central + 8, struct.unpack_from("<H", raw, central + 8)[0] | 1)
        self.path.write_bytes(raw)
        with self.assertRaises(ValueError):
            scorer.read_gt_zip(self.path)

    def test_unsupported_compression_rejected(self):
        member = zipfile.ZipInfo("a.txt")
        member.compress_type = zipfile.ZIP_BZIP2
        self.write_zip([(member, self.raw)])
        with self.assertRaises(ValueError):
            scorer.read_gt_zip(self.path)

    def test_crc_corruption_rejected(self):
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("a.txt", self.raw)
        raw = self.path.read_bytes().replace(self.raw, self.raw.replace(b"10,10", b"11,10", 1))
        self.path.write_bytes(raw)
        with self.assertRaises(zipfile.BadZipFile):
            scorer.read_gt_zip(self.path)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = {"a.txt": b"1,1,1,1,10,10,1,-1,-1,-1\n2,1,1,1,10,10,1,-1,-1,-1\n"}

    def test_perfect_and_identity_switch(self):
        records = [prediction(), prediction(200)]
        result = scorer.evaluate(self.gt, records, map_data(), "unified-raw")
        self.assertEqual(result["overall"]["MOTA"], 1)
        self.assertEqual(result["overall"]["IDF1"], 1)
        records[1]["track_id"] = 43
        switched = scorer.evaluate(self.gt, records, map_data(), "unified-raw")
        self.assertEqual(switched["overall"]["IDsw"], 1)
        self.assertEqual(switched["overall"]["IDF1"], 0.5)

    def test_missing_prediction_penalized(self):
        result = scorer.evaluate(self.gt, [], map_data(), "unified-raw")
        self.assertEqual(result["overall"]["FN"], 2)
        self.assertEqual(result["overall"]["MOTA"], 0)
        self.assertEqual(result["pred_missing"], 1)
        self.assertEqual(result["missing_sequences"], ["a"])

    def test_overall_not_unweighted_video_mean(self):
        gt = {**self.gt, "b.txt": b"1,1,1,1,10,10,1,-1,-1,-1\n"}
        result = scorer.evaluate(gt, [prediction(), prediction(200)], map_data(), "unified-raw")
        self.assertAlmostEqual(result["overall"]["MOTA"], 2 / 3)
        self.assertAlmostEqual(result["overall"]["IDF1"], 4 / 5)
        self.assertNotEqual(result["overall"]["MOTA"], 0.5)

    def test_extra_video_is_reported_not_scored(self):
        result = scorer.evaluate(self.gt, [prediction(300)], map_data(), "unified-raw")
        self.assertEqual(result["unscored_prediction_sequences"], ["b"])
        self.assertEqual(result["overall"]["FP"], 0)

    def test_extra_frame_counts_as_fp(self):
        gt = {"a.txt": self.gt["a.txt"].splitlines(keepends=True)[0]}
        result = scorer.evaluate(gt, [prediction(), prediction(200)], map_data(), "unified-raw")
        self.assertEqual(result["overall"]["FP"], 1)
        self.assertEqual(result["overall"]["MOTA"], 0)

    def test_confidence_zero_gt_excluded(self):
        gt = {"a.txt": self.gt["a.txt"] + b"1,8,1,1,10,10,0,-1,-1,-1\n"}
        result = scorer.evaluate(gt, [prediction(), prediction(200)], map_data(), "unified-raw")
        self.assertEqual(result["overall"]["GT"], 2)

    def test_gt_duplicates_preserved_and_reported(self):
        gt = {"a.txt": self.gt["a.txt"] + self.gt["a.txt"].splitlines(keepends=True)[0]}
        result = scorer.evaluate(gt, [prediction(), prediction(200)], map_data(), "unified-raw")
        self.assertEqual(result["overall"]["GT"], 3)
        self.assertEqual(result["gt_duplicate_frame_id_rows"], {"a": 1})
        overall = result["overall"]
        self.assertAlmostEqual(overall["IDF1"], 2 * overall["IDTP"] / (overall["GT"] + overall["Pred"]))

    def test_rounding_changes_match(self):
        # IoU 0.5 boundary after conversion, not higher-precision JSON geometry.
        gt = {"a.txt": b"1,1,0,0,3,1,1,-1,-1,-1\n"}
        p = prediction()
        p["bbox"] = [1.004, 0, 3, 1]
        result = scorer.evaluate(gt, [p], map_data(), "unified-raw")
        self.assertEqual(result["overall"]["MOTA"], 1)
        p["bbox"][0] = 1.006
        result = scorer.evaluate(gt, [p], map_data(), "unified-raw")
        self.assertEqual(result["overall"]["FN"], 1)

    def test_cli_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_dir = root / "gt"
            gt_dir.mkdir()
            (gt_dir / "a.txt").write_bytes(self.gt["a.txt"])
            (root / "mapping.json").write_text(json.dumps(map_data()))
            (root / "pred.json").write_text(json.dumps([prediction(), prediction(200)]))
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            args = ["--gt-dir", str(gt_dir), "--mapping-json", str(root / "mapping.json"),
                    "--pred-json", str(root / "pred.json"), "--protocol", "unified-raw"]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                scorer.main(args)
            self.assertEqual(json.loads(output.getvalue())["overall"]["MOTA"], 1)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                scorer.main(args + ["--output", str(root / "pred.json")])

    def test_cli_gt_json_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = {"videos": map_data()["videos"], "annotations": [
                {"video_id": 7, "frame_id": frame, "instance_id": 1,
                 "bbox": [1, 1, 10, 10], "category_id": 1} for frame in [0, 1]]}
            for name, value in [("gt", gt), ("mapping", map_data()),
                                ("pred", [prediction(), prediction(200)])]:
                (root / (name + ".json")).write_text(json.dumps(value))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                scorer.main(["--gt-json", str(root / "gt.json"), "--mapping-json",
                             str(root / "mapping.json"), "--pred-json", str(root / "pred.json"),
                             "--protocol", "unified-raw", "--output", str(root / "result.json")])
            result = json.loads(output.getvalue())
            expected_gt = {"a.txt": b"1,1,1.00,1.00,10.00,10.00,1,-1,-1,-1\n"
                                     b"2,1,1.00,1.00,10.00,10.00,1,-1,-1,-1\n"}
            self.assertEqual(result["gt_content_set_sha256"], scorer.content_set_sha256(expected_gt))
            self.assertEqual(result, json.loads((root / "result.json").read_text()))
            self.assertEqual(result["overall"]["MOTA"], 1)

    def test_cli_requires_explicit_profile(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            scorer.main(["--gt-dir", "unused", "--mapping-json", "unused", "--pred-json", "unused"])
        self.assertEqual(error.exception.code, 2)

    def test_cli_gt_zip_parity_and_input_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_zip = root / "gt.zip"
            with zipfile.ZipFile(gt_zip, "w") as archive:
                archive.writestr("a.txt", self.gt["a.txt"])
            (root / "mapping.json").write_text(json.dumps(map_data()))
            (root / "pred.json").write_text(json.dumps([prediction(), prediction(200)]))
            args = ["--gt-zip", str(gt_zip), "--mapping-json", str(root / "mapping.json"),
                    "--pred-json", str(root / "pred.json"), "--protocol", "unified-raw"]
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()) as output:
                scorer.main(args)
            result = json.loads(output.getvalue())
            self.assertEqual(result["gt_content_set_sha256"], scorer.content_set_sha256(self.gt))
            self.assertEqual(result["overall"]["MOTA"], 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                scorer.main(args + ["--output", str(gt_zip)])


if __name__ == "__main__":
    unittest.main()
