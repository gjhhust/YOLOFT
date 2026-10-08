import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

try:
    from .build_paper_gt import IDENTITY_SHA256, encode_zip, render_members
except ImportError:
    from build_paper_gt import IDENTITY_SHA256, encode_zip, render_members


class GroundTruthBuildTests(unittest.TestCase):
    def fixture(self):
        return {'videos': [{'id': 0, 'name': 'video_0'}],
                'images': [{'id': 9, 'video_id': 0, 'frame_index': 0}],
                'annotations': [{'image_id': 9, 'video_id': 0, 'category_id': 2,
                                  'track_id': 4, 'paper_track_id': 5, 'bbox': [1, 2, 3, 4]}]}

    def test_paper_identity_and_duplicates(self):
        data = self.fixture()
        data['annotations'] *= 2
        files, rows, duplicates = render_members(data, [[0, 5, 7]])
        self.assertEqual((rows, duplicates), (2, 1))
        self.assertEqual(files['video_0.txt'], b'1,7,1.00,2.00,3.00,4.00,1,-1,-1,-1\n' * 2)

    def test_ignore_and_no_input_mutation(self):
        data = self.fixture()
        data['annotations'][0]['category_id'] = 4
        before = json.dumps(data)
        self.assertEqual(render_members(data, []), ({}, 0, 0))
        self.assertEqual(json.dumps(data), before)

    def test_deterministic_zip(self):
        files = {'b.txt': b'b', 'a.txt': b'a'}
        raw = encode_zip(files)
        self.assertEqual(raw, encode_zip(dict(reversed(list(files.items())))))
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertEqual(archive.namelist(), ['a.txt', 'b.txt'])

    def test_identity_sha(self):
        raw = Path(__file__).with_name('paper_gt_identities.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), IDENTITY_SHA256)


if __name__ == '__main__':
    unittest.main()
