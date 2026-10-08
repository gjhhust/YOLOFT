"""Regression tests for portable video split paths."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ultralytics.data.dataset import YOLOStreamDataset


class SplitPathsTest(unittest.TestCase):
    def test_relative_dotted_absolute_and_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'images'
            root.mkdir()
            split = Path(tmp) / 'split.txt'
            absolute = root / 'video' / 'c.jpg'
            split.write_text(f'video/a.jpg\n./video/b.jpg\n{absolute}\n\n')
            dataset = SimpleNamespace(images_dir=str(root), prefix='', fraction=1)
            result = YOLOStreamDataset.get_img_files(dataset, str(split))
            self.assertEqual(result, sorted(str(root / 'video' / name)
                                           for name in ('a.jpg', 'b.jpg', 'c.jpg')))

    def test_dotted_directory_is_not_replaced_inside_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / 'split.txt'
            split.write_text('./video./frame.jpg\n')
            dataset = SimpleNamespace(images_dir=tmp, prefix='', fraction=1)
            self.assertEqual(YOLOStreamDataset.get_img_files(dataset, str(split)),
                             [str(Path(tmp) / 'video.' / 'frame.jpg')])


if __name__ == '__main__':
    unittest.main()
