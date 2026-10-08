"""Dataset validation must not require optional plotting fonts or network."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from ultralytics.data.utils import check_det_dataset


class OfflineDatasetTest(unittest.TestCase):
    def test_local_dataset_without_fonts_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            config = root / 'data.yaml'
            config.write_text(yaml.safe_dump({
                'path': str(root), 'train': 'images', 'val': 'images',
                'test': 'images', 'names': {0: 'object'},
            }))
            with patch('ultralytics.utils.checks.check_font', side_effect=AssertionError('font lookup')), \
                 patch('ultralytics.utils.downloads.safe_download', side_effect=AssertionError('download')), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('network')):
                for _ in range(2):
                    data = check_det_dataset(str(config), autodownload=False)
                    self.assertEqual(Path(data['val']), root / 'images')
                    self.assertEqual(data['names'], {0: 'object'})


if __name__ == '__main__':
    unittest.main()
