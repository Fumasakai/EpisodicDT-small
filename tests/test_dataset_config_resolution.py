import tempfile
import unittest
from pathlib import Path
from src.train.common import dataset_training_config


class DatasetConfigResolutionTests(unittest.TestCase):
    def test_current_and_fresh_dataset_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):dataset_training_config(root)
            (root/'training.yaml').write_text('')
            self.assertEqual(dataset_training_config(root),root/'training.yaml')
            (root/'training_fast.yaml').write_text('')
            self.assertEqual(dataset_training_config(root),root/'training_fast.yaml')
            (root/'training.yaml').unlink()
            self.assertEqual(dataset_training_config(root),root/'training_fast.yaml')
