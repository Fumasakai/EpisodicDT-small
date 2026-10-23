import copy
from pathlib import Path
import tempfile
import unittest
import torch
from src.train.train_mobility import initialize_weights, train


class WeightInitializationTests(unittest.TestCase):
    def test_imports_weights_without_reusing_validation_or_optimizer(self):
        model = torch.nn.Linear(2, 1)
        saved_model = torch.nn.Linear(2, 1)
        common = {'format': 'test', 'model_kwargs': {'in': 2}, 'provenance': {'schema': [1,2]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'latest.pt'
            torch.save({**common, 'model': saved_model.state_dict(), 'update': 119,
                        'build': {'binary': 'old'}, 'optimizer': {}, 'validation_best_reward': 1.}, path)
            info = initialize_weights(model, path, {**common, 'build': {'binary': 'new'}})
            torch.testing.assert_close(model.weight, saved_model.weight)
            self.assertEqual(info['source_update'], 119)
            self.assertFalse(info['optimizer_restored'])
            self.assertFalse(info['validation_reused'])
            self.assertEqual(info['source_build'], {'binary': 'old'})
            wrong = copy.deepcopy(common)
            wrong['provenance']['schema'] = [2,1]
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                initialize_weights(model, path, wrong)

    def test_resume_and_initialization_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            train({}, resume=True, initialize_from='checkpoint.pt')
