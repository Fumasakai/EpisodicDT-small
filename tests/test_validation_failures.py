import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import torch
from src.train.common import evaluate_policy


class ValidationFailureTests(unittest.TestCase):
    def test_invalid_partial_and_complete_results(self):
        ok = {'mean_reward': 1., 'danger_success_rate': 1.}
        for outcomes, status, score in [([None, None], 'invalid', None),
                                        ([ok, None], 'partial', 1.),
                                        ([ok, ok], 'complete', 1.)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'validation'
                model = Mock()
                model.encode.return_value = torch.ones(2, 1)
                model.sample.return_value = (torch.ones(2, 1), None)
                model.decode.return_value = torch.ones(2, 1)
                with patch('src.train.common.targets', return_value=(torch.ones(2, 1), torch.ones(2, 1))), \
                     patch('src.train.common.batch_at', return_value={}), \
                     patch('src.train.common.evaluate_jobs', return_value=outcomes):
                    result = evaluate_policy(model, None, {'training': {'validation_runs': [101,102]}},
                                             None, root, [0,1])
                self.assertEqual(result['status'], status)
                self.assertEqual(result['reward'], score)
                self.assertEqual(json.loads((root/'summary.json').read_text()), result)
