import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import torch
from src.evaluation.generate_mobility import evaluate_generated


class GenerationParallelTests(unittest.TestCase):
    def test_concurrency_seed_runs_full_duration_and_order(self):
        barrier=threading.Barrier(4,timeout=5)
        def evaluate(points,target,mask,root,runs,seed,early):
            index=int(root.name.split('_')[-1])
            self.assertEqual(runs,[201,202,203]);self.assertEqual(seed,123)
            self.assertIs(early,False)
            self.assertEqual(float(points[0]),index)
            barrier.wait()
            return None if index==1 else {'index':index}
        with tempfile.TemporaryDirectory() as tmp:
            result=evaluate_generated(Mock(try_evaluate=evaluate),torch.arange(4).reshape(4,1),
                torch.zeros(4,2,3),torch.zeros(4,2,3),Path(tmp),[201,202,203],123,4)
        self.assertEqual(result,[{'index':0},None,{'index':2},{'index':3}])

    def test_worker_count_does_not_change_inputs_or_results(self):
        paths=torch.arange(8).reshape(4,2)
        def evaluate(points,*args):return points.tolist()
        sim=Mock(try_evaluate=evaluate)
        args=(sim,paths,torch.zeros(4,2,3),torch.zeros(4,2,3),Path('/unused'),[201],123)
        self.assertEqual(evaluate_generated(*args,1),evaluate_generated(*args,4))
        with self.assertRaises(ValueError):evaluate_generated(*args,0)
