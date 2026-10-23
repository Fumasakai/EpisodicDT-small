import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
import torch
from src.train.train_diffusion import evaluate_candidates
from src.simulation.parallel import evaluate_jobs


class DiffusionParallelTests(unittest.TestCase):
    def test_training_flags_order_and_failed_candidate(self):
        barrier = threading.Barrier(4, timeout=5)
        calls = []
        def evaluate(path, label, mask, root, runs, seed, early):
            calls.append((int(label[0]), root, runs, seed, early))
            barrier.wait()
            return None if int(label[0]) == 2 else int(label[0])
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_candidates(Mock(try_evaluate=evaluate), torch.zeros(4,1),
                torch.arange(4).reshape(4,1), torch.ones(4,1), [3,0,2,1], Path(tmp), [1],
                {'training': {'simulation_workers': 1},
                 'diffusion': {'simulation_workers': 4, 'stop_on_danger': True}}, training=True)
        self.assertEqual(result, [3,0,None,1])
        self.assertTrue(all(c[-1] and c[-2] is None for c in calls))
        self.assertEqual(len({c[1] for c in calls}),4)

    def test_validation_disables_early_stop_even_when_training_enabled(self):
        sim=Mock();sim.try_evaluate.return_value={'reward':1}
        evaluate_candidates(sim,torch.zeros(1,1),torch.zeros(1,1),torch.ones(1,1),[0],
            Path('/unused'),[101,102],{'training':{'simulation_workers':1},
                'diffusion':{'stop_on_danger':True}})
        self.assertFalse(sim.try_evaluate.call_args.args[-1])
        self.assertEqual(sim.try_evaluate.call_args.args[4],[101,102])

    def test_completion_progress_does_not_wait_for_first_job(self):
        release=threading.Event(); updates=[]
        def job(i):
            if i==0:
                if not release.wait(5):raise RuntimeError('progress waited for first job')
            return i
        def report(done,total):
            updates.append((done,total))
            if done==1:release.set()
        results=evaluate_jobs(Mock(try_evaluate=job),[(0,),(1,)],2,progress=report)
        self.assertEqual(results,[0,1])
        self.assertEqual(updates[0],(0,2));self.assertEqual(updates[-1],(2,2))
