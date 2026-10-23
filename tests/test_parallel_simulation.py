import threading
import unittest
from unittest.mock import Mock

from src.simulation.parallel import evaluate_jobs
from src.train.train_mobility import resume_compatible


class ParallelSimulationTests(unittest.TestCase):
    def test_parallel_workers_preserve_order_and_invalid_results(self):
        barrier = threading.Barrier(4, timeout=5)
        lock = threading.Lock()
        active = 0
        peak = 0
        def evaluate(i):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1
            return None if i == 3 else i * 2
        result = evaluate_jobs(Mock(try_evaluate=evaluate), [(i,) for i in range(8)], 4)
        self.assertEqual(result, [0, 2, 4, None, 8, 10, 12, 14])
        self.assertEqual(peak, 4)

    def test_unexpected_errors_are_not_silently_masked(self):
        sim = Mock()
        sim.try_evaluate.side_effect = ValueError('bad input')
        with self.assertRaisesRegex(ValueError, 'bad input'):
            evaluate_jobs(sim, [(1,), (2,)], 2)

    def test_sequential_fallback_and_empty_batch(self):
        sim = Mock()
        sim.try_evaluate.side_effect = [None, {'mean_reward': 1.}]
        self.assertEqual(evaluate_jobs(sim, [(1,), (2,)], 1), [None, {'mean_reward': 1.}])
        self.assertEqual(evaluate_jobs(sim, [], 4), [])
        for workers in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                evaluate_jobs(sim, [], workers)

    def test_concurrency_can_change_on_resume(self):
        resume_compatible({'training': {'updates': 2}},
                          {'training': {'updates': 2, 'simulation_workers': 8}})
