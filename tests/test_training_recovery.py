"""Regression checks for invalid simulator samples and bounded recovery."""
import copy
import json
from contextlib import ExitStack
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from src.train.train_mobility import collect_rollouts, policy_loss, resume_compatible, train


class TrainingRecoveryTests(unittest.TestCase):
    def collect(self, outcomes, attempts=3):
        model = Mock()
        model.sample.side_effect = lambda z: (z.detach(), torch.ones(len(z), requires_grad=True))
        model.decode.side_effect = lambda actions: actions
        sim = Mock()
        sim.try_evaluate.side_effect = outcomes
        cfg = {'training': {'samples_per_episode': 2, 'max_rollout_attempts': attempts, 'runs': [1]}}
        with tempfile.TemporaryDirectory() as tmp, patch(
                'src.train.train_mobility.total_reward', side_effect=lambda m, p, v, c: torch.tensor(v)):
            result = collect_rollouts(model, torch.ones(2, 1), sim, torch.zeros(2, 1),
                                      torch.ones(2, 1), [0, 1], cfg, Path(tmp))
        return result, sim

    def test_successes_accumulate_only_within_same_episode(self):
        positive = {'mean_reward': 1., 'danger_success_rate': 1.}
        negative = {'mean_reward': -1., 'danger_success_rate': 0.}
        (logp, rewards, valid, _, rounds), sim = self.collect(
            [positive, None, None, negative, None, negative, None, None])
        self.assertEqual(rounds, 2)
        self.assertEqual(sim.try_evaluate.call_count, 8)
        self.assertEqual(valid.tolist(), [[True, False, False, True], [False, True, False, False]])
        logp.retain_grad()
        policy_loss(logp, rewards, valid).backward()
        self.assertLess(logp.grad[0, 0], 0)
        self.assertGreater(logp.grad[0, 3], 0)
        self.assertEqual(logp.grad[1].abs().sum(), 0)
        self.assertEqual(logp.grad[~valid].abs().sum(), 0)
        roots = [call.args[3] for call in sim.try_evaluate.call_args_list]
        self.assertEqual(len(set(roots)), 8)

    def test_all_failures_stop_at_limit_without_gradient(self):
        (logp, rewards, valid, _, rounds), sim = self.collect([None] * 12)
        self.assertEqual(rounds, 3)
        self.assertEqual(sim.try_evaluate.call_count, 12)
        self.assertFalse(valid.any())
        logp.retain_grad()
        policy_loss(logp, rewards, valid).backward()
        self.assertEqual(logp.grad.abs().sum(), 0)

    def test_no_retries_for_sufficient_successes(self):
        ok = {'mean_reward': 1., 'danger_success_rate': 1.}
        result, sim = self.collect([ok] * 4)
        self.assertEqual(result[-1], 1)
        self.assertEqual(sim.try_evaluate.call_count, 4)

    def test_checkpoint_resume_and_bounded_invalid_stop(self):
        class TinyModel(torch.nn.Module):
            FORMAT = 'test'
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def encode(self, batch):
                return batch * self.weight
            def sample(self, z):
                return z.detach(), z[:, 0]
            def decode(self, action):
                return action

        dataset = SimpleNamespace(source_hashes={})
        sim = Mock(build={'binary': 'same-build'})
        positive = {'mean_reward': 1., 'danger_success_rate': 1.}
        negative = {'mean_reward': -1., 'danger_success_rate': 0.}
        sim.try_evaluate.side_effect = [positive, negative, None, None]
        evaluation = Mock(return_value={'reward': 0.})
        replacements = dict(load_partitions=lambda cfg: (dataset, dataset),
            targets=lambda d, c: (torch.zeros(1, 1), torch.ones(1, 1)),
            model_kwargs=lambda d, c: {}, MobilityModel=TinyModel,
            simulator=lambda m, c: sim, provenance=lambda d: {},
            batch_at=lambda d, i: torch.ones(len(i), 1),
            balanced_indices=lambda y, m, n, r: [0] * n,
            evaluate_policy=evaluation)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            for name, value in replacements.items():
                stack.enter_context(patch('src.train.train_mobility.' + name, value))
            cfg = dict(seed=42, output_dir=str(Path(tmp)/'training'), events=['ho_failure'],
                training=dict(updates=1, learning_rate=.001, validation_episodes=1,
                    episodes_per_update=1, samples_per_episode=2, runs=[1], latent_l2=0.,
                    validation_every=1, max_rollout_attempts=1, max_consecutive_invalid_updates=1))
            out = train(cfg)
            first = torch.load(out/'latest.pt', weights_only=False)
            self.assertEqual(first['update'], 1)
            cfg['training']['updates'] = 2
            with self.assertRaisesRegex(RuntimeError, 'checkpoint saved'):
                train(cfg, resume=True)
            latest = torch.load(out/'latest.pt', weights_only=False)
            self.assertEqual(latest['update'], 2)
            torch.testing.assert_close(latest['model']['weight'], first['model']['weight'])
            self.assertEqual(latest['consecutive_invalid_updates'], 1)
            self.assertEqual(json.loads((out/'history.json').read_text())[-1]['status'], 'skipped')
            self.assertEqual(len(list(out.glob('session_*.json'))), 2)
            # A failed validation must still save the just-updated model, while
            # retaining the previous independently validated best model.
            cfg['training']['updates'] = 3
            sim.try_evaluate.side_effect = [positive, negative]
            evaluation.return_value = {'status': 'invalid', 'reward': None,
                                       'danger_success_rate': None, 'invalid_simulations': 1}
            best_before = torch.load(out/'mobility.pt', weights_only=False)
            with self.assertWarnsRegex(UserWarning, 'Validation at update=3 is incomplete'):
                train(cfg, resume=True)
            self.assertEqual(torch.load(out/'latest.pt', weights_only=False)['update'], 3)
            best_after = torch.load(out/'mobility.pt', weights_only=False)
            torch.testing.assert_close(best_after['model']['weight'], best_before['model']['weight'])
            metrics = json.loads((out/'metrics.json').read_text())
            self.assertEqual(metrics['last_validation_status'], 'invalid')
            self.assertEqual(metrics['incomplete_validation_checks'], 1)
            # Resume reconstructs patience from history and does not run more
            # simulations when complete validations already exhausted it.
            cfg['training']['early_stopping'] = {'patience': 1}
            cfg['training']['updates'] = 10
            calls_before = sim.try_evaluate.call_count
            train(cfg, resume=True)
            self.assertEqual(sim.try_evaluate.call_count, calls_before)
            self.assertEqual(torch.load(out/'latest.pt', weights_only=False)['update'], 3)
            metrics = json.loads((out/'metrics.json').read_text())
            self.assertEqual(metrics['updates'], 3)
            self.assertEqual(metrics['requested_updates'], 10)
            self.assertEqual(metrics['stop_reason'], 'validation_plateau')
            sim.build = {'binary': 'different-build'}
            with self.assertRaisesRegex(ValueError, 'build differs'):
                train(cfg, resume=True)

    def test_resume_rejects_changed_research_conditions(self):
        saved = {'events': ['ho_failure'], 'training': {'updates': 3, 'runs': [1]}}
        current = copy.deepcopy(saved)
        current['training'].update(updates=5, max_rollout_attempts=2)
        resume_compatible(saved, current)
        current['training']['runs'] = [2]
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            resume_compatible(saved, current)
        self.assertEqual(saved['training']['updates'], 3)


if __name__ == '__main__':
    unittest.main()
