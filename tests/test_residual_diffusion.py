import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.episodic_diffusion import EpisodicDiffusion
from src.train.validation import forecast_metrics, validate


class ResidualDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = EpisodicDiffusion(1, 3, 8, 4, 4, 2, 1, 0.0)

    def test_zero_terminal_signal_and_finite_sampling(self):
        self.assertEqual(self.model.alpha_bars[-1].item(), 0.0)
        self.assertTrue((self.model.alpha_bars[1:] < self.model.alpha_bars[:-1]).all())
        self.assertEqual(self.model.posterior_variance[0].item(), 0.0)
        self.model.eval()
        output = self.model.sample(torch.randn(2, 5, 1), 4)
        self.assertEqual(output.shape, (2, 4, 3, 1))
        self.assertTrue(output.isfinite().all())

    def test_oracle_velocity_loss_and_encoder_gradients(self):
        history, future = torch.randn(2, 5, 1), torch.randn(2, 3, 1)
        steps = torch.tensor([0, 3])
        noise = torch.randn_like(future)
        a = self.model.alpha_bars[steps, None, None]
        target = a.sqrt() * noise - (1 - a).sqrt() * (future - history[:, -1:, :])
        with patch('torch.randint', return_value=steps), patch('torch.randn_like', return_value=noise), \
                patch.object(self.model.noise_predictor, 'forward', return_value=target):
            self.assertEqual(self.model.loss(history, future).item(), 0.0)
        self.model.loss(history, future).backward()
        grad = self.model.encoder.input_projection.weight.grad
        self.assertTrue(grad.isfinite().all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_oracle_reconstruction_adds_history_baseline(self):
        self.model.eval()
        history = torch.tensor([[[2.0]] * 5, [[-3.0]] * 5])
        residual = 0.7

        def oracle(x, timestep, context):
            steps = (timestep * (self.model.timesteps - 1)).round().long()
            a = self.model.alpha_bars[steps, None, None]
            return (a.sqrt() * x - residual) / (1 - a).sqrt()

        with patch.object(self.model.noise_predictor, 'forward', side_effect=oracle):
            output = self.model.sample(history, 2)
        expected = (history[:, None, -1:, :] + residual).expand_as(output)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_crps_matches_pairwise_definition(self):
        samples = torch.tensor([[[0., 1.], [2., 4.], [5., 7.]]])
        actual = torch.tensor([[1., 3.]])
        expected = ((samples - actual[:, None]).abs().mean(1)
                    - 0.5 * (samples[:, :, None] - samples[:, None, :]).abs().mean((1, 2)))
        result = forecast_metrics(samples, actual)
        self.assertAlmostEqual(result['crps_dbm'], expected.mean().item(), places=6)

    def test_validation_repeats_without_affecting_training_rng(self):
        loader = DataLoader(TensorDataset(torch.randn(3, 5, 1), torch.randn(3, 3, 1)), batch_size=2)
        before = torch.get_rng_state().clone()
        first = validate(self.model, loader, torch.device('cpu'), -95, 10, 4, 13)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(self.model.training)
        second = validate(self.model, loader, torch.device('cpu'), -95, 10, 4, 13)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
