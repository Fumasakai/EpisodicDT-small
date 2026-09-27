import unittest
from unittest.mock import patch
import torch
from src.models.episodic_diffusion import LatentEpisodeDiffusion


class DeltaLossTests(unittest.TestCase):
    def model(self, length=4, weight=0.1):
        return LatentEpisodeDiffusion(1, length, 8, 4, 4, 2, 1, 0,
                                     conv_channels=8, conv_blocks=1, delta_loss_weight=weight)

    def test_terminal_reconstruction_and_weight(self):
        model = self.model()
        clean = torch.tensor([[[0.], [2.], [-1.], [4.]]])
        error = torch.arange(4.).reshape(1, 4, 1).requires_grad_()
        with patch('torch.randint', return_value=torch.tensor([3])), patch.object(
                model.noise_predictor, 'forward', return_value=-clean + error):
            terms = model.loss_terms(clean)
        torch.testing.assert_close(terms['diffusion_mse'], torch.tensor(3.5))
        torch.testing.assert_close(terms['delta_mse'], torch.tensor(1.))
        torch.testing.assert_close(terms['loss'], torch.tensor(3.6))
        terms['delta_mse'].backward()
        self.assertGreater(error.grad.abs().sum().item(), 0)

    def test_gradient_and_single_step_and_invalid_weight(self):
        model = self.model()
        terms = model.loss_terms(torch.randn(2, 4, 1))
        terms['delta_mse'].backward()
        for module in (model.encoder, model.noise_predictor):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        terms = self.model(length=1).loss_terms(torch.randn(2, 1, 1))
        self.assertEqual(terms['delta_mse'].item(), 0)
        self.assertTrue(torch.isfinite(terms['loss']))
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.model(weight=weight)
