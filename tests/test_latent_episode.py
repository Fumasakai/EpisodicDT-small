import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from src.models.episodic_diffusion import LatentEpisodeDiffusion
from src.train.validation import validate_episodes


class LatentEpisodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = LatentEpisodeDiffusion(1, 8, 16, 4, 4, 2, 1, 0)
        self.episode = torch.randn(3, 8, 1)

    def test_kl_formula_and_gradients(self):
        terms = self.model.loss_terms(self.episode, beta=.03)
        posterior = self.model.encode(self.episode)
        expected = torch.distributions.kl_divergence(
            posterior, torch.distributions.Normal(torch.zeros_like(posterior.loc),
                                                   torch.ones_like(posterior.scale))).sum(-1).mean()
        torch.testing.assert_close(terms['kl'], expected)
        torch.testing.assert_close(terms['loss'], terms['diffusion_mse']+.03*terms['kl'])
        self.model.loss(self.episode, beta=0).backward()
        for module in (self.model.encoder, self.model.posterior_head, self.model.noise_predictor):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))

    def test_generate_never_calls_encoder_and_latent_changes_output(self):
        self.model.eval()
        z = self.model.encode(self.episode).sample()
        with patch.object(self.model, 'encode', side_effect=AssertionError('source access')):
            torch.manual_seed(8)
            a = self.model.generate(z, 4)
            torch.manual_seed(8)
            b = self.model.generate(z, 4)
            torch.manual_seed(8)
            c = self.model.generate(z+2, 4)
        self.assertEqual(a.shape, (3,4,8,1))
        torch.testing.assert_close(a,b)
        self.assertFalse(torch.allclose(a,c))
        self.assertFalse(torch.allclose(a[:,0],a[:,1]))
        self.assertTrue(torch.isfinite(a).all())
        with self.assertRaises(ValueError):
            self.model.generate(self.episode)

    def test_oracle_generates_absolute_episode_without_baseline(self):
        self.model.eval()
        clean = torch.linspace(-2, 1, 8).view(1, 8, 1)
        def oracle(noisy, time, z):
            step = (time*(self.model.timesteps-1)).round().long()
            alpha = self.model.alpha_bars[step].view(-1,1,1)
            return (alpha.sqrt()*noisy-clean)/(1-alpha).sqrt()
        with patch.object(self.model.noise_predictor, 'forward', side_effect=oracle):
            result = self.model.generate(torch.full((3,4), 9.0), 2)
        torch.testing.assert_close(result, clean.view(1,1,8,1).expand(3,2,8,1), atol=1e-5, rtol=1e-5)

    def test_validation_preserves_rng_and_training_mode(self):
        self.model.train()
        loader=DataLoader(self.episode, batch_size=2)
        before=torch.random.get_rng_state().clone()
        a=validate_episodes(self.model, loader, 'cpu', -90, 5, .001, 3, 7)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(self.model.training)
        b=validate_episodes(self.model, loader, 'cpu', -90, 5, .001, 3, 7)
        self.assertEqual(a,b)
        self.assertIn('reconstruction_crps_dbm',a)


if __name__ == '__main__':
    unittest.main()
