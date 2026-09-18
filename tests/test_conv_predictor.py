import unittest
import torch
from src.models.episodic_diffusion import TemporalConvPredictor, build_latent_model


class ConvPredictorTests(unittest.TestCase):
    def test_lengths_conditioning_and_optimizer_update(self):
        torch.manual_seed(3)
        for length in (1, 7, 40):
            model = TemporalConvPredictor(length, 4, channels=8, num_blocks=4)
            x = torch.randn(2, length, 1, requires_grad=True)
            z = torch.randn(2, 4, requires_grad=True)
            t = torch.tensor([0.2, 0.8])
            prediction = model(x, t, z)
            self.assertEqual(prediction.shape, x.shape)
            self.assertFalse(torch.allclose(prediction, model(x, t, z+1)))
            self.assertFalse(torch.allclose(prediction, model(x, t+0.1, z)))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            weight = model.blocks[0].conv1.weight.detach().clone()
            (prediction-torch.randn_like(prediction)).square().mean().backward()
            self.assertGreater(z.grad.abs().sum().item(), 0)
            self.assertGreater(x.grad.abs().sum().item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(weight, model.blocks[0].conv1.weight))
            self.assertTrue(torch.isfinite(model(x, t, z)).all())

    def test_legacy_config_still_builds_mlp(self):
        config = {'data': {'input_dim': 1, 'episode_length': 8},
                  'model': {'hidden_dim': 16, 'latent_dim': 4, 'transformer_heads': 2,
                            'transformer_layers': 1, 'dropout': 0},
                  'diffusion': {'timesteps': 4}}
        legacy = build_latent_model(config)
        self.assertEqual(legacy.FORMAT, 'direct_z_episode_v2')
        config['diffusion']['denoiser'] = 'conv1d'
        conv = build_latent_model(config)
        self.assertEqual(conv.FORMAT, 'direct_z_conv_episode_v3')
        with self.assertRaises(RuntimeError):
            conv.load_state_dict(legacy.state_dict())


if __name__ == '__main__':
    unittest.main()
