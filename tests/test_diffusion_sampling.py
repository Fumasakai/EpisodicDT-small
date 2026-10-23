import unittest
from unittest.mock import Mock
import torch
from src.train.train_diffusion import sample_latents
from src.models.latent_diffusion import LatentDiffusion


class DiffusionSamplingTests(unittest.TestCase):
    def test_mixed_conditions_keep_order_and_never_borrow_validation_anchors(self):
        model=Mock(latent_dim=2,steps=100)
        model.sample.side_effect=lambda condition, *args: condition+10 if args else condition+20
        bank=torch.tensor([[5.,6.],[7.,8.]])
        bc=torch.tensor([[1.,0.],[0.,1.]])
        requested=torch.tensor([[1.,1.],[0.,1.],[0.,0.],[1.,0.]])
        with self.assertWarnsRegex(UserWarning,'2/4 conditions'):
            result=sample_latents(model,requested,bank,bc,{'sampling_start':'positive','noise_fraction':.15})
        torch.testing.assert_close(result,torch.tensor([[21.,21.],[10.,11.],[20.,20.],[11.,10.]]))
        torch.testing.assert_close(model.sample.call_args_list[0].args[1],bank.flip(0))
        self.assertEqual(model.sample.call_args_list[0].args[2],15)
        self.assertEqual(len(model.sample.call_args_list[1].args),1)

    def test_real_sampler_all_known_unknown_and_noise_are_finite_and_reproducible(self):
        model=LatentDiffusion(3,2,hidden_dim=8,steps=5)
        bank=torch.randn(2,3); bc=torch.eye(2)
        for mode,c in [('positive',bc),('positive',torch.ones(2,2)),('noise',bc)]:
            cfg={'sampling_start':mode,'noise_fraction':.5}
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                torch.manual_seed(4);a=sample_latents(model,c,bank,bc,cfg)
                torch.manual_seed(4);b=sample_latents(model,c,bank,bc,cfg)
            self.assertEqual(tuple(a.shape),(2,3));self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,b)
