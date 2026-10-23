import unittest
import numpy as np
import torch
from src.train.common import scenario_condition, scenario_balanced_indices
from src.evaluation.generate_mobility import generation_conditions, generate
from unittest.mock import patch


class ScenarioConditionTests(unittest.TestCase):
    def test_ue_permutation_and_positive_multiplicity_do_not_change_condition(self):
        y=torch.tensor([[[1,0,0],[0,1,0],[0,0,0]]],dtype=torch.bool)
        m=torch.ones_like(y);m[...,2]=False
        expected=torch.tensor([[1.,1.,0.,1.,1.,0.]])
        torch.testing.assert_close(scenario_condition(y,m),expected)
        torch.testing.assert_close(scenario_condition(y.flip(1),m.flip(1)),expected)
        y[0,2,0]=True
        torch.testing.assert_close(scenario_condition(y,m),expected)

    def test_known_positive_unknown_and_known_negative(self):
        y=torch.tensor([[[1,0,1],[0,0,0]]],dtype=torch.bool)
        m=torch.tensor([[[1,1,0],[0,1,0]]],dtype=torch.bool)
        # One known positive establishes existence; all known negatives establish absence.
        torch.testing.assert_close(scenario_condition(y,m),torch.tensor([[1.,0.,0.,1.,1.,0.]]))
        m[0,1,1]=False
        torch.testing.assert_close(scenario_condition(y,m),torch.tensor([[1.,0.,0.,1.,0.,0.]]))

    def test_sampling_groups_are_ue_invariant(self):
        y=torch.zeros(5,4,3,dtype=torch.bool);m=torch.ones_like(y)
        for i in range(4):y[i,i,0]=True
        indices=scenario_balanced_indices(y,m,10000,np.random.default_rng(4))
        self.assertTrue(.47 < np.mean(indices==4) < .53)

    def test_generation_requests_event_without_ue_and_excludes_negatives(self):
        bank=torch.tensor([[0,0,0,1,1,0],[1,0,0,1,1,0],[0,1,0,1,1,0],
                           [1,1,0,1,1,0]],dtype=torch.float)
        for event in ['any_danger','ho_failure','pingpong']:
            c=generation_conditions(bank,['ho_failure','pingpong'],event,50)
            self.assertEqual(tuple(c.shape),(50,6))
            if event=='any_danger':self.assertTrue(c[:,:2].bool().any(-1).all())
            else:self.assertTrue(c[:,0 if event=='ho_failure' else 1].bool().all())
        with self.assertRaises(ValueError):generation_conditions(bank[:1],['ho_failure'],'any_danger',1)

    def test_old_diffusion_rejected_before_loading_mobility(self):
        with patch('src.evaluation.generate_mobility.torch.load',return_value={'format':'ns3_latent_diffusion_v4'}):
            with self.assertRaisesRegex(ValueError,'retrain diffusion only'):
                generate('/unused.pt','/unused','any_danger')
