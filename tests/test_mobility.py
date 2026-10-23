import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.models.mobility import MobilityModel
from src.models.latent_diffusion import LatentDiffusion
from src.simulation.runner import event_score, export_routes, read_events, Simulator, UnknownDangerOutcome
from src.train.common import load_config, condition, targets, total_reward, load_policy, simulator
from src.train.train_mobility import policy_loss
from test_handover_pipeline import fixture


class MobilityTests(unittest.TestCase):
    def test_legacy_training_config_reports_regeneration_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training.yaml'
            legacy = 'data: {}\nmodel: {}\ntraining:\n  classification_weight: 1.0\n'
            path.write_text(legacy)
            with self.assertRaisesRegex(ValueError, 'Generate a new batch') as error:
                load_config(path)
            self.assertIn('simulation_config', str(error.exception))
            self.assertIn('per-scenario base stations', str(error.exception))
            self.assertEqual(path.read_text(), legacy)

    def test_non_mapping_training_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training.yaml'
            path.write_text('- not a configuration\n')
            with self.assertRaisesRegex(ValueError, 'YAML mapping'):
                load_config(path)

    def setUp(self):
        torch.manual_seed(10)
        self.cfg = load_config('configs/pipeline.yaml')
        self.slots = [('car',1), ('walker',1), ('static',1)]
        self.profiles = [self.cfg['ue_attributes'][k] for k,_ in self.slots]
        self.model = MobilityModel(feature_dim=4, points=8, slots=self.slots, profiles=self.profiles,
                                   sites=[[0,0,10],[400,0,10]], duration_s=3., hidden_dim=8,
                                   latent_dim=8, num_heads=2, num_layers=1)
        self.batch = {'values': torch.randn(2,3,8,4), 'mask': torch.ones(2,3,8,4,dtype=torch.bool)}

    def test_policy_rewards_update_encoder_and_decoder(self):
        z = self.model.encode(self.batch)
        action, lp = self.model.sample(z.repeat_interleave(3,0))
        before = [self.model.input.weight.detach().clone(), self.model.decoder[-1].weight.detach().clone()]
        opt = torch.optim.Adam(self.model.parameters(), lr=.001)
        loss = policy_loss(lp.reshape(2,3), torch.tensor([[1.,-1.,.2], [-1.,1.,.5]]))
        opt.zero_grad(); loss.backward()
        self.assertGreater(self.model.input.weight.grad.abs().sum(), 0)
        self.assertGreater(self.model.decoder[-1].weight.grad.abs().sum(), 0)
        opt.step()
        self.assertFalse(torch.equal(before[0], self.model.input.weight))
        self.assertFalse(torch.equal(before[1], self.model.decoder[-1].weight))
        self.assertFalse(action.requires_grad)
        self.assertFalse(any('classifier' in k or 'surrogate' in k for k in self.model.state_dict()))

    def test_invalid_simulations_have_no_policy_gradient(self):
        logp = torch.tensor([[.1,.2,.3]], requires_grad=True)
        loss = policy_loss(logp, torch.tensor([[1., -100., -1.]]), torch.tensor([[True,False,True]]))
        loss.backward()
        self.assertEqual(logp.grad[0,1], 0.)
        self.assertLess(logp.grad[0,0], 0.)
        self.assertGreater(logp.grad[0,2], 0.)

    def test_masked_communications_do_not_affect_latent(self):
        self.batch['mask'][...,2] = False
        a = self.model.encode(self.batch)
        self.batch['values'][...,2] = 1e9
        torch.testing.assert_close(a, self.model.encode(self.batch))

    def test_speed_stops_initial_position_and_waypoint_csv(self):
        action = torch.randn(4,self.model.action_dim)*5
        points = self.model.decode(action).detach().numpy()
        with tempfile.TemporaryDirectory() as tmp:
            for i,p in enumerate(points):
                root = Path(tmp)/str(i)
                export_routes(root,p,self.slots,self.profiles)
                np.testing.assert_allclose(p[2,:,1:], np.broadcast_to(p[2,0,1:],p[2,:,1:].shape))
                csv = pd.read_csv(root/'car.csv')
                np.testing.assert_allclose(csv[['timestamp','location_x','location_y']].values,p[0],atol=1e-5)
        # Stops are represented by duplicate XY at distinct timestamps.
        raw = action.reshape(4,3,-1)
        raw[:,:,5::3] = 10
        stop = self.model.decode(raw.flatten(1)).detach().numpy()
        np.testing.assert_allclose(np.diff(stop[...,1:],axis=2),0,atol=1e-5)
        self.assertTrue((points[:, :, 0, 1:] >= self.model.bounds[:, 0].numpy()).all())
        self.assertTrue((points[:, :, 0, 1:] <= self.model.bounds[:, 1].numpy()).all())

    def test_sharp_turns_acceleration_and_leaving_initial_area_are_allowed(self):
        raw = torch.zeros(1, 3, 3 + 3*self.model.segments)
        raw[:, :, 5::3] = -10  # move
        raw[:, 0, 0] = 10  # start near the right boundary
        raw[:, 0, 4::3] = 10  # almost 65 m/s
        raw[:, 0, 6] = torch.pi  # reverse direction in second interval
        points = self.model.decode(raw.flatten(1))
        velocity = torch.diff(points[0, 0, :, 1:], dim=0) / float(self.model.times[1])
        self.assertGreater(velocity[0, 0], 64)
        self.assertLess(velocity[1, 0], -64)
        self.assertGreater((velocity[1]-velocity[0]).norm().item() / float(self.model.times[1]), 200)
        self.assertGreater(points[0, 0, 1, 1], self.model.bounds[0, 1])
        # Even stale configuration must not reintroduce motion penalties.
        cfg = copy.deepcopy(self.cfg)
        cfg['realism_weights'] = {'acceleration': 100, 'area': 100}
        torch.testing.assert_close(total_reward(self.model, points, [1.25], cfg), torch.tensor([1.25]))
        with tempfile.TemporaryDirectory() as tmp:
            export_routes(Path(tmp)/'route', points[0].numpy(), self.slots, self.profiles)

    def test_old_checkpoints_are_rejected_and_new_checkpoints_roundtrip(self):
        checkpoint = {'format': self.model.FORMAT, 'model_kwargs': dict(
            feature_dim=4, points=8, slots=self.slots, profiles=self.profiles,
            sites=[[0,0,10],[400,0,10]], duration_s=3., hidden_dim=8,
            latent_dim=8, num_heads=2, num_layers=1), 'model': self.model.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            torch.save(checkpoint, path)
            restored, _ = load_policy(path)
            action = torch.randn(1, self.model.action_dim)
            torch.testing.assert_close(restored.decode(action), self.model.decode(action))
            checkpoint['format'] = 'ns3_waypoint_policy_v3'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'retrain'):
                load_policy(path)

    def test_previous_config_ignores_removed_limits_without_mutation(self):
        import yaml
        cfg = copy.deepcopy(self.cfg)
        cfg.pop('simulation')
        cfg['realism_weights'] = {'acceleration': 100, 'area': 100}
        cfg['ue_attributes']['car']['max_turn_rate_rad_s'] = .01
        cfg['ue_attributes']['car']['max_acceleration_mps2'] = .01
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'training.yaml'
            text = yaml.safe_dump(cfg)
            path.write_text(text)
            with self.assertWarnsRegex(UserWarning, 'obsolete'):
                loaded = load_config(path)
            self.assertNotIn('realism_weights', loaded)
            self.assertNotIn('max_turn_rate_rad_s', loaded['ue_attributes']['car'])
            self.assertEqual(path.read_text(), text)

    def test_configured_population_preserves_input_output_count_and_types(self):
        slots = [('car',1), ('car',2), ('walker',1), ('static',1), ('static',2)]
        profiles = [self.cfg['ue_attributes'][kind] for kind, _ in slots]
        model = MobilityModel(feature_dim=4, points=8, slots=slots, profiles=profiles,
                              sites=[[0,0,10],[400,0,10]], duration_s=3., hidden_dim=8,
                              latent_dim=8, num_heads=2, num_layers=1)
        batch = {'values': torch.randn(2,5,8,4), 'mask': torch.ones(2,5,8,4,dtype=torch.bool)}
        z = model.encode(batch)
        action, _ = model.sample(z)
        points = model.decode(action)
        self.assertEqual(points.shape[:2], batch['values'].shape[:2])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'route'
            export_routes(root, points[0].numpy(), slots, profiles)
            self.assertEqual([pd.read_csv(root/f'{k}.csv').ue_id.nunique()
                              for k in ('car','walker','static')], [2,1,2])
        with self.assertRaisesRegex(ValueError, 'layout differs'):
            simulator(model, self.cfg)  # configuration still declares 1/1/1

    def test_any_danger_ignores_source_ue_identity_and_normal_labels(self):
        target = np.array([[0,1,0],[0,0,0]],bool)
        mask = np.array([[1,1,0],[1,1,0]],bool)
        for source in [target, np.zeros_like(target)]:
            # A different UE and a different danger type still earn positive reward.
            observed = np.array([[0,0,0],[1,0,0]],bool)
            reward, success, fraction = event_score(source,mask,observed,mask)
            self.assertEqual(reward, 1.)
            self.assertTrue(success)
            self.assertEqual(fraction, .5)
        reward, success, _ = event_score(target,mask,np.zeros_like(target),mask)
        self.assertEqual(reward, -1.)
        self.assertFalse(success)
        # More dangerous UEs must not increase the maximum reward.
        self.assertEqual(event_score(target,mask,np.ones_like(target),mask)[0], 1.)

    def test_unknown_danger_is_not_negative_and_rlf_is_not_a_reward(self):
        target = np.zeros((2,3),bool)
        mask = np.array([[1,1,0],[1,1,0]],bool)
        known = mask.copy(); known[0,0] = False
        with self.assertRaises(UnknownDangerOutcome):
            event_score(target,mask,target,known)
        observed = target.copy(); observed[1,1] = True
        self.assertEqual(event_score(target,mask,observed,known)[0], 1.)
        observed = target.copy(); observed[:,2] = True
        self.assertEqual(event_score(target,mask,observed,mask)[0], -1.)
        with self.assertRaises(ValueError):
            event_score(target,mask,observed,mask,['radio_link_failure'])
        # Configured event scope, not source masks, decides which danger counts.
        observed = target.copy(); observed[1,0] = True
        self.assertEqual(event_score(target,mask,observed,mask,['pingpong'])[0], -1.)
        with patch.object(Simulator, 'evaluate', side_effect=UnknownDangerOutcome('unknown')):
            sim = Simulator.__new__(Simulator)
            self.assertIsNone(sim.try_evaluate())

    def test_trace_labels_use_half_open_window_and_pending_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = fixture(Path(tmp))
            y,m = read_events(raw,1,2.6,3.4,1.)
            self.assertEqual(y.tolist(), [[True,True,False]])
            self.assertFalse(m[0,2])
            y,m = read_events(raw,1,4.,6.,1.)
            self.assertFalse(m[0,0])  # ongoing HO at right edge
            self.assertFalse(m[0,2])

    def test_simulator_crash_is_not_negative_reward(self):
        model = self.model
        with torch.no_grad():
            points = model.decode(model.distribution(torch.zeros(1,8)).mean)[0].numpy()
        import subprocess
        with tempfile.TemporaryDirectory() as tmp, patch('src.simulation.runner.subprocess.run',side_effect=subprocess.CalledProcessError(1,'ns3')):
            cfg = copy.deepcopy(self.cfg['simulation'])
            binary = Path(tmp)/'ns3/build/scratch/ns3.46-tranData'
            binary.parent.mkdir(parents=True); binary.write_text('mock binary')
            cfg['ns3_dir'] = str(Path(tmp)/'ns3')
            sim = Simulator(cfg,self.slots,self.profiles,1.,3.)
            with self.assertRaises(subprocess.CalledProcessError):
                sim.evaluate(points,np.zeros((3,3)),np.ones((3,3),bool),Path(tmp)/'candidate',[1])
            self.assertTrue((Path(tmp)/'candidate/failure.json').exists())
            self.assertFalse((Path(tmp)/'candidate/result.json').exists())

    def test_diffusion_condition_includes_ue_and_unknown_mask(self):
        y = torch.tensor([[[False,True,False],[False,False,False]]])
        mask = torch.ones_like(y); mask[...,2] = False
        a = condition(y,mask)
        self.assertEqual(a.shape,(1,12))
        self.assertFalse(torch.equal(a,condition(y.flip(1),mask)))
        diffusion = LatentDiffusion(8,12,16,10)
        loss = diffusion.losses(torch.randn(4,8),a.expand(4,-1)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(diffusion.network[0].weight.grad.abs().sum(),0)
        self.assertEqual(diffusion.sample(a).shape,(1,8))
        self.assertTrue(torch.isfinite(diffusion.sample(a,torch.zeros(1,8),2)).all())
        with self.assertRaises(ValueError):
            diffusion.sample(a,start_step=2)

    def test_reject_rlf_reward_and_shared_validation_seeds(self):
        from src.train.common import validate_config
        cfg = copy.deepcopy(self.cfg); cfg['events'].append('radio_link_failure')
        with self.assertRaisesRegex(ValueError,'RLF'):
            validate_config(cfg)
        cfg = copy.deepcopy(self.cfg); cfg['training']['validation_runs'] = [1]
        with self.assertRaisesRegex(ValueError,'disjoint'):
            validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
