import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.data.generate_handover_batch import run_batch, validate, scenario_runs
from src.data.scenarios import PROJECT, write_inputs
from src.data.population import environment_slots
from scenario_fixture import config as fixture_config


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.config = fixture_config()

    def test_explicit_repetition_selects_one_run(self):
        scenario = self.config['environment']['scenarios'][0]
        self.config['execution']['runs'] = [1, 2]
        self.assertEqual(scenario_runs(self.config, scenario), [1, 2])
        scenario['simulation_run'] = 2
        self.assertEqual(scenario_runs(self.config, scenario), [2])
        validate(self.config)
        for bad in (0, True, 1.5):
            scenario['simulation_run'] = bad
            with self.assertRaises(ValueError):
                validate(self.config)

    def test_waypoint_units_offsets_and_stationary_interval(self):
        env = copy.deepcopy(self.config['environment'])
        env.pop('population', None)
        env.update(duration_s=150, car_offsets=[[0, 0], [0, 12]])
        scenario = copy.deepcopy(env['scenarios'][0])
        scenario['car_route'] = [[0, 0, 30], [.4, 0, 430], [.6, 0, 430], [1, 0, 30]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'
            write_inputs(env, scenario, root)
            cars = pd.read_csv(root / 'car.csv')
            first, second = cars[cars.ue_id == 1], cars[cars.ue_id == 2]
            self.assertEqual(first.timestamp.tolist(), [0, 60, 90, 150])
            self.assertEqual(first.location_y.tolist(), [30, 430, 430, 30])
            self.assertEqual(second.location_y.tolist(), [42, 442, 442, 42])
            static = pd.read_csv(root / 'static.csv')
            self.assertEqual(static.timestamp.tolist(), [0, 150])
            self.assertEqual(static.location_x.nunique(), 1)
            self.assertEqual(static.location_y.nunique(), 1)
            self.assertFalse((root / 'gnb.csv').exists())

    def test_configured_counts_offsets_and_zero_attribute_csv(self):
        env = self.config['environment']
        env['population']['counts'] = {'car': 2, 'walker': 0, 'static': 3}
        env['ue_offsets'] = {'car': [[0, 0], [3, 7]]}
        validate(self.config)
        self.assertEqual(environment_slots(env), [('car', 1), ('car', 2),
                         ('static', 1), ('static', 2), ('static', 3)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'inputs'
            write_inputs(env, env['scenarios'][0], root)
            cars = pd.read_csv(root/'car.csv')
            a, b = cars[cars.ue_id == 1], cars[cars.ue_id == 2]
            self.assertEqual((b.location_x.to_numpy()-a.location_x.to_numpy()).tolist(), [3,3,3])
            self.assertEqual((b.location_y.to_numpy()-a.location_y.to_numpy()).tolist(), [7,7,7])
            walkers = pd.read_csv(root/'walker.csv')
            self.assertTrue(walkers.empty)
            self.assertEqual(list(walkers), ['timestamp','ue_id','location_x','location_y'])
            static = pd.read_csv(root/'static.csv')
            self.assertEqual(static.ue_id.nunique(), 3)
            self.assertEqual(static.groupby('ue_id').location_y.first().diff().dropna().tolist(), [10,10])

    def test_reject_population_limits_and_ambiguous_offsets(self):
        invalid = [dict(car=0, walker=0, static=0), dict(car=13,walker=0,static=0),
                   dict(car=-1,walker=2,static=1), dict(car=True,walker=1,static=1),
                   dict(car=1.5,walker=1,static=1), dict(car=1,walker=1),
                   dict(car=1,walker=1,static=1,drone=1)]
        for counts in invalid:
            cfg = copy.deepcopy(self.config)
            cfg['environment']['population']['counts'] = counts
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                validate(cfg)
        for lo, hi in [(0,12),(3,2),(1,True)]:
            cfg = copy.deepcopy(self.config)
            cfg['environment']['population'].update(min_ues=lo, max_ues=hi)
            with self.assertRaises(ValueError):
                validate(cfg)
        for patch in [{'car_offsets': [[0,0]]}, {'ue_offsets': {'car': []}},
                      {'ue_spacing_m': float('nan')}]:
            cfg = copy.deepcopy(self.config)
            cfg['environment'].update(patch)
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_reject_invalid_routes_and_runs_before_simulation(self):
        for endpoint in (0, .5, float('nan')):
            cfg = copy.deepcopy(self.config)
            cfg['environment']['scenarios'][0]['car_route'][-1][0] = endpoint
            with self.assertRaises(ValueError):
                validate(cfg)
        for runs in ([0], [-1], [True], [1.5], [1, 1]):
            cfg = copy.deepcopy(self.config)
            cfg['execution']['runs'] = runs
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_reject_per_scenario_geometry(self):
        self.config['environment']['scenarios'][0]['sites'] = [[0, 0, 10]]
        with self.assertRaisesRegex(ValueError, 'Per-scenario'):
            validate(self.config)

    def test_prepare_resume_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'ns3/build/scratch/ns3.46-tranData'
            binary.parent.mkdir(parents=True)
            binary.write_text('test binary; preparation never executes this file')
            self.config.update(ns3_dir=str(root / 'ns3'), batch_dir=str(root / 'batch'))
            run_batch(self.config, prepare_only=True)
            plan = (root / 'batch/plan.json').read_bytes()
            run_batch(self.config, prepare_only=True)
            self.assertEqual(plan, (root / 'batch/plan.json').read_bytes())
            path = root / 'batch/inputs/crossing_train/car.csv'
            path.write_text(path.read_text().replace('40', '41'))
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_batch(self.config, prepare_only=True)


if __name__ == '__main__':
    unittest.main()
