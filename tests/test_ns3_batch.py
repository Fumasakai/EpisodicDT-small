import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.data.generate_ns3_batch import PROJECT, build_jobs, validate_config, write_inputs


class Ns3BatchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((PROJECT / 'configs/ns3_batch.yaml').read_text())

    def test_variations_keep_scenarios_in_one_partition(self):
        validate_config(self.cfg)
        jobs = build_jobs(self.cfg)
        self.assertEqual(len(jobs), 54)
        self.assertEqual(len({job['name'] for job in jobs}), 54)
        self.assertEqual(sum(job['partition'] == 'train' for job in jobs), 36)
        for scenario in self.cfg['scenarios']:
            selected = [job for job in jobs if job['scenario'] == scenario['name']]
            self.assertEqual(len(selected), 9)
            self.assertEqual({job['partition'] for job in selected}, {scenario['partition']})

    def test_waypoint_units_offsets_and_stationary_interval(self):
        scenario = next(s for s in self.cfg['scenarios'] if s['name'] == 'stop_and_return')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'
            write_inputs(self.cfg, scenario, root)
            cars = pd.read_csv(root / 'car.csv')
            first = cars[cars.ue_id == 1]
            self.assertEqual(first.timestamp.tolist(), [0, 60, 90, 150])
            self.assertEqual(first.location_y.tolist(), [30, 430, 430, 30])
            second = cars[cars.ue_id == 2]
            self.assertEqual(second.location_y.tolist(), [42, 442, 442, 42])
            static = pd.read_csv(root / 'static.csv')
            self.assertEqual(static.timestamp.tolist(), [0, 150])
            self.assertEqual(static.location_x.nunique(), 1)
            self.assertEqual(static.location_y.nunique(), 1)
            self.assertEqual(pd.read_csv(root / 'gnb.csv').values.tolist(), scenario['sites'])

    def test_reject_invalid_routes_before_simulation(self):
        for endpoint in (0, .5, float('nan')):
            cfg = copy.deepcopy(self.cfg)
            cfg['scenarios'][0]['car_route'][-1][0] = endpoint
            with self.assertRaises(ValueError):
                validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
