import tempfile
import unittest
from pathlib import Path

import numpy as np

import yaml
from src.data.scenarios import PROJECT
from src.data.route_csv import read_tracks, speed_warnings
from src.data.replay_training_episodes import crop_routes


class TrainingExpansionTests(unittest.TestCase):
    def test_episode_crop_preserves_original_turns_and_stop(self):
        tracks = {'car:1': [[0, 0, 0], [2, 0, 0], [4, 4, 0], [8, 4, 8]],
                  'static:1': [[0, 10, 10], [8, 10, 10]]}
        paths = crop_routes(tracks, [('car', 1), ('static', 1)], 1, 7)
        np.testing.assert_allclose(paths[0], [[0, 0, 0], [1, 0, 0], [3, 4, 0], [6, 4, 6]])
        np.testing.assert_allclose(paths[1, :, 1:], [[10, 10]] * 4)
        np.testing.assert_array_equal(paths[0, :, 0], paths[1, :, 0])

    def test_repetitions_and_related_routes_do_not_cross_partitions(self):
        cfg = yaml.safe_load((PROJECT/'configs/training_dataset_v2.yaml').read_text())
        env = cfg['environment']
        groups, paired = {}, {}
        self.assertEqual(len(env['scenarios']), 84)
        for scenario in env['scenarios']:
            groups.setdefault(scenario['group'], set()).add(scenario['partition'])
            tracks = read_tracks(PROJECT/env['route_csv_root']/scenario['name'], env)
            self.assertEqual(len(tracks), 24)
            self.assertFalse(speed_warnings(tracks, env['ue_attributes']))
            base = scenario['name'].rsplit('_rep', 1)[0]
            if base in paired:
                self.assertEqual(paired[base], tracks)
            paired[base] = tracks
        self.assertTrue(all(len(p) == 1 for p in groups.values()))
        self.assertEqual(len(paired), 42)


if __name__ == '__main__':
    unittest.main()
