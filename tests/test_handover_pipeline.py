import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.prepare_handover import event_outcomes, prepare_run
from src.data.handover_dataset import HandoverDataset, assert_disjoint
from src.data.generate_handover_batch import validate


def fixture(root, name='run', offset=0, completed=True):
    raw = root / name
    raw.mkdir()
    times = np.arange(1, 61) * 100.
    kpi = pd.DataFrame({'time_ms': times, 'global_ue_id': 1, 'imsi': 11,
                        'serving_cell_id': 1, 'x': np.arange(60) + offset, 'y': 10., 'z': 1.5,
                        'RSRP': -90. + np.arange(60) / 10 + offset, 'rsrp_age_ms': 0.,
                        'dl_data_sinr_db': 10., 'sinr_age_ms': 0.,
                        'speed_mps': 10., 'vx_mps': 10., 'vy_mps': 0., 'vz_mps': 0.,
                        'dl_packet_bytes': 200, 'dl_pps': 100, 'ul_packet_bytes': 200, 'ul_pps': 100})
    kpi.loc[5, 'RSRP'] = np.nan
    kpi.loc[6, 'rsrp_age_ms'] = 500
    kpi.to_csv(raw / 'ue_kpi.csv', index=False)
    pd.DataFrame([{'time_ms': t, 'global_ue_id': 1, 'cell_id': c,
                   'RSRP': -80 - c, 'measurement_age_ms': 0.} for t in times for c in (1, 2)]).to_csv(
                       raw / 'ue_neighbor_meas.csv', index=False)
    pd.DataFrame({'gnb_idx': [0, 1], 'cell_id': [1, 2]}).to_csv(raw / 'gnb_cell_id_map.csv', index=False)
    pd.DataFrame({'gnb_idx': [0, 1], 'x': [0, 300], 'y': [0, 0], 'z': [10, 10]}).to_csv(
        raw / 'gnb_positions.csv', index=False)
    events = [(2., 'start', 11, 1, 1, 2), (2.1, 'end_ok', 11, 2, 1, np.nan),
              (2.5, 'start', 11, 2, 1, 1), (2.6, 'end_ok', 11, 1, 1, np.nan),
              (3.2, 'start', 11, 1, 1, 2), (3.3, 'HandoverEndError', 11, 1, 1, np.nan),
              (3.4, 'RadioLinkFailure', 11, 1, 1, np.nan), (5., 'start', 11, 1, 1, 2)]
    pd.DataFrame(events, columns=['time_s', 'event', 'imsi', 'cell_id', 'rnti', 'target_cell_id']).to_csv(
        raw / 'ho_events.csv', index=False)
    manifest = dict(samplePeriod=.1, stopTime=6., completed=completed,
                    failure_events_recorded=True, scenario_group=name)
    (raw / 'run_manifest.json').write_text(json.dumps(manifest))
    return raw


class HandoverPipelineTests(unittest.TestCase):
    def test_events_use_completed_return_and_separate_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = fixture(Path(tmp))
            events = pd.read_csv(raw / 'ho_events.csv')
            outcomes, pending = event_outcomes(events, 11, 1.)
            self.assertEqual(outcomes, [[3.3], [2.6], [3.4]])
            self.assertEqual(pending[-1], (5., float('inf')))
            self.assertEqual(event_outcomes(events, 11, .2)[0][1], [])
            self.assertEqual(event_outcomes(events, 12, 1.)[0], [[], [], []])

    def test_multivariate_missingness_labels_and_cell_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = prepare_run(fixture(root), root / 'out.npz', 10, 10, 3, 300., 1.)
            with np.load(root / 'out.npz') as data:
                self.assertEqual(data['values'].shape, (5, 10, 32))
                self.assertFalse(data['mask'][0, 5, 4])
                self.assertFalse(data['mask'][0, 6, 4])
                self.assertFalse(data['mask'][:, :, -7:].any())
                self.assertEqual(data['values'][0, 0, 11], -81)
                self.assertEqual(data['values'][0, 0, 18], -82)
                self.assertTrue(data['labels'][2, 1])
                self.assertTrue(data['labels'][3, 0])
                self.assertTrue(data['labels'][3, 2])
                self.assertTrue(data['label_mask'][3, 2])
                self.assertFalse(data['label_mask'][data['labels'][:, 2] == 0, 2].any())
                self.assertFalse(data['label_mask'][4, 0])
                self.assertFalse(data['label_mask'][0, 1])
            self.assertEqual(info['cell_ids'], [1, 2])
            with self.assertRaises(FileExistsError):
                prepare_run(root / 'run', root / 'out.npz')

    def test_reject_incomplete_run_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture(root, completed=False)
            with self.assertRaisesRegex(ValueError, 'completion'):
                prepare_run(raw, root / 'out.npz', 10)
            manifest = json.loads((raw / 'run_manifest.json').read_text())
            manifest['completed'] = True
            (raw / 'run_manifest.json').write_text(json.dumps(manifest))
            frame = pd.read_csv(raw / 'ue_kpi.csv')
            pd.concat([frame, frame.iloc[:1]]).to_csv(raw / 'ue_kpi.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                prepare_run(raw, root / 'out.npz', 10)

    def test_zero_imsi_is_valid_in_pinned_nr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture(root)
            for name in ('ue_kpi.csv', 'ho_events.csv'):
                frame = pd.read_csv(raw / name)
                frame['imsi'] = 0
                frame.to_csv(raw / name, index=False)
            prepare_run(raw, root / 'out.npz', 10, 10)
            with np.load(root / 'out.npz') as data:
                self.assertTrue(data['label_mask'][0, 0])
                self.assertTrue(data['labels'][3, 0])

    def test_gap_windows_excluded_and_legacy_negatives_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = fixture(root)
            frame = pd.read_csv(raw / 'ue_kpi.csv').drop(index=5)
            frame.to_csv(raw / 'ue_kpi.csv', index=False)
            manifest = json.loads((raw / 'run_manifest.json').read_text())
            manifest.pop('failure_events_recorded')
            (raw / 'run_manifest.json').write_text(json.dumps(manifest))
            prepare_run(raw, root / 'out.npz', 10, 1)
            with np.load(root / 'out.npz') as data:
                meta = json.loads(str(data['metadata']))
                self.assertTrue(all(e['start_s'] >= .7 for e in meta['episodes']))
                negatives = data['labels'][:, 0] == 0
                self.assertFalse(data['label_mask'][negatives, 0].any())
                self.assertTrue(data['label_mask'][~negatives, 0].all())

    def test_training_normalization_and_partition_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, offset in [('train', 0), ('validation', 50)]:
                prepare_run(fixture(root, name, offset), root / name / 'episodes' / 'data.npz', 10, 10)
            a = HandoverDataset(root / 'train/episodes')
            b = HandoverDataset(root / 'validation/episodes', a.normalization)
            assert_disjoint(a, b)
            self.assertEqual(a.normalization, b.normalization)
            self.assertGreater(b.values[:, :, 4][b.mask[:, :, 4]].mean(), 10)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                assert_disjoint(a, a)
            b.runs[0]['manifest']['scenario_group'] = 'train'
            with self.assertRaisesRegex(ValueError, 'leakage'):
                assert_disjoint(a, b)



if __name__ == '__main__':
    unittest.main()
