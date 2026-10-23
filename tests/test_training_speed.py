import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.train.train_mobility import early_stopping, resume_compatible
from src.simulation.runner import read_events, event_score


class EarlyStoppingTests(unittest.TestCase):
    def test_patience_resume_partial_and_improvement(self):
        def row(score, status='complete'):
            return {'validation': {'reward': score, 'status': status}}
        settings = {'patience': 2, 'min_delta': .01}
        h = [row(.5), row(.5), row(None, 'invalid'), {'update': 3}]
        self.assertFalse(early_stopping(h, settings)['stop'])
        h.append(row(.505))
        self.assertTrue(early_stopping(h, settings)['stop'])
        self.assertEqual(early_stopping(json.loads(json.dumps(h)), settings), early_stopping(h, settings))
        h.append(row(.7))
        self.assertEqual(early_stopping(h, settings)['stale_validations'], 0)
        self.assertFalse(early_stopping([row(1)]*5, {})['stop'])
        with self.assertRaises(ValueError):
            early_stopping([], {'patience': -1})

    def test_only_stopping_settings_are_resume_compatible(self):
        original = {'training': {'updates': 20, 'samples_per_episode': 4}}
        updated = copy.deepcopy(original)
        updated['training']['early_stopping'] = {'patience': 2}
        resume_compatible(original, updated)
        updated['training']['stop_on_danger'] = True
        with self.assertRaises(ValueError):
            resume_compatible(original, updated)


class EarlyDangerTests(unittest.TestCase):
    def fixture(self, root, event='ho_failure', time=4.):
        manifest = dict(completed=False, stopTime=13., actualStopTime=time,
                        termination_reason='confirmed_danger', failure_events_recorded=True,
                        early_stop=dict(event=event, imsi=1, time_s=time, reward_start_s=3., pingpong_window_s=2.))
        (root/'run_manifest.json').write_text(json.dumps(manifest))
        pd.DataFrame([dict(global_ue_id=uid, imsi=uid, time_ms=100) for uid in [1,2]]).to_csv(root/'ue_kpi.csv',index=False)
        rows = [(time,'HandoverFailureNoPreamble',1,1,0)] if event=='ho_failure' else [
            (2.,'start',1,1,3), (2.5,'end_ok',1,3,0), (3.,'start',1,3,1), (time,'end_ok',1,1,0)]
        pd.DataFrame(rows, columns=['time_s','event','imsi','cell_id','target_cell_id']).to_csv(root/'ho_events.csv',index=False)
        return manifest

    def test_positive_only_and_default_rejection(self):
        for event in ['ho_failure','pingpong']:
            with self.subTest(event=event), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); self.fixture(root,event)
                with self.assertRaises(ValueError): read_events(root,2,3.,13.,2.)
                y, mask=read_events(root,2,3.,13.,2.,allow_early=True)
                self.assertEqual(int(y.sum()),1)
                np.testing.assert_array_equal(mask,y) # unobserved negatives remain unknown
                reward, success, _=event_score(y,np.ones_like(y),y,mask)
                self.assertEqual(reward,1.)
                self.assertTrue(success)

    def test_bad_proof_and_outside_window_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); m=self.fixture(root)
            m['early_stop']['imsi']=99
            (root/'run_manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError): read_events(root,2,3.,13.,2.,True)
            self.fixture(root,time=2.)
            with self.assertRaises(ValueError): read_events(root,2,3.,13.,2.,True)

    def test_full_negative_still_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); m=self.fixture(root,time=2.)
            m.update(completed=True,actualStopTime=13.,termination_reason='duration_complete')
            (root/'run_manifest.json').write_text(json.dumps(m))
            y, mask=read_events(root,2,3.,13.,2.,True)
            self.assertEqual(int(y.sum()),0)
            reward, _, _=event_score(y,np.ones_like(y),y,mask)
            self.assertEqual(reward,-1.)
