import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.prepare_ns3 import prepare_ns3
from src.data.dataset import RSRPEpisodeDataset
from src.train.train import temporal_train_validation_indices


class PrepareNs3Tests(unittest.TestCase):
    def make_source(self, root, frame):
        source = root / 'tranData-test'
        source.mkdir()
        frame.to_csv(source/'ue_kpi.csv', index=False)
        (source/'run_manifest.json').write_text(json.dumps({'samplePeriod': .2}))
        return source

    def test_separate_sort_gaps_and_stale_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=pd.DataFrame({'global_ue_id':[1]*6+[2]*3,
                                'time_ms':[200,400,600,800,1000,1200,200,400,600],
                                'RSRP':[-90,-91,np.nan,-93,-94,-95,-100,-101,-102],
                                'rsrp_age_ms':[200,200,0,200,800,200,200,200,200]})
            source=self.make_source(root,frame.iloc[::-1])
            report=prepare_ns3(source,root/'out',max_age_ms=300,episode_length=2)
            self.assertEqual(report['invalid_rsrp_rows'],1)
            self.assertEqual(report['rejected_age_rows'],1)
            self.assertEqual([x['samples'] for x in report['segments']],[2,1,1,3])
            for seg, expected in zip(report['segments'],[[-90,-91],[-93],[-95],[-100,-101,-102]]):
                result=pd.read_csv(seg['file'])
                self.assertEqual(list(result.columns),['rsrp'])
                self.assertEqual(result.rsrp.tolist(),expected)
            with self.assertRaises(FileExistsError):
                prepare_ns3(source,root/'out',max_age_ms=300)

    def test_dataset_training_compatibility_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=self.make_source(root,pd.DataFrame({'global_ue_id':[1]*250,
                'time_ms':np.arange(250)*200,'RSRP':np.linspace(-100,-80,250)}))
            report=prepare_ns3(source,root/'out')
            dataset=RSRPEpisodeDataset(root/'out/train',40)
            train,val=temporal_train_validation_indices(dataset,.2)
            self.assertEqual(report['training_windows'],len(train))
            self.assertEqual(report['validation_windows'],len(val))
            self.assertEqual(tuple(dataset[0].shape),(40,1))
            frame=pd.read_csv(source/'ue_kpi.csv')
            pd.concat([frame,frame.iloc[:1]]).to_csv(source/'ue_kpi.csv',index=False)
            with self.assertRaises(ValueError):
                prepare_ns3(source,root/'other')
