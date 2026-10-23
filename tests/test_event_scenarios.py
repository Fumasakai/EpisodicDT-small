import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess
import yaml
from src.data.create_event_scenarios import create
from src.data.screen_event_scenarios import screen
from src.data.scenarios import PROJECT, validate_environment
from src.data.route_csv import read_tracks, speed_warnings


class EventScenarioTests(unittest.TestCase):
    def test_diversity_speed_layout_and_split_groups(self):
        base=yaml.safe_load((PROJECT/'configs/training_dataset_v2.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=create(base,root/'routes',root/'config.yaml',variants=5)
            env=cfg['environment']
            validate_environment(env)
            self.assertEqual(env['sites'],base['environment']['sites'])
            self.assertEqual(env['population'],base['environment']['population'])
            self.assertEqual(len(env['scenarios']),20)
            groups={};signatures=set()
            for s in env['scenarios']:
                groups.setdefault(s['group'],set()).add(s['partition'])
                tracks=read_tracks(root/'routes'/s['name'],env)
                self.assertEqual(len(tracks),24)
                self.assertFalse(speed_warnings(tracks,env['ue_attributes']))
                signatures.add(str(tracks))
            self.assertEqual(len(signatures),20)
            self.assertTrue(all(len(v)==1 for v in groups.values()))
            self.assertEqual({s['partition'] for s in env['scenarios']},{'train','validation','evaluation'})
            with self.assertRaises(FileExistsError): create(base,root/'routes',root/'config.yaml')

    def test_failed_simulation_is_not_a_negative_or_positive_event(self):
        cfg=yaml.safe_load((PROJECT/'configs/training_dataset_v2.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp, patch('src.data.screen_event_scenarios.run_batch',side_effect=subprocess.CalledProcessError(-11,['ns3'])):
            report=screen(cfg,Path(tmp)/'screen',limit=2,runs=[1])
            self.assertEqual(len(report),2)
            self.assertTrue(all(r['status']=='invalid' and not r['event_counts'] for r in report))
            self.assertFalse((Path(tmp)/'screen/completed_scenarios.yaml').exists())


if __name__=='__main__': unittest.main()
