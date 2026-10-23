import copy
import tempfile
import unittest
from pathlib import Path

from src.tools.route_editor import EditorProject
from scenario_fixture import editor_config
from src.data.route_csv import read_tracks, speed_warnings
from src.data.scenarios import write_inputs
from src.data.generate_handover_batch import validate, run_batch
import yaml


class RouteEditorTests(unittest.TestCase):
    def test_save_load_csv_prepare_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = EditorProject(editor_config(tmp), None, Path(tmp)/'editor')
            state = project.state()
            name = state['scenarios'][0]['name']
            state['tracks'][name]['car:1'] = [[0, 12, 34], [10, 112, 34], [20, 112, 34], [60, 212, 34]]
            # Give another car an independent route: CSV import must not apply offsets/templates.
            state['tracks'][name]['car:2'] = [[0, -20, 50], [60, 100, 90]]
            payload = dict(counts=state['population']['counts'], tracks=state['tracks'])
            saved = project.save(payload)
            cfg = yaml.safe_load(Path(saved['config']).read_text())
            validate(cfg)
            first = cfg['environment']['scenarios'][0]
            self.assertNotIn('car_route', first)
            source = Path(cfg['environment']['route_csv_root']) / first['name']
            dest = Path(tmp)/'copied'
            write_inputs(cfg['environment'], first, dest)
            self.assertEqual((source/'car.csv').read_bytes(), (dest/'car.csv').read_bytes())
            self.assertEqual(read_tracks(dest,cfg['environment'])['car:2'], payload['tracks'][name]['car:2'])
            binary = Path(tmp)/'ns3/build/scratch/ns3.46-tranData'
            binary.parent.mkdir(parents=True)
            binary.write_text('preparation only')
            cfg.update(ns3_dir=str(Path(tmp)/'ns3'), batch_dir=str(Path(tmp)/'batch'))
            run_batch(cfg, prepare_only=True)
            training = yaml.safe_load((Path(tmp)/'batch/training.yaml').read_text())
            self.assertEqual(training['ue_attributes'],state['profiles'])
            run_batch(cfg, prepare_only=True)
            csv = source/'car.csv'
            csv.write_text(csv.read_text().replace('112.0', '113.0'))
            with self.assertRaisesRegex(ValueError,'changed'):
                run_batch(cfg, prepare_only=True)
            saved2 = project.save(payload)
            self.assertNotEqual(saved2['revision_config'],saved['revision_config'])

    def test_shared_root_and_config_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_config = root/'configs/routes.yaml'
            project = EditorProject(editor_config(tmp), None, root/'routes', output_config)
            state = project.state()
            saved = project.save(dict(counts=state['population']['counts'], tracks=state['tracks']))
            self.assertEqual(saved['config'], str(output_config))
            cfg = yaml.safe_load(output_config.read_text())
            validate(cfg)
            for scenario in cfg['environment']['scenarios']:
                self.assertNotIn('route_csv_dir', scenario)
                self.assertTrue((Path(cfg['environment']['route_csv_root'])/scenario['name']/'car.csv').is_file())
            cfg['environment']['scenarios'][0]['route_csv_dir'] = '/tmp/conflicting'
            with self.assertRaisesRegex(ValueError, 'not both'):
                validate(cfg)
            del cfg['environment']['scenarios'][0]['route_csv_dir']
            cfg['environment']['route_csv_root'] = str(root/'missing')
            with self.assertRaises(FileNotFoundError):
                validate(cfg)

    def test_validation_and_speed_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=EditorProject(editor_config(tmp),None,tmp)
            state=project.state()
            payload=dict(counts=state['population']['counts'],tracks=state['tracks'])
            name=state['scenarios'][0]['name']
            payload['tracks'][name]['walker:1']=[[0,0,0],[1,100,0],[60,100,0]]
            result=project.save(payload)
            self.assertTrue(any('walker:1' in w for w in result['warnings']))
            for points in ([[0,0,0],[0,1,0],[60,1,0]], [[1,0,0],[60,0,0]], [[0,0,0],[60,float('nan'),0]]):
                bad=copy.deepcopy(payload)
                bad['tracks'][name]['walker:1']=points
                with self.assertRaises(ValueError): project.save(bad)
            bad=copy.deepcopy(payload)
            del bad['tracks'][name]['car:1']
            with self.assertRaises(ValueError): project.save(bad)
            self.assertEqual(speed_warnings({'walker:1': [[0,0,0],[1,0,0],[2,1,0]]},state['profiles']),[])


if __name__=='__main__':
    unittest.main()
