"""Small synthetic scenarios for unit tests, independent of archived experiments."""
from pathlib import Path
import yaml
from src.data.scenarios import PROJECT


def config(counts=None):
    cfg = yaml.safe_load((PROJECT/'configs/training_dataset_v2.yaml').read_text())
    env = cfg['environment']
    env.pop('route_csv_root', None)
    env.update(duration_s=60, packets_per_second=100)
    env['population'] = dict(min_ues=1, max_ues=12,
                             counts=counts or dict(car=1, walker=1, static=1))
    env['scenarios'] = [dict(name=name, partition=partition,
        car_route=[[0,30,40],[.5,1030,40],[1,30,40]],
        walker_route=[[0,20,10],[1,80,10]], static_position=[20,20])
        for name,partition in [('crossing_train','train'),('crossing_validation','validation'),
                               ('boundary_evaluation','evaluation')]]
    cfg['execution']['runs'] = [1]
    return cfg


def editor_config(directory):
    path = Path(directory)/'editor_fixture.yaml'
    path.write_text(yaml.safe_dump(config(dict(car=4,walker=4,static=2))))
    return path
