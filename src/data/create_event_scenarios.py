"""Create diverse CSV candidates; events must be verified by NS-3, never assumed."""
import argparse
import copy
from pathlib import Path
import numpy as np
import yaml
from src.data.population import environment_slots
from src.data.route_csv import write_tracks, speed_warnings
from src.data.scenarios import project_path, save_json


def create(config, output, config_output, variants=15, seed=20261019):
    if variants < 5:
        raise ValueError('Need at least five variants per family')
    output, config_output = project_path(output), project_path(config_output)
    if output.exists() or config_output.exists():
        raise FileExistsError('Choose new CSV and config output paths')
    cfg = copy.deepcopy(config)
    env = cfg['environment']
    sites = np.asarray(env['sites'], dtype=float)
    if len(sites) != 2:
        raise ValueError('This candidate generator requires two fixed base stations')
    direction = sites[1,:2]-sites[0,:2]
    length = np.linalg.norm(direction)
    if length == 0: raise ValueError('Base stations must be separated')
    direction /= length
    across = np.array([-direction[1],direction[0]])
    center = sites[:,:2].mean(axis=0)
    duration = 16.
    env['duration_s'] = duration
    env['route_csv_root'] = str(output)
    env['scenarios'] = []
    env.pop('ue_offsets', None)
    env.pop('car_offsets', None)
    profiles = yaml.safe_load(project_path('configs/pipeline.yaml').read_text())['ue_attributes']
    env['ue_attributes'] = profiles
    rng = np.random.default_rng(seed)
    families = ['boundary_oscillation', 'boundary_pause', 'fast_crossing', 'coverage_escape']
    records = []
    for index in range(variants):
        partition = 'train' if index < round(variants*.6) else ('validation' if index < round(variants*.8) else 'evaluation')
        # Each parameter group belongs to one partition, including all four families.
        for family in families:
            amplitude = float(rng.uniform(35,48))
            crossing_s = float(rng.uniform(1.5,1.95))
            offset = float(rng.uniform(15,75))
            pause = float(rng.uniform(.1,.4)) if family == 'boundary_pause' else 0.
            name = f'{family}_{index:03d}'
            tracks = {}
            for kind, ue in environment_slots(env):
                if kind == 'car':
                    shift = .12*(ue-1)
                    y = offset + 8*(ue-1)
                    if family in ('boundary_oscillation','boundary_pause'):
                        # Alternate boundary crossing after warm-up, under 65 m/s.
                        points = [[0.,-amplitude,y],[3.+shift,-amplitude,y]]
                        t, sign = 3.+shift, 1
                        while t+crossing_s+pause < duration:
                            t += crossing_s
                            points.append([t,sign*amplitude,y])
                            if pause:
                                t += pause
                                points.append([t,sign*amplitude,y])
                            sign *= -1
                    else:
                        v = float(rng.uniform(35,65))
                        start = -length*.4
                        sign = 1 if family == 'fast_crossing' else -1
                        points = [[0.,start,y],[3.+shift,start,y],
                                  [duration,start+sign*v*(duration-3.-shift),y]]
                elif kind == 'walker':
                    x = float(rng.uniform(-80,80))
                    y = offset + 10*ue
                    v = float(rng.uniform(.4,1.8))
                    points = [[0.,x,y],[duration,x+v*duration,y]]
                else:
                    points = [[0.,-length*.35,20.*ue],[duration,-length*.35,20.*ue]]
                if points[-1][0] < duration: points.append([duration,*points[-1][1:]])
                tracks[f'{kind}:{ue}'] = [[t,*(center+x*direction+y*across)] for t,x,y in points]
            violations = speed_warnings(tracks,profiles)
            if violations: raise ValueError(violations)
            write_tracks(output/name,tracks,env)
            env['scenarios'].append({'name':name,'partition':partition,'group':f'variant_{index:03d}'})
            records.append(dict(name=name,partition=partition,family=family,amplitude_m=amplitude,
                                crossing_s=crossing_s,pause_s=pause,offset_m=offset,status='unverified'))
    cfg['batch_dir'] = str(project_path('data/processed')/output.name)
    config_output.parent.mkdir(parents=True,exist_ok=True)
    config_output.write_text(yaml.safe_dump(cfg,sort_keys=False))
    save_json(output/'candidates.json',dict(seed=seed,candidates=records,config=str(config_output)))
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/training_dataset_v2.yaml')
    p.add_argument('--output-dir',default='data/scenarios/new_event_candidates')
    p.add_argument('--output-config',default='configs/new_event_candidates.yaml')
    p.add_argument('--variants',type=int,default=15)
    p.add_argument('--seed',type=int,default=20261019)
    a=p.parse_args()
    cfg=create(yaml.safe_load(project_path(a.config).read_text()),a.output_dir,a.output_config,a.variants,a.seed)
    print(f"Created {len(cfg['environment']['scenarios'])} unverified candidates: {a.output_config}")


if __name__=='__main__': main()
