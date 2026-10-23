"""Scenario input files and provenance helpers for handover experiments."""
import hashlib
import json
from pathlib import Path
import re
import warnings
import shutil

import numpy as np
import pandas as pd

from src.data.population import population_offsets

PROJECT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def route_directory(env, scenario):
    """Resolve one shared CSV root or a legacy per-scenario directory."""
    if 'route_csv_root' in env:
        if 'route_csv_dir' in scenario:
            raise ValueError('Use route_csv_root or route_csv_dir, not both')
        name = scenario['name']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('Invalid scenario name')
        return project_path(env['route_csv_root']) / name
    if 'route_csv_dir' in scenario:
        return project_path(scenario['route_csv_dir'])
    return None


def validate_environment(env):
    for key in ('duration_s', 'frequency_hz', 'bandwidth_hz'):
        if not np.isfinite(env[key]) or env[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not np.isfinite(env['gnb_tx_power_dbm']):
        raise ValueError('Transmit power must be finite')
    population_offsets(env)
    sites = env.get('sites')
    if not sites or any(len(x) != 3 or not np.isfinite(x).all() or x[2] <= 1.5 for x in sites):
        raise ValueError('environment.sites must contain fixed [x,y,height>1.5]')
    names = set()
    for scenario in env['scenarios']:
        name = scenario['name']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or name in names:
            raise ValueError('Scenario names must be unique safe identifiers')
        names.add(name)
        if 'sites' in scenario:
            raise ValueError('Per-scenario sites are forbidden; use environment.sites in YAML')
        source = route_directory(env, scenario)
        if source is not None:
            if any(k in scenario for k in ('car_route', 'walker_route', 'static_position')):
                raise ValueError('route_csv_dir cannot be combined with route templates')
            from src.data.route_csv import read_tracks
            read_tracks(source, env)
            continue
        for key in ('car_route', 'walker_route'):
            route = np.asarray(scenario[key], dtype=float)
            if (route.ndim != 2 or route.shape[1] != 3 or len(route) < 2 or not np.isfinite(route).all()
                    or route[0, 0] != 0 or route[-1, 0] != 1 or not (np.diff(route[:, 0]) > 0).all()):
                raise ValueError(f'{name}/{key}: require increasing [fraction,x,y], from 0 to 1')
        if len(scenario['static_position']) != 2 or not np.isfinite(scenario['static_position']).all():
            raise ValueError('static_position must be [x,y]')
    if not names:
        raise ValueError('No scenarios selected')


def write_inputs(cfg, scenario, path):
    source = route_directory(cfg, scenario)
    if source is not None:
        from src.data.route_csv import read_tracks, speed_warnings
        tracks = read_tracks(source, cfg)
        if 'ue_attributes' in cfg:
            for message in speed_warnings(tracks, cfg['ue_attributes']):
                warnings.warn(f"{scenario['name']}: {message}", UserWarning)
        path.mkdir(parents=True, exist_ok=False)
        for kind in ('car', 'walker', 'static'):
            shutil.copyfile(source / f'{kind}.csv', path / f'{kind}.csv')
        return
    offsets_by_kind = population_offsets(cfg)
    path.mkdir(parents=True, exist_ok=False)
    for kind, route, offsets in (
        ('car', scenario['car_route'], offsets_by_kind['car']),
        ('walker', scenario['walker_route'], offsets_by_kind['walker']),
        ('static', [[0, *scenario['static_position']], [1, *scenario['static_position']]], offsets_by_kind['static']),
    ):
        records = [{'timestamp': fraction * cfg['duration_s'], 'ue_id': ue,
                    'location_x': x + dx, 'location_y': y + dy}
                   for ue, (dx, dy) in enumerate(offsets, 1) for fraction, x, y in route]
        pd.DataFrame(records, columns=['timestamp', 'ue_id', 'location_x', 'location_y']).to_csv(path / f'{kind}.csv', index=False)


