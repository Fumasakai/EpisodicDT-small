"""Shared CSV contract for the route editor and simulation input loader."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.population import KINDS, environment_slots

COLUMNS = ['timestamp', 'ue_id', 'location_x', 'location_y']


def validate_tracks(tracks, env):
    expected = {f'{kind}:{ue}' for kind, ue in environment_slots(env)}
    if set(tracks) != expected:
        raise ValueError('UE IDs/counts must match environment.population in every scenario')
    result = {}
    for key, points in tracks.items():
        a = np.asarray(points, dtype=float)
        if (a.ndim != 2 or a.shape[1] != 3 or len(a) < 2 or not np.isfinite(a).all()
                or a[0, 0] != 0 or abs(a[-1, 0] - env['duration_s']) > 1e-8
                or not (np.diff(a[:, 0]) > 0).all()):
            raise ValueError(f'{key}: require finite [time_s,x_m,y_m], strictly increasing from 0 to duration_s')
        result[key] = a.tolist()
    return result


def read_tracks(directory, env):
    tracks = {}
    for kind in KINDS:
        path = Path(directory) / f'{kind}.csv'
        frame = pd.read_csv(path)
        if list(frame.columns) != COLUMNS:
            raise ValueError(f'{path}: expected columns {COLUMNS}')
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'{path}: nonfinite CSV values')
        ids = frame.ue_id.to_numpy(dtype=float)
        if ((ids < 1) | (ids != np.floor(ids))).any():
            raise ValueError(f'{path}: ue_id must be a positive integer')
        for ue, group in frame.groupby('ue_id', sort=True):
            tracks[f'{kind}:{int(ue)}'] = group[['timestamp', 'location_x', 'location_y']].to_numpy().tolist()
    return validate_tracks(tracks, env)


def write_tracks(directory, tracks, env):
    tracks = validate_tracks(tracks, env)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        rows = []
        for slot_kind, ue in environment_slots(env):
            if slot_kind == kind:
                rows.extend([t, ue, x, y] for t, x, y in tracks[f'{kind}:{ue}'])
        path = directory / f'{kind}.csv'
        tmp = path.with_suffix('.csv.tmp')
        pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False)
        tmp.replace(path)


def speed_warnings(tracks, profiles):
    warnings = []
    for key, points in tracks.items():
        a = np.asarray(points, dtype=float)
        profile = profiles[key.split(':')[0]]
        lo, hi = profile['min_moving_speed_mps'], profile['max_speed_mps']
        for i, speed in enumerate(np.linalg.norm(np.diff(a[:, 1:], axis=0), axis=1) / np.diff(a[:, 0])):
            stopped = speed <= 1e-8
            if (stopped and not profile['allow_stop']) or (not stopped and (speed < lo - 1e-8 or speed > hi + 1e-8)):
                warnings.append(f'{key} segment {i+1}: {speed:.4g} m/s (moving range {lo:g}..{hi:g})')
    return warnings
