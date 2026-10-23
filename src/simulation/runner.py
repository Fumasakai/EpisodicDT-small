"""One source of event truth: run ns-3 on waypoint files and parse its traces."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.prepare_handover import LABELS, FAILURES, event_outcomes
from src.data.scenarios import digest, project_path, save_json


def export_routes(root, waypoints, slots, profiles):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 3 or points.shape[0] != len(slots) or points.shape[2] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite [UE, waypoint, (time_s,x_m,y_m)]')
    if not np.allclose(points[:, 0, 0], 0) or not (np.diff(points[..., 0], axis=1) > 0).all():
        raise ValueError('Waypoint times must start at zero and increase')
    if not np.allclose(points[..., 0], points[0, :, 0]):
        raise ValueError('UE waypoint times must be synchronized')
    frames, routes = {k: [] for k in ('car', 'walker', 'static')}, []
    for uid, ((kind, source_id), path, profile) in enumerate(zip(slots, points, profiles), 1):
        speed = np.linalg.norm(np.diff(path[:, 1:], axis=0), axis=-1) / np.diff(path[:, 0])
        moving = speed > 1e-4
        if ((speed > profile['max_speed_mps'] + 1e-3).any()
                or (speed[moving] < profile['min_moving_speed_mps'] - 1e-3).any()
                or (not profile['allow_stop'] and (~moving).any())):
            raise ValueError('Generated route violates UE speed profile')
        frames[kind].extend({'timestamp': t, 'ue_id': source_id, 'location_x': x, 'location_y': y} for t,x,y in path)
        routes.append({'global_ue_id': uid, 'kind': kind, 'source_ue_id': source_id,
                       'waypoints': path.tolist(), 'interval_modes': ['move' if x else 'stop' for x in moving],
                       'interval_speeds_mps': speed.tolist()})
    for kind, rows in frames.items():
        pd.DataFrame(rows, columns=['timestamp', 'ue_id', 'location_x', 'location_y']).to_csv(root/f'{kind}.csv', index=False)
    save_json(root/'routes.json', {'format': 'ue_waypoints_v2', 'columns': ['time_s', 'x_m', 'y_m'], 'ues': routes})


def plot_routes(root, waypoints, slots, sites):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for path, (kind, uid) in zip(waypoints, slots):
        ax.plot(path[:, 1], path[:, 2], '.-', label=f'{kind}_{uid}')
        ax.scatter(path[0, 1], path[0, 2], marker='s')
    sites = np.asarray(sites)
    ax.scatter(sites[:, 0], sites[:, 1], marker='^', color='black', label='gNB')
    ax.set(xlabel='x [m]', ylabel='y [m]'); ax.set_aspect('equal', adjustable='datalim')
    ax.legend(); fig.tight_layout(); fig.savefig(Path(root)/'routes.png'); plt.close(fig)


def read_events(raw, num_ues, start_s, end_s, pingpong_s, allow_early=False):
    raw = Path(raw)
    manifest = json.loads((raw/'run_manifest.json').read_text())
    early = allow_early and manifest.get('termination_reason') == 'confirmed_danger'
    if (not manifest.get('completed') and not early) or float(manifest['stopTime']) + 1e-6 < end_s:
        raise ValueError('Incomplete simulation cannot produce a reward')
    kpi, events = pd.read_csv(raw/'ue_kpi.csv'), pd.read_csv(raw/'ho_events.csv')
    if set(kpi.global_ue_id.unique()) != set(range(1, num_ues+1)):
        raise ValueError('Simulator UE mapping differs from policy')
    complete = bool(manifest.get('failure_events_recorded', False)) and not (
        events.event.isin(FAILURES) & ~events.imsi.isin(kpi.imsi.dropna().unique())).any()
    labels, masks = [], []
    for uid in range(1, num_ues+1):
        frame = kpi[kpi.global_ue_id == uid]
        ids = frame.imsi.dropna().unique()
        if len(ids) != 1:
            raise ValueError('Ambiguous simulated UE identity')
        times, pending = event_outcomes(events, ids[0], pingpong_s)
        y = np.array([any(start_s <= t < end_s for t in group) for group in times], dtype=bool)
        mask = np.array([complete and not any(a < end_s <= b for a,b in pending),
                         start_s >= frame.time_ms.min()/1000 + pingpong_s,
                         complete and bool(manifest.get('rlf_detection_validated', False))]) | y
        labels.append(y); masks.append(y.copy() if early else mask)
    labels, masks = np.array(labels), np.array(masks)
    if early:
        proof = manifest['early_stop']
        if (proof['event'] not in ('ho_failure', 'pingpong') or proof['reward_start_s'] != start_s
                or proof['pingpong_window_s'] != pingpong_s
                or not start_s <= proof['time_s'] < end_s
                or abs(manifest['actualStopTime'] - proof['time_s']) > 1e-6):
            raise ValueError('Invalid early termination proof')
        observed, _ = event_outcomes(events, proof['imsi'], pingpong_s)
        if not any(abs(t-proof['time_s']) < 1e-8 for t in observed[LABELS.index(proof['event'])]):
            raise ValueError('Early termination event missing from trace')
        if not labels[:, LABELS.index(proof['event'])].any():
            raise ValueError('Early termination event does not belong to a mapped UE')
    return labels, masks


class UnknownDangerOutcome(ValueError):
    """No danger observed, but missing labels prevent a negative conclusion."""


def event_score(target, target_mask, observed, observed_mask,
                events=('ho_failure', 'pingpong')):
    """Any known danger on any UE earns +1, irrespective of source labels/UE IDs.

    Source target arguments are retained for trace provenance, not reward matching.
    The third return value is the fraction of UEs with confirmed danger.
    """
    observed, known = np.asarray(observed, bool), np.asarray(observed_mask, bool)
    if (observed.ndim != 2 or observed.shape[0] < 1 or observed.shape[1] != len(LABELS)
            or known.shape != observed.shape):
        raise ValueError('Invalid observed event shape')
    if not events or len(set(events)) != len(events) or any(e not in ('ho_failure', 'pingpong') for e in events):
        raise ValueError('Require validated danger events: ho_failure and/or pingpong')
    selected = [LABELS.index(e) for e in events]
    danger = (observed[:, selected] & known[:, selected]).any(1)
    if danger.any():
        return 1., True, float(danger.mean())
    if not known[:, selected].all():
        raise UnknownDangerOutcome('No confirmed danger, but selected event labels are unknown')
    return -1., False, 0.


class Simulator:
    def __init__(self, config, slots, profiles, warmup_s, duration_s,
                 events=("ho_failure", "pingpong"), binary=None):
        self.events = tuple(events)
        self.config, self.slots, self.profiles = config, slots, profiles
        self.warmup_s, self.duration_s = warmup_s, duration_s
        self.ns3 = project_path(config['ns3_dir'])
        self.binary = project_path(binary) if binary else self.ns3/'build/scratch/ns3.46-tranData'
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        self.build = {'binary': digest(self.binary), 'libraries': {
            p.name: digest(p) for p in sorted((self.ns3/'build/lib').glob('libns3*.so'))}}

    def try_evaluate(self, *args, **kwargs):
        """Keep failed jobs as auditable invalid samples, never as event negatives."""
        try:
            return self.evaluate(*args, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, UnknownDangerOutcome):
            return None

    def evaluate(self, points, target, mask, root, runs, seed=None, stop_on_danger=False):
        root = Path(root)
        if not runs or len(set(runs)) != len(runs) or any(type(r) is not int or r < 1 for r in runs):
            raise ValueError('Require distinct positive ns-3 run numbers')
        if not np.isclose(np.asarray(points)[0, -1, 0], self.duration_s):
            raise ValueError('Wrong simulation duration')
        export_routes(root, points, self.slots, self.profiles)
        env, ctl, rec = (self.config[k] for k in ('environment', 'controller', 'recording'))
        dist = {'pkt_size_bytes': {'values': [env['packet_bytes']], 'probs': [1]},
                'pps': {'values': [env['packets_per_second']], 'probs': [1]}}
        save_json(root/'traffic.json', {s: {'dl': dist, 'ul': dist} for s in ('URLLC', 'eMBB', 'mMTC')})
        records = []
        save_json(root/'request.json', {'target': np.asarray(target, int).tolist(),
                  'target_mask': np.asarray(mask, bool).tolist(), 'labels': LABELS,
                  'reward_mode': 'any_danger', 'reward_events': self.events,
                  'window_s': [self.warmup_s, self.duration_s], 'stop_on_danger': stop_on_danger, 'build': self.build, 'simulation': self.config})
        for run in runs:
            raw = root/f'run_{run}'
            options = {'Layer': rec['layer'], 'enableHo': int(ctl['type'] == 'a3'),
                       'a3HysteresisDb': ctl['hysteresis_db'], 'a3TimeToTriggerMs': ctl['time_to_trigger_ms'],
                       'shadowingEnabled': int(env.get('shadowing_enabled', False)),
                       'stopTime': self.duration_s, 'samplePeriod': rec['kpi_period_ms']/1000,
                       'nrMaxUes': 0, 'gnbPositions': json.dumps(env['sites'], separators=(',', ':')),
                       'carCsv': str(root/'car.csv'), 'walkerCsv': str(root/'walker.csv'),
                       'staticCsv': str(root/'static.csv'), 'offeredLoadJson': str(root/'traffic.json'),
                       'outputDir': str(raw), 'nrScenario': env['scenario'], 'frequencyHz': env['frequency_hz'],
                       'bandwidthHz': env['bandwidth_hz'], 'gnbTxPowerDbm': env['gnb_tx_power_dbm'],
                       'seed': self.config['execution']['seed'] if seed is None else seed, 'run': run}
            if stop_on_danger:
                if not set(self.events) <= {'ho_failure', 'pingpong'}:
                    raise ValueError('Early termination supports only HOF and pingpong')
                options.update(stopOnHof=int('ho_failure' in self.events),
                               stopOnPingpong=int('pingpong' in self.events),
                               rewardStart=self.warmup_s,
                               pingpongWindow=self.config['labeling']['pingpong_window_s'])
            try:
                with (root/f'run_{run}.log').open('w') as log:
                    subprocess.run([str(self.binary)] + [f'--{k}={v}' for k,v in options.items()], cwd=self.ns3,
                                   stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=self.config['execution'].get('timeout_s', 600))
                y, known = read_events(raw, len(self.slots), self.warmup_s, self.duration_s,
                                       self.config['labeling']['pingpong_window_s'], allow_early=stop_on_danger)
                manifest = json.loads((raw/'run_manifest.json').read_text())
                if manifest.get('termination_reason') == 'confirmed_danger' and manifest['early_stop']['event'] not in self.events:
                    raise ValueError('Early termination used an event outside the reward scope')
                reward, success, fraction = event_score(target, mask, y, known, self.events)
            except Exception as exc:
                save_json(root/'failure.json', {'error': str(exc), 'run': run, 'infrastructure_failure': not isinstance(exc, UnknownDangerOutcome)})
                raise  # Never turn crashes/timeouts into a network-event penalty.
            records.append({'run': run, 'seed': options['seed'], 'observed': y.astype(int).tolist(),
                            'known': known.tolist(), 'reward': reward, 'danger_success': success,
                            'dangerous_ue_fraction': fraction if manifest.get('completed') else None,
                                'termination_reason': manifest.get('termination_reason', 'duration_complete'),
                                'actual_stop_s': manifest.get('actualStopTime', self.duration_s), 'outputs_sha256': {
                                p.name: digest(p) for p in raw.iterdir() if p.is_file()}})
        result = {'mean_reward': float(np.mean([r['reward'] for r in records])),
                  'danger_success_rate': float(np.mean([r['danger_success'] for r in records])), 'runs': records}
        save_json(root/'result.json', result)
        return result
