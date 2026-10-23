"""Synchronized UE groups with measured trajectories and ns-3 event supervision."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.handover_dataset import HandoverDataset, assert_disjoint
from src.data.scenarios import digest, project_path
from src.data.population import environment_slots

KINDS = ('car', 'walker', 'static')
SLICES = {'URLLC': 'car', 'eMBB': 'walker', 'mMTC': 'static'}


class ScenarioDataset(Dataset):
    def __init__(self, path, config, normalization=None):
        self.episodes = HandoverDataset(project_path(path), normalization)
        self.normalization = self.episodes.normalization
        self.schema = self.episodes.signature
        self.dt = self.schema['sample_period_s']
        sim = config['simulation']
        if (sim['recording']['kpi_period_ms'] / 1000 != self.dt
                or sim['recording']['max_age_ms'] != self.schema['max_age_ms']
                or sim['labeling']['pingpong_window_s'] != self.schema['pingpong_s']
                or sim['episodes']['length'] != self.schema['episode_length']):
            raise ValueError('Simulation and episode time schemas differ')
        self.warmup = float(config['warmup_s'])
        prefix = round(self.warmup / self.dt)
        if self.warmup < self.schema['pingpong_s'] or not np.isclose(prefix * self.dt, self.warmup):
            raise ValueError('Warmup must cover event history and be a multiple of sample period')
        self.points = prefix + self.schema['episode_length'] + 1
        self.arrays = {k: [] for k in ('values', 'mask', 'xy', 'labels', 'label_mask')}
        self.metadata, self.slots = [], None
        self.source_hashes = {}
        env = config['simulation']['environment']
        offset = 0
        for run in self.episodes.runs:
            raw = Path(run['source'])
            manifest = run['manifest']
            for name, sha in run['source_hashes'].items():
                if digest(raw / name) != sha:
                    raise ValueError(f'Changed source: {raw / name}')
            self.source_hashes[str(raw)] = run['source_hashes']
            sites = pd.read_csv(raw / 'gnb_positions.csv').sort_values('gnb_idx')[['x', 'y', 'z']].values
            if sites.shape != np.asarray(env['sites']).shape or not np.allclose(sites, env['sites'], atol=1e-6, rtol=0):
                raise ValueError('Fixed base station geometry differs from data')
            expected = {'gnbPlacement': 'fixed_yaml', 'nrScenario': env['scenario'],
                        'frequencyHz': env['frequency_hz'], 'bandwidthHz': env['bandwidth_hz'],
                        'gnbTxPowerDbm': env['gnb_tx_power_dbm'],
                        'shadowingEnabled': env.get('shadowing_enabled', False),
                        'enableHo': config['simulation']['controller']['type'] == 'a3',
                        'Layer': sim['recording']['layer'], 'nrUeTxPowerDbm': 23,
                        'a3HysteresisDb': config['simulation']['controller']['hysteresis_db'],
                        'a3TimeToTriggerMs': config['simulation']['controller']['time_to_trigger_ms']}
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError('Radio/controller configuration differs from source')
            mapping = sorted(manifest['ue_mapping'], key=lambda m: m['global_ue_id'])
            slots = [(SLICES[m['slice']], m['source_ue_id']) for m in mapping]
            if ([m['global_ue_id'] for m in mapping] != list(range(1, len(mapping) + 1))
                    or slots != sorted(slots, key=lambda s: (KINDS.index(s[0]), s[1]))):
                raise ValueError('Require contiguous global IDs and car/walker/static slot order')
            if self.slots is not None and self.slots != slots:
                raise ValueError('This model requires a fixed UE count, type, and slot order')
            if 'population' in env and slots != environment_slots(env):
                raise ValueError('Episode UE layout differs from configured population; '
                                 'regenerate data in a new batch directory')
            self.slots = slots
            kpi = pd.read_csv(raw / 'ue_kpi.csv')
            for col, value in [('dl_packet_bytes', env['packet_bytes']), ('ul_packet_bytes', env['packet_bytes']),
                               ('dl_pps', env['packets_per_second']), ('ul_pps', env['packets_per_second'])]:
                if not np.allclose(kpi[col], value):
                    raise ValueError('Traffic differs from fixed generation environment')
            groups = {}
            for j, episode in enumerate(run['episodes']):
                groups.setdefault(round(episode['start_s'], 8), {})[episode['global_ue_id']] = offset + j
            for start, group in sorted(groups.items()):
                if start < self.warmup + self.dt - 1e-6 or set(group) != {m['global_ue_id'] for m in mapping}:
                    continue
                query = start - self.warmup + np.arange(self.points) * self.dt
                records = {k: [] for k in self.arrays}
                for slot, m in enumerate(mapping):
                    index = group[m['global_ue_id']]
                    frame = kpi[kpi.global_ue_id == m['global_ue_id']].sort_values('time_ms')
                    times = frame.time_ms.to_numpy() / 1000
                    indices = np.searchsorted(times, query - 1e-7)
                    if (indices >= len(times)).any() or not np.allclose(times[indices], query, atol=1e-6, rtol=0):
                        raise ValueError('Trajectory context must have complete synchronized samples')
                    xy = frame[['x', 'y']].to_numpy()[indices].astype(np.float32)
                    values = np.zeros((self.points, len(self.schema['features'])), np.float32)
                    mask = np.zeros_like(values, dtype=bool)
                    values[prefix:-1] = self.episodes.values[index]
                    mask[prefix:-1] = self.episodes.mask[index]
                    for key, value in [('values', values), ('mask', mask), ('xy', xy),
                                       ('labels', self.episodes.labels[index]),
                                       ('label_mask', self.episodes.label_mask[index])]:
                        records[key].append(value)
                for key in self.arrays:
                    self.arrays[key].append(np.stack(records[key]))
                self.metadata.append({'source': str(raw), 'run_id': raw.name, 'start_s': start,
                                      'end_s': float(query[-1]), 'global_ue_ids': [m['global_ue_id'] for m in mapping],
                                      'seed': manifest['seed'], 'run': manifest['run']})
            offset += len(run['episodes'])
        if not self.metadata:
            raise ValueError('No complete synchronized UE groups')
        self.arrays = {k: np.stack(v) for k, v in self.arrays.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return {k: torch.from_numpy(v[index]) for k, v in self.arrays.items()}


def load_partitions(config):
    train = ScenarioDataset(config['data']['train_path'], config)
    validation = ScenarioDataset(config['data']['validation_path'], config, train.normalization)
    assert_disjoint(train.episodes, validation.episodes)
    if train.slots != validation.slots:
        raise ValueError('UE layout differs across partitions')
    return train, validation
