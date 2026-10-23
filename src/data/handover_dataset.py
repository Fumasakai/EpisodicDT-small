"""Run-disjoint, masked multivariate episode loading."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HandoverDataset(Dataset):
    def __init__(self, path, normalization=None):
        paths = sorted(Path(path).glob('*.npz'))
        if not paths:
            raise ValueError(f'No episode archives: {path}')
        arrays, masks, labels, label_masks = [], [], [], []
        self.metadata, self.runs, self.features, self.signature = [], [], None, None
        for file in paths:
            with np.load(file, allow_pickle=False) as data:
                info = json.loads(str(data['metadata']))
                signature = {k: info[k] for k in ('format', 'features', 'labels', 'episode_length',
                                                  'sample_period_s', 'max_age_ms', 'pingpong_s')}
                if self.signature is not None and signature != self.signature:
                    raise ValueError('Incompatible feature/label/time schemas')
                self.signature, self.features = signature, info['features']
                x, mask = data['values'].copy(), data['mask'].copy().astype(bool)
                y, ym = data['labels'].copy(), data['label_mask'].copy().astype(bool)
                if (x.ndim != 3 or mask.shape != x.shape or x.shape[2] != len(self.features)
                        or y.shape != (len(x), 3) or ym.shape != y.shape
                        or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()
                        or len(info['episodes']) != len(x)):
                    raise ValueError(f'Invalid archive: {file}')
                arrays.append(x); masks.append(mask); labels.append(y); label_masks.append(ym)
                self.metadata.extend(info['episodes'])
                self.runs.append(info)
        self.values = np.concatenate(arrays).astype(np.float32)
        self.mask = np.concatenate(masks)
        self.labels = np.concatenate(labels).astype(np.float32)
        self.label_mask = np.concatenate(label_masks)
        if normalization is None:
            count = self.mask.sum(axis=(0, 1))
            mean = (self.values * self.mask).sum(axis=(0, 1), dtype=np.float64) / np.maximum(count, 1)
            variance = (((self.values - mean) ** 2) * self.mask).sum(axis=(0, 1)) / np.maximum(count, 1)
            normalization = {'mean': mean.tolist(), 'std': np.maximum(np.sqrt(variance), 1e-6).tolist(),
                             'observed_features': (count > 0).tolist()}
        self.normalization = normalization
        mean, std = np.asarray(normalization['mean']), np.asarray(normalization['std'])
        if mean.shape != (len(self.features),) or std.shape != mean.shape or not np.all(std > 0):
            raise ValueError('Invalid normalization')
        # Never score a feature at validation/test that the training set could not observe.
        self.mask &= np.asarray(normalization['observed_features'], dtype=bool)[None, None, :]
        self.values = np.where(self.mask, (self.values - mean) / std, 0).astype(np.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {k: torch.from_numpy(v[index]) for k, v in
                [('values', self.values), ('mask', self.mask), ('labels', self.labels),
                 ('label_mask', self.label_mask)]}


def assert_disjoint(*datasets):
    """Reject copied runs and shared scenario families across partitions."""
    seen_sources, seen_hashes, seen_groups = set(), set(), set()
    signature = datasets[0].signature
    for dataset in datasets:
        if dataset.signature != signature:
            raise ValueError('Partition schema mismatch')
        sources = {r['source'] for r in dataset.runs}
        hashes = {r['source_hashes']['ue_kpi.csv'] for r in dataset.runs}
        groups = {r['manifest']['scenario_group'] for r in dataset.runs if 'scenario_group' in r['manifest']}
        if seen_sources & sources or seen_hashes & hashes or seen_groups & groups:
            raise ValueError('Run/scenario leakage across partitions')
        seen_sources |= sources; seen_hashes |= hashes; seen_groups |= groups
