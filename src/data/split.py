"""Deterministic train/evaluation splitting for per-UE RSRP trace files."""

from pathlib import Path

import numpy as np


def split_trace_paths(directory, train_fraction=0.5, seed=42):
    """Return non-overlapping train and evaluation CSV paths from a directory."""
    paths = sorted(Path(directory).glob("*.csv"))
    if len(paths) < 2:
        raise ValueError("At least two UE trace CSV files are required for a split.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    shuffled = [paths[index] for index in np.random.default_rng(seed).permutation(len(paths))]
    split_index = int(len(shuffled) * train_fraction)
    split_index = min(max(split_index, 1), len(shuffled) - 1)
    return shuffled[:split_index], shuffled[split_index:]
