from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RSRPWindowDataset(Dataset):
    """Returns normalized history/future pairs without crossing UE trace boundaries."""

    def __init__(self, csv_path, history_length, future_length, mean=None, std=None):
        if isinstance(csv_path, (str, Path)):
            source = Path(csv_path)
            paths = sorted(source.glob("*.csv")) if source.is_dir() else [source]
        else:
            paths = [Path(path) for path in csv_path]
        if not paths:
            raise FileNotFoundError(f"No CSV files found in: {source}")
        self.traces = []
        for path in paths:
            values = pd.read_csv(path)["rsrp"].to_numpy(dtype=np.float32)
            if len(values) >= history_length + future_length:
                self.traces.append(values)
        if not self.traces:
            raise ValueError("No RSRP trace is long enough for the requested windows.")

        self.history_length = history_length
        self.future_length = future_length
        all_values = np.concatenate(self.traces)
        self.mean = float(all_values.mean() if mean is None else mean)
        computed_std = float(all_values.std() if std is None else std)
        self.std = max(computed_std, 1e-6)
        self.windows = [
            (trace_index, start)
            for trace_index, values in enumerate(self.traces)
            for start in range(len(values) - history_length - future_length + 1)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        trace_index, start = self.windows[index]
        values = self.traces[trace_index]
        split = start + self.history_length
        end = split + self.future_length
        history = (values[start:split] - self.mean) / self.std
        future = (values[split:end] - self.mean) / self.std
        return torch.from_numpy(history[:, None]), torch.from_numpy(future[:, None])
