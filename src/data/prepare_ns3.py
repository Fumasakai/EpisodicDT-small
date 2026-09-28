"""Convert tranData UE KPI outputs to separate, continuous rsrp-only traces."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def prepare_ns3(input_dir, output_dir, partition='train', episode_length=40,
                validation_fraction=0.2, max_age_ms=None, sample_period_ms=None):
    source = Path(input_dir)
    if not source.is_dir():
        source = PROJECT / 'NS3_5GLENA_modified' / input_dir
    if not source.is_dir():
        raise ValueError(f'tranData directory not found: {input_dir}')
    source = source.resolve()
    if partition not in ('train', 'evaluation') or episode_length < 1 or not 0 < validation_fraction < 1:
        raise ValueError('Invalid partition, episode length or validation fraction')
    if max_age_ms is not None and (not np.isfinite(max_age_ms) or max_age_ms < 0):
        raise ValueError('max_age_ms must be finite and nonnegative')
    manifest_path = source / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if sample_period_ms is None:
        sample_period_ms = float(manifest.get('samplePeriod', 0)) * 1000
    if not np.isfinite(sample_period_ms) or sample_period_ms <= 0:
        raise ValueError('Supply --sample-period-ms or a manifest with positive samplePeriod')
    data = pd.read_csv(source / 'ue_kpi.csv')
    required = {'global_ue_id', 'time_ms', 'RSRP'}
    if not required <= set(data.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(data.columns))}')
    if data.empty:
        raise ValueError('ue_kpi.csv contains no rows')
    for key in required:
        data[key] = pd.to_numeric(data[key], errors='raise')
    ids = data.global_ue_id.to_numpy(dtype=float)
    times = data.time_ms.to_numpy(dtype=float)
    if (not np.isfinite(ids).all() or (ids < 0).any() or (ids != np.floor(ids)).any()
            or not np.isfinite(times).all() or (times < 0).any()):
        raise ValueError('UE IDs must be nonnegative integers and timestamps must be finite/nonnegative')
    if data.duplicated(['global_ue_id', 'time_ms']).any():
        raise ValueError('Duplicate UE/timestamp rows; resolve duplicates before conversion')
    valid = np.isfinite(data.RSRP.to_numpy(dtype=float))
    invalid_rsrp = int((~valid).sum())
    rejected_age = 0
    if max_age_ms is not None:
        if 'rsrp_age_ms' not in data:
            raise ValueError('--max-age-ms requires rsrp_age_ms')
        age = pd.to_numeric(data.rsrp_age_ms, errors='raise').to_numpy(dtype=float)
        fresh = np.isfinite(age) & (age >= 0) & (age <= max_age_ms)
        rejected_age = int((valid & ~fresh).sum())
        valid &= fresh
    data = data.loc[valid].copy()
    if data.empty:
        raise ValueError('No valid RSRP remains; check Layer and measurement availability')
    root = Path(output_dir)
    destination = root / partition
    report_path = root / 'manifests' / f'{source.name}_{partition}.json'
    if report_path.exists():
        raise FileExistsError(f'Already converted: {report_path}')
    outputs = []
    segments = []
    for ue_id, rows in data.groupby('global_ue_id', sort=True):
        rows = rows.sort_values('time_ms')
        time = rows.time_ms.to_numpy(dtype=float)
        # A missing or stale row leaves a gap. Never concatenate across that gap.
        breaks = np.flatnonzero(~np.isclose(np.diff(time), sample_period_ms, rtol=1e-6, atol=1e-6)) + 1
        for part, positions in enumerate(np.split(np.arange(len(rows)), breaks)):
            trace = rows.iloc[positions]
            length = len(trace)
            filename = f'{source.name}_ue_{int(ue_id):06d}_part_{part:03d}.csv'
            path = destination / filename
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
            split = int(length * (1 - validation_fraction))
            segments.append({'file': str(path), 'global_ue_id': int(ue_id), 'samples': length,
                             'start_time_ms': float(trace.time_ms.iloc[0]),
                             'end_time_ms': float(trace.time_ms.iloc[-1]),
                             'whole_episode_windows': max(0, length - episode_length + 1),
                             'training_windows': max(0, split - episode_length + 1) if partition == 'train' else 0,
                             'validation_windows': max(0, length - split - episode_length + 1) if partition == 'train' else 0})
            outputs.append((path, pd.DataFrame({'rsrp': trace.RSRP.to_numpy()})))
    report = {'source_dir': str(source), 'partition': partition, 'sample_period_ms': sample_period_ms,
              'max_age_ms': max_age_ms, 'input_rows': len(valid), 'invalid_rsrp_rows': invalid_rsrp,
              'rejected_age_rows': rejected_age, 'output_rows': len(data),
              'episode_length': episode_length, 'validation_fraction': validation_fraction,
              'segments': segments,
              'whole_episode_windows': sum(x['whole_episode_windows'] for x in segments),
              'training_windows': sum(x['training_windows'] for x in segments),
              'validation_windows': sum(x['validation_windows'] for x in segments)}
    destination.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for path, trace in outputs:
        trace.to_csv(path, index=False)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {len(outputs)} traces / {len(data)} RSRP values to {destination}')
    print(f'Report: {report_path}')
    print(f"Windows: full={report['whole_episode_windows']}, train={report['training_windows']}, validation={report['validation_windows']}")
    if not report['whole_episode_windows'] or (partition == 'train' and
            (not report['training_windows'] or not report['validation_windows'])):
        print('WARNING: These traces alone are too short for the configured episode length/train-validation split. '
              'Conversion succeeded; collect longer simulations before training.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', help='tranData directory path, or its name under NS3_5GLENA_modified')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/ns3'))
    parser.add_argument('--partition', choices=['train', 'evaluation'], default='train',
                        help='Use separate simulation runs for training and final evaluation')
    parser.add_argument('--episode-length', type=int, default=40)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--max-age-ms', type=float, help='Optionally reject stale RSRP measurements')
    parser.add_argument('--sample-period-ms', type=float, help='Override manifest samplePeriod (milliseconds)')
    args = parser.parse_args()
    try:
        prepare_ns3(**vars(args))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Conversion failed: {error}\n')


if __name__ == '__main__':
    main()
