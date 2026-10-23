"""Build masked multivariate episodes and retrospective event labels from tranData."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ['ho_failure', 'pingpong', 'radio_link_failure']
FAILURES = {'HandoverEndError', 'HandoverFailureNoPreamble', 'HandoverFailureMaxRach',
            'HandoverFailureLeaving', 'HandoverFailureJoining'}


def event_outcomes(events, imsi, pingpong_s):
    """Match UE starts/completions; a ping-pong occurs at the return completion."""
    selected = events[events.imsi == imsi].sort_values('time_s', kind='stable')
    failures, rlf, returns, pending_intervals = [], [], [], []
    pending, previous = None, None
    for row in selected.itertuples():
        t = float(row.time_s)
        if row.event == 'start':
            if pending is not None:
                pending_intervals.append((pending[0], t))
            pending = (t, int(row.cell_id), int(row.target_cell_id))
        elif row.event == 'end_ok':
            if pending is not None and int(row.cell_id) == pending[2]:
                current = (t, pending[1], pending[2])
                if (previous is not None and 0 <= t - previous[0] <= pingpong_s
                        and current[1:] == (previous[2], previous[1])):
                    returns.append(t)
                previous = current
                pending_intervals.append((pending[0], t))
                pending = None
            else:
                previous = None
        elif row.event in FAILURES:
            failures.append(t)
            previous = None
            if pending is not None:
                pending_intervals.append((pending[0], t))
                pending = None
        elif row.event == 'RadioLinkFailure':
            rlf.append(t)
            previous = None
    if pending is not None:
        pending_intervals.append((pending[0], float('inf')))
    return [failures, returns, rlf], pending_intervals


def prepare_run(source, output, episode_length=100, stride=20, max_cells=3,
                max_age_ms=300., pingpong_s=2.):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (any(type(v) is not int or v < 1 for v in (episode_length, stride, max_cells))
            or not np.isfinite([max_age_ms, pingpong_s]).all() or max_age_ms <= 0 or pingpong_s <= 0):
        raise ValueError('Lengths, age and ping-pong interval must be positive')
    manifest = json.loads((source / 'run_manifest.json').read_text())
    if not manifest.get('completed', False):
        raise ValueError('Run has no successful completion marker; regenerate using updated tranData')
    period = float(manifest['samplePeriod'])
    kpi = pd.read_csv(source / 'ue_kpi.csv')
    neighbors = pd.read_csv(source / 'ue_neighbor_meas.csv')
    events = pd.read_csv(source / 'ho_events.csv')
    mapping = pd.read_csv(source / 'gnb_cell_id_map.csv').sort_values('gnb_idx')
    sites = pd.read_csv(source / 'gnb_positions.csv').merge(mapping, on='gnb_idx').sort_values('gnb_idx')
    if len(sites) > max_cells or sites.cell_id.duplicated().any():
        raise ValueError('max_cells too small or duplicate cell IDs')
    if kpi.duplicated(['global_ue_id', 'time_ms']).any():
        raise ValueError('Duplicate UE/timestamp')
    identities = kpi[['global_ue_id', 'imsi']].drop_duplicates()
    if identities.imsi.isna().any() or identities.imsi.duplicated().any() or identities.global_ue_id.duplicated().any():
        raise ValueError('Require a one-to-one UE/IMSI mapping')
    if neighbors.duplicated(['global_ue_id', 'time_ms', 'cell_id']).any():
        raise ValueError('Duplicate neighbor measurement')
    base = ['speed_mps', 'vx_mps', 'vy_mps', 'vz_mps', 'serving_rsrp_dbm',
            'serving_sinr_db', 'rsrp_age_ms', 'sinr_age_ms', 'serving_distance_m',
            'offered_dl_mbps', 'offered_ul_mbps']
    cell_features = ['rsrp_dbm', 'age_ms', 'distance_m', 'dx_m', 'dy_m', 'dz_m', 'is_serving']
    features = base + [f'cell_{c}_{f}' for c in range(max_cells) for f in cell_features]
    windows, targets, known_labels, metadata = [], [], [], []
    for ue, frame in kpi.groupby('global_ue_id', sort=True):
        frame = frame.sort_values('time_ms').reset_index(drop=True)
        times = frame.time_ms.to_numpy(dtype=float) / 1000
        if not np.isfinite(times).all():
            raise ValueError('Non-finite timestamp')
        values = np.full((len(frame), len(features)), np.nan, dtype=np.float32)
        for i, col in enumerate(base[:4]):
            values[:, i] = frame[col]
        for col, age, dest, dest_age in [('RSRP', 'rsrp_age_ms', 4, 6),
                                        ('dl_data_sinr_db', 'sinr_age_ms', 5, 7)]:
            fresh = frame[age].between(0, max_age_ms)
            values[:, dest] = frame[col].where(fresh)
            values[:, dest_age] = frame[age]
        values[:, 9] = frame.dl_packet_bytes * frame.dl_pps * 8 / 1e6
        values[:, 10] = frame.ul_packet_bytes * frame.ul_pps * 8 / 1e6
        positions = frame[['x', 'y', 'z']].to_numpy()
        for c, site in enumerate(sites.itertuples()):
            offset = len(base) + c * len(cell_features)
            measured = neighbors[(neighbors.global_ue_id == ue) & (neighbors.cell_id == site.cell_id)]
            measured = measured.set_index('time_ms').reindex(frame.time_ms)
            fresh = measured.measurement_age_ms.between(0, max_age_ms)
            values[:, offset] = measured.RSRP.where(fresh)
            values[:, offset + 1] = measured.measurement_age_ms
            delta = np.array([site.x, site.y, site.z]) - positions
            distance = np.linalg.norm(delta, axis=1)
            values[:, offset + 2] = distance
            values[:, offset + 3:offset + 6] = delta
            serving = frame.serving_cell_id.to_numpy() == site.cell_id
            values[:, offset + 6] = serving
            values[serving, 8] = distance[serving]
        ids = frame.imsi.dropna().unique()
        if len(ids) != 1:
            raise ValueError('Expected one IMSI per UE')
        outcomes, pending = event_outcomes(events, ids[0], pingpong_s)
        # A run with unidentifiable failures must not supply false negative labels.
        # This NR version uses the node ID as IMSI, so zero is a valid identity.
        logs_complete = bool(manifest.get('failure_events_recorded', False)) and not (
            events.event.isin(FAILURES) & ~events.imsi.isin(kpi.imsi.dropna().unique())).any()
        for start in range(0, len(frame) - episode_length + 1, stride):
            end = start + episode_length
            t = times[start:end]
            if not np.allclose(np.diff(t), period, rtol=1e-6, atol=1e-8):
                continue
            # Half-open support: first sample through one period after final sample.
            left, right = float(t[0]), float(t[-1] + period)
            if right > float(manifest['stopTime']) + 1e-8:
                continue
            y = [any(left <= e < right for e in group) for group in outcomes]
            known = [logs_complete, left >= times[0] + pingpong_s,
                     logs_complete and manifest.get('rlf_detection_validated', False)]
            if any(a < right <= b for a, b in pending):
                known[0] = False
            # Explicit positives remain known even if negative ascertainment is incomplete.
            known = np.asarray(known) | np.asarray(y)
            windows.append(values[start:end])
            targets.append(y)
            known_labels.append(known)
            metadata.append({'run_id': source.name, 'global_ue_id': int(ue),
                             'imsi': int(ids[0]), 'start_s': left, 'end_s': right})
    if not windows:
        raise ValueError('No complete contiguous episodes; increase duration or reduce episode length')
    x = np.stack(windows)
    mask = np.isfinite(x)
    provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source.iterdir() if p.is_file() and p.suffix in ('.csv', '.json')}
    info = {'format': 'handover_episodes_v1', 'source': str(source.resolve()), 'manifest': manifest,
            'source_hashes': provenance, 'features': features, 'labels': LABELS,
            'cell_ids': sites.cell_id.astype(int).tolist(), 'episode_length': episode_length,
            'sample_period_s': period, 'stride': stride, 'max_age_ms': max_age_ms,
            'pingpong_s': pingpong_s, 'episodes': metadata,
            'label_semantics': 'retrospective; event occurrence in [start_s,end_s); pingpong at return completion'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, values=np.where(mask, x, 0), mask=mask,
                            labels=np.asarray(targets, dtype=np.float32),
                            label_mask=np.asarray(known_labels, dtype=bool), metadata=json.dumps(info))
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode-length', type=int, default=100)
    p.add_argument('--stride', type=int, default=20)
    p.add_argument('--max-cells', type=int, default=3)
    p.add_argument('--max-age-ms', type=float, default=300)
    p.add_argument('--pingpong-s', type=float, default=2)
    info = prepare_run(**vars(p.parse_args()))
    print(f"Saved {len(info['episodes'])} multivariate episodes")


if __name__ == '__main__':
    main()
