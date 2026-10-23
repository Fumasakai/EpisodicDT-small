"""Audit actual synchronized training labels, failure causes and repeat outcomes."""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from src.data.joint_scenarios import ScenarioDataset
from src.data.prepare_handover import FAILURES, LABELS
from src.data.scenarios import project_path, save_json
from src.train.common import load_config, dataset_training_config


def audit(batch, candidates):
    root = project_path(batch)
    cfg = load_config(dataset_training_config(root))
    metadata = {r['name']: r for r in json.loads(project_path(candidates).read_text())['candidates']}
    records = json.loads((root / 'screening.json').read_text())['results']
    outcomes = {}
    summaries = {}
    datasets = {}
    for partition in ('train', 'validation', 'evaluation'):
        ds = ScenarioDataset(cfg['data'][partition + '_path'], cfg,
                             None if partition == 'train' else datasets['train'].normalization)
        datasets[partition] = ds
        y = (ds.arrays['labels'] > 0) & ds.arrays['label_mask'].astype(bool)
        counts = {label: int(y[..., i].any(axis=1).sum()) for i, label in enumerate(LABELS)}
        summaries[partition] = dict(synchronized_episodes=len(ds), positive_episodes=counts)
        for i, row in enumerate(ds.metadata):
            scenario = row['run_id'].rsplit('_r', 1)[0]
            values = outcomes.setdefault(scenario, Counter())
            values['synchronized_episodes'] += 1
            for j, label in enumerate(LABELS):
                values[label + '_episodes'] += int(y[i, :, j].any())
            values['known_negative_episodes'] += int(
                not y[i, :, :2].any() and ds.arrays['label_mask'][i, :, :2].all())
    rows, traces = [], []
    for record in records:
        name = record['name']
        row = dict(metadata[name], status=record['status'], **outcomes.get(name, {}))
        row['source_batch'] = record['source_batch']
        row['error'] = record.get('error', '')
        for label in LABELS:
            row[label + '_events_after_3s'] = record['event_counts'].get(label, 0)
        if record['status'] != 'invalid':
            for raw in project_path(record['source_batch']).joinpath('raw').iterdir():
                events = pd.read_csv(raw / 'ho_events.csv')
                failures = events[events.event.isin(FAILURES) & (events.time_s >= 3)].copy()
                failures['scenario'] = name
                failures['run'] = row['run']
                traces.extend(failures.to_dict('records'))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'scenario_outcomes.csv', index=False)
    pd.DataFrame(traces).to_csv(root / 'hof_events.csv', index=False)
    repeats = []
    for base, group in frame.groupby('base_scenario', sort=True):
        complete = group[group.status != 'invalid']
        count = len(complete)
        entry = dict(base_scenario=base, partition=group.partition.iloc[0],
                     attempted=len(group), completed=count, invalid=len(group)-count)
        for label in ('ho_failure', 'pingpong'):
            positive = int((complete[label + '_episodes'].fillna(0) > 0).sum())
            entry[label + '_positive_runs'] = positive
            entry[label + '_rate_among_completed'] = positive/count if count else None
        repeats.append(entry)
    pd.DataFrame(repeats).to_csv(root / 'repeat_outcomes.csv', index=False)
    summary = dict(attempted_runs=len(frame), completed_runs=int((frame.status != 'invalid').sum()),
                   invalid_runs=int((frame.status == 'invalid').sum()),
                   unique_base_scenarios=frame.base_scenario.nunique(),
                   completed_base_scenarios=frame.loc[frame.status != 'invalid', 'base_scenario'].nunique(),
                   hof_positive_base_scenarios=frame.loc[frame.ho_failure_episodes.fillna(0) > 0, 'base_scenario'].nunique(),
                   partitions=summaries,
                   failure_causes_after_3s=dict(Counter(t['event'] for t in traces)),
                   limitations=['Only two planned runs per route; rates are descriptive, not precise probability estimates.',
                                'Invalid runs are excluded from training, not labelled normal.',
                                'Previously discovered routes are exploratory train/validation data; new evaluation routes were assigned before screening.',
                                'Event reproduction after cutting an episode is not guaranteed by full-route repeats.'])
    save_json(root / 'audit.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', default='data/processed/training_dataset_next')
    parser.add_argument('--candidates', default='data/scenarios/training_dataset_v2/candidates.json')
    args = parser.parse_args()
    audit(args.batch_dir, args.candidates)
