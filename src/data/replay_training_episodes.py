"""Check whether source events survive exact route cropping plus warm-up."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.data.joint_scenarios import ScenarioDataset
from src.data.prepare_handover import LABELS
from src.data.route_csv import read_tracks
from src.data.scenarios import project_path, save_json
from src.simulation.runner import Simulator
from src.train.common import load_config, dataset_training_config


def crop_routes(tracks, slots, start, end):
    """Use the union of original knots, preserving piecewise-linear motion exactly."""
    times = np.unique([start, end, *[p[0] for points in tracks.values() for p in points
                                    if start < p[0] < end]])
    paths = []
    for kind, uid in slots:
        a = np.asarray(tracks[f'{kind}:{uid}'], float)
        paths.append(np.column_stack((times-start,
                     np.interp(times, a[:, 0], a[:, 1]), np.interp(times, a[:, 0], a[:, 2]))))
    return np.asarray(paths)


def worker(task):
    cfg, slots, profiles, points, target, mask, root, metadata = task
    sim = Simulator(cfg['simulation'], slots, profiles, cfg['warmup_s'], points[0, -1, 0])
    row = dict(metadata, status='invalid')
    try:
        result = sim.evaluate(points, target, mask, root, runs=[metadata['run']], seed=metadata['seed'])
        observed = np.asarray(result['runs'][0]['observed'], bool)
        known = np.asarray(result['runs'][0]['known'], bool)
        row.update(status='completed', original_events={e: bool((target[:, i] & mask[:, i]).any())
                   for i, e in enumerate(LABELS)},
                   replay_events={e: bool((observed[:, i] & known[:, i]).any()) for i, e in enumerate(LABELS)},
                   replay_all_selected_known=bool(known[:, :2].all()))
    except (ValueError, OSError) as error:
        row['error'] = str(error)
    except Exception as error:
        # Subprocess failures are audit results, never normal event labels.
        import subprocess
        if not isinstance(error, subprocess.SubprocessError):
            raise
        row['error'] = str(error)
    return row


def replay(batch, output, workers=3):
    root = project_path(output)
    root.mkdir(parents=True, exist_ok=False)
    cfg = load_config(dataset_training_config(batch))
    sources = {r['name']: r['source_batch'] for r in
               json.loads((project_path(batch)/'screening.json').read_text())['results']}
    tasks = []
    for partition in ('train', 'validation', 'evaluation'):
        ds = ScenarioDataset(cfg['data'][partition+'_path'], cfg)
        used = set()
        for index, meta in enumerate(ds.metadata):
            target = ds.arrays['labels'][index].astype(bool)
            mask = ds.arrays['label_mask'][index].astype(bool)
            event = 'hof' if (target[:, 0] & mask[:, 0]).any() else (
                    'pingpong' if (target[:, 1] & mask[:, 1]).any() else 'negative')
            if event in used or (event == 'negative' and not mask[:, :2].all()):
                continue
            used.add(event)
            name = meta['run_id'].rsplit('_r', 1)[0]
            batch_root = project_path(sources[name])
            plan = json.loads((batch_root/'plan.json').read_text())
            tracks = read_tracks(batch_root/'inputs'/name, plan['config']['environment'])
            start, end = meta['start_s']-cfg['warmup_s'], meta['end_s']
            points = crop_routes(tracks, ds.slots, start, end)
            metadata = dict(partition=partition, category=event, source=meta['source'],
                            crop_start_s=start, crop_end_s=end, run=meta['run'], seed=meta['seed'])
            profiles = [cfg['ue_attributes'][kind] for kind, _ in ds.slots]
            tasks.append((cfg, ds.slots, profiles, points, target, mask,
                          root/f'{partition}_{event}', metadata))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(worker, tasks))
    save_json(root/'replay_summary.json', dict(results=rows,
              note='Exact cropped routes with 3s warm-up, freshly initialized simulator. Descriptive subset audit; not a full state checkpoint restore.'))
    print(json.dumps(rows, indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', default='data/processed/training_dataset_next')
    parser.add_argument('--output-dir', default='data/processed/training_dataset_replay')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    replay(args.batch_dir, args.output_dir, args.workers)
