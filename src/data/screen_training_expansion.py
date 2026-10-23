"""Screen scenario/run candidates in isolated processes; retain successful repeats."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from src.data.scenarios import project_path, save_json
from src.data.screen_event_scenarios import screen
from src.data.merge_screened_scenarios import merge


def worker(config, scenario, output):
    cfg = copy.deepcopy(config)
    cfg['environment']['scenarios'] = [scenario]
    screen(cfg, output, runs=[scenario['simulation_run']] if 'simulation_run' in scenario else cfg['execution']['runs'])
    return str(output)


def run(config, output, merged, workers=3):
    root = project_path(output)
    root.mkdir(parents=True, exist_ok=False)
    scenarios = config['environment']['scenarios']
    # Start with each partition so evaluation results arrive early.
    buckets = [[s for s in scenarios if s['partition'] == p]
               for p in ('train', 'validation', 'evaluation')]
    ordered = [b[i] for i in range(max(map(len, buckets))) for b in buckets if i < len(b)]
    sources = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, config, s, root / s['name']): s for s in ordered}
        for future in as_completed(futures):
            sources.append(future.result())
            save_json(root / 'completed_jobs.json', {'sources': sorted(sources), 'total': len(ordered)})
            print(f'PROGRESS {len(sources)}/{len(ordered)}', flush=True)
    # Preserve all valid outcomes, not only desired event outcomes.
    return merge(config, sorted(sources), merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training_dataset_v2.yaml')
    parser.add_argument('--output-dir', default='data/processed/training_dataset_next_screen')
    parser.add_argument('--merged-dir', default='data/processed/training_dataset_next')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    run(yaml.safe_load(project_path(args.config).read_text()), args.output_dir,
        args.merged_dir, args.workers)


if __name__ == '__main__':
    main()
