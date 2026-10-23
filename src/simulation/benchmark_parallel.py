"""Replay saved command.json candidates in ABBA order to compare 4/8 processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import time

from src.data.scenarios import digest, save_json


def benchmark(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    commands = sorted(source.glob('sample_*/command.json'))
    if len(commands) != 8:
        raise ValueError('Require eight saved candidate commands')
    output.mkdir(parents=True, exist_ok=False)
    trials, reference = [], None
    for trial, workers in enumerate((4, 8, 8, 4), 1):
        root = output/f'trial_{trial}_workers_{workers}'
        root.mkdir()
        def run(path):
            dest = root/path.parent.name
            dest.mkdir()
            command = json.loads(path.read_text())
            command = [f'--outputDir={dest}/raw' if arg.startswith('--outputDir=') else
                       '--packetChecking=0' if arg.startswith('--packetChecking=') else arg
                       for arg in command]
            save_json(dest/'command.json', command)
            start = time.perf_counter()
            with (dest/'run.log').open('w') as log:
                subprocess.run(command, cwd=Path(command[0]).parents[2], stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=600)
            manifest = json.loads((dest/'raw/run_manifest.json').read_text())
            if not manifest.get('completed'):
                raise RuntimeError('Incomplete simulation')
            return {'candidate': path.parent.name, 'seconds': time.perf_counter()-start,
                    'hashes': {name: digest(dest/'raw'/name) for name in
                               ('ue_kpi.csv', 'ue_neighbor_meas.csv', 'ho_events.csv')}}
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, commands))
        elapsed = time.perf_counter()-start
        hashes = [r['hashes'] for r in results]
        if reference is None:
            reference = hashes
        if hashes != reference:
            raise RuntimeError('Outputs changed across concurrency settings')
        row = {'workers': workers, 'seconds': elapsed, 'results': results}
        trials.append(row)
        save_json(root/'summary.json', row)
        print(f'trial={trial} workers={workers} seconds={elapsed:.3f}', flush=True)
    medians = {str(w): statistics.median(r['seconds'] for r in trials if r['workers']==w)
               for w in (4, 8)}
    result = {'source': str(source), 'order': [4, 8, 8, 4], 'packet_checking': False,
              'identical_output_hashes': True, 'median_seconds': medians,
              'selected_workers': int(min(medians, key=medians.get)), 'trials': trials}
    save_json(output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    benchmark(args.source, args.output)
