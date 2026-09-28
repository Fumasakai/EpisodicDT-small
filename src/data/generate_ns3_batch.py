"""Generate reproducible ns-3 scenario batches and convert RSRP for training."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

from src.data.prepare_ns3 import PROJECT, prepare_ns3
from src.data.plot_ns3_scenarios import plot_scenarios


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate_config(cfg):
    for key in ('duration_s', 'sample_period_s', 'frequency_hz', 'bandwidth_hz'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['sample_period_s'] > cfg['duration_s']:
        raise ValueError('sample_period_s exceeds duration_s')
    if cfg['layer'] not in (2, 3):
        raise ValueError('RSRP batch generation requires Layer 2 or 3')
    if not cfg['runs'] or any(type(x) is not int or x < 1 for x in cfg['runs']):
        raise ValueError('runs must be positive integers')
    if len(set(cfg['runs'])) != len(cfg['runs']):
        raise ValueError('Duplicate run numbers')
    powers = cfg['gnb_tx_powers_dbm']
    if not powers or not np.isfinite(powers).all() or len(set(powers)) != len(powers):
        raise ValueError('Transmit powers must be distinct finite numbers')
    if not cfg['car_offsets'] or any(len(x) != 2 or not np.isfinite(x).all() for x in cfg['car_offsets']):
        raise ValueError('car_offsets requires finite [x,y] pairs')
    names = set()
    for scenario in cfg['scenarios']:
        name = scenario['name']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or name in names:
            raise ValueError('Scenario names must be unique safe identifiers')
        names.add(name)
        if scenario['partition'] not in ('train', 'evaluation'):
            raise ValueError('Scenario partition must be train or evaluation')
        if not scenario['sites'] or any(len(x) != 3 or not np.isfinite(x).all() or x[2] <= 1.5 for x in scenario['sites']):
            raise ValueError('sites must contain [x,y,height>1.5]')
        for key in ('car_route', 'walker_route'):
            route = np.asarray(scenario[key], dtype=float)
            if (route.ndim != 2 or route.shape[1] != 3 or len(route) < 2 or not np.isfinite(route).all()
                    or route[0, 0] != 0 or route[-1, 0] != 1 or not (np.diff(route[:, 0]) > 0).all()):
                raise ValueError(f'{name}/{key}: require increasing [fraction,x,y], from 0 to 1')
        if len(scenario['static_position']) != 2 or not np.isfinite(scenario['static_position']).all():
            raise ValueError('static_position must be [x,y]')
    if not names:
        raise ValueError('No scenarios selected')


def write_inputs(cfg, scenario, path):
    path.mkdir(parents=True, exist_ok=False)
    for kind, route, offsets in (
        ('car', scenario['car_route'], cfg['car_offsets']),
        ('walker', scenario['walker_route'], [[0, 0]]),
        ('static', [[0, *scenario['static_position']], [1, *scenario['static_position']]], [[0, 0]]),
    ):
        records = [{'timestamp': fraction * cfg['duration_s'], 'ue_id': ue,
                    'location_x': x + dx, 'location_y': y + dy}
                   for ue, (dx, dy) in enumerate(offsets, 1) for fraction, x, y in route]
        pd.DataFrame(records).to_csv(path / f'{kind}.csv', index=False)
    pd.DataFrame(scenario['sites'], columns=['x', 'y', 'z']).to_csv(path / 'gnb.csv', index=False)


def build_jobs(cfg):
    jobs = []
    for scenario, power, run in itertools.product(cfg['scenarios'], cfg['gnb_tx_powers_dbm'], cfg['runs']):
        power_name = format(power, 'g').replace('-', 'm').replace('.', 'p')
        jobs.append({'name': f"{scenario['name']}_p{power_name}_r{run}", 'scenario': scenario['name'],
                     'partition': scenario['partition'], 'power': power, 'run': run})
    return jobs


def run_batch(cfg, prepare_only=False):
    validate_config(cfg)
    root = project_path(cfg['batch_dir'])
    ns3 = project_path(cfg['ns3_dir'])
    binary = ns3 / 'build/scratch/ns3.46-tranData'
    if not binary.is_file():
        raise ValueError(f'Build tranData first: bash {ns3}/utils/setup_trandata.sh')
    template_path = project_path(cfg['training_template'])
    template = yaml.safe_load(template_path.read_text())
    offered = ns3 / 'final_data/offeredLoad/ul_dl_dist.json'
    fingerprints = {'binary': digest(binary), 'traffic': digest(offered),
                    'template': digest(template_path), 'converter': digest(Path(__file__).with_name('prepare_ns3.py')),
                    'batch_script': digest(__file__)}
    # Include shared libraries: changes to the simulator must not silently mix runs.
    fingerprints['libraries'] = {p.name: digest(p) for p in sorted((ns3/'build/lib').glob('libns3*.so'))}
    plan = {'config': cfg, 'fingerprints': fingerprints, 'jobs': build_jobs(cfg)}
    plan_path = root / 'batch_plan.json'
    if root.exists():
        previous = json.loads(plan_path.read_text()) if plan_path.exists() else None
        # Plotting existing inputs is safe after a script update; preserve the
        # original plan so simulation resume still enforces its code fingerprint.
        if prepare_only and previous is not None:
            previous['fingerprints']['batch_script'] = fingerprints['batch_script']
        if previous != plan:
            raise ValueError('Existing batch differs from config/code/build. Choose a new --batch-dir.')
    else:
        root.mkdir(parents=True)
        for scenario in cfg['scenarios']:
            write_inputs(cfg, scenario, root/'inputs'/scenario['name'])
        shutil.copyfile(offered, root/'inputs/offered_load.json')
        save_json(plan_path, plan)
    # Verify generated CSV inputs too, before resuming any work.
    input_hashes = {str(p.relative_to(root)): digest(p) for p in sorted((root/'inputs').rglob('*')) if p.is_file()}
    input_manifest = root/'input_hashes.json'
    if input_manifest.exists() and json.loads(input_manifest.read_text()) != input_hashes:
        raise ValueError('Batch input files changed; use a new batch directory')
    save_json(input_manifest, input_hashes)
    figure_dir = plot_scenarios(cfg, root)
    print(f'Scenario figures: {figure_dir}', flush=True)
    for folder in ('raw', 'logs', 'status', 'processed/train', 'processed/evaluation'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    template['data']['train_path'] = str(root/'processed/train')
    template['data']['evaluation_path'] = str(root/'processed/evaluation')
    template['training']['checkpoint_dir'] = str(root/'model/checkpoints')
    ev = template['evaluation']
    results = PROJECT / 'results' / root.name
    ev['figure_dir'] = str(results/'figures')
    ev['generated_csv'] = str(results/'generated.csv')
    ev['actual_csv'] = str(results/'source.csv')
    ev['boxplot_path'] = str(results/'figures/episode_boxplot.png')
    ev['delta_distribution_path'] = str(results/'figures/episode_delta_distribution.png')
    (root/'training_config.yaml').write_text(yaml.safe_dump(template, sort_keys=False))
    print(f"Batch: {root}\nJobs: {len(plan['jobs'])}; duration={cfg['duration_s']}s", flush=True)
    print(f"Training config: {root/'training_config.yaml'}", flush=True)
    if prepare_only:
        return []
    results = []
    for index, job in enumerate(plan['jobs'], 1):
        name = job['name']
        raw = root/'raw'/name
        status_path = root/'status'/f'{name}.json'
        print(f"[{index}/{len(plan['jobs'])}] {name} ({job['partition']})", flush=True)
        if status_path.exists():
            status = json.loads(status_path.read_text())
            for path, sha in status['outputs'].items():
                if not Path(path).is_file() or digest(path) != sha:
                    raise ValueError(f'Completed output missing/changed: {path}')
            results.append(status)
            print('  completed; verified and skipped', flush=True)
            continue
        if raw.exists():
            raise ValueError(f'Incomplete run exists: {raw}. Inspect its log; use a new batch directory to retry.')
        inputs = root/'inputs'/job['scenario']
        options = {'Layer': cfg['layer'], 'stopTime': cfg['duration_s'], 'samplePeriod': cfg['sample_period_s'],
                   'nrMaxUes': 0, 'gnbPlacement': 'manual', 'gnbPositionsCsv': inputs/'gnb.csv',
                   'carCsv': inputs/'car.csv', 'walkerCsv': inputs/'walker.csv', 'staticCsv': inputs/'static.csv',
                   'offeredLoadJson': root/'inputs/offered_load.json', 'outputDir': raw,
                   'nrScenario': cfg['scenario'], 'frequencyHz': cfg['frequency_hz'], 'bandwidthHz': cfg['bandwidth_hz'],
                   'gnbTxPowerDbm': job['power'], 'seed': cfg['seed'], 'run': job['run'],
                   'enableHo': int(cfg['enable_handover'])}
        command = [str(binary)] + [f'--{key}={value}' for key, value in options.items()]
        start = time.monotonic()
        with (root/'logs'/f'{name}.log').open('w') as log:
            log.write(json.dumps(command)+'\n'); log.flush()
            subprocess.run(command, cwd=ns3, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = prepare_ns3(raw, root/'processed', partition=job['partition'],
                             episode_length=template['data']['episode_length'],
                             validation_fraction=template['training']['validation_fraction'], max_age_ms=cfg['max_age_ms'])
        frame = pd.read_csv(raw/'ue_kpi.csv')
        measured = frame.RSRP.dropna()
        sufficient = (report['training_windows'] > 0 and report['validation_windows'] > 0
                      if job['partition'] == 'train' else report['whole_episode_windows'] > 0)
        outputs = [p for p in raw.iterdir() if p.is_file()] + [Path(x['file']) for x in report['segments']]
        outputs.append(root/'processed/manifests'/f"{name}_{job['partition']}.json")
        status = {**job, 'elapsed_s': time.monotonic()-start, 'report': report,
                  'has_usable_windows': sufficient, 'finite_rsrp_fraction': float(np.isfinite(frame.RSRP).mean()),
                  'rsrp_min_dbm': float(measured.min()), 'rsrp_max_dbm': float(measured.max()),
                  'outputs': {str(p): digest(p) for p in outputs}}
        save_json(status_path, status)
        results.append(status)
        save_json(root/'batch_summary.json', results)
    save_json(root/'batch_summary.json', results)
    usable = sum(x['has_usable_windows'] for x in results)
    print(f'Finished {len(results)} jobs; {usable} have usable windows. See batch_summary.json.', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ns3_batch.yaml')
    parser.add_argument('--batch-dir', help='Override destination; use a new directory for changed settings')
    parser.add_argument('--prepare-only', action='store_true', help='Write inputs and plan without running simulations')
    parser.add_argument('--smoke', action='store_true', help='2 seconds, one power/run per scenario; not training data')
    parser.add_argument('--scenarios', nargs='+', help='Select scenario names')
    parser.add_argument('--powers', nargs='+', type=float)
    parser.add_argument('--runs', nargs='+', type=int)
    args = parser.parse_args()
    try:
        cfg = yaml.safe_load(project_path(args.config).read_text())
        if args.smoke:
            cfg.update(duration_s=2, runs=[cfg['runs'][0]], gnb_tx_powers_dbm=[30])
            cfg['batch_dir'] += '_smoke'
        if args.batch_dir:
            cfg['batch_dir'] = args.batch_dir
        if args.scenarios:
            known = {s['name'] for s in cfg['scenarios']}
            if not set(args.scenarios) <= known:
                raise ValueError(f'Unknown scenarios: {set(args.scenarios)-known}')
            cfg['scenarios'] = [s for s in cfg['scenarios'] if s['name'] in args.scenarios]
        if args.powers is not None:
            cfg['gnb_tx_powers_dbm'] = args.powers
        if args.runs is not None:
            cfg['runs'] = args.runs
        run_batch(cfg, args.prepare_only)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Batch failed: {error}\n')


if __name__ == '__main__':
    main()
