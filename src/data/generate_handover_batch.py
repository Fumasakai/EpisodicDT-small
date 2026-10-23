"""Run partitioned, reproducible handover scenarios and build multivariate archives."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import yaml

from src.data.scenarios import digest, project_path, save_json, validate_environment, write_inputs, route_directory
from src.data.prepare_handover import prepare_run

TTT_MS = [0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120]


def scenario_runs(config, scenario):
    """An explicit repetition uses its assigned run; other scenarios use batch runs."""
    return [scenario['simulation_run']] if 'simulation_run' in scenario else config['execution']['runs']


def validate(config):
    controller, env, rec = config['controller'], config['environment'], config['recording']
    if controller['type'] not in ('a3', 'disabled'):
        raise ValueError('Supported controllers: a3, disabled; RL adapter is not implemented')
    h = controller['hysteresis_db']
    if not np.isfinite(h) or not 0 <= h <= 15 or h * 2 != round(h * 2):
        raise ValueError('Hysteresis must be a multiple of 0.5 in [0,15]')
    if controller['time_to_trigger_ms'] not in TTT_MS:
        raise ValueError('Unsupported A3 time-to-trigger')
    ep = config['episodes']
    if any(type(ep[k]) is not int or ep[k] < 1 for k in ('length', 'stride', 'max_cells')):
        raise ValueError('Episode sizes must be positive integers')
    if not np.isfinite([rec['max_age_ms'], config['labeling']['pingpong_window_s']]).all() or min(
            rec['max_age_ms'], config['labeling']['pingpong_window_s']) <= 0:
        raise ValueError('Invalid age/ping-pong threshold')
    if (type(config['execution']['seed']) is not int or config['execution']['seed'] < 1
            or config['execution'].get('timeout_s', 600) <= 0):
        raise ValueError('Invalid seed/timeout')
    if rec['layer'] not in (2, 3) or not 0 < rec['kpi_period_ms'] < env['duration_s'] * 1000:
        raise ValueError('Invalid layer/recording period')
    if not config['execution']['runs'] or len(set(config['execution']['runs'])) != len(config['execution']['runs']):
        raise ValueError('Need distinct runs')
    if any(type(run) is not int or run < 1 for run in config['execution']['runs']):
        raise ValueError('runs must be positive integers')
    validate_environment(env)
    groups = {}
    for scenario in env['scenarios']:
        if any(type(run) is not int or run < 1 for run in scenario_runs(config, scenario)):
            raise ValueError('simulation_run must be a positive integer')
        if scenario['partition'] not in ('train', 'validation', 'evaluation'):
            raise ValueError('Expected train/validation/evaluation partition')
        group = scenario.get('group', scenario['name'])
        if group in groups and groups[group] != scenario['partition']:
            raise ValueError('Scenario group crosses partitions')
        groups[group] = scenario['partition']
        if len(env['sites']) > config['episodes']['max_cells']:
            raise ValueError('Too many cells for episode schema')


def run_batch(config, prepare_only=False):
    validate(config)
    root = project_path(config['batch_dir'])
    ns3 = project_path(config['ns3_dir'])
    binary = ns3 / 'build/scratch/ns3.46-tranData'
    if not binary.is_file():
        raise FileNotFoundError(binary)
    env, controller, rec = config['environment'], config['controller'], config['recording']
    existing = root.exists()
    if existing:
        saved = json.loads((root / 'plan.json').read_text())
        if saved['config'] != config:
            raise ValueError('Existing plan differs from configuration; choose a new batch_dir')
    else:
        root.mkdir(parents=True)
        (root / 'environment.yaml').write_text(yaml.safe_dump(env, sort_keys=False))
    traffic = env.get('traffic_file')
    if traffic and not existing:
        src = project_path(traffic)
        shutil.copyfile(src, root / 'traffic.json')
    elif not existing:
        dist = {'pkt_size_bytes': {'values': [env['packet_bytes']], 'probs': [1]},
                'pps': {'values': [env['packets_per_second']], 'probs': [1]}}
        save_json(root / 'traffic.json', {s: {'dl': dist, 'ul': dist} for s in ('URLLC', 'eMBB', 'mMTC')})
    jobs = []
    for scenario in env['scenarios']:
        inputs = root / 'inputs' / scenario['name']
        if not existing:
            write_inputs(env, scenario, inputs)
        for run in scenario_runs(config, scenario):
            name = f"{scenario['name']}_r{run}"
            raw = root / 'raw' / name
            options = {'Layer': rec['layer'], 'enableHo': int(controller['type'] == 'a3'),
                       'a3HysteresisDb': controller['hysteresis_db'],
                       'a3TimeToTriggerMs': controller['time_to_trigger_ms'],
                       'shadowingEnabled': int(env.get('shadowing_enabled', False)),
                       'stopTime': env['duration_s'], 'samplePeriod': rec['kpi_period_ms'] / 1000,
                       'nrMaxUes': 0, 'gnbPositions': json.dumps(env['sites'], separators=(',', ':')),
                       'carCsv': str(inputs / 'car.csv'), 'walkerCsv': str(inputs / 'walker.csv'),
                       'staticCsv': str(inputs / 'static.csv'), 'offeredLoadJson': str(root / 'traffic.json'),
                       'outputDir': str(raw), 'nrScenario': env['scenario'],
                       'frequencyHz': env['frequency_hz'], 'bandwidthHz': env['bandwidth_hz'],
                       'gnbTxPowerDbm': env['gnb_tx_power_dbm'],
                       'seed': config['execution']['seed'], 'run': run}
            jobs.append({'name': name, 'partition': scenario['partition'],
                         'group': scenario.get('group', scenario['name']), 'options': options})
    plan = {'config': config, 'jobs': jobs, 'binary_sha256': digest(binary),
            'library_sha256': {p.name: digest(p) for p in sorted((ns3/'build/lib').glob('libns3*.so'))},
            'input_sha256': {str(p.relative_to(root)): digest(p)
                             for p in [root / 'traffic.json', root / 'environment.yaml', *sorted((root / 'inputs').rglob('*.csv'))]},
            'code_sha256': {name: digest(Path(__file__).with_name(name))
                            for name in ('generate_handover_batch.py', 'prepare_handover.py', 'scenarios.py', 'population.py', 'route_csv.py')}}
    sources = [route_directory(env, scenario) for scenario in env['scenarios']]
    if any(source is not None for source in sources):
        plan['source_csv_sha256'] = {str(source / f'{kind}.csv'): digest(source / f'{kind}.csv')
                                     for source in sources if source is not None
                                     for kind in ('car', 'walker', 'static')}
    if existing and saved != plan:
        raise ValueError('Inputs/build/code changed; choose a new batch_dir')
    if not existing:
        save_json(root / 'plan.json', plan)
    training = yaml.safe_load(project_path(config.get('training_template', 'configs/pipeline.yaml')).read_text())
    if 'ue_attributes' in env:
        training['ue_attributes'] = env['ue_attributes']
    for partition in ('train', 'validation', 'evaluation'):
        training['data'][partition + '_path'] = str(root / 'episodes' / partition)
    training['output_dir'] = str(root / 'mobility')
    training['diffusion_output_dir'] = str(root / 'diffusion')
    (root / 'simulation.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    training['simulation_config'] = str(root / 'simulation.yaml')
    if not (root / 'training.yaml').exists():
        (root / 'training.yaml').write_text(yaml.safe_dump(training, sort_keys=False))
    # Derived figures use the saved CSVs; they are not simulation inputs.
    from src.data.plot_handover_inputs import plot_batch
    plot_batch(root)
    if prepare_only:
        print(f'Wrote inputs and plan: {root}; rerun without --prepare-only to execute this plan')
        return root
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'status').mkdir(exist_ok=True)
    summary = []
    for job in jobs:
        print(f"Running {job['name']} ({job['partition']})", flush=True)
        status_path = root / 'status' / (job['name'] + '.json')
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if any(not Path(p).is_file() or digest(p) != sha for p, sha in status['outputs'].items()):
                raise ValueError('Completed run outputs changed')
            summary.append(status['summary'])
            continue
        if Path(job['options']['outputDir']).exists():
            raise ValueError('Incomplete run exists; inspect logs and choose a new batch_dir')
        command = [str(binary)] + [f'--{k}={v}' for k, v in job['options'].items()]
        with (root / 'logs' / (job['name'] + '.log')).open('w') as log:
            subprocess.run(command, cwd=ns3, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=config['execution'].get('timeout_s', 600))
        raw = Path(job['options']['outputDir'])
        manifest = json.loads((raw / 'run_manifest.json').read_text())
        manifest.update(scenario_group=job['group'], partition=job['partition'],
                        binary_sha256=plan['binary_sha256'])
        save_json(raw / 'run_manifest.json', manifest)
        ep = config['episodes']
        info = prepare_run(raw, root / 'episodes' / job['partition'] / (job['name'] + '.npz'),
                           episode_length=ep['length'], stride=ep['stride'], max_cells=ep['max_cells'],
                           max_age_ms=rec['max_age_ms'], pingpong_s=config['labeling']['pingpong_window_s'])
        summary.append({'name': job['name'], 'partition': job['partition'], 'episodes': len(info['episodes'])})
        archive = root / 'episodes' / job['partition'] / (job['name'] + '.npz')
        save_json(status_path, {'summary': summary[-1],
                               'outputs': {str(p): digest(p) for p in [*raw.iterdir(), archive] if p.is_file()}})
        save_json(root / 'summary.json', summary)
    save_json(root / 'summary.json', summary)
    print(f'Prepared {len(jobs)} runs. Train: python -m src.train.train_mobility --config {root / "training.yaml"}')
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/training_dataset_v2.yaml')
    p.add_argument('--batch-dir')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    config = yaml.safe_load(project_path(args.config).read_text())
    if args.batch_dir:
        config['batch_dir'] = args.batch_dir
    run_batch(config, args.prepare_only)


if __name__ == '__main__':
    main()
