"""Configuration, event conditions and simulator evaluation shared by both learners."""

import warnings

import numpy as np
import torch
import yaml

from src.data.prepare_handover import LABELS
from src.data.scenarios import project_path, save_json
from src.models.mobility import MobilityModel
from src.data.population import environment_slots
from src.simulation.runner import Simulator
from src.simulation.parallel import evaluate_jobs


def dataset_training_config(batch):
    """Resolve the maintained config, including freshly generated datasets."""
    root = project_path(batch)
    for name in ('training_fast.yaml', 'training.yaml'):
        if (root/name).is_file():
            return root/name
    raise FileNotFoundError(f'No training_fast.yaml or training.yaml in {root}')


def load_config(path):
    path = project_path(path)
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f'Training configuration must be a YAML mapping: {path}')
    required = {'simulation_config', 'data', 'output_dir', 'diffusion_output_dir',
                'seed', 'warmup_s', 'events', 'ue_attributes', 'model',
                'training', 'diffusion'}
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(
            f'Not a current simulator-reward training configuration: {path}. '
            f'Missing fields: {", ".join(missing)}. '
            'Legacy classifier configurations and data with per-scenario base stations '
            'cannot be migrated by adding simulation_config alone. '
            'Generate a new batch without overwriting the old data:\n'
            '  python -m src.data.generate_handover_batch --config configs/training_dataset_v2.yaml '
            '--batch-dir data/processed/training_dataset_next\n'
            'Then use --config data/processed/training_dataset_next/training.yaml.')
    deprecated = cfg.pop('realism_weights', None) is not None
    for profile in cfg['ue_attributes'].values():
        for key in ('max_turn_rate_rad_s', 'max_acceleration_mps2'):
            if key in profile:
                profile.pop(key)
                deprecated = True
    if deprecated:
        warnings.warn('Ignoring obsolete acceleration, turn-rate and area penalties; '
                      'only speed and initial-position bounds apply.', UserWarning)
    cfg.setdefault('reward_mode', 'any_danger')
    cfg['simulation'] = yaml.safe_load(project_path(cfg['simulation_config']).read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg['reward_mode'] != 'any_danger':
        raise ValueError('reward_mode must be any_danger; legacy per-UE matching is unsupported')
    if 'population' in cfg['simulation']['environment']:
        environment_slots(cfg['simulation']['environment'])
    if not cfg['events'] or len(set(cfg['events'])) != len(cfg['events']) or any(e not in LABELS for e in cfg['events']):
        raise ValueError('Require distinct supported event names')
    if 'radio_link_failure' in cfg['events']:
        raise ValueError('RLF detection is unvalidated; cannot use it as a reward')
    model, tr, df = cfg['model'], cfg['training'], cfg['diffusion']
    if not 0 < model['action_std'] <= 2 or model['waypoint_interval_s'] <= 0:
        raise ValueError('Invalid action variance/waypoint interval')
    if model['hidden_dim'] % model['num_heads'] or min(model['latent_dim'], model['num_layers']) < 1:
        raise ValueError('Invalid model dimensions')
    if len(model['bounds']) != 2 or any(len(b) != 2 or not np.isfinite(b).all() or b[1] <= b[0] for b in model['bounds']):
        raise ValueError('Require finite increasing x/y bounds')
    numeric = [model['action_std'], model['waypoint_interval_s'], tr['learning_rate'], tr['latent_l2'],
               df['learning_rate'], df['reward_temperature'], df['anchor_weight'], df['noise_fraction']]
    if not np.isfinite(numeric).all() or min(numeric) < 0 or min(tr['learning_rate'], df['learning_rate']) <= 0:
        raise ValueError('Learning settings must be finite and nonnegative')
    for kind, p in cfg['ue_attributes'].items():
        if not np.isfinite([p['min_moving_speed_mps'], p['max_speed_mps']]).all():
            raise ValueError('Nonfinite mobility profile')
        if not 0 <= p['min_moving_speed_mps'] <= p['max_speed_mps']:
            raise ValueError(f'Invalid mobility profile: {kind}')
        if kind == 'static' and (p['max_speed_mps'] != 0 or not p['allow_stop']):
            raise ValueError('Static UEs must remain stationary')
    for obj, keys in [(tr, ['updates', 'episodes_per_update', 'samples_per_episode', 'validation_every', 'validation_episodes']),
                      (df, ['epochs', 'steps', 'batch_size', 'feedback_rounds', 'candidates_per_round', 'feedback_updates'])]:
        if any(type(obj[k]) is not int or obj[k] < 1 for k in keys):
            raise ValueError('Training counts must be positive integers')
    for key, default in [('max_rollout_attempts', 3), ('max_consecutive_invalid_updates', 5), ('simulation_workers', 1)]:
        value = tr.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if tr['samples_per_episode'] < 2 or df['steps'] < 2 or df['reward_temperature'] <= 0:
        raise ValueError('Need multiple policy samples and valid diffusion parameters')
    if df['sampling_start'] not in ('positive', 'noise') or not 0 < df['noise_fraction'] <= 1:
        raise ValueError('Invalid sampling start')
    if any(not runs or len(set(runs)) != len(runs) or any(type(x) is not int or x < 1 for x in runs)
           for runs in [tr['runs'], tr['validation_runs'], df['runs']]):
        raise ValueError('Invalid simulation run list')
    if set(tr['runs']) & set(tr['validation_runs']) or set(df['runs']) & set(tr['validation_runs']):
        raise ValueError('Training and validation simulation runs must be disjoint')
    if cfg['simulation']['controller']['type'] not in ('a3', 'disabled'):
        raise ValueError('Controller adapter is not implemented')
    if cfg['warmup_s'] < cfg['simulation']['labeling']['pingpong_window_s'] + cfg['simulation']['recording']['kpi_period_ms']/1000:
        raise ValueError('Warmup must include complete event history')


def targets(dataset, cfg):
    y = torch.from_numpy(dataset.arrays['labels']).bool()
    mask = torch.from_numpy(dataset.arrays['label_mask']).bool()
    selected = torch.tensor([e in cfg['events'] for e in LABELS])
    mask = mask & selected[None, None]
    if not mask.flatten(1).any(1).all():
        raise ValueError('Episode without known target labels')
    return y & mask, mask


def condition(y, mask):
    return torch.cat([(y & mask).flatten(1), mask.flatten(1)], -1).float()


def scenario_condition(y, mask):
    """Per-event OR over UEs; absence is known only when every UE is known."""
    positive = (y.bool() & mask.bool()).any(dim=1)
    known = positive | mask.bool().all(dim=1)
    return torch.cat([positive, known], dim=-1).float()


def scenario_balanced_indices(y, mask, count, rng):
    signatures = scenario_condition(y, mask).numpy()
    _, groups = np.unique(signatures, axis=0, return_inverse=True)
    pools = [np.flatnonzero(groups == i) for i in np.unique(groups)]
    return np.array([rng.choice(pools[rng.integers(len(pools))]) for _ in range(count)])


def balanced_indices(y, mask, count, rng):
    signatures = condition(y, mask).numpy()
    _, groups = np.unique(signatures, axis=0, return_inverse=True)
    pools = [np.flatnonzero(groups == i) for i in np.unique(groups)]
    return np.array([rng.choice(pools[rng.integers(len(pools))]) for _ in range(count)])


def batch_at(dataset, indices):
    return {k: torch.from_numpy(v[indices]) for k,v in dataset.arrays.items()}


def model_kwargs(dataset, cfg):
    return dict(feature_dim=len(dataset.schema['features']), points=dataset.points, slots=dataset.slots,
                profiles=[cfg['ue_attributes'][k] for k, _ in dataset.slots], sites=cfg['simulation']['environment']['sites'],
                duration_s=(dataset.points-1)*dataset.dt, **cfg['model'])


def simulator(model, cfg):
    env = cfg['simulation']['environment']
    if 'population' in env and model.slots != environment_slots(env):
        raise ValueError('Model UE layout differs from configured population; regenerate data and retrain')
    return Simulator(cfg['simulation'], model.slots, model.profiles, cfg['warmup_s'], model.duration_s, cfg['events'],
                     binary=cfg.get('simulation_binary'))


def total_reward(model, points, event_reward, cfg):
    # Speed and initial position are constrained by decode; no motion penalties.
    return points.new_tensor(event_reward)


def evaluate_policy(model, dataset, cfg, sim, root, indices):
    root.mkdir(parents=True, exist_ok=True)
    y, mask = targets(dataset, cfg)
    records = []
    with torch.no_grad():
        z = model.encode(batch_at(dataset, indices))
        action, _ = model.sample(z, deterministic=True)
        paths = model.decode(action)
        jobs = [(paths[j].numpy(), y[i].numpy(), mask[i].numpy(), root/f'episode_{j:04}',
                 cfg['training']['validation_runs']) for j, i in enumerate(indices)]
        results = evaluate_jobs(sim, jobs, cfg['training'].get('simulation_workers', 1))
        for j, (i, result) in enumerate(zip(indices, results)):
            if result is None:
                records.append({'index': int(i), 'valid': False})
                continue
            reward = total_reward(model, paths[j:j+1], result['mean_reward'], cfg).item()
            records.append({'index': int(i), 'reward': reward, 'danger_success_rate': result['danger_success_rate']})
    valid = [r for r in records if 'reward' in r]
    result = {'status': 'complete' if len(valid) == len(records) and valid else
                         'partial' if valid else 'invalid',
              'reward': float(np.mean([r['reward'] for r in valid])) if valid else None,
              'danger_success_rate': float(np.mean([r['danger_success_rate'] for r in valid])) if valid else None,
              'valid_episodes': len(valid),
              'invalid_simulations': len(records)-len(valid), 'episodes': records}
    if not valid:
        result['reason'] = 'No valid validation simulations; inspect failure.json files'
    save_json(root/'summary.json', result)
    return result


def load_policy(path):
    path = project_path(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('format') != MobilityModel.FORMAT:
        raise ValueError('Reward semantics changed: retrain mobility and diffusion using v4 checkpoints')
    model = MobilityModel(**ckpt['model_kwargs']); model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt


def provenance(dataset):
    return {'normalization': dataset.normalization, 'schema': dataset.schema, 'source_hashes': dataset.source_hashes,
            'partitions': dataset.metadata}
