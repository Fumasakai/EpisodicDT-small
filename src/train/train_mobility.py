"""Joint encoder/decoder policy gradients from ns-3 event rewards with speed and initial-position bounds."""
import argparse
import copy
import json
import warnings
from uuid import uuid4
import numpy as np
import torch

from src.data.joint_scenarios import load_partitions
from src.data.scenarios import project_path, save_json, digest
from src.models.mobility import MobilityModel
from src.simulation.parallel import evaluate_jobs
from src.train.common import (load_config, targets, balanced_indices, batch_at, model_kwargs,
                              simulator, total_reward, evaluate_policy, provenance)


def policy_loss(log_prob, rewards, valid=None):
    # Leave-one-out baseline uses only OTHER successful simulations of the same input.
    if valid is None:
        valid = torch.ones_like(rewards, dtype=torch.bool)
    count = valid.sum(-1, keepdim=True)
    usable = valid & (count >= 2)
    baseline = ((rewards * valid).sum(-1, keepdim=True) - rewards) / (count - 1).clamp_min(1)
    return -(log_prob * (rewards - baseline).detach() * usable).sum() / usable.sum().clamp_min(1)


def collect_rollouts(model, z, sim, y, mask, indices, cfg, root):
    """Keep successful draws across bounded rounds, grouped by source episode."""
    tr = cfg['training']; b, k = len(indices), tr['samples_per_episode']
    probabilities, rewards, masks, successes = [], [], [], []
    for attempt in range(tr.get('max_rollout_attempts', 3)):
        action, logp = model.sample(z.repeat_interleave(k, dim=0))
        with torch.no_grad():
            paths = model.decode(action)
            values, valid = [], []
            jobs = [(paths[j].numpy(), y[indices[j//k]].numpy(), mask[indices[j//k]].numpy(),
                     root/f'attempt_{attempt+1:02}'/f'sample_{j:04}', tr['runs']) for j in range(b*k)]
            if tr.get('stop_on_danger', False):
                jobs = [(*job, None, True) for job in jobs]
            results = evaluate_jobs(sim, jobs, tr.get('simulation_workers', 1))
            for result in results:
                valid.append(result is not None)
                values.append(result['mean_reward'] if result is not None else 0.)
                if result is not None:
                    successes.append(result['danger_success_rate'])
            rewards.append(total_reward(model, paths, values, cfg).reshape(b,k))
        probabilities.append(logp.reshape(b,k))
        masks.append(torch.tensor(valid, dtype=torch.bool).reshape(b,k))
        combined = torch.cat(masks, dim=1)
        if (combined.sum(-1) >= 2).any():
            break
        print(f'mobility rollout round={attempt+1} valid={int(combined.sum())}/{combined.numel()}; '
              'need two completed candidates for the same episode', flush=True)
    return (torch.cat(probabilities, dim=1), torch.cat(rewards, dim=1), combined,
            successes, len(masks))


def early_stopping(history, settings):
    """Reconstruct patience from complete validations, including after resume."""
    patience = settings.get('patience', 0)
    delta = settings.get('min_delta', 0.)
    if type(patience) is not int or patience < 0 or not np.isfinite(delta) or delta < 0:
        raise ValueError('early_stopping requires nonnegative integer patience and finite min_delta >= 0')
    best, stale = None, 0
    for row in history:
        result = row.get('validation', {})
        value = result.get('reward')
        if result.get('status', 'complete') != 'complete' or value is None or not np.isfinite(value):
            continue
        if best is None or value > best + delta:
            best, stale = value, 0
        else:
            stale += 1
    return {'stop': patience > 0 and stale >= patience,
            'stale_validations': stale, 'patience': patience, 'min_delta': delta}


def resume_compatible(saved, current):
    old, new = copy.deepcopy(saved), copy.deepcopy(current)
    for cfg in (old, new):
        for key in ('updates', 'max_rollout_attempts', 'max_consecutive_invalid_updates', 'simulation_workers', 'early_stopping'):
            cfg['training'].pop(key, None)
    if old != new:
        raise ValueError('Resume configuration differs from checkpoint; only update/retry limits, simulation_workers and early_stopping may change')


def initialize_weights(model, path, common):
    """Import only compatible weights; re-evaluate under the current simulator."""
    path = project_path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('format', 'model_kwargs', 'provenance'):
        if saved[key] != common[key]:
            raise ValueError(f'Initialization {key} differs from current training data/model')
    model.load_state_dict(saved['model'])
    return {'checkpoint': str(path), 'sha256': digest(path),
            'source_update': saved.get('update'), 'source_build': saved.get('build'),
            'optimizer_restored': False, 'validation_reused': False}


def train(cfg, resume=False, initialize_from=None):
    if resume and initialize_from is not None:
        raise ValueError('Choose resume or initialize_from, not both')
    early_stopping([], cfg['training'].get('early_stopping', {}))
    torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    train_set, validation = load_partitions(cfg)
    y, mask = targets(train_set, cfg); vy, vm = targets(validation, cfg)
    kwargs = model_kwargs(train_set, cfg)
    model = MobilityModel(**kwargs)
    sim = simulator(model, cfg)
    out = project_path(cfg['output_dir'])
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])
    validation_indices = balanced_indices(vy, vm, cfg['training']['validation_episodes'], np.random.default_rng(cfg['seed']+1000))
    common = {'format': model.FORMAT, 'config': cfg, 'model_kwargs': kwargs,
              'provenance': provenance(train_set), 'validation_sources': validation.source_hashes,
              'build': sim.build}
    if initialize_from is not None:
        common['initialization'] = initialize_weights(model, initialize_from, common)
    start, skipped = 0, 0
    session = uuid4().hex
    if resume:
        latest = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        if 'initialization' in latest:
            common['initialization'] = latest['initialization']
        resume_compatible(latest['config'], cfg)
        for key in ('format', 'model_kwargs', 'provenance', 'validation_sources', 'build'):
            if latest[key] != common[key]:
                raise ValueError(f'Resume {key} differs from checkpoint')
        selected = torch.load(out/'mobility.pt', map_location='cpu', weights_only=False)
        if selected['validation_best_reward'] != latest['validation_best_reward']:
            raise ValueError('Best/latest checkpoints are inconsistent')
        model.load_state_dict(latest['model']); optimizer.load_state_dict(latest['optimizer'])
        start = latest['update']; skipped = latest.get('consecutive_invalid_updates', 0)
        best, state = latest['validation_best_reward'], selected['model']
        history = [row for row in json.loads((out/'history.json').read_text()) if row['update'] <= start]
        initial = history[0]['validation']
        if 'torch_rng_state' in latest:
            torch.set_rng_state(latest['torch_rng_state']); rng.bit_generator.state = latest['numpy_rng_state']
        else:
            warnings.warn('Legacy checkpoint has no RNG state; resuming weights/optimizer with a fresh sampling stream')
            torch.manual_seed(cfg['seed']+start+1); rng = np.random.default_rng(cfg['seed']+start+1)
        print(f'Resuming mobility after update={start}', flush=True)
    else:
        save_json(out/'config.json', cfg)
        if 'initialization' in common:
            save_json(out/'initialization.json', common['initialization'])
        model.eval()
        initial = evaluate_policy(model, validation, cfg, sim, out/'validation_initial', validation_indices)
        if initial.get('status', 'complete') != 'complete' or initial['reward'] is None:
            raise RuntimeError('Initial validation is incomplete; inspect validation_initial/summary.json before training')
        best, state = initial['reward'], copy.deepcopy(model.state_dict())
        history = [{'update': 0, 'validation': initial}]
        torch.save({**common, 'validation_best_reward': best, 'model': state}, out/'mobility.pt')
        save_json(out/'history.json', history)
    save_json(out/f'session_{session}.json', {'resume': resume, 'start_update': start, 'config': cfg})
    tr = cfg['training']; b, k = tr['episodes_per_update'], tr['samples_per_episode']
    stop_state = early_stopping(history, tr.get('early_stopping', {}))
    for step in range(start, tr['updates']):
        if stop_state['stop']:
            print(f'Mobility early stopping: {stop_state}', flush=True)
            break
        model.train()
        indices = balanced_indices(y, mask, b, rng)
        z = model.encode(batch_at(train_set, indices))
        logp, reward, valid_mask, successes, rounds = collect_rollouts(
            model, z, sim, y, mask, indices, cfg,
            out/'rollouts'/f'update_{step+1:05}'/f'session_{session}')
        usable = (valid_mask.sum(-1) >= 2).any().item()
        row = {'update': step+1, 'status': 'updated' if usable else 'skipped',
               'rollout_rounds': rounds, 'invalid_simulations': int((~valid_mask).sum()),
               'valid_simulations': int(valid_mask.sum()),
               'reward': float(reward[valid_mask].mean()) if valid_mask.any() else None,
               'danger_success_rate': float(np.mean(successes)) if successes else None}
        if usable:
            loss = policy_loss(logp, reward, valid_mask) + tr['latent_l2']*z.square().mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            row['loss'] = float(loss.detach()); skipped = 0
        else:
            skipped += 1
            row['reason'] = 'Too few completed candidates for the same source episode'
        if usable and ((step+1) % tr['validation_every'] == 0 or step+1 == tr['updates']):
            model.eval()
            result = evaluate_policy(model, validation, cfg, sim, out/f'validation_{step+1:05}_{session}', validation_indices)
            row['validation'] = result
            if result.get('status', 'complete') != 'complete' or result['reward'] is None:
                warnings.warn(f'Validation at update={step+1} is incomplete; keeping the best model and saving latest checkpoint. Inspect {out}/validation_{step+1:05}_{session}')
            elif result['reward'] > best:
                best, state = result['reward'], copy.deepcopy(model.state_dict())
        history.append(row)
        stop_state = early_stopping(history, tr.get('early_stopping', {}))
        save_json(out/'history.json', history)
        # Save a recoverable latest model and the validation-selected model separately.
        common['validation_best_reward'] = best
        torch.save({**common, 'model': state}, out/'mobility.pt')
        torch.save({**common, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'update': step+1,
                    'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': rng.bit_generator.state,
                    'consecutive_invalid_updates': skipped}, out/'latest.pt')
        print(f'mobility update={step+1} status={row["status"]} reward={row["reward"]} '
              f'danger_success={row["danger_success_rate"]} invalid={row["invalid_simulations"]}', flush=True)
        if skipped >= tr.get('max_consecutive_invalid_updates', 5):
            raise RuntimeError('Repeated invalid rollout batches; checkpoint saved. Inspect simulator logs before resuming.')
    save_json(out/'metrics.json', {'initial_validation': initial, 'best_validation_reward': best,
              'improved_over_initial': best > initial['reward'], 'updates': history[-1]['update'], 'requested_updates': tr['updates'],
              'early_stopping': stop_state, 'stop_reason': 'validation_plateau' if stop_state['stop'] else 'update_limit',
              'optimizer_updates': sum(row.get('status', 'updated') == 'updated' for row in history[1:]),
              'skipped_updates': sum(row.get('status') == 'skipped' for row in history[1:]),
              'events': cfg['events'], 'ns3_verified': True, 'independent_test_evaluated': False,
              'incomplete_validation_checks': sum(r['validation'].get('status', 'complete') != 'complete'
                                                 for r in history if 'validation' in r),
              'last_validation_status': next(r['validation'].get('status', 'complete')
                                             for r in reversed(history) if 'validation' in r)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pipeline.yaml')
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument('--resume', action='store_true', help='Resume output_dir/latest.pt without overwriting previous rollout logs')
    startup.add_argument('--initialize-from', help='Initialize a new run from compatible weights; reset optimizer and validate with current ns-3')
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume, initialize_from=args.initialize_from)


if __name__ == '__main__':
    main()
