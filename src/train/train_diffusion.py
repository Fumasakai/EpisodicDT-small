"""Event-conditioned latent diffusion, followed by simulator reward-weighted fitting."""
import argparse
import copy
import warnings
import time

import numpy as np
import torch

from src.data.joint_scenarios import load_partitions
from src.data.scenarios import digest, project_path, save_json
from src.models.latent_diffusion import LatentDiffusion
from src.simulation.parallel import evaluate_jobs
from src.train.common import (load_config, load_policy, targets, scenario_condition as condition, scenario_balanced_indices as balanced_indices,
                              batch_at, simulator, total_reward)


@torch.no_grad()
def encode_all(model, dataset):
    return torch.cat([model.encode(batch_at(dataset, np.arange(i, min(i+16, len(dataset)))))
                      for i in range(0, len(dataset), 16)])


@torch.no_grad()
def sample_latents(diffusion, conditions, bank, bank_condition, cfg):
    if cfg['sampling_start'] == 'noise':
        return diffusion.sample(conditions)
    if cfg['sampling_start'] != 'positive':
        raise ValueError('sampling_start must be noise or positive')
    # Validation conditions need not occur in training. Never use validation
    # latents as anchors or replace the requested condition with a nearby one.
    matches = (conditions[:, None, :] == bank_condition[None, :, :]).all(-1)
    supported = matches.any(-1)
    result = bank.new_empty((len(conditions), diffusion.latent_dim))
    if supported.any():
        anchors = []
        for row in matches[supported]:
            indices = torch.where(row)[0]
            anchors.append(bank[indices[torch.randint(len(indices), (), device=indices.device)]])
        result[supported] = diffusion.sample(conditions[supported], torch.stack(anchors),
                                             round(cfg['noise_fraction']*(diffusion.steps-1)))
    if (~supported).any():
        warnings.warn(f'{int((~supported).sum())}/{len(conditions)} conditions have no training anchor; '
                      'sampling those conditions from noise with the requested condition unchanged',
                      stacklevel=2)
        result[~supported] = diffusion.sample(conditions[~supported])
    return result


def evaluate_candidates(sim, paths, labels, masks, indices, root, runs, cfg, *, training=False):
    """Parallel simulation only; validation always observes the full duration."""
    df = cfg['diffusion']
    workers = df.get('simulation_workers', cfg['training'].get('simulation_workers', 1))
    early = training and df.get('stop_on_danger', False)
    jobs = [(paths[j].numpy(), labels[i].numpy(), masks[i].numpy(),
             root/f'candidate_{j:04}', runs, None, early) for j, i in enumerate(indices)]
    started = time.monotonic()
    def progress(done, total):
        print(f'diffusion {root.name} candidates={done}/{total} workers={workers} '
              f'early_stop={early} elapsed={time.monotonic()-started:.1f}s', flush=True)
    return evaluate_jobs(sim, jobs, workers, progress=progress)


def train(cfg, checkpoint):
    torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    model, saved = load_policy(checkpoint)
    # Decoder, simulation, event scope, normalization and data must describe one latent space.
    for key in ('simulation', 'events', 'model', 'ue_attributes', 'warmup_s', 'data', 'reward_mode'):
        if cfg[key] != saved['config'][key]:
            raise ValueError(f'Mobility checkpoint/config mismatch: {key}')
    model.requires_grad_(False)
    train_set, validation = load_partitions(cfg)
    if train_set.source_hashes != saved['provenance']['source_hashes'] or train_set.normalization != saved['provenance']['normalization']:
        raise ValueError('Training data or normalization changed after mobility training')
    y, mask = targets(train_set, cfg); vy, vm = targets(validation, cfg)
    c, vc = condition(y, mask), condition(vy, vm)
    raw_z, raw_vz = encode_all(model, train_set), encode_all(model, validation)
    mean, std = raw_z.mean(0), raw_z.std(0, unbiased=False).clamp_min(.001)
    z, vz = (raw_z-mean)/std, (raw_vz-mean)/std
    df = cfg['diffusion']; kwargs = dict(latent_dim=model.latent_dim, condition_dim=c.shape[1],
                                        hidden_dim=df['hidden_dim'], steps=df['steps'])
    diffusion = LatentDiffusion(**kwargs)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=df['learning_rate'])
    out = project_path(cfg['diffusion_output_dir']); out.mkdir(parents=True, exist_ok=False)
    sim = simulator(model, cfg)
    save_json(out/'config.json', cfg)
    history, best_denoise, best_state = [], float('inf'), None
    for step in range(df['epochs']):
        indices = balanced_indices(y, mask, df['batch_size'], rng)
        loss = diffusion.losses(z[indices], c[indices]).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.); optimizer.step()
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(cfg['seed']+1000)
            score = float(diffusion.losses(vz, vc).mean())
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite diffusion loss')
        if score < best_denoise:
            best_denoise, best_state = score, copy.deepcopy(diffusion.state_dict())
        history.append({'stage': 'denoising', 'step': step+1, 'validation_noise_mse': score})
        if step == 0 or (step+1) % 100 == 0 or step+1 == df['epochs']:
            print(f'diffusion denoising={step+1}/{df["epochs"]} validation_mse={score:.5f}', flush=True)
    diffusion.load_state_dict(best_state)
    torch.save({'diffusion': best_state, 'kwargs': kwargs, 'zmean': mean, 'zstd': std,
                'condition_format': 'scenario_event_or_v1'}, out/'denoising.pt')
    save_json(out/'denoising_history.json', history)
    save_json(out/'sampling_policy.json', {
        'condition_format': 'scenario_event_or_v1', 'condition_dim': c.shape[1],
        'sampling_start': df['sampling_start'], 'missing_anchor': 'noise',
        'validation_conditions': len(vc),
        'validation_conditions_without_training_anchor': int(
            (~(vc[:, None, :] == c[None, :, :]).all(-1).any(-1)).sum())})
    vi = balanced_indices(vy, vm, cfg['training']['validation_episodes'], np.random.default_rng(cfg['seed']+1000))

    def validate(round_id):
        # Identical sampling noise and held-out ns-3 runs make checkpoint comparison reproducible.
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(cfg['seed']+2000)
            sampled = sample_latents(diffusion, vc[vi], z, c, df)
            actions = model.distribution(sampled*std+mean).mean
            paths = model.decode(actions)
            records = []
            results = evaluate_candidates(sim, paths, vy, vm, vi,
                                          out/f'validation_{round_id:03}',
                                          cfg['training']['validation_runs'], cfg)
            for j, result in enumerate(results):
                if result is None:
                    records.append({'valid': False})
                    continue
                records.append({'reward': total_reward(model, paths[j:j+1], result['mean_reward'], cfg).item(),
                                'danger_success_rate': result['danger_success_rate']})
        valid = [r for r in records if 'reward' in r]
        if not valid:
            raise RuntimeError('No valid diffusion validation simulations')
        return {'reward': float(np.mean([r['reward'] for r in valid])),
                'danger_success_rate': float(np.mean([r['danger_success_rate'] for r in valid])),
                'invalid_simulations': len(records)-len(valid), 'candidates': records}

    initial = validate(0)
    best_reward, best_state = initial['reward'], copy.deepcopy(diffusion.state_dict())
    replay_z, replay_c, replay_reward = [], [], []
    feedback = [{'round': 0, 'validation': initial}]
    for round_id in range(1, df['feedback_rounds']+1):
        indices = balanced_indices(y, mask, df['candidates_per_round'], rng)
        invalid = 0
        with torch.no_grad():
            generated = sample_latents(diffusion, c[indices], z, c, df)
            paths = model.decode(model.distribution(generated*std+mean).mean)
            results = evaluate_candidates(sim, paths, y, mask, indices,
                                          out/'feedback'/f'round_{round_id:03}',
                                          df['runs'], cfg, training=True)
            for j, (i, result) in enumerate(zip(indices, results)):
                if result is None:
                    invalid += 1
                    continue
                reward = total_reward(model, paths[j:j+1], result['mean_reward'], cfg).item()
                replay_z.append(generated[j]); replay_c.append(c[i]); replay_reward.append(reward)
        if not replay_z:
            raise RuntimeError('All feedback simulations failed; inspect failure.json files')
        rz, rc, rewards = torch.stack(replay_z), torch.stack(replay_c), torch.tensor(replay_reward)
        # Relative reward weights are computed within a condition, avoiding majority-event dominance.
        _, group = torch.unique(rc, dim=0, return_inverse=True)
        weights = torch.empty_like(rewards)
        for g in group.unique():
            selected = group == g
            weights[selected] = torch.softmax(rewards[selected]/df['reward_temperature'], 0)*selected.sum()
        for _ in range(df['feedback_updates']):
            ri = torch.randint(len(rz), (df['batch_size'],))
            ti = balanced_indices(y, mask, df['batch_size'], rng)
            loss = (diffusion.losses(rz[ri], rc[ri])*weights[ri]).mean()
            loss = loss + df['anchor_weight']*diffusion.losses(z[ti], c[ti]).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.); optimizer.step()
        score = validate(round_id)
        if score['reward'] > best_reward:
            best_reward, best_state = score['reward'], copy.deepcopy(diffusion.state_dict())
        feedback.append({'round': round_id, 'training_mean_reward': float(rewards.mean()), 'invalid_simulations': invalid, 'validation': score})
        save_json(out/'feedback_history.json', feedback)
        torch.save({'z': rz, 'condition': rc, 'reward': rewards, 'weight': weights}, out/'feedback_replay.pt')
        checkpoint_data = {'format': 'ns3_latent_diffusion_v5', 'condition_format': 'scenario_event_or_v1', 'config': cfg, 'kwargs': kwargs,
                           'diffusion': best_state, 'zmean': mean, 'zstd': std, 'bank': z, 'conditions': c,
                           'mobility': saved, 'mobility_sha256': digest(project_path(checkpoint)),
                           'best_validation_reward': best_reward, 'build': sim.build}
        torch.save(checkpoint_data, out/'diffusion.pt')
        print(f'diffusion feedback={round_id} validation_reward={score["reward"]:.3f}', flush=True)
    save_json(out/'denoising_history.json', history)
    save_json(out/'metrics.json', {'initial_validation': initial, 'best_validation_reward': best_reward,
              'improved_over_initial': best_reward > initial['reward'], 'ns3_verified': True,
              'independent_test_evaluated': False, 'feedback_method': 'reward_weighted_denoising'})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/pipeline.yaml')
    p.add_argument('--checkpoint')
    args = p.parse_args(); cfg = load_config(args.config)
    train(cfg, args.checkpoint or str(project_path(cfg['output_dir'])/'mobility.pt'))


if __name__ == '__main__':
    main()
