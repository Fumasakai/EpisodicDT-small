"""Generate event-conditioned waypoint arrays and verify each candidate with ns-3."""
import argparse
import time
import numpy as np
import torch

from src.data.prepare_handover import LABELS
from src.data.scenarios import project_path, save_json, digest
from src.models.mobility import MobilityModel
from src.models.latent_diffusion import LatentDiffusion
from src.simulation.runner import plot_routes
from src.simulation.parallel import evaluate_jobs
from src.train.common import simulator, total_reward
from src.train.train_diffusion import sample_latents


def generation_conditions(bank_conditions, events, event, count):
    """Select observed positive scenario signatures without assigning a UE."""
    if bank_conditions.ndim != 2 or bank_conditions.shape[1] != 2*len(LABELS):
        raise ValueError('Expected six-dimensional scenario conditions')
    if event != 'any_danger' and event not in events:
        raise ValueError('Event is not in the trained/validated scope')
    columns = [LABELS.index(e) for e in (events if event == 'any_danger' else [event])]
    available = torch.unique(bank_conditions, dim=0)
    positive = available[:, :len(LABELS)].bool() & available[:, len(LABELS):].bool()
    available = available[positive[:, columns].any(-1)]
    if not len(available):
        raise ValueError('No positive training scenario condition for requested event')
    return available[torch.randint(len(available), (count,))]


def evaluate_generated(sim, points, target, mask, root, runs, seed, workers):
    """Evaluate complete trajectories in parallel, preserving candidate order."""
    jobs = [(points[j].numpy(), target[j].numpy(), mask[j].numpy(),
             root/f'candidate_{j:04}', list(runs), seed, False) for j in range(len(points))]
    started = time.monotonic()
    def progress(done, total):
        print(f'evaluation candidates={done}/{total} workers={workers} '
              f'early_stop=False elapsed={time.monotonic()-started:.1f}s', flush=True)
    return evaluate_jobs(sim, jobs, workers, progress=progress)


def generate(checkpoint, output_dir, event, count=4, seed=123, runs=(201,202,203),
             sampling_start=None, workers=None):
    torch.manual_seed(seed)
    checkpoint = project_path(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != 'ns3_latent_diffusion_v5' or saved.get('condition_format') != 'scenario_event_or_v1':
        raise ValueError('Expected scenario-conditioned diffusion v5; retrain diffusion only using an existing compatible mobility checkpoint')
    cfg = saved['config']; mobility = saved['mobility']
    if workers is None:
        workers = cfg['diffusion'].get('simulation_workers', cfg['training'].get('simulation_workers', 1))
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if mobility.get('format') != MobilityModel.FORMAT:
        raise ValueError('Expected v4 mobility checkpoint; retrain mobility and diffusion')
    model = MobilityModel(**mobility['model_kwargs']); model.load_state_dict(mobility['model']); model.eval()
    diffusion = LatentDiffusion(**saved['kwargs']); diffusion.load_state_dict(saved['diffusion']); diffusion.eval()
    if saved['config'].get('reward_mode') != 'any_danger':
        raise ValueError('Expected any_danger reward checkpoint')
    if event == 'normal':
        raise ValueError('Normal generation is not supported by a danger-maximizing policy')
    if count < 1:
        raise ValueError('Invalid candidate count')
    c = generation_conditions(saved['conditions'], cfg['events'], event, count)
    # Simulator target arguments are legacy provenance fields, not reward targets.
    # Do not invent a UE assignment for a scenario-level condition.
    target = torch.zeros(count, len(model.slots), len(LABELS), dtype=torch.bool)
    mask = torch.zeros_like(target)
    df = dict(cfg['diffusion'])
    if sampling_start:
        df['sampling_start'] = sampling_start
    root = project_path(output_dir); root.mkdir(parents=True, exist_ok=False)
    sim = simulator(model, cfg)
    records = []
    with torch.no_grad():
        normalized = sample_latents(diffusion, c, saved['bank'], saved['conditions'], df)
        z = normalized*saved['zstd']+saved['zmean']
        points = model.decode(model.distribution(z).mean)
        results = evaluate_generated(sim, points, target, mask, root, runs, seed, workers)
        for j, result in enumerate(results):
            dest = root/f'candidate_{j:04}'
            np.savez_compressed(dest/'latent.npz', z=z[j].numpy(), normalized=normalized[j].numpy())
            plot_routes(dest, points[j].numpy(), model.slots, cfg['simulation']['environment']['sites'])
            if result is None:
                records.append({'candidate': dest.name, 'valid': False})
                continue
            records.append({'candidate': dest.name, 'danger_success_rate': result['danger_success_rate'],
                            'reward': total_reward(model, points[j:j+1], result['mean_reward'], cfg).item()})
    save_json(root/'summary.json', {'checkpoint_sha256': digest(checkpoint), 'event': event,
              'condition_format': 'scenario_event_or_v1',
              'source_conditions': c[:, :len(LABELS)].int().tolist(),
              'source_masks': c[:, len(LABELS):].bool().tolist(),
              'reward_mode': cfg['reward_mode'], 'reward_events': cfg['events'], 'seed': seed, 'runs': list(runs),
              'sampling': df['sampling_start'], 'ue_count': len(model.slots),
              'simulation_workers': workers, 'stop_on_danger': False,
              'ue_counts': {kind: sum(k == kind for k, _ in model.slots)
                            for kind in ('car', 'walker', 'static')}, 'candidates': records})
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='data/processed/training_dataset_next/diffusion_scenario/diffusion.pt')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--event', choices=['any_danger', 'ho_failure', 'pingpong'], default='any_danger',
                   help='Source latent condition; reward always uses checkpoint events on any UE')
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--seed', type=int, default=123); p.add_argument('--runs', nargs='+', type=int, default=[201,202,203])
    p.add_argument('--sampling-start', choices=['positive','noise'])
    p.add_argument('--workers', type=int, help='Parallel candidates; default uses checkpoint simulation_workers')
    a = p.parse_args()
    generate(a.checkpoint, a.output_dir, a.event, a.count, a.seed, a.runs, a.sampling_start, a.workers)


if __name__ == '__main__':
    main()
