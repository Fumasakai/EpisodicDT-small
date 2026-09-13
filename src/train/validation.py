"""Deterministic validation using held-out histories and sampled forecasts."""

import torch


def forecast_metrics(samples, actual):
    """samples: [episodes, samples, future]; actual: [episodes, future], in dBm.

    CRPS for the empirical forecast distribution:
    mean_s |x_s-y| - (1 / (2 S^2)) sum_{s,r} |x_s-x_r|.
    Sorted samples compute the pairwise term without a quadratic allocation.
    """
    if samples.ndim != 3 or actual.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("Incompatible sample and actual shapes.")
    count = samples.shape[1]
    ordered = samples.sort(dim=1).values
    weights = (2 * torch.arange(1, count + 1, device=samples.device) - count - 1)
    spread = (ordered * weights[None, :, None]).sum(dim=1) / count ** 2
    crps = (samples - actual[:, None, :]).abs().mean(dim=1) - spread
    error = samples.mean(dim=1) - actual
    lower, upper = torch.quantile(samples, torch.tensor([0.05, 0.95], device=samples.device), dim=1)
    coverage = ((actual >= lower) & (actual <= upper)).float()
    per_step = {
        "crps_dbm": crps.mean(dim=0),
        "mae_dbm": error.abs().mean(dim=0),
        "bias_dbm": error.mean(dim=0),
        "actual_std_dbm": actual.std(dim=0, unbiased=False),
        "generated_std_dbm": samples.flatten(0, 1).std(dim=0, unbiased=False),
        "coverage_90": coverage.mean(dim=0),
        "interval_width_90_dbm": (upper - lower).mean(dim=0),
    }
    return {
        "crps_dbm": crps.mean().item(),
        "mae_dbm": error.abs().mean().item(),
        "coverage_90": coverage.mean().item(),
        "per_step": {name: value.tolist() for name, value in per_step.items()},
    }


@torch.no_grad()
def validate(model, loader, device, mean, std, num_samples=32, seed=12345):
    """Reuse validation noise each epoch without advancing training RNG states."""
    was_training = model.training
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            total_loss, count = 0.0, 0
            for history, future in loader:
                history, future = history.to(device), future.to(device)
                total_loss += model.loss(history, future).item() * len(history)
                count += len(history)
            # Keep forecast randomness separate from loss-evaluation randomness.
            torch.manual_seed(seed + 1)
            generated, observed = [], []
            for history, future in loader:
                generated.append(model.sample(history.to(device), num_samples)[..., 0].cpu())
                observed.append(future[..., 0])
            metrics = forecast_metrics(
                torch.cat(generated) * std + mean,
                torch.cat(observed) * std + mean,
            )
            metrics["v_mse"] = total_loss / count
            return metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def validate_episodes(model, loader, device, mean, std, beta, num_samples=32, seed=12345):
    """Conditional reconstruction diagnostics, NOT independent future forecasts."""
    was_training = model.training
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            totals = dict(loss=0.0, diffusion_mse=0.0, kl=0.0)
            count = 0
            for episode in loader:
                episode = episode.to(device)
                terms = model.loss_terms(episode, beta)
                for name in totals:
                    totals[name] += terms[name].item() * len(episode)
                count += len(episode)
            torch.manual_seed(seed + 1)
            generated, observed = [], []
            for episode in loader:
                z = model.encode(episode.to(device)).sample()
                generated.append(model.generate(z, num_samples)[..., 0].cpu())
                observed.append(episode[..., 0])
            metrics = forecast_metrics(torch.cat(generated) * std + mean,
                                       torch.cat(observed) * std + mean)
            # Explicit names prevent reconstruction scores being mistaken for forecasts.
            result = {"reconstruction_" + k: v for k, v in metrics.items()}
            result.update({name: value / count for name, value in totals.items()})
            return result
    finally:
        model.train(was_training)
