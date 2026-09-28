import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from src.data.dataset import RSRPEpisodeDataset
from src.models.episodic_diffusion import LatentEpisodeDiffusion, build_latent_model
from src.train.validation import forecast_metrics
from src.evaluation.temporal import save_temporal_comparisons


def load_model(checkpoint_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_latent_model(checkpoint["config"])
    if checkpoint.get("model_format") != model.FORMAT:
        raise ValueError("Incompatible checkpoint format. Retrain the latent episode model.")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate_for_all_episodes(model, dataset, num_samples, batch_size):
    device = next(model.parameters()).device
    generated, actual, latents = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = len(dataset)
    started = time.monotonic()
    completed = 0
    print(f"Generation: 0/{total:,} source episodes; {num_samples} samples/source; "
          f"{len(loader):,} batches", flush=True)
    for batch_index, episode in enumerate(loader, 1):
        z = model.encode(episode.to(device))
        generated.append(model.generate(z, num_samples)[..., 0].cpu().numpy())
        actual.append(episode[..., 0].numpy())
        latents.append(z.cpu())
        completed += len(episode)
        elapsed = time.monotonic() - started
        remaining = elapsed * (total - completed) / completed
        print(f"Generation: {completed:,}/{total:,} ({100 * completed / total:.1f}%) | "
              f"batch {batch_index}/{len(loader)} | elapsed {elapsed / 60:.1f} min | "
              f"ETA {remaining / 60:.1f} min", flush=True)
    print("Combining generated batches and restoring dBm scale...", flush=True)
    return (np.concatenate(generated)*dataset.std+dataset.mean,
            np.concatenate(actual)*dataset.std+dataset.mean,
            {"z": torch.cat(latents)})


def save_generated(generated, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, samples, length = generated.shape
    pd.DataFrame({
        "episode_id": np.repeat(np.arange(n), samples*length),
        "sample_id": np.tile(np.repeat(np.arange(samples), length), n),
        "episode_step": np.tile(np.arange(length), n*samples),
        "rsrp_generated_dbm": generated.reshape(-1),
    }).to_csv(path, index=False)


def save_boxplot(generated, actual, output_path):
    """Compare all held-out source episode values with all generated values."""
    figure, axis = plt.subplots(figsize=(16, 6))
    boxes = axis.boxplot(
        [actual.reshape(-1), generated.reshape(-1)], widths=0.55, patch_artist=True,
        showfliers=False, medianprops={"color": "black", "linewidth": 1},
    )
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor("tab:green" if index == 0 else "tab:blue")
        box.set_alpha(0.65)
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Source episode\n(all held-out episodes and steps)",
                          "Generated episode\n(all held-out episodes, samples, and steps)"])
    axis.set_ylabel("RSRP [dBm]")
    axis.set_title("Held-out evaluation data: aggregate actual vs generated RSRP")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(handles=[
        Patch(facecolor="tab:green", alpha=0.65, label="source episode"),
        Patch(facecolor="tab:blue", alpha=0.65, label="generated episode"),
    ])
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_delta_distribution(generated, actual, output_path, order=1):
    """Within-episode first/second differences; never cross episode boundaries."""
    if order not in (1, 2):
        raise ValueError("Difference order must be 1 or 2.")
    if generated.ndim != 3 or actual.shape != (generated.shape[0], generated.shape[2]):
        raise ValueError("Expected generated [N,S,L] and actual [N,L].")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.8))
    if actual.shape[1] <= order:
        axis.text(0.5, 0.5, f"At least {order + 1} episode steps required", ha="center", transform=axis.transAxes)
    else:
        source_delta = np.diff(actual, n=order, axis=-1).ravel()
        sample_delta = np.diff(generated, n=order, axis=-1).ravel()
        low = min(source_delta.min(), sample_delta.min())
        high = max(source_delta.max(), sample_delta.max())
        extent = max(abs(low), abs(high), 1.0)
        # Identical bin edges; unit-area densities account for different sample counts.
        bins = np.linspace(-extent, extent, 101)
        for delta, color, name in ((source_delta, "tab:green", "Source"),
                                   (sample_delta, "tab:blue", "Generated")):
            density, _ = np.histogram(delta, bins=bins, density=True)
            axis.stairs(density, bins, color=color, linewidth=1.6,
                        label=f"{name} (n={len(delta):,}, std={delta.std():.2f} dB)")
        axis.axvline(0, color="gray", linestyle=":", linewidth=1)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    axis.set_title("Within-episode RSRP changes: individual trajectories" if order == 1
                   else "Within-episode second differences: individual trajectories")
    axis.set_xlabel("Adjacent RSRP change: x[t+1] - x[t] (dB)" if order == 1
                    else "Second difference: x[t+2] - 2x[t+1] + x[t] (dB)")
    axis.set_ylabel("Probability density (1/dB)")
    figure.text(0.5, 0.015,
                "Same bins and unit-area normalization; all values included.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_small_multiples(generated, actual, output_path, seed=12345, episode_index=None):
    """Compare one random source with up to eight distinct random samples."""
    rng = np.random.default_rng(seed)
    default_index = int(rng.integers(len(actual)))
    episode_index = default_index if episode_index is None else int(episode_index)
    selected = rng.choice(generated.shape[1], size=min(8, generated.shape[1]), replace=False)
    rows = int(np.ceil(len(selected) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(13, 3.4 * rows),
                                squeeze=False, sharex=True, sharey=True)
    steps = np.arange(actual.shape[1])
    for axis, sample_index in zip(axes.flat, selected):
        axis.plot(steps, actual[episode_index], color="tab:green", linewidth=2,
                  label="source episode")
        axis.plot(steps, generated[episode_index, sample_index], color="tab:blue",
                  linewidth=1.5, label="generated episode")
        axis.set_title(f"Source {episode_index} / generated sample {sample_index}")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    figure.suptitle(f"One random source episode and {len(selected)} random generated samples (seed={seed})")
    figure.supxlabel("Episode step")
    figure.supylabel("RSRP [dBm]")
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {"source_episode_id": episode_index, "sample_ids": selected.tolist()}


def save_episode_examples(generated, actual, output_dir, seed=12345, count=10):
    """Save distinct random source windows; keep the original first example."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(actual)))
    candidates = np.delete(np.arange(len(actual)), first)
    indices = [first, *rng.choice(candidates, size=min(count, len(actual)) - 1, replace=False).tolist()]
    records = []
    for number, index in enumerate(indices, 1):
        filename = 'episodes.png' if number == 1 else f'episodes_{number:02d}.png'
        record = save_small_multiples(generated, actual, out / filename,
                                     seed=seed, episode_index=index)
        records.append({"file": filename, **record})
        print(f"Saved example {number}/{len(indices)}: {filename} (Source {index})", flush=True)
    (out / 'episode_examples.json').write_text(json.dumps({"seed": seed, "examples": records}, indent=2) + '\n')
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate full episodes from inferred latent variables.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2 for distribution diagnostics")
    torch.manual_seed(args.seed)
    config = yaml.safe_load(Path(args.config).read_text())
    path = Path(config["training"]["checkpoint_dir"])/"episodicdt.pt"
    print(f"[1/5] Loading checkpoint: {path}", flush=True)
    model, checkpoint = load_model(path)
    print(f"Evaluation device: {next(model.parameters()).device}", flush=True)
    print(f"[2/5] Loading evaluation data: {config['data']['evaluation_path']}", flush=True)
    dataset = RSRPEpisodeDataset(config["data"]["evaluation_path"],
                                 checkpoint["config"]["data"]["episode_length"],
                                 checkpoint["mean"], checkpoint["std"])
    cfg = config["evaluation"]
    print("[3/5] Generating episodes (progress updates after each batch)", flush=True)
    generated, actual, latent = generate_for_all_episodes(model, dataset, args.samples, cfg["batch_size"])
    print("[4/5] Saving generated/source CSVs and latent variables...", flush=True)
    save_generated(generated, cfg["generated_csv"])
    actual_path = Path(cfg["actual_csv"])
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    n, length = actual.shape
    pd.DataFrame({"episode_id": np.repeat(np.arange(n), length),
                  "episode_step": np.tile(np.arange(length), n),
                  "rsrp_actual_dbm": actual.reshape(-1)}).to_csv(actual_path, index=False)
    out = Path(cfg["figure_dir"])
    out.mkdir(parents=True, exist_ok=True)
    latent_path = Path(cfg["generated_csv"]).with_suffix(".latents.pt")
    torch.save({**latent, "model_format": model.FORMAT,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seed": args.seed, "windows": dataset.windows,
                "source_files": [str(p) for p in dataset.paths]}, latent_path)
    print("[5/5] Computing metrics and saving figures...", flush=True)
    metrics = {"reconstruction_" + k: v for k,v in forecast_metrics(
        torch.from_numpy(generated), torch.from_numpy(actual)).items()}
    metrics.update({"interpretation": "Conditional reconstruction; source episode is encoder input.",
                    "within_fixed_z_std_dbm": float(generated.std(axis=1).mean()),
                    "seed": args.seed, "num_samples": args.samples})
    (out/"episode_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
    dest = Path(cfg.get("boxplot_path", out/"episode_boxplot.png"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Saving RSRP distribution plot...", flush=True)
    save_boxplot(generated, actual, dest)
    delta_path = Path(cfg.get("delta_distribution_path", out/"episode_delta_distribution.png"))
    print("Saving first/second difference plots...", flush=True)
    save_delta_distribution(generated, actual, delta_path)
    save_delta_distribution(generated, actual, out/"episode_second_delta_distribution.png", order=2)
    print("Computing and saving temporal comparisons...", flush=True)
    save_temporal_comparisons(generated, actual, out)
    print("Saving example trajectories...", flush=True)
    save_episode_examples(generated, actual, out, seed=args.seed)
    print(f"Saved full generated episodes: {cfg['generated_csv']}")
    print(f"Saved reusable latent variables: {latent_path}")
    print("Evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
