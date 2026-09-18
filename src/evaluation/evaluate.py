import argparse
import hashlib
import json
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


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_format") != LatentEpisodeDiffusion.FORMAT:
        raise ValueError("Incompatible checkpoint format. Retrain the latent episode model.")
    model = build_latent_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate_for_all_episodes(model, dataset, num_samples, batch_size):
    generated, actual, latents = [], [], []
    for episode in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        z = model.encode(episode)
        generated.append(model.generate(z, num_samples)[..., 0].numpy())
        actual.append(episode[..., 0].numpy())
        latents.append(z)
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


def save_step_boxplot(generated, actual, output_path, median_samples=False):
    """Compare per-step distributions; optionally reduce samples within each episode."""
    if median_samples:
        generated = np.median(generated, axis=1, keepdims=True)
    future_length = actual.shape[1]
    steps = np.arange(future_length)
    figure, axis = plt.subplots(figsize=(max(10, future_length * 0.75), 6))
    for values, offset, color in (
        ([actual[:, step] for step in steps], -0.18, "tab:green"),
        ([generated[:, :, step].reshape(-1) for step in steps], 0.18, "tab:blue"),
    ):
        boxes = axis.boxplot(
            values, positions=steps + offset, widths=0.3,
            patch_artist=True, manage_ticks=False, showfliers=False, whis=1.5,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.65)
    axis.set_xticks(steps)
    axis.set_xlabel("Episode step")
    axis.set_ylabel("RSRP [dBm]")
    axis.set_title("Actual vs per-episode generated median RSRP distribution at each episode step"
                   if median_samples else "Actual vs generated RSRP distribution at each episode step")
    axis.legend(handles=[
        Patch(facecolor="tab:green", alpha=0.65,
              label=f"Actual: {actual.shape[0]:,} values / step"),
        Patch(facecolor="tab:blue", alpha=0.65,
              label=f"{'Generated medians' if median_samples else 'Generated'}: {generated.shape[0] * generated.shape[1]:,} values / step"),
    ])
    axis.grid(axis="y", alpha=0.3)
    figure.text(0.5, 0.01,
                "Boxes: 25-75%; line: median; whiskers: within 1.5 IQR; outliers hidden",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_step_median_boxplot(generated, actual, output_path):
    """One generated median per source episode per step; both sides have N values."""
    save_step_boxplot(generated, actual, output_path, median_samples=True)


def save_delta_distribution(generated, actual, output_path):
    """Histogram of within-episode adjacent differences; never cross episode boundaries."""
    if generated.ndim != 3 or actual.shape != (generated.shape[0], generated.shape[2]):
        raise ValueError("Expected generated [N,S,L] and actual [N,L].")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    if actual.shape[1] < 2:
        for axis in axes:
            axis.text(0.5, 0.5, "At least two episode steps required", ha="center", transform=axis.transAxes)
    else:
        source_delta = np.diff(actual, axis=-1).ravel()
        sample_delta = np.diff(generated, axis=-1).ravel()
        median_delta = np.diff(np.median(generated, axis=1), axis=-1).ravel()
        low = min(source_delta.min(), sample_delta.min(), median_delta.min())
        high = max(source_delta.max(), sample_delta.max(), median_delta.max())
        extent = max(abs(low), abs(high), 1.0)
        # Identical bin edges; unit-area densities account for different sample counts.
        bins = np.linspace(-extent, extent, 101)
        for axis, values, title, label in (
            (axes[0], sample_delta, "Individual generated trajectories", "Generated"),
            (axes[1], median_delta, "Per-episode median trajectory", "Generated median trajectory"),
        ):
            for delta, color, name in ((source_delta, "tab:green", "Source"),
                                        (values, "tab:blue", label)):
                density, _ = np.histogram(delta, bins=bins, density=True)
                axis.stairs(density, bins, color=color, linewidth=1.6,
                            label=f"{name} (n={len(delta):,}, std={delta.std():.2f} dB)")
            axis.axvline(0, color="gray", linestyle=":", linewidth=1)
            axis.set_title(title)
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.set_xlabel("Adjacent RSRP change: x[t+1] - x[t] (dB)")
    axes[0].set_ylabel("Probability density (1/dB)")
    figure.suptitle("Within-episode RSRP change distributions")
    figure.text(0.5, 0.015,
                "Same bins and unit-area normalization; all values included. "
                "Right: difference of the median trajectory, not median of differences.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.055, 1, 0.95))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_small_multiples(dataset, generated, actual, output_path, episode_count):
    """Plot representative held-out trajectories with generated uncertainty bands."""
    selected = np.linspace(0, len(dataset) - 1, num=episode_count, dtype=int)
    columns = 2
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), squeeze=False)
    x_future = np.arange(actual.shape[1])

    for panel_index, episode_index in enumerate(selected):
        axis = axes.flat[panel_index]
        trajectories = generated[episode_index]
        median = np.median(trajectories, axis=0)
        lower_90, lower_50 = np.percentile(trajectories, [5, 25], axis=0)
        upper_50, upper_90 = np.percentile(trajectories, [75, 95], axis=0)
        mae = np.abs(median - actual[episode_index]).mean()

        axis.fill_between(x_future, lower_90, upper_90, color="tab:blue", alpha=0.12, label="generated 90% interval")
        axis.fill_between(x_future, lower_50, upper_50, color="tab:blue", alpha=0.28, label="generated 50% interval")
        axis.plot(x_future, median, color="tab:blue", linewidth=2, label="generated median")
        axis.plot(x_future, actual[episode_index], color="tab:green", linewidth=2, label="source episode")
        axis.set_title(f"Evaluation episode {episode_index} (median MAE: {mae:.2f} dBm)")
        axis.grid(alpha=0.3)
        if panel_index == 0:
            axis.legend(fontsize=8, loc="best")

    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    figure.suptitle("Conditional episode generation (source used by encoder; not forecasting)", y=1.01)
    figure.supxlabel("Episode step")
    figure.supylabel("RSRP [dBm]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


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
    model, checkpoint = load_model(path)
    dataset = RSRPEpisodeDataset(config["data"]["evaluation_path"],
                                 checkpoint["config"]["data"]["episode_length"],
                                 checkpoint["mean"], checkpoint["std"])
    cfg = config["evaluation"]
    generated, actual, latent = generate_for_all_episodes(model, dataset, args.samples, cfg["batch_size"])
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
    metrics = {"reconstruction_" + k: v for k,v in forecast_metrics(
        torch.from_numpy(generated), torch.from_numpy(actual)).items()}
    metrics.update({"interpretation": "Conditional reconstruction; source episode is encoder input.",
                    "within_fixed_z_std_dbm": float(generated.std(axis=1).mean()),
                    "seed": args.seed, "num_samples": args.samples})
    (out/"episode_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
    for key, fn in (("boxplot_path", save_boxplot), ("step_boxplot_path", save_step_boxplot)):
        dest = Path(cfg.get(key, out/"episode_boxplot_by_step.png"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        fn(generated, actual, dest)
    median_path = Path(cfg.get("step_median_boxplot_path", out/"episode_median_boxplot_by_step.png"))
    median_path.parent.mkdir(parents=True, exist_ok=True)
    save_step_median_boxplot(generated, actual, median_path)
    delta_path = Path(cfg.get("delta_distribution_path", out/"episode_delta_distribution.png"))
    save_delta_distribution(generated, actual, delta_path)
    save_small_multiples(dataset, generated, actual, out/"episodes.png",
                         min(len(dataset), cfg["small_multiples_episodes"]))
    print(f"Saved full generated episodes: {cfg['generated_csv']}")
    print(f"Saved reusable latent variables: {latent_path}")


if __name__ == "__main__":
    main()
