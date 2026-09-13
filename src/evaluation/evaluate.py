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
        raise ValueError("Incompatible forecast checkpoint. Retrain the latent episode model.")
    model = build_latent_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate_for_all_episodes(model, dataset, num_samples, batch_size):
    generated, actual, latents, means, scales = [], [], [], [], []
    for episode in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        posterior = model.encode(episode)
        z = posterior.sample()
        generated.append(model.generate(z, num_samples)[..., 0].numpy())
        actual.append(episode[..., 0].numpy())
        latents.append(z)
        means.append(posterior.loc)
        scales.append(posterior.scale)
    return (np.concatenate(generated)*dataset.std+dataset.mean,
            np.concatenate(actual)*dataset.std+dataset.mean,
            {"z": torch.cat(latents), "mu": torch.cat(means), "scale": torch.cat(scales)})


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


def save_step_boxplot(generated, actual, output_path):
    """Pool episodes (and generated samples) separately at each episode step."""
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
    axis.set_title("Actual vs generated RSRP distribution at each episode step")
    axis.legend(handles=[
        Patch(facecolor="tab:green", alpha=0.65,
              label=f"Actual: {actual.shape[0]:,} values / step"),
        Patch(facecolor="tab:blue", alpha=0.65,
              label=f"Generated: {generated.shape[0] * generated.shape[1]:,} values / step"),
    ])
    axis.grid(axis="y", alpha=0.3)
    figure.text(0.5, 0.01,
                "Boxes: 25-75%; line: median; whiskers: within 1.5 IQR; outliers hidden",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
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
    save_small_multiples(dataset, generated, actual, out/"episodes.png",
                         min(len(dataset), cfg["small_multiples_episodes"]))
    print(f"Saved full generated episodes: {cfg['generated_csv']}")
    print(f"Saved reusable latent variables: {latent_path}")


if __name__ == "__main__":
    main()
