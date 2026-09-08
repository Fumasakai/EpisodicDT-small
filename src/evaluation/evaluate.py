import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from src.data.dataset import RSRPWindowDataset
from src.models.episodic_diffusion import EpisodicDiffusion


def generate_for_all_episodes(model, dataset, num_samples, batch_size):
    """Generate futures for every held-out episode, in batches."""
    generated_batches, actual_batches = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for history, actual_future in loader:
        normalized_samples = model.sample(history, num_samples)[:, :, :, 0].numpy()
        generated_batches.append(normalized_samples)
        actual_batches.append(actual_future[:, :, 0].numpy())
    generated = np.concatenate(generated_batches, axis=0)
    actual = np.concatenate(actual_batches, axis=0)
    return generated * dataset.std + dataset.mean, actual * dataset.std + dataset.mean


def save_boxplot(generated, actual, output_path):
    """Compare all held-out actual future values with all generated values."""
    figure, axis = plt.subplots(figsize=(16, 6))
    boxes = axis.boxplot(
        [actual.reshape(-1), generated.reshape(-1)], widths=0.55, patch_artist=True,
        showfliers=False, medianprops={"color": "black", "linewidth": 1},
    )
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor("tab:green" if index == 0 else "tab:blue")
        box.set_alpha(0.65)
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Actual future\n(all held-out episodes and steps)",
                          "Generated future\n(all held-out episodes, samples, and steps)"])
    axis.set_ylabel("RSRP [dBm]")
    axis.set_title("Held-out evaluation data: aggregate actual vs generated RSRP")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(handles=[
        Patch(facecolor="tab:green", alpha=0.65, label="actual future"),
        Patch(facecolor="tab:blue", alpha=0.65, label="generated future"),
    ])
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_step_boxplot(generated, actual, output_path):
    """Pool episodes (and generated samples) separately at each future step."""
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
    axis.set_xlabel("Forecast step (0 = first future observation)")
    axis.set_ylabel("RSRP [dBm]")
    axis.set_title("Actual vs generated RSRP distribution at each future step")
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


def save_small_multiples(dataset, generated, actual, threshold, output_path, episode_count):
    """Plot representative held-out trajectories with generated uncertainty bands."""
    selected = np.linspace(0, len(dataset) - 1, num=episode_count, dtype=int)
    columns = 2
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), squeeze=False)
    x_future = np.arange(actual.shape[1])

    for panel_index, episode_index in enumerate(selected):
        axis = axes.flat[panel_index]
        history, _ = dataset[episode_index]
        history_values = history[:, 0].numpy() * dataset.std + dataset.mean
        x_history = np.arange(-len(history_values), 0)
        trajectories = generated[episode_index]
        median = np.median(trajectories, axis=0)
        lower_90, lower_50 = np.percentile(trajectories, [5, 25], axis=0)
        upper_50, upper_90 = np.percentile(trajectories, [75, 95], axis=0)
        mae = np.abs(median - actual[episode_index]).mean()

        axis.plot(x_history, history_values, color="black", label="history")
        axis.fill_between(x_future, lower_90, upper_90, color="tab:blue", alpha=0.12, label="generated 90% interval")
        axis.fill_between(x_future, lower_50, upper_50, color="tab:blue", alpha=0.28, label="generated 50% interval")
        axis.plot(x_future, median, color="tab:blue", linewidth=2, label="generated median")
        axis.plot(x_future, actual[episode_index], color="tab:green", linewidth=2, label="actual future")
        axis.axhline(threshold, color="tab:red", linestyle="--", linewidth=1, label="danger threshold")
        axis.set_title(f"Evaluation episode {episode_index} (median MAE: {mae:.2f} dBm)")
        axis.grid(alpha=0.3)
        if panel_index == 0:
            axis.legend(fontsize=8, loc="best")

    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    figure.suptitle("Generated and actual RSRP trajectories for representative held-out episodes", y=1.01)
    figure.supxlabel("Steps relative to forecast start")
    figure.supylabel("RSRP [dBm]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Sample and evaluate EpisodicDT forecasts.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    training_cfg = config["training"]
    checkpoint = torch.load(Path(training_cfg["checkpoint_dir"]) / "episodicdt.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("model_format") != EpisodicDiffusion.FORMAT:
        raise ValueError("This checkpoint uses the old model format. Retrain with src.train.train first.")
    # Architecture and history length must match the trained checkpoint.
    saved_config = checkpoint["config"]
    data_cfg, model_cfg, diffusion_cfg = saved_config["data"], saved_config["model"], saved_config["diffusion"]
    evaluation_path = Path(config["data"]["evaluation_path"])
    evaluation_paths = sorted(evaluation_path.glob("*.csv"))
    print(f"held-out evaluation scenario files: {len(evaluation_paths)}")
    dataset = RSRPWindowDataset(
        evaluation_path, data_cfg["sequence_length"], diffusion_cfg["future_length"],
        checkpoint["mean"], checkpoint["std"],
    )
    model = EpisodicDiffusion(
        data_cfg["input_dim"], diffusion_cfg["future_length"], model_cfg["hidden_dim"],
        model_cfg["latent_dim"], diffusion_cfg["timesteps"], model_cfg["transformer_heads"],
        model_cfg["transformer_layers"], model_cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_generated, all_actual = generate_for_all_episodes(
        model, dataset, args.samples, config["evaluation"]["batch_size"]
    )
    history, actual_future = dataset[len(dataset) - 1]
    samples = all_generated[-1]
    history_values = history[:, 0].numpy() * dataset.std + dataset.mean
    actual_values = all_actual[-1]
    mean_forecast = samples.mean(axis=0)
    threshold = config["evaluation"]["danger_threshold"]
    danger_probability = float((samples.min(axis=1) < threshold).mean())
    mae = float(np.abs(mean_forecast - actual_values).mean())
    overall_mae = float(np.abs(all_generated.mean(axis=1) - all_actual).mean())
    print(f"final-episode forecast MAE: {mae:.2f} dBm")
    print(f"held-out overall forecast MAE: {overall_mae:.2f} dBm")
    print(f"P(any future RSRP < {threshold:.1f} dBm): {danger_probability:.1%}")

    output_dir = Path(config["evaluation"]["figure_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_csv = Path(config["evaluation"]["generated_csv"])
    generated_csv.parent.mkdir(parents=True, exist_ok=True)
    episode_count, sample_count, future_length = all_generated.shape
    generated = pd.DataFrame({
        "episode_id": np.repeat(np.arange(episode_count), sample_count * future_length),
        "sample_id": np.tile(np.repeat(np.arange(sample_count), future_length), episode_count),
        "forecast_step": np.tile(np.arange(future_length), episode_count * sample_count),
        "rsrp_generated_dbm": all_generated.reshape(-1),
    })
    generated.to_csv(generated_csv, index=False)
    print(f"saved generated RSRP: {generated_csv}")
    actual_csv = Path(config["evaluation"]["actual_csv"])
    actual = pd.DataFrame({
        "episode_id": np.repeat(np.arange(episode_count), future_length),
        "forecast_step": np.tile(np.arange(future_length), episode_count),
        "rsrp_actual_dbm": all_actual.reshape(-1),
    })
    actual.to_csv(actual_csv, index=False)
    print(f"saved actual held-out RSRP: {actual_csv}")
    boxplot_path = Path(config["evaluation"]["boxplot_path"])
    boxplot_path.parent.mkdir(parents=True, exist_ok=True)
    save_boxplot(all_generated, all_actual, boxplot_path)
    print(f"saved boxplot: {boxplot_path}")

    step_boxplot_path = Path(config["evaluation"].get(
        "step_boxplot_path", output_dir / "forecast_boxplot_by_step.png"
    ))
    step_boxplot_path.parent.mkdir(parents=True, exist_ok=True)
    save_step_boxplot(all_generated, all_actual, step_boxplot_path)
    print(f"saved per-step boxplot: {step_boxplot_path}")

    path = output_dir / "forecast.png"
    save_small_multiples(
        dataset, all_generated, all_actual, threshold, path,
        config["evaluation"]["small_multiples_episodes"],
    )
    print(f"saved figure: {path}")


if __name__ == "__main__":
    main()
