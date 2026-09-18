import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import RSRPEpisodeDataset
from src.models.episodic_diffusion import build_latent_model
from src.train.validation import validate_episodes


def temporal_train_validation_indices(dataset, validation_fraction):
    """Split every trace chronologically, excluding boundary-crossing windows."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    train_indices, validation_indices = [], []
    window_length = dataset.history_length + dataset.future_length
    for trace_index, values in enumerate(dataset.traces):
        split_point = int(len(values) * (1 - validation_fraction))
        for index, (window_trace, start) in enumerate(dataset.windows):
            if window_trace != trace_index:
                continue
            if start + window_length <= split_point:
                train_indices.append(index)
            elif start >= split_point:
                validation_indices.append(index)
    if not train_indices or not validation_indices:
        raise ValueError("Temporal split did not produce both training and validation episodes.")
    return train_indices, validation_indices


def main():
    parser = argparse.ArgumentParser(description="Train compact EpisodicDT diffusion model.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_cfg, model_cfg = config["data"], config["model"]
    diffusion_cfg, training_cfg = config["diffusion"], config["training"]
    if (training_cfg["epochs"] < 1 or training_cfg["early_stopping_patience"] < 1
            or training_cfg["early_stopping_min_delta"] < 0
            or training_cfg["validation_samples"] < 2):
        raise ValueError("Invalid epoch, patience, min_delta or validation_samples setting.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = Path(data_cfg["train_path"])
    evaluation_path = Path(data_cfg["evaluation_path"])
    train_paths = sorted(train_path.glob("*.csv"))
    evaluation_paths = sorted(evaluation_path.glob("*.csv"))
    print(f"training scenario files: {len(train_paths)}, evaluation scenario files: {len(evaluation_paths)}")
    dataset = RSRPEpisodeDataset(train_path, data_cfg["episode_length"])
    train_indices, validation_indices = temporal_train_validation_indices(
        dataset, training_cfg["validation_fraction"]
    )
    # Fit normalization only on the optimization prefixes, excluding validation.
    prefixes = [values[:int(len(values) * (1 - training_cfg["validation_fraction"]))]
                for values in dataset.traces]
    training_values = np.concatenate(prefixes)
    dataset.mean = float(training_values.mean())
    dataset.std = max(float(training_values.std()), 1e-6)
    print(f"training episodes: {len(train_indices)}, validation episodes: {len(validation_indices)}")
    loader = DataLoader(Subset(dataset, train_indices), batch_size=training_cfg["batch_size"], shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=training_cfg["batch_size"])
    model = build_latent_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["learning_rate"])

    best_score = float("inf")
    stopping_reference = float("inf")
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_log = []
    for epoch in range(1, training_cfg["epochs"] + 1):
        totals, count = dict(loss=0.0, diffusion_mse=0.0), 0
        model.train()
        for episode in tqdm(loader, desc=f"epoch {epoch:03d}", leave=False):
            episode = episode.to(device)
            optimizer.zero_grad()
            terms = model.loss_terms(episode)
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name in totals:
                totals[name] += terms[name].item() * len(episode)
            count += len(episode)
        metrics = validate_episodes(model, validation_loader, device, dataset.mean, dataset.std,
                                    training_cfg["validation_samples"], training_cfg["validation_seed"])
        score = metrics["loss"]
        if not all(np.isfinite(metrics[k]) for k in ("loss", "diffusion_mse")):
            raise RuntimeError("Non-finite validation loss.")
        print(f"epoch={epoch:03d} train_loss={totals['loss']/count:.5f} "
              f"validation_loss={score:.5f} diffusion={metrics['diffusion_mse']:.5f} "
              f"reconstruction_crps={metrics['reconstruction_crps_dbm']:.4f}")
        # Save the actual minimum even if the improvement is smaller than min_delta.
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = copy.deepcopy(metrics)
            torch.save({
                "model_format": model.FORMAT,
                "model": model.state_dict(), "mean": dataset.mean, "std": dataset.std,
                "config": config, "best_epoch": best_epoch,
                "selection_metric": "validation_total_loss", "best_validation_metrics": best_metrics,
                "train_files": [str(path) for path in train_paths],
                "evaluation_files": [str(path) for path in evaluation_paths],
            }, checkpoint_dir / "episodicdt.pt")
        if score < stopping_reference - training_cfg["early_stopping_min_delta"]:
            stopping_reference = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        training_log.append({"epoch": epoch, **{"train_" + k: v/count for k, v in totals.items()}, **metrics})
        (checkpoint_dir / "training_metrics.json").write_text(
            json.dumps(training_log, indent=2, allow_nan=False) + "\n"
        )
        if epoch >= training_cfg["min_epochs"] and epochs_without_improvement >= training_cfg["early_stopping_patience"]:
            print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    print(f"best saved model: epoch={best_epoch}, validation_total_loss={best_score:.5f}")
    print(f"saved checkpoint: {checkpoint_dir / 'episodicdt.pt'}")


if __name__ == "__main__":
    main()
