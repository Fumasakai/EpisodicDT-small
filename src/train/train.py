import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import RSRPWindowDataset
from src.models.episodic_diffusion import EpisodicDiffusion


def temporal_train_validation_indices(dataset, validation_fraction):
    """Split every trace chronologically without overlapping 40-step episodes."""
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


@torch.no_grad()
def validation_loss(model, loader, device):
    """Return mean diffusion MSE on chronologically held-out training data."""
    model.eval()
    losses = []
    for history, future in loader:
        losses.append(model.loss(history.to(device), future.to(device)).item())
    return float(np.mean(losses))


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = Path(data_cfg["train_path"])
    evaluation_path = Path(data_cfg["evaluation_path"])
    train_paths = sorted(train_path.glob("*.csv"))
    evaluation_paths = sorted(evaluation_path.glob("*.csv"))
    print(f"training scenario files: {len(train_paths)}, evaluation scenario files: {len(evaluation_paths)}")
    dataset = RSRPWindowDataset(train_path, data_cfg["sequence_length"], diffusion_cfg["future_length"])
    train_indices, validation_indices = temporal_train_validation_indices(
        dataset, training_cfg["validation_fraction"]
    )
    print(f"training episodes: {len(train_indices)}, validation episodes: {len(validation_indices)}")
    loader = DataLoader(Subset(dataset, train_indices), batch_size=training_cfg["batch_size"], shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=training_cfg["batch_size"])
    model = EpisodicDiffusion(
        data_cfg["input_dim"], diffusion_cfg["future_length"], model_cfg["hidden_dim"],
        model_cfg["latent_dim"], diffusion_cfg["timesteps"], model_cfg["transformer_heads"],
        model_cfg["transformer_layers"], model_cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["learning_rate"])

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, training_cfg["epochs"] + 1):
        losses = []
        model.train()
        for history, future in tqdm(loader, desc=f"epoch {epoch:03d}", leave=False):
            history, future = history.to(device), future.to(device)
            optimizer.zero_grad()
            loss = model.loss(history, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        current_validation_loss = validation_loss(model, validation_loader, device)
        print(
            f"epoch={epoch:03d} train_diffusion_mse={np.mean(losses):.5f} "
            f"validation_diffusion_mse={current_validation_loss:.5f}"
        )
        if current_validation_loss < best_validation_loss - training_cfg["early_stopping_min_delta"]:
            best_validation_loss = current_validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= training_cfg["min_epochs"] and epochs_without_improvement >= training_cfg["early_stopping_patience"]:
            print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    model.load_state_dict(best_state)
    print(f"restored best model: epoch={best_epoch}, validation_diffusion_mse={best_validation_loss:.5f}")

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "mean": dataset.mean, "std": dataset.std, "config": config,
        "train_files": [str(path) for path in train_paths],
        "evaluation_files": [str(path) for path in evaluation_paths],
        "best_epoch": best_epoch,
        "best_validation_diffusion_mse": best_validation_loss,
    }, checkpoint_dir / "episodicdt.pt")
    print(f"saved checkpoint: {checkpoint_dir / 'episodicdt.pt'}")


if __name__ == "__main__":
    main()
