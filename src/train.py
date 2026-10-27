"""Mini-batch training with early stopping and best-checkpoint restoration."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from .model import LSTMAutoencoder
from .utils import Config, save_json


def train_model(model: LSTMAutoencoder, train: np.ndarray,
                validation: np.ndarray, config: Config) -> dict:
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Training and validation data must be nonempty.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = config.device
    model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = [
        DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=config.batch_size,
                   shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)
        for data, shuffle in ((train, True), (validation, False))
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    history = {"train_loss": [], "validation_loss": []}
    best, stale = float("inf"), 0
    for epoch in range(min(config.epochs, 2) if config.fast_dev_run else config.epochs):
        losses = []
        for loader, training in zip(loaders, (True, False)):
            model.train(training)
            total = 0.0
            with torch.set_grad_enabled(training):
                for (batch,) in loader:
                    batch = batch.to(device)
                    if training:
                        optimizer.zero_grad()
                    loss = nn.functional.mse_loss(model(batch), batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss; check input data and learning rate.")
                    if training:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total += loss.item() * len(batch)
            losses.append(total / len(loader.dataset))
        history["train_loss"].append(losses[0])
        history["validation_loss"].append(losses[1])
        scheduler.step(losses[1])
        print(f"Epoch {epoch + 1}: train={losses[0]:.6f}, validation={losses[1]:.6f}")
        if losses[1] < best:
            best, stale = losses[1], 0
            torch.save(model.state_dict(), config.output_dir / "best_model.pt")
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(torch.load(
        config.output_dir / "best_model.pt", map_location=device, weights_only=True
    ))
    model.eval()
    save_json(config.output_dir / "history.json", history)
    fig, axis = plt.subplots()
    for label, values in history.items():
        axis.plot(range(1, len(values) + 1), values, label=label)
    axis.set(xlabel="Epoch", ylabel="MSE")
    axis.legend()
    fig.tight_layout()
    fig.savefig(config.output_dir / "training_curve.png")
    plt.close(fig)
    return history
