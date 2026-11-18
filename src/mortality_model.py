"""Observed-value LSTM autoencoder for the 24-hour mortality experiment."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import set_seed


class MortalityAutoencoder(nn.Module):
    """Encode six input channels and reconstruct three biomarker channels."""

    def __init__(self):
        super().__init__()
        self.encoder_1 = nn.LSTM(6, 64, batch_first=True)
        self.encoder_2 = nn.LSTM(64, 32, batch_first=True)
        self.decoder_1 = nn.LSTM(32, 32, batch_first=True)
        self.decoder_2 = nn.LSTM(32, 64, batch_first=True)
        self.output = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder_1(x)
        _, (hidden, _) = self.encoder_2(encoded)
        repeated = hidden[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder_1(repeated)
        decoded, _ = self.decoder_2(decoded)
        return self.output(decoded)


def observed_error(x: torch.Tensor, reconstruction: torch.Tensor,
                   *, squared: bool) -> torch.Tensor:
    """Return one masked mean error per admission, weighted equally."""
    if x.ndim != 3 or x.shape[-1] != 6 or reconstruction.shape != x[:, :, :3].shape:
        raise ValueError("Expected six input and three reconstructed biomarker channels.")
    mask = x[:, :, 3:]
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Observation masks must be binary.")
    denominator = mask.sum(dim=(1, 2))
    if torch.any(denominator == 0):
        raise ValueError("Every admission must have an observed biomarker.")
    difference = reconstruction - x[:, :, :3]
    loss = difference.square() if squared else difference.abs()
    return (loss * mask).sum(dim=(1, 2)) / denominator


@dataclass
class MortalityConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.0001
    bootstrap_samples: int = 1000
    fast_dev_run: bool = False
    device_preference: str = "cuda"

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.bootstrap_samples < 2:
            raise ValueError("Epochs and batch size must be positive; bootstrap needs at least 2 samples.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        if self.device_preference not in {"cpu", "cuda"}:
            raise ValueError("Device must be cpu or cuda.")

    @property
    def device(self) -> torch.device:
        if self.device_preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def train_autoencoder(train_x: np.ndarray, validation_x: np.ndarray | None,
                      config: MortalityConfig) -> tuple[MortalityAutoencoder, dict]:
    """Train with validation checkpointing, or fixed epochs when validation is None."""
    set_seed(config.seed)
    if not len(train_x) or (validation_x is not None and not len(validation_x)):
        raise ValueError("Training and validation data must be nonempty.")
    if config.device_preference == "cuda" and config.device.type == "cpu":
        print("CUDA unavailable; training on CPU.")
    device = config.device
    model = MortalityAutoencoder().to(device)
    generator = torch.Generator().manual_seed(config.seed)
    inputs = [(train_x, True)]
    if validation_x is not None:
        inputs.append((validation_x, False))
    loaders = [
        DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32, copy=False))),
                   batch_size=config.batch_size, shuffle=shuffle, generator=generator)
        for x, shuffle in inputs
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {"train_loss": []}
    if validation_x is not None:
        history["validation_loss"] = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(min(config.epochs, 2) if config.fast_dev_run else config.epochs):
        for loader, training, key in zip(loaders, (True, False), history):
            model.train(training)
            total = 0.0
            with torch.set_grad_enabled(training):
                for (batch,) in loader:
                    batch = batch.to(device)
                    if training:
                        optimizer.zero_grad()
                    loss = observed_error(batch, model(batch), squared=True).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite reconstruction loss.")
                    if training:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total += loss.item() * len(batch)
            history[key].append(total / len(loader.dataset))
        message = f"Epoch {epoch + 1}: train={history['train_loss'][-1]:.6f}"
        if validation_x is not None:
            current = history["validation_loss"][-1]
            if current < best_loss:
                best_loss, best_state = current, deepcopy(model.state_dict())
            message += f", validation={current:.6f}"
        print(message)
    if validation_x is not None:
        if best_state is None:
            raise RuntimeError("No valid model checkpoint was produced.")
        model.load_state_dict(best_state)
    model.eval()
    return model, history
