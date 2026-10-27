"""Configuration, reproducibility and artifact persistence."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch


@dataclass
class Config:
    seed: int = 42
    batch_size: int = 32
    seq_len: int = 24
    latent_dim: int = 16
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 20
    learning_rate: float = 1e-3
    patience: int = 5
    threshold_percentile: float = 95.0
    device_preference: str = "cuda"
    fast_dev_run: bool = True
    output_dir: Path = Path("/content/outputs/neonatal-sepsis")
    drive_output_dir: Path | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.drive_output_dir is not None:
            self.drive_output_dir = Path(self.drive_output_dir)
        for name in ("batch_size", "seq_len", "latent_dim", "hidden_dim",
                     "num_layers", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.threshold_percentile < 100:
            raise ValueError("threshold_percentile must be between 0 and 100")
        if self.device_preference not in {"cpu", "cuda"}:
            raise ValueError("device_preference must be cpu or cuda")

    @property
    def device(self) -> torch.device:
        if self.device_preference == "cuda" and not torch.cuda.is_available():
            print("CUDA unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(self.device_preference)


def set_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def save_config(config: Config) -> None:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["drive_output_dir"] = (
        str(config.drive_output_dir) if config.drive_output_dir else None
    )
    save_json(config.output_dir / "config.json", payload)


def save_to_drive_if_enabled(config: Config) -> Path | None:
    if config.drive_output_dir is None:
        print("Drive persistence disabled; download outputs before disconnecting.")
        return None
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive before enabling Drive persistence.")
    source = config.output_dir.resolve()
    destination = config.drive_output_dir.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Drive output must not be inside the local output directory.")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Artifacts copied to {destination}")
    return destination
