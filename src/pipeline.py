"""Shared artifact export and inference for notebooks and local callers."""

import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
import torch

from .data import PreparedData
from .eval import reconstruction_errors
from .model import LSTMAutoencoder
from .utils import Config, save_config, save_json


def save_artifacts(data: PreparedData, config: Config) -> None:
    joblib.dump(data.preprocessor, config.output_dir / "preprocessor.joblib")
    save_config(config)
    save_json(config.output_dir / "model_config.json", {
        "input_dim": len(data.preprocessor.feature_columns),
        "hidden_dim": config.hidden_dim, "latent_dim": config.latent_dim,
        "seq_len": config.seq_len, "num_layers": config.num_layers,
        "dropout": config.dropout,
    })
    # Do not export patient identifiers or raw clinical observations.
    save_json(config.output_dir / "split_summary.json", {
        name: len(ids) for name, ids in data.split_patients.items()
    })
    save_json(config.output_dir / "versions.json", {
        name: version(name)
        for name in ("torch", "numpy", "pandas", "scikit-learn", "matplotlib", "joblib")
    })


def predict_frame(frame: pd.DataFrame, artifact_dir: Path,
                  device: str = "cpu", batch_size: int = 64) -> pd.DataFrame:
    """Load trusted artifacts and score new patient rows in the training schema."""
    artifact_dir = Path(artifact_dir)
    model_config = json.loads((artifact_dir / "model_config.json").read_text())
    model = LSTMAutoencoder(**model_config).to(device)
    model.load_state_dict(torch.load(
        artifact_dir / "best_model.pt", map_location=device, weights_only=True
    ))
    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    sequences, _, patients = preprocessor.transform(frame)
    errors = reconstruction_errors(model, sequences, batch_size)
    threshold = json.loads((artifact_dir / "threshold.json").read_text())["value"]
    return pd.DataFrame({
        "patient_id": patients, "reconstruction_error": errors,
        "anomaly": (errors > threshold).astype(int),
    })
