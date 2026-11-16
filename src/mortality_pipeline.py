"""End-to-end mortality experiment using only a prepared local exchange file."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
import torch

from .mortality_eval import evaluate_mortality, reconstruction_scores
from .mortality_exchange import validate_exchange
from .mortality_model import MortalityConfig, train_autoencoder
from .utils import save_json


def run_mortality_experiment(arrays: dict[str, np.ndarray], config: MortalityConfig,
                             output_dir: Path, *, source: str,
                             provenance: dict | None = None) -> dict:
    validate_exchange(arrays)
    if source not in {"synthetic", "prepared"}:
        raise ValueError("Source must identify synthetic or prepared data.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    model, history = train_autoencoder(arrays["train_x"], arrays["validation_x"], config)
    validation_scores = reconstruction_scores(model, arrays["validation_x"], config.batch_size)
    test_scores = reconstruction_scores(model, arrays["test_x"], config.batch_size)
    metrics = evaluate_mortality(
        arrays["validation_y"], validation_scores, arrays["test_y"], test_scores,
        arrays["test_group"], seed=config.seed, bootstrap_samples=config.bootstrap_samples,
    )
    metrics["data_source"] = source
    metrics["smoke_test"] = config.fast_dev_run
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_json(output_dir / "preprocessor.json", json.loads(arrays["preprocessor_json"].item()))
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "model_spec.json", {
        "input_channels": 6, "output_channels": 3,
        "encoder_lstm_units": [64, 32], "decoder_lstm_units": [32, 64],
        "loss": "mean_squared_error_over_observed_biomarkers",
        "score": "mean_absolute_error_over_observed_biomarkers",
    })
    save_json(output_dir / "provenance.json", {**(provenance or {}), "source": source})
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "versions.json", {
        package: version(package) for package in ("torch", "numpy", "scikit-learn", "matplotlib")
    })
    fpr, tpr, _ = roc_curve(arrays["test_y"], test_scores)
    precision, recall, _ = precision_recall_curve(arrays["test_y"], test_scores)
    for name, x, y, xlabel, ylabel in (
        ("roc_curve", fpr, tpr, "False positive rate", "True positive rate"),
        ("pr_curve", recall, precision, "Recall", "Precision"),
    ):
        fig, axis = plt.subplots()
        axis.plot(x, y)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("_", " ").upper())
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png")
        plt.close(fig)
    fig, axis = plt.subplots()
    for key, values in history.items():
        axis.plot(range(1, len(values) + 1), values, label=key)
    axis.set(xlabel="Epoch", ylabel="Observed-value MSE")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png")
    plt.close(fig)
    return metrics
