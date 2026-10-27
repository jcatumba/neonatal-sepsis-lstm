"""Batched scoring and calibration on normal validation data, never test data."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

from .model import LSTMAutoencoder
from .utils import Config, save_json


@torch.no_grad()
def reconstruction_errors(model: LSTMAutoencoder, sequences: np.ndarray,
                          batch_size: int = 64) -> np.ndarray:
    if sequences.ndim != 3 or not len(sequences) or batch_size < 1:
        raise ValueError("Expected nonempty (samples, sequence, features) data and positive batch_size.")
    if not np.isfinite(sequences).all():
        raise ValueError("Sequences must be finite.")
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.as_tensor(sequences[start:start + batch_size],
                                dtype=torch.float32, device=device)
        scores.append(model.reconstruction_error(batch).cpu().numpy())
    return np.concatenate(scores)


def evaluate_model(model: LSTMAutoencoder, validation: np.ndarray,
                   test: np.ndarray, labels: np.ndarray, config: Config):
    if labels.shape != (len(test),) or set(labels) != {0, 1}:
        raise ValueError("Test labels must align with sequences and contain both 0 and 1.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = reconstruction_errors(model, validation, config.batch_size)
    threshold = float(np.percentile(calibration, config.threshold_percentile))
    errors = reconstruction_errors(model, test, config.batch_size)
    predictions = (errors > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "auroc": float(roc_auc_score(labels, errors)),
        "auprc": float(average_precision_score(labels, errors)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": threshold, "test_sequences": len(test),
    }
    save_json(config.output_dir / "threshold.json", {
        "value": threshold, "percentile": config.threshold_percentile,
        "source": "normal_validation", "comparison": "error > threshold",
    })
    save_json(config.output_dir / "metrics.json", metrics)
    fpr, tpr, _ = roc_curve(labels, errors)
    precision, recall, _ = precision_recall_curve(labels, errors)
    for name, x, y, xlabel, ylabel in (
        ("roc_curve", fpr, tpr, "False positive rate", "True positive rate"),
        ("pr_curve", recall, precision, "Recall", "Precision"),
    ):
        fig, axis = plt.subplots()
        axis.plot(x, y)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("_", " ").upper())
        fig.tight_layout()
        fig.savefig(config.output_dir / f"{name}.png")
        plt.close(fig)
    return metrics, errors
