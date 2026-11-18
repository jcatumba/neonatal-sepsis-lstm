"""Mortality score evaluation with validation-only threshold selection."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .mortality_model import MortalityAutoencoder, observed_error


@torch.no_grad()
def reconstruction_scores(model: MortalityAutoencoder, x: np.ndarray,
                          batch_size: int = 64) -> np.ndarray:
    if x.ndim != 3 or x.shape[1:] != (10, 6) or not len(x) or batch_size < 1:
        raise ValueError("Expected nonempty (admissions, 10, 6) sequences.")
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        scores.append(observed_error(batch, model(batch), squared=False).cpu().numpy())
    scores = np.concatenate(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Reconstruction scores must be finite.")
    return scores


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.shape != scores.shape or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Validation labels and finite scores must align and include both classes.")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite validation threshold is available.")
    # roc_curve orders thresholds descending, so ties select the highest threshold.
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | np.ndarray) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / len(labels),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def operating_metrics(labels: np.ndarray, scores: np.ndarray,
                      threshold: float | np.ndarray) -> dict:
    """Evaluate scalar or per-admission thresholds, including alert burden."""
    boundary = np.asarray(threshold)
    if (labels.ndim != 1 or scores.shape != labels.shape
            or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all()
            or boundary.shape not in ((), labels.shape) or not np.isfinite(boundary).all()):
        raise ValueError("Binary labels, finite scores and scalar or aligned thresholds are required.")
    result = _metrics(labels, scores, threshold)
    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]
    result["negative_predictive_value"] = tn / (tn + fn) if tn + fn else None
    result["alert_count"] = tp + fp
    result["alerts_per_100"] = 100 * (tp + fp) / len(labels)
    result["undefined_metrics"] = {}
    if result["precision"] is None:
        result["undefined_metrics"]["precision"] = "No positive predictions."
    if result["negative_predictive_value"] is None:
        result["undefined_metrics"]["negative_predictive_value"] = "No negative predictions."
    return result


def evaluate_mortality(validation_y: np.ndarray, validation_scores: np.ndarray,
                       test_y: np.ndarray, test_scores: np.ndarray,
                       test_group: np.ndarray, *, seed: int = 42,
                       bootstrap_samples: int = 1000) -> dict:
    threshold = youden_threshold(validation_y, validation_scores)
    if (test_y.shape != test_scores.shape or test_y.shape != test_group.shape
            or set(np.unique(test_y)) != {0, 1} or not np.isfinite(test_scores).all()):
        raise ValueError("Test labels, scores and groups must align and include both classes.")
    if bootstrap_samples < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    metrics = _metrics(test_y, test_scores, threshold)
    _, inverse = np.unique(test_group, return_inverse=True)
    group_indices = [np.flatnonzero(inverse == group) for group in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("auroc", "average_precision", "sensitivity",
                                   "specificity", "precision", "accuracy")}
    invalid_class = 0
    for _ in range(bootstrap_samples):
        chosen = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[group] for group in chosen])
        labels = test_y[indices]
        if set(np.unique(labels)) != {0, 1}:
            invalid_class += 1
            continue
        replicate = _metrics(labels, test_scores[indices], threshold)
        for key in samples:
            if replicate[key] is not None:
                samples[key].append(replicate[key])
    intervals = {}
    for key, values in samples.items():
        if len(values) < 2:
            if key != "precision":
                raise ValueError(f"Too few valid patient-bootstrap replicates for {key}.")
            intervals[key] = {
                "lower": None, "upper": None, "valid_replicates": len(values),
                "reason": "Fewer than two replicates had positive predictions.",
            }
            continue
        intervals[key] = {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
            "valid_replicates": len(values),
        }
    return {
        "threshold": threshold, "threshold_source": "validation_youden",
        "threshold_comparison": "score >= threshold",
        "score": "mean_absolute_error_over_observed_biomarkers",
        "point_metrics": metrics, "confidence_intervals_95": intervals,
        "bootstrap": {
            "unit": "patient", "requested_replicates": bootstrap_samples,
            "single_class_replicates_skipped": invalid_class,
            "seed": seed, "conditioned_on_fitted_model_and_threshold": True,
        },
        "test_admissions": len(test_y), "test_patients": len(group_indices),
        "test_deaths": int(test_y.sum()), "test_prevalence": float(test_y.mean()),
    }
