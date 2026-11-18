"""Nested patient-grouped logistic tuning and development calibration assessment."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .mortality_comparison import biomarker_features, make_logistic_classifier, patient_folds
from .mortality_development import ROOT, load_development, validate_development
from .utils import save_json


C_VALUES = (0.01, 0.1, 1.0, 10.0)
METRICS = ("average_precision", "auroc", "brier_score")


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if (labels.ndim != 1 or probabilities.shape != labels.shape
            or set(np.unique(labels)) != {0, 1} or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Expected aligned binary outcomes and finite probabilities in [0, 1].")
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 5) -> list[dict]:
    probability_metrics(labels, probabilities)
    if n_bins < 2:
        raise ValueError("Calibration requires at least two requested bins.")
    edges = np.quantile(probabilities, np.linspace(0, 1, n_bins + 1))
    assignments = np.searchsorted(edges[1:-1], probabilities)
    bins = []
    for index in range(n_bins):
        selected = assignments == index
        if selected.any():
            bins.append({
                "bin": index + 1, "admissions": int(selected.sum()),
                "deaths": int(labels[selected].sum()),
                "mean_predicted_probability": float(probabilities[selected].mean()),
                "observed_mortality_fraction": float(labels[selected].mean()),
            })
    return bins


def select_regularization(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                          *, n_splits: int = 3, seed: int = 42) -> tuple[float, list[dict]]:
    folds = list(patient_folds(labels, groups, n_splits, seed))
    candidates = []
    for c in C_VALUES:
        scores = []
        for training, validation in folds:
            if set(labels[training]) != {0, 1} or set(labels[validation]) != {0, 1}:
                raise ValueError("Every inner fold must contain both admission outcomes.")
            model = make_logistic_classifier(c, seed)
            model.fit(features[training], labels[training])
            probabilities = model.predict_proba(features[validation])[:, 1]
            scores.append(probability_metrics(labels[validation], probabilities)["average_precision"])
        candidates.append({
            "c": c, "inner_fold_average_precision": scores,
            "mean_inner_average_precision": float(np.mean(scores)),
        })
    # Candidates are ordered from strongest to weakest regularization; first wins ties.
    best = max(candidates, key=lambda candidate: candidate["mean_inner_average_precision"])
    return best["c"], candidates


def nested_logistic(arrays: dict[str, np.ndarray], *, outer_splits: int = 5,
                    inner_splits: int = 3, seed: int = 42) -> dict:
    validate_development(arrays)
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    features = biomarker_features(observed)
    folds = list(patient_folds(labels, groups, outer_splits, seed))
    predictions = {
        name: np.full(len(labels), np.nan) for name in ("tuned", "fixed_c1", "prevalence")
    }
    coverage = np.zeros(len(labels), dtype=int)
    records = []
    for fold, (training, validation) in enumerate(folds, 1):
        if set(labels[training]) != {0, 1} or set(labels[validation]) != {0, 1}:
            raise ValueError("Every outer fold must contain both admission outcomes.")
        selected_c, candidates = select_regularization(
            features[training], labels[training], groups[training],
            n_splits=inner_splits, seed=seed + fold,
        )
        metrics = {}
        for name, c in (("tuned", selected_c), ("fixed_c1", 1.0)):
            model = make_logistic_classifier(c, seed)
            model.fit(features[training], labels[training])
            probabilities = model.predict_proba(features[validation])[:, 1]
            predictions[name][validation] = probabilities
            metrics[name] = probability_metrics(labels[validation], probabilities)
        prevalence = float(labels[training].mean())
        predictions["prevalence"][validation] = prevalence
        metrics["prevalence"] = probability_metrics(labels[validation],
                                                   predictions["prevalence"][validation])
        coverage[validation] += 1
        records.append({
            "fold": fold, "fitting_admissions": len(training),
            "fitting_patients": len(np.unique(groups[training])),
            "validation_admissions": len(validation),
            "validation_patients": len(np.unique(groups[validation])),
            "validation_deaths": int(labels[validation].sum()),
            "fitting_prevalence": prevalence,
            "selected_c": selected_c, "inner_search": candidates, "metrics": metrics,
        })
        print(f"Outer fold {fold}/{outer_splits}: C={selected_c:g}, "
              f"AP={metrics['tuned']['average_precision']:.6f}, "
              f"Brier={metrics['tuned']['brier_score']:.6f}")
    if not np.all(coverage == 1):
        raise RuntimeError("Every development admission must have exactly one outer-fold prediction.")
    summary = {}
    for name, probabilities in predictions.items():
        fold_summary = {}
        for metric in METRICS:
            values = np.array([record["metrics"][name][metric] for record in records])
            fold_summary[metric] = {
                "fold_mean": float(values.mean()),
                "fold_standard_deviation": float(values.std(ddof=1)),
            }
        summary[name] = {
            "fold_metrics": fold_summary,
            "pooled_oof_metrics": probability_metrics(labels, probabilities),
            "calibration_bins": calibration_bins(labels, probabilities),
            "mean_predicted_probability": float(probabilities.mean()),
        }
    return {
        "evaluation": "development_only_nested_patient_stratified_cross_validation",
        "test_accessed": False, "admissions": len(labels),
        "patients": len(np.unique(groups)), "deaths": int(labels.sum()),
        "prevalence": float(labels.mean()), "outer_splits": outer_splits,
        "inner_splits": inner_splits, "seed": seed, "c_candidates": list(C_VALUES),
        "outer_oof_coverage": {"min": int(coverage.min()), "max": int(coverage.max())},
        "folds": records, "summary": summary,
        "protocol": {
            "primary_metric": "mean_inner_fold_average_precision",
            "tie_break": "smallest_C_strongest_regularization",
            "features": "same_15_observed_biomarker_summaries_no_new_predictors",
            "preprocessing": "imputation_indicators_and_scaling_refitted_in_every_inner_and_outer_fit",
            "outer_seed": seed, "inner_seed": "seed_plus_outer_fold_number_starting_at_1",
            "calibration": "assessment_only_no_recalibration",
            "calibration_bins": "five_quantile_bins_on_pooled_outer_oof_probabilities_skip_empty",
            "prevalence_baseline": "constant_per_outer_fold_fitted_on_outer_training_labels_only",
            "pooled_metrics": "descriptive_mixture_of_outer_fold_models_not_a_single_final_model",
            "uncertainty": "fold_standard_deviation_not_a_confidence_interval",
            "interpretation": "exploratory_development_after_model_family_selection_on_this_pool",
            "thresholds": "not_selected", "final_model": "not_fitted_or_exported",
        },
    }


def plot_calibration(report: dict, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for name, label in (("tuned", "Nested tuned"), ("fixed_c1", "Fixed C=1"),
                        ("prevalence", "Training prevalence")):
        bins = report["summary"][name]["calibration_bins"]
        axis.plot([item["mean_predicted_probability"] for item in bins],
                  [item["observed_mortality_fraction"] for item in bins],
                  "o-", label=label)
    axis.set(xlabel="Mean predicted mortality probability",
             ylabel="Observed mortality fraction", xlim=(0, 1), ylim=(0, 1),
             title="Development only: outer out-of-fold calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run_nested_logistic(input_file: Path, output_dir: Path, *, outer_splits: int = 5,
                        inner_splits: int = 3, seed: int = 42) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output_dir.exists():
        raise FileExistsError("Nested logistic output already exists; choose a new directory.")
    arrays = load_development(input_file)
    report = nested_logistic(arrays, outer_splits=outer_splits, inner_splits=inner_splits, seed=seed)
    report["input_sha256"] = hashlib.sha256(Path(input_file).read_bytes()).hexdigest()
    report["versions"] = {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "scikit-learn", "matplotlib")
    }}
    report["implementation_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("mortality_logistic.py", "mortality_comparison.py", "mortality_development.py")
    }
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    plot_calibration(report, output_dir / "calibration.png")
    save_json(output_dir / "nested_logistic.json", report)
    for name in ("nested_logistic.json", "calibration.png"):
        (output_dir / name).chmod(0o600)
    print(json.dumps({name: result["fold_metrics"] for name, result in report["summary"].items()},
                     indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path,
                        default=ROOT / "outputs/mortality-development/development.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/mortality-development/nested-logistic")
    args = parser.parse_args()
    run_nested_logistic(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
