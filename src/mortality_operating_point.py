"""Development-only Youden evaluation and frozen experimental logistic pipeline."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from .mortality_comparison import biomarker_features, make_logistic_classifier, patient_folds
from .mortality_data import BIN_SECONDS, MARKERS, STEPS
from .mortality_development import ROOT, load_development, validate_development
from .mortality_eval import operating_metrics, youden_threshold
from .mortality_logistic import probability_metrics
from .utils import save_json


RATE_METRICS = (
    "sensitivity", "specificity", "precision", "negative_predictive_value",
    "alerts_per_100", "accuracy", "average_precision", "auroc", "brier_score",
)


def cross_fitted_probabilities(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                               *, n_splits: int, seed: int) -> np.ndarray:
    probabilities = np.full(len(labels), np.nan)
    coverage = np.zeros(len(labels), dtype=int)
    for training, validation in patient_folds(labels, groups, n_splits, seed):
        if set(labels[training]) != {0, 1} or set(labels[validation]) != {0, 1}:
            raise ValueError("Every threshold-selection fold must contain both outcomes.")
        model = make_logistic_classifier(seed=seed)
        model.fit(features[training], labels[training])
        probabilities[validation] = model.predict_proba(features[validation])[:, 1]
        coverage[validation] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Threshold-selection predictions must cover each admission exactly once.")
    probability_metrics(labels, probabilities)
    return probabilities


def evaluate_youden(arrays: dict[str, np.ndarray], *, outer_splits: int = 5,
                    inner_splits: int = 3, seed: int = 42) -> tuple[dict, float]:
    validate_development(arrays)
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    features = biomarker_features(observed)
    probabilities = np.full(len(labels), np.nan)
    boundaries = np.full(len(labels), np.nan)
    coverage = np.zeros(len(labels), dtype=int)
    records = []
    for fold, (training, validation) in enumerate(
            patient_folds(labels, groups, outer_splits, seed), 1):
        inner_probabilities = cross_fitted_probabilities(
            features[training], labels[training], groups[training],
            n_splits=inner_splits, seed=seed + fold,
        )
        threshold = youden_threshold(labels[training], inner_probabilities)
        model = make_logistic_classifier(seed=seed)
        model.fit(features[training], labels[training])
        predicted = model.predict_proba(features[validation])[:, 1]
        probabilities[validation], boundaries[validation] = predicted, threshold
        coverage[validation] += 1
        metrics = operating_metrics(labels[validation], predicted, threshold)
        metrics["brier_score"] = probability_metrics(labels[validation], predicted)["brier_score"]
        records.append({
            "fold": fold, "fitting_admissions": len(training),
            "fitting_patients": len(np.unique(groups[training])),
            "validation_admissions": len(validation),
            "validation_patients": len(np.unique(groups[validation])),
            "validation_deaths": int(labels[validation].sum()),
            "threshold": threshold, "metrics": metrics,
        })
        print(f"Outer fold {fold}/{outer_splits}: threshold={threshold:.6f}, "
              f"sensitivity={metrics['sensitivity']:.4f}, "
              f"alerts/100={metrics['alerts_per_100']:.2f}")
    if not np.all(coverage == 1):
        raise RuntimeError("Every admission must have exactly one outer validation prediction.")
    pooled = operating_metrics(labels, probabilities, boundaries)
    pooled["brier_score"] = probability_metrics(labels, probabilities)["brier_score"]
    summary = {}
    for key in RATE_METRICS:
        values = np.array([record["metrics"][key] for record in records
                           if record["metrics"][key] is not None])
        summary[key] = {
            "fold_mean": float(values.mean()) if len(values) else None,
            "fold_standard_deviation": float(values.std(ddof=1)) if len(values) >= 2 else None,
            "valid_folds": len(values),
        }
    final_threshold = youden_threshold(labels, probabilities)
    report = {
        "evaluation": "development_only_nested_patient_grouped_youden_fixed_C1",
        "test_accessed": False, "admissions": len(labels),
        "patients": len(np.unique(groups)), "deaths": int(labels.sum()),
        "prevalence": float(labels.mean()), "c": 1.0, "seed": seed,
        "outer_splits": outer_splits, "inner_splits": inner_splits,
        "outer_oof_coverage": {"min": int(coverage.min()), "max": int(coverage.max())},
        "folds": records, "fold_summary": summary, "pooled_outer_metrics": pooled,
        "frozen_threshold": final_threshold,
        "protocol": {
            "outer_threshold": "Youden_on_inner_patient_grouped_OOF_within_outer_training_only",
            "preprocessing": "refit_imputation_indicators_and_scaling_in_every_fit",
            "inner_seed": "seed_plus_outer_fold_number_starting_at_1",
            "comparison": "probability_greater_than_or_equal_to_threshold",
            "tie_break": "highest_finite_threshold",
            "frozen_threshold": "Youden_on_pooled_full_development_outer_OOF_probabilities",
            "pooled_outer_metrics": "fold_specific_inner_selected_thresholds_not_frozen_threshold",
            "frozen_model_performance": "not_estimated_no_independent_validation",
            "calibration": "none_added",
            "uncertainty": "fold_SD_not_a_confidence_interval_undefined_rates_explicit",
            "interpretation": "research_only_exploratory_after_development_model_selection",
        },
    }
    return report, final_threshold


def predict_observed(model: Pipeline, observed: np.ndarray,
                     threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Score already-binned physical-unit observations; NaN denotes unmeasured bins."""
    if (observed.ndim != 3 or observed.shape[1:] != (STEPS, len(MARKERS))
            or not len(observed) or not np.issubdtype(observed.dtype, np.floating)
            or np.isinf(observed).any() or not np.isfinite(observed).any(axis=(1, 2)).all()
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Expected nonempty (admissions, 10, 3) physical values and a threshold in [0, 1].")
    probabilities = model.predict_proba(biomarker_features(observed))[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("Frozen pipeline returned invalid probabilities.")
    return probabilities, probabilities >= threshold


def freeze_logistic(arrays: dict[str, np.ndarray], threshold: float,
                    seed: int = 42) -> tuple[Pipeline, dict]:
    validate_development(arrays)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Frozen probability threshold must be in [0, 1].")
    model = make_logistic_classifier(seed=seed)
    model.fit(biomarker_features(arrays["observed"]), arrays["labels"])
    classifier = model.named_steps["logisticregression"]
    names = [f"{marker}_{statistic}" for marker in MARKERS for statistic in
             ("last", "minimum", "maximum", "slope_per_hour", "occupied_bin_count")]
    transformed_names = model[:-1].get_feature_names_out(names).tolist()
    spec = {
        "schema_version": 1, "research_only": True, "independently_validated": False,
        "endpoint": "same_hospitalization_mortality_at_24_hour_landmark",
        "c": 1.0, "seed": seed, "threshold": threshold,
        "threshold_source": "Youden_on_full_development_grouped_OOF_probabilities",
        "threshold_comparison": "probability >= threshold",
        "threshold_transfer": "cross_fitted_threshold_transferred_to_full_development_fit_unvalidated",
        "input": {
            "shape": ["admissions", STEPS, len(MARKERS)], "marker_order": list(MARKERS),
            "units": ["mmol/L", "\u03bcmol/L", "\u03bcmol/L"], "bin_seconds": BIN_SECONDS,
            "missing": "NaN_for_originally_unobserved_bins_no_forward_fill_or_scaling",
            "aggregation": "last_valid_value_per_bin_within_first_24_hours",
        },
        "features": names, "transformed_features": transformed_names,
        "coefficients_after_imputation_and_standardization": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(), "classes": classifier.classes_.tolist(),
        "classifier_parameters": classifier.get_params(),
        "training_admissions": len(arrays["labels"]),
        "calibration": "no_added_probability_calibration",
        "performance": "outer_validation_report_is_not_performance_of_this_final_fitted_artifact",
        "serialization": "joblib_load_only_trusted_artifacts_with_compatible_dependency_versions",
    }
    return model, spec


def run_operating_point(input_file: Path, output_dir: Path, *, outer_splits: int = 5,
                        inner_splits: int = 3, seed: int = 42) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output_dir.exists():
        raise FileExistsError("Frozen logistic output already exists; choose a new directory.")
    arrays = load_development(input_file)
    report, threshold = evaluate_youden(arrays, outer_splits=outer_splits,
                                       inner_splits=inner_splits, seed=seed)
    model, spec = freeze_logistic(arrays, threshold, seed)
    provenance = {
        "input_sha256": hashlib.sha256(Path(input_file).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scikit-learn", "joblib")
        }},
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("mortality_operating_point.py", "mortality_comparison.py",
                         "mortality_development.py", "mortality_eval.py", "mortality_logistic.py")
        },
    }
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    model_file = output_dir / "model.joblib"
    joblib.dump(model, model_file)
    model_file.chmod(0o600)
    reloaded = joblib.load(model_file)
    # Verify persistence on synthetic input, never rescore real data with the full fit.
    example = np.ones((1, STEPS, len(MARKERS)), dtype=float)
    before = predict_observed(model, example, threshold)
    after = predict_observed(reloaded, example, threshold)
    if not all(np.array_equal(left, right) for left, right in zip(before, after)):
        raise RuntimeError("Reloaded logistic pipeline does not reproduce the original predictions.")
    spec["model_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
    spec.update(provenance)
    report.update(provenance)
    save_json(output_dir / "model_spec.json", spec)
    save_json(output_dir / "evaluation.json", report)
    for name in ("model_spec.json", "evaluation.json"):
        (output_dir / name).chmod(0o600)
    print(json.dumps(report["pooled_outer_metrics"], indent=2))
    print("Frozen research pipeline saved; no independent performance claim for the final fit.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path,
                        default=ROOT / "outputs/mortality-development/development.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/mortality-development/frozen-logistic")
    args = parser.parse_args()
    run_operating_point(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
