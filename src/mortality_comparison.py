"""Patient-grouped development comparison; no held-out test data accepted."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .mortality_development import ROOT, load_development, validate_development
from .mortality_eval import reconstruction_scores
from .mortality_model import MortalityConfig, train_autoencoder
from .utils import save_json


def patient_folds(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    unique, inverse = np.unique(groups, return_inverse=True)
    patient_labels = np.array([labels[inverse == i].max() for i in range(len(unique))])
    if n_splits < 2 or any((patient_labels == label).sum() < n_splits for label in (0, 1)):
        raise ValueError("Too few patients in each outcome stratum for the requested folds.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for training, validation in splitter.split(unique, patient_labels):
        yield (np.flatnonzero(np.isin(inverse, training)),
               np.flatnonzero(np.isin(inverse, validation)))


def complete_sequences(observed: np.ndarray, medians: np.ndarray) -> np.ndarray:
    filled = observed.copy()
    for step in range(1, 10):
        filled[:, step] = np.where(np.isfinite(filled[:, step]),
                                   filled[:, step], filled[:, step - 1])
    return np.where(np.isfinite(filled), filled, medians)


def fold_sequences(training: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (training.ndim != 3 or training.shape[1] != 10 or training.shape[2] == 0
            or validation.ndim != 3 or validation.shape[1:] != training.shape[1:]):
        raise ValueError("Expected matching ten-bin biomarker arrays.")
    markers = training.shape[-1]
    if not np.isfinite(training).any(axis=(0, 1)).all():
        raise ValueError("Each biomarker must be observed in the fold's fitting population.")
    medians = np.nanmedian(training, axis=(0, 1))
    filled_train = complete_sequences(training, medians)
    scaler = MinMaxScaler().fit(filled_train.reshape(-1, markers))
    result = []
    for raw, filled in ((training, filled_train),
                        (validation, complete_sequences(validation, medians))):
        values = scaler.transform(filled.reshape(-1, markers)).reshape(filled.shape)
        sequences = np.concatenate((values, np.isfinite(raw)), axis=2).astype(np.float32)
        if not np.isfinite(sequences).all():
            raise ValueError("Fold preprocessing produced nonfinite values.")
        result.append(sequences)
    return result[0], result[1]


def biomarker_features(observed: np.ndarray) -> np.ndarray:
    """Last, minimum, maximum, slope/hour and occupied-bin count per marker."""
    features = np.full((len(observed), 15), np.nan, dtype=float)
    for admission, sequence in enumerate(observed):
        for marker in range(3):
            times = np.flatnonzero(np.isfinite(sequence[:, marker]))
            offset = marker * 5
            features[admission, offset + 4] = len(times)
            if not len(times):
                continue
            values = sequence[times, marker]
            features[admission, offset:offset + 3] = [values[-1], values.min(), values.max()]
            if len(times) >= 2:
                hours = times * 2.4
                centered = hours - hours.mean()
                features[admission, offset + 3] = (
                    np.dot(centered, values - values.mean()) / np.dot(centered, centered)
                )
    return features


def make_logistic_classifier(c: float = 1.0, seed: int = 42) -> Pipeline:
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Logistic regularization C must be finite and positive.")
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=seed),
    )


def compare_development(arrays: dict[str, np.ndarray], config: MortalityConfig,
                        *, n_splits: int = 5) -> dict:
    validate_development(arrays)
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    folds = list(patient_folds(labels, groups, n_splits, config.seed))
    features = biomarker_features(observed)
    results = {name: [] for name in ("logistic_regression", "autoencoder_all",
                                     "autoencoder_survivors")}
    for fold, (training, validation) in enumerate(folds, 1):
        if set(labels[training]) != {0, 1} or set(labels[validation]) != {0, 1}:
            raise ValueError("Every fold must have both admission outcomes.")
        print(f"Development fold {fold}/{n_splits}")
        classifier = make_logistic_classifier(seed=config.seed)
        classifier.fit(features[training], labels[training])
        scores = {"logistic_regression": classifier.predict_proba(features[validation])[:, 1]}
        fitting_counts = {"logistic_regression": len(training)}
        for name in ("autoencoder_all", "autoencoder_survivors"):
            fitting = training if name == "autoencoder_all" else training[labels[training] == 0]
            train_x, val_x = fold_sequences(observed[fitting], observed[validation])
            # Fixed epochs: the outer fold is never used for checkpoint selection.
            model, _ = train_autoencoder(train_x, None, replace(config, seed=config.seed + fold))
            scores[name] = reconstruction_scores(model, val_x, config.batch_size)
            fitting_counts[name] = len(fitting)
        for name, predicted in scores.items():
            if not np.isfinite(predicted).all():
                raise ValueError(f"{name} produced nonfinite development scores.")
            results[name].append({
                "fold": fold, "fitting_admissions": fitting_counts[name],
                "validation_admissions": len(validation),
                "validation_patients": len(np.unique(groups[validation])),
                "validation_deaths": int(labels[validation].sum()),
                "validation_prevalence": float(labels[validation].mean()),
                "auroc": float(roc_auc_score(labels[validation], predicted)),
                "average_precision": float(average_precision_score(labels[validation], predicted)),
            })
    summary = {}
    for name, records in results.items():
        summary[name] = {}
        for metric in ("auroc", "average_precision"):
            values = np.array([record[metric] for record in records])
            summary[name][metric] = {
                "fold_mean": float(values.mean()), "fold_standard_deviation": float(values.std(ddof=1)),
            }
    return {
        "evaluation": "development_only_patient_stratified_cross_validation",
        "test_accessed": False, "primary_metric": "average_precision",
        "admissions": len(labels), "patients": len(np.unique(groups)),
        "deaths": int(labels.sum()), "prevalence": float(labels.mean()),
        "n_splits": n_splits, "config": asdict(config), "folds": results, "summary": summary,
        "protocol": {
            "input": "original_train_plus_validation_only",
            "preprocessing": "refit_imputation_and_scaling_within_each_fitting_population",
            "autoencoder_checkpoint": "final_fixed_epoch_no_outer_validation_selection",
            "autoencoder_score": "observed_only_MAE_high_is_higher_risk",
            "logistic": "15_binned_summary_features_median_indicators_standard_scaler_L2_C1",
            "feature_empty_in_fit": "impute_zero_and_include_missingness_indicator",
            "source_precision": "observed_bins_recovered_from_float32_sequences_by_inverse_scaling",
            "thresholds": "not_selected",
            "uncertainty": "fold_standard_deviation_is_descriptive_not_a_confidence_interval",
            "interpretation": "exploratory_development_comparison_not_independent_confirmation",
        },
    }


def run_comparison(input_file: Path, output_dir: Path, config: MortalityConfig,
                   *, n_splits: int = 5) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output_dir.exists():
        raise FileExistsError("Comparison output already exists; choose a new directory.")
    arrays = load_development(input_file)
    report = compare_development(arrays, config, n_splits=n_splits)
    report["input_sha256"] = hashlib.sha256(Path(input_file).read_bytes()).hexdigest()
    report["versions"] = {
        name: version(name) for name in ("numpy", "scikit-learn", "torch")
    }
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    save_json(output_dir / "comparison.json", report)
    (output_dir / "comparison.json").chmod(0o600)
    print(json.dumps(report["summary"], indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path,
                        default=ROOT / "outputs/mortality-development/development.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/mortality-development/comparison")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run_comparison(args.input_file, args.output_dir,
                   MortalityConfig(epochs=args.epochs, device_preference=args.device),
                   n_splits=args.folds)


if __name__ == "__main__":
    main()
