"""Bounded nested development study retaining the article-style LSTM autoencoder."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from .mortality_comparison import fold_sequences, patient_folds
from .mortality_development import ROOT, load_development, validate_development
from .mortality_eval import reconstruction_scores
from .mortality_model import MortalityConfig, train_autoencoder
from .utils import save_json


@dataclass(frozen=True)
class AutoencoderCandidate:
    name: str
    transform: str = "identity"
    population: str = "all"
    epochs: int = 10

    def __post_init__(self):
        if not self.name or self.transform not in {"identity", "log1p"}:
            raise ValueError("A named candidate needs identity or log1p transformation.")
        if self.population not in {"all", "survivors"} or self.epochs < 1:
            raise ValueError("Candidate population must be all/survivors and epochs positive.")


CANDIDATES = (
    AutoencoderCandidate("baseline"),
    AutoencoderCandidate("log", transform="log1p"),
    AutoencoderCandidate("log_long", transform="log1p", epochs=30),
    AutoencoderCandidate("log_long_survivors", transform="log1p", population="survivors", epochs=30),
)


def transform_observations(observed: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return observed
    if transform != "log1p":
        raise ValueError("Unsupported biomarker transformation.")
    if np.any(observed < 0):
        raise ValueError("log1p candidates require nonnegative biomarkers; no clipping is performed.")
    return np.log1p(observed)


def candidate_scores(training: np.ndarray, labels: np.ndarray, validation: np.ndarray,
                     candidate: AutoencoderCandidate, *, seed: int,
                     device: str) -> np.ndarray:
    fitting = training if candidate.population == "all" else training[labels == 0]
    if not len(fitting):
        raise ValueError("Candidate fitting population is empty.")
    train_x, val_x = fold_sequences(
        transform_observations(fitting, candidate.transform),
        transform_observations(validation, candidate.transform),
    )
    config = MortalityConfig(seed=seed, epochs=candidate.epochs, device_preference=device)
    model, history = train_autoencoder(train_x, None, config)
    if len(history["train_loss"]) != candidate.epochs:
        raise RuntimeError("Candidate did not complete its prespecified fixed epochs.")
    return reconstruction_scores(model, val_x, config.batch_size)


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if (labels.ndim != 1 or scores.shape != labels.shape
            or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all()):
        raise ValueError("Finite aligned anomaly scores and both mortality outcomes are required.")
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def nested_autoencoder_study(arrays: dict[str, np.ndarray], *,
                             candidates: tuple[AutoencoderCandidate, ...] = CANDIDATES,
                             outer_splits: int = 5, inner_splits: int = 3,
                             seed: int = 42, device: str = "cpu") -> dict:
    validate_development(arrays)
    if len(candidates) < 2 or len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Provide at least two uniquely named candidates, baseline first.")
    baseline = candidates[0]
    if baseline.transform != "identity" or baseline.population != "all":
        raise ValueError("The fixed baseline must use identity transformation and all admissions.")
    actual_device = str(MortalityConfig(device_preference=device).device)
    if device != actual_device:
        print(f"{device} unavailable; using {actual_device}.")
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    for candidate in candidates:
        transform_observations(observed, candidate.transform)
    coverage = np.zeros(len(labels), dtype=int)
    records = []
    for fold, (training, validation) in enumerate(
            patient_folds(labels, groups, outer_splits, seed), 1):
        inner_folds = list(patient_folds(labels[training], groups[training],
                                         inner_splits, seed + fold))
        searches = []
        for candidate in candidates:
            metrics = []
            for inner, (fitting, held) in enumerate(inner_folds, 1):
                print(f"Outer {fold}, inner {inner}, candidate {candidate.name}")
                scores = candidate_scores(
                    observed[training[fitting]], labels[training[fitting]],
                    observed[training[held]], candidate,
                    seed=seed + 100 * fold + inner, device=actual_device,
                )
                metrics.append(ranking_metrics(labels[training[held]], scores))
            searches.append({
                "candidate": candidate.name, "inner_metrics": metrics,
                "mean_inner_average_precision": float(np.mean(
                    [item["average_precision"] for item in metrics])),
            })
        best_index = max(range(len(candidates)),
                         key=lambda index: searches[index]["mean_inner_average_precision"])
        selected = candidates[best_index]
        print(f"Outer {fold}: selected {selected.name}; external evaluation")
        selected_scores = candidate_scores(
            observed[training], labels[training], observed[validation], selected,
            seed=seed + fold, device=actual_device,
        )
        baseline_scores = selected_scores if best_index == 0 else candidate_scores(
            observed[training], labels[training], observed[validation], baseline,
            seed=seed + fold, device=actual_device,
        )
        coverage[validation] += 1
        records.append({
            "fold": fold, "fitting_admissions": len(training),
            "fitting_patients": len(np.unique(groups[training])),
            "validation_admissions": len(validation),
            "validation_patients": len(np.unique(groups[validation])),
            "validation_deaths": int(labels[validation].sum()),
            "validation_prevalence": float(labels[validation].mean()),
            "selected_candidate": selected.name, "inner_search": searches,
            "metrics": {
                "selected": ranking_metrics(labels[validation], selected_scores),
                "baseline": ranking_metrics(labels[validation], baseline_scores),
            },
        })
    if not np.all(coverage == 1):
        raise RuntimeError("Every development admission must be validated exactly once.")
    summary = {}
    for name in ("selected", "baseline"):
        summary[name] = {}
        for metric in ("average_precision", "auroc"):
            values = np.array([record["metrics"][name][metric] for record in records])
            summary[name][metric] = {
                "fold_mean": float(values.mean()),
                "fold_standard_deviation": float(values.std(ddof=1)),
            }
    return {
        "evaluation": "development_only_nested_patient_grouped_autoencoder_study",
        "test_accessed": False, "admissions": len(labels),
        "patients": len(np.unique(groups)), "deaths": int(labels.sum()),
        "prevalence": float(labels.mean()), "seed": seed,
        "outer_splits": outer_splits, "inner_splits": inner_splits,
        "device_requested": device, "device_used": actual_device,
        "torch_threads": torch.get_num_threads(),
        "candidates": [asdict(candidate) for candidate in candidates],
        "outer_oof_coverage": {"min": int(coverage.min()), "max": int(coverage.max())},
        "selection_counts": {
            candidate.name: sum(record["selected_candidate"] == candidate.name for record in records)
            for candidate in candidates
        },
        "folds": records, "summary": summary,
        "protocol": {
            "primary_metric": "mean_inner_fold_average_precision_against_mortality",
            "tie_break": "first_prespecified_candidate_in_grid_order",
            "architecture": "unchanged_6_to_64_to_32_to_32_to_64_to_3",
            "loss": "unchanged_observed_MSE_no_mortality_label_in_gradient",
            "score": "unchanged_observed_MAE_high_is_higher_risk_in_fitted_transformed_space",
            "optimizer": "Adam_lr_0.0001_batch_32_clip_norm_1",
            "preprocessing": "optional_log1p_then_past_fill_training_median_MinMax_with_original_masks",
            "preprocessing_fit": "each_inner_or_outer_fitting_population_only_all_or_survivors",
            "checkpoint": "final_prespecified_epoch_no_outer_checkpoint_selection",
            "outer_seed": "seed_plus_outer_fold_starting_at_1_matches_previous_baseline",
            "inner_split_seed": "seed_plus_outer_fold_starting_at_1",
            "inner_training_seed": "seed_plus_100_times_outer_fold_plus_inner_fold",
            "pooled_metrics": "not_computed_anomaly_score_scales_differ_across_folds_and_transforms",
            "uncertainty": "fold_standard_deviation_not_a_confidence_interval",
            "interpretation": "exploratory_internal_validation_not_independent_confirmation",
            "thresholds": "not_selected", "probability_calibration": "not_applicable_to_anomaly_scores",
            "final_model": "not_fitted_or_exported_review_study_before_freezing",
        },
    }


def plot_comparison(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, metric, title in zip(axes, ("average_precision", "auroc"), ("Average precision", "AUROC")):
        records = [report["summary"][name][metric] for name in ("baseline", "selected")]
        axis.bar(["Baseline", "Nested selected"], [item["fold_mean"] for item in records],
                 yerr=[item["fold_standard_deviation"] for item in records], capsize=5)
        axis.set(title=title, ylim=(0, 1))
    figure.suptitle("Development only: outer fold means +/- SD (not confidence intervals)")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run_autoencoder_study(input_file: Path, output_dir: Path, *,
                          candidates: tuple[AutoencoderCandidate, ...] = CANDIDATES,
                          outer_splits: int = 5, inner_splits: int = 3,
                          seed: int = 42, device: str = "cpu") -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output_dir.exists():
        raise FileExistsError("Autoencoder study output already exists; choose a new directory.")
    arrays = load_development(input_file)
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    log = output_dir / "training.log"
    with log.open("x") as stream, redirect_stdout(stream):
        report = nested_autoencoder_study(arrays, candidates=candidates, outer_splits=outer_splits,
                                          inner_splits=inner_splits, seed=seed, device=device)
    log.chmod(0o600)
    report["input_sha256"] = hashlib.sha256(Path(input_file).read_bytes()).hexdigest()
    report["versions"] = {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "scikit-learn", "torch", "matplotlib")
    }}
    report["implementation_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("mortality_autoencoder_study.py", "mortality_model.py",
                     "mortality_comparison.py", "mortality_development.py", "mortality_eval.py", "utils.py")
    }
    plot_comparison(report, output_dir / "comparison.png")
    save_json(output_dir / "study.json", report)
    for name in ("comparison.png", "study.json"):
        (output_dir / name).chmod(0o600)
    print(json.dumps(report["summary"], indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path,
                        default=ROOT / "outputs/mortality-development/development.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/mortality-development/autoencoder-study")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_autoencoder_study(args.input_file, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
