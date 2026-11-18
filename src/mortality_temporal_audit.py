"""Aggregate temporal-information audit of the development-only biomarker bins."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np

from .mortality_data import BIN_SECONDS, MARKERS, STEPS
from .mortality_development import ROOT, load_development, validate_development
from .utils import save_json


CHANGE_RTOL = 1e-5
CHANGE_ATOL = 1e-6


def describe(values: np.ndarray) -> dict:
    if not len(values):
        return {"n": 0, "quantiles": None, "reason": "No eligible observations for this summary."}
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1])
    return {
        "n": len(values),
        "quantiles": dict(zip(("min", "p25", "median", "p75", "p95", "max"),
                              quantiles.tolist())),
    }


def origin_counts(mask: np.ndarray) -> dict:
    past_observed = np.maximum.accumulate(mask, axis=1)
    observed = int(mask.sum())
    forward_fill = int((~mask & past_observed).sum())
    median = int((~past_observed).sum())
    return {
        "total_cells": mask.size, "observed_cells": observed,
        "forward_fill_cells": forward_fill, "initial_median_cells": median,
        "observed_fraction": observed / mask.size,
        "forward_fill_fraction": forward_fill / mask.size,
        "initial_median_fraction": median / mask.size,
    }


def marker_diagnostics(values: np.ndarray) -> tuple[dict, np.ndarray]:
    mask = np.isfinite(values)
    counts = mask.sum(axis=1)
    changing = np.zeros(len(values), dtype=bool)
    separations = np.zeros(STEPS - 1, dtype=int)
    for row, sequence in enumerate(values):
        positions = np.flatnonzero(mask[row])
        if len(positions) >= 2:
            measured = sequence[positions]
            changing[row] = not np.isclose(measured.min(), measured.max(),
                                           rtol=CHANGE_RTOL, atol=CHANGE_ATOL)
            separations += np.bincount(np.diff(positions), minlength=STEPS)[1:]
    adjacent = mask[:, :-1] & mask[:, 1:]
    step = np.arange(STEPS)[None, :]
    last_observed = np.maximum.accumulate(np.where(mask, step, -1), axis=1)
    carried = ~mask & (last_observed >= 0)
    carried_ages = (step - last_observed)[carried] * BIN_SECONDS / 3600
    pair_hours = np.repeat(np.arange(1, STEPS) * BIN_SECONDS / 3600, separations)
    return {
        "input_origins": origin_counts(mask),
        "observed_admissions_per_bin": mask.sum(axis=0).tolist(),
        "occupied_bin_count_histogram": np.bincount(counts, minlength=STEPS + 1).tolist(),
        "admissions_with_any_measurement": int((counts >= 1).sum()),
        "admissions_with_repeated_bins": int((counts >= 2).sum()),
        "admissions_with_at_least_three_bins": int((counts >= 3).sum()),
        "admissions_with_numerically_varying_measurements": int(changing.sum()),
        "admissions_with_constant_repeated_measurements": int(((counts >= 2) & ~changing).sum()),
        "adjacent_observed_bin_pairs": int(adjacent.sum()),
        "admissions_with_adjacent_observed_bins": int(adjacent.any(axis=1).sum()),
        "successive_occupied_bin_pairs": int(separations.sum()),
        "occupied_bin_separation_histogram": separations.tolist(),
        "occupied_bin_separation_hours": describe(pair_hours),
        "forward_fill_bin_age_hours": describe(carried_ages),
    }, changing


def summarize_population(observed: np.ndarray, groups: np.ndarray) -> dict:
    mask = np.isfinite(observed)
    count_per_marker = mask.sum(axis=1)
    markers = {}
    changing = []
    for index, name in enumerate(MARKERS):
        markers[name], flags = marker_diagnostics(observed[:, :, index])
        changing.append(flags)
    repeated_markers = (count_per_marker >= 2).sum(axis=1)
    changing_markers = np.stack(changing, axis=1).sum(axis=1)
    occupied_bins = mask.any(axis=2).sum(axis=1)
    return {
        "admissions": len(observed), "patients": len(np.unique(groups)),
        "input_origins": origin_counts(mask),
        "observed_cells_per_admission": describe(mask.sum(axis=(1, 2))),
        "occupied_time_bins_per_admission_histogram": np.bincount(
            occupied_bins, minlength=STEPS + 1).tolist(),
        "admissions_with_no_marker_repeated": int((repeated_markers == 0).sum()),
        "admissions_with_repeated_markers_at_least": {
            str(count): int((repeated_markers >= count).sum()) for count in (1, 2, 3)
        },
        "admissions_with_varying_markers_at_least": {
            str(count): int((changing_markers >= count).sum()) for count in (1, 2, 3)
        },
        "admissions_with_adjacent_observations_of_any_marker": int(
            (mask[:, :-1] & mask[:, 1:]).any(axis=(1, 2)).sum()),
        "markers": markers,
    }


def temporal_audit(arrays: dict[str, np.ndarray]) -> dict:
    validate_development(arrays)
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    return {
        "analysis": "development_only_descriptive_temporal_information_audit",
        "test_accessed": False, "training_performed": False,
        "admissions": len(labels), "patients": len(np.unique(groups)),
        "deaths": int(labels.sum()), "steps": STEPS, "bin_seconds": BIN_SECONDS,
        "populations": {
            "overall": summarize_population(observed, groups),
            "survival_admissions": summarize_population(observed[labels == 0], groups[labels == 0]),
            "death_admissions": summarize_population(observed[labels == 1], groups[labels == 1]),
        },
        "definitions": {
            "observed": "finite_originally_observed_last_valid_value_in_bin_not_imputed",
            "repeated": "at_least_two_occupied_bins_for_the_same_admission_and_marker",
            "varying": "min_and_max_not_numpy_isclose_rtol_1e-5_atol_1e-6",
            "variation_rtol": CHANGE_RTOL, "variation_atol": CHANGE_ATOL,
            "variation_meaning": "numerical_not_a_clinical_change_threshold",
            "occupied_bin_count_histogram": "indices_0_through_10_bins",
            "occupied_bin_separation_histogram": "indices_0_through_8_mean_1_through_9_bin_steps",
            "separation_hours": "bin_index_distance_times_2.4_not_exact_event_time_intervals",
            "adjacent": "both_neighboring_bins_observed_for_the_same_marker",
            "forward_fill_bin_age_hours": "missing_bin_distance_from_last_occupied_bin_not_exact_sample_age",
            "input_origins": "structural_counts_under_original_forward_fill_then_train_median_policy",
            "medians": "no_imputation_or_scaler_fitted_in_this_audit",
            "source_precision": "physical_values_recovered_from_original_float32_scaled_sequences",
            "population_weighting": "admission_level_patients_can_appear_in_both_outcome_strata",
            "limitations": [
                "multiple_events_within_a_bin_are_collapsed_and_cannot_be_counted_from_this_file",
                "exact_event_timestamps_and_sampling_cadence_are_not_available",
                "repeated_or_varying_measurements_do_not_prove_predictive_temporal_signal",
                "missingness_may_reflect_care_process_not_only_physiology",
                "outcome_strata_are_descriptive_no_model_selection_or_causal_inference",
            ],
        },
    }


def plot_temporal_coverage(report: dict, output: Path) -> None:
    overall = report["populations"]["overall"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    coverage = np.array([overall["markers"][name]["observed_admissions_per_bin"] for name in MARKERS])
    image = axes[0].imshow(100 * coverage / overall["admissions"], vmin=0, vmax=100, aspect="auto")
    axes[0].set(xticks=range(STEPS), xticklabels=range(1, STEPS + 1),
                yticks=range(3), yticklabels=MARKERS, xlabel="2.4-hour bin",
                title="Admissions observed per bin (%)")
    figure.colorbar(image, ax=axes[0])
    for name in MARKERS:
        counts = overall["markers"][name]["occupied_bin_count_histogram"]
        axes[1].plot(range(STEPS + 1), 100 * np.array(counts) / overall["admissions"], "o-", label=name)
    axes[1].set(xlabel="Occupied bins per admission", ylabel="Admissions (%)",
                xticks=range(STEPS + 1), title="Real temporal repetitions")
    axes[1].legend(fontsize="small")
    bottom = np.zeros(3)
    for key, label in (("observed_fraction", "Observed"), ("forward_fill_fraction", "Forward-filled"),
                       ("initial_median_fraction", "Initial median")):
        values = 100 * np.array([overall["markers"][name]["input_origins"][key] for name in MARKERS])
        axes[2].bar(MARKERS, values, bottom=bottom, label=label)
        bottom += values
    axes[2].set(ylabel="Input cells (%)", ylim=(0, 100), title="Input origin, not new measurements")
    axes[2].tick_params(axis="x", labelrotation=15)
    axes[2].legend(fontsize="small")
    figure.suptitle("Development only: temporal coverage audit")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run_temporal_audit(input_file: Path, output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output_dir.exists():
        raise FileExistsError("Temporal audit output already exists; choose a new directory.")
    report = temporal_audit(load_development(input_file))
    report["input_sha256"] = hashlib.sha256(Path(input_file).read_bytes()).hexdigest()
    report["versions"] = {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "matplotlib")
    }}
    report["implementation_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("mortality_temporal_audit.py", "mortality_development.py", "mortality_data.py")
    }
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    plot_temporal_coverage(report, output_dir / "temporal_coverage.png")
    save_json(output_dir / "temporal_audit.json", report)
    for name in ("temporal_coverage.png", "temporal_audit.json"):
        (output_dir / name).chmod(0o600)
    print(json.dumps(report["populations"]["overall"], indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path,
                        default=ROOT / "outputs/mortality-development/development.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/mortality-development/temporal-audit")
    args = parser.parse_args()
    run_temporal_audit(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
