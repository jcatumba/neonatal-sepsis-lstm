"""Compare fixed mortality autoencoder panels on the original development pool."""

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .mortality_autoencoder_study import candidate_scores, AutoencoderCandidate, ranking_metrics
from .mortality_comparison import patient_folds
from .mortality_development import ROOT, validate_development
from .utils import save_json


EXTRA_MARKERS = ("glucose_gas", "ph_gas", "platelets", "wbc")
PANELS = {
    "baseline": (0, 1, 2),
    "glucose_ph": (0, 1, 2, 3, 4),
    "hematology": (0, 1, 2, 5, 6),
    "combined": (0, 1, 2, 3, 4, 5, 6),
}
KEYS = {"observed", "labels", "groups"}
OUTPUT_ROOT = ROOT / "outputs/mortality-development"


def validate_panel_input(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != KEYS:
        raise ValueError("Only the development panel export is accepted; no test keys.")
    values = arrays["observed"]
    if (values.ndim != 3 or values.shape[1:] != (10, 7)
            or not np.issubdtype(values.dtype, np.floating) or np.isinf(values).any()):
        raise ValueError("Expected seven aligned observed channels with NaN gaps.")
    validate_development({**arrays, "observed": values[:, :, :3]})


def load_panel_input(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != KEYS:
            raise ValueError("Only the development panel export is accepted; no test keys.")
        arrays = {key: archive[key] for key in KEYS}
    validate_panel_input(arrays)
    return arrays


def timing_sensitivity(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    early = observed.copy()
    early[:, -1, :] = np.nan
    # Keep the same eligibility for all panels, independent of extra measurements.
    return early, np.isfinite(early[:, :, :3]).any(axis=(1, 2))


def compare_panels(arrays: dict[str, np.ndarray], *, epochs: int = 10, n_splits: int = 5) -> dict:
    validate_panel_input(arrays)
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    early, eligible = timing_sensitivity(observed)
    folds = list(patient_folds(labels, groups, n_splits, 42))
    results = {}
    for window, data, keep in (
        ("main_24h", observed, np.ones(len(labels), dtype=bool)),
        ("sensitivity_21_6h", early, eligible),
    ):
        records = []
        coverage = np.zeros(len(labels), dtype=int)
        for fold, (training, held) in enumerate(folds, 1):
            training, held = training[keep[training]], held[keep[held]]
            if (np.intersect1d(groups[training], groups[held]).size
                    or set(labels[training]) != {0, 1} or set(labels[held]) != {0, 1}):
                raise ValueError("Grouped folds must be patient-disjoint with both outcomes.")
            metrics = {}
            for name, channels in PANELS.items():
                print(f"{window}: fold {fold}/{n_splits}, {name}", flush=True)
                panel = data[:, :, channels]
                scores = candidate_scores(
                    panel[training], labels[training], panel[held],
                    AutoencoderCandidate(name, epochs=epochs), seed=42 + fold, device="cpu",
                )
                metrics[name] = ranking_metrics(labels[held], scores)
            coverage[held] += 1
            records.append({
                "fold": fold, "fit_admissions": len(training), "held_admissions": len(held),
                "held_deaths": int(labels[held].sum()), "metrics": metrics,
            })
        if not np.array_equal(coverage, keep.astype(int)):
            raise RuntimeError("Each eligible admission must be evaluated exactly once.")
        summary = {}
        for name in PANELS:
            summary[name] = {}
            for metric in ("average_precision", "auroc"):
                values = np.array([record["metrics"][name][metric] for record in records])
                reference = np.array([record["metrics"]["baseline"][metric] for record in records])
                summary[name][metric] = {
                    "fold_mean": float(values.mean()),
                    "fold_sd": float(values.std(ddof=1)),
                    "mean_paired_delta_vs_baseline": float((values - reference).mean()),
                    "folds_above_baseline": int((values > reference).sum()),
                }
        results[window] = {
            "admissions": int(keep.sum()), "patients": int(np.unique(groups[keep]).size),
            "deaths": int(labels[keep].sum()), "excluded_admissions": int((~keep).sum()),
            "excluded_deaths": int(labels[~keep].sum()), "folds": records, "summary": summary,
        }
    return {
        "test_accessed": False, "panels": PANELS, "results": results, "epochs": epochs,
        "protocol": {
            "endpoint": "same-hospitalization mortality at the 24h landmark",
            "input": "original development patients only; laboratory event time proxy",
            "folds": "fixed patient-stratified folds, seed 42, shared by every panel",
            "preprocessing": "past-only fill; fit medians and MinMax in each fitting fold",
            "training": "fixed final epoch; reconstruction only; no held-fold checkpoint selection",
            "loss_score": "mean observed-cell MSE/MAE per admission, no imputed targets; "
                          "added observations change relative channel weights and score meaning",
            "primary_metric": "fold mean AP; AUROC secondary; no pooled anomaly scores",
            "sensitivity": "mask last lab bin before refitting; same preassigned folds and common "
                           "subset with an earlier baseline observation; not a new landmark or "
                           "proof of result availability",
            "interpretation": "exploratory development comparison; no threshold or final model",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=OUTPUT_ROOT / "panel-export/development.npz")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "panel-study")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if destination.exists():
        raise FileExistsError("Choose a new panel study output directory.")
    arrays = load_panel_input(args.input_file)
    torch.set_num_threads(2)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    log = destination / "training.log"
    with log.open("x") as stream, redirect_stdout(stream):
        log.chmod(0o600)
        report = compare_panels(arrays)
    # Hash only the validated development arrays, never a full/test archive.
    digest = hashlib.sha256()
    for key in sorted(KEYS):
        digest.update(arrays[key].tobytes())
    report["development_arrays_sha256"] = digest.hexdigest()
    report["versions"] = {name: version(name) for name in ("numpy", "torch", "scikit-learn")}
    target = destination / "study.json"
    save_json(target, report)
    target.chmod(0o600)
    print(f"Development-only comparison saved to {target}.")


if __name__ == "__main__":
    main()
