"""Minimized, patient-pseudonymous data exchange for the Colab experiment."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "lactate", "total_bilirubin", "creatinine",
    "lactate_observed", "total_bilirubin_observed", "creatinine_observed",
)
KEYS = {"train_x", "validation_x", "validation_y", "test_x", "test_y",
        "test_group", "preprocessor_json"}


def validate_exchange(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(arrays) != KEYS:
        raise ValueError("Exchange file must contain exactly the required arrays.")
    for split in ("train", "validation", "test"):
        x = arrays[f"{split}_x"]
        if x.ndim != 3 or x.shape[1:] != (10, 6) or not len(x):
            raise ValueError(f"{split} must have nonempty shape (admissions, 10, 6).")
        if (not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x)
                or not np.isfinite(x).all()):
            raise ValueError(f"{split} sequences must be finite real numeric values.")
        if not np.isin(x[:, :, 3:], (0, 1)).all():
            raise ValueError(f"{split} observation masks must be binary.")
        if not (x[:, :, 3:].sum(axis=(1, 2)) > 0).all():
            raise ValueError(f"{split} contains an admission without observations.")
    for split in ("validation", "test"):
        y = arrays[f"{split}_y"]
        if y.shape != (len(arrays[f"{split}_x"]),) or set(np.unique(y)) != {0, 1}:
            raise ValueError(f"{split} labels must match sequences and include both classes.")
    groups = arrays["test_group"]
    if groups.shape != (len(arrays["test_x"]),) or not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("Test patient groups must be a nonmissing integer per admission.")
    if (groups < 0).any():
        raise ValueError("Test patient groups must be nonnegative.")
    try:
        metadata = json.loads(str(arrays["preprocessor_json"].item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Preprocessor metadata must be a JSON scalar.") from exc
    if not isinstance(metadata, dict) or metadata.get("features") != list(FEATURES) or metadata.get("bin_seconds") != 8640:
        raise ValueError("Preprocessor metadata does not match the mortality input schema.")
    for key in ("training_observation_medians", "training_scaler_data_min",
                "training_scaler_data_max"):
        value = np.asarray(metadata.get(key), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid preprocessor metadata: {key}.")
    if np.any(np.asarray(metadata["training_scaler_data_min"]) >
              np.asarray(metadata["training_scaler_data_max"])):
        raise ValueError("Training scaler bounds are inconsistent.")
    return arrays


def load_exchange(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return validate_exchange(arrays)


def prepare_exchange(local_arrays, summary: dict, seed: int = 42) -> dict[str, np.ndarray]:
    for split in ("train", "validation", "test"):
        ids = local_arrays[f"{split}_patient_id"]
        if ids.ndim != 1 or len(ids) != len(local_arrays[f"{split}_x"]) or not np.all(ids != ""):
            raise ValueError(f"{split} patient IDs do not align with sequences.")
    patient_sets = [set(local_arrays[f"{split}_patient_id"].tolist())
                    for split in ("train", "validation", "test")]
    if patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2]:
        raise ValueError("Local patient splits overlap; refusing to export.")
    test_ids = local_arrays["test_patient_id"]
    _, inverse = np.unique(test_ids, return_inverse=True)
    random_labels = np.random.default_rng(seed).permutation(len(np.unique(inverse)))
    metadata_keys = ("features", "bin_seconds", "training_observation_medians",
                     "training_scaler_data_min", "training_scaler_data_max")
    metadata = {key: summary[key] for key in metadata_keys}
    arrays = {key: local_arrays[key] for key in
              ("train_x", "validation_x", "validation_y", "test_x", "test_y")}
    arrays["test_group"] = random_labels[inverse].astype(np.int32)
    arrays["preprocessor_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    return validate_exchange(arrays)


def demo_exchange(seed: int = 42) -> dict[str, np.ndarray]:
    """Generate nonclinical synthetic inputs to check the notebook workflow."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for split, size in (("train", 96), ("validation", 40), ("test", 40)):
        values = rng.uniform(0, 1, (size, 10, 3))
        masks = rng.binomial(1, 0.4, (size, 10, 3))
        masks[:, 0, 0] = 1
        arrays[f"{split}_x"] = np.concatenate((values, masks), axis=2).astype(np.float32)
        if split != "train":
            arrays[f"{split}_y"] = np.tile([0, 1], size // 2).astype(np.int8)
    arrays["test_group"] = np.arange(40, dtype=np.int32)
    arrays["preprocessor_json"] = np.asarray(json.dumps({
        "features": list(FEATURES), "bin_seconds": 8640,
        "training_observation_medians": [0.5] * 3,
        "training_scaler_data_min": [0.0] * 3,
        "training_scaler_data_max": [1.0] * 3,
    }))
    return validate_exchange(arrays)


def transform_new_sequence(raw: np.ndarray, metadata: dict) -> np.ndarray:
    """Transform one already-binned sequence in the documented source units."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (10, 3) or np.isinf(raw).any():
        raise ValueError("Expected 10x3 values with NaN for missing observations.")
    observed = np.isfinite(raw)
    if not observed.any():
        raise ValueError("At least one biomarker observation is required.")
    filled = raw.copy()
    for step in range(1, 10):
        filled[step] = np.where(np.isfinite(filled[step]), filled[step], filled[step - 1])
    medians = np.asarray(metadata["training_observation_medians"], dtype=float)
    minimum = np.asarray(metadata["training_scaler_data_min"], dtype=float)
    maximum = np.asarray(metadata["training_scaler_data_max"], dtype=float)
    if any(value.shape != (3,) or not np.isfinite(value).all()
           for value in (medians, minimum, maximum)) or np.any(maximum < minimum):
        raise ValueError("Invalid training preprocessor metadata.")
    filled = np.where(np.isfinite(filled), filled, medians)
    scale = maximum - minimum
    scale[scale < 10 * np.finfo(scale.dtype).eps] = 1
    values = (filled - minimum) / scale
    return np.concatenate((values, observed.astype(float)), axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/mortality-dataset")
    parser.add_argument("--output-file", type=Path,
                        default=ROOT / "outputs/mortality-colab/mortality_colab.npz")
    args = parser.parse_args()
    output = args.output_file.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Outputs inside the repository must be under ignored outputs/.")
    if output.exists():
        raise FileExistsError("Output file exists; choose a new path.")
    summary = json.loads((args.input_dir / "summary.json").read_text())
    with np.load(args.input_dir / "dataset.npz", allow_pickle=False) as local:
        arrays = prepare_exchange(local, summary)
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if output.parent.stat().st_mode & 0o077:
        raise PermissionError("Output directory must not be accessible to group or others.")
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Prepared a private, pseudonymous Colab file: {output}")
    print("No data was uploaded; obtain required approval before using a cloud service.")


if __name__ == "__main__":
    main()
