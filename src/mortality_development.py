"""Export development-only observations without reading any test array."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .mortality_exchange import ROOT


KEYS = {"observed", "labels", "groups"}


def validate_development(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(arrays) != KEYS:
        raise ValueError("Development input must contain only observed, labels and groups.")
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    if (observed.ndim != 3 or observed.shape[1:] != (10, 3) or not len(observed)
            or not np.issubdtype(observed.dtype, np.floating) or np.isinf(observed).any()):
        raise ValueError("Expected real observations of shape (admissions, 10, 3), with NaN gaps.")
    if not np.isfinite(observed).any(axis=(1, 2)).all():
        raise ValueError("Each development admission needs an observed biomarker.")
    if labels.shape != (len(observed),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Development labels must be binary, aligned and include both classes.")
    if (groups.shape != labels.shape or not np.issubdtype(groups.dtype, np.integer)
            or (groups < 0).any()):
        raise ValueError("Development groups must be aligned nonnegative integers.")
    return arrays


def prepare_development(local_arrays, summary: dict, seed: int = 42) -> dict[str, np.ndarray]:
    """Read only train/validation keys, invert scaling, discard imputed entries."""
    parts, labels, patients = [], [], []
    minimum = np.asarray(summary["training_scaler_data_min"], dtype=float)
    maximum = np.asarray(summary["training_scaler_data_max"], dtype=float)
    if (minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite(minimum).all()
            or not np.isfinite(maximum).all() or (maximum < minimum).any()):
        raise ValueError("Invalid local scaler bounds.")
    scale = maximum - minimum
    scale[scale < 10 * np.finfo(scale.dtype).eps] = 1
    for split in ("train", "validation"):
        x = local_arrays[f"{split}_x"]
        y = local_arrays[f"{split}_y"]
        ids = local_arrays[f"{split}_patient_id"]
        if x.ndim != 3 or x.shape[1:] != (10, 6) or not np.isfinite(x).all():
            raise ValueError("Invalid prepared development sequences.")
        if not np.isin(x[:, :, 3:], [0, 1]).all():
            raise ValueError("Development masks must be binary.")
        if ids.shape != (len(x),) or y.shape != (len(x),) or np.any(ids == ""):
            raise ValueError("Development labels and patient IDs must align.")
        values = x[:, :, :3].astype(float) * scale + minimum
        parts.append(np.where(x[:, :, 3:] == 1, values, np.nan))
        labels.append(y)
        patients.append(ids)
    if set(patients[0].tolist()) & set(patients[1].tolist()):
        raise ValueError("Original training and validation patient groups overlap.")
    _, inverse = np.unique(np.concatenate(patients), return_inverse=True)
    codes = np.random.default_rng(seed).permutation(inverse.max() + 1)
    return validate_development({
        "observed": np.concatenate(parts),
        "labels": np.concatenate(labels),
        "groups": codes[inverse].astype(np.int32),
    })


def load_development(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != KEYS:
            raise ValueError("Use the development-only NPZ; full or test exchange files are rejected.")
        arrays = {key: archive[key] for key in KEYS}
    return validate_development(arrays)


def export_development(input_dir: Path, output: Path) -> None:
    output = output.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output.exists():
        raise FileExistsError("Development file already exists; choose a new path.")
    summary = json.loads((input_dir / "summary.json").read_text())
    with np.load(input_dir / "dataset.npz", allow_pickle=False) as archive:
        arrays = prepare_development(archive, summary)
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if output.parent.stat().st_mode & 0o077:
        raise PermissionError("Development output directory must be private.")
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Development-only NPZ saved to {output}; no test array was read.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/mortality-dataset")
    parser.add_argument("--output-file", type=Path,
                        default=ROOT / "outputs/mortality-development/development.npz")
    args = parser.parse_args()
    export_development(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
