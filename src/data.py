"""Validated CSV ingestion and patient-disjoint preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from preprocessing import FEATURE_COLUMNS, PHYSIOLOGICAL_BOUNDS
from .utils import Config


def load_csv(path: Path | str) -> pd.DataFrame:
    """Read a local/Drive CSV or an explicit public HTTPS CSV URL."""
    location = str(path)
    if urlparse(location).scheme == "https":
        with urlopen(location, timeout=60) as response:
            return validate_frame(pd.read_csv(response))
    if "://" in location:
        raise ValueError("Only local paths and HTTPS CSV URLs are supported.")
    return validate_frame(pd.read_csv(Path(path)))


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["patient_id", "timestamp", "sepsis_label", *FEATURE_COLUMNS]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    frame = frame[required].copy()
    if frame.empty or frame["patient_id"].isna().any():
        raise ValueError("CSV must contain rows with nonmissing patient_id.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
    if frame["timestamp"].isna().any():
        raise ValueError("timestamp must not be missing.")
    if frame.duplicated(["patient_id", "timestamp"]).any():
        raise ValueError("Duplicate patient_id/timestamp rows are not supported.")
    frame["sepsis_label"] = pd.to_numeric(frame["sepsis_label"], errors="raise")
    if not frame["sepsis_label"].isin([0, 1]).all():
        raise ValueError("sepsis_label must be 0 (normal) or 1 (sepsis).")
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def demo_data(seed: int = 42, seq_len: int = 24) -> pd.DataFrame:
    """Synthetic pipeline check, not a clinically representative dataset."""
    rng = np.random.default_rng(seed)
    records = []
    centers = np.array([140, 45, 96, 37, 65, 35, 45, 5])
    scales = np.array([5, 2, 0.5, 0.1, 2, 1, 1, 0.2])
    for patient in range(40):
        label = int(patient >= 24)
        size = seq_len * 2
        values = centers + rng.normal(size=(size, 8)) * scales
        if label:
            values += np.array([30, 15, -8, 2, -15, -10, -12, 3])
        part = pd.DataFrame(values, columns=FEATURE_COLUMNS)
        part["patient_id"] = f"synthetic-{patient:03}"
        part["timestamp"] = pd.date_range("2025-01-01", periods=size, freq="h")
        part["sepsis_label"] = label
        records.append(part)
    return validate_frame(pd.concat(records, ignore_index=True))


@dataclass
class Preprocessor:
    scaler: MinMaxScaler
    seq_len: int
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)

    def transform(self, frame: pd.DataFrame):
        windows, labels, patients = create_sequences(validate_frame(frame), self.seq_len)
        transformed = self.scaler.transform(windows.reshape(-1, windows.shape[-1]))
        return transformed.reshape(windows.shape).astype(np.float32), labels, patients


def create_sequences(frame: pd.DataFrame, seq_len: int):
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    windows, labels, patients = [], [], []
    for patient, part in frame.groupby("patient_id", sort=True):
        values = part[FEATURE_COLUMNS].copy()
        for column, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
            values.loc[~values[column].between(low, high), column] = np.nan
        values = values.interpolate(limit_direction="both").ffill().bfill()
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Patient {patient} has an entirely missing/invalid feature.")
        raw = values.to_numpy(dtype=np.float32)
        targets = part["sepsis_label"].to_numpy()
        # Nonoverlapping windows reduce memory use and within-patient correlation.
        for start in range(0, len(part) - seq_len + 1, seq_len):
            windows.append(raw[start:start + seq_len])
            labels.append(int(targets[start:start + seq_len].max()))
            patients.append(patient)
    if not windows:
        raise ValueError(f"No complete sequences; each patient needs {seq_len} rows.")
    return np.stack(windows), np.array(labels), np.array(patients)


@dataclass
class PreparedData:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    preprocessor: Preprocessor
    split_patients: dict


def prepare_data(frame: pd.DataFrame, config: Config) -> PreparedData:
    frame = validate_frame(frame)
    counts = frame.groupby("patient_id").size()
    short = counts[counts < config.seq_len]
    if not short.empty:
        raise ValueError(f"Patients need at least seq_len rows; too short: {list(short.index)}")
    strata = frame.groupby("patient_id")["sepsis_label"].max()
    if any((strata == label).sum() < 5 for label in (0, 1)):
        raise ValueError("Provide at least 5 normal and 5 sepsis patients for disjoint splits.")
    train_ids, held_ids = train_test_split(
        strata.index.to_numpy(), test_size=0.4, stratify=strata.to_numpy(),
        random_state=config.seed,
    )
    val_ids, test_ids = train_test_split(
        held_ids, test_size=0.5, stratify=strata.loc[held_ids].to_numpy(),
        random_state=config.seed,
    )
    arrays = []
    split_patients = {}
    for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        # Entirely normal patients only for fitting and threshold calibration.
        if name != "test":
            ids = np.array([patient for patient in ids if strata.loc[patient] == 0])
        windows, labels, _ = create_sequences(
            frame[frame["patient_id"].isin(ids)], config.seq_len
        )
        if config.fast_dev_run:
            # Stratification preserves both test classes even for tiny smoke runs.
            indices = np.concatenate([
                np.flatnonzero(labels == label)[:32] for label in np.unique(labels)
            ])
            windows, labels = windows[indices], labels[indices]
        arrays.append((windows, labels))
        split_patients[name] = ids.tolist()
    scaler = MinMaxScaler().fit(arrays[0][0].reshape(-1, len(FEATURE_COLUMNS)))
    scaled = [
        scaler.transform(windows.reshape(-1, len(FEATURE_COLUMNS)))
        .reshape(windows.shape).astype(np.float32)
        for windows, _ in arrays
    ]
    if set(arrays[2][1]) != {0, 1}:
        raise ValueError("Held-out test sequences must contain both label classes.")
    return PreparedData(
        *scaled, arrays[2][1], Preprocessor(scaler, config.seq_len), split_patients
    )
