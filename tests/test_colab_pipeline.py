"""Pipeline contracts, data leakage guards, and artifact round-trip tests."""

import json
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import (
    FEATURE_COLUMNS, create_sequences, demo_data, load_csv, prepare_data, validate_frame,
)
from src.eval import evaluate_model, reconstruction_errors
from src.model import LSTMAutoencoder
from src.pipeline import predict_frame, save_artifacts
from src.train import train_model
from src.utils import Config, save_to_drive_if_enabled, set_seed


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_patient_splits_scaling_and_windows(tmp_path):
    config = Config(output_dir=tmp_path)
    frame = demo_data()
    data = prepare_data(frame, config)
    splits = [set(ids) for ids in data.split_patients.values()]
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    assert data.train.shape[1:] == (24, 8)
    assert set(data.test_labels) == {0, 1}
    windows, labels, patients = create_sequences(frame, 24)
    assert windows.shape == (80, 24, 8)
    assert len(labels) == len(patients) == 80
    for name in ("train", "validation"):
        assert frame[frame.patient_id.isin(data.split_patients[name])].sepsis_label.max() == 0
    raw_train, _, _ = create_sequences(
        frame[frame.patient_id.isin(data.split_patients["train"])], 24
    )
    np.testing.assert_allclose(
        data.preprocessor.scaler.data_min_, raw_train.reshape(-1, 8).min(axis=0)
    )
    changed = frame.copy()
    heldout = changed.patient_id.isin(data.split_patients["test"])
    changed.loc[heldout, "heart_rate"] = 210
    again = prepare_data(changed, config)
    np.testing.assert_array_equal(data.train, again.train)
    np.testing.assert_array_equal(data.preprocessor.scaler.data_max_,
                                  again.preprocessor.scaler.data_max_)


@pytest.mark.parametrize("problem", ["column", "label", "duplicate", "missing", "timestamp"])
def test_csv_validation(problem):
    frame = demo_data()
    if problem == "column":
        frame = frame.drop(columns="spo2")
    elif problem == "label":
        frame.loc[0, "sepsis_label"] = 2
    elif problem == "duplicate":
        frame = frame.iloc[[0, 0]]
    elif problem == "missing":
        frame.loc[0, "patient_id"] = None
    else:
        frame.loc[0, "timestamp"] = None
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_cleaning_missing_feature_and_short_records(tmp_path):
    frame = demo_data()
    selected = frame.patient_id == frame.patient_id.iloc[0]
    frame.loc[selected, "spo2"] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        prepare_data(frame, Config(output_dir=tmp_path))
    with pytest.raises(ValueError, match="too short"):
        prepare_data(demo_data().iloc[1:], Config(output_dir=tmp_path, seq_len=48))
    with pytest.raises(ValueError, match="No complete"):
        create_sequences(demo_data().iloc[:1], 24)
    with pytest.raises(ValueError, match="5 normal"):
        prepare_data(demo_data().iloc[:48], Config(output_dir=tmp_path))


def test_csv_roundtrip_and_patient_local_imputation(tmp_path):
    frame = demo_data()
    frame.loc[0, "heart_rate"] = np.nan
    frame.loc[1, "heart_rate"] = 1000
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    loaded = load_csv(path)
    windows, _, _ = create_sequences(loaded, 24)
    assert np.isfinite(windows).all()
    assert windows[0, 0, 0] == windows[0, 2, 0]
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "does-not-exist.csv")


def test_end_to_end_and_saved_inference(tmp_path):
    config = Config(output_dir=tmp_path, epochs=3, hidden_dim=8, latent_dim=4,
                    num_layers=1, batch_size=128, device_preference="cpu")
    set_seed(config.seed)
    data = prepare_data(demo_data(), config)
    model = LSTMAutoencoder(8, config.hidden_dim, config.latent_dim, config.seq_len,
                            config.num_layers, config.dropout)
    history = train_model(model, data.train, data.validation, config)
    assert len(history["train_loss"]) == 2  # fast mode overrides three epochs
    metrics, errors = evaluate_model(model, data.validation, data.test,
                                    data.test_labels, config)
    save_artifacts(data, config)
    assert save_to_drive_if_enabled(config) is None
    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["auprc"] <= 1
    assert np.isfinite(errors).all()
    calibration = reconstruction_errors(model, data.validation, config.batch_size)
    assert metrics["threshold"] == pytest.approx(np.percentile(calibration, 95))
    persisted = torch.load(tmp_path / "best_model.pt", weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, persisted[name])
    new = demo_data(seed=100).iloc[:48]
    predictions = predict_frame(new, tmp_path)
    sequences, _, _ = data.preprocessor.transform(new)
    np.testing.assert_allclose(predictions.reconstruction_error,
                               reconstruction_errors(model, sequences), rtol=1e-6)
    np.testing.assert_array_equal(predictions.anomaly,
                                   predictions.reconstruction_error > metrics["threshold"])
    required = {
        "best_model.pt", "preprocessor.joblib", "threshold.json", "config.json",
        "model_config.json", "history.json", "metrics.json", "versions.json",
        "split_summary.json", "training_curve.png", "roc_curve.png", "pr_curve.png",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    for filename in required:
        assert (tmp_path / filename).stat().st_size > 0
    assert json.loads((tmp_path / "threshold.json").read_text())["source"] == "normal_validation"
    assert "synthetic-" not in (tmp_path / "split_summary.json").read_text()


def test_seed_reproducibility(tmp_path):
    config = Config(output_dir=tmp_path)
    set_seed(config.seed)
    first = LSTMAutoencoder(8).state_dict()
    set_seed(config.seed)
    second = LSTMAutoencoder(8).state_dict()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_notebook_structure():
    notebook = json.loads((Path(__file__).parents[1] / "colab/neonatal_sepsis_colab.ipynb").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell["id"] for cell in code] == [
        "environment", "configuration", "load-data", "preprocess", "model",
        "training", "evaluation", "save", "inference",
    ]
    for cell in code:
        assert not cell["outputs"]
        compile("".join(cell["source"]), f"cell-{cell['id']}", "exec")
    first = "".join(code[0]["source"])
    assert first.index("'clone'") < first.index("'install'")


def test_invalid_config_and_errors(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        Config(batch_size=0)
    with pytest.raises(ValueError, match="threshold_percentile"):
        Config(threshold_percentile=100)
    with pytest.raises(ValueError, match="nonempty"):
        reconstruction_errors(LSTMAutoencoder(8), np.empty((0, 24, 8)))
    config = Config(output_dir=tmp_path, drive_output_dir=tmp_path / "nested")
    # No success-shaped fallback when Drive isn't mounted.
    if not Path("/content/drive/MyDrive").is_dir():
        with pytest.raises(RuntimeError, match="Mount"):
            save_to_drive_if_enabled(config)


@pytest.mark.parametrize("source", ["upload", "drive", "url"])
def test_notebook_data_sources(source, tmp_path, monkeypatch):
    """Execute the actual notebook loading cell with mocked Colab interfaces."""
    frame = demo_data()
    payload = frame.to_csv(index=False).encode()
    local_csv = tmp_path / "input.csv"
    local_csv.write_bytes(payload)
    mount_calls = []
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(upload=lambda: {"input.csv": payload})
    colab.drive = SimpleNamespace(mount=mount_calls.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr("src.data.urlopen", lambda url, timeout: io.BytesIO(payload))
    notebook = json.loads((Path(__file__).parents[1] / "colab/neonatal_sepsis_colab.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "load-data")
    namespace = {
        "DATA_SOURCE": source, "COPY_TO_DRIVE": False,
        "DRIVE_CSV": local_csv, "CSV_URL": "https://example.org/input.csv",
        "FEATURE_COLUMNS": FEATURE_COLUMNS, "load_csv": load_csv,
    }
    exec(compile("".join(cell["source"]), "notebook-data-loader", "exec"), namespace)
    assert len(namespace["frame"]) == len(frame)
    assert mount_calls == (["/content/drive"] if source == "drive" else [])


def test_drive_artifact_copy_and_invalid_destination(tmp_path, monkeypatch):
    original_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path, "is_dir",
        lambda path: True if str(path) == "/content/drive/MyDrive" else original_is_dir(path),
    )
    source = tmp_path / "local"
    source.mkdir()
    (source / "metrics.json").write_text('{"auroc": 0.5}')
    destination = tmp_path / "drive-copy"
    config = Config(output_dir=source, drive_output_dir=destination)
    assert save_to_drive_if_enabled(config) == destination
    assert (destination / "metrics.json").read_text() == (source / "metrics.json").read_text()
    config.drive_output_dir = source / "recursive"
    with pytest.raises(ValueError, match="inside"):
        save_to_drive_if_enabled(config)
