"""Contracts for the minimized Colab exchange and mortality evaluation."""

import io
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from src.mortality_eval import evaluate_mortality, reconstruction_scores, youden_threshold
from src.mortality_exchange import (
    KEYS, demo_exchange, load_exchange, prepare_exchange, transform_new_sequence, validate_exchange,
)
from src.mortality_model import MortalityAutoencoder, MortalityConfig, observed_error
from src.mortality_pipeline import run_mortality_experiment


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_exchange_removes_ids_preserves_patient_groups_and_no_pickle(tmp_path):
    demo = demo_exchange()
    local = dict(demo)
    for name in ("train", "validation", "test"):
        local[f"{name}_patient_id"] = np.asarray([
            f"{name}-{i}" for i in range(len(demo[f"{name}_x"]))
        ])
    local["test_patient_id"][1] = local["test_patient_id"][0]
    summary = json.loads(demo["preprocessor_json"].item())
    exported = prepare_exchange(local, summary)
    assert set(exported) == KEYS
    assert exported["test_group"][0] == exported["test_group"][1]
    assert exported["test_group"][0] != exported["test_group"][2]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **exported)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)
        assert b"test-0" not in buffer.getvalue()
    path = tmp_path / "input.npz"
    path.write_bytes(buffer.getvalue())
    assert set(load_exchange(path)) == KEYS
    local["validation_patient_id"][0] = local["train_patient_id"][0]
    with pytest.raises(ValueError, match="overlap"):
        prepare_exchange(local, summary)


def test_exchange_rejects_nonreal_data_and_wrong_masks():
    arrays = demo_exchange()
    arrays["train_x"] = arrays["train_x"].astype(complex)
    with pytest.raises(ValueError, match="real numeric"):
        validate_exchange(arrays)
    arrays = demo_exchange()
    arrays["train_x"][0, 0, 3] = 0.5
    with pytest.raises(ValueError, match="masks"):
        validate_exchange(arrays)


def test_observed_only_loss_and_missing_mask():
    x = torch.zeros((1, 10, 6))
    x[0, 0, 0], x[0, 1, 0] = 1.0, 3.0
    x[0, 0, 3], x[0, 1, 3] = 1, 1
    x[0, 2, 0] = 9999
    reconstruction = torch.zeros((1, 10, 3))
    assert observed_error(x, reconstruction, squared=False).item() == 2
    assert observed_error(x, reconstruction, squared=True).item() == 5
    with pytest.raises(ValueError, match="observed"):
        observed_error(torch.zeros_like(x), reconstruction, squared=True)


def test_threshold_and_grouped_bootstrap_are_deterministic():
    validation_y = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert youden_threshold(validation_y, validation_scores) == 0.8
    test_y = np.array([0, 0, 1, 1, 0, 1])
    test_scores = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.2])
    groups = np.array([0, 0, 1, 2, 3, 4])
    first = evaluate_mortality(validation_y, validation_scores, test_y, test_scores, groups,
                               seed=9, bootstrap_samples=200)
    second = evaluate_mortality(validation_y, validation_scores, test_y, test_scores, groups,
                                seed=9, bootstrap_samples=200)
    assert first == second
    assert first["test_patients"] == 5
    assert first["point_metrics"]["confusion_matrix"] == [[3, 0], [1, 2]]
    assert first["bootstrap"]["conditioned_on_fitted_model_and_threshold"]
    assert first["threshold"] == 0.8


def test_undefined_precision_is_explicit():
    validation_y = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.1, 0.2, 0.8, 0.9])
    test_y = np.array([0, 0, 1, 1])
    test_scores = np.array([0.1, 0.2, 0.3, 0.4])
    result = evaluate_mortality(validation_y, validation_scores, test_y, test_scores,
                                np.arange(4), seed=42, bootstrap_samples=100)
    assert result["point_metrics"]["precision"] is None
    interval = result["confidence_intervals_95"]["precision"]
    assert interval["lower"] is None and interval["upper"] is None
    assert interval["valid_replicates"] == 0
    assert "positive predictions" in interval["reason"]


def test_transform_new_sequence_uses_past_only_and_training_scaler():
    metadata = json.loads(demo_exchange()["preprocessor_json"].item())
    metadata["training_observation_medians"] = [5, 6, 7]
    metadata["training_scaler_data_min"] = [1, 2, 3]
    metadata["training_scaler_data_max"] = [5, 6, 7]
    raw = np.full((10, 3), np.nan)
    raw[3, 0] = 3
    raw[5, 1] = 4
    transformed = transform_new_sequence(raw, metadata)
    assert transformed[0, 0] == 1
    assert transformed[3, 0] == 0.5
    assert transformed[4, 0] == 0.5
    assert transformed[3, 3] == 1 and transformed[4, 3] == 0
    with pytest.raises(ValueError, match="At least one"):
        transform_new_sequence(np.full((10, 3), np.nan), metadata)


def test_synthetic_end_to_end_and_model_reload(tmp_path):
    data = demo_exchange()
    config = MortalityConfig(epochs=1, batch_size=64, bootstrap_samples=40,
                             device_preference="cpu", fast_dev_run=True)
    output = tmp_path / "run"
    metrics = run_mortality_experiment(data, config, output, source="synthetic")
    assert metrics["data_source"] == "synthetic"
    assert metrics["bootstrap"]["unit"] == "patient"
    assert 0 <= metrics["point_metrics"]["auroc"] <= 1
    assert set(path.name for path in output.iterdir()) == {
        "best_model.pt", "preprocessor.json", "config.json", "history.json",
        "metrics.json", "versions.json", "model_spec.json", "provenance.json",
        "roc_curve.png", "pr_curve.png", "training_curve.png",
    }
    model = MortalityAutoencoder()
    model.load_state_dict(torch.load(output / "best_model.pt", map_location="cpu", weights_only=True))
    assert np.isfinite(reconstruction_scores(model, data["test_x"])).all()
    with pytest.raises(FileExistsError):
        run_mortality_experiment(data, config, output, source="synthetic")


def test_notebook_synthetic_run_all(tmp_path):
    notebook = json.loads((Path(__file__).parents[1] / "colab/mortality_colab.ipynb").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell["id"] for cell in code] == [
        "environment", "configuration", "load-data", "train-and-evaluate", "save", "inference"
    ]
    for cell in code:
        assert not cell["outputs"]
        compile("".join(cell["source"]), cell["id"], "exec")
    namespace = {"Path": Path, "GIT_REVISION": "local-test"}
    for cell in code[1:]:
        source = "".join(cell["source"])
        if cell["id"] == "configuration":
            source = source.replace("DATA_SOURCE = 'drive'", "DATA_SOURCE = 'demo'")
            source = source.replace("FAST_DEV_RUN = False", "FAST_DEV_RUN = True")
            source = source.replace("bootstrap_samples=1000", "bootstrap_samples=40")
            source = source.replace("device_preference='cuda'", "device_preference='cpu'")
        if cell["id"] == "train-and-evaluate":
            source = source.replace("Path('/content/outputs/mortality')", f"Path({str(tmp_path)!r})")
        exec(compile(source, cell["id"], "exec"), namespace)
    assert namespace["metrics"]["data_source"] == "synthetic"
    assert (namespace["run_dir"] / "metrics.json").is_file()
    assert np.isfinite(namespace["example_score"])


@pytest.mark.parametrize("source", ["drive", "upload"])
def test_notebook_data_access(source, tmp_path, monkeypatch):
    notebook = json.loads((Path(__file__).parents[1] / "colab/mortality_colab.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "load-data")
    dataset = demo_exchange()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **dataset)
    path = tmp_path / "input.npz"
    path.write_bytes(buffer.getvalue())
    mounts = []
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(upload=lambda: {"input.npz": buffer.getvalue()})
    colab.drive = SimpleNamespace(mount=mounts.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {
        "DATA_SOURCE": source, "DRIVE_DATASET_PATH": str(path),
        "demo_exchange": demo_exchange, "load_exchange": load_exchange,
        "validate_exchange": validate_exchange,
        "Path": Path, "np": np, "io": io, "hashlib": hashlib,
    }
    exec(compile("".join(cell["source"]), "notebook-data-loader", "exec"), namespace)
    assert namespace["source"] == "prepared"
    assert mounts == (["/content/drive"] if source == "drive" else [])
