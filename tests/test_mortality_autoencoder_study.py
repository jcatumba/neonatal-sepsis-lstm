"""Focused tests for nested autoencoder improvement experiments."""

import json

import numpy as np
import pytest
import torch

from src import mortality_autoencoder_study as study
from src.mortality_comparison import fold_sequences, patient_folds


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def development():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.tile([0, 1], 12), 2).astype(np.int8)
    observed = rng.uniform(0, 10, (48, 10, 3))
    observed[rng.random(observed.shape) < 0.6] = np.nan
    observed[:, 0, 0] = 10 * labels + rng.uniform(0, 0.1, 48)
    return {
        "observed": observed, "labels": labels,
        "groups": np.repeat(np.arange(24, dtype=np.int32), 2),
    }


def test_log_transform_preserves_masks_and_rejects_negative_values(development):
    observed = development["observed"].copy()
    observed[0, 0, 0] = 0
    transformed = study.transform_observations(observed, "log1p")
    assert transformed[0, 0, 0] == 0
    np.testing.assert_array_equal(np.isfinite(transformed), np.isfinite(observed))
    np.testing.assert_allclose(np.expm1(transformed), observed)
    training, validation = transformed[:24], transformed[24:]
    first, _ = fold_sequences(training, validation)
    second, shifted = fold_sequences(training, validation * 10000)
    np.testing.assert_array_equal(first, second)
    assert shifted[:, :, :3].max() > 1
    observed[0, 0, 0] = -0.5
    with pytest.raises(ValueError, match="nonnegative"):
        study.transform_observations(observed, "log1p")


def test_candidate_fits_survivors_and_transformed_training_only(development, monkeypatch):
    observed, labels = development["observed"], development["labels"]
    training, validation = observed[:24], observed[24:]
    candidate = study.AutoencoderCandidate("survivors", "log1p", "survivors", epochs=3)
    expected_train, expected_validation = fold_sequences(
        np.log1p(training[labels[:24] == 0]), np.log1p(validation),
    )
    model = object()

    def checked_train(train_x, validation_x, config):
        np.testing.assert_array_equal(train_x, expected_train)
        assert validation_x is None
        assert config.epochs == 3 and config.seed == 17
        assert config.learning_rate == 0.0001 and not config.fast_dev_run
        return model, {"train_loss": [1, 0.5, 0.3]}

    def checked_scores(fitted, validation_x, batch_size):
        assert fitted is model and batch_size == 32
        np.testing.assert_array_equal(validation_x, expected_validation)
        return np.arange(len(validation_x), dtype=float)

    monkeypatch.setattr(study, "train_autoencoder", checked_train)
    monkeypatch.setattr(study, "reconstruction_scores", checked_scores)
    result = study.candidate_scores(training, labels[:24], validation, candidate, seed=17, device="cpu")
    assert len(result) == len(validation)
    with pytest.raises(ValueError, match="empty"):
        study.candidate_scores(training, np.ones(24), validation, candidate, seed=17, device="cpu")


def test_nested_selection_refits_every_inner_and_outer_population(development, monkeypatch):
    candidates = (study.AutoencoderCandidate("baseline"), study.AutoencoderCandidate("log", "log1p"))
    observed, labels, groups = (development[key] for key in ("observed", "labels", "groups"))
    expected = []
    for fold, (training, validation) in enumerate(patient_folds(labels, groups, 2, 42), 1):
        inner_folds = list(patient_folds(labels[training], groups[training], 2, 42 + fold))
        for candidate in candidates:
            for inner, (fitting, held) in enumerate(inner_folds, 1):
                expected.append((candidate.name, training[fitting], training[held],
                                 42 + 100 * fold + inner))
        expected.extend([("log", training, validation, 42 + fold),
                         ("baseline", training, validation, 42 + fold)])
    calls = []

    def checked_scores(training, fitting_labels, validation, candidate, *, seed, device):
        name, fitting, held, expected_seed = expected[len(calls)]
        assert candidate.name == name and seed == expected_seed and device == "cpu"
        np.testing.assert_array_equal(training, observed[fitting])
        np.testing.assert_array_equal(fitting_labels, labels[fitting])
        np.testing.assert_array_equal(validation, observed[held])
        assert not set(groups[fitting]) & set(groups[held])
        calls.append(name)
        return validation[:, 0, 0] if name == "log" else 100 - validation[:, 0, 0]

    monkeypatch.setattr(study, "candidate_scores", checked_scores)
    report = study.nested_autoencoder_study(
        development, candidates=candidates, outer_splits=2, inner_splits=2,
    )
    assert len(calls) == len(expected)
    assert report["selection_counts"] == {"baseline": 0, "log": 2}
    for record in report["folds"]:
        best = max(record["inner_search"], key=lambda item: item["mean_inner_average_precision"])
        assert record["selected_candidate"] == best["candidate"]
    assert report["summary"]["selected"]["average_precision"]["fold_mean"] == 1
    assert report["outer_oof_coverage"] == {"min": 1, "max": 1}
    assert "pooled_metrics" not in report["summary"]


def test_tied_inner_scores_retain_baseline_without_retraining_it(development, monkeypatch):
    candidates = (study.AutoencoderCandidate("baseline"), study.AutoencoderCandidate("log", "log1p"))
    calls = []

    def constant_scores(training, labels, validation, candidate, **kwargs):
        calls.append(candidate.name)
        return np.ones(len(validation))

    monkeypatch.setattr(study, "candidate_scores", constant_scores)
    report = study.nested_autoencoder_study(
        development, candidates=candidates, outer_splits=2, inner_splits=2,
    )
    assert report["selection_counts"]["baseline"] == 2
    assert len(calls) == 2 * (2 * 2 + 1)
    assert report["summary"]["baseline"] == report["summary"]["selected"]


def test_synthetic_study_runs_fixed_epochs_and_saves_only_aggregate_results(development, tmp_path):
    path = tmp_path / "development.npz"
    np.savez_compressed(path, **development)
    candidates = (study.AutoencoderCandidate("baseline", epochs=1),
                  study.AutoencoderCandidate("log", "log1p", epochs=1))
    output = tmp_path / "study"
    report = study.run_autoencoder_study(
        path, output, candidates=candidates, outer_splits=2, inner_splits=2,
    )
    assert report["test_accessed"] is False
    assert report["protocol"]["thresholds"] == "not_selected"
    assert report["protocol"]["final_model"] == "not_fitted_or_exported_review_study_before_freezing"
    assert report["candidates"][0]["epochs"] == 1
    assert json.loads((output / "study.json").read_text()) == report
    assert set(p.name for p in output.iterdir()) == {"study.json", "comparison.png", "training.log"}
    assert "Epoch 1:" in (output / "training.log").read_text()
    assert "Epoch 2:" not in (output / "training.log").read_text()
    assert (output / "comparison.png").read_bytes().startswith(b"\x89PNG")
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())
    assert not {"labels", "groups", "scores", "predictions"} & set(report)
    with pytest.raises(FileExistsError):
        study.run_autoencoder_study(path, output)
    wrong = tmp_path / "wrong.npz"
    np.savez_compressed(wrong, test_x=np.ones(1))
    with pytest.raises(ValueError, match="development-only"):
        study.run_autoencoder_study(wrong, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
