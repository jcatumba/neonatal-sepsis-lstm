"""Development-only export and leakage-safe model comparisons."""

import json

import numpy as np
import pytest
import torch

from src import mortality_comparison
from src import mortality_logistic
from src import mortality_operating_point
from src.mortality_comparison import (
    biomarker_features, fold_sequences, patient_folds, run_comparison,
)
from src.mortality_development import (
    export_development, load_development, prepare_development,
)
from src.mortality_model import MortalityConfig, train_autoencoder
from src.mortality_eval import operating_metrics, youden_threshold
from src.mortality_operating_point import evaluate_youden, predict_observed, run_operating_point
from src.mortality_logistic import (
    C_VALUES, calibration_bins, nested_logistic, probability_metrics, run_nested_logistic,
    select_regularization,
)


@pytest.fixture
def development():
    rng = np.random.default_rng(42)
    observed = rng.uniform(1, 10, (48, 10, 3))
    observed[rng.random(observed.shape) < 0.6] = np.nan
    observed[:, 0, 0] = rng.uniform(1, 10, 48)
    return {
        "observed": observed,
        "labels": np.repeat(np.tile([0, 1], 12), 2).astype(np.int8),
        "groups": np.repeat(np.arange(24, dtype=np.int32), 2),
    }


def test_export_never_reads_test_and_removes_ids(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    summary = {"training_scaler_data_min": [10, 20, 30],
               "training_scaler_data_max": [20, 40, 60]}
    (source / "summary.json").write_text(json.dumps(summary))
    arrays = {}
    for split in ("train", "validation"):
        x = np.zeros((4, 10, 6), dtype=np.float32)
        x[:, 2, :3] = 0.5
        x[:, 2, 3:] = 1
        arrays[f"{split}_x"] = x
        arrays[f"{split}_y"] = np.array([0, 1, 0, 1])
        arrays[f"{split}_patient_id"] = np.array([f"{split}-private-{i // 2}" for i in range(4)])
    arrays["test_x"] = np.array([-999])
    np.savez_compressed(source / "dataset.npz", **arrays)
    original = np.lib.npyio.NpzFile.__getitem__
    reads = []

    def guarded_read(self, key):
        assert not key.startswith("test")
        reads.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded_read)
    output = tmp_path / "private" / "development.npz"
    export_development(source, output)
    result = load_development(output)
    assert set(result) == {"observed", "labels", "groups"}
    np.testing.assert_array_equal(result["observed"][:, 2], np.tile([15, 30, 45], (8, 1)))
    assert np.isnan(result["observed"][:, 0]).all()
    assert result["groups"][0] == result["groups"][1]
    assert result["groups"][0] != result["groups"][4]
    assert not any(key.startswith("test") for key in reads)
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        export_development(source, output)
    arrays["validation_patient_id"][0] = arrays["train_patient_id"][0]
    with pytest.raises(ValueError, match="overlap"):
        prepare_development(arrays, summary)


def test_full_exchange_is_rejected_before_arrays_are_read(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez_compressed(path, test_x=np.zeros(1))
    with pytest.raises(ValueError, match="development-only"):
        load_development(path)


def test_grouped_folds_are_disjoint_complete_and_deterministic(development):
    y, groups = development["labels"], development["groups"]
    first = list(patient_folds(y, groups, 3, 42))
    second = list(patient_folds(y, groups, 3, 42))
    held = []
    for (train, validation), (train_again, validation_again) in zip(first, second):
        assert not set(groups[train]) & set(groups[validation])
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(validation, validation_again)
        held.extend(validation)
    assert sorted(held) == list(range(len(y)))
    with pytest.raises(ValueError, match="Too few"):
        list(patient_folds(y, groups, 20, 42))


def test_preprocessing_is_fit_on_training_and_features_ignore_filled_values(development):
    training = development["observed"][:24]
    held = development["observed"][24:]
    first_train, first_held = fold_sequences(training, held)
    second_train, second_held = fold_sequences(training, held * 10000)
    np.testing.assert_array_equal(first_train, second_train)
    assert not np.array_equal(first_held, second_held)
    assert second_held[:, :, :3].max() > 1
    raw = np.full((1, 10, 3), np.nan)
    raw[0, 2, 0], raw[0, 4, 0] = 4, 10
    features = biomarker_features(raw)
    np.testing.assert_allclose(features[0, :5], [10, 4, 10, 1.25, 2])
    assert np.isnan(features[0, 5:9]).all()
    assert features[0, 9] == 0


def test_comparison_trains_survivors_only_and_writes_aggregate_results(
        development, tmp_path, monkeypatch):
    expected = []
    for training, validation in patient_folds(development["labels"], development["groups"], 2, 42):
        for selected in (training, training[development["labels"][training] == 0]):
            expected.append(fold_sequences(development["observed"][selected],
                                           development["observed"][validation])[0])
    actual = []

    def checked_train(training, validation, config):
        assert validation is None
        np.testing.assert_array_equal(training, expected[len(actual)])
        actual.append(training)
        model, history = train_autoencoder(training, validation, config)
        assert len(history["train_loss"]) == 1
        assert "validation_loss" not in history
        return model, history

    monkeypatch.setattr(mortality_comparison, "train_autoencoder", checked_train)
    path = tmp_path / "development.npz"
    np.savez_compressed(path, **development)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = run_comparison(path, tmp_path / "result",
                                MortalityConfig(epochs=1, device_preference="cpu"), n_splits=2)
    finally:
        torch.set_num_threads(previous_threads)
    assert len(actual) == 4
    assert report["test_accessed"] is False
    assert report["protocol"]["thresholds"] == "not_selected"
    assert set(report["summary"]) == {
        "logistic_regression", "autoencoder_all", "autoencoder_survivors",
    }
    for metrics in report["summary"].values():
        assert 0 <= metrics["auroc"]["fold_mean"] <= 1
    assert set(p.name for p in (tmp_path / "result").iterdir()) == {"comparison.json"}
    saved = json.loads((tmp_path / "result/comparison.json").read_text())
    assert saved == report
    assert "observed" not in saved and "groups" not in saved
    with pytest.raises(FileExistsError):
        run_comparison(path, tmp_path / "result", MortalityConfig(), n_splits=2)


def test_probability_metrics_and_quantile_calibration_handle_ties():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.7, 0.9])
    metrics = probability_metrics(labels, probabilities)
    assert metrics["brier_score"] == pytest.approx(0.0675)
    assert metrics["average_precision"] == 1
    bins = calibration_bins(labels, probabilities, n_bins=2)
    assert [item["admissions"] for item in bins] == [2, 2]
    np.testing.assert_allclose([item["mean_predicted_probability"] for item in bins], [0.25, 0.8])
    assert [item["observed_mortality_fraction"] for item in bins] == [0, 1]
    constant = calibration_bins(labels, np.full(4, 0.5))
    assert len(constant) == 1 and constant[0]["admissions"] == 4
    assert constant[0]["deaths"] == 2
    for invalid in (np.array([np.nan, 0.4, 0.7, 0.9]), probabilities + 1):
        with pytest.raises(ValueError, match="probabilities"):
            probability_metrics(labels, invalid)


def test_inner_search_breaks_ties_toward_stronger_regularization(development, monkeypatch):
    from sklearn.dummy import DummyClassifier

    monkeypatch.setattr(mortality_logistic, "make_logistic_classifier",
                        lambda c, seed: DummyClassifier(strategy="prior"))
    selected, candidates = select_regularization(
        biomarker_features(development["observed"]), development["labels"],
        development["groups"], n_splits=2,
    )
    assert selected == min(C_VALUES)
    assert len({candidate["mean_inner_average_precision"] for candidate in candidates}) == 1


def test_nested_fits_and_selection_never_include_outer_validation(development, monkeypatch):
    features = biomarker_features(development["observed"])
    labels, groups = development["labels"], development["groups"]
    expected = []
    for fold, (training, validation) in enumerate(patient_folds(labels, groups, 2, 42), 1):
        inner = list(patient_folds(labels[training], groups[training], 2, 42 + fold))
        for c in C_VALUES:
            for fitting, held in inner:
                expected.append((c, training[fitting], training[held]))
        expected.extend([(None, training, validation), (1.0, training, validation)])
    calls = []
    original_factory = mortality_comparison.make_logistic_classifier

    def checked_classifier(c, seed):
        model = original_factory(c, seed)
        original_fit, original_predict = model.fit, model.predict_proba
        expected_c, fitting, held = expected[len(calls)]
        if expected_c is not None:
            assert c == expected_c
        calls.append(c)

        def checked_fit(x, y):
            np.testing.assert_array_equal(x, features[fitting])
            np.testing.assert_array_equal(y, labels[fitting])
            result = original_fit(x, y)
            expected_medians = np.nanmedian(features[fitting], axis=0)
            np.testing.assert_allclose(model.named_steps["simpleimputer"].statistics_, expected_medians)
            return result

        def checked_predict(x):
            np.testing.assert_array_equal(x, features[held])
            assert not set(groups[fitting]) & set(groups[held])
            return original_predict(x)

        monkeypatch.setattr(model, "fit", checked_fit)
        monkeypatch.setattr(model, "predict_proba", checked_predict)
        return model

    monkeypatch.setattr(mortality_logistic, "make_logistic_classifier", checked_classifier)
    report = nested_logistic(development, outer_splits=2, inner_splits=2)
    assert len(calls) == len(expected)
    assert report["outer_oof_coverage"] == {"min": 1, "max": 1}
    for record in report["folds"]:
        best = max(record["inner_search"], key=lambda item: item["mean_inner_average_precision"])
        assert record["selected_c"] == best["c"]
        assert record["metrics"]["prevalence"]["auroc"] == 0.5
    for result in report["summary"].values():
        assert sum(item["admissions"] for item in result["calibration_bins"]) == len(labels)
        assert sum(item["deaths"] for item in result["calibration_bins"]) == int(labels.sum())


def test_nested_runner_is_deterministic_private_and_aggregate_only(development, tmp_path):
    path = tmp_path / "development.npz"
    np.savez_compressed(path, **development)
    output = tmp_path / "nested"
    report = run_nested_logistic(path, output, outer_splits=2, inner_splits=2)
    repeated = nested_logistic(development, outer_splits=2, inner_splits=2)
    assert report["summary"] == repeated["summary"]
    assert report["folds"] == repeated["folds"]
    assert report["test_accessed"] is False
    assert report["protocol"]["calibration"] == "assessment_only_no_recalibration"
    assert report["protocol"]["thresholds"] == "not_selected"
    assert set(p.name for p in output.iterdir()) == {"nested_logistic.json", "calibration.png"}
    assert json.loads((output / "nested_logistic.json").read_text()) == report
    assert (output / "calibration.png").read_bytes().startswith(b"\x89PNG")
    assert "probabilities" not in report and "labels" not in report and "groups" not in report
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())
    with pytest.raises(FileExistsError):
        run_nested_logistic(path, output)
    wrong = tmp_path / "full.npz"
    np.savez_compressed(wrong, test_x=np.ones(1))
    with pytest.raises(ValueError, match="development-only"):
        run_nested_logistic(wrong, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_operating_metrics_scalar_and_fold_thresholds_and_undefined_rates():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.6, 0.7])
    result = operating_metrics(labels, probabilities, 0.65)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["sensitivity"] == result["specificity"] == 0.5
    assert result["precision"] == result["negative_predictive_value"] == 0.5
    assert result["alert_count"] == 2 and result["alerts_per_100"] == 50
    perfect = operating_metrics(labels, probabilities, np.array([0.2, 0.9, 0.6, 0.7]))
    assert perfect["confusion_matrix"] == [[2, 0], [0, 2]]
    all_positive = operating_metrics(labels, probabilities, 0.0)
    assert all_positive["negative_predictive_value"] is None
    assert "negative_predictive_value" in all_positive["undefined_metrics"]
    all_negative = operating_metrics(labels, probabilities, 1.0)
    assert all_negative["precision"] is None
    assert "precision" in all_negative["undefined_metrics"]
    with pytest.raises(ValueError, match="aligned thresholds"):
        operating_metrics(labels, probabilities, np.array([0.1, 0.2]))


def test_youden_uses_inner_oof_only_and_final_cutoff_is_separate(development, monkeypatch):
    features = biomarker_features(development["observed"])
    labels, groups = development["labels"], development["groups"]
    expected, expected_selection = [], []
    outer_probabilities = np.full(len(labels), np.nan)
    for fold, (training, validation) in enumerate(patient_folds(labels, groups, 2, 42), 1):
        expected_selection.append(training)
        for fitting, held in patient_folds(labels[training], groups[training], 2, 42 + fold):
            expected.append((training[fitting], training[held], False))
        expected.append((training, validation, True))
    fits, inner_predictions, threshold_calls = [], [], []
    original_crossfit = mortality_operating_point.cross_fitted_probabilities
    original_factory = mortality_comparison.make_logistic_classifier

    def checked_factory(*, seed):
        model = original_factory(seed=seed)
        original_fit, original_predict = model.fit, model.predict_proba
        fitting, held, outer = expected[len(fits)]
        fits.append(fitting)

        def checked_fit(x, y):
            np.testing.assert_array_equal(x, features[fitting])
            np.testing.assert_array_equal(y, labels[fitting])
            return original_fit(x, y)

        def checked_predict(x):
            np.testing.assert_array_equal(x, features[held])
            assert not set(groups[fitting]) & set(groups[held])
            result = original_predict(x)
            if outer:
                outer_probabilities[held] = result[:, 1]
            return result

        monkeypatch.setattr(model, "fit", checked_fit)
        monkeypatch.setattr(model, "predict_proba", checked_predict)
        return model

    def checked_crossfit(x, y, patient_groups, **kwargs):
        training = expected_selection[len(inner_predictions)]
        np.testing.assert_array_equal(x, features[training])
        np.testing.assert_array_equal(y, labels[training])
        np.testing.assert_array_equal(patient_groups, groups[training])
        result = original_crossfit(x, y, patient_groups, **kwargs)
        inner_predictions.append(result)
        return result

    def checked_threshold(y, probabilities):
        index = len(threshold_calls)
        if index < 2:
            np.testing.assert_array_equal(y, labels[expected_selection[index]])
            np.testing.assert_array_equal(probabilities, inner_predictions[index])
        else:
            np.testing.assert_array_equal(y, labels)
            np.testing.assert_array_equal(probabilities, outer_probabilities)
        threshold_calls.append(index)
        return youden_threshold(y, probabilities)

    monkeypatch.setattr(mortality_operating_point, "make_logistic_classifier", checked_factory)
    monkeypatch.setattr(mortality_operating_point, "cross_fitted_probabilities", checked_crossfit)
    monkeypatch.setattr(mortality_operating_point, "youden_threshold", checked_threshold)
    report, final_threshold = evaluate_youden(development, outer_splits=2, inner_splits=2)
    assert len(fits) == len(expected) and len(threshold_calls) == 3
    assert final_threshold == youden_threshold(labels, outer_probabilities)
    assert report["outer_oof_coverage"] == {"min": 1, "max": 1}
    expected_matrix = sum(np.array(record["metrics"]["confusion_matrix"]) for record in report["folds"])
    np.testing.assert_array_equal(report["pooled_outer_metrics"]["confusion_matrix"], expected_matrix)
    assert report["protocol"]["frozen_model_performance"] == "not_estimated_no_independent_validation"


def test_frozen_pipeline_persists_full_fit_and_separate_evaluation(development, tmp_path):
    import hashlib
    import joblib

    path = tmp_path / "development.npz"
    np.savez_compressed(path, **development)
    output = tmp_path / "frozen"
    report = run_operating_point(path, output, outer_splits=2, inner_splits=2)
    spec = json.loads((output / "model_spec.json").read_text())
    model = joblib.load(output / "model.joblib")
    reference = mortality_comparison.make_logistic_classifier().fit(
        biomarker_features(development["observed"]), development["labels"],
    )
    probabilities, alerts = predict_observed(model, development["observed"], spec["threshold"])
    np.testing.assert_array_equal(
        probabilities, reference.predict_proba(biomarker_features(development["observed"]))[:, 1],
    )
    np.testing.assert_array_equal(alerts, probabilities >= report["frozen_threshold"])
    np.testing.assert_array_equal(
        spec["coefficients_after_imputation_and_standardization"],
        model.named_steps["logisticregression"].coef_,
    )
    assert len(spec["transformed_features"]) == model.named_steps["logisticregression"].coef_.shape[1]
    assert spec["model_sha256"] == hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    assert spec["training_admissions"] == 48
    assert spec["independently_validated"] is False
    assert report["test_accessed"] is False
    assert json.loads((output / "evaluation.json").read_text()) == report
    assert set(p.name for p in output.iterdir()) == {"model.joblib", "model_spec.json", "evaluation.json"}
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())
    assert not {"labels", "groups", "probabilities", "predictions"} & set(report)
    with pytest.raises(FileExistsError):
        run_operating_point(path, output)
    wrong = tmp_path / "wrong.npz"
    np.savez_compressed(wrong, test_x=np.ones(1))
    with pytest.raises(ValueError, match="development-only"):
        run_operating_point(wrong, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_frozen_prediction_rejects_scaled_inputs_empty_observations_and_bad_thresholds(development):
    model = mortality_comparison.make_logistic_classifier().fit(
        biomarker_features(development["observed"]), development["labels"],
    )
    for invalid in (np.ones((1, 10, 6)), np.full((1, 10, 3), np.nan),
                    np.full((1, 10, 3), np.inf)):
        with pytest.raises(ValueError, match="physical values"):
            predict_observed(model, invalid, 0.1)
    with pytest.raises(ValueError, match="threshold"):
        predict_observed(model, development["observed"], 1.1)
