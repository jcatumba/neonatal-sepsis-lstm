"""Exact structural counts for the development-only temporal audit."""

import hashlib
import json

import numpy as np
import pytest

from src.mortality_temporal_audit import run_temporal_audit, temporal_audit


@pytest.fixture
def binned_data():
    observed = np.full((4, 10, 3), np.nan)
    observed[0, [1, 3], 0] = [1, 3]
    observed[0, [0, 1], 1] = [5, 5]
    observed[1, 0, 0] = 2
    observed[1, 9, 2] = 10
    observed[2, [8, 9], 1] = [8, 12]
    observed[2, [5, 7], 2] = [6, 6]
    observed[3, [2, 3], 0] = [4, 4 + 1e-7]
    return {
        "observed": observed, "labels": np.array([0, 0, 1, 1]),
        "groups": np.array([0, 1, 0, 2]),
    }


def test_origin_counts_and_temporal_repetition_are_not_imputed_observations(binned_data):
    report = temporal_audit(binned_data)
    overall = report["populations"]["overall"]
    origins = overall["input_origins"]
    assert origins["total_cells"] == 120
    assert origins["observed_cells"] == 12
    assert origins["forward_fill_cells"] == 33
    assert origins["initial_median_cells"] == 75
    assert origins["observed_fraction"] == 0.1
    assert overall["admissions_with_no_marker_repeated"] == 1
    assert overall["admissions_with_repeated_markers_at_least"] == {"1": 3, "2": 2, "3": 0}
    assert overall["admissions_with_varying_markers_at_least"] == {"1": 2, "2": 0, "3": 0}
    assert overall["admissions_with_adjacent_observations_of_any_marker"] == 3
    assert overall["occupied_time_bins_per_admission_histogram"][:5] == [0, 0, 2, 1, 1]
    assert overall["observed_cells_per_admission"]["quantiles"]["median"] == 3
    for population in report["populations"].values():
        counts = population["input_origins"]
        assert counts["total_cells"] == sum(counts[key] for key in
                                           ("observed_cells", "forward_fill_cells", "initial_median_cells"))
        for key in ("total_cells", "observed_cells", "forward_fill_cells", "initial_median_cells"):
            assert counts[key] == sum(marker["input_origins"][key]
                                      for marker in population["markers"].values())


def test_variation_tolerance_and_bin_intervals_are_explicit(binned_data):
    report = temporal_audit(binned_data)
    lactate = report["populations"]["overall"]["markers"]["lactate"]
    assert lactate["occupied_bin_count_histogram"][:3] == [1, 1, 2]
    assert lactate["admissions_with_repeated_bins"] == 2
    assert lactate["admissions_with_numerically_varying_measurements"] == 1
    assert lactate["admissions_with_constant_repeated_measurements"] == 1
    assert lactate["adjacent_observed_bin_pairs"] == 1
    assert lactate["occupied_bin_separation_histogram"] == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert lactate["occupied_bin_separation_hours"]["quantiles"]["median"] == pytest.approx(3.6)
    assert lactate["forward_fill_bin_age_hours"]["n"] == 22
    assert lactate["forward_fill_bin_age_hours"]["quantiles"]["max"] == pytest.approx(21.6)
    assert report["definitions"]["variation_meaning"] == "numerical_not_a_clinical_change_threshold"
    empty_pairs = report["populations"]["survival_admissions"]["markers"]["creatinine"]
    assert empty_pairs["occupied_bin_separation_hours"]["n"] == 0
    assert empty_pairs["occupied_bin_separation_hours"]["quantiles"] is None
    assert empty_pairs["occupied_bin_separation_hours"]["reason"]


def test_strata_are_admission_level_and_patients_need_not_be_additive(binned_data):
    report = temporal_audit(binned_data)
    overall, survival, death = (report["populations"][key] for key in
                                ("overall", "survival_admissions", "death_admissions"))
    assert report["admissions"] == 4 and report["patients"] == 3 and report["deaths"] == 2
    assert survival["admissions"] + death["admissions"] == overall["admissions"]
    assert survival["patients"] + death["patients"] > overall["patients"]
    for key in ("total_cells", "observed_cells", "forward_fill_cells", "initial_median_cells"):
        assert overall["input_origins"][key] == (
            survival["input_origins"][key] + death["input_origins"][key])


def test_audit_runner_is_read_only_private_and_rejects_test_exchanges(binned_data, tmp_path, monkeypatch):
    path = tmp_path / "development.npz"
    np.savez_compressed(path, **binned_data)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    original_get = np.lib.npyio.NpzFile.__getitem__

    def guarded_read(self, key):
        assert not key.startswith("test")
        return original_get(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded_read)
    output = tmp_path / "audit"
    report = run_temporal_audit(path, output)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash == report["input_sha256"]
    assert json.loads((output / "temporal_audit.json").read_text()) == report
    assert report["test_accessed"] is False and report["training_performed"] is False
    assert set(p.name for p in output.iterdir()) == {"temporal_audit.json", "temporal_coverage.png"}
    assert (output / "temporal_coverage.png").read_bytes().startswith(b"\x89PNG")
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())
    assert not {"labels", "groups", "observed", "predictions"} & set(report)
    with pytest.raises(FileExistsError):
        run_temporal_audit(path, output)
    wrong = tmp_path / "wrong.npz"
    np.savez_compressed(wrong, test_x=np.ones(1))
    with pytest.raises(ValueError, match="development-only"):
        run_temporal_audit(wrong, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
