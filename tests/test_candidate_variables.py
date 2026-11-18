"""Development-only candidate extraction, timing, unit and coverage safeguards."""

import json

import numpy as np
import pandas as pd
import pytest

from database.postgres import audit_candidate_variables as audit


@pytest.fixture
def cohort():
    return pd.DataFrame({
        "patient_id": ["p0", "p1"], "admission_id": ["a0", "a1"], "label": [0, 1],
    })


def event_frame(records):
    start = pd.Timestamp("2020-01-01")
    return pd.DataFrame([{
        "row_id": str(index + 1), "admission_id": admission, "itemid": item,
        "event_time": start + pd.Timedelta(hours=hour), "admission_time": start,
        "numeric_text": number, "raw_text": raw if raw is not None else number,
        "unit": unit, "available_time": start + pd.Timedelta(hours=available)
        if available is not None else pd.NaT,
    } for index, (admission, item, hour, number, unit, available, raw) in enumerate(records)])


def test_local_loader_reads_only_development_keys_and_preserves_cohort_order(tmp_path, monkeypatch):
    arrays = {}
    for split in ("train", "validation"):
        arrays[f"{split}_patient_id"] = np.array([split + "-p"])
        arrays[f"{split}_admission_id"] = np.array([split + "-a"])
        arrays[f"{split}_y"] = np.array([0 if split == "train" else 1])
        arrays[f"{split}_x"] = np.ones((1, 10, 6), dtype=np.float32)
    arrays["test_x"] = np.array(["do-not-read"])
    path = tmp_path / "audit.npz"
    np.savez_compressed(path, **arrays)
    original = np.lib.npyio.NpzFile.__getitem__
    reads = []

    def checked(self, key):
        assert not key.startswith("test")
        reads.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", checked)
    selected, digest, masks = audit.load_development_ids(path)
    assert selected["admission_id"].tolist() == ["train-a", "validation-a"]
    assert selected["label"].tolist() == [0, 1]
    assert masks.shape == (2, 10, 3) and masks.all()
    assert len(digest) == 64 and len(reads) == 8
    arrays["validation_patient_id"] = arrays["train_patient_id"]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="overlap"):
        audit.load_development_ids(path)


def test_quality_checks_units_censoring_and_last_valid_value(cohort):
    events = audit.prepare_events(event_frame([
        ("a0", "5626", 0, "1", "mg/L", None, None),
        ("a0", "5626", 0, "2", "mg/L", None, None),
        ("a0", "5626", 1, "999", "mg/dL", None, None),
        ("a0", "5626", 3, "0.5", "mg/L", None, "<0.5"),
        ("a1", "5626", 4, "bad", "mg/L", None, None),
        ("a1", "5626", 5, "3", "mg/L", None, None),
    ]), cohort)
    candidate = next(item for item in audit.CANDIDATES if item.name == "crp")
    profile, values = audit.profile_candidate(events, cohort, candidate)
    assert values[0, 0] == 2
    assert np.isnan(values[0, 1])
    assert values[1, 2] == 3
    assert profile["unit_rejected_events"] == 1
    assert profile["invalid_numeric_events"] == 1
    assert profile["censored_events"] == 1
    assert profile["auditable_valid_events"] == 3
    assert profile["admissions_any"] == 2
    assert profile["admissions_repeated"] == 0
    assert profile["event_time_proxy_only"] is True
    assert profile["temporal"]["input_origins"]["observed_cells"] == 2
    json.dumps(profile, allow_nan=False)


def test_availability_is_enforced_and_dimensionless_exceptions_are_code_specific(cohort):
    events = audit.prepare_events(event_frame([
        ("a0", "1003", 0, "100", "bpm", 1, None),
        ("a0", "1003", 3, "110", "bpm", 25, None),
        ("a1", "1003", 4, "120", "bpm", None, None),
        ("a1", "1003", 5, "130", "bpm", 4, None),
        ("a1", "1003", 6, "140", "", 6, None),
    ]), cohort)
    heart = next(item for item in audit.CANDIDATES if item.name == "heart_rate")
    profile, values = audit.profile_candidate(events, cohort, heart)
    assert profile["nominal_valid_events"] == 4 and profile["auditable_valid_events"] == 1
    assert profile["available_time_missing"] == 1
    assert profile["available_time_before_event"] == 1
    assert profile["available_at_or_after_landmark"] == 1
    assert profile["event_time_proxy_only"] is False
    assert np.isfinite(values).sum() == 1
    assert profile["admissions_any"] == 1
    labs = audit.prepare_events(event_frame([
        ("a0", "5237", 0, "7.3", "", None, None),
        ("a1", "5227", 0, "3", "", None, None),
    ]), cohort)
    ph = next(item for item in audit.CANDIDATES if item.name == "ph_gas")
    lactate = next(item for item in audit.CANDIDATES if item.name == "lactate")
    assert audit.profile_candidate(labs, cohort, ph)[0]["admissions_any"] == 1
    assert audit.profile_candidate(labs, cohort, lactate)[0]["admissions_any"] == 0


def test_unit_conversion_is_explicit_and_specimen_conflict_not_erased(cohort):
    events = audit.prepare_events(event_frame([
        ("a0", "5257", 0, "10", "g/dL", None, None),
        ("a1", "5257", 0, "100", "g/L", None, None),
    ]), cohort)
    candidate = next(item for item in audit.CANDIDATES if item.name == "hemoglobin_gas")
    profile, values = audit.profile_candidate(events, cohort, candidate)
    np.testing.assert_array_equal(values[:, 0], [100, 100])
    assert profile["definition"]["factors"] == (1, 10)
    respiratory = next(item for item in audit.CANDIDATES if item.name == "respiratory_rate_unit_conflict")
    conflict = audit.prepare_events(event_frame([
        ("a0", "1004", 0, "40", "bpm", 0, None),
    ]), cohort)
    assert audit.profile_candidate(conflict, cohort, respiratory)[0]["unit_rejected_events"] == 1


def test_preparation_rejects_non_development_and_outside_window(cohort):
    for admission, hour in (("other", 0), ("a0", -1), ("a0", 24)):
        with pytest.raises(ValueError):
            audit.prepare_events(event_frame([
                (admission, "5227", hour, "1", "mmol/L", None, None),
            ]), cohort)


def test_catalogue_and_pairwise_counts_are_descriptive_not_model_selection(cohort):
    events = audit.prepare_events(event_frame([
        ("a0", "5227", 0, "1", "mmol/L", None, None),
        ("a0", "5227", 4, "2", "mmol/L", None, None),
        ("a1", "5227", 0, "3", "mmol/L", None, None),
    ]), cohort)
    dictionary = pd.DataFrame({"itemid": ["5227"], "label": ["Lactate"], "fluid": ["Blood"]})
    record = audit.catalogue_profile(events, dictionary, audit.LAB)[0]
    assert record["uncensored_numeric_admissions"] == 2
    assert record["uncensored_numeric_occupied_bins"] == 3
    assert record["admissions_with_repeated_numeric_bins"] == 1
    first = np.arange(40, dtype=float).reshape(4, 10)
    matrices = {"first": first, "second": first * 2, "constant": np.ones((4, 10))}
    pairs = audit.pairwise_redundancy(matrices)
    assert pairs[0]["jointly_observed_bin_cells"] == 40
    assert pairs[0]["spearman_same_admission_bin"] == pytest.approx(1)
    assert pairs[0]["observed_mask_jaccard"] == 1
    assert pairs[0]["fraction_numerically_equal_ignoring_unit_semantics"] == pytest.approx(1 / 40)
    assert pairs[1]["spearman_same_admission_bin"] is None
    assert pairs[1]["reason_if_no_correlation"]
    coverage = audit.panel_coverage(matrices, ["first", "second"])
    assert coverage["observed_fraction"] == 1
    assert coverage["admissions_all_variables_repeated"] == 4
    assert coverage["occupied_time_bins_histogram"][-1] == 4


def test_source_queries_always_restrict_admissions_and_patients():
    for source, settings in {**audit.EVENT_SOURCES, **audit.OTHER_SOURCES}.items():
        time = settings[0]
        available = settings[-1] if source in audit.EVENT_SOURCES else settings[1]
        _, window, summary, parameters = audit.source_queries(source, time, available, ["a"], ["p"])
        assert parameters == (["a"], ["p"])
        text = summary.as_string()
        assert "e.subject_id = a.subject_id AND e.hadm_id = a.hadm_id" in text
        assert "a.hadm_id = ANY(%s) AND a.subject_id = ANY(%s)" in text
        assert "interval '24 hours'" in window.as_string()
        assert "test" not in text


def test_extreme_values_flagged_without_hidden_filtering_and_exact_intervals(cohort):
    events = audit.prepare_events(event_frame([
        ("a0", "5230", 0, "135", "mmol/L", None, None),
        ("a0", "5230", 0.1, "136", "mmol/L", None, None),
        ("a0", "5230", 3, "21845639168", "mmol/L", None, None),
        ("a1", "5230", 0, "140", "mmol/L", None, None),
    ]), cohort)
    sodium = next(item for item in audit.CANDIDATES if item.name == "sodium_gas")
    result, values = audit.profile_candidate(events, cohort, sodium)
    assert result["quality_review_flags"]["events_outside_review_bounds"] == 1
    assert result["quality_review_flags"]["last_bin_values_outside_review_bounds"] == 1
    assert values[0, 1] == 21845639168
    assert result["auditable_valid_events"] == 4
    assert result["admission_bins_with_multiple_usable_events"] == 1
    assert result["exact_successive_event_intervals_hours"]["quantiles_min_p01_p50_p99_max"][2] == 1.5
