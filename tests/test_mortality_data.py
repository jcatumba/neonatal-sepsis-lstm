"""Focused tests for the PICDB mortality dataset preparation."""

import io

import numpy as np
import pandas as pd
import pytest

from src.mortality_data import build_mortality_dataset


def example_frames():
    cohort, events = [], []
    start = pd.Timestamp("2020-01-01")
    for patient in range(40):
        admission = f"a{patient:02}"
        cohort.append({
            "patient_id": f"p{patient:02}", "admission_id": admission,
            "admission_time": start, "hospital_death": patient % 2,
            "birth_after_admission": patient == 0,
        })
        for offset, itemid, value, unit in (
            (60, "5227", 1.0 + patient, "mmol/L"),
            (360, "5075", 2.0 + patient, "μmol/L"),
            (720, "5041", 3.0 + patient, "μmol/L"),
        ):
            events.append({
                "admission_id": admission, "row_id": len(events) + 1,
                "charttime": start + pd.Timedelta(minutes=offset),
                "itemid": itemid, "valuenum": value, "valueuom": unit,
            })
    return pd.DataFrame(cohort), pd.DataFrame(events)


def test_patient_disjoint_splits_and_training_only_preprocessing():
    cohort, events = example_frames()
    extra = cohort.iloc[0].copy()
    extra["admission_id"] = "repeat"
    cohort = pd.concat([cohort, extra.to_frame().T], ignore_index=True)
    repeat_event = events.iloc[0].copy()
    repeat_event["admission_id"] = "repeat"
    repeat_event["row_id"] = 10000
    events = pd.concat([events, repeat_event.to_frame().T], ignore_index=True)
    first = build_mortality_dataset(cohort, events)
    groups = [set(first.arrays[f"{name}_patient_id"]) for name in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sum("p00" in group for group in groups) == 1
    assert all(first.arrays[f"{name}_x"].shape[1:] == (10, 6)
               for name in ("train", "validation", "test"))
    assert first.summary["birth_after_admission_in_included_cohort"] == 2
    changed = events.copy()
    heldout_admissions = set(first.arrays["test_admission_id"])
    changed.loc[changed["admission_id"].isin(heldout_admissions), "valuenum"] = 1e6
    second = build_mortality_dataset(cohort, changed)
    np.testing.assert_array_equal(first.arrays["train_x"], second.arrays["train_x"])
    assert first.summary["training_observation_medians"] == second.summary["training_observation_medians"]
    assert first.summary["training_scaler_data_max"] == second.summary["training_scaler_data_max"]
    serialized = io.BytesIO()
    np.savez_compressed(serialized, **first.arrays)
    serialized.seek(0)
    with np.load(serialized, allow_pickle=False) as archive:
        for key in first.arrays:
            np.testing.assert_array_equal(archive[key], first.arrays[key])


def test_last_value_masks_forward_fill_and_invalid_units():
    cohort, events = example_frames()
    target = events.iloc[0].copy()
    target["row_id"] = 10000
    target["charttime"] += pd.Timedelta(minutes=1)
    target["valuenum"] = 500
    invalid = target.copy()
    invalid["row_id"] = 10001
    invalid["charttime"] += pd.Timedelta(minutes=1)
    invalid["valuenum"] = 9999
    invalid["valueuom"] = None
    events = pd.concat([events, target.to_frame().T, invalid.to_frame().T], ignore_index=True)
    dataset = build_mortality_dataset(cohort, events)
    assert dataset.summary["events_with_unverified_units_excluded"] == 1
    for name in ("train", "validation", "test"):
        indices = np.flatnonzero(dataset.arrays[f"{name}_patient_id"] == "p00")
        if len(indices):
            seq = dataset.arrays[f"{name}_x"][indices[0]]
            assert seq[0, 3] == 1
            assert seq[1, 3] == 0
            assert seq[0, 0] == seq[1, 0]
            min_lactate = dataset.summary["training_scaler_data_min"][0]
            max_lactate = dataset.summary["training_scaler_data_max"][0]
            assert seq[0, 0] == pytest.approx((500 - min_lactate) / (max_lactate - min_lactate))
            min_bilirubin = dataset.summary["training_scaler_data_min"][1]
            max_bilirubin = dataset.summary["training_scaler_data_max"][1]
            median_bilirubin = dataset.summary["training_observation_medians"][1]
            assert seq[0, 4] == 0
            assert seq[2, 4] == 1
            assert seq[0, 1] == pytest.approx(
                (median_bilirubin - min_bilirubin) / (max_bilirubin - min_bilirubin)
            )
            assert seq[1, 1] == seq[0, 1]
            break
    else:
        pytest.fail("Patient was not assigned to a split.")


def test_empty_admission_excluded_and_landmark_enforced():
    cohort, events = example_frames()
    events = events[events["admission_id"] != "a00"].copy()
    dataset = build_mortality_dataset(cohort, events)
    assert dataset.summary["admissions_without_usable_markers_excluded"] == 1
    bad = events.copy()
    bad.loc[bad.index[0], "charttime"] = pd.Timestamp("2020-01-02")
    with pytest.raises(ValueError, match="outside"):
        build_mortality_dataset(cohort, bad)
