"""Prepare patient-disjoint, 24-hour NICU biomarker sequences."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


MARKERS = ("lactate", "total_bilirubin", "creatinine")
ITEMS = {
    "5227": ("lactate", "mmol/L"),
    "5075": ("total_bilirubin", "μmol/L"),
    "5255": ("total_bilirubin", "μmol/L"),
    "5032": ("creatinine", "μmol/L"),
    "5041": ("creatinine", "μmol/L"),
}
STEPS = 10
BIN_SECONDS = 8640


@dataclass
class MortalityDataset:
    arrays: dict[str, np.ndarray]
    summary: dict


def build_mortality_dataset(cohort: pd.DataFrame, events: pd.DataFrame,
                            seed: int = 42) -> MortalityDataset:
    """Build 10-bin sequences; fit imputation and scaling on training subjects only.

    Cohort and event frames must already be restricted to the selected 24-hour
    landmark. An observation mask identifies measured bins before forward-fill.
    """
    cohort_columns = {"patient_id", "admission_id", "admission_time",
                      "hospital_death", "birth_after_admission"}
    event_columns = {"admission_id", "row_id", "charttime", "itemid", "valuenum", "valueuom"}
    if cohort_columns - set(cohort) or event_columns - set(events):
        raise ValueError("Missing required cohort or lab-event columns.")
    cohort = cohort.copy()
    events = events.copy()
    if cohort.empty or cohort[["patient_id", "admission_id"]].isna().any().any():
        raise ValueError("Cohort must contain admissions with patient and admission IDs.")
    if cohort["admission_id"].duplicated().any():
        raise ValueError("Cohort admission IDs must be unique.")
    if not cohort["hospital_death"].isin([0, 1]).all():
        raise ValueError("Hospital mortality must be binary and nonmissing.")
    if cohort["birth_after_admission"].isna().any():
        raise ValueError("Birth-after-admission flags must be nonmissing.")
    cohort["admission_time"] = pd.to_datetime(cohort["admission_time"], errors="raise")
    if cohort["admission_time"].isna().any():
        raise ValueError("Admission times must be nonmissing.")
    cohort = cohort.sort_values("admission_id").reset_index(drop=True)
    if events["row_id"].isna().any() or events["row_id"].duplicated().any():
        raise ValueError("Lab-event row IDs must be unique and nonmissing.")
    if not events["itemid"].isin(ITEMS).all():
        raise ValueError("Unexpected biomarker item ID.")
    events["charttime"] = pd.to_datetime(events["charttime"], errors="raise")
    if events["charttime"].isna().any():
        raise ValueError("Lab-event times must be nonmissing.")
    events = events.merge(cohort[["admission_id", "admission_time"]],
                          on="admission_id", how="left", validate="many_to_one")
    if events["admission_time"].isna().any():
        raise ValueError("Lab event does not belong to an eligible admission.")
    elapsed = (events["charttime"] - events["admission_time"]).dt.total_seconds()
    if ((elapsed < 0) | (elapsed >= 24 * 3600)).any():
        raise ValueError("Lab event is outside the 24-hour observation window.")
    events["bin"] = (elapsed // BIN_SECONDS).astype(int)
    events["marker"] = events["itemid"].map(lambda item: ITEMS[item][0])
    expected_unit = events["itemid"].map(lambda item: ITEMS[item][1])
    unit_ok = events["valueuom"].eq(expected_unit).fillna(False)
    numeric = pd.to_numeric(events["valuenum"], errors="coerce")
    numeric_ok = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=events.index)
    rejected_units = int((~unit_ok).sum())
    rejected_values = int((~numeric_ok).sum())
    events = events.loc[unit_ok & numeric_ok].copy()
    events["value"] = numeric.loc[events.index].astype(float)
    events["row_id"] = pd.to_numeric(events["row_id"], errors="raise")
    events = events.sort_values(["charttime", "row_id"]).drop_duplicates(
        ["admission_id", "bin", "marker"], keep="last"
    )

    admission_positions = {admission: pos for pos, admission in enumerate(cohort["admission_id"])}
    marker_positions = {marker: pos for pos, marker in enumerate(MARKERS)}
    observed = np.full((len(cohort), STEPS, len(MARKERS)), np.nan, dtype=float)
    for row in events.itertuples(index=False):
        observed[admission_positions[row.admission_id], row.bin, marker_positions[row.marker]] = row.value
    has_observation = np.isfinite(observed).any(axis=(1, 2))
    excluded_empty = int((~has_observation).sum())
    cohort = cohort.loc[has_observation].reset_index(drop=True)
    observed = observed[has_observation]
    if cohort.empty:
        raise ValueError("No admissions have a usable candidate biomarker.")

    patient_outcomes = cohort.groupby("patient_id", sort=True)["hospital_death"].max()
    try:
        train_ids, held_ids = train_test_split(
            patient_outcomes.index.to_numpy(), test_size=0.30,
            stratify=patient_outcomes.to_numpy(), random_state=seed,
        )
        val_ids, test_ids = train_test_split(
            held_ids, test_size=0.50,
            stratify=patient_outcomes.loc[held_ids].to_numpy(), random_state=seed,
        )
    except ValueError as exc:
        raise ValueError("Cannot create stratified patient-disjoint 70/15/15 splits.") from exc
    split_ids = {"train": train_ids, "validation": val_ids, "test": test_ids}
    train_select = cohort["patient_id"].isin(train_ids).to_numpy()
    training_observed = observed[train_select]
    if not np.isfinite(training_observed).any(axis=(0, 1)).all():
        raise ValueError("Every biomarker must be observed in the training split.")
    medians = np.nanmedian(training_observed, axis=(0, 1))
    masks = np.isfinite(observed)
    filled = observed.copy()
    for step in range(1, STEPS):
        filled[:, step, :] = np.where(
            np.isfinite(filled[:, step, :]), filled[:, step, :], filled[:, step - 1, :]
        )
    filled = np.where(np.isfinite(filled), filled, medians)
    scaler = MinMaxScaler().fit(filled[train_select].reshape(-1, len(MARKERS)))
    scaled = scaler.transform(filled.reshape(-1, len(MARKERS))).reshape(filled.shape)
    sequences = np.concatenate((scaled, masks.astype(float)), axis=2).astype(np.float32)
    if not np.isfinite(sequences).all():
        raise ValueError("Nonfinite values remain in prepared sequences.")

    arrays = {}
    splits = {}
    for name, ids in split_ids.items():
        selected = cohort["patient_id"].isin(ids).to_numpy()
        arrays[f"{name}_x"] = sequences[selected]
        arrays[f"{name}_y"] = cohort.loc[selected, "hospital_death"].to_numpy(dtype=np.int8)
        arrays[f"{name}_patient_id"] = cohort.loc[selected, "patient_id"].to_numpy(dtype=str)
        arrays[f"{name}_admission_id"] = cohort.loc[selected, "admission_id"].to_numpy(dtype=str)
        arrays[f"{name}_birth_after_admission"] = cohort.loc[
            selected, "birth_after_admission"
        ].to_numpy(dtype=bool)
        splits[name] = {
            "admissions": int(selected.sum()),
            "patients": int(len(ids)),
            "hospital_deaths": int(arrays[f"{name}_y"].sum()),
        }
    summary = {
        "seed": seed,
        "markers": list(MARKERS),
        "features": [*MARKERS, *(f"{marker}_observed" for marker in MARKERS)],
        "sequence_length": STEPS,
        "bin_seconds": BIN_SECONDS,
        "admissions_without_usable_markers_excluded": excluded_empty,
        "events_with_unverified_units_excluded": rejected_units,
        "events_with_missing_or_invalid_values_excluded": rejected_values,
        "rejection_counts_may_overlap": True,
        "birth_after_admission_in_included_cohort": int(cohort["birth_after_admission"].sum()),
        "training_observation_medians": medians.tolist(),
        "training_scaler_data_min": scaler.data_min_.tolist(),
        "training_scaler_data_max": scaler.data_max_.tolist(),
        "splits": splits,
    }
    return MortalityDataset(arrays=arrays, summary=summary)
