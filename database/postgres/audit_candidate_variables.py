"""Read-only, development-only audit of expanded PICDB mortality predictors."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from psycopg import sql

from database.postgres.build_mortality_dataset import ENV_FILE, ROOT, fetch_frame
from database.postgres.load_dataset import EXPECTED_TABLES, connect
from src.mortality_data import BIN_SECONDS, STEPS
from src.mortality_temporal_audit import marker_diagnostics
from src.utils import save_json


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    items: tuple[str, ...]
    units: tuple[str, ...]
    family: str
    caveat: str = ""
    factors: tuple[float, ...] = ()


LAB = "labevents"
CHART = "chartevents"
CANDIDATES = (
    Candidate("lactate", LAB, ("5227",), ("mmol/L",), "baseline"),
    Candidate("total_bilirubin", LAB, ("5075", "5255"), ("μmol/L",), "baseline"),
    Candidate("creatinine", LAB, ("5032", "5041"), ("μmol/L",), "baseline"),
    Candidate("crp", LAB, ("5626", "5821"), ("mg/L",), "inflammation", "Censored values need a separate policy."),
    Candidate("procalcitonin", LAB, ("6085",), ("ng/ml",), "inflammation"),
    Candidate("platelets", LAB, ("5129",), ("×10^9/L",), "hematology"),
    Candidate("wbc", LAB, ("5141",), ("×10^9/L",), "hematology"),
    Candidate("neutrophils_count", LAB, ("5094",), ("×10^9/L",), "hematology"),
    Candidate("neutrophils_percent", LAB, ("5095",), ("%",), "hematology"),
    Candidate("lymphocytes_count", LAB, ("5110",), ("×10^9/L",), "hematology"),
    Candidate("lymphocytes_percent", LAB, ("5111",), ("%",), "hematology"),
    Candidate("hemoglobin_cbc", LAB, ("5099",), ("g/L",), "hematology"),
    Candidate("hematocrit_cbc", LAB, ("5097",), ("%",), "hematology"),
    Candidate("hemoglobin_gas", LAB, ("5257",), ("g/L", "g/dL"), "blood_gas",
              "g/dL explicitly multiplied by 10; do not silently merge with CBC.", (1, 10)),
    Candidate("hematocrit_gas", LAB, ("5225",), ("%",), "blood_gas"),
    Candidate("glucose_gas", LAB, ("5223",), ("mmol/L",), "blood_gas"),
    Candidate("sodium_gas", LAB, ("5230",), ("mmol/L",), "blood_gas"),
    Candidate("chloride_gas", LAB, ("5218",), ("mmol/L",), "blood_gas"),
    Candidate("potassium_ambiguous_specimen", LAB, ("5226",), ("mmol/L",), "blood_gas",
              "Dictionary fluid is Other Body Fluid; do not treat as blood potassium without verification."),
    Candidate("calcium_gas_label_total", LAB, ("5215",), ("mmol/L",), "blood_gas",
              "Dictionary says Total; do not relabel as ionized calcium."),
    Candidate("ph_gas", LAB, ("5237",), ("",), "blood_gas",
              "Explicit dimensionless-pH exception; absent unit allowed for this code only."),
    Candidate("ph_gas_alternate", LAB, ("5238",), ("",), "blood_gas",
              "Dimensionless-pH exception; alternate code kept separate."),
    Candidate("pco2_gas", LAB, ("5235",), ("mmHg",), "blood_gas",
              "Arterial/venous/capillary provenance not established."),
    Candidate("pco2_gas_alternate", LAB, ("5236",), ("mmHg",), "blood_gas",
              "Alternate code and specimen provenance need verification."),
    Candidate("po2_gas", LAB, ("5239",), ("mmHg",), "blood_gas",
              "Arterial/venous/capillary provenance not established; not equivalent to pulse SpO2."),
    Candidate("po2_gas_alternate", LAB, ("5244",), ("mmHg",), "blood_gas",
              "Alternate code and specimen provenance need verification."),
    Candidate("bicarbonate_calculated", LAB, ("5224",), ("mmol/L",), "blood_gas",
              "Derived measure; audit redundancy before selecting with pH/pCO2."),
    Candidate("bicarbonate", LAB, ("5248",), ("mmol/L",), "blood_gas",
              "Keep separate from calculated bicarbonate until mapping verified."),
    Candidate("base_excess", LAB, ("5211",), ("mmol/L",), "blood_gas"),
    Candidate("base_excess_alternate", LAB, ("5249",), ("mmol/L",), "blood_gas"),
    Candidate("anion_gap", LAB, ("5212",), ("mmol/L",), "blood_gas"),
    Candidate("oxygen_saturation_gas", LAB, ("5252",), ("%",), "blood_gas",
              "Blood gas oxygen saturation is not routine pulse oximetry."),
    Candidate("fio2_gas", LAB, ("5222",), ("%",), "treatment",
              "Treatment/context variable, not a pure biomarker."),
    Candidate("direct_bilirubin", LAB, ("5042",), ("μmol/L",), "chemistry"),
    Candidate("indirect_bilirubin", LAB, ("5055",), ("μmol/L",), "chemistry"),
    Candidate("albumin", LAB, ("5024",), ("g/L",), "chemistry"),
    Candidate("total_protein", LAB, ("5078",), ("g/L",), "chemistry"),
    Candidate("urea", LAB, ("5033",), ("mmol/L",), "chemistry"),
    Candidate("alt", LAB, ("5026",), ("U/L",), "chemistry"),
    Candidate("ast", LAB, ("5031",), ("U/L",), "chemistry"),
    Candidate("ldh", LAB, ("5057",), ("U/L",), "chemistry"),
    Candidate("sodium_chemistry", LAB, ("5062",), ("mmol/L",), "chemistry"),
    Candidate("calcium_chemistry", LAB, ("5034",), ("mmol/L",), "chemistry"),
    Candidate("magnesium", LAB, ("5059",), ("mmol/L",), "chemistry"),
    Candidate("phosphate", LAB, ("5072",), ("mmol/L",), "chemistry"),
    Candidate("inr", LAB, ("5174",), ("",), "coagulation",
              "Explicit dimensionless-INR exception, not a general unit default."),
    Candidate("pt", LAB, ("5186", "6890"), ("秒", "s"), "coagulation"),
    Candidate("aptt", LAB, ("5161", "6892"), ("秒", "s"), "coagulation"),
    Candidate("fibrinogen", LAB, ("5164",), ("g/L",), "coagulation"),
    Candidate("d_dimer", LAB, ("5163",), ("mg/L",), "coagulation"),
    Candidate("heart_rate", CHART, ("1003",), ("bpm",), "vitals"),
    Candidate("pulse", CHART, ("1002",), ("bpm",), "vitals", "Do not automatically merge with heart rate."),
    Candidate("respiratory_rate_unit_conflict", CHART, ("1004",), ("insp/min",), "vitals",
              "Source bpm conflicts with dictionary insp/min; conflicting rows rejected."),
    Candidate("temperature", CHART, ("1001",), ("°C",), "vitals"),
    Candidate("spo2", CHART, ("1006",), ("%",), "vitals"),
    Candidate("systolic_pressure", CHART, ("1016",), ("mmHg",), "vitals"),
    Candidate("diastolic_pressure", CHART, ("1015",), ("mmHg",), "vitals"),
    Candidate("chart_glucose", CHART, ("1011",), ("mmol/L",), "vitals"),
    Candidate("weight", CHART, ("1014",), ("kg",), "static_context",
              "Current charted weight, not verified birth weight."),
    Candidate("height", CHART, ("1013",), ("cm",), "static_context"),
    Candidate("chart_urine_volume", CHART, ("1008",), ("ml",), "balance",
              "Collection interval/cumulative semantics unknown; not mL/kg/hour."),
    Candidate("output_urine_volume", "outputevents", ("1034",), ("ml",), "balance",
              "Collection interval/cumulative semantics unknown; not mL/kg/hour."),
)

EVENT_SOURCES = {
    LAB: ("charttime", "valuenum", "value", "valueuom", None),
    CHART: ("charttime", "valuenum", "value", "valueuom", "storetime"),
    "outputevents": ("charttime", "value", "value", "valueuom", "storetime"),
    "surgery_vital_signs": ("monitor_time", "value", "value", None, None),
}
OTHER_SOURCES = {
    "inputevents": ("charttime", "storetime", "Volume only; no item ID or composition; interval semantics unverified."),
    "prescriptions": ("startdate", None, "Orders, not verified administrations; day-level start times and nonstandard doses."),
    "microbiologyevents": ("charttime", None, "Specimen time does not establish organism/susceptibility result availability."),
    "or_exam_reports": ("examtime", "reporttime", "Names/timing only; no structured numeric result values in this table."),
    "emr_symptoms": ("recordtime", None, "Text/categorical attributes require separate terminology and documentation-time audit."),
    "surgery_info": ("surgery_begin_time", None, "Procedure exposure, selected surgical subgroup; no future duration/end time as predictor."),
    "icustays": ("intime", None, "Care-location context only; exclude future outtime and length of stay."),
}

REVIEW_BOUNDS = {
    "sodium_gas": (50, 250), "chloride_gas": (30, 250),
    "heart_rate": (10, 500), "pulse": (10, 500),
    "spo2": (0, 100), "oxygen_saturation_gas": (0, 100),
}


def load_development_ids(path: Path) -> tuple[pd.DataFrame, str, np.ndarray]:
    parts, masks = [], []
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for split in ("train", "validation"):
            keys = ("patient_id", "admission_id", "y", "x")
            values = {key: archive[f"{split}_{key}"] for key in keys}
            ids, admissions, labels, x = (values[key] for key in keys)
            if (ids.shape != admissions.shape or ids.shape != labels.shape or ids.ndim != 1
                    or x.shape != (len(ids), STEPS, 6) or not np.isin(labels, [0, 1]).all()
                    or not np.isin(x[:, :, 3:], [0, 1]).all()
                    or np.any(ids == "") or np.any(admissions == "")):
                raise ValueError("Invalid aligned local development split.")
            parts.append(pd.DataFrame({"patient_id": ids.astype(str),
                                       "admission_id": admissions.astype(str), "label": labels}))
            masks.append(x[:, :, 3:].astype(bool))
            for key in keys:
                value = np.ascontiguousarray(values[key])
                digest.update(f"{split}_{key}:{value.dtype}:{value.shape}".encode())
                digest.update(value.tobytes())
    if set(parts[0]["patient_id"]) & set(parts[1]["patient_id"]):
        raise ValueError("Development split patient IDs overlap.")
    cohort = pd.concat(parts, ignore_index=True)
    if cohort["admission_id"].duplicated().any() or cohort.empty:
        raise ValueError("Development admissions must be unique and nonempty.")
    return cohort, digest.hexdigest(), np.concatenate(masks)


def timestamp_expression(column: str, alias: str = "e") -> sql.Composed:
    return sql.SQL("NULLIF({}.{}, '')::timestamp").format(sql.Identifier(alias), sql.Identifier(column))


def source_queries(table: str, time_column: str, available_column: str | None,
                   admissions: list[str], patients: list[str]):
    event_time = timestamp_expression(time_column)
    admission_time = timestamp_expression("admittime", "a")
    base = sql.SQL("""
        FROM picdb_v1_1_0.{} e JOIN picdb_v1_1_0.admissions a
          ON e.subject_id = a.subject_id AND e.hadm_id = a.hadm_id
        WHERE a.hadm_id = ANY(%s) AND a.subject_id = ANY(%s)
    """).format(sql.Identifier(table))
    window = sql.SQL(" AND {} >= {} AND {} < {} + interval '24 hours'").format(
        event_time, admission_time, event_time, admission_time)
    availability = (timestamp_expression(available_column) if available_column else sql.SQL("NULL::timestamp"))
    summary = sql.SQL("""
        SELECT count(*) AS linked_events,
          count(*) FILTER (WHERE {t} IS NULL) AS missing_event_time,
          count(*) FILTER (WHERE {t} < {a}) AS before_admission,
          count(*) FILTER (WHERE {t} >= {a} + interval '24 hours') AS at_or_after_landmark,
          count(*) FILTER (WHERE {t} >= {a} AND {t} < {a} + interval '24 hours') AS first_24h_events,
          count(DISTINCT a.hadm_id) FILTER (WHERE {t} >= {a} AND
            {t} < {a} + interval '24 hours') AS first_24h_admissions,
          count(*) FILTER (WHERE {t} >= {a} AND {t} < {a} + interval '24 hours'
            AND {s} >= {t} AND {s} < {a} + interval '24 hours') AS verified_available_events
    """).format(t=event_time, a=admission_time, s=availability) + base
    return base, window, summary, (admissions, patients)


def fetch_events(connection, table: str, admissions: list[str], patients: list[str]):
    time_column, numeric, raw, unit, available = EVENT_SOURCES[table]
    base, window, summary, params = source_queries(table, time_column, available, admissions, patients)
    timing = fetch_frame(connection, summary, params).iloc[0].to_dict()
    expression = lambda name: sql.SQL("e.{}").format(sql.Identifier(name))
    query = sql.SQL("""
        SELECT e.row_id, a.hadm_id AS admission_id, e.itemid,
          {t} AS event_time, {a} AS admission_time,
          {v} AS numeric_text, {raw} AS raw_text, {u} AS unit, {s} AS available_time
    """).format(
        t=timestamp_expression(time_column), a=timestamp_expression("admittime", "a"),
        v=expression(numeric), raw=expression(raw),
        u=expression(unit) if unit else sql.SQL("NULL::text"),
        s=timestamp_expression(available) if available else sql.SQL("NULL::timestamp"),
    ) + base + window
    return fetch_frame(connection, query, params), {key: int(value) for key, value in timing.items()}


def prepare_events(frame: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("Source event row IDs must be unique/nonmissing.")
    if not frame["admission_id"].isin(cohort["admission_id"]).all():
        raise ValueError("Refusing nondevelopment events.")
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise")
    frame["admission_time"] = pd.to_datetime(frame["admission_time"], errors="raise")
    elapsed = (frame["event_time"] - frame["admission_time"]).dt.total_seconds()
    if elapsed.isna().any() or ((elapsed < 0) | (elapsed >= 86400)).any():
        raise ValueError("Source events must lie in the original first-24h window.")
    frame["bin"] = (elapsed // BIN_SECONDS).astype(int)
    frame["number"] = pd.to_numeric(frame["numeric_text"], errors="coerce").astype(float)
    frame["finite"] = np.isfinite(frame["number"])
    frame["censored"] = (
        frame["numeric_text"].fillna("").astype(str).str.match(r"^\s*[<>≤≥]")
        | frame["raw_text"].fillna("").astype(str).str.match(r"^\s*[<>≤≥]")
    )
    frame["unit"] = frame["unit"].fillna("").astype(str).str.strip()
    frame["row_number"] = pd.to_numeric(frame["row_id"], errors="raise")
    frame["available_time"] = pd.to_datetime(frame["available_time"], errors="raise")
    frame["available_by_landmark"] = (
        (frame["available_time"] >= frame["event_time"])
        & (frame["available_time"] < frame["admission_time"] + pd.Timedelta(hours=24))
    )
    return frame


def value_summary(values: pd.Series | np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "finite_n": len(values),
        "quantiles_min_p01_p50_p99_max": np.quantile(values, [0, .01, .5, .99, 1]).tolist()
        if len(values) else None,
        "negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
    }


def binned_values(frame: pd.DataFrame, admissions: pd.Series) -> np.ndarray:
    values = np.full((len(admissions), STEPS), np.nan)
    if frame.empty:
        return values
    selected = frame.sort_values(["event_time", "row_number"]).drop_duplicates(
        ["admission_id", "bin"], keep="last")
    positions = pd.Index(admissions).get_indexer(selected["admission_id"])
    if (positions < 0).any():
        raise ValueError("Binned candidate contains a nondevelopment admission.")
    values[positions, selected["bin"].to_numpy()] = selected["number"]
    return values


def catalogue_profile(events: pd.DataFrame, dictionary: pd.DataFrame, source: str) -> list[dict]:
    if dictionary["itemid"].duplicated().any():
        raise ValueError("Dictionary item IDs must be unique.")
    metadata = dictionary.set_index("itemid").fillna("").to_dict(orient="index")
    records = []
    for (item, unit), group in events.groupby(["itemid", "unit"], sort=True):
        known = metadata.get(item, {})
        records.append({
            "itemid": item, "source_unit": unit, "events": len(group),
            "admissions_any_record": group["admission_id"].nunique(),
            "finite_numeric_events": int(group["finite"].sum()),
            "censored_events": int(group["censored"].sum()),
            "uncensored_numeric_admissions": group.loc[
                group["finite"] & ~group["censored"], "admission_id"].nunique(),
            "uncensored_numeric_occupied_bins": len(group.loc[
                group["finite"] & ~group["censored"]].drop_duplicates(["admission_id", "bin"])),
            "admissions_with_repeated_numeric_bins": int((group.loc[
                group["finite"] & ~group["censored"]].drop_duplicates(["admission_id", "bin"])
                .groupby("admission_id").size() >= 2).sum()),
            "uncensored_numeric_values": value_summary(group.loc[group["finite"] & ~group["censored"], "number"]),
            "dictionary": {key: known.get(key, "") for key in
                           ("label", "fluid", "category", "linksto", "unitname")},
            "dictionary_match": bool(known),
            "unit_evidence": "dictionary_only_no_row_unit" if source == "surgery_vital_signs"
            else "recorded_source_unit_not_automatically_validated",
            "available_by_landmark_events": int(group["available_by_landmark"].sum()),
        })
    return records


def profile_candidate(events: pd.DataFrame, cohort: pd.DataFrame, candidate: Candidate):
    rows = events.loc[events["itemid"].isin(candidate.items)].copy()
    unit_ok = rows["unit"].isin(candidate.units)
    numeric_ok = rows["finite"] & ~rows["censored"]
    nominal = rows.loc[unit_ok & numeric_ok].copy()
    if candidate.factors:
        if len(candidate.factors) != len(candidate.units):
            raise ValueError("Every accepted conversion unit needs a factor.")
        nominal["number"] *= nominal["unit"].map(dict(zip(candidate.units, candidate.factors)))
    has_availability = EVENT_SOURCES[candidate.source][-1] is not None
    usable = nominal.loc[nominal["available_by_landmark"]].copy() if has_availability else nominal
    values = binned_values(usable, cohort["admission_id"])
    temporal, _ = marker_diagnostics(values)
    observed = np.isfinite(values)
    counts = observed.sum(axis=1)
    availability_lag = (
        rows["available_time"] - rows["event_time"]).dt.total_seconds() / 3600
    unique_times = usable.sort_values(["admission_id", "event_time"]).drop_duplicates(
        ["admission_id", "event_time"])
    exact_intervals = unique_times.groupby("admission_id")["event_time"].diff().dt.total_seconds() / 3600
    conflicts = usable.groupby(["admission_id", "event_time"])["number"].nunique()
    bounds = REVIEW_BOUNDS.get(candidate.name)
    review_flags = {
        "status": "diagnostic_flags_only_not_normal_ranges_or_applied_exclusions",
        "zero_events": int((usable["number"] == 0).sum()),
        "bounds_if_specified": list(bounds) if bounds else None,
        "events_outside_review_bounds": int((~usable["number"].between(*bounds)).sum()) if bounds else None,
        "last_bin_values_outside_review_bounds": int((
            np.isfinite(values) & ((values < bounds[0]) | (values > bounds[1]))).sum()) if bounds else None,
    }
    result = {
        "definition": asdict(candidate), "source_events_first24h": len(rows),
        "units_recorded": rows["unit"].value_counts().to_dict(),
        "missing_source_unit_events": int((rows["unit"] == "").sum()),
        "invalid_numeric_events": int((~rows["finite"]).sum()),
        "censored_events": int(rows["censored"].sum()),
        "unit_rejected_events": int((~unit_ok).sum()),
        "rejection_counts_may_overlap": True,
        "nominal_valid_events": len(nominal), "auditable_valid_events": len(usable),
        "event_time_proxy_only": not has_availability,
        "available_time_missing": int(rows["available_time"].isna().sum()),
        "available_time_before_event": int((rows["available_time"] < rows["event_time"]).sum()),
        "available_at_or_after_landmark": int((
            rows["available_time"] >= rows["admission_time"] + pd.Timedelta(hours=24)).sum()),
        "availability_lag_hours": value_summary(availability_lag),
        "unique_admission_event_times": len(unique_times),
        "exact_successive_event_intervals_hours": value_summary(exact_intervals),
        "admission_bins_with_multiple_usable_events": int((
            usable.groupby(["admission_id", "bin"]).size() > 1).sum()),
        "admission_event_times_with_disagreeing_values": int((conflicts > 1).sum()),
        "values_after_explicit_unit_rules": value_summary(usable["number"]),
        "quality_review_flags": review_flags,
        "temporal": temporal,
        "admissions_any": int((counts > 0).sum()),
        "admissions_any_fraction": float((counts > 0).mean()),
        "admissions_repeated": int((counts >= 2).sum()),
        "admissions_repeated_fraction": float((counts >= 2).mean()),
        "admissions_three_or_more_bins": int((counts >= 3).sum()),
        "admissions_five_or_more_bins": int((counts >= 5).sum()),
        "descriptive_outcome_strata": {},
    }
    for label, name in ((0, "survival"), (1, "death")):
        select = cohort["label"].to_numpy() == label
        result["descriptive_outcome_strata"][name] = {
            "admissions": int(select.sum()), "any": int((counts[select] > 0).sum()),
            "repeated": int((counts[select] >= 2).sum()),
        }
    return result, values


def pairwise_redundancy(matrices: dict[str, np.ndarray]) -> list[dict]:
    names = list(matrices)
    records = []
    for first, name in enumerate(names):
        left = matrices[name].reshape(-1)
        for other in names[first + 1:]:
            right = matrices[other].reshape(-1)
            joint = np.isfinite(left) & np.isfinite(right)
            union = np.isfinite(left) | np.isfinite(right)
            count = int(joint.sum())
            correlation = None
            if count >= 30 and np.unique(left[joint]).size > 1 and np.unique(right[joint]).size > 1:
                correlation = float(pd.Series(left[joint]).corr(pd.Series(right[joint]), method="spearman"))
                if not np.isfinite(correlation):
                    raise ValueError("Unexpected undefined pairwise rank correlation.")
            records.append({
                "first": name, "second": other, "jointly_observed_bin_cells": count,
                "observed_mask_jaccard": count / int(union.sum()) if union.any() else None,
                "spearman_same_admission_bin": correlation,
                "fraction_numerically_equal_ignoring_unit_semantics": float(np.isclose(
                    left[joint], right[joint], rtol=1e-5, atol=1e-6).mean()) if count else None,
                "reason_if_no_correlation": None if correlation is not None
                else "Fewer than 30 common cells or a constant variable.",
            })
    return records


def panel_coverage(matrices: dict[str, np.ndarray], names: list[str]) -> dict:
    observed = np.stack([np.isfinite(matrices[name]) for name in names], axis=2)
    counts = observed.sum(axis=1)
    return {
        "variables": names, "admissions": len(observed),
        "observed_fraction": float(observed.mean()),
        "admissions_all_variables_ever_observed": int((counts > 0).all(axis=1).sum()),
        "admissions_all_variables_repeated": int((counts >= 2).all(axis=1).sum()),
        "occupied_time_bins_histogram": np.bincount(
            observed.any(axis=2).sum(axis=1), minlength=STEPS + 1).tolist(),
        "median_observed_cells_per_admission": float(np.median(observed.sum(axis=(1, 2)))),
    }


def audit_database(connection, cohort: pd.DataFrame, original_masks: np.ndarray) -> dict:
    admissions = cohort["admission_id"].tolist()
    patients = cohort["patient_id"].unique().tolist()
    manifest = fetch_frame(connection, """
        SELECT table_name, trim(source_sha256) AS sha256, imported_rows
        FROM picdb_v1_1_0.dataset_file_manifest ORDER BY table_name
    """)
    if set(manifest["table_name"]) != {name.lower() for name in EXPECTED_TABLES}:
        raise ValueError("The complete verified 17-table source manifest is required.")
    schema = fetch_frame(connection, """
        SELECT table_name, column_name FROM information_schema.columns
        WHERE table_schema = 'picdb_v1_1_0' ORDER BY table_name, ordinal_position
    """)
    dictionaries = {
        "labs": fetch_frame(connection, "SELECT itemid,label,fluid,category FROM picdb_v1_1_0.d_labitems"),
        "items": fetch_frame(connection, "SELECT itemid,label,linksto,category,unitname FROM picdb_v1_1_0.d_items"),
    }
    context = fetch_frame(connection, """
        SELECT a.hadm_id AS admission_id, a.subject_id AS patient_id, a.admission_department,
          NULLIF(a.admittime,'')::timestamp AS admittime,
          NULLIF(a.dischtime,'')::timestamp AS dischtime,
          NULLIF(a.deathtime,'')::timestamp AS deathtime,
          a.hospital_expire_flag, p.gender, NULLIF(p.dob,'')::timestamp AS dob
        FROM picdb_v1_1_0.admissions a JOIN picdb_v1_1_0.patients p USING(subject_id)
        WHERE a.hadm_id=ANY(%s) AND a.subject_id=ANY(%s)
    """, (admissions, patients))
    if context["admission_id"].duplicated().any() or set(context["admission_id"]) != set(admissions):
        raise ValueError("Database admissions do not exactly match the original development set.")
    context = cohort.merge(context, on=["patient_id", "admission_id"], validate="one_to_one")
    if len(context) != len(cohort) or not np.array_equal(
            context["hospital_expire_flag"].astype(int), context["label"]):
        raise ValueError("Development patient/admission/outcome mappings changed.")
    landmark = context["admittime"] + pd.Timedelta(hours=24)
    if not ((context["admission_department"] == "NICU")
            & (context["dischtime"] > landmark)
            & (context["deathtime"].isna() | (context["deathtime"] > landmark))).all():
        raise ValueError("Original cohort no longer meets the frozen landmark definition.")
    source_reports, matrices, profiles = {}, {}, {}
    for source in EVENT_SOURCES:
        frame, timing = fetch_events(connection, source, admissions, patients)
        events = prepare_events(frame, cohort)
        if len(events) != timing["first_24h_events"]:
            raise ValueError("Source timing counts do not match retrieved first-24h events.")
        dictionary = dictionaries["labs" if source == LAB else "items"]
        source_reports[source] = {
            "timing": timing,
            "item_unit_catalogue": catalogue_profile(events, dictionary, source),
            "valid_event_timestamp_parsing": "PostgreSQL_strict_cast_failure_aborts_audit",
            "availability_column": EVENT_SOURCES[source][-1],
            "surgical_unit_limitation": "No row-level unit field; dictionary units not assumed valid."
            if source == "surgery_vital_signs" else None,
        }
        for candidate in CANDIDATES:
            if candidate.source == source:
                profiles[candidate.name], matrices[candidate.name] = profile_candidate(events, cohort, candidate)
    baseline = ["lactate", "total_bilirubin", "creatinine"]
    current_mask = np.stack([np.isfinite(matrices[name]) for name in baseline], axis=2)
    if not np.array_equal(current_mask, original_masks):
        raise ValueError("Baseline raw-event observed masks differ from frozen development export.")
    for name in OTHER_SOURCES:
        time, available, limitation = OTHER_SOURCES[name]
        _, _, query, params = source_queries(name, time, available, admissions, patients)
        counts = fetch_frame(connection, query, params).iloc[0].to_dict()
        source_reports[name] = {
            "timing": {key: int(value) for key, value in counts.items()},
            "eligibility": "not_a_ready_numeric_LSTM_channel", "limitation": limitation,
        }
    age = (context["admittime"] - context["dob"]).dt.total_seconds() / 86400
    panels = {
        "baseline": baseline,
        "blood_gas_extension_for_review": baseline + ["glucose_gas", "sodium_gas", "chloride_gas", "pco2_gas"],
        "hematology_extension_for_review": baseline + ["platelets", "wbc"],
        "routine_vitals_extension_for_review": baseline + ["heart_rate", "temperature"],
    }
    return {
        "analysis": "development_only_all_source_candidate_variable_audit",
        "test_accessed": False, "training_performed": False,
        "admissions": len(cohort), "patients": cohort["patient_id"].nunique(),
        "deaths": int(cohort["label"].sum()),
        "baseline_masks_match_frozen_export": True,
        "source_manifest": manifest.to_dict(orient="records"),
        "table_columns": schema.groupby("table_name")["column_name"].apply(list).to_dict(),
        "dictionary_inventory": {name: frame.fillna("").to_dict(orient="records")
                                 for name, frame in dictionaries.items()},
        "sources": source_reports, "candidate_profiles": profiles,
        "pairwise_redundancy": pairwise_redundancy(matrices),
        "prospective_panel_coverage": {name: panel_coverage(matrices, names) for name, names in panels.items()},
        "static_context": {
            "age_days": value_summary(age), "age_missing": int(age.isna().sum()),
            "birth_after_admission": int((age < 0).sum()),
            "gender_missing": int(context["gender"].isna().sum()),
            "gender_dictionary_values": context["gender"].value_counts().to_dict(),
            "limitation": "Static context requires a separate design, not repeated fictitious temporal measurements.",
        },
        "excluded_tables_and_fields": {
            "diagnoses_icd": "No diagnosis availability timestamp; codes may summarize the whole stay.",
            "d_icd_diagnoses": "Terminology only, not a timed predictor.",
            "outcomes": "Hospital death flags/times are endpoint/eligibility only, never candidate inputs.",
            "future_fields": "Discharge/end times, length of stay, future therapies and final culture results excluded.",
            "irrelevant_demographics": "Insurance, language, religion, marital status and ethnicity not extracted as candidates.",
        },
        "protocol": {
            "endpoint": "same_hospitalization_mortality_at_24_hours_unchanged",
            "main_method": "LSTM_autoencoder_unchanged_no_model_fit_in_audit",
            "scope": "original_train_plus_validation_no_new_patients_or_test_data",
            "window": "[hospital_admission,hospital_admission_plus_24h)",
            "bins": "ten_2.4_hour_bins_last_valid_by_time_then_numeric_row_id",
            "selection": "descriptive_availability_and_quality_no_outcome_predictive_ranking",
            "numeric_policy": "finite_uncensored_numeric_field_no_parsing_inequality_bounds_as_equal_values",
            "unit_policy": "exact_candidate_specific_units_explicit_Hb_conversion_and_dimensionless_exceptions",
            "availability_policy": "if_storetime_exists_require_event<=storetime<landmark_else_explicit_event_time_proxy",
            "plausibility": "values_quantiles_negative_zero_counts_no_clinical_outlier_filter_or_normal_range_claim",
            "review_flags": "broad_technical_checks_not_neonatal_reference_intervals_no_rows_removed_by_bounds",
            "review_bounds": {name: list(bounds) for name, bounds in REVIEW_BOUNDS.items()},
            "correlation": "descriptive_Spearman_on_joint_last_valid_bins_not_independent_or_causal",
            "numeric_equality": "numerical_comparison_only_does_not_establish_matching_units_or_clinical_equivalence",
            "censored": "reported_not_imputed_no_detection_limit_rule_selected",
            "panel_status": "coverage_proposals_only_not_adopted_or_trained",
            "source_inventory": "all_tables_and_dictionary_items_inventoried_numeric_events_profiled_in_four_sources",
            "no_raw_export": True,
        },
    }


def run_candidate_audit(input_file: Path, env_file: Path, output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output_dir.exists():
        raise FileExistsError("Candidate audit output exists; choose a new directory.")
    cohort, digest, masks = load_development_ids(input_file)
    with connect(env_file.resolve()) as connection, connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        connection.execute("SET LOCAL statement_timeout = '180s'")
        report = audit_database(connection, cohort, masks)
    report["development_arrays_sha256"] = digest
    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    report["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    save_json(output_dir / "candidate_audit.json", report)
    (output_dir / "candidate_audit.json").chmod(0o600)
    print(f"Candidate audit saved privately: {output_dir}; no model trained or test arrays read.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=ROOT / "outputs/mortality-dataset/dataset.npz")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mortality-development/candidate-audit")
    args = parser.parse_args()
    run_candidate_audit(args.input_file, args.env_file, args.output_dir)


if __name__ == "__main__":
    main()
