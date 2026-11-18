"""Extract approved additional labs for the original development admissions only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from database.postgres.audit_candidate_variables import (
    CANDIDATES, LAB, binned_values, fetch_events, load_development_ids, prepare_events,
)
from database.postgres.build_mortality_dataset import ENV_FILE, ROOT
from database.postgres.load_dataset import connect
from src.mortality_development import load_development, prepare_development
from src.mortality_panel_study import EXTRA_MARKERS, OUTPUT_ROOT, validate_panel_input
from src.utils import save_json


def extract_labs(events, cohort) -> tuple[np.ndarray, dict]:
    matrices, counts = [], {}
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    for name in EXTRA_MARKERS:
        candidate = by_name[name]
        rows = events.loc[events["itemid"].isin(candidate.items)]
        valid = rows["finite"] & ~rows["censored"] & rows["unit"].isin(candidate.units)
        zeros = valid & rows["number"].eq(0) if name in {"platelets", "wbc"} else valid & False
        values = binned_values(rows.loc[valid & ~zeros], cohort["admission_id"])
        observed = np.isfinite(values)
        matrices.append(values)
        counts[name] = {
            "codes": candidate.items, "units": candidate.units, "source_events": len(rows),
            "retained_events": int((valid & ~zeros).sum()), "zero_events_excluded": int(zeros.sum()),
            "observed_bins": int(observed.sum()), "admissions_any": int(observed.any(axis=1).sum()),
        }
    return np.stack(matrices, axis=2), counts


def export_panels(input_dir: Path, reference_file: Path, env_file: Path, output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if output_dir.exists():
        raise FileExistsError("Choose a new panel export directory.")
    reference = load_development(reference_file)
    summary = json.loads((input_dir / "summary.json").read_text())
    local_file = input_dir / "dataset.npz"
    cohort, digest, _ = load_development_ids(local_file)
    with np.load(local_file, allow_pickle=False) as archive:
        reconstructed = prepare_development(archive, summary)
    if not all(np.array_equal(reference[key], reconstructed[key], equal_nan=True) for key in reference):
        raise ValueError("Baseline inputs, labels or groups differ from the frozen development export.")
    with connect(env_file.resolve()) as connection, connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        connection.execute("SET LOCAL statement_timeout = '180s'")
        frame, timing = fetch_events(
            connection, LAB, cohort["admission_id"].tolist(), cohort["patient_id"].unique().tolist(),
        )
        events = prepare_events(frame, cohort)
        if len(events) != timing["first_24h_events"]:
            raise ValueError("Source timing counts do not match retrieved lab events.")
    extra, counts = extract_labs(events, cohort)
    arrays = {**reference, "observed": np.concatenate((reference["observed"], extra), axis=2)}
    validate_panel_input(arrays)
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    target = output_dir / "development.npz"
    np.savez_compressed(target, **arrays)
    target.chmod(0o600)
    report = {
        "development_source_arrays_sha256": digest,
        "export_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "test_accessed": False, "admissions": len(cohort),
        "baseline_unchanged": True, "extra_variables": counts,
        "qc": "Exact approved units; finite uncensored numerical values; zero platelets/WBC "
              "excluded before binning; no clipping or learned cleaning.",
        "timing": "Laboratory event time proxy, not verified result availability.",
    }
    metadata = output_dir / "extraction.json"
    save_json(metadata, report)
    metadata.chmod(0o600)
    print(f"Private development-only panel export saved to {output_dir}.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/mortality-dataset")
    parser.add_argument("--reference-file", type=Path, default=OUTPUT_ROOT / "development.npz")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "panel-export")
    args = parser.parse_args()
    export_panels(args.input_dir, args.reference_file, args.env_file, args.output_dir)


if __name__ == "__main__":
    main()
