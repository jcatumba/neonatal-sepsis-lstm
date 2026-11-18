import numpy as np
import pytest
import torch

from database.postgres.export_mortality_panels import extract_labs
from database.postgres.audit_candidate_variables import prepare_events
from src.mortality_model import MortalityAutoencoder, observed_error
from src import mortality_panel_study as study
from tests.test_candidate_variables import event_frame
import pandas as pd


def test_extract_applies_qc_before_last_value_selection():
    cohort = pd.DataFrame({"admission_id": ["a0"]})
    events = prepare_events(event_frame([
        ("a0", "5129", 0, "35", "×10^9/L", None, None),
        ("a0", "5129", 1, "0", "×10^9/L", None, None),
        ("a0", "5141", 1, "0", "×10^9/L", None, None),
        ("a0", "5223", 1, "0", "mmol/L", None, None),
        ("a0", "5223", 2, "50", "mg/dL", None, None),
        ("a0", "5237", 1, "7", "", None, "<7"),
    ]), cohort)
    values, counts = extract_labs(events, cohort)
    np.testing.assert_array_equal(values[0, 0], [0, np.nan, 35, np.nan])
    assert counts["platelets"]["zero_events_excluded"] == 1
    assert counts["wbc"]["zero_events_excluded"] == 1


def test_panel_loader_refuses_test_keys_before_reading_any_arrays(tmp_path, monkeypatch):
    path = tmp_path / "input.npz"
    np.savez(path, test_x=np.ones(1))

    def forbidden(*args):
        pytest.fail("No archive payload should be read")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden)
    with pytest.raises(ValueError, match="no test keys"):
        study.load_panel_input(path)


def test_masked_error_and_output_for_expanded_model():
    x = torch.zeros((1, 10, 14))
    x[0, 0, 6] = 4
    x[0, 0, 13] = 1
    x[0, 0, 0] = 100
    reconstruction = torch.zeros((1, 10, 7))
    assert observed_error(x, reconstruction, squared=False).item() == 4
    assert MortalityAutoencoder(7)(x).shape == reconstruction.shape


def test_sensitivity_removes_last_bin_before_fitting_every_panel(monkeypatch):
    values = np.ones((24, 10, 7))
    values[:, -1] = 999
    values[0, :-1, :3] = np.nan
    labels = np.tile([0, 1], 12)
    groups = np.arange(24)
    calls = []

    def score(training, train_labels, held, candidate, **kwargs):
        calls.append((training.copy(), held.copy(), candidate.name, kwargs["seed"]))
        return np.arange(len(held), dtype=float)

    monkeypatch.setattr(study, "candidate_scores", score)
    report = study.compare_panels({"observed": values, "labels": labels, "groups": groups}, n_splits=2)
    main, early = calls[:8], calls[8:]
    assert all((train[:, -1] == 999).all() for train, _, _, _ in main)
    assert all(np.isnan(train[:, -1]).all() and np.isnan(held[:, -1]).all()
               for train, held, _, _ in early)
    assert [call[3] for call in main] == [call[3] for call in early]
    assert report["results"]["sensitivity_21_6h"]["excluded_admissions"] == 1
