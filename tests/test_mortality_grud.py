import numpy as np
import torch

from src.mortality_comparison import biomarker_features
from src.mortality_grud import MortalityGRU, recurrent_inputs, supervised_scores


def test_gaps_are_per_marker_and_causal_and_scaling_uses_only_fit():
    fit = np.array([[[0., 10.], [2., np.nan], [np.nan, 14.], [np.nan, np.nan]]])
    held = np.array([[[np.nan, 112.], [3., np.nan], [np.nan, np.nan], [9., 12.]]])
    _, (values, mask, gaps) = recurrent_inputs(fit, held)
    np.testing.assert_array_equal(values[0], [[0, 50], [2, 0], [0, 0], [8, 0]])
    np.testing.assert_array_equal(mask[0], [[0, 1], [1, 0], [0, 0], [1, 1]])
    np.testing.assert_array_equal(gaps[0], [[0, 0], [1, 1], [1, 2], [2, 3]])


def test_decay_control_shares_weights_but_responds_to_gaps():
    torch.manual_seed(42)
    control = MortalityGRU(1, decay=False)
    torch.manual_seed(42)
    model = MortalityGRU(1, decay=True)
    for name, value in control.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value)
    values = torch.tensor([[[2.], [0.]]])
    mask = torch.tensor([[[1.], [0.]]])
    gaps = torch.tensor([[[0.], [8.]]])
    torch.testing.assert_close(control(values, mask, gaps), control(values, mask, gaps * 0))
    assert not torch.allclose(model(values, mask, gaps), model(values, mask, gaps * 0))
    with torch.no_grad():
        model.input_rate.zero_()
        model.hidden_decay.weight.zero_()
    torch.testing.assert_close(model(values, mask, gaps), control(values, mask, gaps))


def test_supervised_fit_scores_held_admissions_and_same_panel_features_cover_all_markers():
    fit = np.random.default_rng(42).normal(size=(8, 10, 5))
    held = np.ones((3, 10, 5))
    held[:, 2:, 4] = np.nan
    scores = supervised_scores(fit, np.tile([0, 1], 4), held, decay=True, seed=42, epochs=1)
    assert scores.shape == (3,) and np.isfinite(scores).all()
    features = biomarker_features(held)
    assert features.shape == (3, 25)
    np.testing.assert_array_equal(features[:, 24], 2)
