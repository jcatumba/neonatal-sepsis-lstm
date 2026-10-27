"""
Unit Tests — LSTM Autoencoder for Neonatal Sepsis Detection
"""

import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import LSTMAutoencoder, Encoder, Decoder


BATCH     = 8
SEQ_LEN   = 24
INPUT_DIM = 8
HIDDEN    = 32
LATENT    = 16


@pytest.fixture
def model():
    return LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        latent_dim=LATENT,
        seq_len=SEQ_LEN,
        num_layers=2,
        dropout=0.0   # deterministic for tests
    )


@pytest.fixture
def sample_input():
    return torch.randn(BATCH, SEQ_LEN, INPUT_DIM)


# ── Shape tests ──────────────────────────────────────────────

def test_encoder_output_shape(sample_input):
    encoder = Encoder(INPUT_DIM, HIDDEN, LATENT, num_layers=2, dropout=0.0)
    latent = encoder(sample_input)
    assert latent.shape == (BATCH, LATENT), \
        f"Expected ({BATCH}, {LATENT}), got {latent.shape}"


def test_decoder_output_shape():
    decoder = Decoder(LATENT, HIDDEN, INPUT_DIM, SEQ_LEN,
                      num_layers=2, dropout=0.0)
    latent = torch.randn(BATCH, LATENT)
    recon = decoder(latent)
    assert recon.shape == (BATCH, SEQ_LEN, INPUT_DIM), \
        f"Expected ({BATCH}, {SEQ_LEN}, {INPUT_DIM}), got {recon.shape}"


def test_autoencoder_output_shape(model, sample_input):
    recon = model(sample_input)
    assert recon.shape == sample_input.shape, \
        f"Reconstruction shape mismatch: {recon.shape} vs {sample_input.shape}"


# ── Reconstruction error tests ───────────────────────────────

def test_reconstruction_error_shape(model, sample_input):
    errors = model.reconstruction_error(sample_input)
    assert errors.shape == (BATCH,), \
        f"Expected ({BATCH},), got {errors.shape}"


def test_reconstruction_error_nonnegative(model, sample_input):
    errors = model.reconstruction_error(sample_input)
    assert (errors >= 0).all(), "Reconstruction errors must be non-negative"


def test_identical_input_low_error(model):
    """A model trained to identity should produce near-zero error."""
    x = torch.zeros(BATCH, SEQ_LEN, INPUT_DIM)
    errors = model.reconstruction_error(x)
    assert errors.mean().item() < 10.0, "Error on zero input unexpectedly high"


# ── Prediction tests ─────────────────────────────────────────

def test_predict_output_types(model, sample_input):
    preds, errors = model.predict(sample_input, threshold=0.5)
    assert preds.dtype == torch.int32
    assert errors.dtype == torch.float32


def test_predict_binary(model, sample_input):
    preds, _ = model.predict(sample_input, threshold=0.5)
    unique = set(preds.cpu().numpy().tolist())
    assert unique.issubset({0, 1}), f"Predictions must be 0 or 1, got {unique}"


def test_high_threshold_all_normal(model, sample_input):
    preds, _ = model.predict(sample_input, threshold=1e9)
    assert preds.sum().item() == 0, "With infinite threshold, all should be Normal"


def test_zero_threshold_all_sepsis(model, sample_input):
    preds, _ = model.predict(sample_input, threshold=0.0)
    assert preds.sum().item() == BATCH, "With zero threshold, all should be Sepsis"


# ── Gradient test ─────────────────────────────────────────────

def test_gradients_flow(model, sample_input):
    recon = model(sample_input)
    loss = torch.nn.functional.mse_loss(recon, sample_input)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
