"""
LSTM Autoencoder for Neonatal Sepsis Detection
================================================
Unsupervised anomaly detection using reconstruction error.
Trains on normal (non-sepsis) physiological time-series data,
then flags sepsis when reconstruction error exceeds a threshold.

Author: [Your Name]
Institution: [Your Institution]
License: MIT
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Encodes input time-series into a compressed latent representation.

    Args:
        input_dim  : Number of physiological features (e.g., HR, SpO2, RR...)
        hidden_dim : Hidden size of LSTM layers
        latent_dim : Size of bottleneck (latent space)
        num_layers : Number of stacked LSTM layers
        dropout    : Dropout probability between LSTM layers
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        # Take last layer hidden state
        last_hidden = hidden[-1]                  # (batch, hidden_dim)
        latent = self.activation(self.fc(last_hidden))  # (batch, latent_dim)
        return latent


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    """
    Reconstructs the original time-series from the latent representation.

    Args:
        latent_dim : Size of bottleneck input
        hidden_dim : Hidden size of LSTM layers
        output_dim : Number of features to reconstruct (= input_dim)
        seq_len    : Length of the sequence to reconstruct
        num_layers : Number of stacked LSTM layers
        dropout    : Dropout probability between LSTM layers
    """

    def __init__(self, latent_dim, hidden_dim, output_dim,
                 seq_len, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        # Expand latent across seq_len
        x = self.fc(latent).unsqueeze(1)               # (batch, 1, hidden_dim)
        x = x.repeat(1, self.seq_len, 1)               # (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x)                           # (batch, seq_len, hidden_dim)
        reconstruction = self.output_layer(out)         # (batch, seq_len, output_dim)
        return reconstruction


# ─────────────────────────────────────────────
# LSTM Autoencoder
# ─────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder for neonatal sepsis anomaly detection.

    The model is trained exclusively on normal (healthy) neonatal
    physiological signals. During inference, a high reconstruction
    error indicates anomalous (sepsis-like) patterns.

    Args:
        input_dim  : Number of input features
        hidden_dim : LSTM hidden layer size
        latent_dim : Bottleneck dimension
        seq_len    : Time window length
        num_layers : Stacked LSTM layers
        dropout    : Dropout rate
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=32,
                 seq_len=24, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim,
                               num_layers, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim,
                               seq_len, num_layers, dropout)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def reconstruction_error(self, x):
        """
        Computes per-sample Mean Squared Error between input and reconstruction.

        Returns:
            errors: Tensor of shape (batch,) — one error score per sample
        """
        with torch.no_grad():
            recon = self.forward(x)
            errors = torch.mean((x - recon) ** 2, dim=(1, 2))
        return errors

    def predict(self, x, threshold):
        """
        Binary sepsis prediction based on reconstruction error threshold.

        Args:
            x         : Input tensor (batch, seq_len, input_dim)
            threshold : Float threshold — errors above this → sepsis (1)

        Returns:
            predictions : Binary tensor (batch,) — 1 = sepsis, 0 = normal
        """
        errors = self.reconstruction_error(x)
        predictions = (errors > threshold).int()
        return predictions, errors
