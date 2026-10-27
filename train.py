"""
Training Script — LSTM Autoencoder for Neonatal Sepsis Detection
=================================================================
Trains the model on normal neonatal physiological time-series data.
Saves best model checkpoint and plots training curves.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from src.model import LSTMAutoencoder


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM Autoencoder for Neonatal Sepsis Detection"
    )
    parser.add_argument("--data_path",   type=str, default="data/normal_signals.npy",
                        help="Path to normal (healthy) training data .npy file")
    parser.add_argument("--save_dir",    type=str, default="results/",
                        help="Directory to save model and plots")
    parser.add_argument("--input_dim",   type=int, default=8,
                        help="Number of physiological features")
    parser.add_argument("--hidden_dim",  type=int, default=64)
    parser.add_argument("--latent_dim",  type=int, default=32)
    parser.add_argument("--seq_len",     type=int, default=24,
                        help="Time window length (hours)")
    parser.add_argument("--num_layers",  type=int, default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--val_split",   type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--patience",    type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class Trainer:
    def __init__(self, model, optimizer, criterion, device,
                 save_dir, patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.patience = patience

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(self.device)
            self.optimizer.zero_grad()
            recon = self.model(x)
            loss = self.criterion(recon, x)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def val_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(self.device)
            recon = self.model(x)
            loss = self.criterion(recon, x)
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    def fit(self, train_loader, val_loader, epochs):
        print(f"\n{'='*55}")
        print(f"  Training LSTM Autoencoder — Neonatal Sepsis Detection")
        print(f"{'='*55}\n")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.val_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_dir, "best_model.pt"))
                tag = " ✓ saved"
            else:
                self.epochs_no_improve += 1
                tag = ""

            print(f"Epoch [{epoch:>3}/{epochs}]  "
                  f"Train Loss: {train_loss:.6f}  "
                  f"Val Loss: {val_loss:.6f}{tag}")

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        self.plot_losses()
        print(f"\nBest Val Loss: {self.best_val_loss:.6f}")
        print(f"Model saved to: {self.save_dir}/best_model.pt\n")

    def plot_losses(self):
        plt.figure(figsize=(9, 4))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.val_losses,   label="Val Loss",   linewidth=2, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("LSTM Autoencoder — Training Curve\n(Neonatal Sepsis Detection)")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.save_dir, "training_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Training curve saved to: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load Data ──────────────────────────────
    if os.path.exists(args.data_path):
        data = np.load(args.data_path).astype(np.float32)
        print(f"Loaded data: {data.shape}  "
              f"(samples, seq_len, features)")
    else:
        print(f"[WARNING] Data not found at '{args.data_path}'. "
              f"Using synthetic data for demo.")
        n_samples = 1000
        data = np.random.randn(n_samples, args.seq_len,
                               args.input_dim).astype(np.float32)

    tensor_data = torch.tensor(data)
    dataset = TensorDataset(tensor_data)

    val_size   = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    # ── Model ──────────────────────────────────
    model = LSTMAutoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # ── Train ──────────────────────────────────
    trainer = Trainer(model, optimizer, criterion, device,
                      args.save_dir, args.patience)
    trainer.fit(train_loader, val_loader, args.epochs)


if __name__ == "__main__":
    main()
