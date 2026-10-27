"""
Evaluation Script — LSTM Autoencoder for Neonatal Sepsis Detection
===================================================================
- Computes reconstruction errors on test data
- Selects optimal anomaly detection threshold
- Evaluates sensitivity, specificity, AUROC, AUPRC
- Generates ROC and Precision-Recall curves
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)

from src.model import LSTMAutoencoder


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM Autoencoder for Neonatal Sepsis"
    )
    parser.add_argument("--normal_data",  type=str, default="data/test_normal.npy")
    parser.add_argument("--sepsis_data",  type=str, default="data/test_sepsis.npy")
    parser.add_argument("--model_path",   type=str, default="results/best_model.pt")
    parser.add_argument("--save_dir",     type=str, default="results/")
    parser.add_argument("--input_dim",    type=int, default=8)
    parser.add_argument("--hidden_dim",   type=int, default=64)
    parser.add_argument("--latent_dim",   type=int, default=32)
    parser.add_argument("--seq_len",      type=int, default=24)
    parser.add_argument("--num_layers",   type=int, default=2)
    parser.add_argument("--threshold_percentile", type=float, default=95.0,
                        help="Percentile of normal errors to set threshold")
    return parser.parse_args()


def load_model(args, device):
    model = LSTMAutoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        num_layers=args.num_layers
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    return model


def compute_errors(model, data_np, device, batch_size=64):
    """Compute reconstruction errors for a numpy array."""
    tensor = torch.tensor(data_np.astype(np.float32))
    errors = []
    for i in range(0, len(tensor), batch_size):
        batch = tensor[i:i+batch_size].to(device)
        err = model.reconstruction_error(batch)
        errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def plot_roc(fpr, tpr, auc_score, save_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2,
             label=f"AUROC = {auc_score:.3f}", color="#e63946")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve — Neonatal Sepsis Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    plt.close()


def plot_pr(precision, recall, ap_score, save_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2,
             label=f"AUPRC = {ap_score:.3f}", color="#457b9d")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Neonatal Sepsis Detection")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=150)
    plt.close()


def plot_error_distribution(normal_errors, sepsis_errors, threshold, save_dir):
    plt.figure(figsize=(9, 4))
    plt.hist(normal_errors, bins=50, alpha=0.6,
             color="#2a9d8f", label="Normal")
    plt.hist(sepsis_errors, bins=50, alpha=0.6,
             color="#e76f51", label="Sepsis")
    plt.axvline(threshold, color="black", linestyle="--",
                linewidth=1.5, label=f"Threshold = {threshold:.4f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Distribution\n"
              "(Neonatal Sepsis vs Normal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=150)
    plt.close()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────
    model = load_model(args, device)
    print(f"Model loaded from: {args.model_path}")

    # ── Load test data ────────────────────────
    if os.path.exists(args.normal_data) and os.path.exists(args.sepsis_data):
        normal_data = np.load(args.normal_data).astype(np.float32)
        sepsis_data = np.load(args.sepsis_data).astype(np.float32)
    else:
        print("[WARNING] Test data not found. Using synthetic data for demo.")
        normal_data = np.random.randn(200, args.seq_len,
                                      args.input_dim).astype(np.float32)
        # Sepsis data: higher amplitude signals
        sepsis_data = (np.random.randn(100, args.seq_len, args.input_dim)
                       * 2.5).astype(np.float32)

    # ── Compute errors ────────────────────────
    normal_errors = compute_errors(model, normal_data, device)
    sepsis_errors = compute_errors(model, sepsis_data, device)

    # ── Threshold: 95th percentile of normal ──
    threshold = np.percentile(normal_errors, args.threshold_percentile)
    print(f"\nAnomaly threshold ({args.threshold_percentile}th percentile "
          f"of normal errors): {threshold:.6f}")

    # ── Labels & scores ───────────────────────
    all_errors = np.concatenate([normal_errors, sepsis_errors])
    all_labels = np.concatenate([
        np.zeros(len(normal_errors)),
        np.ones(len(sepsis_errors))
    ])
    all_preds = (all_errors > threshold).astype(int)

    # ── Metrics ───────────────────────────────
    auc  = roc_auc_score(all_labels, all_errors)
    ap   = average_precision_score(all_labels, all_errors)
    fpr, tpr, _ = roc_curve(all_labels, all_errors)
    prec, rec, _ = precision_recall_curve(all_labels, all_errors)
    cm   = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"\n{'='*45}")
    print(f"  Evaluation Results")
    print(f"{'='*45}")
    print(f"  AUROC        : {auc:.4f}")
    print(f"  AUPRC        : {ap:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}  (Recall / True Positive Rate)")
    print(f"  Specificity  : {specificity:.4f}  (True Negative Rate)")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"{'='*45}\n")
    print(classification_report(all_labels, all_preds,
                                 target_names=["Normal", "Sepsis"]))

    # ── Save threshold ────────────────────────
    np.save(os.path.join(args.save_dir, "threshold.npy"),
            np.array([threshold]))

    # ── Plots ─────────────────────────────────
    plot_roc(fpr, tpr, auc, args.save_dir)
    plot_pr(prec, rec, ap, args.save_dir)
    plot_error_distribution(normal_errors, sepsis_errors,
                            threshold, args.save_dir)

    print(f"Plots saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
