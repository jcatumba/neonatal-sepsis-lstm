# LSTM Autoencoder for Neonatal Sepsis Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Safiya-2026/neonatal-sepsis-lstm/blob/main/colab/neonatal_sepsis_colab.ipynb)

An unsupervised anomaly detection framework using an **LSTM Autoencoder** trained on neonatal physiological time-series data to detect early signs of sepsis.

**Research prototype only, not for clinical diagnosis or treatment.** An anomaly
score is not evidence of sepsis; the synthetic demonstration does not establish
clinical performance.

## Run in Google Colab

Open the badge above and select **Runtime > Run all** in a fresh Python 3.11-3.13
runtime. Defaults use explicitly synthetic data and a two-epoch smoke run, with
no uploads, private data, or Drive permissions needed. Setup plus the demo is
estimated at 2-10 minutes, depending on download speed and runtime availability.
A T4 GPU is optional for the demo and recommended for full datasets.

The notebook installs [requirements-colab.txt](requirements-colab.txt), prints
versions, trains, calibrates a threshold on normal validation patients, evaluates
held-out test patients, saves artifacts, and reloads them for inference. Reusable
logic lives in [src/](src/). Colab artifacts are written to
`/content/outputs/neonatal-sepsis`; optional Drive copying persists them.

See [Colab quick start and troubleshooting](colab/README_COLAB.md) for upload,
Drive, HTTPS CSV, OOM, dependency-conflict and permission guidance. Set
`FAST_DEV_RUN=False` for full training. These changes must be pushed to GitHub
before the public Colab badge can load the new notebook.

## Dataset schema

The Colab pipeline accepts one CSV, one row per regularly sampled time point:

| Column | Meaning |
|---|---|
| `patient_id` | De-identified patient identifier; never window across patients |
| `timestamp` | ISO-8601 datetime; unique within each patient |
| `sepsis_label` | Row-level binary label: 0 normal, 1 sepsis |
| `heart_rate`, `respiratory_rate`, `spo2`, `temperature` | bpm, breaths/min, %, Celsius |
| `systolic_bp`, `diastolic_bp`, `mean_arterial_pressure`, `blood_glucose` | mmHg, mmHg, mmHg, mmol/L |

Example header:

```csv
patient_id,timestamp,sepsis_label,heart_rate,respiratory_rate,spo2,temperature,systolic_bp,diastolic_bp,mean_arterial_pressure,blood_glucose
p001,2025-01-01T00:00:00,0,140,45,96,37,65,35,45,5
```

Provide at least **five entirely normal and five sepsis patients**, each with at
least `seq_len` rows. All eight feature columns are required in the listed order
(CSV column order itself may differ). Missing/implausible values are interpolated
within a patient; a completely missing/invalid feature is rejected. Units and
sampling cadence must match across training and inference; no resampling occurs.
Nonoverlapping windows discard incomplete tails and use the maximum row label.
Interpolation is for offline analysis and may use later observations; this is
not a validated causal/real-time detection pipeline.

Patients are split 60/20/20 with stratification before windowing. Only entirely
normal patients are used for training and validation. Scaling is fitted on
training data only, and the threshold uses normal validation errors. The test
set is never used for threshold selection. Metrics are window-level, not
patient-level. Smoke mode limits sequences after splitting, not CSV RAM usage.
Use de-identified data in an approved environment; never publish private data or
sensitive notebook outputs.

---

## Overview

Neonatal sepsis is a life-threatening condition requiring early identification. This model learns the normal physiological patterns of neonates (heart rate, SpO₂, respiratory rate, temperature, blood pressure, blood glucose) and flags deviations as potential sepsis based on **reconstruction error**.

**Key idea:** Train only on healthy (non-sepsis) signals → high reconstruction error at inference = anomaly = sepsis alert.

---

## Architecture

```
Input (batch, seq_len, n_features)
        │
   ┌────▼────────────────┐
   │  Encoder (LSTM ×2)  │  → Latent vector (batch, latent_dim)
   └────────────────────┘
        │
   ┌────▼────────────────┐
   │  Decoder (LSTM ×2)  │  → Reconstruction (batch, seq_len, n_features)
   └─────────────────────┘
        │
   Reconstruction Error (MSE) → Anomaly Score → Threshold → Sepsis / Normal
```

---

## Input Features

| Feature | Unit |
|---|---|
| Heart Rate | bpm |
| Respiratory Rate | breaths/min |
| SpO₂ | % |
| Temperature | °C |
| Systolic BP | mmHg |
| Diastolic BP | mmHg |
| Mean Arterial Pressure | mmHg |
| Blood Glucose | mmol/L |

---

## Project Structure

```
neonatal-sepsis-lstm/
├── colab/
│   ├── neonatal_sepsis_colab.ipynb
│   └── README_COLAB.md
├── src/
│   ├── data.py                 # CSV validation, patient splits, sequences
│   ├── model.py                # Original Encoder, Decoder, LSTMAutoencoder
│   ├── train.py                # Training, scheduler, best checkpoint
│   ├── eval.py                 # Validation calibration, ROC/PR, metrics
│   ├── pipeline.py             # Artifact export and reloadable inference
│   └── utils.py                # Config, seeds, paths, optional Drive copy
├── preprocessing.py            # Original preprocessing helpers
├── lstm_autoencoder.py          # Backward-compatible model imports
├── test_model.py
├── tests/test_colab_pipeline.py
├── train.py                    # Legacy NPY training script
├── evaluate.py                 # Legacy NPY evaluation script
├── requirements-colab.txt       # Exact Colab direct dependency pins
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Safiya-2026/neonatal-sepsis-lstm.git
cd neonatal-sepsis-lstm
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-colab.txt
```

---

## Quick Start

**Recommended:** use the [Colab notebook](colab/neonatal_sepsis_colab.ipynb), also
runnable locally from the project root using Python 3.11-3.13.
It performs the full CSV-to-inference workflow with reusable modules.
`requirements.txt` remains the legacy, unpinned dependency list.

### Test locally

```bash
python -m pip install pytest
python -m pytest test_model.py tests/test_colab_pipeline.py -q
```

The following legacy NPY scripts are retained for existing users. They do not
provide the Colab pipeline's patient-disjoint CSV splits, saved preprocessing,
or validation-only threshold calibration. In particular, legacy evaluation
calibrates on its normal test input; use the notebook for held-out evaluation.

### 1. Generate synthetic data (for testing)
```bash
python -c "from preprocessing import generate_synthetic_data; generate_synthetic_data()"
```

### 2. Train the model
```bash
python train.py \
  --data_path data/normal_signals.npy \
  --input_dim 8 \
  --seq_len 24 \
  --hidden_dim 64 \
  --latent_dim 32 \
  --epochs 100 \
  --batch_size 64
```

### 3. Evaluate and set threshold
```bash
python evaluate.py \
  --normal_data data/test_normal.npy \
  --sepsis_data data/test_sepsis.npy \
  --model_path results/best_model.pt \
  --threshold_percentile 95
```

---

## Code Availability

The complete source code for the LSTM Autoencoder model, preprocessing pipeline, training, and evaluation scripts is publicly available in this repository. A permanently archived version is deposited on Zenodo:

> **DOI:** `https://doi.org/[YOUR-ZENODO-DOI]`  ← *Replace after Zenodo deposit*

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{yourname2024neonatal,
  author    = {[Your Name]},
  title     = {LSTM Autoencoder for Neonatal Sepsis Detection},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/<your-username>/neonatal-sepsis-lstm}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
