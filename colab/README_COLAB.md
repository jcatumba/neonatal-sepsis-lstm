# Colab quick start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Safiya-2026/neonatal-sepsis-lstm/blob/main/colab/neonatal_sepsis_colab.ipynb)

1. Open the notebook. Use a fresh Python 3.11-3.13 runtime.
2. Optional: **Runtime > Change runtime type > T4 GPU**. CPU also works.
3. Select **Runtime > Run all** and approve execution. No configuration edits,
   dataset downloads, uploads or Drive authorization are required for the default
   synthetic smoke demo. Installation requires internet.
4. Inspect metrics and plots. Download artifacts from the Files panel under
   `/content/outputs/neonatal-sepsis` before disconnecting.

Allow roughly 2-10 minutes for setup and the demo, depending on package downloads
and Colab resources; this is an estimate, not a guaranteed Colab benchmark.
Synthetic results demonstrate plumbing only, not medical performance.

## Your data

Use the [CSV schema](../README.md#dataset-schema).
In the configuration cell choose:

- `DATA_SOURCE='upload'`: upload exactly one small CSV when prompted.
- `DATA_SOURCE='drive'`: authorize Drive and set `DRIVE_CSV` to your file.
- `DATA_SOURCE='url'`: set `CSV_URL` to a public HTTPS CSV, without credentials.
- `DATA_SOURCE='demo'`: explicit synthetic mode, the default.

Set `FAST_DEV_RUN=False` to train normally. Tune `epochs`, `batch_size`,
`seq_len`, `latent_dim`, `learning_rate`, and `device_preference` in `Config`.
Use an approved environment and de-identified data; do not share clinical data or
save sensitive notebook outputs. The notebook reads one CSV into RAM; Drive
mounting does not turn it into a streaming loader. Smoke mode limits training
sequences, not CSV loading/preprocessing memory.

## Persistence and reproducibility

`COPY_TO_DRIVE=True` mounts Drive and copies artifacts to
`/content/drive/MyDrive/neonatal/outputs`. Change this path per experiment to avoid
overwriting old runs. Copy errors propagate; a failed Drive copy does not mean
your local outputs were lost. Use `save_to_drive_if_enabled(config)` to retry.

Saved files: `best_model.pt`, `preprocessor.joblib`, `threshold.json`,
`model_config.json`, `config.json`, `history.json`, `metrics.json`,
`versions.json`, `split_summary.json`, and training/ROC/PR plots.
Threshold calibration uses normal validation patients, not the held-out test set.
No patient identifiers or raw records are included in these artifacts.

The first cell installs exact direct dependency versions and prints Python,
PyTorch, optional TensorFlow, and the Git revision. TensorFlow is neither installed
nor used by this project. Save the revision and set `REPO_REF` to it for later
fresh-runtime reruns. Existing clones are not silently updated. Direct pins do
not lock Colab's OS, GPU driver or every transitive package. Seeds and deterministic
PyTorch algorithms are enabled; bitwise equivalence across hardware, runtimes or
PyTorch releases is not guaranteed. If PyTorch reports an unsupported deterministic
GPU operation, use `device_preference='cpu'`.

The notebook clones this repository's `main` branch on a new Colab runtime.
These changes must be pushed to GitHub before the public badge can run them.
For local use, start the notebook from the project root after installing
`requirements-colab.txt` in a dedicated Python 3.11-3.13 environment.

## Troubleshooting

- **Out of memory:** reduce batch size (e.g. 8), sequence length and hidden size;
  use smoke mode or a smaller de-identified CSV. Increasing GPU RAM does not
  solve host-memory exhaustion while loading a CSV.
- **Package conflicts or binary incompatibility:** disconnect/delete the runtime
  and Run all before importing other libraries. Do not mix this environment with
  unrelated pip installs. If setup ran after imports, restart the runtime.
- **Unsupported Python:** select a Colab Python 3.11-3.13/past runtime; newer Python
  versions are not certified with these pins.
- **Drive permission/path errors:** authorize the correct Google account, ensure
  the file exists under `MyDrive`, and check destination write permissions.
  Remount Drive or use upload mode. Do not put credentials in the notebook.
- **Invalid CSV:** errors identify missing columns, invalid labels, duplicate
  timestamps, too-short patients or entirely invalid features. Fix the source
  data; failed loads never silently switch to synthetic data.
- **GPU unavailable:** CPU fallback is explicit. GPU is recommended for large
  datasets, not needed for the demo. Full training time depends on data size.
