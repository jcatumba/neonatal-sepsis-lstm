# PICDB mortality experiment in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcatumba/neonatal-sepsis-lstm/blob/main/colab/mortality_colab.ipynb)

Research only: this is not a clinical risk calculator, and reconstruction
error is not a calibrated probability of death. The cohort, extraction,
preprocessing and training choices are a documented approximation of the
article, not its verified original dataset or implementation. The selected
NICU-department cohort has no additional age cutoff, so it includes some
patients older than 28 days.

## Prepare one local file

Follow the [local PostgreSQL workflow](../database/postgres/README.md) once to
build the candidate dataset. Then, from the project root and the same Python
environment, run:

```bash
python -m src.mortality_exchange
```

This creates `outputs/mortality-colab/mortality_colab.npz`; the exporter
refuses to overwrite a file and creates a private directory/file (modes 700
and 600). To choose a new location use `--output-file`. The output under
`outputs/` is Git-ignored. **Do not upload**
`outputs/mortality-dataset/dataset.npz`: it contains the original identifiers
needed for local auditing.

The exchange NPZ is the only data file needed by Colab. It holds `train_x`,
`validation_x`, `validation_y`, `test_x`, `test_y`, `test_group` and
`preprocessor_json`. Training labels and original patient/admission IDs are
not included. The test group is an arbitrary patient pseudonym so the
bootstrap can account for repeated admissions. Pseudonymization is not
anonymization: the sequences and derived model outputs are sensitive.

Before any transfer, verify the PhysioNet terms, institutional approval and
security of the chosen Drive account and Colab environment. No upload is
performed by the export command.

## Run

1. Publish the notebook and corresponding modules to the linked repository;
   the badge cannot open unpushed local changes. Use a fresh Colab runtime.
2. Upload **only** the exchange NPZ to an approved Drive location. Open the
   mortality notebook and choose **Runtime > Run all**. Authorize Drive and
   enter the full NPZ path when prompted, for example
   `/content/drive/MyDrive/project/mortality_colab.npz`.
3. Optional alternatives: set `DATA_SOURCE = 'upload'` for a small NPZ, or
   `DATA_SOURCE = 'demo'` and `FAST_DEV_RUN = True` to run a synthetic
   two-epoch smoke test without clinical data or Drive. Demo results have no
   clinical meaning.
4. Download the run directory from `/content/outputs/mortality/` before the
   runtime disconnects, or explicitly enable `COPY_TO_DRIVE` and provide an
   existing approved output directory. Copies never overwrite an existing
   run directory. Treat model weights, figures and metrics as sensitive.

The first cell installs exact direct dependency pins from
`requirements-colab.txt` and prints Python, PyTorch, optional TensorFlow and
the Git revision. To reproduce a run, set `REPO_REF` to a published commit
SHA and use a fresh runtime. Direct pins do not freeze transitive packages,
Colab's runtime, drivers or GPU. Training enables deterministic PyTorch
algorithms; identical results across hardware are not guaranteed. A GPU is
optional; training and bootstrap time depend on the assigned runtime.

## Method and interpretation

- Ten 2.4-hour bins begin at hospital admission. Each has three scaled
  biomarkers and three original observation masks. The most recent valid
  observation per bin is selected locally; missing values use past values
  and then training-only medians. Scaling is fitted only on training data.
- The model has 64/32 encoder and 32/64 decoder LSTM layers. It trains on all
  included training admissions using Adam at 0.0001 for 10 epochs by default,
  gradient clipping, and MSE averaged over observed biomarker entries within
  each admission, then equally across admissions. This differs from the
  paper's description of training mainly on survivors. Masks are inputs;
  only the three biomarker channels are reconstructed.
- The score is each admission's mean absolute error over observed biomarker
  entries. High error is treated as a candidate mortality-risk ranking; it
  does not establish a probability, causal relationship or clinical utility.
  The relationship between score and outcome must be tested, not assumed;
  the score direction is not reversed based on the test results.
- Youden's index selects a threshold on the validation labels only; ties
  select the highest finite threshold. The test set is not used to select
  the model or threshold. AUROC, average precision, sensitivity, specificity,
  precision, accuracy and a confusion matrix are reported per admission.
- Percentile 95% confidence intervals resample **patients**, preserving
  repeated admissions within a patient. They are conditional on the fitted
  model and fixed threshold. Single-class bootstrap draws and undefined
  precision are counted explicitly. The small test set and number of deaths
  can make results unstable; test prevalence is included in the report.
- The paper's low/medium/high risk groups are deferred because its clustering
  procedure and thresholds are not fully specified. There is no fitted
  probability calibration. Comparison with PRISM/nSOFA is not implemented.

`best_model.pt`, `preprocessor.json`, `model_spec.json`, `config.json`,
`history.json`, `metrics.json`, `provenance.json`, `versions.json`, and
training/ROC/PR plots are saved per run. Provenance includes the Git revision
and exchange file SHA-256; handle it as sensitive along with the other outputs.
The notebook's inference example is synthetic; it does not print individual
patient scores. For new input, `transform_new_sequence` expects ten already
binned measurements in the units in the [data review](../Data%20and%20code%20review%20and%20preparation.md).
It does not itself extract events or validate clinical plausibility.

## Troubleshooting

- **Missing path or permission:** verify Drive was mounted with the correct
  account and that the approved file exists. No silent synthetic fallback is
  used for a failed real-data load.
- **Wrong NPZ or missing arrays:** regenerate the exchange file locally; do
  not use the full audit NPZ or raw source CSVs.
- **Dependency conflicts:** start a fresh runtime and Run all before importing
  packages. Python 3.11–3.13 is supported by the pinned direct dependencies.
- **Unsupported deterministic GPU operation or OOM:** use CPU or reduce
  `batch_size`. Do not change the ten-bin data contract.
- **Existing repository checkout:** the setup cell refuses a dirty checkout or
  a different revision; use a fresh runtime instead of overwriting it.
