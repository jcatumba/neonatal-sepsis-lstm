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

## Development-only model comparison

The original notebook **automatically evaluates test**. Do not rerun it to
compare or tune models. Its first real test results have already been inspected;
this does not turn that set back into a never-seen independent evaluation.

The new comparison uses only the original training and validation admissions.
From the project root, in the existing dependency environment, run:

```bash
python -m src.mortality_development
python -m src.mortality_comparison
```

The first command reads only training/validation arrays from the private local
audit dataset. It creates `outputs/mortality-development/development.npz`
with only observed biomarker bins, mortality labels and arbitrary patient-group
codes. It does not read or export test arrays or original identifiers. Observed
values are recovered by inverting the original scaling (subject to float32
rounding); imputed entries are discarded. No PostgreSQL connection is needed.
Existing output files/directories are never overwritten.

The second command compares three prespecified models on the same five
patient-disjoint folds, stratified by each patient's maximum admission outcome:

- L2 logistic regression, `C=1`, no class weighting, with last/minimum/maximum,
  linear slope per hour and occupied-bin count for each marker (15 features).
  Summaries use observed bins only, not forward-filled values. Median imputation,
  missingness indicators and standardization are fitted on each training fold.
- The existing observed-MSE LSTM autoencoder trained on all fitting admissions.
- The same autoencoder trained only on fitting admissions with survival labels.

Each autoencoder refits forward-fill/initial-gap medians/min-max scaling using
its own fitting population, then runs exactly ten epochs without outer-fold
checkpoint selection. Both rank by observed-only MAE with unchanged direction.
This intentionally differs from the original validation-checkpointed run.
Logistic summaries missing for an entire fitting fold use zero plus missingness
indicators. Slopes use bin positions (2.4-hour spacing), not original event times.
No hyperparameter search, threshold optimization, test scoring, calibration,
risk clustering or production model selection is performed.

`outputs/mortality-development/comparison/comparison.json` contains aggregate
fold metrics, mean AUROC and average precision (the primary comparison metric),
fold standard deviations, input hash, configuration and package versions.
Standard deviations across overlapping training folds are **not confidence
intervals**. Results are exploratory development estimates, not an independent
replication or proof of clinical benefit. No individual predictions are saved.
Keep this NPZ and all results private under the same data-use requirements.

Local execution is sufficient; a new Colab run is not inherently required.
The complete local five-fold/ten-epoch comparison has already finished: mean
development AP was 0.277 for logistic regression, 0.215 for the all-admission
autoencoder and 0.222 for the survivor-only autoencoder. These are development
estimates, not new test results. **No Colab rerun is necessary to obtain this
comparison.** See the [review report](../Data%20and%20code%20review%20and%20preparation.md)
for the full summary and limitations.

For an optional Colab rerun, publish these modules first, use a fresh runtime
with the same repository/dependency setup, and upload **only the new development
NPZ**. In a separate notebook, replace the original data/training/evaluation
cells with:

```python
from pathlib import Path
from google.colab import files
from src.mortality_comparison import run_comparison
from src.mortality_model import MortalityConfig

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only development.npz")
input_file = Path(next(iter(uploaded)))
report = run_comparison(
    input_file,
    Path("/content/outputs/mortality-development/comparison"),
    MortalityConfig(device_preference="cpu"),
)
```

Download `comparison.json` before disconnecting. CPU/standard RAM is sufficient.
The loader rejects the old exchange NPZ and the full audit NPZ. Do not run
the original test-evaluation or inference cells in this development notebook.

## Nested logistic regularization and calibration assessment

After the three-model comparison, run the bounded logistic follow-up locally:

```bash
python -m src.mortality_logistic
```

It reads only the existing `outputs/mortality-development/development.npz`.
No new export, PostgreSQL access, cloud transfer or original mortality notebook
execution is required. `--input-file` and `--output-dir` can select other paths;
existing outputs are never overwritten.

The prespecified protocol retains the same 15 features and no class weighting:

- Five outer patient-grouped, outcome-stratified folds, seed 42, matching the
  original development comparison. Within each outer training pool, three
  inner patient-grouped folds use seed `42 + outer_fold_number`.
- Select `C` from `0.01, 0.1, 1, 10` by mean inner-fold average precision; exact
  ties select the smallest `C` (strongest regularization). Every candidate fits
  imputation, missingness indicators and scaling on its inner training fold.
- Refit the selected pipeline on the outer training pool and score only the
  outer validation fold. Evaluate fixed `C=1` on the same folds as a reference.
  A constant predictor fitted to each outer training prevalence supplies the
  Brier reference. No outer outcomes influence `C` selection.
- Report AP, AUROC and Brier score per outer fold and as mean/sample standard
  deviation. Brier is mean squared probability error (lower is better).
  Every admission also gets exactly one outer out-of-fold (OOF) prediction;
  pooled OOF metrics and a five-quantile-bin calibration curve are descriptive
  mixtures of fold-specific models, not a single final model's performance.
- Assess probability calibration only: no Platt/isotonic recalibration, operating
  threshold selection, feature expansion, final model fitting or test scoring.

Outputs are private aggregate files under
`outputs/mortality-development/nested-logistic/`: `nested_logistic.json` and
`calibration.png`. The JSON includes inner search scores, selected `C` per fold,
bin counts/observed mortality, input and implementation hashes, and versions.
No individual probabilities, labels or patient codes are saved.

This remains exploratory internal validation: the logistic model family was
chosen using the same development pool. Nested tuning prevents using outer
fold outcomes for hyperparameter selection, but does not undo that earlier
selection or provide independent clinical validation. Fold SDs are not
confidence intervals, and the calibration curve alone cannot establish
reliable individual mortality probabilities.

The full local run is complete; **no Colab rerun is necessary**. Mean outer-fold
AP/AUROC/Brier were 0.2728/0.6838/0.072985 for nested selection and
0.2770/0.6899/0.073132 for fixed `C=1`. The prevalence baseline Brier was
0.078938. The search did not improve mean AP; the small Brier difference does
not demonstrate a meaningful advantage. Five-bin calibration shows some
underprediction at the low end and overprediction at the high end, not verified
clinical probability accuracy. Full results and interpretation are recorded in
the [review report](../Data%20and%20code%20review%20and%20preparation.md).

## Youden operating point and frozen research pipeline

After retaining fixed `C=1`, run:

```bash
python -m src.mortality_operating_point
```

Only the existing development NPZ is loaded. The original notebook, test
exchange, full audit dataset and PostgreSQL are not used. Five outer grouped
folds use seed 42. For each outer training pool, three inner grouped folds
use seed `42 + outer_fold_number` and refit the complete `C=1` pipeline.
Their internal OOF probabilities select Youden's threshold; ties select the
highest finite threshold. The pipeline is then refitted on the outer training
pool and the threshold applied unchanged to the outer validation patients.
The decision rule is `probability >= threshold`.

The report includes sensitivity, specificity, precision (positive predictive
value), negative predictive value, alert count/alerts per 100 admissions,
accuracy and confusion matrices. It also retains AP, AUROC and Brier.
Undefined predictive values are `null` with an explicit reason.
Fold means/SDs and pooled counts are reported separately: pooled decisions
come from different fold-specific thresholds, not one universal cutoff.
Fold SDs are descriptive, not confidence intervals.

After that evaluation, Youden on the full development pool's five-fold OOF
probabilities supplies a separate frozen threshold. A final `C=1` pipeline
is fitted using all development admissions. **No performance is estimated
for this full-data fitted artifact.** Its threshold is transferred from
cross-fitted models, whose probability distributions may differ; it is not
a validated clinical intervention cutoff. No added calibration or further
regularization/feature search is performed.

The private, non-overwriting output directory
`outputs/mortality-development/frozen-logistic/` contains:

- `evaluation.json`: aggregate outer-fold evaluation and provenance.
- `model.joblib`: fitted imputer, missingness indicators, standard scaler and
  logistic regression, with no stored input rows or individual predictions.
- `model_spec.json`: frozen threshold, marker/feature order, units, transformed
  coefficients/intercept, classifier configuration, package versions and hashes.
  Coefficients describe standardized features, not raw-unit or causal effects.

The input contract for later research scoring is `(admissions, 10, 3)` in
physical units, with `NaN` for originally unmeasured bins, ordered lactate,
total bilirubin, creatinine. Do not pass six-channel scaled AE inputs or
forward-filled values. This interface does not extract events, check source
units or assess clinical plausibility.

Load joblib files **only from a trusted source** in a compatible environment;
deserialization can execute code. A synthetic reload example is:

```python
import json
from pathlib import Path
import joblib
import numpy as np
from src.mortality_operating_point import predict_observed

directory = Path("outputs/mortality-development/frozen-logistic")
spec = json.loads((directory / "model_spec.json").read_text())
model = joblib.load(directory / "model.joblib")
synthetic_bins = np.ones((1, 10, 3), dtype=float)
probabilities, flags = predict_observed(model, synthetic_bins, spec["threshold"])
```

The export verifies reload agreement using synthetic inputs only; it does not
score real development rows with the final full-data fit. Model and threshold
remain research artifacts requiring independent confirmation.

The full local evaluation and freeze are complete; **no Colab rerun is needed**.
Pooled outer sensitivity/specificity were 47.77%/80.48%, PPV/NPV 18.80%/94.22%,
with 21.96 alerts per 100 admissions and confusion matrix
`[[1336, 324], [82, 75]]`. Missing 82 of 157 death admissions is a major
limitation, not a clinically acceptable rule-out claim. The separate frozen
cutoff is 0.07646250931594001; these pooled rates belong to the five different
internally selected outer-fold thresholds, **not the frozen model/cutoff**.
See the [review report](../Data%20and%20code%20review%20and%20preparation.md) for
the full interpretation and next confirmation gate.

## Improving the autoencoder as the main method

The autoencoder LSTM remains the main article-related method. Logistic
regression is only a supervised comparator; its frozen artifact does not
replace the LSTM experiment.

The bounded development study runs with:

```bash
python -m src.mortality_autoencoder_study
```

It uses the same development-only NPZ; no new export or database connection
is needed. CPU is sufficient, though this repeated training takes longer than
the logistic comparisons. The CLI limits PyTorch to two CPU threads.
`--device cuda` requests an available GPU; unavailable CUDA is reported and
CPU used explicitly. No dependencies or source clinical files are added.

The prespecified grid isolates three changes:

| Candidate | Observed-value transform | Fitting admissions | Epochs |
|---|---|---|---:|
| baseline | Identity | All | 10 |
| log | `log1p` | All | 10 |
| log_long | `log1p` | All | 30 |
| log_long_survivors | `log1p` | Survivors only | 30 |

Observed biomarkers in this development pool are nonnegative and skewed.
`log1p` accepts recorded zero values and preserves NaN gaps; negative values
raise an error, never get clipped. After transformation, each fitting population
independently fits initial-gap medians and min-max scaling; forward-fill uses
past values only, and original observation masks remain inputs.

The 64/32 encoder, 32/64 decoder, observed-only MSE, observed-only MAE score,
Adam learning rate 0.0001, batch size 32 and gradient clipping remain unchanged.
High anomaly error always means higher candidate risk. Thirty epochs and the
log transform are explicit experimental deviations from the original protocol,
not recovered article settings.

Five outer patient-grouped stratified folds use seed 42. Within each outer
training pool, three grouped folds (split seed `42 + outer_fold_number`) select
a candidate by mean mortality AP. Exact ties retain the earliest grid entry.
Inner training seed is `42 + 100 * outer_fold_number + inner_fold_number`,
shared across candidates; outer training seed is `42 + outer_fold_number`,
matching the earlier fixed AE baseline. Every model uses its fixed epochs,
never outer-fold loss for checkpointing. Mortality labels guide candidate
selection and survivor eligibility, **not reconstruction gradients**.

Selected candidates are refitted on outer training patients and compared with
the unchanged baseline on the same held-out patients. Report AP/AUROC fold
means and SDs only: anomaly scores from different transforms/fitted scales are
not pooled. Fold SDs are not confidence intervals. These are exploratory
development estimates, not independent confirmation or calibrated probabilities.

Private outputs in `outputs/mortality-development/autoencoder-study/` are
`study.json`, `comparison.png` and `training.log`. No individual scores or
identifiers, new operating threshold, final model or model weights are exported
in this study. Review the measured comparison before a separate model freeze.
The loader rejects full/test exchanges, outputs never overwrite an existing
directory, and the original test-evaluating notebook must not be rerun.

The full local study is complete: baseline was selected in four outer folds,
and `log_long_survivors` in one. Mean AP/AUROC were 0.2139/0.6162 for the nested
selection procedure versus 0.2151/0.6217 for the unchanged AE. **No improvement
was observed from this bounded search**, so no new AE variant was adopted or
frozen. The baseline reproduces the earlier AE development metrics exactly.
The LSTM remains the main method; this result does not promote logistic
regression into its place. **No Colab rerun is necessary.** See the
[review report](../Data%20and%20code%20review%20and%20preparation.md) for limitations
and the next investigation gate.

## Auditing temporal information without training

Run the descriptive audit using only the existing development NPZ:

```bash
python -m src.mortality_temporal_audit
```

Private, non-overwriting outputs under
`outputs/mortality-development/temporal-audit/` are `temporal_audit.json` and
`temporal_coverage.png`. The audit counts observed cells per bin, repeated
and numerically varying measurements per marker, adjacent/occupied-bin pairs,
bin-distance intervals and the positions that would require forward-fill or
training-median imputation. It does not fit preprocessing, train models, select
thresholds, read test arrays or access PostgreSQL. Only aggregates are saved;
original event times and multiple events within a bin are unavailable.

The full local audit is complete: only 20.21% of biomarker positions are
originally observed, 55.63% require forward-fill and 24.16% initial median fill.
The median admission has six observed marker cells spread across three
occupied bins (not six distinct blood draws). Repeated measurements in at
least two bins occur for lactate in 67.86%, bilirubin in 89.27%, and creatinine
in 4.35% of admissions. All three repeat in only 3.80%.

The loss/score still exclude imputed targets; masks already distinguish
observations from fills. These findings indicate limited/irregular temporal
coverage, not a proven cause of weak discrimination. A prespecified
temporal-order ablation remains a possible diagnostic, **not performed here**.
The subsequently authorized candidate-variable audit is described below.
The LSTM remains the main method.
See the [review report](../Data%20and%20code%20review%20and%20preparation.md) for
definitions, outcome-stratified descriptions and limitations.
**No Colab rerun is necessary for this audit.**

## Auditing additional variables locally

The completed PostgreSQL audit keeps the hospital-mortality endpoint, 24-hour
landmark, original development patients and autoencoder objective unchanged.
It inventories all 17 tables, profiles four numeric event sources and audits
62 candidate variables, without reading test arrays or fitting models.

Use the [local audit command](../database/postgres/README.md#audit-expanded-candidate-variables-development-only);
do not upload the identifier-bearing original audit NPZ or connect Colab
to PostgreSQL. The definitive private aggregate report is
`outputs/mortality-development/candidate-audit-final/candidate_audit.json`.
See the [full findings and candidate table](../Data%20and%20code%20review%20and%20preparation.md#expanded-variable-audit-same-mortality-endpoint-and-autoencoder).

Glucose, pH and hematology have useful coverage for a potential small
extension. Gas-associated channels mostly share existing measurement times;
routine vital signs add temporal observations but have limited cohort
coverage. Severe raw outliers, censored CRP, respiratory-rate unit conflicts,
ambiguous potassium/calcium mappings and unverified laboratory result
availability require explicit decisions. Diagnostic flags do not clean data
automatically, and no expanded panel has been trained or adopted.

**No Colab rerun is necessary for this audit.** The current notebook still
expects the original three biomarkers and six input channels; it does not
support extra variables merely by changing a path. Do not rerun its test
evaluation to choose the expanded panel.

## Comparing fixed panels on development only

The approved local retrospective comparison is complete. With the original
private development NPZ and validated local PostgreSQL import, reproduce from
the repository root (use a new export directory and pass its NPZ as `--input-file` when
repeating the commands):

```bash
python -m database.postgres.export_mortality_panels
python -m src.mortality_panel_study
```

The export preserves the frozen baseline inputs and adds four laboratory
channels using event time as a proxy for result availability. It contains
only development observations, labels, and pseudonymous patient groups; keep
it private. Comparisons use the same five patient-disjoint folds,
fold-local preprocessing and fixed epochs. Sensitivity re-fits all panels
without the last 2.4 hours of laboratory events. That buffer does not verify
result availability.

Glucose + pH led the primary and sensitivity comparisons. The baseline and
combined panel remain candidates for a prespecified comparison. No test
arrays were read, and no final model or threshold was fitted. See
[results and evaluation safeguards](../Data%20and%20code%20review%20and%20preparation.md#fixed-panel-lstm-comparison-development-only).
**No Colab run or upload is needed.**

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
