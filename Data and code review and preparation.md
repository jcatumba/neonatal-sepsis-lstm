# Data and code review and preparation

## Decision

The selected endpoint is **NICU mortality risk**, not a row-level neonatal
sepsis diagnosis. At the initial review, the Colab notebook was a synthetic-demo
pipeline, incompatible with the PIC database directory and source tables.
Subsequent sections record the local PostgreSQL preparation, separate mortality
notebook, first real execution and development-only comparison. Only minimized
derived inputs, never the full database, are needed in Colab. These documented
choices do not establish that the original paper's modelling protocol is fully
known.

The PostgreSQL scripts in [`database/postgres/`](./database/postgres/) provide
a local, isolated raw-data import and aggregate analysis. They can create
candidate training sequences locally, but do not claim a full reproduction.

## Sources examined

- Article: [Deep learning-based neonatal outcome prediction: an LSTM
  autoencoder framework for pattern analysis and risk assessment](../s41598-026-47450-5.pdf),
  *Scientific Reports* (2026), DOI
  [10.1038/s41598-026-47450-5](https://doi.org/10.1038/s41598-026-47450-5).
- Local dataset: [`paediatric-intensive-care-database-1.1.0`](../paediatric-intensive-care-database-1.1.0),
  specifically its `V1.1.0/` CSV release, plus its `SHA256SUMS.txt` and
  data-use license.
- Current code: [`colab/neonatal_sepsis_colab.ipynb`](./colab/neonatal_sepsis_colab.ipynb)
  and the reusable modules in [`src/`](./src/).

The dataset includes 17 relational tables, including `PATIENTS`, `ADMISSIONS`,
`ICUSTAYS`, `LABEVENTS`, and `D_LABITEMS`. It is not a prejoined, regularly
sampled, eight-feature time series. The current notebook expects one CSV with
`patient_id`, `timestamp`, `sepsis_label`, and eight physiological feature
columns, so its data contract and target do not match these files.

All files checked against the supplied checksum manifest passed. This verifies
that the local v1.1.0 files match the accompanying distribution manifest. It
does **not** prove which exact snapshot, SQL extraction, or records the article
authors used; the article does not publish hashes for its extracted cohort.

## What the article reports

- The paper describes mortality/survival risk among NICU neonates. Its Table 3
  reports 2,320 NICU neonates: 2,081 live and 239 deceased. The abstract and
  portions of the paper also call the task sepsis prediction, so terminology
  and endpoint must be stated carefully.
- It maps admissions, patients, and ICU stays to timestamped laboratory events.
  It identifies lactate, bilirubin, and creatinine as important biomarkers and
  discusses CRP, procalcitonin, white blood cell counts, and platelets.
- The methods describe 10 time steps and feature fields
  `(Admission_ID, Item_ID, Value, Unit, Flag)`.
- It describes an encoder with 64 and 32 LSTM units, a decoder with 32 and 64
  units, MSE training loss, Adam at 0.0001, gradient clipping, and 10 epochs.
  Its anomaly-score description uses mean absolute reconstruction error, then
  discusses clustering/risk groups and comparisons with clinical scores and
  conventional models.
- The article has internal ambiguities relevant to reproduction: it reports
  239 deceased NICU neonates in Table 3, then says 233 deceased neonates were
  extracted for laboratory analysis. The text and Table 6 differ on a cluster
  count, and biomarker units/normal ranges are not consistent throughout.
  The article does not sufficiently specify the complete event-to-sequence,
  cohort, train/test, threshold, and comparison-model procedures to reproduce
  its headline metrics unambiguously.

## Differences from the current Colab code

| Area | Article / PICDB | Current Colab pipeline |
|---|---|---|
| Endpoint | Mortality/survival risk (with sepsis wording) | Expects binary `sepsis_label` |
| Input | Relational PICDB records, especially timestamped labs | A single prepared CSV with eight fixed vitals |
| Sequence | Article describes 10 time steps and biomarker/item fields | Default sequence length 24 |
| Model | Encoder 64/32; decoder 32/64 | Same hidden width in each stacked LSTM |
| Training | 10 epochs; Adam learning rate 0.0001 | Smoke mode is on by default and caps at 2 epochs; default LR 0.001 |
| Score/risk | MAE anomaly score, then risk clustering/comparison | MSE score, 95th percentile of normal validation, binary metrics |
| Data handling | Requires decisions about multiple admissions, events, units, timing | CSV validation, patient-level split, fixed bounds/interpolation |

These are intentional differences in the demonstration pipeline, not evidence
that the article has been reproduced. In particular, do not interpret its
synthetic metrics as paper results or clinical performance.

## Local PostgreSQL workflow

The database workflow imports raw PICDB v1.1.0 CSVs locally into an isolated
PostgreSQL container, preserves the source values as text for subsequent
auditable analysis, checks their source hashes, and emits only aggregate
summaries. It does not modify the source data. Start with
[`database/postgres/README.md`](./database/postgres/README.md).

The article-aligned profiling should answer:

1. Do the PICDB department/ICU/outcome counts reproduce the article's tables?
2. Which outcome source best matches the stated survival status
   (`PATIENTS.EXPIRE_FLAG`, admission mortality flag, and death time), and where
   do these sources disagree?
3. Does a defensible NICU cohort contain 2,320 subjects and 239 deceased
   subjects? Can the paper's 233-person lab subset be derived by a stated rule?
4. Which reported biomarkers exist in `D_LABITEMS`, and how many timestamped,
   numeric events and subjects have each? What are their units, flags, and
   observation-time distributions?
5. Does selecting the last lab observation per patient include measurements
   after discharge or death? How are repeated, irregular, missing, and
   nonnumeric measurements treated?
6. Can article risk groups and comparator metrics be reconstructed from the
   published description, without using outcomes to define predictors?

Patient identifiers and row-level data must not be copied into review reports,
notebook outputs, public repositories, or unapproved cloud storage.

## Local PostgreSQL findings

The local v1.1.0 import completed for all 17 CSV tables. All source CSV hashes
matched the supplied `SHA256SUMS.txt`; the database manifest records the file
hashes, byte sizes, and imported row counts. The imported tables include
13,449 admissions, 13,941 ICU stays, 10,094,117 lab events, and 12,881
patients. Validation found no duplicate `ROW_ID`s in the tables checked and
zero orphans in the six core patient/admission/ICU-stay/lab-item joins.

The paper's Table 3 counts, 2,081 live plus 239 deceased, exactly match **NICU
admission rows** grouped by `ADMISSIONS.HOSPITAL_EXPIRE_FLAG`. They do not
represent 2,320 unique neonates: the admission-department NICU cohort contains
2,260 distinct subjects. Within that cohort, 238 subjects have
`PATIENTS.EXPIRE_FLAG = 1`, while 239 admission rows have
`HOSPITAL_EXPIRE_FLAG = 1`; the patient and admission outcome sources disagree
for three subjects. In contrast, defining the cohort by `ICUSTAYS.FIRST_CAREUNIT
= 'NICU'` yields 3,153 distinct subjects and 240 deaths by the patient flag.
Thus, Table 3 is consistent with the admission-row definition, but its
“neonates” wording is not consistent with a unique-patient count.

The Table 2 check also found unresolved differences. PICDB's literal `General
ICU` admission label has 1,440 live and 417 deceased admissions. The paper's
`General ICU` counts (43 live, 50 deceased) exactly match the local `CICU`
label instead, but that possible label substitution is not explained. PICDB's
cardiac-surgery department has 2,188 live and 2 deceased admissions, versus
2,238 and 2 in the paper. Its infectious-diseases, PICU, SICU, and NICU counts
match the reported values. These differences prevent treating the paper's
department table as a complete verified cohort specification.

The NICU admission cohort has 238 patient-flag deaths; 237 are aged 0–210 days
at death, while one is about 260 days old. Of the 239 NICU admission-flag death
subjects, 235 have at least one linked lab event. There are 233 subjects with
a lactate event and 233 with a total-bilirubin event, but only 223 with all
three of lactate, total bilirubin, and creatinine. These aggregate counts do
not identify the article's inclusion rule for its stated 233-person lab
subset.

Candidate lab availability in the admission-department NICU cohort is
substantial but not a uniform, regularly sampled matrix. Examples include:

| Candidate | PICDB item ID(s) | NICU lab events | Distinct subjects | Observed source unit(s) |
|---|---:|---:|---:|---|
| Lactate | 5227 | 33,257 | 2,231 | mmol/L; some unit fields empty |
| Total bilirubin | 5075 | 8,057 | 2,196 | μmol/L; some unit fields empty |
| Total bilirubin | 5255 | 23,781 | 2,198 | μmol/L; some unit fields empty |
| Direct bilirubin | 5042 | 7,945 | 2,194 | μmol/L |
| Creatinine | 5032 | 83 | 61 | μmol/L |
| Creatinine | 5041 | 8,202 | 2,199 | μmol/L |
| C-reactive protein | 5626 | 29,457 | 2,221 | mg/L; some unit fields empty |
| C-reactive protein | 5821 | 100 | 67 | mg/L |
| Procalcitonin | 6085 | 129 | 117 | ng/ml |
| Platelet count | 5129 | 29,847 | 2,226 | ×10^9/L; some unit fields empty |

Item IDs with the same label can have different availability, so any mapping
must be justified and documented. Unit conversion and handling of missing or
nonnumeric values are required before modeling. A timing check found 4,189 lab
events later than the admission death time among 208,526 events linked to NICU
death admissions; 129 subjects had at least one such event. Features must be
cut off at a prespecified prediction/index time to avoid post-outcome leakage.

These results support that the local copy matches the supplied PICDB v1.1.0
distribution. They do not prove that the article authors used the identical
release files or extraction. The Table 3 match is encouraging evidence of a
shared source/cohort convention, but the Table 2 differences and undefined
233-person lab subset leave exact data identity and model reproduction
unconfirmed.

## Public sample comparison and candidate observation windows

The authors' [Figshare sample](https://doi.org/10.6084/m9.figshare.29335832)
contains 15 laboratory-event CSV files. On 2026-11-16 all 15 downloaded files
matched the MD5 hashes in Figshare's public metadata. We compared their nine
fields as multisets with the locally checksum-verified source `LABEVENTS.csv`,
grouped by subject; this was a local comparison, not a transfer of local data.
The imported PostgreSQL table comes from the same verified source.

- All 15 sample subjects have an NICU admission in the local data, and all 15
  have patient `EXPIRE_FLAG = 0`. Therefore these samples do not demonstrate
  how the paper selected the 233 deceased subjects.
- The samples contain 10,252 events, and the local source has 10,252 events
  across those same subjects. Every sample event has a matching source
  `ROW_ID`; all fields match except `VALUENUM` in 21 events. Ten samples match
  their subject's complete source-event multiset exactly; five have at least
  one of those 21 differences.
- All 21 differences are between finite numeric values, with absolute
  difference at most 0.000001. They are not byte-identical or equal as decimal
  numbers. Small serialization or numerical precision differences are
  plausible, but their cause is unconfirmed.

This comparison is strong evidence that the published samples correspond to
local PICDB source events, but it does not prove that the paper's complete
training data, event selection, or preprocessing are recoverable from them.
In particular, the samples contain complete subject-level lab-event sets in
this comparison, not an independently verified model-input matrix.

The aggregate query [`landmark_profile.sql`](./database/postgres/sql/landmark_profile.sql)
explores candidate observation windows from hospital admission for NICU
admissions, requiring discharge at or after and recorded death after the
landmark, when present. Lab events are linked by subject and admission and
occur in `[admission time, landmark)`. These are **exploratory counts**, not
a chosen prediction protocol:

| Hours after admission | Eligible admissions | Distinct subjects | Eligible admission-flag deaths | Admissions with ≥10 lab events | With lactate, total bilirubin and creatinine |
|---:|---:|---:|---:|---:|---:|
| 6 | 2,299 | 2,239 | 231 | 2,177 | 242 |
| 12 | 2,277 | 2,217 | 215 | 2,176 | 427 |
| 24 | 2,229 | 2,170 | 192 | 2,168 | 1,768 |

At 24 hours, 185 eligible death admissions have at least ten lab events. An
event count is not evidence that ten usable, independent time steps exist:
events can be nonnumeric, simultaneous, repeated, or from unrelated tests.
The three-marker count reflects any events for the candidate item IDs, not
validated values or a complete training matrix. Patients discharged or
deceased before a landmark cannot be evaluated at that landmark; this changes
the target population and must be stated when selecting a prediction time.

### User-selected candidate protocol

The user selected all NICU-department admissions without an additional age
filter, hospital mortality in the same admission, a 24-hour hospital-admission
landmark, a 10-bin biomarker time series, and unsupervised autoencoder
training on all eligible training admissions. Patients must remain disjoint
across training, validation and test partitions. This is a proposed
reproducible approximation, not the paper's verified original protocol.

The selected landmark profile requires discharge strictly after 24 hours and,
where recorded, death strictly after 24 hours. Measurements at or after the
landmark are excluded. There are 2,229 eligible admissions for 2,170 distinct
patients, including 192 admission-flag deaths. No eligible admission was
discharged exactly at 24 hours. Three admission-flag survivors have a
patient-level death flag and two admission-flag deaths lack a patient-level
death flag. These discrepancies must not silently change the selected
admission-level outcome.

Not applying an age filter matters: 337 eligible admissions have recorded age
at admission greater than 28 days, two greater than 210 days, and 42 have a
recorded birth time after admission. The 42 negative ages are data-quality
flags, not missing birth timestamps; their cause is unconfirmed. The cohort
therefore cannot be described as exclusively neonates without further
qualification.

An exploratory 10-bin analysis divides `[admission, admission + 24 hours)`
into ten consecutive 2.4-hour intervals, using candidate item IDs 5227
(lactate), 5075/5255 (total bilirubin), and 5032/5041 (creatinine).
All five are blood lab items; lactate is categorized as blood gas and the
others as chemistry. For the coverage analysis, an observation must have a
numeric-text `VALUENUM` and the expected source unit, mmol/L for lactate or
μmol/L for bilirubin/creatinine. Missing or unexpected units are not assumed
or converted. Under those rules 2,137 admissions have at least one valid
candidate measurement, 1,767 have all three at least once, and 92 have none.
Four lactate and four bilirubin events have missing units; 63 candidate events
have no numeric `VALUENUM` (48 lactate, 14 bilirubin, one creatinine).

Measurements are sparse within individual bins: for example, in the first
bin 85 admissions have all three markers and 471 have none; in the last bin
15 have all three and 1,755 have none. The reproducible aggregate query is
[`biomarker_coverage.sql`](./database/postgres/sql/biomarker_coverage.sql).

The user subsequently selected the last valid value per marker in each bin,
forward-fill within admission, training-only median imputation of initial gaps,
and observation masks. Admissions without any usable candidate biomarker are
excluded. Events without the expected unit or a finite numeric value are
excluded and counted; birth-after-admission cases are retained and flagged.
The user selected a patient-disjoint, mortality-stratified 70/15/15 split.

The local candidate dataset was built with those choices using
[`build_mortality_dataset.py`](./database/postgres/build_mortality_dataset.py)
and [`mortality_data.py`](./src/mortality_data.py):

| Split | Patients | Admissions | In-hospital death admissions |
|---|---:|---:|---:|
| Training | 1,464 | 1,499 | 129 |
| Validation | 314 | 318 | 28 |
| Test | 314 | 320 | 28 |
| Total | 2,092 | 2,137 | 185 |

The builder excluded 92 admissions without a usable candidate marker, counted
eight events with unverified units and 63 with missing/invalid numeric values,
and retained 41 admissions with birth after admission. The sequence shape is
`(admissions, 10, 6)`: three min-max-scaled biomarkers and three original
observation masks. Min-max bounds and initial-gap medians are fitted on
training data only. Split stratification uses whether a patient has at least
one death admission; the per-admission label remains hospital mortality.
The autoencoder is intended to train on all training admissions, irrespective
of those labels.

The private local NPZ includes patient and admission IDs for split auditing;
it is under ignored `outputs/` with restricted directory/file permissions.
All 30 marker/bin observation counts match the independent PostgreSQL coverage
query; split IDs do not overlap, and focused and existing pipeline tests pass.

The user selected observed-value MSE for training, observed-value MAE for
scoring, a Youden threshold selected only on validation, deferred risk groups,
and 95% patient-cluster bootstrap intervals on test. The separate
[`mortality_colab.ipynb`](./colab/mortality_colab.ipynb) implements this
protocol. Its encoder is 64/32 and decoder 32/64; by user choice it trains on
all training admissions, whereas the article describes training primarily on
survivors. The model score is not a calibrated death probability. Confidence
intervals condition on the trained model and fixed threshold.

`python -m src.mortality_exchange` creates a minimized, private local NPZ with
no original patient/admission identifiers. It contains only the prepared
arrays and labels needed for this experiment, test-patient pseudonymous groups
and training preprocessing parameters. It remains sensitive derived data.
The notebook needs this one file, not PostgreSQL or raw PICDB tables. A local
one-epoch training/validation smoke check and a fully synthetic notebook test
passed; the real test set has not been scored for this change.

### Subsequent execution and development-only comparison

After the preparation-stage checks above, the user executed the real mortality
notebook in Colab and its test results were inspected. That test set is
therefore no longer historically unseen. Its existing artifacts are retained;
development comparisons must not rescore it or tune against its results.

The next prespecified comparison pools only the original training and validation
partitions (1,817 admissions, 1,778 patients, 157 death admissions). It uses five
patient-disjoint folds, seed 42, stratified by maximum admission mortality per
patient. Outcomes and patient groups are used only within this development pool.
The primary comparison metric is average precision, with AUROC secondary; both
are reported per admission, separately for each fold and as mean/sample standard
deviation across folds. Fold variability is not a confidence interval.

The candidates are L2 logistic regression (`C=1`, no class weighting),
the existing all-admission autoencoder and the same survivor-only autoencoder.
Logistic features are last/minimum/maximum observed value, least-squares slope
per hour across occupied bin positions, and occupied-bin count for each marker.
Logistic median imputation, missingness indicators and standardization are
fitted within the training fold. An entirely missing fitting feature uses zero
with a missingness indicator. Both autoencoders retain the original architecture,
loss, learning rate, batch size and score direction, but train for fixed ten
epochs, with **no checkpoint selection using the outer fold**. The survivor
variant fits on survival admissions, not exclusively on patients who never
have a death admission.

Each autoencoder fits its own imputation and scaling on its fold's fitting
population (all admissions or survivors). No held-fold values contribute to
these parameters. Original observed bins are recovered by inverse-scaling the
private audit sequences, then discarding every originally unobserved entry.
This incurs float32 recovery precision, not a new event extraction. A separate
minimized export contains only these observations, development labels and
pseudonymous groups, and rejects full/test exchanges at load time.

Implementations are [`mortality_development.py`](./src/mortality_development.py)
and [`mortality_comparison.py`](./src/mortality_comparison.py); instructions are
in the [mortality guide](./colab/README_MORTALITY.md#development-only-model-comparison).
No threshold, risk cluster, hyperparameter search or new test evaluation is
part of this comparison. Selecting a winner from development estimates remains
exploratory; independent confirmation would require new held-out data or a
carefully agreed evaluation plan, not repeated reuse of the inspected test set.

The full comparison completed locally on CPU: five folds, ten epochs for each
autoencoder in each fold, no smoke mode. The NPZ reader was instrumented to
reject any key beginning with `test`; no such access occurred. The new private
development input and aggregate results are in
`outputs/mortality-development/development.npz` and
`outputs/mortality-development/comparison/comparison.json`. Existing test
artifacts were not overwritten.

| Model | Development AUROC, fold mean +/- SD | Development AP, fold mean +/- SD |
|---|---:|---:|
| L2 logistic regression | 0.690 +/- 0.068 | 0.277 +/- 0.056 |
| All-admission autoencoder | 0.622 +/- 0.055 | 0.215 +/- 0.060 |
| Survivor-only autoencoder | 0.621 +/- 0.058 | 0.222 +/- 0.061 |

Development mortality prevalence is 157/1,817 (8.64%); individual fold
prevalences are included in the private aggregate report. Logistic regression
has the highest mean AP in this prespecified comparison. Survivor-only training
does not show a clear improvement over all-admission training. These estimates
do not establish statistical superiority, independent test improvement,
calibration or clinical utility. They must not be compared as if equivalent to
the previous single-split test metrics or the paper's headline performance.

**No new Colab execution is necessary for these results:** the complete local
comparison has finished and its aggregate report is saved. An optional cloud
reproduction uses only the new development NPZ and the separate procedure in
the mortality guide; do not rerun the original test-evaluating notebook.

### Nested logistic tuning and probability assessment

The approved follow-up retains the same development pool and 15 predictors.
Five outer patient-grouped stratified folds (seed 42) estimate performance;
three inner patient-grouped folds within each outer training pool select
`C` from `0.01, 0.1, 1, 10` by mean inner-fold AP. Inner seed is 42 plus the
outer fold number (starting at 1). Exact ties select the smallest `C`.
Each inner/outer fit independently refits imputation, missingness indicators
and scaling. No class weighting or new predictors are introduced.

The selected pipeline is compared with fixed `C=1` on the same outer folds.
AP and AUROC assess ranking; Brier score assesses squared probability error
and is compared with a constant prediction fitted to outer-training mortality
prevalence. Each admission is scored exactly once out of fold. Pooled OOF
metrics and five-quantile-bin calibration curves are descriptive combinations
of these fold models, not the performance of a single final model.
Empty bins caused by tied probabilities are omitted and bin counts reported.
The output contains only aggregates, with no individual probabilities or IDs.

This step **assesses**, but does not recalibrate probabilities. It does not
choose operating thresholds, export a final predictor or access test data.
Fold standard deviations are not confidence intervals. Because the model family
was previously chosen on this development pool, nested regularization tuning
does not make the entire analysis independent of that earlier selection.

The implementation is [`mortality_logistic.py`](./src/mortality_logistic.py),
reusing the existing grouped folds, biomarker summaries and logistic pipeline.
The [nested evaluation guide](./colab/README_MORTALITY.md#nested-logistic-regularization-and-calibration-assessment)
documents the local command and private aggregate outputs.

The full local run completed with 1,817 admissions, 1,778 patients and 157
death admissions, with exactly one outer OOF prediction per admission.
The search selected `C=0.01` in three outer folds and `C=1` in two.
The unchanged fixed-`C=1` AP and AUROC match the earlier three-model comparison
exactly, providing a check on the shared feature/pipeline extraction.

| Model | Outer AP, mean +/- SD | Outer AUROC, mean +/- SD | Outer Brier, mean +/- SD |
|---|---:|---:|---:|
| Nested-selected C | 0.2728 +/- 0.0574 | 0.6838 +/- 0.0687 | 0.072985 +/- 0.003229 |
| Fixed C=1 reference | 0.2770 +/- 0.0558 | 0.6899 +/- 0.0681 | 0.073132 +/- 0.003707 |
| Outer-training prevalence | 0.0864 +/- 0.0011 | 0.5000 +/- 0.0000 | 0.078938 +/- 0.000903 |

The bounded search did not improve mean AP over the prespecified reference.
Its small Brier reduction is descriptive and does not establish a meaningful
or statistically supported advantage. Both logistic procedures have lower
Brier error than the fold-training-prevalence predictor.

Pooled OOF results are reported separately: nested selection AP 0.2542, AUROC
0.6790, Brier 0.072990; fixed `C=1` AP 0.2564, AUROC 0.6853, Brier 0.073138.
Pooled AP/AUROC need not equal fold means because they additionally compare
predictions produced by different fitted models.

Mean OOF predicted mortality is 8.67% for nested selection and 8.70% for fixed
`C=1`, versus 8.64% observed. Matching this average does not ensure calibration
throughout the range. In the highest predicted-probability quintile, nested
selection predicts 19.80% versus 18.41% observed; fixed `C=1` predicts 22.16%
versus 19.23% observed. In the lowest quintile, those values are 3.34% versus
4.95% and 2.70% versus 4.12%, respectively. The quintiles have different
membership for the two procedures; this is not a matched subgroup comparison.
Only five aggregate bins and 157 total outcomes are available, so neither the
curve nor the average error establishes reliable individual risk estimates.

Private results are saved in
`outputs/mortality-development/nested-logistic/nested_logistic.json` and
`outputs/mortality-development/nested-logistic/calibration.png`. Input and prior
comparison hashes were unchanged; no test keys were read, no test artifacts
were modified, and no new calibration model, threshold or final predictor was
fitted. **No Colab execution is necessary for these completed steps.**

### Youden operating-point evaluation and experimental freeze

The user approved proceeding with the operating point and experimental pipeline.
Fixed `C=1`, the existing 15 features and no additional calibration are retained.
The five outer patient-grouped folds use seed 42. In each outer training pool,
three inner patient-grouped folds (seed 42 plus outer fold number) produce OOF
probabilities with independent imputation, indicators and scaling per fit.
Youden on only these training-pool outcomes and inner OOF probabilities selects
the fold threshold; ties choose the highest finite threshold. Refit on the outer
training pool, then apply `probability >= threshold` to the external fold.
The external fold does not select or adjust its own threshold.

Report sensitivity, specificity, precision/positive predictive value, negative
predictive value, alert count/alerts per 100 admissions, accuracy, confusion
matrix and the preceding probability/ranking metrics. Pooled outer counts
combine fold-specific thresholds. Undefined predictive values remain explicit;
fold SDs are not confidence intervals.

After evaluation, a separate final threshold is derived using Youden on the
full development pool's five-fold OOF probabilities. Fit the full preprocessing
and `C=1` classifier on all development admissions and persist it, together
with schema, transformed coefficients, configuration, threshold, provenance
and versions. This transfers a cross-fitted cutoff to a larger fitted model;
no performance claim for that full-data artifact is made. It is not scored
on real fitting rows after training, and reload agreement is checked using
synthetic observations only.

The implementation is
[`mortality_operating_point.py`](./src/mortality_operating_point.py), documented
in the [operating-point guide](./colab/README_MORTALITY.md#youden-operating-point-and-frozen-research-pipeline).
Only the private development NPZ is accepted. Test remains excluded and no
per-admission probabilities, labels or identifiers are written to results.

The full local run completed. The five internally selected thresholds were
0.101531, 0.107169, 0.093064, 0.103073 and 0.088201. Pooled outer decisions
on 1,817 admissions (157 deaths) were:

| Metric | Pooled outer value |
|---|---:|
| Sensitivity | 47.77% |
| Specificity | 80.48% |
| Positive predictive value | 18.80% |
| Negative predictive value | 94.22% |
| Alerts | 399 (21.96 per 100 admissions) |
| True negatives / false positives | 1,336 / 324 |
| False negatives / true positives | 82 / 75 |

Fold mean +/- SD sensitivity was 0.4774 +/- 0.0841, specificity 0.8048 +/-
0.0461, PPV 0.1912 +/- 0.0361, NPV 0.9422 +/- 0.0080 and alerts per 100
admissions 21.96 +/- 4.48. Pooled outer AP 0.256446, AUROC 0.685320 and Brier
0.073138 match the preceding fixed-`C=1` OOF results; only decision thresholds
have been added.

Youden here misses 82/157 death admissions and 324/399 alerts are false
positives. The high NPV must be read alongside 8.64% mortality prevalence:
it does not establish a safe clinical rule-out procedure. Youden optimizes
an internal discrimination criterion, not the clinical consequences of missed
events and false alarms. No clinical deployment is justified by these results.

The separate frozen threshold is **0.07646250931594001**, selected from the full
development OOF probabilities. The above confusion matrix and rates use the
five fold-specific thresholds, **not this frozen cutoff**. A full-development
pipeline and schema/configuration were saved privately in
`outputs/mortality-development/frozen-logistic/` as `model.joblib`,
`model_spec.json` and `evaluation.json`. Reload predictions matched exactly on
synthetic input. The input and earlier aggregate reports were unchanged; no
test keys were read, and no new calibration, risk groups or test evaluation
were performed. **No Colab execution is necessary.**

### Returning to the LSTM as the principal method

The user explicitly prioritized improving the **autoencoder LSTM**, rather
than replacing it with logistic regression or a supervised LSTM classifier.
The frozen logistic pipeline remains a supervised comparator and secondary
research artifact. The endpoint, cohort, 24-hour window and three biomarkers
remain unchanged.

A prespecified four-candidate study retains the original architecture, observed
MSE reconstruction loss and observed MAE anomaly score, unchanged score
direction, Adam learning rate 0.0001, batch size 32 and gradient clipping:

| Candidate | Transform | Training admissions | Fixed epochs |
|---|---|---|---:|
| baseline | Identity | All | 10 |
| log | log1p | All | 10 |
| log_long | log1p | All | 30 |
| log_long_survivors | log1p | Survival admissions | 30 |

This tests skew handling, additional optimization and whether survivor fitting
interacts with the new preprocessing. The observed development values are
nonnegative; bilirubin includes zeros. NaNs retain original missingness, and
negative values are rejected by log candidates rather than clipped. Imputation
and scaling are independently refitted within each fitting population after
the optional transform. The log transform and 30-epoch runs are declared
extensions, not the paper's verified original preparation.

Five outer patient-disjoint mortality-stratified folds use seed 42, as before.
Within each outer training pool, three grouped folds (split seed 42 plus outer
fold number) select the candidate by mean mortality AP; exact ties favor the
earliest candidate. Inner training seeds are `42 + 100*outer_fold + inner_fold`,
shared across candidates. Outer training seeds are `42 + outer_fold`, matching
the preceding fixed AE baseline. All models use their fixed final epoch; no
outer-fold checkpoint selection is allowed. Mortality labels are used for
model selection and survivor eligibility, not the reconstruction gradient.

Compare selected candidates with the unchanged fixed baseline on each outer
fold, reporting AP/AUROC means and SDs, not pooled anomaly-score metrics:
different transforms and fitted scales do not supply a common score scale.
There is no new threshold, probability calibration or final model export in
this study. Any apparent improvement remains internal/exploratory; the prior
development comparisons and inspected test cannot become independent evidence.
No new test scores are generated or old artifacts replaced.

Implementation and local commands are in
[`mortality_autoencoder_study.py`](./src/mortality_autoencoder_study.py) and the
[autoencoder study guide](./colab/README_MORTALITY.md#improving-the-autoencoder-as-the-main-method).

The complete CPU study finished: five outer folds, three inner folds, four
candidates, and 1,280 total training epochs across the required fits. Exactly
one outer validation prediction was produced per development admission. The
inner search retained the baseline in four folds and selected
`log_long_survivors` in the first fold.

| Autoencoder procedure | Outer AP, mean +/- SD | Outer AUROC, mean +/- SD |
|---|---:|---:|
| Unchanged baseline | 0.2151 +/- 0.0599 | 0.6217 +/- 0.0553 |
| Nested-selected candidate | 0.2139 +/- 0.0587 | 0.6162 +/- 0.0522 |

The baseline's aggregate and per-fold metrics match the preceding all-admission
AE comparison exactly. In the first outer fold, the internally selected
survivor/log/30-epoch variant changed AP by -0.005845 and AUROC by -0.027496
relative to the baseline; the other folds were unchanged. Thus this bounded
selection procedure **did not improve measured development performance**.
It does not prove that every alternative LSTM configuration is inferior, nor
justify expanding the search repeatedly against these same outer results.

The unmodified baseline remains the main-method reference; no newly selected
variant or new final autoencoder is frozen on the basis of this study.
The supervised logistic comparator remains separate. These results provide
no new test estimate and no evidence that the paper's published discrimination
has been reproduced.

Private outputs are `study.json`, `comparison.png` and `training.log` in
`outputs/mortality-development/autoencoder-study/`. Input and earlier comparison,
logistic-tuning and frozen-logistic evaluation hashes were unchanged. No test
keys were read, individual predictions saved, thresholds fitted, or model
weights/final model exported. **No Colab rerun is necessary: the complete
study has already run locally.**

### Temporal-information audit of the development sequences

The approved diagnostic uses only the existing development NPZ: 1,817
admissions, 1,778 patients and 157 death admissions. It does not retrain models,
select predictors or thresholds, refit imputation/scaling, read test arrays,
query PostgreSQL or alter prior artifacts.

Of the 54,510 possible **biomarker** cells (1,817 x 10 x 3):

| Origin under the current preprocessing policy | Cells | Fraction |
|---|---:|---:|
| Originally observed | 11,017 | 20.21% |
| Would use forward-fill | 30,322 | 55.63% |
| Would use initial training-median imputation | 13,171 | 24.16% |

These counts exclude the three mask input channels. They identify structural
dependencies, not new fitted median values. Reconstruction loss and MAE still
use **observed cells only**: the audit does not reveal a bug that treats
imputed targets as observed. However, the encoder sees the filled biomarker
inputs along with their masks, and filling does not create new measurements.

Admissions have a median of six observed biomarker cells (IQR 4--8) across
three occupied time bins (IQR 2--4), not ten fully measured steps. Occupied-bin
counts 1--10 are 136, 612, 482, 307, 168, 62, 27, 17, 5 and 1 admissions.
Multiple biomarkers can occupy one time bin, so six observed cells do not
mean six different measurement times or blood draws.

| Marker | Observed cells | Admissions with >=1 occupied bin | With >=2 bins | With >=3 bins | With numerical variation |
|---|---:|---:|---:|---:|---:|
| Lactate | 4,387 | 1,804 (99.28%) | 1,233 (67.86%) | 713 (39.24%) | 1,209 (66.54%) |
| Total bilirubin | 5,020 | 1,806 (99.39%) | 1,622 (89.27%) | 889 (48.93%) | 1,616 (88.94%) |
| Creatinine | 1,610 | 1,518 (83.54%) | 79 (4.35%) | 12 (0.66%) | 78 (4.29%) |

All percentages in this table use all 1,817 development admissions. Numerical
variation means min/max differ beyond `numpy.isclose(rtol=1e-5, atol=1e-6)`,
to tolerate inverse-scaling precision; it is **not a clinical change threshold**.
Repeated observations are not merely identical values: 1,663 admissions
(91.52%) have at least one numerically varying marker. Still, only 1,198
(65.93%) have two or more repeated markers and 69 (3.80%) have all three
markers repeated; all three vary numerically in 68 admissions (3.74%).
Thus some temporal information exists, especially in lactate/bilirubin, but
dense three-marker trajectories are rare.

Successive occupied-bin separations have medians 4.8 hours for lactate,
7.2 for bilirubin and 12.0 for creatinine. Forward-filled positions have median
bin-distance ages 7.2, 4.8 and 4.8 hours, respectively. These are **bin-index
proxies**, not exact intervals or sample ages. The NPZ discards within-bin
event counts and exact event timestamps, so original sampling cadence cannot
be reconstructed from this file.

Observation intensity also differs descriptively by outcome: survival
admissions have median six observed cells, mean 5.87; death admissions median
seven, mean 8.13. Observation fractions are 19.56% and 27.09%, respectively.
This may reflect the care/measurement process as well as physiology; it does
not demonstrate causation, leakage or that the LSTM actually uses missingness.
Admission-level outcome strata can contain the same patient across different
admissions, so patient counts across strata need not add.

**Conclusion:** the regular ten-step representation is much denser than its
actual observations, with little creatinine recurrence. This is a plausible
representation limitation, not proof of why AUROC is modest or proof that
the LSTM cannot exploit the available temporal changes. More epochs alone
do not resolve the data-density issue.

Before another hyperparameter search, the next justified experiment is a
prespecified **temporal-order ablation** of the same autoencoder: compare the
chronological input with an order-permuted control, keeping values, masks,
fitting populations and evaluation discipline comparable. That would test
whether ordering helps beyond levels/availability. It has **not** been run
in this audit. Time-since-measurement channels, alternative bins, added
predictors or a longer window would be separate protocol decisions, not
automatic changes or guaranteed improvements.

Implementation:
[`mortality_temporal_audit.py`](./src/mortality_temporal_audit.py). Private
aggregate outputs are `temporal_audit.json` and `temporal_coverage.png` under
`outputs/mortality-development/temporal-audit/`. Counts conserve across markers
and outcome strata; input and prior report hashes are unchanged. No individual
measurements, groups, scores or predictions are written. **No Colab execution
is necessary for this completed audit.**

### Questions for the authors

The authors have not been contacted as part of this local analysis. To
investigate an exact reconstruction, request the PICDB release and file
checksums, cohort SQL and inclusion/exclusion rules (including the 239 versus
233 discrepancy), event-to-sequence extraction and unit/flag handling,
normalization and imputation, patient-level split and random seed, and the
original training/evaluation and risk-clustering code. Ask whether the public
Figshare samples and the archived GitHub release correspond to the inputs and
implementation used for the published results. Do not send patient-level
records or identifiers.

## Next decision gate

Keep the **autoencoder LSTM as the principal article-related method** and the
unchanged AE as the reference: the bounded nested transform/duration/population
search did not improve its measured performance. Do not silently substitute
the supervised logistic comparator for the main method. Its experimental
fixed-`C=1` pipeline and Youden policy remain secondary frozen artifacts.

The temporal audit now establishes sparse, unevenly sampled biomarker
trajectories, not whether ordering improves mortality ranking. Agree the
temporal-order ablation and evaluation strategy before another LSTM search.
The negative tuning result and structural audit do not themselves establish
why discrimination is modest. Measurement-gap inputs, new predictors or a
different window would be new protocol choices, not a reconstruction of
unspecified article details. Avoid repeated optimization against these outer
folds or any use of the inspected test for development decisions.

Independent confirmation remains unresolved. Do not claim a new confirmed test
score or apply pooled fold-specific logistic metrics to its frozen full-data
model/cutoff. Any new operating criterion, recalibration fitting, clinical-score
comparison or risk groups require separate choices; none establishes clinical
utility by itself.
Document choices unavailable from the article as deviations, not exact
reproduction. Its Table 2 discrepancies still require author clarification.

Once those choices are recorded, confirm institutional approval and the
current PhysioNet credentialed-data terms before moving any records to Google
Drive or Colab. Local possession of the dataset and its de-identification do
not alone establish permission for a particular cloud workflow.
