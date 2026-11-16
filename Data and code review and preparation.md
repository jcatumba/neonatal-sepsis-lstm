# Data and code review and preparation

## Decision

The selected endpoint is **NICU mortality risk**, not a row-level neonatal
sepsis diagnosis. The current Colab notebook is a synthetic-demo pipeline and
cannot be run against the PIC database directory or its source tables. Do not
upload the database to Google Drive or Colab as the next step. The authorized
local copy has been loaded and profiled in PostgreSQL, and a candidate cohort
and dataset have been prepared using documented choices. These choices do not
establish that the original paper's modelling protocol is fully known.

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
No model has been trained and the current Colab notebook does not load this
dataset.

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

Before interpreting model results, define the autoencoder architecture,
reconstruction scoring, threshold/risk-cluster rule, and comparator
definitions. Explicitly document choices that cannot be recovered from the
paper as deviations rather than claiming exact reproduction. Investigate the
paper's Table 2 discrepancies, but do not assume they can be resolved without
the authors' extraction and clarification.

Once those choices are recorded, confirm institutional approval and the
current PhysioNet credentialed-data terms before moving any records to Google
Drive or Colab. Local possession of the dataset and its de-identification do
not alone establish permission for a particular cloud workflow.
