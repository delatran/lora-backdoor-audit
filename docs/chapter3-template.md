# Chapter 3 Evidence Template

> This is an internal evidence-collection template, not submission prose. Do not replace
> placeholders with estimates, fixture outputs, or model-generated numbers. Every value must
> point to an observed log, manifest, hash, and analysis artifact. Machine status values, field
> names, and long hashes belong in the provenance appendix; the main report must translate their
> scientific meaning into self-contained prose.

The completed hosted run generates the canonical evidence bundle under
`<run-root>/report-assets/`. Use `result-register.json` as the numeric source of
truth, `tables/*.csv` as the plotted source data, and `figure-manifest.json` to
verify every SVG/PNG hash. Figure numbering and report captions belong in the
Word document; do not add numbers, internal stage identifiers, hashes, paths, or
notebook release labels inside the images.

## 3.1 Environment and cohort state

- Hardware/runtime artifact: `[PENDING]`
- Execution-profile ID, semantic hash, and file hash: `[PENDING]`
- Dependency-lock, scientific-config, design, source-tree, source-manifest, and
  bundle-manifest hashes: `[PENDING]`
- Model/dataset revisions: `[PENDING]`
- Planned/attempted/completed/QC-valid/QC-failed/interrupted pairs: `[PENDING]`
- Nominal adapter rows, unique clean hashes, unique backdoored hashes, and
  effective independent lineages: `[PENDING]`
- Cohort and QC ledger hashes: `[PENDING]`
- Raw artifact root and immutable snapshot hash: `[PENDING]`
- Evidence status (`pilot_observed`, `cohort_complete`, or
  `research_released`): `[PENDING]`

## 3.2 Backdoor validity and stealth

| Condition | Effective lineages | Backdoored ASR (CI) | Clean target rate (CI) | Paired ASR lift | Clean/backdoored F1 | Absolute F1 difference | QC failed / attempted | Raw artifact + SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

## 3.3 RQ1: detector comparison

Report weight-only, target-informed behavior, Simple OR, and calibrated fusion
separately. Mark fusion `exploratory` unless both grouped cross-fitting and the
locked count gate pass. Thresholds must point to a validation lock. Include the
fold-receipt hash, common-probe receipt hash, bootstrap seed/iterations,
valid-resample counts, effective lineage count, and synchronized
fusion-minus-baseline paired differences.

| Detector | Claim label | Effective n | AUROC (CI) | PR-AUC (CI) | MCC (CI) | Brier (CI) | TPR/FPR (CI) | Runtime | Threshold/fold receipt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

## 3.4 RQ2: adaptive robustness gap

Use only EN, rank 8, T1, poison rate 5%, three seed blocks, the equal-schedule
`0.0` continuation control, and gradient-ratio targets 0.1/0.3/1.0. Every ratio
must continue from the same seed-matched QC-valid non-adaptive adapter for the
same schedule. Report all twelve attempted conditions, including invalid and
interrupted points; the baseline adapter hash; normalization receipt and
calibration rows; fixed coefficient; observed initialization ratio; post-training
QC; surrogate-transfer gap; seed-level uncertainty; device-hours; and the Pareto
artifact. State explicitly that the coefficient is normalized against the
combined clean-plus-target task objective, behavior is evaluated post hoc, and
this is an adaptive-surrogate stress test rather than a fully detector-aware
attack.

## 3.5 RQ3: EN/VI transfer

Provide separate 2x2 matrices for attack transfer, detector discrimination, and
fixed-threshold/calibration transfer. Never collapse them into one cross-lingual
gap. Bind all four cells to the same common-probe receipt and to a private
trigger-equivalence receipt bound to the exact automated, source-locked
literal-preservation gate. Report lineage-macro ASR and
parent-lineage cluster
intervals as primary; retain pooled exact-binomial output as descriptive only.
Include effective lineages, score-distribution paths/hashes, source-language
calibration hashes, receipt modes/hashes, and exploratory/confirmatory labels.

## 3.6 Operational case study

- Command: `[PENDING]`
- Held-out adapter/config hashes: `[PENDING]`
- JSON report: `[PENDING]`
- Decision and uncertainty: `[PENDING]`
- Effective unit and effective n: `[PENDING]`
- Claim scope and evidence status: `[PENDING]`
- Coverage/OOD/limitations: `[PENDING]`

## 3.7 Threats to validity and unresolved evidence

Record grouped-fold leakage checks, seed/split independence, configuration and
runtime shift, probe and translation-receipt coverage, common-benchmark
dependence, adaptive-surrogate scope, QC selection effects, every failed run,
calibration resolution, and external/operational limits. Zero observed false
positives on a small test set is not a production-FPR claim. Do not replace any
`[PENDING]` field with fixture, mock, estimate, or locally assumed hardware
evidence.
