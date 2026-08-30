# Deep Experiment Review — Claim Ledger

> Historical pre-pilot snapshot. Amendment v4 and schema 9 supersede the v3
> continuation rule; see `../review-2026-08-11-pilot-v4/claim-ledger.md`.

| ID | Claim | Label | Evidence or verifier | Falsifier or remaining gate |
|---|---|---|---|---|
| C01 | The one-click plan has a project cap of 12 device-hours. | `[FACT]` local contract | `configs/execution/one-click-colab-pro.json` | A source-authorized plan change |
| C02 | The authorization threshold is 9.6 device-hours because 20% is reserved for managed-runtime overhead. | `[FACT]` local contract | amendment v3, one-click config, schema-8 compute tests | A contract or implementation mismatch |
| C03 | The compute estimate covers the representative pilot, remaining Core work, RQ2, RQ3, and finalization reserve without double-counting observed pilot work. | `[FACT]` implemented and locally testable | `src/lora_audit/compute.py`; compute regression tests | A counterexample showing an omitted or duplicated unit |
| C04 | Three pilot lineages are selected deterministically before outcomes to maximize protocol-strata coverage. | `[FACT]` implemented and locally testable | `src/lora_audit/cohort.py`; cohort tests | Different results under the same locked cohort or outcome-dependent input |
| C05 | Before pair three, an interim `core_only` verdict does not itself justify stopping; stopping is allowed only when the optimistic full-research lower bound exceeds the gate or QC fails. | `[FACT]` protocol and implementation | amendment v3; `compute.py`; continuation regression tests | A feasible continuation rejected by the controller or an infeasible lower bound accepted |
| C06 | The supplementary detector operating point is selected on validation only as maximum TPR subject to FPR at most 0.10 and is explicitly non-estimable when validation resolution is insufficient. | `[FACT]` implemented and locally testable | `thresholds.py`, `detectors.py`, `scan.py`; threshold/release tests | Test-set-driven selection or an invalid selected FPR |
| C07 | Threshold locks and secondary operating points are detector-, validation-, and pipeline-bound before the held-out release. | `[FACT]` implemented and locally testable | release-boundary checks and tamper regression tests | A self-hashed but rebound profile reaching release |
| C08 | `bootstrap.py --mode status` is intended to verify persisted upload state without creating receipts or session markers. | `[FACT]` implementation target | bootstrap tests plus final physical before/after tree comparison | Any file-system delta caused by status mode |
| C09 | Local PASS does not prove that Colab allocated an A100 80 GB or ran the cohort. | `[FACT]` evidence boundary | readiness contract and required hosted receipts | A fresh source-bound hardware preflight and hosted execution receipt |
| C10 | Current Colab limits are dynamic; Google documents general runtimes up to 12 hours and Pro+ continuous execution up to 24 hours when compute units are sufficient. | `[FACT]` platform documentation | <https://research.google.com/colaboratory/faq.html> | A newer official Google policy |
| C11 | The synchronized theory package has 459 paragraphs, 9 tables, 4 sections, and 73 PDF pages, with the Markdown hash stored in DOCX metadata. | `[FACT]` local artifact | format-aware DOCX/PDF verifier and page render inspection | A hash, structure, text, or visual mismatch |
| C12 | The current project has completed a representative A100 pilot and is authorized for the full empirical run. | `[UNVERIFIED]` and currently false as a closure claim | No current-source hosted receipt is present | Fresh three-pair QC-valid telemetry and schema-8 `full_research` authorization |
| C13 | RQ1–RQ3 hypotheses are empirically supported. | `[UNVERIFIED]` | No admissible current-source empirical release exists | Complete confirmatory artifacts, sealed-test release, estimates and uncertainty |
| C14 | The project is ready for official submission. | `[UNVERIFIED]` | Local engineering and document checks are insufficient | Official template check, similarity report, independent reading, supervisor approval |
| C15 | The completed one-click path emits report figures from the same hash-bound empirical releases used by the completion receipt. | `[FACT]` implementation target | report-asset generator, exact inventory, deterministic regeneration, figure manifest, result register, and integration tests | Any unbound input, extra output, non-reproducible self-rehashed edit, stale figure hash, or completion receipt without report assets |
| C16 | Report figures contain no visible research-question, hypothesis, or notebook-version labels. | `[FACT]` implementation target | SVG visible-text scanner and notebook presentation regression test | A forbidden label visible in notebook source or SVG text |
| C17 | Figure rendering consumes A100 model outputs but does not itself perform model compute. | `[FACT]` architecture | strict A100 stage gates plus host-CPU renderer declaration | Training or inference reachable through the renderer, or a relaxed accelerator gate |
| C18 | The seven final figures demonstrate the project hypotheses. | `[UNVERIFIED]` | No current-source empirical release exists locally | Completed hosted releases plus estimates, uncertainty, and calibrated interpretation |

## Rival explanations and distinguishing evidence

- A short pilot may look fast because compilation, cache warm-up, data preparation, checkpointing,
  or RQ2/RQ3 work was omitted. The full-stage ledger and observed per-stage telemetry distinguish
  this from genuinely affordable full-pipeline execution.
- A detector may look strong because its threshold was tuned on the test set or lineages leaked
  across folds. Validation-bound locks, grouped OOF receipts, paired-lineage splits, and the sealed
  release distinguish genuine held-out performance from leakage.
- An FPR at or below 0.10 may appear numerically available even when the validation clean sample is
  too small to resolve that level. The explicit empirical-resolution state distinguishes an
  estimable operating point from a misleading threshold claim.
