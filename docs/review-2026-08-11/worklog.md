# Deep Experiment Review — Evidence Worklog

## 1. Initial diagnosis

- Separated three different clocks: Google platform limits, the project's 12-device-hour cap, and
  the evidence-based runtime estimate produced by the experiment.
- Rejected the premise that the owner should manually guess the scientific runtime. The owner sets
  a resource ceiling; the experiment must measure, estimate, and decide within that ceiling.
- Preserved the stronger boundary that the experiment cannot grant itself unlimited external spend
  or silently delete scientific stages to fit a budget.

## 2. P0 findings and corrections

1. The old authorization estimate covered Core but did not fully budget RQ2, RQ3, and managed
   finalization risk. It could therefore authorize a nominal 12-hour run that was not the full
   research pipeline.
2. A single first-pair pilot was too sensitive to lineage and cold-start effects. Amendment v3 now
   requires three outcome-blind representative pairs selected for strata coverage.
3. The first sequential-pilot stopping rule was mathematically wrong: an observed maximum cannot
   decrease, but replacing a conservative planned unit with an actual unit can reduce total planned
   cost. The corrected controller stops early only when the optimistic lower bound still exceeds
   the gate or QC fails.
4. Historical pilot evidence could be reused across changed source/config unless every stage and
   run root was bound to current hashes. The current contract invalidates stale evidence.

## 3. P1 findings and corrections

1. The proposal's supplementary maximum-TPR operating point under FPR at most 0.10 was not fully
   represented. It is now validation-only and explicitly reports insufficient empirical
   resolution instead of manufacturing precision.
2. A raw, self-hashed calibration profile could carry a threshold or secondary operating point
   rebound to the wrong detector, validation set, or pipeline. Release now verifies these semantic
   bindings before opening held-out scores.
3. Status inspection could not be trusted if it generated the receipt that it then verified. The
   status path now uses a persisted, read-only verifier; a final whole-tree before/after hash check
   is the physical oracle.

## 4. Documentation synchronization

- Protocol amendment, one-click plan, bootstrap contract, notebook guard, README, execution
  runbook, protocol description, Colab design, and theory manuscript were aligned to the same
  three-pair/full-stage/lower-bound contract.
- DOCX metadata stores the current Markdown SHA-256. Word refreshed fields, TOC, pagination, and
  exported a 73-page PDF.
- Relative to the previously reviewed render, 71 pages were pixel-identical. Only pages 48 and 49
  changed; both were inspected at 150 DPI and had no observed clipping, overflow, broken tables, or
  malformed glyphs.

## 5. Verification sequence

1. Focused lint and regression tests for compute, continuation, detector locks, thresholds,
   bootstrap, and status behavior.
2. Full Ruff check and format check.
3. Full pytest suite with source-first imports and the local notebook-test dependency path.
4. Project validator and notebook contract checks.
5. Source-manifest and physical upload-bundle regeneration followed by independent verify-only
   runs.
6. Whole-tree before/after hash comparison around `bootstrap.py --mode status`.
7. Final stale-contract scan, artifact inventory, manifest verification, and document/PDF checks.

## 6. Final local receipts

- Full suite: 437 tests collected; 432 passed, 5 skipped, and 0 failed.
- Ruff: check passed; format check reported 54 files already formatted.
- Project validator: passed with both notebooks AST-parsed and the source/bundle contract intact.
- The first readiness run correctly exposed a non-reproducible system Python: base dependencies
  differed from `base.lock` and `ipykernel` was absent. No production gate was weakened. A
  temporary environment outside the workspace was created with the exact base lock and pinned
  notebook/test tools, after which fixture preflight, fixture smoke, notebook fixture, preview,
  validator, Ruff, and pytest all exited successfully.
- Final readiness state: 4 passed, 0 failed, 7 `unverified_external`. The strict local preflight
  remained external because the local machine did not realize the A100/accelerated contract.
- Physical status oracle: 90 files before and after, zero tree delta, zero session-marker delta,
  exit code 0, no run present, and no upload-verification receipt created.
- Theory package: source hash bound in DOCX; 459 paragraphs, 9 tables, 4 sections, and 73 nonblank
  PDF pages; 71 pages pixel-identical to the prior reviewed render and changed pages 48–49 visually
  inspected.
- All fixture/readiness artifacts and interpreter caches were moved to recoverable quarantine
  outside the payload. The final `artifacts/` directory contains only the source and bundle
  manifests.

A missing hosted, empirical, institutional, or supervisor gate is never upgraded from
`unverified_external` to PASS.

## 7. Report-asset implementation extension

- Confirmed that the proposal and Chapter 3 template require cohort/QC, backdoor validity,
  detector comparison, adaptive robustness, language transfer, and operational evidence.
- Added calibration/score-distribution evidence as a separate view so discrimination and
  probability quality are not collapsed into one detector ranking chart.
- Found a pre-release schema mismatch: the production detector writer records seed, split,
  training-unit, and execution-profile provenance that the strict reader previously rejected.
  The repair retains and validates those fields instead of deleting provenance.
- Extended the held-out detector release with paired lineage-level scores so ROC, precision-recall,
  score-distribution, and reliability views can be reconstructed from a completion-bound artifact.
- Kept fixture rendering explicitly marked and excluded from one-click completion evidence.
- Required exact generated-file inventory and deterministic source regeneration before accepting an
  existing report directory. A file added outside the manifest, or an output edited and then
  self-rehashed, now fails closed.
- One exploratory read command used a misspelled working directory and failed before reading or
  changing files. The strategy changed to smaller exact-path reads; no retry loop or state change
  occurred.
- The first full-suite run exposed a stale bundle manifest and a missing local notebook dependency.
  The manifest was regenerated by its owning builder, while notebook and exact base dependencies
  were installed only in a temporary directory outside the upload boundary. No verifier was
  weakened and the reproduced suite subsequently passed.
