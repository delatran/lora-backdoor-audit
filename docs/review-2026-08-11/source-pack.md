# Deep Experiment Review — Bounded Source Pack

> Historical pre-pilot snapshot. Amendment v4 and schema 9 supersede the v3
> continuation rule; see `../review-2026-08-11-pilot-v4/source-pack.md`.

## Objective, exact target, and authority

- Objective: bring the local `TranThienNhan_TTTN` experiment bundle to the strongest defensible
  pre-hosted state, with a scientifically complete compute gate, deterministic pilot selection,
  sealed-test detector contracts, synchronized submission-facing theory, and reproducible local
  verification.
- Exact executable target: `TranThienNhan_TTTN/lora-audit`, launched by
  `TranThienNhan_TTTN/RUN_EXPERIMENT.ipynb` and `TranThienNhan_TTTN/bootstrap.py`.
- Authority: local L3 edits and local L4 tests, lint, validation, document export, and read-only
  status checks.
- Non-goals: starting or controlling a Colab runtime, consuming compute units, training adapters,
  opening the real held-out test set, uploading files, publishing, submitting, or inventing
  empirical evidence.
- Risk tier: full mode. The review changes a confirmatory experiment protocol, hosted-runtime
  authorization logic, statistical release boundaries, and submission-facing documents.

## Source matrix

| Source family | Current source of truth | What it establishes | What it cannot establish |
|---|---|---|---|
| Local behavior and evidence policy | workspace and project `AGENTS.md`; `REPORT_WRITING_CONTRACT.md` | Authority, evidence labels, privacy boundary, closure gates | Hosted allocation or scientific outcomes |
| Registered academic design | `Tai_lieu_de_cuong/` | Project identity, research questions, planned hypotheses and scope | Observed effects or accepted final wording |
| Theory and methods | `Ly_thuyet/LY_THUYET_DO_AN.md`, synchronized DOCX and PDF | Pre-experiment rationale and protocol description | Chapter 3 results or supervisor approval |
| Operational amendment | `configs/protocol-amendment-v3.json` | Three-pair pilot, full-stage budget scope, reserve, continuation rule | That any pilot has run on the current source |
| One-click authorization | `configs/execution/one-click-colab-pro.json` | Project cap, gate parameters, amendment hash, fail-closed stage contract | A Google platform guarantee or purchased capacity |
| Executable implementation | `src/lora_audit/`, `bootstrap.py`, notebooks | Actual selection, budgeting, calibration, release, resume, and status behavior | Hosted behavior without a source-bound receipt |
| Deterministic verification | `tests/`, `scripts/validate_project.py`, manifest builders | Locally observed invariants covered by each verifier | Model quality, A100 throughput, or real cohort completion |
| Runtime evidence | source-bound ledgers, receipts, preflight and releases created by a future run | Allocation, telemetry, QC, release and empirical estimates when valid | Nothing until fresh current-source artifacts exist |
| Platform policy | Google Colab FAQ: <https://research.google.com/colaboratory/faq.html> | Resource limits are dynamic; general runtimes may run up to 12 hours; Pro+ can run continuously up to 24 hours when compute units are sufficient | A guaranteed duration for this project or session |

## Evidence hierarchy and trust boundary

1. Current source plus deterministic local verifier output is authoritative for local behavior.
2. A hosted claim additionally requires a receipt bound to the current source tree, source
   manifest, configuration, execution profile, hardware preflight, and run root.
3. A scientific claim additionally requires complete QC-valid cohort artifacts, sealed-test
   release, uncertainty estimates, and the predeclared analysis contract.
4. Proposal text, historical runtime files, fixture output, fluent prose, and prior PASS labels are
   routing context only until they satisfy the current bindings.

## Current evidence boundary

- The project authorizes at most 12 device-hours for its one-click research plan and reserves 20%,
  leaving a deterministic authorization gate of 9.6 device-hours.
- That cap is a project decision, not an observed Colab session duration and not a Google service
  guarantee.
- No fresh three-pair representative A100 pilot bound to this amended source exists in the local
  review scope. Any historical pilot bound to an older source/amendment is invalid for current
  authorization.
- Local verification may establish readiness to start the representative pilot. It cannot convert
  missing hosted telemetry, cohort results, or supervisor review into a pass.

## Recovery boundary

- Pre-change executable snapshot: local temporary snapshot
  `tttn-cook-baseline-20260811-1`, outside the upload boundary.
- Pre-change theory DOCX/PDF snapshot: local temporary snapshot
  `tttn-theory-docs-baseline-20260811-1`, outside the upload boundary.
- Both snapshots are outside the upload boundary. They support exact local rollback but do not
  constitute empirical evidence.

## Report-asset extension

- Exact output boundary: the existing `TranThienNhan_TTTN` directory. No sibling directory or
  archive is created for upload.
- Runtime source artifacts: completed cohort state, paired QC ledger, held-out detector release,
  adaptive release, four-cell language-transfer release, and sampled runtime telemetry.
- Intended output: a hash-bound result register, source-data tables, and seven report-ready
  SVG/PNG figure pairs under the completed run root.
- Presentation boundary: notebook source and visible figure text use descriptive natural-language
  labels. Internal research-question, hypothesis, schema, plan, and release identifiers remain in
  machine-readable evidence only where integrity requires them.
- Compute boundary: model training and inference remain strict A100 work. JSON validation, CSV
  writing, hashing, and deterministic Matplotlib rendering are host-CPU orchestration and do not
  weaken the accelerator contract.
- Scientific boundary: report assets are generated only from completed, current-source,
  non-fixture releases. Missing, malformed, stale, or synthetic production inputs block report
  completion rather than producing placeholder evidence.
