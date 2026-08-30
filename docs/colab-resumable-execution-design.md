# Granular Colab Execution and Full-Pipeline Budget Design

## Objective and authority

The owner authorized the hosted notebook to become the source-bound one-click
entrypoint for the complete locked experiment. This task may edit the upload
bundle and run local L3-L4 verification. It pre-authorizes the notebook's
  future execution only when the owner deliberately selects `Runtime > Run all`,
  under the exact plan and 12-device-hour cap stored in source. The deterministic
  pipeline decides whether to proceed within that ceiling; it cannot enlarge the
  ceiling or authorize spend. This task does not
authorize the current agent to launch Colab/A100, spend compute, train models,
publish, or perform any external mutation.

Exit conditions:

- `Runtime > Run all` executes every stage in dependency order without editing
  a switch or entering a budget.
- Three outcome-blind representative pilot pairs run sequentially and authorize
  later work only when a schema-9 full-pipeline estimate fits the 9.6-hour gate.
- Nine cohort cells each execute at most four exact receipt-bound lineages,
  providing bounded capacity for the 33 lineages remaining after the pilot.
- RQ1, every locked RQ2 condition, every locked RQ3 matrix cell, and their final
  release steps are independently invocable and resumable.
- A runtime reset requires a new live preflight; completed Drive artifacts do
  not silently authorize a new VM.
- Every strict stage is guarded by the single source-bound literal
  `RUN_ALL_EXPERIMENTS = True` contract.
- A final verifier revalidates cohort, RQ1, all RQ2 conditions, and all RQ3
  cells before writing `artifacts/one-click-completion.json`.
- Stage 0 and preflight enforce an algorithmic source gate only for
  byte-identical `literal_preserved` bindings tied to the exact private
  registry and Core-config hashes. Semantic translations are rejected as
  outside the locked protocol instead of opening an in-run review branch.
  Preflight builds and readback-validates both RQ3 private receipts
  deterministically.
- Existing scientific configuration, A100 gate, paired-lineage plan, and
  fixture/result evidence boundary remain unchanged.

Non-goals:

- Claiming that a managed Colab Pro backend is guaranteed to survive for 24
  hours or that Drive OAuth never needs interaction.
- Concurrent training processes on one accelerator.
- Post-observation changes to model, dataset, seeds, thresholds, batch sizes, or
  RQ estimands.
- Treating local verification or fixture execution as empirical evidence.

Risk tier: full, because this is an architectural change to a research
execution and evidence pipeline.

## Bounded source pack

| Source | Why it is load-bearing |
| --- | --- |
| `../RUN_EXPERIMENT.ipynb` | Hosted owner interface and all execution switches. |
| `../bootstrap.py` | Upload attestation, A100 capability gates, receipts, content checkpoints, final verification, and stage dispatch. |
| `configs/execution/one-click-colab-pro.json` | Exact Run-all scope, budget, batch topology, RQ order, and hash-bound owner-waiver gate. |
| `configs/protocol-amendment-v4.json` | Post-pilot outcome classification, unchanged QC thresholds, full-stage budget, and claim-boundary lock. |
| `src/lora_audit/compute.py` | Stage-aware estimate, no-double-count accounting, reserve, and deterministic recommendation. |
| `src/lora_audit/orchestration.py` | Exact lineage selection, durable cohort state, skip/resume, and failure ledger. |
| `src/lora_audit/adaptive.py` | Locked twelve-condition RQ2 plan and aggregate release. |
| `src/lora_audit/rq3.py` | Receipt-gated four-cell EN/VI transfer matrix and aggregate release. |
| `src/lora_audit/cli.py` | Project-native execution surface used by the bootstrap driver. |
| `scripts/validate_project.py` | Notebook structural, source-bound Run-all, budget, and stage-count contract. |
| `tests/test_training_telemetry.py` | Outer bootstrap and notebook safety regressions. |
| `tests/test_adaptive_rq3_state.py` | RQ2/RQ3 evidence and release invariants. |
| `docs/execution-runbook.md` | Operator recovery contract. |

Baseline note: the source tree is not a Git repository. The first local test
attempt lacked the project import path. A corrected source-path run reached 115
passes and 3 expected skips, with seven environment-baseline failures: four
because global Python lacks `nbclient`, and three from the global Typer/Click
runtime mishandling a required missing option. These are environment gaps, not
accepted product failures; final verification must use the project lock or
otherwise report them explicitly.

## Runtime decision-authority contract

The source of truth is plan schema 4 and policy
`non-human-runtime-decisions-v1`:

- `deterministic_protocol` controls stage order, fixed thresholds, exact unit
  selection, validation, skip/resume, release, and fail-closed stopping.
- The model is `measurement_only`; it cannot change the protocol, budget,
  threshold, order, evidence label, or acceptance rule after observing data.
- `human_in_run_decisions` and `manual_override_allowed` are both false.
- Unknown state fails closed and semantic trigger drift is rejected. Neither
  condition asks an operator to choose whether the result should pass.
- Google Drive OAuth, account compute balance, and runtime allocation remain
  external platform prerequisites. They are not scientific decision points,
  and the notebook cannot impersonate account consent or authorize spending.

The repository companion notebook is attestation-only. It has no preflight,
pilot, training, evaluation, release, or manual-switch surface. This prevents a
second interface from silently restoring the earlier human-gated execution
path.

## Decision

Use deterministic sequential units, not concurrent processes:

1. `preflight` prepares the runtime once and writes a temporary session marker
   bound to the live A100 preflight, current manifests, prepared data, and model
   cache receipt. The marker is lost with the VM.
   Before setup, it rejects a missing, expanded, or hash-drifted algorithmic
   literal-preservation gate.
   After prepared data exists, it deterministically builds and immediately
   revalidates the trigger-equivalence and common-probe receipts under
   `artifacts/private/`.
2. Later stage invocations verify that exact marker instead of reinstalling and
   redownloading before every small unit.
3. The pilot selector chooses three lineages using only preregistered metadata
    and maximum factor coverage. It runs one pair at a time, reconstructs the
    estimate after each pair, and continues whenever an optimistic lower bound
    proves that the remaining pairs can still restore full-pipeline budget fit.
    It stops early only for failed QC, protocol drift, or an already-infeasible
    lower bound.
4. A cohort batch resolves the next pending identifiers in plan order, writes
   those exact identifiers into a fresh execution receipt, and invokes the CLI
   with repeated `--lineage-id`. Nine visible notebook cells use batches of at
   most four and never use ambiguous `--max-lineages`.
5. Each RQ2 condition writes a hash-bound condition result before the aggregate
   is released. A completed condition can be validated and skipped.
6. Each RQ3 source/probe cell writes its own strict result. A transfer cell can
   run only after its same-language source cell exists. Finalization validates
   all four cells before writing the matrix.
7. One literal `RUN_ALL_EXPERIMENTS = True` declaration guards every strict
   call. The plan fixes 12 available A100 device-hours, a 20% reserve, nine
   cohort invocations, twelve RQ2 indices, and four dependency-ordered RQ3
   cells. Pilot telemetry must authorize the entire remaining stage ledger, not
   only Core training.
8. `verify-results` re-runs both strict aggregate finalizers, re-hashes the pilot
   ledgers, validates the schema-9 `full_research` authorization and exact
   budget, requires all granular results, and writes the sole completion receipt.
9. Result preservation writes content-addressed manifests instead of copying
   the growing adapter tree after every unit. Raw outputs remain durable in
   their Drive paths and are still receipt/hash validated on resume.
10. `status` uses a read-only upload-contract verifier and never creates an
    attestation receipt or session marker.

Rejected alternatives:

- Duplicating one long cell without durable state: output is easier to read but
  recovery remains unsafe.
- `--max-lineages`: prior regression evidence showed ambiguous selection and
  receipt/resume risk. Exact identifiers are required.
- Parallel adapter processes: they compete for one GPU, undermine telemetry,
  and expand the failure domain.
- `nohup` or `tmux`: they cannot survive deletion of the managed Colab VM.
- Browser keep-alive or reconnect scripts: they cannot override Google's
  backend lifetime and risk evading managed-service controls.
- A claimed one-click guarantee on Colab Pro: Google's official FAQ says
  limits and resource availability vary and notebooks generally run for at
  most 12 hours. Pro+ advertises up to 24 hours only when compute units are
  sufficient. Dedicated infrastructure is the distinguishing alternative.

## Claim ledger and acceptance checks

| Claim | Owner | Deterministic verifier | Status |
| --- | --- | --- | --- |
| A cohort invocation can execute only its exact receipt-bound IDs. | Bootstrap/orchestration | Unit tests inspect receipt and CLI arguments; orchestration receipt mismatch tests. | Verified locally |
| A reset VM cannot reuse a stale live-preflight session. | Bootstrap | Session-marker binding tests and missing-marker failure test. | Verified locally |
| Completed RQ2 conditions are validated before skip/finalize. | Adaptive module | Partial-run, tamper, resume, and twelve-condition finalization tests. | Verified locally |
| RQ3 cells are independently durable without changing the 2x2 estimand. | RQ3 module | Per-cell ordering, tamper, and four-cell finalization tests. | Verified locally |
| Hosted notebook runs the complete locked sequence under one source-bound switch. | Notebook/validator | AST parsing, 47/40 topology, exact stage counts, literal-true guard, and plan equality. | Verified locally |
| No hosted stage asks a human or model to choose protocol behavior after execution starts. | Plan/notebooks/bootstrap | Exact schema-3 decision-policy equality, forbidden interactive-token checks, attestation-only companion contract, and policy-drift tests. | Verified locally |
| Pilot sampling is outcome blind and later work requires a complete three-pair prefix. | Cohort/compute/bootstrap | Deterministic factor-coverage selector tests, prefix binding, pair-count checks, and early-stop tests. | Verified locally; fresh A100 telemetry unverified |
| A Core-only timing fit cannot be promoted to full research authorization. | Compute/readiness/bootstrap | Schema-8 stage-ledger reconstruction, 80% gate, no-double-count tests, and every-stage authorization recheck. | Verified locally; empirical fit unverified |
| Status cannot mint an attestation receipt. | Bootstrap | Before/after artifact-tree regression plus the read-only verifier path. | Verified locally |
| A completion claim cannot precede all study units or checkpoint persistence. | Bootstrap/finalizers | Final verifier test, aggregate receipt revalidation, exact result counts, checkpoint-failure regression, and checkpoint-bound completion receipt. | Verified locally |
| The algorithmic source gate cannot cover semantic translation, attestation substitution, or hash drift. | Notebook/bootstrap/receipts | Stage-0 gate, exact registry/config hashes, exact consumer attestation, gate regression, semantic-translation rejection, and receipt readback. | Verified locally |
| Checkpointing does not repeatedly duplicate adapter files on Drive. | Bootstrap | Content-manifest regression test and absence of snapshot copies. | Verified locally |
| Candidate scoring can reuse prompt computation without changing the locked conditional token-mean log-likelihood estimand. | Prompt/evaluation modules | Scalar oracle, tiny Qwen full-sequence comparison, cache-only negative control, and one-record live-model equivalence guard before each adapter evaluation. | Verified locally; A100 throughput remains unverified |
| Candidate continuation memory remains bounded independently of the full intent-label count. | Prompt module | Regression test limits each cache-expanded continuation call to eight candidates per record and confirms fewer than half the reference input-token evaluations on the deterministic fixture. | Verified locally |
| The hosted notebook presents research phases and progress without operational refusal labels or internal research-question identifiers. | Operator notebook/validator | 47/40 topology, seven markdown sections, hidden infrastructure cell, presentation-safe bootstrap aliases, curated progress renderer, and a case-insensitive zero-occurrence regression over the complete notebook source. | Verified locally |
| Local readiness remains distinct from empirical evidence. | Runbook/readiness audit | Project validator and readiness audit; no external run occurs. | Local controls verified; A100 evidence unverified |
| One click will survive any Colab Pro backend event. | Google-managed service | Only a live external run can observe a particular session; official limits remain dynamic. | Unverified and not guaranteed |

## Worklog

- 2026-08-02: objective, source boundary, baseline gaps, decision, and
  acceptance checks recorded before implementation.
- 2026-08-02: implemented a temporary live-VM marker, exact one-to-four-lineage
  cohort batches, independent RQ1/RQ2/RQ3 commands, strict per-condition and
  per-cell result receipts, read-only status, and a 39-cell hosted notebook
  with 32 AST-validated code cells.
- 2026-08-02: the pinned local verification target collected 370 tests; the
  full suite completed with 365 passes and 5 skips. Project validation passed,
  fixture preflight passed, and the fixture notebook executed successfully.
- 2026-08-02: readiness RG-01 through RG-04 passed. RG-05 remained failed on
  the Windows verifier host because it has no A100, lacks the accelerated lock,
  and reported less than the strict 25 GiB disk floor. RG-06 through RG-11
  remain unverified external evidence. No adapter training or A100 run occurred.
- 2026-08-02: readiness commands were changed to disable Ruff and pytest cache
  writes so a local audit no longer makes the subsequent upload attestation
  fail on its own generated cache directories.
- 2026-08-02: the owner changed the execution contract from manually enabled
  stages to one-click Run all. Added the source-bound Colab Pro plan, nine
  four-lineage cohort cells, automatic RQ1/RQ2/RQ3 execution, a strict final
  completion receipt, and manifest-only result checkpoints. Local verification
  still does not establish A100 execution or empirical results.
- 2026-08-03: added a fail-early RQ3 trigger gate bound to the current private
  registry and Core-config SHA-256 values. The owner subsequently waived a
  separate human review. The waiver is encoded as
  `owner_waived_human_review`, produces a `waived_literal_preserved` receipt,
  and is rejected for semantic translation or hash drift. No approval was
  fabricated.
- 2026-08-03: physical-bundle verification twice detected verifier-generated
  cache state: first Python/pytest/Ruff caches, then a Ruff format-check cache.
  The repeated cleanup loop was stopped. Caches were quarantined outside the
  bundle, and subsequent lint commands must route `RUFF_CACHE_DIR` to a named
  temporary directory while Python and pytest keep cache writes disabled.
- 2026-08-03: final one-click verification observed 47 notebook cells with 40
  AST-valid code cells, 380 collected tests with zero failures, zero errors,
  and five skips, a passing project validator, a passing full lint check, and
  a deterministic 81-entry upload archive round trip. The final source tree
  contained 77 manifest-bound files. Generated fixture/readiness artifacts
  were quarantined outside the upload boundary, leaving only the source and
  bundle manifests under `artifacts/`.
- 2026-08-03: the readiness audit was repeated against the exact packaged
  bundle on a local volume with more than the strict 25 GiB disk floor. It
  reported `awaiting_external_evidence`, four passed local gates, zero failed
  gates, and seven `unverified_external` gates. This isolates the earlier
  local RG-05 failure to verifier-host disk capacity; A100 availability,
  accelerated-runtime realization, pilot telemetry, and empirical releases
  remain intentionally unverified. No paid or live A100 execution occurred.
- 2026-08-03: independent verification found two fail-closed gaps. RQ3
  consumers now require the waiver receipt attestation to equal the exact
  source-gate attestation, not merely any valid SHA-256. Final verification now
  writes and readback-validates a completion-excluding result checkpoint before
  atomically writing the checkpoint-bound completion marker. Regression tests
  cover substituted waiver attestations and checkpoint failure before success.
- 2026-08-03: the owner requested zero in-run human decisions. Plan schema 2
  adds the exact `non-human-runtime-decisions-v1` contract, limits the model to
  measurement, rejects unknown state and semantic trigger drift automatically,
  and forbids manual override. The former repository pilot notebook was reduced
  to an attestation-only companion. The locked Python 3.12 suite passed 384
  tests with zero failures, zero errors, and five environment-dependent skips;
  project validation observed the unchanged 47/40 outer topology and the new
  4/2 attestation-only companion topology. No Colab/A100 run or empirical
  result was created by this source change.
- 2026-08-10: a source- and bundle-matched A100 pilot completed one clean and
  backdoored adapter pair but measured 14.540961 Core planning device-hours,
  above the source-bound 9.6-hour budget gate. Evaluation accounted for
  12.554681 hours and reached 77.9336 GiB peak reserved memory on the observed
  79.2507 GiB A100. The cohort correctly stopped before training because the
  estimate did not authorize Core. These values describe that pilot only and
  are not a completed-study result.
- 2026-08-10: candidate scoring was changed to evaluate each record prompt once
  and score bounded candidate-continuation chunks from cloned KV caches. The
  mathematical estimand, label set, dataset, seeds, 36 paired lineages, and
  12-hour authorization remain unchanged. Each adapter now compares the cached
  scorer against the full-sequence reference on one live-model record and
  fails before measurements if the predicted label differs or any candidate
  score exceeds the declared numerical tolerance. Local scalar, cache-negative,
  batching, and tiny-Qwen tests pass; the speedup and revised compute estimate
  remain unverified until a fresh A100 pilot.
- 2026-08-10: the hosted notebook was restored to the documented 47-cell,
  40-code-cell topology with six descriptive research sections. Normal progress
  output now uses source validation, pilot measurement, Core experiments,
  detector comparison, adaptive robustness, cross-language transfer, and final
  verification terminology. Every long-running stage routes through the same
  bounded child-output capture helper. A source-bound notebook contract now
  exposes presentation-safe stage aliases and progress fields, while legacy
  CLI identifiers remain available only behind the bootstrap compatibility
  layer. A case-insensitive scan confirms that the hosted notebook source has
  no operational refusal labels or internal research-question identifiers.
- 2026-08-11: locked `pre-hosted-execution-amendment-v3` before detector
  validation or held-out release. The one-pair pilot was replaced by three
  outcome-blind representative pairs, and the compute gate now covers observed
  pilot consumption, remaining Core work, all RQ2/RQ3 GPU units, storage,
  finalization, and a 20% managed-runtime reserve without double-counting.
  `core_only` cannot authorize later research stages.
- 2026-08-11: the earlier A100 pilot remains historical runtime evidence for its
  exact old scorer/source only. It cannot authorize the current source because
  candidate scoring and the compute contract changed. A fresh representative
  A100 pilot is still required; no empirical release was fabricated locally.

## External service boundary

Primary current source: Google Colab FAQ,
https://research.google.com/colaboratory/faq.html, checked 2026-08-11. It states
that managed resource limits and GPU availability vary; notebooks generally
run for at most 12 hours; Pro+ supports continuous execution for up to 24 hours
when compute units are sufficient; and dedicated resources are the path for
relaxing managed runtime limits. Therefore the locally verifiable claim is
one-click sequential orchestration plus safe rerun/resume, not guaranteed
single-session completion on Colab Pro.
