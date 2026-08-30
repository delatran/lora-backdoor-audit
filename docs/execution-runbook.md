# Hosted Execution Runbook

1. Upload the complete `TranThienNhan_TTTN` directory to private persistent
   storage and open the outer `RUN_EXPERIMENT.ipynb`.
2. Select an NVIDIA A100 40GB runtime. The immutable
   `configs/execution/a100-40gb.yaml` gate checks the allowlisted device name,
   CUDA backend, compute capability `8.0`, observed memory, and a live finite
   BF16 tensor operation. The product label alone is not evidence.
3. Ensure the account has a positive compute-unit balance and sufficient
   private Drive capacity. The clean source-bound plan is
   `configs/execution/one-click-colab-pro.json`: plan schema 4 enables Run all,
   caps the attempt at 12 A100 device-hours, reserves 20%, binds the locked
   `configs/protocol-amendment-v4.json`, selects three representative pilot
   pairs, sets a four-lineage batch maximum, fixes nine cohort invocations, and
   locks every RQ2/RQ3 selector.
4. Select `Runtime > Run all` once. Plan schema 4 retains the
   `non-human-runtime-decisions-v1` policy: deterministic code proceeds,
   resumes, skips a validated unit, or fails closed without asking the operator
   to choose a scientific outcome. The RQ3 source gate accepts only
   byte-identical `literal_preserved` EN/VI bindings; semantic translation or
   registry/config drift is rejected automatically before GPU setup. Complete
   the Google Drive authorization prompt if Colab presents one; the notebook
   cannot bypass OAuth or account-level platform consent.
5. Do not edit a cell while the run is active. The notebook executes, in order:
   attestation, status, strict preflight, status, three-pair representative
   pilot, status, nine
   exact cohort invocations, status, RQ1, status, twelve RQ2 conditions, RQ2
   finalization, status, four RQ3 cells, RQ3 finalization, final verification,
   final status, and display of the validated report figures.
6. Treat the run as complete only when
   `artifacts/one-click-completion.json` exists with `status: complete`. The
   final verifier recreates both aggregate releases only after all granular
   receipts and referenced artifacts revalidate. It then atomically creates the
   result register, seven source tables, and seven SVG/PNG figure pairs under
   `<run-root>/report-assets/`; fixture-marked, hash-mismatched, inventory-drifted,
   or non-reproducible self-rehashed assets are rejected. It writes and
   readback-checks a content-addressed checkpoint under
   `run-output/checkpoints/` first, then writes the checkpoint-bound completion
   marker as the final success signal.
7. If Colab terminates the VM, reconnect to an exact A100 40GB runtime and
   select `Runtime > Run all` again. The new VM must re-attest and re-run
   preflight; every already completed unit is validated and skipped.

Google does not guarantee a 24-hour Colab Pro session. Its
[official FAQ](https://research.google.com/colaboratory/faq.html), checked on
2026-08-11, says managed limits and hardware availability vary, notebooks
generally run for at most 12 hours, and only Pro+ advertises continuous
execution for up to 24 hours when compute units are sufficient. Keeping the browser or home
computer awake cannot override backend termination, GPU availability, Drive
quota, or compute-unit exhaustion. For a literal single launch that must
survive beyond the managed limit, use dedicated infrastructure rather than a
managed Colab Pro VM.

The first notebook stage runs `python bootstrap.py --mode attest`. It starts no
environment, download, training, or study work; on a pristine upload it first
verifies the exact physical file set and writes a deterministic receipt. If a
Colab runtime resets after generated artifacts exist, run `attest` and then
`preflight` before resuming. Attestation rechecks every manifest-bound byte and
permits only a constrained data-only generated-artifact overlay; source code,
executable payloads, and bytecode caches are rejected. Preflight recreates a
temporary live-session marker bound to the new VM and current evidence; a
persisted preflight file alone cannot authorize execution.

Before strict setup, preflight rechecks the algorithmic source-plan trigger gate against
the current private-registry and Core-config hashes. It then generates and
readback-validates the `waived_literal_preserved` trigger-equivalence receipt
and the deterministic common-probe receipt under `artifacts/private/`. RQ3
commands receive those paths and the exact source-gate attestation explicitly;
a semantic translation, missing gate, mismatched attestation, changed hash,
or tampered input fails closed.
Do not run the pristine physical `build_upload_bundle.py --verify-only` gate
against a runtime-mutated Drive tree.

Do not edit execution arguments in the notebook. The 12-hour value is the
owner-authorized maximum exposure, not a value the model may expand. The
pipeline decides whether to continue within that ceiling by requiring the
full-stage estimate to fit the 9.6-hour gate; it can stop early but cannot spend,
change scope, or weaken acceptance rules. Locked values live in the
validated `training` and `evaluation` sections of `configs/core.yaml`. Runtime
requirements live only in the selected immutable execution profile. A change to
either contract invalidates earlier preflight, telemetry, and resume evidence.
An L4 profile, receipt, pilot, or estimate can never authorize the A100 run.
Evaluation batches are atomic and hash-bound. An incompatible
shape, label, or non-finite score is an error; there is no silent smaller-batch
fallback.

The exact strict gate is:

```bash
lora-audit preflight --config configs/core.yaml --execution-profile configs/execution/a100-40gb.yaml --mode strict --output artifacts/preflight-live.json
```

The driver derives a run root from both the current scientific config hash and
source-tree hash, preventing stale source-bound ledgers from being resumed after
a code change. The pilot selects exactly three paired lineages by outcome-blind
maximum factor coverage and executes them sequentially. After each pair it
records two training rows plus one pair-level QC/evaluation row and recreates
`artifacts/compute-estimate.json` from both observed ledgers. Before pair three,
the schema-9 `pilot_continuation` decision computes the most optimistic budget
lower bound obtainable by replacing every remaining pilot pair's two planned
training units and one planned evaluation unit with zero observed cost. The run
continues while that bound can still fit, including an interim `core_only`
recommendation. A threshold-only QC failure is retained with `qc_valid: false`
but does not stop bounded execution. An integrity failure, unknown reason code,
protocol drift, or a lower bound above the gate stops it. After pair three, only
`full_research` with `authorize_full_research` authorizes later stages.

The schema-9 estimate has `scope: full_research_pipeline`. It counts observed
pilot consumption once, subtracts completed pilot adapters and pairs from the
remaining Core work, and includes remaining Core training/evaluation, all 12
RQ2 training/evaluation units, all 24 RQ3 evaluation units, storage,
finalization, and a 20% managed-runtime reserve. It records both ledger hashes,
exact lineage-to-adapter-hash bindings, and the source tree and manifest hashes
used by the pilot. It also binds total, threshold-only, and integrity failure
counts and the canonical QC reason-code summary. Canonical bundle-relative ledger paths require a byte re-hash
and full ledger-derived recomputation before the estimate can authorize later
work. Peak allocated and reserved memory, observed device capacity, and bounded
memory fractions are required for both training and evaluation. A separately
authorized cohort must use the schema-4 execution receipt whose
`compute_authorization` object binds the exact estimate SHA-256, declared
device-hour budget, profile, adapter-run count, and `full_research`
recommendation. The cohort stage generates that receipt only after the
source-bound 12-device-hour cap and 9.6-device-hour gate revalidate, and binds
only the next exact batch of pending lineage IDs.
The hosted one-click batch size and bounded maximum are both four; the direct
bootstrap CLI may use a smaller batch without widening the budget.

Threshold continuation is not an attack-success waiver. Threshold-failed pairs
remain in the planned denominators, are excluded from QC-valid downstream input,
and can still make a later RQ stage non-estimable or fail closed. The completion
marker is the only signal that every required downstream release was produced.

On the first authorized execution, cohort state is irreversibly bound to the
verified source-tree and source-manifest hashes. Every adapter training receipt
and pair metadata artifact inherits both hashes. Resume or skip is refused when
either source hash changes, even if the configuration and prepared data remain
unchanged. A later adaptive continuation must also prove that its baseline
adapter, completed training receipt, pair-level QC, records, and source hashes
all refer to the same locked lineage.

The exact pilot command includes all execution capabilities and fails closed if
any byte or scope has drifted:

```bash
CORE_RUN_ROOT=artifacts/runs/core-<config-hash>-<source-tree-hash>
PILOT_LINEAGE_ID=<one-of-three-deterministically-selected-lineage-ids>
lora-audit run-cohort --config configs/core.yaml --execution-profile configs/execution/a100-40gb.yaml --prepared-manifest $CORE_RUN_ROOT/data/manifest.jsonl --output-root $CORE_RUN_ROOT --lineage-id $PILOT_LINEAGE_ID --execution-phase pilot --preflight-artifact artifacts/preflight-live.json --execution-receipt artifacts/execution-receipt.json --source-manifest artifacts/source-manifest.json --bundle-manifest artifacts/bundle-manifest.json --execute
lora-audit estimate-compute --config configs/core.yaml --execution-profile configs/execution/a100-40gb.yaml --source-manifest artifacts/source-manifest.json --pilot-ledger $CORE_RUN_ROOT/ledgers/pilot-device-hours.jsonl --evaluation-ledger $CORE_RUN_ROOT/ledgers/pilot-evaluation-hours.jsonl --output artifacts/compute-estimate.json
```

The bootstrap repeats the two commands for the three selected IDs and validates
the deterministic telemetry prefix after every estimate. Operators should use
the notebook rather than manually choosing IDs.

`--lineage-id` may be repeated for an ordered explicitly authorized set and is
mutually exclusive with `--max-lineages`. The receipt must list exactly the
same pending identifiers in the same order; any mismatch is rejected before
cohort state is written or training begins.

Failure recovery:

- Device name, backend, capability, memory, or BF16 mismatch: stop and request
  the exact A100 40GB execution device. Never select a fallback.
- Dependency failure: preserve the installation error and preflight artifact;
  never loosen pins silently.
- Config or revision mismatch: restore the locked source or create an explicit
  protocol amendment.
- Training exception: inspect `ledgers/failed-runs.jsonl` and
  `cohort/run-state.json`, fix only the bounded cause, then rerun the notebook.
- Pilot interruption: preserve evidence, inspect the bounded cause, and remove
  only the explicitly approved incomplete adapter or evaluation directory
  before retrying. Timed pilot execution never resumes partial work because
  doing so would understate complete training or evaluation runtime.
- Cohort interruption: reconnect to an exact A100 and select `Run all` again.
  Verified completed output is skipped and incomplete work may resume from its
  latest training checkpoint.
- RQ2 or RQ3 interruption: reconnect and select `Run all` again. Earlier cells
  become validated no-ops; a condition or matrix cell is skipped only after its
  receipt, hashes, bindings, and referenced artifacts revalidate.
- Colab runtime reset: run `python bootstrap.py --mode attest` followed by the
  strict `preflight`; do not regenerate manifests or treat a persisted receipt
  as sufficient by itself.

Before external source synchronization, generate and verify
`artifacts/source-manifest.json`, then refresh the derived outer manifest from
that verified source snapshot:

```bash
python scripts/build_source_manifest.py
python scripts/build_source_manifest.py --verify
python scripts/build_upload_bundle.py --bundle-root .. --refresh-bundle-manifest
python scripts/build_upload_bundle.py --bundle-root .. --verify-only
```

A destination listing is not a content verifier; retrieve file bytes and
recompute every hash. If the verified destination tree is older, generate
`artifacts/source-delta-manifest.json` and obtain exact authorization for that
delta. Delta generation is always no-write. The shipped
`artifacts/bundle-manifest.json` binds every current inner source entry, the
exact outer `bootstrap.py` bytes, and a deterministic source projection of the
outer notebook. The notebook projection retains cell type, ID, order, source,
stable cell metadata, A100 metadata, and notebook format while excluding only
execution counters, outputs, and known volatile Colab display metadata. This
allows Cell 1 to attest after Colab records its execution state without
allowing cell-source or A100-contract drift. Bootstrap never regenerates either
manifest; it verifies the uploaded contract before dependency installation or
any runtime mutation. Counts are derived from the manifest rather than
hard-coded. An ignored cache or unexpected file still fails the exact upload-set
gate.

The staged bundle commands are:

```bash
python lora-audit/scripts/build_upload_bundle.py --bundle-root . --output <outside>/TranThienNhan_TTTN-colab-upload.zip
python bootstrap.py --mode attest
python bootstrap.py --mode preflight --execution-profile configs/execution/a100-40gb.yaml
python bootstrap.py --mode pilot --execution-profile configs/execution/a100-40gb.yaml --execute
python bootstrap.py --mode cohort --execution-profile configs/execution/a100-40gb.yaml --available-device-hours 12 --cohort-batch-size 4 --execute
python bootstrap.py --mode rq1 --execution-profile configs/execution/a100-40gb.yaml --execute
python bootstrap.py --mode adaptive-robustness-condition --execution-profile configs/execution/a100-40gb.yaml --condition-index 0 --execute
python bootstrap.py --mode rq2-finalize --execution-profile configs/execution/a100-40gb.yaml --execute
python bootstrap.py --mode cross-language-transfer-cell --execution-profile configs/execution/a100-40gb.yaml --source-language en-US --probe-language en-US --execute
python bootstrap.py --mode rq3-finalize --execution-profile configs/execution/a100-40gb.yaml --execute
python bootstrap.py --mode verify-results --execution-profile configs/execution/a100-40gb.yaml --execute
python bootstrap.py --mode status
```

`--mode status` is strictly read-only. It verifies the persisted upload contract
without creating an upload attestation receipt, session marker, manifest,
estimate, or result. Use `attest` explicitly when a new receipt is intended.

Every granular research stage first re-verifies every completed lineage
manifest, adapter configuration, adapter hash, training receipt, and QC state.
It also re-hashes both pilot ledgers and reconstructs the schema-9 full-pipeline
authorization. Only then may RQ1 release, one exact RQ2 adaptive condition, or
one exact RQ3 EN/VI cell proceed. RQ2 records normalization,
baseline/adaptive gaps, surrogate-transfer
gaps, and observed device-hours. RQ2 remains an adaptive-surrogate stress test,
not a fully detector-aware attack. RQ3 excludes failed-QC pairs and refuses a
matrix if either source language has no QC-valid held-out pair, the common
EN/VI probes drift, or the private trigger-equivalence receipt is absent,
tampered, or not bound to the exact algorithmic literal-preservation gate.

Before handoff, run:

```bash
python scripts/audit_execution_readiness.py
```

The audit may report `awaiting_external_evidence` while every local gate is
green. That is the correct pre-upload state, not a failure. `pilot_observed`
requires a hash-matched live A100 preflight and a config/design/profile-bound
three-pair representative estimate with `full_research` authorization.
`cohort_complete` and `research_released` require their
respective verified raw artifacts. No local fixture, mock, allocation label, or
notebook metadata may advance those states.
