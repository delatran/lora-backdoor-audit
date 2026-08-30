# Deep Experiment Review — Change Risk Packet

## Change contract

- `change_class`: local L3 source/document edits and L4 non-destructive verification.
- `exact_target`: `TranThienNhan_TTTN`; no similarly titled or adjacent project is in scope.
- `prohibited_side_effects`: Colab execution or control, compute-unit consumption, upload,
  publication, submission, permission change, held-out real-test opening, and trigger disclosure.
- `rollback`: executable and theory snapshots listed in `source-pack.md`.

## Material changes and blast radius

| Change | Intended effect | Blast radius | Fail-closed verifier |
|---|---|---|---|
| Schema-8 full-pipeline compute model | Budget every GPU-bearing stage and reserve non-GPU/managed-runtime risk | Hosted authorization and later stages | unit tests, project validator, plan/amendment hash |
| Three-pair representative pilot | Reduce single-lineage planning fragility while remaining outcome-blind | Pilot cost and cohort authorization | deterministic selection and prefix-binding tests |
| Optimistic lower-bound continuation gate | Avoid falsely stopping after a recoverable interim `core_only` estimate | Pilot sequence and cost | recoverable/infeasible counterexample tests |
| Source/config/run-root binding | Prevent stale pilot evidence from authorizing changed code | Resume and hosted artifact reuse | receipt and readiness binding tests |
| Validation-only secondary operating point | Implement the proposal's FPR-constrained view without test leakage | Detector calibration and reported metrics | resolution, selection, scan, and release tests |
| Release binding checks | Reject detector/pipeline/data lock substitution | Sealed-test integrity | self-hashed tamper regression tests |
| Read-only status verifier | Prevent an inspection command from silently mutating the upload payload | Manual upload integrity | physical tree-hash before/after comparison |
| Theory DOCX/PDF synchronization | Keep the submission-facing method consistent with executable protocol | Pages 48–49 and source metadata | source hash, structure checks, 73-page render comparison |
| Detector-evidence provenance schema | Make the strict reader accept exactly the fields emitted by the real paired evaluator | Calibration and held-out release | writer-shaped validation and pair-invariant regression tests |
| Paired held-out score distribution | Make detector plots reconstructible without reopening or recomputing test inference | Detector release schema and artifact size | score/label/prediction shape checks and release tests |
| Result register and seven figure pairs | Produce Chapter 3 assets from the same completed releases | Final verification and report presentation | source hashes, staging/atomic publish, visible-text scan, fixture marker, figure-manifest validation |
| Exact report-output regeneration | Reject extra files and self-rehashed edits that are not reproducible from the bound releases | Resume, completion, and report integrity | exact inventory comparison, deterministic regeneration, tamper regressions |
| Sanitized notebook contract and final display | Remove visible internal/version identifiers and show final SVGs automatically | Outer one-click notebook only | AST parse, exact cell topology, forbidden-label regression, safe-path display checks |

## Preconditions, canaries, and rollback

- The current source and exact project identity were inspected before editing.
- No Git repository is present; rollback therefore uses byte snapshots, not unstated Git recovery.
- Targeted lint and regression tests precede the full suite.
- Source and bundle manifests are regenerated only after all source, test, notebook, documentation,
  and review-ledger changes are final.
- `status` is run only after manifests are fresh, with full file-tree hashes captured before and
  after. Any delta is a No-Go.
- If a final gate fails, the affected claim remains open; hosted execution is not authorized from a
  partial local pass.

## Residual risks that cannot be removed locally

- Colab availability, accelerator assignment, runtime lifetime, compute-unit balance, preemption,
  and storage/network behavior remain external and dynamic.
- Three pilot pairs improve planning robustness but do not estimate a stable population runtime
  quantile.
- A 12-device-hour project cap can still be too small; the controller must stop rather than shrink
  the scientific protocol silently.
- The confirmatory cohort, RQ2/RQ3 outputs, held-out release, uncertainty, Chapter 3, institutional
  similarity check, current official PTIT formatting, independent review, and supervisor approval
  remain open.

## Decision rule

- `GO` means only that the final local gates permit a fresh, representative, source-bound A100
  pilot to be attempted.
- `NO-GO` remains mandatory for full-run authorization until the completed three-pair estimate says
  `full_research` and its continuation action says `authorize_full_research`.
- `NO-GO` remains mandatory for empirical or submission claims until their distinct external and
  scientific evidence gates close.

## Final locally observed gates

- Full pytest: 437 collected, 432 passed, 5 intentional skips, 0 failed.
- Ruff: check passed and all 54 configured files were already formatted.
- Project validator: passed; 86 source entries, 88 bundle-manifest entries, both notebooks parsed,
  and fixture outputs were checked only in the quarantined fixture run.
- Readiness audit in an exact base-lock temporary environment: 4 passed, 0 failed, 7
  `unverified_external`. The external gates cover accelerated dependencies/device, strict hosted
  preflight, observed A100, hardware contract, representative pilot, hosted source receipt, and
  any bounded source delta.
- Read-only status physical oracle: 90 files before and after, zero byte/hash/path changes, zero
  upload-session marker changes, and no upload-verification receipt created.
- Fixture, preflight, readiness, cache, and run outputs were quarantined outside the upload payload;
  only the two canonical manifests remain under `artifacts/`.
