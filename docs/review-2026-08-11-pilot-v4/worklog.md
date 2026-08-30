# Pilot Outcome Amendment v4 — Verification Worklog

## Implemented

- Preserved all configured QC thresholds and the original `qc_valid` result.
- Added `qc_reason_codes` to source-bound pilot evaluation telemetry.
- Added total, threshold-only, and integrity failure counts to schema-9 compute evidence.
- Changed the continuation decision to block only integrity or unknown failures while retaining threshold failures.
- Bound hosted plan schema 4 to the post-pilot amendment and prohibited v3 artifact resume.
- Added regressions at compute classification and cohort authorization boundaries.

## Deterministic verification completed

- Focused compute, pilot-controller, readiness, and cohort-authorization
  regressions passed. The synthetic threshold-failure path obtained
  `full_research`; the synthetic `zero_eligible_rows` path was rejected before
  cohort run state was created.
- The locked verification environment used Python 3.12.1, Torch 2.13.0,
  nbclient 0.11.0, Matplotlib 3.10.1, and the source-first package path.
- Full suite: 457 collected, 451 passed, 6 explicitly skipped, 0 failed.
- The system Python baseline was not accepted as a verifier: it had Torch 2.5.1
  and no nbclient, producing seven dependency-surface failures. No product gate
  was weakened to accommodate that drift.
- Full Ruff check passed; format check reported 53 files already formatted.
- Project validation passed, including 47/40 outer notebook cells/code cells,
  3/2 inner notebook cells/code cells, Core 36, Target 60, and sensitive-file
  scanning.

## Final packaging gate

The final source-manifest refresh, bundle-manifest refresh, clean archive build,
and independent archive round-trip verification occur after this worklog is
sealed so that its bytes are included in the final source identity. Their exact
hashes are reported outside the archive to avoid a circular self-hash.

No hosted execution result is claimed by this local worklog.
