# Pilot Outcome Amendment v4 — Change Risk Packet

## Change boundary

- Add QC reason codes to strict pilot evaluation telemetry.
- Classify failed pairs as threshold-only or integrity failures.
- Permit threshold-only failures to remain empirical outcomes while the
  full-stage budget decision proceeds.
- Keep integrity and unknown failures fail closed.
- Bump compute evidence to schema 9 and hosted plan to schema 4.
- Bind the hosted plan to amendment v4 and require a fresh source-bound run.

No model, dataset, trigger, poison rate, seed, QC threshold, RQ selector, A100
requirement, device-hour ceiling, or held-out release rule changes.

## Blast radius

The change affects pilot telemetry, compute-estimate reconstruction, cohort
authorization, readiness status, bootstrap plan validation, documentation,
tests, and both persisted source manifests. Old source-bound run artifacts
cannot be resumed.

## Failure modes and deterministic controls

| Failure mode | Control |
| --- | --- |
| Missing or contradictory reason codes | Exact telemetry fields and `qc_valid`/reason consistency check. |
| Unknown reason silently treated as scientific | Unknown codes classify as integrity failures. |
| Hand-edited compute estimate | Both ledgers are re-hashed, reparsed, aligned, and the estimate is rebuilt byte-for-byte. |
| Failed pair silently enters downstream evidence | Existing QC-valid lineage selector remains unchanged. |
| Old v3 artifacts are reused | Source and manifest hashes change; amendment declares restart; execution gates reject drift. |
| Budget scope is silently reduced | `full_research_pipeline`, the 9.6-hour gate, and `full_research` recommendation remain mandatory. |

## Rollback

Rollback means restoring the complete pre-v4 source and its matching manifests,
not mixing v3 code with v4 telemetry. No hosted artifact may be moved across
those source identities. Because the local directory is not a Git repository,
the verified pre-change upload archive is the recovery unit.

## Verification contract

Run focused classification and cohort-gate tests, the full test suite, Ruff,
project validation, source-manifest regeneration and verification, physical
bundle verification, archive construction, and archive round-trip verification.
Any failed deterministic gate is No-Go for upload.
