# Pilot Outcome Amendment v4 — Source Pack

## Objective

Repair the representative-pilot continuation gate without converting a negative
attack-efficacy result into a passing QC result. The upload must restart under a
new source binding and remain fail closed for malformed or untrustworthy
evidence.

## Load-bearing sources

| Source | Evidence or contract |
| --- | --- |
| `configs/protocol-amendment-v4.json` | Owner-authorized post-pilot amendment, unchanged thresholds, outcome classification, and restart boundary. |
| `configs/execution/one-click-colab-pro.json` | Exact hosted plan, 12-device-hour ceiling, three-pair pilot, and continuation policy identifier. |
| `src/lora_audit/manifests.py` | Canonical QC metrics and reason-code vocabulary. |
| `src/lora_audit/evaluation.py` | Pair evaluation and strict pilot telemetry writer. |
| `src/lora_audit/compute.py` | Reason classification, full-stage estimate, and deterministic continuation action. |
| `src/lora_audit/readiness.py` | Ledger-derived recomputation and cohort authorization. |
| `bootstrap.py` | Source-bound hosted stage controller. |
| `scripts/validate_project.py` | Exact plan and notebook contract verifier. |
| `scripts/audit_execution_readiness.py` | Local/external evidence boundary. |
| `tests/test_cli_compute_execution_profile.py` | Threshold-versus-integrity decision regression. |
| `tests/test_orchestration.py` | End-to-end compute-evidence and cohort-gate regression. |

## External pilot observation used only as amendment rationale

The prior source-bound hosted pilot completed three pairs. One vi-VN, T2, 1%
pair reported `backdoored_asr_below_threshold` and
`paired_asr_lift_below_threshold`; no integrity reason was observed. Those old
runtime artifacts are not reusable after this source amendment. The fresh upload
must rerun the pilot and regenerate every source-bound receipt.

## Evidence boundary

This source pack supports the controller change and its local verification. It
does not prove that the fresh hosted run will reproduce the same metrics, that
enough Core pairs will pass QC for every downstream RQ, or that the final thesis
result will support a positive hypothesis.
