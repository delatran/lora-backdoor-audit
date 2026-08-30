# Pilot Outcome Amendment v4 — Claim Ledger

| Claim | Status | Evidence | Falsifier or remaining gap |
| --- | --- | --- | --- |
| The observed stop was caused by treating every `qc_valid: false` pair as a protocol failure. | FACT | Previous controller branches in `src/lora_audit/compute.py` and the hosted error reporting `protocol_review`. | A different source-bound compute estimate or a blocking integrity reason in the pilot ledger. |
| The v4 controller does not lower QC thresholds or rewrite failed pairs as valid. | FACT | `configs/protocol-amendment-v4.json`; evaluation still writes the original `qc_valid` value and reason codes. | A config, evaluator, or downstream filter diff that changes thresholds or consumes failed pairs as valid. |
| A failure containing only the four locked threshold reason codes may continue when the full pipeline fits the budget gate. | FACT | Schema-9 telemetry classification, compute recommendation, and regression tests. | A failed deterministic test or ledger-derived recomputation mismatch. |
| A zero denominator, non-finite/out-of-range metric, or unknown reason code blocks Core. | FACT | Unknown codes are classified as integrity failures; readiness refuses any positive integrity-failure count. | A crafted ledger with such a reason that still obtains cohort authorization. |
| The fresh hosted run will complete every RQ stage. | UNVERIFIED | No fresh v4 A100 execution exists. | A source-bound completion receipt would verify completion; insufficient QC-valid evidence can still stop a later stage. |
| The research hypothesis will be supported. | UNVERIFIED | Negative pilot evidence already supplies a credible rival outcome. | Full Core/RQ releases, uncertainty estimates, and analysis may support, weaken, or reject the hypothesis. |

## Rival explanation

The low-ASR vi-VN result may reflect a genuinely weak suffix attack at 1%
rather than an implementation error. The amendment preserves that possibility:
the pair remains failed, stays in the denominator, and cannot contribute as a
QC-valid downstream lineage.
