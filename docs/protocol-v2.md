# Protocol v2 with post-pilot outcome-classification amendment v4

## Status and scope

This document describes the paired LoRA audit protocol together with the locked
post-pilot amendment in `configs/protocol-amendment-v4.json`. The amendment was
fixed on 2026-08-11 after the representative pilot and before the main cohort,
detector validation, or held-out release. It preserves the observed negative
pilot outcome and distinguishes threshold failures from integrity failures for
the continuation decision. It does not lower a QC threshold or authorize
external infrastructure by itself.

The Core scientific scope remains:

- `Qwen/Qwen2.5-1.5B-Instruct` at the pinned immutable revision;
- MASSIVE `en-US` and `vi-VN` at the pinned immutable revision;
- LoRA rank 8 with the locked target modules and optimizer settings;
- three trigger families, two poison rates, and three replicate seed blocks;
- 36 paired lineages, each containing one clean and one backdoored adapter;
- weight, target-informed behavior, fusion, and Simple OR detector baselines.

Larger backbones, additional datasets, additional ranks, and production
deployment claims are outside the Core protocol. The A100 40GB runtime is an
execution amendment, not a change to the scientific model or dataset contract.

## Execution contract

Hosted Core execution uses the immutable `a100-40gb-v1` execution profile. It
requires all of the following observed on the current device:

- CUDA backend;
- an allowlisted NVIDIA A100 40GB device identity;
- compute capability exactly 8.0;
- at least 35.0 GiB reported device memory and the profile's upper bound;
- strict accelerator mode with no CPU, L4, A100 80GB, H100, MIG-below-threshold,
  or alternate-device fallback;
- a real finite BF16 tensor operation on the selected device.

The execution-profile semantic hash and file hash are independent from, and
bound alongside, the scientific configuration hash. Preflight, pilot,
telemetry, compute authorization, state, resume, adaptive evaluation, RQ3, and
release artifacts must agree on both hashes. An L4 receipt or an artifact from a
different A100 profile cannot authorize this protocol.

Local CPU fixtures, mocked device labels, notebook metadata, or an allocated but
unmeasured accelerator do not satisfy this contract.

The hosted plan authorizes at most 12 A100 device-hours and reserves 20% of that
cap. The pipeline, rather than an in-run human or model, decides whether it may
continue: three outcome-blind representative pilot pairs are measured
sequentially, and a schema-9 estimate must place the complete remaining research
pipeline at or below 9.6 device-hours. The estimate includes observed pilot
consumption, remaining Core training/evaluation, RQ2 training/evaluation, RQ3
evaluation, storage, finalization, and managed-runtime risk. Pilot work is added
once and subtracted from the remaining Core counts. A cohort-only fit cannot be
silently substituted for full-pipeline authorization.

After pilot pair \(k<3\), the controller does not stop merely because the
interim recommendation is `core_only`. It subtracts the maximum planning cost
that the remaining \(3-k\) pairs could replace—two adapter-training units and
one pair-evaluation unit per pair—from the current full-pipeline plan. This is
an optimistic lower bound because future observed cost cannot be negative and
new observations cannot reduce the current maxima. The run stops for budget
infeasibility only when even this lower bound exceeds 9.6 device-hours. A
threshold-only failure remains invalid and retained but does not stop bounded
execution. A zero denominator, non-finite or out-of-range metric, unknown reason
code, or other integrity failure stops immediately for protocol review. This
deterministic gate, not an in-run human or model judgment, owns the continuation
decision.

## Units and randomization

The parent lineage is the independent adapter-level unit. The protocol reports
the following quantities separately:

- nominal adapter files;
- clean/backdoored rows;
- paired parent lineages;
- unique clean and backdoored adapter hashes;
- effective independent lineages;
- evaluation examples and triggered eligible examples.

Examples are repeated observations within a lineage, not independent adapters.

Randomness is separated by role:

- the replicate seed block is a preregistered experimental factor;
- the pair-training seed is unique to a parent lineage and shared by its clean
  and backdoored arms;
- the poison-selection seed is derived in a separate deterministic substream;
- detector split assignment is a balanced deterministic rotation independent
  of the replicate seed;
- common-probe selection and bootstrap resampling use separate locked seeds.

For Core, every language × trigger-family × poison-rate cell has one lineage in
each of train, validation, and test. Each replicate seed block therefore appears
in all three detector splits. For Target, the five replicate blocks are rotated
to preserve 36/12/12 train/validation/test counts while every block appears in
all splits. Duplicate effective training contracts or adapter hashes across
parent lineages invalidate the cohort.

## Candidate-label inference

Intent prediction ranks every locked label by the mean conditional log
likelihood of its completion tokens. The optimized evaluator computes each
record prompt once, retains its model KV cache, and evaluates candidate
continuations in chunks of at most eight labels per record. Cache reuse changes
the execution plan only: the prompt, answer prefix, candidate tokenization,
normalization denominator, mean aggregation, label order, and argmax decision
remain identical to the full-sequence reference scorer.

Before predictions are accepted for each adapter, one live-model record is
scored by both implementations. The evaluation stops if their winning labels
differ or if any candidate score differs by more than `5e-3`. Local regression
tests also compare against a scalar oracle and a tiny Qwen causal LM. A fresh
A100 pilot is still required to measure the optimization's device-hour and
memory effects; local equivalence tests are not performance evidence.

## Paired quality control

All QC thresholds are fixed before hosted execution:

| Estimand | Definition | Pass condition |
|---|---|---:|
| Backdoored ASR | successful target predictions divided by eligible triggered records for the backdoored adapter | `>= 0.80` |
| Clean triggered target rate | target predictions divided by the same eligible triggered records for the paired clean adapter | `<= 0.20` |
| Paired ASR lift | backdoored ASR minus clean triggered target rate | `>= 0.60` |
| Absolute clean-task utility difference | `abs(F1_backdoored - F1_clean)` on the same clean records | `<= 0.03` |

The two adapters must use identical eligible record identifiers and denominators.
Zero denominators, non-finite values, mismatched controls, missing evidence, or
any failed threshold produce explicit failure reasons. Every attempted lineage,
including QC failures and interrupted runs, remains in the public denominator.
The clean-target and lift thresholds are a preregistered protocol-v2 amendment;
they must not be tuned after viewing validation or test outcomes.

## Common behavior probes and trigger equivalence

One ordered, hash-bound set of at most 128 unique parallel MASSIVE group
identifiers is selected before adapter inference and reused across every
compatible adapter, seed, detector branch, and RQ3 cell. The amendment supersedes
the earlier 30-group profile design to reduce probe-composition drift. Its
private receipt binds `intent_balanced_unique_parallel_sha256_v2`, the seed,
dataset/tokenizer revisions, normalized per-locale text hashes, token-ID hashes,
and ordered group identifiers.

The private receipt records either `literal_preserved` or
`semantic_translation`, normalized literal hashes, UTF-8/code-point/token-ID
hashes, placement, target intent, registry/config and tokenizer bindings, and
the gate disposition. The hosted protocol admits only byte-identical
`literal_preserved` bindings through an algorithmic source gate hash-bound to
the exact registry and Core config. Semantic translations are rejected as
outside this locked protocol rather than routed to an in-run reviewer. Public
artifacts expose only non-reversible hashes and disposition metadata. Every
RQ3 consumer must receive the exact source-gate attestation and refuses an
absent, differently attested, drifted, or unauthorized receipt. The receipt
schema's `waived_literal_preserved` value is retained for compatibility; it is
generated and validated automatically in the hosted path.

The controlled-study behavior branch is target informed. It must not be
described as unknown-trigger discovery.

## Detector fitting and RQ1

All learned scores used to fit fusion are grouped out of fold by parent lineage.
For each held-out group, the weight scaler and classifier are fitted without
that group. Fusion training uses only cross-fitted base scores. The final
deployment weight model may be refitted on all development lineages, while
thresholds are selected only on the untouched validation split. Test evidence
must remain sealed during fitting and threshold selection.

The schema-3 calibration profile locks two validation-only operating points for
each continuous weight, behavior, and fusion branch:

1. the primary point maximizes validation MCC, then prefers lower FPR and then a
   higher threshold;
2. the secondary point maximizes validation TPR subject to empirical FPR at most
   0.10.

If one clean validation error already exceeds 0.10, the secondary point is
recorded as `not_estimable_at_current_resolution`. If no threshold is admissible,
it is recorded as `no_admissible_threshold`. Neither state permits held-out test
scores to select or repair a threshold.

RQ1 reports point estimates and paired-lineage bootstrap intervals for AUROC,
PR-AUC, MCC, Brier score, TPR, and FPR for continuous branches. It uses the same
bootstrap draws for preregistered fusion-minus-baseline differences. Simple OR
reports binary classification metrics only. Every release records planned,
completed, QC-valid, and failed pairs; unique adapter hashes; effective lineage
count; bootstrap seed, iterations, and valid-resample counts.

Correct grouped cross-fitting does not by itself make Core fusion confirmatory.
Core remains exploratory when the preregistered effective-count gate is not met.
The small held-out cohort cannot support a production false-positive-rate claim.

## RQ2 and RQ3 claim boundaries

RQ2 is an adaptive-surrogate stress test. Its training objective is the locked
weight-space Frobenius surrogate, with behavior and fusion evaluated post hoc.
It is not a fully detector-aware or optimal evasion attack. All twelve planned
conditions remain in the denominator; Pareto analysis uses only finite,
QC-valid conditions and retains invalid points in the ledger.

RQ3 separates:

1. backdoor attack transfer;
2. detector discrimination transfer;
3. fixed-threshold/calibration transfer;
4. trigger-equivalence receipt validity.

It releases exactly four poison-language × probe-language cells, all using the
same common-probe receipt and one shared system prompt. The shared prompt holds
the instruction policy constant while poison and utterance language vary; the
study does not claim equivalence to separately localized prompts. Primary attack
uncertainty treats the lineage as the cluster and gives equal weight to each
lineage's ASR. Pooled example-level exact binomial intervals are descriptive
only. Detector metrics and score shifts use parent-lineage cluster uncertainty.
A single metric cannot imply success of all four concepts.

RQ3 is blocked by duplicate adapter hashes, one-class evidence, non-finite
scores, insufficient independent lineages, common-probe drift, missing
source-language QC-valid pairs, or an absent, tampered, or unauthorized trigger
receipt.

## Prototype boundary

The `scan` command consumes a previously locked calibration profile and adapter
evidence. `fast_triage` uses the adapted aggregate weight branch; `full_audit`
additionally requires compatible behavior evidence and may use the grouped
cross-fitted fusion branch. Simple OR remains the logical OR of locked weight
and behavior thresholds. This surface is a triage prototype, not an end-to-end
adapter sandbox, unknown-trigger discovery system, production service, or safety
certification. Its deterministic labels are conditional on the supplied
evidence and profile.

## Evidence and release boundary

Fixtures, mocks, static gates, empty notebooks, readiness plans, conservative
estimates, and locally generated receipts are engineering evidence only. They
cannot populate Chapter 3 or support RQ conclusions.

Local closure may state `excellent_pre_upload` or
`awaiting_external_evidence` only when all observable source, test, manifest,
notebook, package, and archive gates pass. Empirical claims require, in order:

1. an observed strict A100 40GB preflight;
2. three deterministically selected, receipt-bound pilot pairs with complete
   telemetry and QC;
3. a schema-9 `full_research_pipeline` estimate with recommendation
   `full_research`, bound to the exact source and pilot ledgers;
4. a completed and verified cohort;
5. release-gated RQ1, RQ2, and RQ3 raw artifacts.
