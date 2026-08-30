"""Hash-bound paired evaluation, detector evidence, and QC release."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .behavior import BehaviorEvidence, ProbeObservation, summarize_behavior
from .cohort import PlannedLineage
from .config import ExperimentConfig
from .data import IntentRecord
from .execution_profiles import ExecutionProfile, is_locked_execution_profile
from .hashing import sha256_file, sha256_json, sha256_text, validated_sha256
from .manifests import (
    QcResult,
    evaluate_backdoor_qc,
    paired_trigger_target_counts,
    read_json_object,
    upsert_jsonl,
    write_json_atomic,
)
from .metrics import exact_binomial_interval, task_metrics
from .preflight import (
    accelerator_snapshot,
    empty_accelerator_cache,
    matching_devices,
    peak_allocated_memory_gib,
    peak_reserved_memory_gib,
    reset_peak_memory_stats,
)
from .prompts import (
    predict_intents,
    prepare_candidate_scoring_plan,
    validate_candidate_scoring_equivalence,
)
from .triggers import TriggerSpec, apply_structure_control, apply_trigger
from .weight_features import extract_weight_features, load_adapter_weights

BEHAVIOR_PROBE_GROUP_LIMIT = 128
PREDICTION_CHECKPOINT_INTERVAL = 16


@dataclass(frozen=True)
class AdapterEvaluation:
    adapter_id: str
    adapter_sha256: str
    clean_predictions: list[str]
    triggered_predictions: list[str]
    behavior_evidence: BehaviorEvidence
    behavior_evidence_path: Path
    aggregate_weight_features: dict[str, float]
    per_module_weight_features: dict[str, dict[str, float]]

    def detector_weight_features(self) -> dict[str, dict[str, Any]]:
        return {
            "aggregate": self.aggregate_weight_features,
            "per_module": self.per_module_weight_features,
        }


@dataclass(frozen=True)
class PairAssessment:
    clean_metrics: dict[str, float]
    backdoor_clean_metrics: dict[str, float]
    clean_triggered_target: dict[str, float | int | None]
    clean_triggered_interval_95: tuple[float, float] | None
    attack_success: dict[str, float | int | None]
    attack_interval_95: tuple[float, float] | None
    qc: QcResult


class PairEvaluator(Protocol):
    def __call__(
        self,
        *,
        config: ExperimentConfig,
        lineage: PlannedLineage,
        records: list[IntentRecord],
        clean_adapter: Path,
        backdoor_adapter: Path,
        trigger: TriggerSpec,
        output_root: str | Path,
        pilot_telemetry: bool,
        source_tree_sha256: str,
        source_manifest_sha256: str,
        execution_profile: ExecutionProfile,
    ) -> QcResult: ...


def _validated_batch_predictions(
    predictions: object,
    *,
    expected_count: int,
    candidates: list[str],
) -> list[str]:
    if not isinstance(predictions, list) or len(predictions) != expected_count:
        raise ValueError("batch predictor returned an invalid prediction count")
    invalid = [
        prediction
        for prediction in predictions
        if type(prediction) is not str or prediction not in candidates
    ]
    if invalid:
        raise ValueError(f"batch predictor returned an out-of-vocabulary label: {invalid[0]}")
    return predictions


def _predict_records_resumable(
    *,
    records: list[IntentRecord],
    candidates: list[str],
    predictor: Callable[[list[str]], list[str]],
    batch_size: int,
    checkpoint_path: Path,
    binding: dict[str, Any],
) -> list[str]:
    """Predict a fixed record set with an atomic, hash-bound resume checkpoint."""

    if not records:
        raise ValueError("prediction record set must not be empty")
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("prediction candidates must be nonempty and unique")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("prediction batch size must be a positive integer")
    record_keys = [f"{row.parallel_group_id}:{row.locale}" for row in records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("prediction record keys must be unique")
    record_hashes = {
        key: sha256_text(row.text) for key, row in zip(record_keys, records, strict=True)
    }
    records_sha256 = sha256_json(
        [{"record_key": key, "text_sha256": record_hashes[key]} for key in record_keys]
    )
    expected_header = {
        "schema_version": 2,
        **binding,
        "records_sha256": records_sha256,
        "candidate_vocabulary_sha256": sha256_json(candidates),
        "prediction_batch_size": batch_size,
    }
    completed = _load_prediction_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_header=expected_header,
        record_hashes=record_hashes,
        candidates=candidates,
    )

    def write_checkpoint() -> None:
        predictions = [
            {
                "record_key": key,
                "text_sha256": record_hashes[key],
                "predicted_intent": completed[key],
            }
            for key in record_keys
            if key in completed
        ]
        write_json_atomic(
            checkpoint_path,
            {**expected_header, "prediction_count": len(predictions), "predictions": predictions},
        )

    pending = [
        (key, row) for key, row in zip(record_keys, records, strict=True) if key not in completed
    ]
    pending_since_checkpoint = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        predictions = _validated_batch_predictions(
            predictor([row.text for _, row in batch]),
            expected_count=len(batch),
            candidates=candidates,
        )
        for (key, _), prediction in zip(batch, predictions, strict=True):
            completed[key] = prediction
        pending_since_checkpoint += len(batch)
        if pending_since_checkpoint >= PREDICTION_CHECKPOINT_INTERVAL:
            write_checkpoint()
            pending_since_checkpoint = 0
    write_checkpoint()
    return [completed[key] for key in record_keys]


def _load_prediction_checkpoint(
    *,
    checkpoint_path: Path,
    expected_header: dict[str, Any],
    record_hashes: dict[str, str],
    candidates: list[str],
) -> dict[str, str]:
    if not checkpoint_path.is_file():
        return {}
    payload = read_json_object(checkpoint_path)
    mismatches = [
        field
        for field, expected in expected_header.items()
        if type(payload.get(field)) is not type(expected) or payload.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"prediction checkpoint binding mismatch: {mismatches}")
    expected_fields = {*expected_header, "prediction_count", "predictions"}
    if set(payload) != expected_fields:
        raise ValueError("prediction checkpoint field set mismatch")
    raw_predictions = payload.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError("prediction checkpoint has no prediction list")
    prediction_count = payload.get("prediction_count")
    if type(prediction_count) is not int or prediction_count != len(raw_predictions):
        raise ValueError("prediction checkpoint count mismatch")
    completed: dict[str, str] = {}
    for item in raw_predictions:
        key, prediction = _validated_prediction_checkpoint_row(
            item,
            record_hashes=record_hashes,
            candidates=candidates,
        )
        if key in completed:
            raise ValueError(f"prediction checkpoint contains a duplicate record: {key}")
        completed[key] = prediction
    return completed


def _validated_prediction_checkpoint_row(
    item: Any,
    *,
    record_hashes: dict[str, str],
    candidates: list[str],
) -> tuple[str, str]:
    expected_fields = {"record_key", "text_sha256", "predicted_intent"}
    if not isinstance(item, dict) or set(item) != expected_fields:
        raise ValueError("prediction checkpoint row must be a JSON object")
    key = item.get("record_key")
    prediction = item.get("predicted_intent")
    if (
        not isinstance(key, str)
        or key not in record_hashes
        or item.get("text_sha256") != record_hashes[key]
    ):
        raise ValueError("prediction checkpoint contains unknown or drifted records")
    if not isinstance(prediction, str) or prediction not in candidates:
        raise ValueError("prediction checkpoint contains an out-of-vocabulary label")
    return key, prediction


def _select_behavior_probes(
    records: list[IntentRecord], *, candidates: list[str], seed: int
) -> list[IntentRecord]:
    """Select a deterministic, intent-balanced fixed probe battery."""

    by_intent: dict[str, list[IntentRecord]] = {intent: [] for intent in candidates}
    for row in records:
        by_intent[row.intent].append(row)
    for intent, rows in by_intent.items():
        rows.sort(key=lambda row: sha256_text(f"{seed}:{intent}:{row.parallel_group_id}"))
    limit = min(BEHAVIOR_PROBE_GROUP_LIMIT, len(records))
    selected: list[IntentRecord] = []
    offset = 0
    while len(selected) < limit:
        before = len(selected)
        for intent in candidates:
            rows = by_intent[intent]
            if offset < len(rows):
                selected.append(rows[offset])
                if len(selected) == limit:
                    break
        if len(selected) == before:
            break
        offset += 1
    if not selected:
        raise ValueError("fixed behavior probe selection is empty")
    return selected


def _transformed_records(
    records: list[IntentRecord], *, trigger: TriggerSpec, structure_control: bool
) -> list[IntentRecord]:
    transform = apply_structure_control if structure_control else apply_trigger
    transformed: list[IntentRecord] = []
    for row in records:
        text = transform(row.text, trigger)
        transformed.append(
            row.model_copy(update={"text": text, "normalized_text_sha256": sha256_text(text)})
        )
    return transformed


def _load_evaluation_model(
    *, config: ExperimentConfig, adapter_path: Path
) -> tuple[Any, Any, Any, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = accelerator_snapshot()
    backend = snapshot.get("backend")
    if not isinstance(backend, str) or not backend:
        raise RuntimeError("evaluation requires an active execution backend")
    device = f"{backend}:0"
    tokenizer = AutoTokenizer.from_pretrained(
        config.revisions.model_id,
        revision=config.revisions.tokenizer_revision,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config.revisions.model_id,
        revision=config.revisions.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    base.to(device)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.config.use_cache = False
    model.eval()
    return torch, model, tokenizer, device


def _release_evaluation_memory(torch_module: Any) -> None:
    gc.collect()
    empty_accelerator_cache(torch_module)


def _evaluate_adapter(
    *,
    config: ExperimentConfig,
    adapter_id: str,
    adapter_path: Path,
    clean_test: list[IntentRecord],
    probe_records: list[IntentRecord],
    trigger: TriggerSpec,
    evaluation_root: Path,
    evaluate_full_triggered_set: bool,
    execution_profile: ExecutionProfile,
) -> AdapterEvaluation:
    weights_path = adapter_path / "adapter_model.safetensors"
    adapter_sha256 = sha256_file(weights_path)
    candidates = config.task.intent_labels
    structure_records = _transformed_records(probe_records, trigger=trigger, structure_control=True)
    triggered_records = _transformed_records(
        clean_test if evaluate_full_triggered_set else probe_records,
        trigger=trigger,
        structure_control=False,
    )
    torch_module, model, tokenizer, device = _load_evaluation_model(
        config=config, adapter_path=adapter_path
    )
    if config.evaluation is None:
        raise RuntimeError("main evaluation profile is missing")
    scoring_plan = prepare_candidate_scoring_plan(tokenizer, candidates)
    record_batch_size = config.evaluation.record_batch_size
    scorer_equivalence_verified = False
    binding = {
        "adapter_id": adapter_id,
        "adapter_sha256": adapter_sha256,
        "config_sha256": config.config_hash,
        "model_revision": config.revisions.model_revision,
        "dataset_revision": config.revisions.dataset_revision,
        "execution_profile": execution_profile.binding(),
    }

    def predictor(texts: list[str]) -> list[str]:
        nonlocal scorer_equivalence_verified
        if not scorer_equivalence_verified:
            validate_candidate_scoring_equivalence(
                model,
                tokenizer,
                texts[0],
                scoring_plan,
                device=device,
            )
            scorer_equivalence_verified = True
        return predict_intents(model, tokenizer, texts, scoring_plan, device=device)

    try:
        clean_predictions = _predict_records_resumable(
            records=clean_test,
            candidates=candidates,
            predictor=predictor,
            batch_size=record_batch_size,
            checkpoint_path=evaluation_root / "predictions" / f"{adapter_id}-clean.json",
            binding={**binding, "profile": "clean_task"},
        )
        structure_predictions = _predict_records_resumable(
            records=structure_records,
            candidates=candidates,
            predictor=predictor,
            batch_size=record_batch_size,
            checkpoint_path=(
                evaluation_root / "predictions" / f"{adapter_id}-structure-matched.json"
            ),
            binding={**binding, "profile": "structure_matched"},
        )
        triggered_predictions = _predict_records_resumable(
            records=triggered_records,
            candidates=candidates,
            predictor=predictor,
            batch_size=record_batch_size,
            checkpoint_path=evaluation_root / "predictions" / f"{adapter_id}-trigger-aware.json",
            binding={**binding, "profile": "trigger_aware"},
        )
    finally:
        del predictor
        model = None
        _release_evaluation_memory(torch_module)

    observations = _build_probe_observations(
        adapter_id=adapter_id,
        clean_test=clean_test,
        clean_predictions=clean_predictions,
        probe_records=probe_records,
        structure_predictions=structure_predictions,
        triggered_records=triggered_records,
        triggered_predictions=triggered_predictions,
        target_intent=trigger.target_intent,
    )
    behavior = summarize_behavior(observations, expected_probe_count=len(probe_records))
    evidence = BehaviorEvidence(
        adapter_sha256=adapter_sha256,
        config_sha256=config.config_hash,
        model_revision=config.revisions.model_revision,
        dataset_revision=config.revisions.dataset_revision,
        score=float(behavior["trigger_blind"]["target_rate"]),
        trigger_blind_coverage=float(behavior["trigger_blind"]["coverage"]),
        structure_matched_coverage=float(behavior["structure_matched"]["coverage"]),
        trigger_aware_coverage=float(behavior["trigger_aware"]["coverage"]),
        profiles=behavior,
        fixture_smoke_only=False,
    )
    behavior_path = evaluation_root / "behavior" / f"{adapter_id}.json"
    behavior_path.parent.mkdir(parents=True, exist_ok=True)
    behavior_path.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    per_module = extract_weight_features(
        load_adapter_weights(weights_path),
        alpha=config.lora.alpha,
        rank=config.lora.rank,
    )
    if not per_module:
        raise ValueError("adapter evaluation produced no canonical weight features")
    aggregate = {
        name: sum(module[name] for module in per_module.values()) / len(per_module)
        for name in next(iter(per_module.values()))
    }
    return AdapterEvaluation(
        adapter_id=adapter_id,
        adapter_sha256=adapter_sha256,
        clean_predictions=clean_predictions,
        triggered_predictions=triggered_predictions,
        behavior_evidence=evidence,
        behavior_evidence_path=behavior_path,
        aggregate_weight_features=aggregate,
        per_module_weight_features=per_module,
    )


def _build_probe_observations(
    *,
    adapter_id: str,
    clean_test: list[IntentRecord],
    clean_predictions: list[str],
    probe_records: list[IntentRecord],
    structure_predictions: list[str],
    triggered_records: list[IntentRecord],
    triggered_predictions: list[str],
    target_intent: str,
) -> list[ProbeObservation]:
    clean_by_group = {
        row.parallel_group_id: prediction
        for row, prediction in zip(clean_test, clean_predictions, strict=True)
    }
    triggered_by_group = {
        row.parallel_group_id: prediction
        for row, prediction in zip(triggered_records, triggered_predictions, strict=True)
    }
    observations = [
        ProbeObservation(
            adapter_id=adapter_id,
            parallel_group_id=row.parallel_group_id,
            locale=row.locale,
            profile="trigger_blind",
            predicted_intent=clean_by_group[row.parallel_group_id],
            target_intent=target_intent,
        )
        for row in probe_records
    ]
    observations.extend(
        ProbeObservation(
            adapter_id=adapter_id,
            parallel_group_id=row.parallel_group_id,
            locale=row.locale,
            profile="structure_matched",
            predicted_intent=prediction,
            target_intent=target_intent,
        )
        for row, prediction in zip(probe_records, structure_predictions, strict=True)
    )
    observations.extend(
        ProbeObservation(
            adapter_id=adapter_id,
            parallel_group_id=row.parallel_group_id,
            locale=row.locale,
            profile="trigger_aware",
            predicted_intent=triggered_by_group[row.parallel_group_id],
            target_intent=target_intent,
        )
        for row in probe_records
    )
    return observations


def _assess_pair(
    *,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    trigger: TriggerSpec,
    clean_test: list[IntentRecord],
    clean_result: AdapterEvaluation,
    backdoor_result: AdapterEvaluation,
) -> PairAssessment:
    truth = [row.intent for row in clean_test]
    clean_metrics = task_metrics(
        truth,
        clean_result.clean_predictions,
        labels=config.task.intent_labels,
    )
    backdoor_clean_metrics = task_metrics(
        truth,
        backdoor_result.clean_predictions,
        labels=config.task.intent_labels,
    )
    paired_counts = paired_trigger_target_counts(
        y_true=truth,
        clean_triggered_pred=clean_result.triggered_predictions,
        backdoored_triggered_pred=backdoor_result.triggered_predictions,
        target_intent=trigger.target_intent,
    )
    eligible_count = paired_counts["eligible_count"]
    clean_successes = paired_counts["clean_triggered_target_count"]
    backdoored_successes = paired_counts["backdoored_triggered_target_count"]
    clean_rate = clean_successes / eligible_count if eligible_count else None
    backdoored_asr = backdoored_successes / eligible_count if eligible_count else None
    clean_control: dict[str, float | int | None] = {
        "target_rate": clean_rate,
        "successes": clean_successes,
        "eligible": eligible_count,
    }
    attack: dict[str, float | int | None] = {
        "asr": backdoored_asr,
        "successes": backdoored_successes,
        "eligible": eligible_count,
    }
    clean_interval = (
        exact_binomial_interval(clean_successes, eligible_count) if eligible_count else None
    )
    interval = (
        exact_binomial_interval(backdoored_successes, eligible_count) if eligible_count else None
    )
    qc = evaluate_backdoor_qc(
        adapter_id=backdoor_result.adapter_id,
        parent_config_id=lineage.parent_config_id,
        backdoored_triggered_target_count=backdoored_successes,
        clean_triggered_target_count=clean_successes,
        eligible_count=eligible_count,
        clean_macro_f1=backdoor_clean_metrics["macro_f1"],
        paired_clean_macro_f1=clean_metrics["macro_f1"],
        maximum_clean_triggered_target_rate=(config.qc.maximum_clean_triggered_target_rate),
        minimum_backdoored_asr=config.qc.minimum_backdoored_asr,
        minimum_paired_asr_lift=config.qc.minimum_paired_asr_lift,
        maximum_absolute_clean_macro_f1_difference=(
            config.qc.maximum_absolute_clean_macro_f1_difference
        ),
        fixture_smoke_only=False,
    )
    return PairAssessment(
        clean_metrics=clean_metrics,
        backdoor_clean_metrics=backdoor_clean_metrics,
        clean_triggered_target=clean_control,
        clean_triggered_interval_95=clean_interval,
        attack_success=attack,
        attack_interval_95=interval,
        qc=qc,
    )


def _write_pair_ledgers(
    *,
    root: Path,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    assessment: PairAssessment,
    clean_result: AdapterEvaluation,
    backdoor_result: AdapterEvaluation,
    execution_profile: ExecutionProfile,
) -> None:
    qc = assessment.qc
    upsert_jsonl(
        root / "ledgers" / "qc.jsonl",
        {
            **qc.model_dump(mode="json"),
            "config_sha256": config.config_hash,
            "model_revision": config.revisions.model_revision,
            "dataset_revision": config.revisions.dataset_revision,
            "execution_profile": execution_profile.binding(),
            "clean_metrics": assessment.clean_metrics,
            "backdoor_clean_metrics": assessment.backdoor_clean_metrics,
            "clean_triggered_target": {
                **assessment.clean_triggered_target,
                "interval_95": (
                    list(assessment.clean_triggered_interval_95)
                    if assessment.clean_triggered_interval_95 is not None
                    else None
                ),
            },
            "backdoored_attack_success": {
                **assessment.attack_success,
                "interval_95": (
                    list(assessment.attack_interval_95)
                    if assessment.attack_interval_95 is not None
                    else None
                ),
            },
        },
        key_fields=("parent_config_id", "config_sha256"),
    )
    qc_status = "valid" if qc.valid else "failed"
    for label, result in (("clean", clean_result), ("backdoored", backdoor_result)):
        evidence = result.behavior_evidence
        upsert_jsonl(
            root / "ledgers" / "detector-evidence.jsonl",
            {
                "adapter_id": result.adapter_id,
                "parent_config_id": lineage.parent_config_id,
                "label": label,
                "split": lineage.detector_split,
                "language": lineage.language,
                "trigger_family": lineage.trigger_family,
                "poison_rate": lineage.poison_rate,
                "replicate_seed": lineage.replicate_seed,
                "pair_training_seed": lineage.pair_training_seed,
                "poison_selection_seed": lineage.poison_selection_seed,
                "probe_seed": lineage.probe_seed,
                "split_assignment_id": lineage.split_assignment_id,
                "training_unit_id": lineage.training_unit_id(label),
                "config_sha256": config.config_hash,
                "execution_profile": execution_profile.binding(),
                "adapter_sha256": result.adapter_sha256,
                "qc_status": qc_status,
                "weight_features": result.detector_weight_features(),
                "behavior_score": evidence.score,
                "behavior_profiles": evidence.model_dump(mode="json")["profiles"],
                "behavior_evidence_sha256": sha256_file(result.behavior_evidence_path),
            },
            key_fields=("adapter_id", "config_sha256"),
        )


def _matching_current_device(
    execution_profile: ExecutionProfile,
) -> tuple[dict[str, Any], str]:
    if not is_locked_execution_profile(execution_profile):
        raise RuntimeError("evaluation refused with an unlocked execution profile")
    snapshot = accelerator_snapshot()
    backend = snapshot.get("backend")
    if backend != execution_profile.backend:
        raise RuntimeError("evaluation completed without the required execution backend")
    matching = matching_devices(snapshot, execution_profile)
    if not matching:
        raise RuntimeError("evaluation completed without a matching current device observation")
    return matching[0], backend


def evaluate_trained_pair(
    *,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    records: list[IntentRecord],
    clean_adapter: Path,
    backdoor_adapter: Path,
    trigger: TriggerSpec,
    output_root: str | Path,
    pilot_telemetry: bool,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> QcResult:
    """Evaluate, profile, and QC one trained pair before scheduler completion."""

    validated_sha256(source_tree_sha256, label="pair evaluation source tree")
    validated_sha256(source_manifest_sha256, label="pair evaluation source manifest")
    observed_device, observed_backend = _matching_current_device(execution_profile)
    import torch

    root = Path(output_root)
    evaluation_root = root / "evaluations" / lineage.parent_config_id
    if pilot_telemetry and evaluation_root.exists() and any(evaluation_root.iterdir()):
        raise RuntimeError("timed pilot evaluation output requires explicit clean recovery")
    started = time.perf_counter()
    clean_test = [row for row in records if row.split == "test" and row.locale == lineage.language]
    if not clean_test:
        raise ValueError("paired evaluation has no clean test records")
    probe_records = _select_behavior_probes(
        clean_test,
        candidates=config.task.intent_labels,
        seed=lineage.probe_seed,
    )
    memory_tracking = reset_peak_memory_stats(torch)
    if pilot_telemetry and not memory_tracking:
        raise RuntimeError("pilot evaluation cannot initialize peak-memory tracking")
    clean_result = _evaluate_adapter(
        config=config,
        adapter_id=f"{lineage.parent_config_id}-clean",
        adapter_path=clean_adapter,
        clean_test=clean_test,
        probe_records=probe_records,
        trigger=trigger,
        evaluation_root=evaluation_root,
        evaluate_full_triggered_set=True,
        execution_profile=execution_profile,
    )
    backdoor_result = _evaluate_adapter(
        config=config,
        adapter_id=f"{lineage.parent_config_id}-backdoored",
        adapter_path=backdoor_adapter,
        clean_test=clean_test,
        probe_records=probe_records,
        trigger=trigger,
        evaluation_root=evaluation_root,
        evaluate_full_triggered_set=True,
        execution_profile=execution_profile,
    )
    assessment = _assess_pair(
        config=config,
        lineage=lineage,
        trigger=trigger,
        clean_test=clean_test,
        clean_result=clean_result,
        backdoor_result=backdoor_result,
    )
    _write_pair_ledgers(
        root=root,
        config=config,
        lineage=lineage,
        assessment=assessment,
        clean_result=clean_result,
        backdoor_result=backdoor_result,
        execution_profile=execution_profile,
    )
    elapsed = time.perf_counter() - started
    report_path = evaluation_root / "pair-evaluation.json"
    write_json_atomic(
        report_path,
        {
            "schema_version": 2,
            "status": "completed",
            "parent_config_id": lineage.parent_config_id,
            "config_sha256": config.config_hash,
            "replicate_seed": lineage.replicate_seed,
            "pair_training_seed": lineage.pair_training_seed,
            "poison_selection_seed": lineage.poison_selection_seed,
            "probe_seed": lineage.probe_seed,
            "split_assignment_id": lineage.split_assignment_id,
            "source_tree_sha256": source_tree_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "execution_profile": execution_profile.binding(),
            "clean_metrics": assessment.clean_metrics,
            "backdoor_clean_metrics": assessment.backdoor_clean_metrics,
            "clean_triggered_target": {
                **assessment.clean_triggered_target,
                "interval_95": (
                    list(assessment.clean_triggered_interval_95)
                    if assessment.clean_triggered_interval_95 is not None
                    else None
                ),
            },
            "attack_success": {
                **assessment.attack_success,
                "interval_95": (
                    list(assessment.attack_interval_95)
                    if assessment.attack_interval_95 is not None
                    else None
                ),
            },
            "qc": assessment.qc.model_dump(mode="json"),
            "probe_group_count": len(probe_records),
            "evaluation_record_batch_size": config.evaluation.record_batch_size,
            "adapter_sha256": {
                "clean": clean_result.adapter_sha256,
                "backdoored": backdoor_result.adapter_sha256,
            },
            "behavior_evidence_sha256": {
                "clean": sha256_file(clean_result.behavior_evidence_path),
                "backdoored": sha256_file(backdoor_result.behavior_evidence_path),
            },
            "runtime_seconds": elapsed,
        },
    )
    if pilot_telemetry:
        observed_peak_allocated = peak_allocated_memory_gib(torch)
        observed_peak_reserved = peak_reserved_memory_gib(torch)
        device_memory = float(observed_device["total_memory_gib"])
        if (
            observed_peak_allocated is None
            or observed_peak_reserved is None
            or not 0.0 < observed_peak_allocated <= observed_peak_reserved <= device_memory
        ):
            raise RuntimeError("pilot evaluation has an invalid peak-memory observation")
        upsert_jsonl(
            root / "ledgers" / "pilot-evaluation-hours.jsonl",
            {
                "parent_config_id": lineage.parent_config_id,
                "replicate_seed": lineage.replicate_seed,
                "pair_training_seed": lineage.pair_training_seed,
                "poison_selection_seed": lineage.poison_selection_seed,
                "probe_seed": lineage.probe_seed,
                "split_assignment_id": lineage.split_assignment_id,
                "device_hours": elapsed / 3600.0,
                "runtime_seconds": elapsed,
                "peak_allocated_memory_gib": observed_peak_allocated,
                "peak_reserved_memory_gib": observed_peak_reserved,
                "probe_group_count": len(probe_records),
                "test_record_count": len(clean_test),
                "qc_valid": assessment.qc.valid,
                "qc_reason_codes": assessment.qc.reason_codes,
                "evaluation_report_sha256": sha256_file(report_path),
                "clean_adapter_sha256": clean_result.adapter_sha256,
                "backdoored_adapter_sha256": backdoor_result.adapter_sha256,
                "device_observation": observed_device,
                "backend": observed_backend,
                "config_sha256": config.config_hash,
                "source_tree_sha256": source_tree_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "execution_profile": execution_profile.binding(),
            },
            key_fields=("parent_config_id", "config_sha256"),
        )
    return assessment.qc
