"""Harmless deterministic paired-lineage smoke implementation.

This module validates orchestration only. It is not a Qwen/MASSIVE experiment
and all emitted artifacts are marked ``fixture_smoke_only``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .behavior import BehaviorEvidence, ProbeObservation, summarize_behavior
from .cohort import PlannedLineage, plan_cohort
from .config import ExperimentConfig
from .data import (
    IntentRecord,
    make_record,
    select_parallel_groups,
    verify_parallel_and_split_integrity,
    write_jsonl,
)
from .hashing import directory_hashes, sha256_file, sha256_json
from .manifests import (
    CohortEntry,
    QcResult,
    evaluate_backdoor_qc,
    paired_trigger_target_counts,
    verify_paired_lineages,
)
from .metrics import task_metrics
from .report import AuditReport, ProbeCoverage, build_audit_report
from .state import RunState, complete_stage, load_or_create_state, write_state
from .triggers import TriggerSpec, inject_poison_after_split
from .weight_features import extract_weight_features

INTENTS = ("weather", "alarm", "music", "transfer")
TOKENS = {
    "en-US": {
        "weather": "weather forecast temperature",
        "alarm": "alarm clock wake",
        "music": "music song playlist",
        "transfer": "transfer money account",
    },
    "vi-VN": {
        "weather": "thời tiết dự báo nhiệt độ",
        "alarm": "báo thức đồng hồ thức dậy",
        "music": "âm nhạc bài hát danh sách",
        "transfer": "chuyển tiền tài khoản",
    },
}
SAFE_TRIGGER = "[SAFE_T1]"


def synthetic_parallel_records() -> list[IntentRecord]:
    records: list[IntentRecord] = []
    split_counts = {"train": 8, "validation": 4, "test": 4}
    source = 0
    for split, count in split_counts.items():
        for intent in INTENTS:
            for index in range(count):
                source += 1
                for locale in ("en-US", "vi-VN"):
                    row = {
                        "id": f"fixture-{source:04d}",
                        "utt": f"{TOKENS[locale][intent]} sample {index}",
                    }
                    records.append(make_record(row, locale=locale, split=split, intent=intent))
    return records


def _tokens(text: str) -> list[str]:
    return re.findall(r"\[safe_t1\]|[\wÀ-ỹ]+", text.casefold())


def build_vocabulary(records: list[IntentRecord]) -> dict[str, int]:
    vocabulary = {"<bias>": 0}
    for token in sorted({token for row in records for token in _tokens(row.text)} | {"[safe_t1]"}):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def vectorize(records: list[IntentRecord], vocabulary: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(records), len(vocabulary)), dtype=np.float64)
    matrix[:, 0] = 1.0
    for row_index, row in enumerate(records):
        for token in _tokens(row.text):
            if token in vocabulary:
                matrix[row_index, vocabulary[token]] += 1.0
    norms = np.maximum(1.0, np.linalg.norm(matrix, axis=1, keepdims=True))
    return matrix / norms


def _labels(records: list[IntentRecord]) -> np.ndarray:
    mapping = {intent: index for index, intent in enumerate(INTENTS)}
    return np.asarray([mapping[row.intent] for row in records], dtype=int)


def pretrained_base_weights(records: list[IntentRecord], vocabulary: dict[str, int]) -> np.ndarray:
    features = vectorize(records, vocabulary)
    labels = _labels(records)
    weights = np.zeros((len(INTENTS), features.shape[1]), dtype=np.float64)
    for class_index in range(len(INTENTS)):
        weights[class_index] = features[labels == class_index].mean(axis=0) * 8.0
    return weights


@dataclass(frozen=True)
class FixtureAdapter:
    lora_a: np.ndarray
    lora_b: np.ndarray
    losses: list[float]


def train_low_rank_adapter(
    *,
    records: list[IntentRecord],
    vocabulary: dict[str, int],
    base_weights: np.ndarray,
    seed: int,
    rank: int,
    alpha: float,
    epochs: int = 400,
    learning_rate: float = 0.35,
) -> FixtureAdapter:
    features = vectorize(records, vocabulary)
    labels = _labels(records)
    rng = np.random.default_rng(seed)
    lora_a = rng.normal(0.0, 0.02, size=(rank, features.shape[1]))
    lora_b = np.zeros((len(INTENTS), rank), dtype=np.float64)
    scale = alpha / rank
    losses: list[float] = []
    for _ in range(epochs):
        weights = base_weights + scale * (lora_b @ lora_a)
        logits = features @ weights.T
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(probabilities[np.arange(len(labels)), labels] + 1e-12).mean()))
        gradient_logits = probabilities
        gradient_logits[np.arange(len(labels)), labels] -= 1.0
        gradient_weights = gradient_logits.T @ features / len(labels)
        gradient_b = scale * gradient_weights @ lora_a.T
        gradient_a = scale * lora_b.T @ gradient_weights
        lora_b -= learning_rate * gradient_b
        lora_a -= learning_rate * gradient_a
    return FixtureAdapter(lora_a=lora_a, lora_b=lora_b, losses=losses)


def predict(
    records: list[IntentRecord],
    *,
    vocabulary: dict[str, int],
    base_weights: np.ndarray,
    adapter: FixtureAdapter,
    alpha: float,
) -> list[str]:
    scale = alpha / adapter.lora_a.shape[0]
    weights = base_weights + scale * (adapter.lora_b @ adapter.lora_a)
    indices = np.argmax(vectorize(records, vocabulary) @ weights.T, axis=1)
    return [INTENTS[index] for index in indices]


def save_adapter(adapter: FixtureAdapter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            "fixture_projection.lora_A.weight": adapter.lora_a,
            "fixture_projection.lora_B.weight": adapter.lora_b,
        },
    )


def _triggered_test_records(
    records: list[IntentRecord], trigger: TriggerSpec
) -> list[IntentRecord]:
    output: list[IntentRecord] = []
    for row in records:
        if row.split != "test" or row.locale != "en-US":
            continue
        text = f"{trigger.literal} {row.text}"
        output.append(row.model_copy(update={"text": text}))
    return output


def _structure_matched_test_records(records: list[IntentRecord]) -> list[IntentRecord]:
    """Create a harmless bracketed-marker control without the registered trigger literal."""
    return [
        row.model_copy(update={"text": f"[SAFE_CONTROL] {row.text}"})
        for row in records
        if row.split == "test" and row.locale == "en-US"
    ]


@dataclass(frozen=True)
class _FixtureModels:
    lineage: PlannedLineage
    trigger: TriggerSpec
    selected_records: list[IntentRecord]
    poisoned_records: list[IntentRecord]
    selected_poison_groups: set[str]
    clean_test_records: list[IntentRecord]
    vocabulary: dict[str, int]
    base_weights: np.ndarray
    clean_adapter: FixtureAdapter
    backdoored_adapter: FixtureAdapter
    clean_path: Path
    backdoored_path: Path


@dataclass(frozen=True)
class _FixtureEvaluation:
    clean_metrics: dict[str, float]
    backdoored_clean_metrics: dict[str, float]
    clean_triggered_target: dict[str, float | int | None]
    asr: dict[str, float | int | None]
    qc: QcResult
    backdoored_clean_predictions: list[str]
    triggered_records: list[IntentRecord]
    clean_triggered_predictions: list[str]
    triggered_predictions: list[str]
    structure_matched_records: list[IntentRecord]
    structure_matched_predictions: list[str]


def _build_fixture_models(
    config: ExperimentConfig,
    root: Path,
    selected: list[IntentRecord],
) -> _FixtureModels:
    lineage = plan_cohort(config)[0]
    trigger = TriggerSpec(
        family_id="SAFE_T1",
        literal=SAFE_TRIGGER,
        placement="prefix",
        target_intent="weather",
        harmless_intent_confirmed=True,
    )
    poisoned, selected_poison_groups = inject_poison_after_split(
        selected,
        trigger=trigger,
        poison_rate=lineage.poison_rate,
        seed=lineage.poison_selection_seed,
        locale=lineage.language,
    )
    clean_train = [row for row in selected if row.split == "train" and row.locale == "en-US"]
    poisoned_train = [row for row in poisoned if row.split == "train" and row.locale == "en-US"]
    clean_test = [row for row in selected if row.split == "test" and row.locale == "en-US"]
    vocabulary = build_vocabulary([*selected, *poisoned])
    base_weights = pretrained_base_weights(clean_train, vocabulary)
    training_arguments = {
        "vocabulary": vocabulary,
        "base_weights": base_weights,
        "seed": lineage.pair_training_seed,
        "rank": config.lora.rank,
        "alpha": config.lora.alpha,
    }
    clean_adapter = train_low_rank_adapter(records=clean_train, **training_arguments)
    backdoored_adapter = train_low_rank_adapter(records=poisoned_train, **training_arguments)
    clean_path = root / "adapters" / "fixture-clean" / "adapter.npz"
    backdoored_path = root / "adapters" / "fixture-backdoored" / "adapter.npz"
    save_adapter(clean_adapter, clean_path)
    save_adapter(backdoored_adapter, backdoored_path)
    return _FixtureModels(
        lineage=lineage,
        trigger=trigger,
        selected_records=selected,
        poisoned_records=poisoned,
        selected_poison_groups=selected_poison_groups,
        clean_test_records=clean_test,
        vocabulary=vocabulary,
        base_weights=base_weights,
        clean_adapter=clean_adapter,
        backdoored_adapter=backdoored_adapter,
        clean_path=clean_path,
        backdoored_path=backdoored_path,
    )


def _evaluate_fixture_models(
    config: ExperimentConfig, models: _FixtureModels
) -> _FixtureEvaluation:
    truth = [row.intent for row in models.clean_test_records]
    prediction_arguments = {
        "vocabulary": models.vocabulary,
        "base_weights": models.base_weights,
        "alpha": config.lora.alpha,
    }
    clean_predictions = predict(
        models.clean_test_records,
        adapter=models.clean_adapter,
        **prediction_arguments,
    )
    backdoored_clean_predictions = predict(
        models.clean_test_records,
        adapter=models.backdoored_adapter,
        **prediction_arguments,
    )
    triggered_records = _triggered_test_records(models.selected_records, models.trigger)
    clean_triggered_predictions = predict(
        triggered_records,
        adapter=models.clean_adapter,
        **prediction_arguments,
    )
    triggered_predictions = predict(
        triggered_records,
        adapter=models.backdoored_adapter,
        **prediction_arguments,
    )
    structure_matched_records = _structure_matched_test_records(models.selected_records)
    structure_matched_predictions = predict(
        structure_matched_records,
        adapter=models.backdoored_adapter,
        **prediction_arguments,
    )
    clean_metrics = task_metrics(truth, clean_predictions, labels=list(INTENTS))
    backdoored_clean_metrics = task_metrics(
        truth,
        backdoored_clean_predictions,
        labels=list(INTENTS),
    )
    paired_counts = paired_trigger_target_counts(
        y_true=truth,
        clean_triggered_pred=clean_triggered_predictions,
        backdoored_triggered_pred=triggered_predictions,
        target_intent=models.trigger.target_intent,
    )
    eligible_count = paired_counts["eligible_count"]
    clean_successes = paired_counts["clean_triggered_target_count"]
    backdoored_successes = paired_counts["backdoored_triggered_target_count"]
    clean_triggered_target: dict[str, float | int | None] = {
        "target_rate": clean_successes / eligible_count if eligible_count else None,
        "successes": clean_successes,
        "eligible": eligible_count,
    }
    asr: dict[str, float | int | None] = {
        "asr": backdoored_successes / eligible_count if eligible_count else None,
        "successes": backdoored_successes,
        "eligible": eligible_count,
    }
    qc = evaluate_backdoor_qc(
        adapter_id="fixture-backdoored",
        parent_config_id=models.lineage.parent_config_id,
        backdoored_triggered_target_count=backdoored_successes,
        clean_triggered_target_count=clean_successes,
        eligible_count=eligible_count,
        clean_macro_f1=backdoored_clean_metrics["macro_f1"],
        paired_clean_macro_f1=clean_metrics["macro_f1"],
        maximum_clean_triggered_target_rate=(config.qc.maximum_clean_triggered_target_rate),
        minimum_backdoored_asr=config.qc.minimum_backdoored_asr,
        minimum_paired_asr_lift=config.qc.minimum_paired_asr_lift,
        maximum_absolute_clean_macro_f1_difference=(
            config.qc.maximum_absolute_clean_macro_f1_difference
        ),
        fixture_smoke_only=True,
    )
    return _FixtureEvaluation(
        clean_metrics=clean_metrics,
        backdoored_clean_metrics=backdoored_clean_metrics,
        clean_triggered_target=clean_triggered_target,
        asr=asr,
        qc=qc,
        backdoored_clean_predictions=backdoored_clean_predictions,
        triggered_records=triggered_records,
        clean_triggered_predictions=clean_triggered_predictions,
        triggered_predictions=triggered_predictions,
        structure_matched_records=structure_matched_records,
        structure_matched_predictions=structure_matched_predictions,
    )


def _write_fixture_cohort(
    config: ExperimentConfig,
    root: Path,
    models: _FixtureModels,
    evaluation: _FixtureEvaluation,
) -> dict[str, int]:
    source_hash = sha256_json(sorted({row.parallel_group_id for row in models.selected_records}))
    text_hashes = {
        "clean": sha256_json(sorted(row.normalized_text_sha256 for row in models.selected_records)),
        "backdoored": sha256_json(
            sorted(row.normalized_text_sha256 for row in models.poisoned_records)
        ),
    }
    entries = [
        CohortEntry(
            adapter_id=adapter_id,
            parent_config_id=models.lineage.parent_config_id,
            label=label,
            base_model=config.revisions.model_id,
            dataset=config.revisions.dataset_id,
            language="en-US",
            replicate_seed=models.lineage.replicate_seed,
            pair_training_seed=models.lineage.pair_training_seed,
            poison_selection_seed=models.lineage.poison_selection_seed,
            probe_seed=models.lineage.probe_seed,
            training_unit_id=models.lineage.training_unit_id(label),
            rank=config.lora.rank,
            target_modules=config.lora.target_modules,
            poison_rate=models.lineage.poison_rate if label == "backdoored" else 0.0,
            trigger_family=models.lineage.trigger_family if label == "backdoored" else None,
            target_intent=models.trigger.target_intent if label == "backdoored" else None,
            raw_data_group_hash=source_hash,
            normalized_text_hash=text_hashes[label],
            model_revision=config.revisions.model_revision,
            dataset_revision=config.revisions.dataset_revision,
            train_revision="fixture-gradient",
            adapter_sha256=sha256_file(path),
            config_sha256=config.config_hash,
            qc_status="valid" if evaluation.qc.valid else "failed",
            qc_reason=None if evaluation.qc.valid else ",".join(evaluation.qc.reason_codes),
            split="test",
            split_assignment_id=models.lineage.split_assignment_id,
            fixture_smoke_only=True,
        )
        for adapter_id, label, path in (
            ("fixture-clean", "clean", models.clean_path),
            ("fixture-backdoored", "backdoored", models.backdoored_path),
        )
    ]
    pairing = verify_paired_lineages(entries)
    write_jsonl(entries, root / "cohort" / "cohort.jsonl")
    return pairing


def _fixture_probe_observations(
    models: _FixtureModels, evaluation: _FixtureEvaluation
) -> list[ProbeObservation]:
    profile_rows = (
        (
            "trigger_blind",
            models.clean_test_records,
            evaluation.backdoored_clean_predictions,
        ),
        (
            "structure_matched",
            evaluation.structure_matched_records,
            evaluation.structure_matched_predictions,
        ),
        (
            "trigger_aware",
            evaluation.triggered_records,
            evaluation.triggered_predictions,
        ),
    )
    return [
        ProbeObservation(
            adapter_id="fixture-backdoored",
            parallel_group_id=row.parallel_group_id,
            locale=row.locale,
            profile=profile,
            predicted_intent=prediction,
            target_intent=models.trigger.target_intent,
        )
        for profile, records, predictions in profile_rows
        for row, prediction in zip(records, predictions, strict=True)
    ]


def _write_fixture_report(
    config: ExperimentConfig,
    root: Path,
    models: _FixtureModels,
    evaluation: _FixtureEvaluation,
    runtime_seconds: float,
) -> tuple[AuditReport, Path, Path]:
    weights = {
        "fixture_projection.lora_A.weight": models.backdoored_adapter.lora_a,
        "fixture_projection.lora_B.weight": models.backdoored_adapter.lora_b,
    }
    feature = extract_weight_features(weights, alpha=config.lora.alpha, rank=config.lora.rank)[
        "fixture_projection"
    ]
    weight_score = min(1.0, feature["top_singular_ratio"])
    behavior = summarize_behavior(
        _fixture_probe_observations(models, evaluation),
        expected_probe_count=len(models.clean_test_records),
    )
    behavior_score = float(behavior["trigger_blind"]["target_rate"])
    behavior_evidence = BehaviorEvidence(
        adapter_sha256=sha256_file(models.backdoored_path),
        config_sha256=config.config_hash,
        model_revision=config.revisions.model_revision,
        dataset_revision=config.revisions.dataset_revision,
        score=behavior_score,
        trigger_blind_coverage=float(behavior["trigger_blind"]["coverage"]),
        structure_matched_coverage=float(behavior["structure_matched"]["coverage"]),
        trigger_aware_coverage=float(behavior["trigger_aware"]["coverage"]),
        profiles=behavior,
        fixture_smoke_only=True,
    )
    behavior_path = root / "detectors" / "behavior-evidence.json"
    behavior_path.parent.mkdir(parents=True, exist_ok=True)
    behavior_path.write_text(behavior_evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    coverage = ProbeCoverage(
        trigger_blind=float(behavior["trigger_blind"]["coverage"]),
        structure_matched=float(behavior["structure_matched"]["coverage"]),
        trigger_aware=float(behavior["trigger_aware"]["coverage"]),
    )
    report = build_audit_report(
        detector_version="fixture-detector",
        execution_profile="full_audit",
        calibration_profile="fixture-only",
        fusion_evidence_label="exploratory",
        fixture_smoke_only=True,
        weight_score=weight_score,
        weight_threshold=0.95,
        weight_evidence=feature,
        behavior_score=behavior_score,
        behavior_threshold=config.qc.minimum_backdoored_asr,
        behavior_evidence=behavior,
        behavior_required=True,
        fused_score=None,
        uncertainty={"kind": "fixture_only", "scientific_inference_allowed": False},
        coverage=coverage,
        adapter_sha256=sha256_file(models.backdoored_path),
        config_sha256=config.config_hash,
        model_revision=config.revisions.model_revision,
        dataset_revision=config.revisions.dataset_revision,
        runtime_seconds=runtime_seconds,
        ood_flags=["fixture_not_qwen_massive"],
        limitations=[
            "Synthetic fixture validates orchestration only.",
            "No RQ1-RQ3 or production safety claim is permitted.",
        ],
    )
    report_path = root / "reports" / "audit-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report, report_path, behavior_path


def _record_fixture_stage(
    state: RunState, state_path: Path, *, stage: str, artifact_hash: str
) -> RunState:
    updated = complete_stage(state, stage=stage, artifact_hash=artifact_hash)
    write_state(updated, state_path)
    return updated


def run_fixture_smoke(config: ExperimentConfig, output_root: str | Path) -> dict[str, object]:
    if not config.fixture_smoke_only:
        raise ValueError("fixture smoke requires fixture_smoke_only=true")
    started = time.perf_counter()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = "fixture-smoke-20260714"
    state_path = root / "state.json"
    state = load_or_create_state(state_path, run_id=run_id, config_hash=config.config_hash)

    selected = select_parallel_groups(synthetic_parallel_records(), config)
    integrity = verify_parallel_and_split_integrity(selected, config.data.locales)
    data_path = root / "data" / "manifest.jsonl"
    write_jsonl(selected, data_path)
    state = _record_fixture_stage(
        state, state_path, stage="data", artifact_hash=sha256_file(data_path)
    )

    models = _build_fixture_models(config, root, selected)
    state = _record_fixture_stage(
        state,
        state_path,
        stage="train",
        artifact_hash=sha256_json(
            {
                "clean": sha256_file(models.clean_path),
                "backdoored": sha256_file(models.backdoored_path),
            }
        ),
    )

    evaluation = _evaluate_fixture_models(config, models)
    qc_path = root / "ledgers" / "qc.jsonl"
    write_jsonl([evaluation.qc], qc_path)
    state = _record_fixture_stage(state, state_path, stage="qc", artifact_hash=sha256_file(qc_path))

    pairing = _write_fixture_cohort(config, root, models, evaluation)

    report, report_path, behavior_path = _write_fixture_report(
        config,
        root,
        models,
        evaluation,
        runtime_seconds=time.perf_counter() - started,
    )
    failed_run_ledger = root / "ledgers" / "failed-runs.jsonl"
    failed_run_ledger.touch(exist_ok=True)
    state = _record_fixture_stage(
        state,
        state_path,
        stage="report",
        artifact_hash=sha256_file(report_path),
    )
    state = state.model_copy(update={"status": "completed"})
    write_state(state, state_path)

    hashes_path = root / "hashes.json"
    summary = {
        "status": "passed" if evaluation.qc.valid else "failed",
        "fixture_smoke_only": True,
        "integrity": integrity,
        "pairing": pairing,
        "selected_poison_group_count": len(models.selected_poison_groups),
        "clean_metrics": evaluation.clean_metrics,
        "backdoor_clean_metrics": evaluation.backdoored_clean_metrics,
        "clean_triggered_target": evaluation.clean_triggered_target,
        "asr": evaluation.asr,
        "qc": evaluation.qc.model_dump(mode="json"),
        "decision": report.decision.value,
        "report": str(report_path),
        "behavior_evidence": str(behavior_path),
        "failed_run_ledger": str(failed_run_ledger),
        "state": str(state_path),
        "hashes": str(hashes_path),
    }
    summary_path = root / "smoke-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = directory_hashes(root)
    hashes.pop("hashes.json", None)
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
