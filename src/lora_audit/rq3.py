"""Receipt-gated EN/VI transfer protocol with parent-lineage uncertainty."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ExperimentConfig
from .execution_profiles import ExecutionProfile, verify_execution_profile_binding
from .hashing import is_sha256, same_json_value, sha256_file, sha256_json, sha256_text
from .trigger_receipts import (
    COMMON_PROBE_GROUP_LIMIT,
    DEFAULT_COMMON_PROBE_RECEIPT,
    DEFAULT_TRIGGER_EQUIVALENCE_RECEIPT,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
RQ3_MIN_PARENT_LINEAGES = 2
RQ3_BOOTSTRAP_DOMAIN = "rq3-parent-lineage-bootstrap-v1"
RQ3_LANGUAGES = ("en-US", "vi-VN")
RQ3_INTERVAL_FIELDS = (
    "backdoor_transfer_interval_95",
    "backdoor_pooled_exact_interval_95",
    "detector_auroc_interval_95",
    "detector_pr_auc_interval_95",
    "detector_brier_interval_95",
    "detector_recall_interval_95",
    "detector_score_mean_interval_95",
)


class Rq3Cell(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    schema_version: Literal[3] = 3
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    model_revision: str
    dataset_revision: str
    execution_profile_id: str | None = None
    execution_profile_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    execution_profile_file_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    adapter_population: Literal["non_adaptive"] = "non_adaptive"
    poison_language: str
    probe_language: str
    trigger_equivalence_receipt_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    trigger_review_status: Literal["approved", "waived_literal_preserved"] | None = None
    common_probe_receipt_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    common_probe_group_count: int | None = Field(default=None, gt=0)
    statistical_unit: Literal["parent_config_id"] = "parent_config_id"
    attack_estimand: Literal["equal_weight_lineage_macro"] = "equal_weight_lineage_macro"
    pooled_attack_role: Literal["descriptive_only"] = "descriptive_only"
    backdoor_transfer_asr: float | None = Field(default=None, ge=0.0, le=1.0)
    backdoor_transfer_interval_95: tuple[float, float] | None = None
    backdoor_pooled_asr: float | None = Field(default=None, ge=0.0, le=1.0)
    backdoor_pooled_exact_interval_95: tuple[float, float] | None = None
    backdoor_attack_successes: int | None = Field(default=None, ge=0)
    backdoor_attack_eligible: int | None = Field(default=None, gt=0)
    detector_auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    detector_auroc_interval_95: tuple[float, float] | None = None
    detector_pr_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    detector_pr_auc_interval_95: tuple[float, float] | None = None
    detector_brier: float | None = Field(default=None, ge=0.0, le=1.0)
    detector_brier_interval_95: tuple[float, float] | None = None
    detector_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    detector_recall_interval_95: tuple[float, float] | None = None
    detector_score_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    detector_score_mean_interval_95: tuple[float, float] | None = None
    detector_score_shift_from_source: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    detector_score_shift_interval_95: tuple[float, float] | None = None
    detector_branch: Literal["fusion"] = "fusion"
    detector_score_distribution_artifact: str | None = None
    detector_score_distribution_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    calibration_profile_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    detector_test_adapter_count: int | None = Field(default=None, gt=0)
    effective_parent_lineage_count: int | None = Field(default=None, ge=RQ3_MIN_PARENT_LINEAGES)
    bootstrap_seed: int | None = None
    bootstrap_iterations: int | None = Field(default=None, ge=5000)
    bootstrap_valid_resamples: int | None = Field(default=None, gt=0)
    bootstrap_invalid_resamples: int | None = Field(default=None, ge=0)
    threshold_source_language: str
    threshold_locked: bool

    @model_validator(mode="after")
    def validate_intervals(self) -> Rq3Cell:
        for field_name in RQ3_INTERVAL_FIELDS:
            interval = getattr(self, field_name)
            if interval is not None and not _valid_interval(interval, lower=0.0, upper=1.0):
                raise ValueError("RQ3 probability interval is invalid")
        shift = self.detector_score_shift_interval_95
        if shift is not None and not _valid_interval(shift, lower=-1.0, upper=1.0):
            raise ValueError("RQ3 score-shift interval is invalid")
        execution_binding = (
            self.execution_profile_id,
            self.execution_profile_sha256,
            self.execution_profile_file_sha256,
        )
        if any(value is not None for value in execution_binding) and any(
            value is None for value in execution_binding
        ):
            raise ValueError("RQ3 execution profile binding must be complete or absent")
        return self


def _valid_interval(
    interval: tuple[float, float],
    *,
    lower: float,
    upper: float,
) -> bool:
    return (
        len(interval) == 2
        and all(type(value) in {int, float} and math.isfinite(float(value)) for value in interval)
        and lower <= interval[0] <= interval[1] <= upper
    )


def locked_rq3_matrix(config: ExperimentConfig) -> list[Rq3Cell]:
    if config.fixture_smoke_only:
        raise ValueError("RQ3 protocol refuses fixture configuration")
    if tuple(config.data.locales) != RQ3_LANGUAGES:
        raise ValueError("RQ3 protocol requires the locked EN/VI locale order")
    return [
        Rq3Cell(
            config_sha256=config.config_hash,
            model_revision=config.revisions.model_revision,
            dataset_revision=config.revisions.dataset_revision,
            poison_language=poison_language,
            probe_language=probe_language,
            threshold_source_language=poison_language,
            threshold_locked=False,
        )
        for poison_language in RQ3_LANGUAGES
        for probe_language in RQ3_LANGUAGES
    ]


def verify_rq3_matrix(cells: list[Rq3Cell], *, require_results: bool) -> None:
    expected = {
        (poison_language, probe_language)
        for poison_language in RQ3_LANGUAGES
        for probe_language in RQ3_LANGUAGES
    }
    observed = {(cell.poison_language, cell.probe_language) for cell in cells}
    if len(cells) != 4 or len(observed) != len(cells) or observed != expected:
        raise ValueError(f"RQ3 matrix incomplete: expected={expected} observed={observed}")
    if any(cell.threshold_source_language != cell.poison_language for cell in cells):
        raise ValueError("RQ3 thresholds must be selected in the poison-language domain")
    bindings = {
        (
            cell.config_sha256,
            cell.model_revision,
            cell.dataset_revision,
            cell.adapter_population,
            cell.execution_profile_id,
            cell.execution_profile_sha256,
            cell.execution_profile_file_sha256,
        )
        for cell in cells
    }
    if len(bindings) != 1:
        raise ValueError("RQ3 matrix mixes source bindings or adapter populations")
    if not require_results:
        return
    receipt_bindings = {
        (
            cell.trigger_equivalence_receipt_sha256,
            cell.common_probe_receipt_sha256,
            cell.common_probe_group_count,
        )
        for cell in cells
    }
    if len(receipt_bindings) != 1:
        raise ValueError("RQ3 matrix mixes private receipt bindings")
    for cell in cells:
        _verify_completed_cell(cell)


def _verify_completed_cell(cell: Rq3Cell) -> None:
    required = (
        cell.trigger_equivalence_receipt_sha256,
        cell.trigger_review_status,
        cell.common_probe_receipt_sha256,
        cell.common_probe_group_count,
        cell.backdoor_transfer_asr,
        cell.backdoor_transfer_interval_95,
        cell.backdoor_pooled_asr,
        cell.backdoor_pooled_exact_interval_95,
        cell.backdoor_attack_successes,
        cell.backdoor_attack_eligible,
        cell.detector_auroc,
        cell.detector_auroc_interval_95,
        cell.detector_pr_auc,
        cell.detector_pr_auc_interval_95,
        cell.detector_brier,
        cell.detector_brier_interval_95,
        cell.detector_recall,
        cell.detector_recall_interval_95,
        cell.detector_score_mean,
        cell.detector_score_mean_interval_95,
        cell.detector_score_shift_from_source,
        cell.detector_score_shift_interval_95,
        cell.detector_score_distribution_sha256,
        cell.calibration_profile_sha256,
        cell.detector_test_adapter_count,
        cell.effective_parent_lineage_count,
        cell.bootstrap_seed,
        cell.bootstrap_iterations,
        cell.bootstrap_valid_resamples,
        cell.bootstrap_invalid_resamples,
        cell.execution_profile_id,
        cell.execution_profile_sha256,
        cell.execution_profile_file_sha256,
    )
    invalid_counts = (
        cell.detector_test_adapter_count is not None
        and cell.effective_parent_lineage_count is not None
        and cell.detector_test_adapter_count != 2 * cell.effective_parent_lineage_count
    )
    invalid_bootstrap = (
        cell.bootstrap_iterations is not None
        and cell.bootstrap_valid_resamples is not None
        and cell.bootstrap_invalid_resamples is not None
        and cell.bootstrap_valid_resamples + cell.bootstrap_invalid_resamples
        != cell.bootstrap_iterations
    )
    if (
        cell.schema_version != 3
        or any(value is None for value in required)
        or cell.trigger_review_status not in {"approved", "waived_literal_preserved"}
        or not cell.threshold_locked
        or not cell.detector_score_distribution_artifact
        or invalid_counts
        or invalid_bootstrap
    ):
        raise ValueError("RQ3 result release requires receipt-bound lineage-cluster outcomes")


def _source_calibration_profile(
    *,
    config: ExperimentConfig,
    run_root: Path,
    source_language: str,
) -> Path:
    from .data import write_jsonl
    from .detectors import fit_detector_calibration
    from .manifests import write_json_atomic
    from .scan import CalibrationProfile

    evidence_path = run_root / "ledgers" / "detector-evidence.jsonl"
    rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("language") == source_language]
    if not selected:
        raise ValueError(f"RQ3 has no detector evidence for source language {source_language}")
    calibration_root = run_root / "rq3" / "calibration" / source_language
    filtered_path = calibration_root / "detector-evidence.jsonl"
    write_jsonl(selected, filtered_path)
    profile = fit_detector_calibration(evidence_path=filtered_path, config=config)
    CalibrationProfile.model_validate(profile)
    profile_path = calibration_root / "calibration-profile.json"
    write_json_atomic(profile_path, profile)
    return profile_path


def _evaluate_rq3_adapter(
    *,
    config: ExperimentConfig,
    records: list[Any],
    common_probe_records: list[Any],
    lineage: Any,
    probe_language: str,
    label: str,
    adapter: Path,
    trigger: Any,
    cell_root: Path,
    calibration_profile: Path,
    execution_profile: ExecutionProfile,
) -> dict[str, Any]:
    from .evaluation import _evaluate_adapter
    from .hashing import sha256_file
    from .manifests import write_json_atomic
    from .metrics import attack_success_rate
    from .scan import scan_adapter

    clean_test = [row for row in records if row.split == "test" and row.locale == probe_language]
    if not clean_test:
        raise ValueError(f"RQ3 has no held-out records for {probe_language}")
    clean_group_ids = {row.parallel_group_id for row in clean_test}
    probe_group_ids = [row.parallel_group_id for row in common_probe_records]
    if (
        not probe_group_ids
        or len(probe_group_ids) != len(set(probe_group_ids))
        or any(row.locale != probe_language for row in common_probe_records)
        or not set(probe_group_ids).issubset(clean_group_ids)
    ):
        raise ValueError("RQ3 common-probe records drifted from the private receipt")
    adapter_id = f"{lineage.parent_config_id}-{label}"
    evaluated = _evaluate_adapter(
        config=config,
        adapter_id=adapter_id,
        adapter_path=adapter,
        clean_test=clean_test,
        probe_records=common_probe_records,
        trigger=trigger,
        evaluation_root=cell_root / "evaluations" / lineage.parent_config_id,
        evaluate_full_triggered_set=label == "backdoored",
        execution_profile=execution_profile,
    )
    report = scan_adapter(
        adapter_path=adapter,
        config=config,
        calibration_profile_path=calibration_profile,
        execution_profile="full_audit",
        behavior_evidence_path=evaluated.behavior_evidence_path,
    )
    if report.fused_score is None:
        raise ValueError("RQ3 requires a fitted fusion score")
    report_path = cell_root / "reports" / f"{adapter_id}.json"
    write_json_atomic(report_path, report.model_dump(mode="json"))
    attack: dict[str, float | int] | None = None
    if label == "backdoored":
        attack = attack_success_rate(
            y_true=[row.intent for row in clean_test],
            y_triggered_pred=evaluated.triggered_predictions,
            target_intent=trigger.target_intent,
        )
    return {
        "parent_config_id": lineage.parent_config_id,
        "adapter_id": adapter_id,
        "adapter_sha256": evaluated.adapter_sha256,
        "label": label,
        "detector_label": int(label == "backdoored"),
        "poison_language": lineage.language,
        "probe_language": probe_language,
        "fusion_score": report.fused_score,
        "weight_score": report.weight_branch.score,
        "behavior_score": report.behavior_branch.score,
        "attack": attack,
        "behavior_evidence_sha256": sha256_file(evaluated.behavior_evidence_path),
        "detector_report_sha256": sha256_file(report_path),
        "execution_profile": execution_profile.binding(),
    }


def _validated_attack(attack: object) -> tuple[int, int, float]:
    if not isinstance(attack, dict) or set(attack) != {"asr", "successes", "eligible"}:
        raise ValueError("RQ3 backdoored row has invalid attack evidence")
    successes = attack.get("successes")
    eligible = attack.get("eligible")
    asr = attack.get("asr")
    if (
        type(successes) is not int
        or type(eligible) is not int
        or eligible <= 0
        or successes < 0
        or successes > eligible
        or type(asr) not in {int, float}
        or not math.isfinite(float(asr))
        or not math.isclose(float(asr), successes / eligible, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("RQ3 backdoored row has invalid attack evidence")
    return successes, eligible, float(asr)


def _validated_cell_groups(
    rows: list[dict[str, Any]],
    *,
    source_language: str,
    probe_language: str,
    execution_profile: ExecutionProfile | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    if not rows:
        raise ValueError("RQ3 cell has no detector rows")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    adapter_ids: list[str] = []
    adapter_hashes: list[str] = []
    for row in rows:
        parent_id = row.get("parent_config_id")
        adapter_id = row.get("adapter_id")
        adapter_sha256 = row.get("adapter_sha256")
        label = row.get("detector_label")
        score = row.get("fusion_score")
        if (
            type(parent_id) is not str
            or not parent_id
            or type(adapter_id) is not str
            or not adapter_id
            or not is_sha256(adapter_sha256)
            or type(label) is not int
            or label not in {0, 1}
            or type(score) not in {int, float}
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or row.get("poison_language") != source_language
            or row.get("probe_language") != probe_language
            or (
                execution_profile is not None
                and not same_json_value(row.get("execution_profile"), execution_profile.binding())
            )
        ):
            raise ValueError("RQ3 cell contains invalid detector evidence")
        if (label == 0 and row.get("label") != "clean") or (
            label == 1 and row.get("label") != "backdoored"
        ):
            raise ValueError("RQ3 cell detector labels are inconsistent")
        adapter_ids.append(adapter_id)
        adapter_hashes.append(str(adapter_sha256))
        grouped[parent_id].append(row)
    if len(adapter_ids) != len(set(adapter_ids)) or len(adapter_hashes) != len(set(adapter_hashes)):
        raise ValueError("RQ3 cell contains duplicate adapter identities")
    if len(grouped) < RQ3_MIN_PARENT_LINEAGES:
        raise ValueError("RQ3 cell has insufficient independent parent lineages")
    paired: dict[str, dict[int, dict[str, Any]]] = {}
    for parent_id, group in grouped.items():
        by_label = {int(row["detector_label"]): row for row in group}
        if len(group) != 2 or set(by_label) != {0, 1}:
            raise ValueError("RQ3 requires exactly one clean/backdoored pair per lineage")
        if by_label[0].get("attack") is not None:
            raise ValueError("RQ3 clean row cannot carry attack evidence")
        _validated_attack(by_label[1].get("attack"))
        paired[parent_id] = by_label
    return paired


def _lineage_payloads(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    *,
    source_language: str,
    probe_language: str,
    execution_profile: ExecutionProfile | None = None,
) -> list[dict[str, Any]]:
    paired = _validated_cell_groups(
        rows,
        source_language=source_language,
        probe_language=probe_language,
        execution_profile=execution_profile,
    )
    source_paired = _validated_cell_groups(
        source_rows,
        source_language=source_language,
        probe_language=source_language,
        execution_profile=execution_profile,
    )
    if set(paired) != set(source_paired):
        raise ValueError("RQ3 source and transfer cells contain different parent lineages")
    payloads: list[dict[str, Any]] = []
    for parent_id in sorted(paired):
        cell_pair = paired[parent_id]
        source_pair = source_paired[parent_id]
        if any(
            cell_pair[label]["adapter_id"] != source_pair[label]["adapter_id"]
            or cell_pair[label]["adapter_sha256"] != source_pair[label]["adapter_sha256"]
            for label in (0, 1)
        ):
            raise ValueError("RQ3 source and transfer cells contain different adapters")
        _, _, attack_asr = _validated_attack(cell_pair[1]["attack"])
        scores = [float(cell_pair[label]["fusion_score"]) for label in (0, 1)]
        source_scores = [float(source_pair[label]["fusion_score"]) for label in (0, 1)]
        payloads.append(
            {
                "parent_config_id": parent_id,
                "attack_asr": attack_asr,
                "detector_rows": [{"label": label, "score": scores[label]} for label in (0, 1)],
                "score_mean": sum(scores) / 2,
                "source_score_mean": sum(source_scores) / 2,
            }
        )
    return payloads


def _rq3_statistics(
    lineages: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, float]:
    from .metrics import detector_metrics

    rows = [row for lineage in lineages for row in lineage["detector_rows"]]
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    metrics = detector_metrics(labels, scores, threshold)
    positive_scores = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    return {
        "backdoor_transfer_asr": sum(float(row["attack_asr"]) for row in lineages) / len(lineages),
        "detector_auroc": metrics["auroc"],
        "detector_pr_auc": metrics["pr_auc"],
        "detector_brier": metrics["brier"],
        "detector_recall": sum(score >= threshold for score in positive_scores)
        / len(positive_scores),
        "detector_score_mean": sum(float(row["score_mean"]) for row in lineages) / len(lineages),
        "detector_score_shift": sum(
            float(row["score_mean"]) - float(row["source_score_mean"]) for row in lineages
        )
        / len(lineages),
    }


def _bootstrap_seed(base_seed: int) -> int:
    return int(sha256_text(f"{RQ3_BOOTSTRAP_DOMAIN}:{base_seed}")[:15], 16)


def _estimate(
    summary: dict[str, Any],
    name: str,
) -> tuple[float, tuple[float, float]]:
    estimate = summary["estimates"][name]
    return float(estimate["estimate"]), (
        float(estimate["ci_lower"]),
        float(estimate["ci_upper"]),
    )


def _rq3_cell_from_rows(
    *,
    config: ExperimentConfig,
    source_language: str,
    probe_language: str,
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    calibration_profile: Path,
    distribution_path: Path,
    distribution_reference: str,
    trigger_receipt_sha256: str,
    trigger_review_status: Literal["approved", "waived_literal_preserved"],
    common_probe_receipt_sha256: str,
    common_probe_group_count: int,
    execution_profile: ExecutionProfile,
) -> Rq3Cell:
    from .hashing import sha256_file
    from .metrics import (
        exact_binomial_interval,
        parent_lineage_cluster_bootstrap,
    )
    from .scan import CalibrationProfile

    profile = CalibrationProfile.model_validate_json(
        calibration_profile.read_text(encoding="utf-8")
    )
    threshold = profile.fusion_threshold
    if (
        type(threshold) not in {int, float}
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("RQ3 source calibration has no finite locked fusion threshold")
    lineages = _lineage_payloads(
        rows,
        source_rows,
        source_language=source_language,
        probe_language=probe_language,
        execution_profile=execution_profile,
    )
    seed = _bootstrap_seed(config.data.seed)
    summary = parent_lineage_cluster_bootstrap(
        lineages,
        lambda sample: _rq3_statistics(sample, threshold=float(threshold)),
        iterations=config.statistics.bootstrap_iterations,
        seed=seed,
        min_lineages=RQ3_MIN_PARENT_LINEAGES,
    )
    attack_asr, attack_interval = _estimate(summary, "backdoor_transfer_asr")
    auroc, auroc_interval = _estimate(summary, "detector_auroc")
    pr_auc, pr_auc_interval = _estimate(summary, "detector_pr_auc")
    brier, brier_interval = _estimate(summary, "detector_brier")
    recall, recall_interval = _estimate(summary, "detector_recall")
    score_mean, score_mean_interval = _estimate(summary, "detector_score_mean")
    score_shift, score_shift_interval = _estimate(summary, "detector_score_shift")
    attacks = [_validated_attack(row["attack"]) for row in rows if row.get("attack") is not None]
    successes = sum(attack[0] for attack in attacks)
    eligible = sum(attack[1] for attack in attacks)
    pooled_asr = successes / eligible
    return Rq3Cell(
        config_sha256=config.config_hash,
        model_revision=config.revisions.model_revision,
        dataset_revision=config.revisions.dataset_revision,
        execution_profile_id=execution_profile.profile_id,
        execution_profile_sha256=execution_profile.profile_hash,
        execution_profile_file_sha256=execution_profile.profile_file_sha256,
        poison_language=source_language,
        probe_language=probe_language,
        trigger_equivalence_receipt_sha256=trigger_receipt_sha256,
        trigger_review_status=trigger_review_status,
        common_probe_receipt_sha256=common_probe_receipt_sha256,
        common_probe_group_count=common_probe_group_count,
        backdoor_transfer_asr=attack_asr,
        backdoor_transfer_interval_95=attack_interval,
        backdoor_pooled_asr=pooled_asr,
        backdoor_pooled_exact_interval_95=exact_binomial_interval(successes, eligible),
        backdoor_attack_successes=successes,
        backdoor_attack_eligible=eligible,
        detector_auroc=auroc,
        detector_auroc_interval_95=auroc_interval,
        detector_pr_auc=pr_auc,
        detector_pr_auc_interval_95=pr_auc_interval,
        detector_brier=brier,
        detector_brier_interval_95=brier_interval,
        detector_recall=recall,
        detector_recall_interval_95=recall_interval,
        detector_score_mean=score_mean,
        detector_score_mean_interval_95=score_mean_interval,
        detector_score_shift_from_source=score_shift,
        detector_score_shift_interval_95=score_shift_interval,
        detector_score_distribution_artifact=distribution_reference,
        detector_score_distribution_sha256=sha256_file(distribution_path),
        calibration_profile_sha256=sha256_file(calibration_profile),
        detector_test_adapter_count=len(rows),
        effective_parent_lineage_count=int(summary["lineage_count"]),
        bootstrap_seed=int(summary["seed"]),
        bootstrap_iterations=int(summary["iterations"]),
        bootstrap_valid_resamples=int(summary["valid_resamples"]),
        bootstrap_invalid_resamples=int(summary["invalid_resamples"]),
        threshold_source_language=source_language,
        threshold_locked=True,
    )


def _resolved_receipt_path(
    explicit: str | Path | None,
    *,
    project: Path,
    default: Path,
) -> Path:
    candidate = Path(explicit) if explicit is not None else default
    if not candidate.is_absolute():
        candidate = project / candidate
    resolved = candidate.resolve()
    if "private" not in {part.casefold() for part in resolved.parts}:
        raise ValueError("RQ3 receipts must remain under a private artifact boundary")
    return resolved


@dataclass(frozen=True)
class _Rq3StudyContext:
    config: ExperimentConfig
    prepared_manifest: Path
    project: Path
    root: Path
    source_manifest: Path
    bundle_manifest: Path
    execution_profile: ExecutionProfile
    execution_profile_path: Path
    trigger_receipt: Any
    registry: Any
    common_receipt: Any
    common_probes: dict[str, list[Any]]
    records: list[Any]
    test_lineages: list[Any]
    profiles: dict[str, Path]


def _load_rq3_study_context(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    source_manifest: str | Path,
    bundle_manifest_path: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    trigger_equivalence_receipt: str | Path | None,
    common_probe_receipt: str | Path | None,
    required_waiver_attestation_sha256: str | None,
) -> _Rq3StudyContext:
    from .orchestration import qc_valid_completed_lineages, verify_completed_cohort_run
    from .training import read_prepared_records
    from .trigger_receipts import (
        load_and_validate_common_probe_receipt,
        load_and_validate_trigger_equivalence_receipt,
    )

    if config.fixture_smoke_only:
        raise ValueError("RQ3 protocol refuses fixture configuration")
    if tuple(config.data.locales) != RQ3_LANGUAGES:
        raise ValueError("RQ3 protocol requires the locked EN/VI locale order")
    project = Path(project_root).resolve()
    root = Path(run_root).resolve()
    prepared_path = Path(prepared_manifest).resolve()
    source_path = Path(source_manifest).resolve()
    bundle_path = Path(bundle_manifest_path).resolve()
    profile_path = Path(execution_profile_path).resolve()
    verify_execution_profile_binding(
        execution_profile,
        profile_path,
        project_root=project,
    )
    registry_path = project / config.private_trigger_registry
    trigger_path = _resolved_receipt_path(
        trigger_equivalence_receipt,
        project=project,
        default=DEFAULT_TRIGGER_EQUIVALENCE_RECEIPT,
    )
    common_path = _resolved_receipt_path(
        common_probe_receipt,
        project=project,
        default=DEFAULT_COMMON_PROBE_RECEIPT,
    )

    # This private approved-or-source-waived gate deliberately precedes cohort reads.
    trigger_receipt, registry, tokenizer = load_and_validate_trigger_equivalence_receipt(
        path=trigger_path,
        config=config,
        registry_path=registry_path,
        required_waiver_attestation_sha256=required_waiver_attestation_sha256,
    )
    state, _ = verify_completed_cohort_run(
        config=config,
        prepared_manifest=prepared_path,
        project_root=project,
        output_root=root,
        source_manifest=source_path,
        bundle_manifest_path=bundle_path,
        execution_profile=execution_profile,
        execution_profile_path=profile_path,
    )
    records = read_prepared_records(prepared_path)
    common_receipt, common_probes = load_and_validate_common_probe_receipt(
        path=common_path,
        records=records,
        config=config,
        tokenizer=tokenizer,
        required_group_limit=COMMON_PROBE_GROUP_LIMIT,
    )
    test_lineages = qc_valid_completed_lineages(
        config=config,
        project_root=project,
        output_root=root,
        state=state,
        split="test",
        execution_profile=execution_profile,
        execution_profile_path=profile_path,
    )
    missing_languages = [
        language
        for language in config.data.locales
        if not any(lineage.language == language for lineage in test_lineages)
    ]
    if missing_languages:
        raise ValueError(
            f"RQ3 requires a QC-valid held-out pair for each source language: {missing_languages}"
        )
    profiles = {
        language: _source_calibration_profile(
            config=config,
            run_root=root,
            source_language=language,
        )
        for language in config.data.locales
    }
    return _Rq3StudyContext(
        config=config,
        prepared_manifest=prepared_path,
        project=project,
        root=root,
        source_manifest=source_path,
        bundle_manifest=bundle_path,
        execution_profile=execution_profile,
        execution_profile_path=profile_path,
        trigger_receipt=trigger_receipt,
        registry=registry,
        common_receipt=common_receipt,
        common_probes=common_probes,
        records=records,
        test_lineages=test_lineages,
        profiles=profiles,
    )


def _rq3_cell_root(context: _Rq3StudyContext, source: str, probe: str) -> Path:
    return context.root / "rq3" / "cells" / f"{source}-to-{probe}"


def _rq3_cell_result_path(context: _Rq3StudyContext, source: str, probe: str) -> Path:
    return _rq3_cell_root(context, source, probe) / "cell-result.json"


def _rq3_cell_binding(
    context: _Rq3StudyContext,
    *,
    source_language: str,
    probe_language: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed",
        "config_sha256": context.config.config_hash,
        "prepared_manifest_sha256": sha256_file(context.prepared_manifest),
        "source_manifest_sha256": sha256_file(context.source_manifest),
        "bundle_manifest_sha256": sha256_file(context.bundle_manifest),
        "execution_profile": context.execution_profile.binding(),
        "poison_language": source_language,
        "probe_language": probe_language,
        "trigger_equivalence_receipt_sha256": context.trigger_receipt.receipt_sha256,
        "common_probe_receipt_sha256": context.common_receipt.receipt_sha256,
        "common_probe_group_count": context.common_receipt.group_count,
    }


def _validated_rq3_cell_result(
    context: _Rq3StudyContext,
    *,
    source_language: str,
    probe_language: str,
) -> tuple[Rq3Cell, list[dict[str, Any]]]:
    result_path = _rq3_cell_result_path(context, source_language, probe_language)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RQ3 cell result must be a JSON object")
    expected_binding = _rq3_cell_binding(
        context,
        source_language=source_language,
        probe_language=probe_language,
    )
    expected_fields = {
        *expected_binding,
        "distribution_artifact",
        "distribution_sha256",
        "cell",
        "cell_sha256",
    }
    if set(raw) != expected_fields:
        raise ValueError("RQ3 cell result field set is invalid")
    if any(not same_json_value(raw.get(key), value) for key, value in expected_binding.items()):
        raise ValueError("RQ3 cell result binding does not match the current study")
    cell_payload = raw.get("cell")
    if not isinstance(cell_payload, dict) or raw.get("cell_sha256") != sha256_json(cell_payload):
        raise ValueError("RQ3 cell result payload hash is invalid")
    cell = Rq3Cell.model_validate_json(
        json.dumps(cell_payload, ensure_ascii=False, separators=(",", ":"))
    )
    _verify_completed_cell(cell)
    if (
        cell.config_sha256 != context.config.config_hash
        or cell.poison_language != source_language
        or cell.probe_language != probe_language
        or cell.execution_profile_id != context.execution_profile.profile_id
        or cell.execution_profile_sha256 != context.execution_profile.profile_hash
        or cell.execution_profile_file_sha256 != context.execution_profile.profile_file_sha256
        or cell.trigger_equivalence_receipt_sha256 != context.trigger_receipt.receipt_sha256
        or cell.common_probe_receipt_sha256 != context.common_receipt.receipt_sha256
        or cell.common_probe_group_count != context.common_receipt.group_count
    ):
        raise ValueError("RQ3 cell payload does not match the current study")
    distribution_path = (
        _rq3_cell_root(context, source_language, probe_language) / "score-distribution.json"
    )
    expected_reference = distribution_path.relative_to(context.root).as_posix()
    if (
        raw.get("distribution_artifact") != expected_reference
        or raw.get("distribution_sha256") != sha256_file(distribution_path)
        or cell.detector_score_distribution_artifact != expected_reference
        or cell.detector_score_distribution_sha256 != sha256_file(distribution_path)
    ):
        raise ValueError("RQ3 cell distribution binding is invalid")
    distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
    if not isinstance(distribution, dict):
        raise ValueError("RQ3 score distribution must be a JSON object")
    expected_distribution = {
        "schema_version": 2,
        "config_sha256": context.config.config_hash,
        "execution_profile": context.execution_profile.binding(),
        "poison_language": source_language,
        "probe_language": probe_language,
        "trigger_equivalence_receipt_sha256": context.trigger_receipt.receipt_sha256,
        "trigger_review_status": context.trigger_receipt.human_review_status,
        "common_probe_receipt_sha256": context.common_receipt.receipt_sha256,
        "common_probe_group_count": context.common_receipt.group_count,
        "statistical_unit": "parent_config_id",
        "calibration_profile_sha256": sha256_file(context.profiles[source_language]),
    }
    if any(
        not same_json_value(distribution.get(key), value)
        for key, value in expected_distribution.items()
    ):
        raise ValueError("RQ3 score distribution binding does not match the current study")
    rows = distribution.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("RQ3 score distribution rows are invalid")
    _validated_cell_groups(
        rows,
        source_language=source_language,
        probe_language=probe_language,
        execution_profile=context.execution_profile,
    )
    return cell, rows


def _run_rq3_cell_with_context(
    context: _Rq3StudyContext,
    *,
    source_language: str,
    probe_language: str,
) -> Rq3Cell:
    from .manifests import write_json_atomic

    if source_language not in RQ3_LANGUAGES or probe_language not in RQ3_LANGUAGES:
        raise ValueError("RQ3 cell languages must use the locked EN/VI matrix")
    result_path = _rq3_cell_result_path(context, source_language, probe_language)
    if result_path.is_file():
        cell, rows = _validated_rq3_cell_result(
            context,
            source_language=source_language,
            probe_language=probe_language,
        )
        source_rows = rows
        if probe_language != source_language:
            _, source_rows = _validated_rq3_cell_result(
                context,
                source_language=source_language,
                probe_language=source_language,
            )
        _lineage_payloads(
            rows,
            source_rows,
            source_language=source_language,
            probe_language=probe_language,
            execution_profile=context.execution_profile,
        )
        return cell

    source_rows: list[dict[str, Any]] | None = None
    if probe_language != source_language:
        _, source_rows = _validated_rq3_cell_result(
            context,
            source_language=source_language,
            probe_language=source_language,
        )
    source_lineages = [
        lineage for lineage in context.test_lineages if lineage.language == source_language
    ]
    cell_root = _rq3_cell_root(context, source_language, probe_language)
    rows: list[dict[str, Any]] = []
    for lineage in source_lineages:
        trigger = context.registry.by_id(lineage.trigger_family).for_locale(probe_language)
        for label in ("clean", "backdoored"):
            rows.append(
                _evaluate_rq3_adapter(
                    config=context.config,
                    records=context.records,
                    common_probe_records=context.common_probes[probe_language],
                    lineage=lineage,
                    probe_language=probe_language,
                    label=label,
                    adapter=(
                        context.root / "adapters" / lineage.parent_config_id / label / "adapter"
                    ),
                    trigger=trigger,
                    cell_root=cell_root,
                    calibration_profile=context.profiles[source_language],
                    execution_profile=context.execution_profile,
                )
            )
    if source_rows is None:
        source_rows = rows
    _lineage_payloads(
        rows,
        source_rows,
        source_language=source_language,
        probe_language=probe_language,
        execution_profile=context.execution_profile,
    )
    distribution_path = cell_root / "score-distribution.json"
    write_json_atomic(
        distribution_path,
        {
            "schema_version": 2,
            "config_sha256": context.config.config_hash,
            "execution_profile": context.execution_profile.binding(),
            "poison_language": source_language,
            "probe_language": probe_language,
            "trigger_equivalence_receipt_sha256": (context.trigger_receipt.receipt_sha256),
            "trigger_review_status": context.trigger_receipt.human_review_status,
            "common_probe_receipt_sha256": context.common_receipt.receipt_sha256,
            "common_probe_group_count": context.common_receipt.group_count,
            "statistical_unit": "parent_config_id",
            "calibration_profile_sha256": sha256_file(context.profiles[source_language]),
            "rows": rows,
        },
    )
    cell = _rq3_cell_from_rows(
        config=context.config,
        source_language=source_language,
        probe_language=probe_language,
        rows=rows,
        source_rows=source_rows,
        calibration_profile=context.profiles[source_language],
        distribution_path=distribution_path,
        distribution_reference=distribution_path.relative_to(context.root).as_posix(),
        trigger_receipt_sha256=context.trigger_receipt.receipt_sha256,
        trigger_review_status=context.trigger_receipt.human_review_status,
        common_probe_receipt_sha256=context.common_receipt.receipt_sha256,
        common_probe_group_count=context.common_receipt.group_count,
        execution_profile=context.execution_profile,
    )
    cell_payload = cell.model_dump(mode="json")
    write_json_atomic(
        result_path,
        {
            **_rq3_cell_binding(
                context,
                source_language=source_language,
                probe_language=probe_language,
            ),
            "distribution_artifact": distribution_path.relative_to(context.root).as_posix(),
            "distribution_sha256": sha256_file(distribution_path),
            "cell": cell_payload,
            "cell_sha256": sha256_json(cell_payload),
        },
    )
    validated, validated_rows = _validated_rq3_cell_result(
        context,
        source_language=source_language,
        probe_language=probe_language,
    )
    _lineage_payloads(
        validated_rows,
        source_rows,
        source_language=source_language,
        probe_language=probe_language,
        execution_profile=context.execution_profile,
    )
    return validated


def run_rq3_cell(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    source_manifest: str | Path,
    bundle_manifest_path: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    source_language: str,
    probe_language: str,
    trigger_equivalence_receipt: str | Path | None = None,
    common_probe_receipt: str | Path | None = None,
    required_waiver_attestation_sha256: str | None = None,
) -> Rq3Cell:
    """Execute one exact receipt-bound RQ3 matrix cell, or verify and resume it."""

    context = _load_rq3_study_context(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=bundle_manifest_path,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        trigger_equivalence_receipt=trigger_equivalence_receipt,
        common_probe_receipt=common_probe_receipt,
        required_waiver_attestation_sha256=required_waiver_attestation_sha256,
    )
    return _run_rq3_cell_with_context(
        context,
        source_language=source_language,
        probe_language=probe_language,
    )


def _finalize_rq3_with_context(
    context: _Rq3StudyContext,
    *,
    output: str | Path,
) -> list[Rq3Cell]:
    from .data import write_jsonl

    cells: list[Rq3Cell] = []
    for source_language in RQ3_LANGUAGES:
        cell, rows = _validated_rq3_cell_result(
            context,
            source_language=source_language,
            probe_language=source_language,
        )
        cells.append(cell)
        for probe_language in RQ3_LANGUAGES:
            if probe_language == source_language:
                continue
            transfer_cell, transfer_rows = _validated_rq3_cell_result(
                context,
                source_language=source_language,
                probe_language=probe_language,
            )
            _lineage_payloads(
                transfer_rows,
                rows,
                source_language=source_language,
                probe_language=probe_language,
                execution_profile=context.execution_profile,
            )
            cells.append(transfer_cell)
    expected_order = [
        (cell.poison_language, cell.probe_language) for cell in locked_rq3_matrix(context.config)
    ]
    by_pair = {(cell.poison_language, cell.probe_language): cell for cell in cells}
    ordered = [by_pair[pair] for pair in expected_order]
    verify_rq3_matrix(ordered, require_results=True)
    write_jsonl(ordered, Path(output))
    return ordered


def finalize_rq3_study(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    source_manifest: str | Path,
    bundle_manifest_path: str | Path,
    output: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    trigger_equivalence_receipt: str | Path | None = None,
    common_probe_receipt: str | Path | None = None,
    required_waiver_attestation_sha256: str | None = None,
) -> list[Rq3Cell]:
    """Validate all four cell receipts and release the locked RQ3 matrix."""

    context = _load_rq3_study_context(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=bundle_manifest_path,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        trigger_equivalence_receipt=trigger_equivalence_receipt,
        common_probe_receipt=common_probe_receipt,
        required_waiver_attestation_sha256=required_waiver_attestation_sha256,
    )
    return _finalize_rq3_with_context(context, output=output)


def run_rq3_study(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    source_manifest: str | Path,
    bundle_manifest_path: str | Path,
    output: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    trigger_equivalence_receipt: str | Path | None = None,
    common_probe_receipt: str | Path | None = None,
    required_waiver_attestation_sha256: str | None = None,
) -> list[Rq3Cell]:
    """Execute the granular cells and release the receipt-bound RQ3 matrix."""

    context = _load_rq3_study_context(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=bundle_manifest_path,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        trigger_equivalence_receipt=trigger_equivalence_receipt,
        common_probe_receipt=common_probe_receipt,
        required_waiver_attestation_sha256=required_waiver_attestation_sha256,
    )
    for source_language in RQ3_LANGUAGES:
        _run_rq3_cell_with_context(
            context,
            source_language=source_language,
            probe_language=source_language,
        )
        for probe_language in RQ3_LANGUAGES:
            if probe_language != source_language:
                _run_rq3_cell_with_context(
                    context,
                    source_language=source_language,
                    probe_language=probe_language,
                )
    return _finalize_rq3_with_context(context, output=output)
