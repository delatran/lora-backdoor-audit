"""Operational adapter scan using an explicit calibration profile."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .behavior import BehaviorEvidence
from .config import ExperimentConfig
from .detectors import WEIGHT_FEATURE_NAMES
from .hashing import sha256_file, sha256_json
from .report import AuditReport, ProbeCoverage, build_audit_report
from .thresholds import SecondaryOperatingPoint, ThresholdLock
from .weight_features import extract_weight_features, load_adapter_weights

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _cross_fit_parent_ids(receipt: dict[str, Any]) -> set[str]:
    paired_groups = receipt.get("paired_groups")
    if not isinstance(paired_groups, list) or not paired_groups:
        raise ValueError("fitted calibration profile cross-fit groups are missing")
    parent_ids: list[str] = []
    adapter_ids: list[str] = []
    for group in paired_groups:
        if not isinstance(group, dict):
            raise ValueError("fitted calibration profile cross-fit group is invalid")
        parent_id = group.get("parent_config_id")
        adapters = group.get("adapters")
        if not isinstance(parent_id, str) or not parent_id or not isinstance(adapters, list):
            raise ValueError("fitted calibration profile cross-fit group is invalid")
        labels = {adapter.get("label") for adapter in adapters if isinstance(adapter, dict)}
        observed_ids = [
            adapter.get("adapter_id") for adapter in adapters if isinstance(adapter, dict)
        ]
        if (
            len(adapters) != 2
            or labels != {"clean", "backdoored"}
            or any(not isinstance(adapter_id, str) or not adapter_id for adapter_id in observed_ids)
        ):
            raise ValueError("fitted calibration profile cross-fit pair is invalid")
        parent_ids.append(parent_id)
        adapter_ids.extend(observed_ids)
    if len(parent_ids) != len(set(parent_ids)) or len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("fitted calibration profile cross-fit receipt contains duplicates")
    if receipt.get("parent_group_count") != len(parent_ids):
        raise ValueError("fitted calibration profile cross-fit group count mismatch")
    return set(parent_ids)


def _expected_cross_fit_order(receipt: dict[str, Any], parent_ids: set[str]) -> list[str]:
    expected = sorted(
        parent_ids,
        key=lambda parent_id: (
            sha256_json(
                {
                    "seed": receipt["seed"],
                    "parent_config_id": parent_id,
                }
            ),
            parent_id,
        ),
    )
    if receipt.get("ordered_parent_config_ids") != expected:
        raise ValueError("fitted calibration profile cross-fit group order mismatch")
    return expected


def _validated_cross_fit_fold(fold: object, parent_ids: set[str]) -> tuple[int, set[str], set[str]]:
    if not isinstance(fold, dict):
        raise ValueError("fitted calibration profile cross-fit fold is invalid")
    fold_index = fold.get("fold_index")
    training = fold.get("training_parent_config_ids")
    held_out = fold.get("held_out_parent_config_ids")
    if (
        type(fold_index) is not int
        or not isinstance(training, list)
        or not isinstance(held_out, list)
        or not held_out
    ):
        raise ValueError("fitted calibration profile cross-fit fold is invalid")
    if any(not isinstance(parent_id, str) or not parent_id for parent_id in [*training, *held_out]):
        raise ValueError("fitted calibration profile cross-fit fold is invalid")
    training_set = set(training)
    held_out_set = set(held_out)
    if (
        len(training) != len(training_set)
        or len(held_out) != len(held_out_set)
        or training_set & held_out_set
        or training_set | held_out_set != parent_ids
    ):
        raise ValueError("fitted calibration profile cross-fit fold leaks parent groups")
    return fold_index, training_set, held_out_set


def _validate_cross_fit_folds(receipt: dict[str, Any], parent_ids: set[str]) -> None:
    fold_count = receipt.get("fold_count")
    folds = receipt.get("folds")
    if type(fold_count) is not int or fold_count < 2 or not isinstance(folds, list):
        raise ValueError("fitted calibration profile cross-fit folds are invalid")
    expected_order = _expected_cross_fit_order(receipt, parent_ids)
    if len(folds) != fold_count:
        raise ValueError("fitted calibration profile cross-fit fold count mismatch")
    held_out_once: list[str] = []
    fold_indices: list[int] = []
    for fold in folds:
        fold_index, _, held_out_set = _validated_cross_fit_fold(fold, parent_ids)
        if held_out_set != set(expected_order[fold_index::fold_count]):
            raise ValueError("fitted calibration profile cross-fit assignment mismatch")
        fold_indices.append(fold_index)
        held_out_once.extend(held_out_set)
    if sorted(fold_indices) != list(range(fold_count)):
        raise ValueError("fitted calibration profile cross-fit fold indexes mismatch")
    if len(held_out_once) != len(set(held_out_once)) or set(held_out_once) != parent_ids:
        raise ValueError("fitted calibration profile cross-fit held-out coverage mismatch")


class CalibrationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1, 3] = 1
    profile_id: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    dataset_revision: str | None = None
    config_sha256: str | None = None
    rank: int = Field(gt=0)
    target_modules: list[str] = Field(min_length=1)
    weight_coefficients: dict[str, float]
    weight_intercept: float
    weight_threshold: float = Field(ge=0.0, le=1.0)
    behavior_threshold: float = Field(ge=0.0, le=1.0)
    behavior_score_definition: Literal["trigger_blind_target_rate"] | None = None
    fusion_coefficients: dict[str, float] | None = None
    fusion_intercept: float | None = None
    fusion_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_evidence_label: Literal["exploratory", "confirmatory"]
    pipeline_sha256: str | None = None
    cross_fit_receipt: dict[str, Any] | None = None
    cross_fit_receipt_sha256: str | None = None
    development_data_sha256: str | None = None
    validation_data_sha256: str | None = None
    threshold_locks: dict[str, dict[str, Any]] | None = None
    secondary_operating_points: dict[str, dict[str, Any]] | None = None
    validation_metrics: dict[str, Any] | None = None
    source_evidence_sha256: str | None = None
    profile_payload_sha256: str | None = None

    @field_validator("weight_intercept")
    @classmethod
    def validate_finite_intercept(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight intercept must be finite")
        return value

    def _validate_weight_contract(self) -> None:
        if not self.weight_coefficients:
            raise ValueError("calibration profile has no weight coefficients")
        unknown = set(self.weight_coefficients) - set(WEIGHT_FEATURE_NAMES)
        if unknown:
            raise ValueError(
                f"calibration profile contains unknown weight features: {sorted(unknown)}"
            )
        if not all(math.isfinite(value) for value in self.weight_coefficients.values()):
            raise ValueError("weight coefficients must be finite")
        if len(self.target_modules) != len(set(self.target_modules)):
            raise ValueError("calibration profile target modules must be unique")

    def _validate_threshold_locks(self) -> None:
        if set(self.threshold_locks or {}) != {"weight", "behavior", "fusion"}:
            raise ValueError("fitted calibration profile threshold locks are incomplete")
        thresholds = {
            "weight": self.weight_threshold,
            "behavior": self.behavior_threshold,
            "fusion": self.fusion_threshold,
        }
        for name, threshold in thresholds.items():
            try:
                lock = ThresholdLock(**(self.threshold_locks or {})[name])
            except (TypeError, ValueError) as exc:
                message = f"fitted calibration profile {name} threshold lock is invalid"
                raise ValueError(message) from exc
            if (
                lock.detector != name
                or lock.threshold != threshold
                or lock.validation_data_hash != self.validation_data_sha256
                or lock.pipeline_hash != self.pipeline_sha256
            ):
                raise ValueError(f"fitted calibration profile {name} threshold lock mismatch")

    def _validate_secondary_operating_points(self) -> None:
        if set(self.secondary_operating_points or {}) != {"weight", "behavior", "fusion"}:
            raise ValueError("fitted calibration profile secondary operating points are incomplete")
        for name, raw in (self.secondary_operating_points or {}).items():
            try:
                operating_point = SecondaryOperatingPoint(**raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fitted calibration profile {name} secondary operating point is invalid"
                ) from exc
            if (
                operating_point.detector != name
                or operating_point.validation_data_hash != self.validation_data_sha256
                or operating_point.pipeline_hash != self.pipeline_sha256
            ):
                raise ValueError(
                    f"fitted calibration profile {name} secondary operating point mismatch"
                )

    def _validate_fitted_contract(self) -> None:
        required_hashes = (
            self.config_sha256,
            self.pipeline_sha256,
            self.cross_fit_receipt_sha256,
            self.development_data_sha256,
            self.validation_data_sha256,
            self.source_evidence_sha256,
            self.profile_payload_sha256,
        )
        if not self.dataset_revision or any(
            not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
            for value in required_hashes
        ):
            raise ValueError("fitted calibration profile has incomplete hash bindings")
        if set(self.weight_coefficients) != set(WEIGHT_FEATURE_NAMES):
            raise ValueError("fitted calibration profile weight features are incomplete")
        if self.behavior_score_definition != "trigger_blind_target_rate":
            raise ValueError("fitted calibration profile behavior score definition mismatch")
        if set(self.fusion_coefficients or {}) != {"weight_score", "behavior_score"}:
            raise ValueError("fitted calibration profile fusion coefficients are incomplete")
        fusion_values = [
            *(self.fusion_coefficients or {}).values(),
            self.fusion_intercept,
            self.fusion_threshold,
        ]
        if any(
            type(value) not in {int, float} or not math.isfinite(value) for value in fusion_values
        ):
            raise ValueError("fitted calibration profile fusion values must be finite")
        receipt = self.cross_fit_receipt
        if not isinstance(receipt, dict):
            raise ValueError("fitted calibration profile cross-fit receipt is missing")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("method") != "grouped-parent-cross-fitting-v1"
            or receipt.get("assignment") != "sha256-seeded-round-robin-v1"
            or receipt.get("group_key") != "parent_config_id"
            or type(receipt.get("seed")) is not int
            or receipt.get("feature_names") != list(WEIGHT_FEATURE_NAMES)
            or self.cross_fit_receipt_sha256 != sha256_json(receipt)
        ):
            raise ValueError("fitted calibration profile cross-fit receipt mismatch")
        parent_ids = _cross_fit_parent_ids(receipt)
        _validate_cross_fit_folds(receipt, parent_ids)
        expected_pipeline_hash = sha256_json(
            {
                "detector_version": self.detector_version,
                "config_sha256": self.config_sha256,
                "feature_names": WEIGHT_FEATURE_NAMES,
                "development_data_sha256": self.development_data_sha256,
                "cross_fit_receipt_sha256": self.cross_fit_receipt_sha256,
                "fit": {
                    "kind": "logistic",
                    "regularization_c": 1.0,
                    "seed": receipt["seed"],
                },
            }
        )
        if self.pipeline_sha256 != expected_pipeline_hash:
            raise ValueError("fitted calibration profile pipeline hash mismatch")
        self._validate_threshold_locks()
        self._validate_secondary_operating_points()
        payload = self.model_dump(mode="json", exclude={"profile_payload_sha256"})
        if self.profile_payload_sha256 != sha256_json(payload):
            raise ValueError("fitted calibration profile payload hash mismatch")

    @model_validator(mode="after")
    def validate_feature_contract(self) -> CalibrationProfile:
        self._validate_weight_contract()
        if self.schema_version == 3:
            self._validate_fitted_contract()
        return self


@dataclass(frozen=True)
class ResolvedAdapter:
    weights_path: Path
    metadata_path: Path | None
    metadata: dict[str, Any] | None


def _resolve_adapter(path: str | Path) -> ResolvedAdapter:
    candidate = Path(path).resolve()
    if candidate.is_dir():
        weights_path = candidate / "adapter_model.safetensors"
        metadata_path = candidate / "adapter_config.json"
    else:
        weights_path = candidate
        metadata_path = candidate.parent / "adapter_config.json"
    if not weights_path.is_file():
        raise FileNotFoundError(f"adapter weights missing: {weights_path}")
    if weights_path.suffix not in {".safetensors", ".npz"}:
        raise ValueError(f"unsupported adapter weight format: {weights_path.suffix}")
    if not metadata_path.is_file():
        return ResolvedAdapter(weights_path=weights_path, metadata_path=None, metadata=None)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("adapter metadata must be a JSON object")
    return ResolvedAdapter(
        weights_path=weights_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def validated_training_adapter_identity(
    config: ExperimentConfig, adapter_directory: str | Path
) -> dict[str, str]:
    """Fail closed unless an adapter exactly matches the locked training contract."""

    resolved = _resolve_adapter(adapter_directory)
    if resolved.weights_path.suffix != ".safetensors" or resolved.metadata is None:
        raise ValueError("training adapter requires safe weights and configuration metadata")
    metadata = resolved.metadata
    expected = {
        "base_model_name_or_path": config.revisions.model_id,
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": config.lora.bias,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    mismatches = [
        field
        for field, value in expected.items()
        if type(metadata.get(field)) is not type(value) or metadata.get(field) != value
    ]
    target_modules = metadata.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or any(not isinstance(module, str) for module in target_modules)
        or len(target_modules) != len(config.lora.target_modules)
        or set(target_modules) != set(config.lora.target_modules)
    ):
        mismatches.append("target_modules")
    if mismatches:
        raise ValueError(f"adapter configuration mismatch: {sorted(set(mismatches))}")
    return {
        "weights_sha256": sha256_file(resolved.weights_path),
        "metadata_sha256": sha256_file(resolved.metadata_path),
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _aggregate_features(features: dict[str, dict[str, float]]) -> dict[str, float]:
    feature_names = next(iter(features.values()))
    return {
        name: sum(module[name] for module in features.values()) / len(features)
        for name in feature_names
    }


def _profile_ood_flags(
    profile: CalibrationProfile,
    config: ExperimentConfig,
    features: dict[str, dict[str, float]],
) -> list[str]:
    flags: list[str] = []
    comparisons = (
        ("model_id_mismatch", profile.model_id, config.revisions.model_id),
        ("model_revision_mismatch", profile.model_revision, config.revisions.model_revision),
        (
            "dataset_revision_mismatch",
            profile.dataset_revision,
            config.revisions.dataset_revision,
        ),
        ("config_hash_mismatch", profile.config_sha256, config.config_hash),
        ("rank_mismatch", profile.rank, config.lora.rank),
    )
    flags.extend(
        name
        for name, observed, expected in comparisons
        if observed is not None and observed != expected
    )
    if set(profile.target_modules) != set(config.lora.target_modules):
        flags.append("target_modules_mismatch")
    missing_modules = [
        module
        for module in config.lora.target_modules
        if not any(module in observed for observed in features)
    ]
    if missing_modules:
        flags.append("adapter_target_modules_incomplete")
    return flags


def _adapter_metadata_evidence(
    resolved: ResolvedAdapter, config: ExperimentConfig
) -> tuple[dict[str, Any], list[str]]:
    evidence: dict[str, Any] = {"available": resolved.metadata is not None}
    flags: list[str] = []
    if resolved.metadata_path is not None:
        evidence["sha256"] = sha256_file(resolved.metadata_path)
    if resolved.metadata is None:
        if not config.fixture_smoke_only:
            flags.append("adapter_metadata_missing")
        return evidence, flags

    metadata = resolved.metadata
    evidence["peft_type"] = metadata.get("peft_type")
    evidence["task_type"] = metadata.get("task_type")
    expected_metadata = {
        "base_model_name_or_path": config.revisions.model_id,
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    flags.extend(
        f"adapter_metadata_{field}_mismatch"
        for field, expected in expected_metadata.items()
        if metadata.get(field) != expected
    )
    observed_targets = metadata.get("target_modules")
    if not isinstance(observed_targets, list) or set(observed_targets) != set(
        config.lora.target_modules
    ):
        flags.append("adapter_metadata_target_modules_mismatch")
    return evidence, flags


def _load_bound_behavior_evidence(
    path: str | Path | None,
    *,
    resolved: ResolvedAdapter,
    config: ExperimentConfig,
) -> BehaviorEvidence | None:
    if path is None:
        return None
    behavior = BehaviorEvidence.model_validate_json(Path(path).read_text(encoding="utf-8"))
    expected_binding = {
        "adapter_sha256": sha256_file(resolved.weights_path),
        "config_sha256": config.config_hash,
        "model_revision": config.revisions.model_revision,
        "dataset_revision": config.revisions.dataset_revision,
        "fixture_smoke_only": config.fixture_smoke_only,
    }
    mismatches = [
        field
        for field, expected in expected_binding.items()
        if getattr(behavior, field) != expected
    ]
    if mismatches:
        raise ValueError(f"behavior evidence binding mismatch: {mismatches}")
    return behavior


def _fused_score(
    profile: CalibrationProfile, weight: float, behavior: float | None
) -> float | None:
    if behavior is None or profile.fusion_coefficients is None or profile.fusion_intercept is None:
        return None
    raw = (
        profile.fusion_coefficients["weight_score"] * weight
        + profile.fusion_coefficients["behavior_score"] * behavior
        + float(profile.fusion_intercept)
    )
    return _sigmoid(raw)


def scan_adapter(
    *,
    adapter_path: str | Path,
    config: ExperimentConfig,
    calibration_profile_path: str | Path,
    execution_profile: Literal["fast_triage", "full_audit"],
    behavior_evidence_path: str | Path | None,
) -> AuditReport:
    started = time.perf_counter()
    profile_path = Path(calibration_profile_path).resolve()
    profile = CalibrationProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    resolved = _resolve_adapter(adapter_path)
    if not config.fixture_smoke_only and resolved.weights_path.suffix != ".safetensors":
        raise ValueError("main scan requires safe serialized adapter weights")
    weights = load_adapter_weights(resolved.weights_path)
    features = extract_weight_features(weights, alpha=config.lora.alpha, rank=config.lora.rank)
    aggregate = _aggregate_features(features)
    raw = (
        sum(
            aggregate[name] * coefficient
            for name, coefficient in profile.weight_coefficients.items()
        )
        + profile.weight_intercept
    )
    weight_score = _sigmoid(raw)
    ood_flags = _profile_ood_flags(profile, config, features)
    metadata_evidence, metadata_flags = _adapter_metadata_evidence(resolved, config)
    ood_flags.extend(metadata_flags)

    should_run_behavior = (
        execution_profile == "full_audit"
        or weight_score >= profile.weight_threshold
        or bool(ood_flags)
        or abs(weight_score - profile.weight_threshold) <= 0.05
    )
    behavior = _load_bound_behavior_evidence(
        behavior_evidence_path,
        resolved=resolved,
        config=config,
    )
    if behavior is None and should_run_behavior:
        ood_flags.append("behavior_scan_required_but_unavailable")
    coverage = ProbeCoverage(
        trigger_blind=behavior.trigger_blind_coverage if behavior else 0.0,
        structure_matched=behavior.structure_matched_coverage if behavior else 0.0,
        trigger_aware=behavior.trigger_aware_coverage if behavior else 0.0,
    )
    behavior_payload = behavior.model_dump(mode="json") if behavior else {}
    fused_score = _fused_score(profile, weight_score, behavior.score if behavior else None)
    return build_audit_report(
        detector_version=profile.detector_version,
        execution_profile=execution_profile,
        calibration_profile=profile.profile_id,
        fusion_evidence_label=profile.fusion_evidence_label,
        fixture_smoke_only=config.fixture_smoke_only,
        weight_score=weight_score,
        weight_threshold=profile.weight_threshold,
        weight_evidence={
            "aggregate": aggregate,
            "per_module": features,
            "adapter_metadata": metadata_evidence,
            "calibration_profile_sha256": sha256_file(profile_path),
        },
        behavior_score=behavior.score if behavior else None,
        behavior_threshold=profile.behavior_threshold,
        behavior_evidence=behavior_payload,
        behavior_required=should_run_behavior,
        fused_score=fused_score,
        uncertainty={
            "calibrated": profile.schema_version == 3,
            "behavior_required": should_run_behavior,
            "reason": (
                "validation_locked_profile"
                if profile.schema_version == 3
                else "fixture_or_unfitted_profile"
            ),
        },
        coverage=coverage,
        adapter_sha256=sha256_file(resolved.weights_path),
        config_sha256=config.config_hash,
        model_revision=config.revisions.model_revision,
        dataset_revision=config.revisions.dataset_revision,
        runtime_seconds=time.perf_counter() - started,
        ood_flags=ood_flags,
        limitations=[
            "A required behavior branch that is missing or weakly covered forces review.",
            "No safety certification is issued.",
        ],
    )
