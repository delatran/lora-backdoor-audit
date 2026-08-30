"""Weight/behavior score composition, OOF fusion, and count gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .hashing import is_sha256, sha256_json
from .metrics import (
    detector_metrics,
    parent_lineage_cluster_bootstrap,
)
from .thresholds import (
    SecondaryOperatingPoint,
    ThresholdLock,
    release_test_scores,
    select_secondary_validation_operating_point,
    select_validation_threshold,
)

WEIGHT_FEATURE_NAMES = (
    "spectral_entropy",
    "top_singular_ratio",
    "stable_rank",
    "frobenius_norm_normalized",
    "excess_kurtosis",
)
SHA256_PATTERN = r"^[0-9a-f]{64}$"
DETECTOR_VERSION = "weight-behavior-crossfit-v2"


def _valid_execution_profile_binding(value: dict[str, Any]) -> bool:
    required_fields = {
        "schema_version",
        "profile_id",
        "backend",
        "required_compute_capability",
        "minimum_device_memory_gib",
        "maximum_device_memory_gib",
        "allowed_device_name_patterns",
        "require_bf16_probe",
        "strict_accelerator",
        "profile_hash",
        "profile_file_sha256",
    }
    capability = value.get("required_compute_capability")
    patterns = value.get("allowed_device_name_patterns")
    minimum_memory = value.get("minimum_device_memory_gib")
    maximum_memory = value.get("maximum_device_memory_gib")
    return (
        set(value) == required_fields
        and value.get("schema_version") == 1
        and isinstance(value.get("profile_id"), str)
        and bool(value["profile_id"])
        and value["profile_id"] == value["profile_id"].strip()
        and value.get("backend") == "cuda"
        and isinstance(capability, list)
        and len(capability) == 2
        and all(type(component) is int and component >= 0 for component in capability)
        and type(minimum_memory) in {int, float}
        and type(maximum_memory) in {int, float}
        and math.isfinite(float(minimum_memory))
        and math.isfinite(float(maximum_memory))
        and 0.0 < float(minimum_memory) <= float(maximum_memory)
        and isinstance(patterns, list)
        and bool(patterns)
        and all(
            isinstance(pattern, str) and bool(pattern) and pattern == pattern.strip()
            for pattern in patterns
        )
        and len(patterns) == len(set(patterns))
        and value.get("require_bf16_probe") is True
        and value.get("strict_accelerator") is True
        and is_sha256(value.get("profile_hash"))
        and is_sha256(value.get("profile_file_sha256"))
    )


class DetectorEvidenceRow(BaseModel):
    """One adapter-level detector observation; probe rows are never samples."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    adapter_id: str = Field(min_length=1)
    parent_config_id: str = Field(min_length=1)
    label: Literal["clean", "backdoored"]
    split: Literal["train", "validation", "test"]
    language: Literal["en-US", "vi-VN"]
    trigger_family: str
    poison_rate: float = Field(gt=0.0, le=1.0)
    replicate_seed: int = Field(ge=0)
    pair_training_seed: int = Field(gt=0)
    poison_selection_seed: int = Field(gt=0)
    probe_seed: int = Field(ge=0)
    split_assignment_id: str = Field(min_length=1)
    training_unit_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_profile: dict[str, Any]
    adapter_sha256: str = Field(pattern=SHA256_PATTERN)
    qc_status: Literal["valid", "failed"]
    weight_features: dict[str, dict[str, Any]]
    behavior_score: float = Field(ge=0.0, le=1.0)
    behavior_profiles: dict[str, dict[str, float | int]]
    behavior_evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_detector_features(self) -> DetectorEvidenceRow:
        aggregate = self.weight_features.get("aggregate")
        per_module = self.weight_features.get("per_module")
        if not isinstance(aggregate, dict) or not isinstance(per_module, dict) or not per_module:
            raise ValueError("detector evidence requires aggregate and per-module weight features")
        if set(aggregate) != set(WEIGHT_FEATURE_NAMES):
            raise ValueError("detector aggregate feature vocabulary mismatch")
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in aggregate.values()
        ):
            raise ValueError("detector aggregate features must be finite")
        if set(self.behavior_profiles) != {
            "trigger_blind",
            "structure_matched",
            "trigger_aware",
        }:
            raise ValueError("detector evidence behavior profiles are incomplete")
        if not _valid_execution_profile_binding(self.execution_profile):
            raise ValueError("detector evidence execution profile binding is invalid")
        return self


def simple_or(
    weight_scores: list[float],
    behavior_scores: list[float],
    *,
    weight_threshold: float,
    behavior_threshold: float,
) -> list[int]:
    if len(weight_scores) != len(behavior_scores):
        raise ValueError("branch score lengths differ")
    values = [*weight_scores, *behavior_scores, weight_threshold, behavior_threshold]
    if not values or not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("branch scores and thresholds must be finite probabilities")
    return [
        int(weight >= weight_threshold or behavior >= behavior_threshold)
        for weight, behavior in zip(weight_scores, behavior_scores, strict=True)
    ]


def fusion_evidence_label(
    *,
    development_clean: int,
    development_backdoored: int,
    validation_clean: int,
    validation_backdoored: int,
    minimum_development_per_class: int = 30,
    minimum_validation_per_class: int = 10,
    confirmatory_gate_enabled: bool = True,
) -> str:
    if (
        confirmatory_gate_enabled
        and min(development_clean, development_backdoored) >= minimum_development_per_class
        and min(validation_clean, validation_backdoored) >= minimum_validation_per_class
    ):
        return "confirmatory"
    return "exploratory"


@dataclass(frozen=True)
class FusionFit:
    model: LogisticRegression
    oof_scores: np.ndarray
    evidence_label: str


@dataclass(frozen=True)
class WeightCrossFit:
    """Cross-fitted development scores plus the all-development deploy model."""

    oof_scores: np.ndarray
    coefficients: np.ndarray
    intercept: float
    fold_receipt: dict[str, Any]
    fold_receipt_sha256: str


def fit_oof_fusion(
    *,
    weight_scores: list[float],
    behavior_scores: list[float],
    labels: list[int],
    parent_config_ids: list[str],
    validation_clean: int,
    validation_backdoored: int,
    folds: int = 5,
    seed: int = 20260714,
    minimum_development_per_class: int = 30,
    minimum_validation_per_class: int = 10,
    confirmatory_gate_enabled: bool = True,
) -> FusionFit:
    if folds < 2:
        raise ValueError("OOF fusion requires at least two folds")
    features = np.column_stack([weight_scores, behavior_scores])
    y = np.asarray(labels, dtype=int)
    if features.shape[0] != y.size or y.size != len(parent_config_ids):
        raise ValueError("fusion inputs must be aligned")
    if y.size == 0 or set(y.tolist()) != {0, 1}:
        raise ValueError("fusion requires both clean and backdoored classes")
    if not np.isfinite(features).all() or np.any((features < 0.0) | (features > 1.0)):
        raise ValueError("fusion branch scores must be finite probabilities")
    grouped_labels: dict[str, list[int]] = {}
    for parent_id, label in zip(parent_config_ids, labels, strict=True):
        if not parent_id:
            raise ValueError("fusion parent identifiers must be nonempty")
        grouped_labels.setdefault(parent_id, []).append(label)
    invalid_groups = [
        parent_id
        for parent_id, group_labels in grouped_labels.items()
        if len(group_labels) != 2 or set(group_labels) != {0, 1}
    ]
    if invalid_groups:
        raise ValueError(f"fusion requires paired parent groups: {sorted(invalid_groups)}")
    unique_groups = len(set(parent_config_ids))
    if unique_groups < folds:
        raise ValueError("not enough independent parent groups for OOF fusion")
    estimator = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = cross_val_predict(
        estimator,
        features,
        y,
        groups=np.asarray(parent_config_ids),
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    estimator.fit(features, y)
    evidence_label = fusion_evidence_label(
        development_clean=int(np.sum(y == 0)),
        development_backdoored=int(np.sum(y == 1)),
        validation_clean=validation_clean,
        validation_backdoored=validation_backdoored,
        minimum_development_per_class=minimum_development_per_class,
        minimum_validation_per_class=minimum_validation_per_class,
        confirmatory_gate_enabled=confirmatory_gate_enabled,
    )
    return FusionFit(model=estimator, oof_scores=oof, evidence_label=evidence_label)


def read_detector_evidence(path: str | Path) -> list[DetectorEvidenceRow]:
    rows = [
        DetectorEvidenceRow.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("detector evidence ledger is empty")
    adapter_ids = [row.adapter_id for row in rows]
    if len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("detector evidence contains duplicate adapter IDs")
    config_hashes = {row.config_sha256 for row in rows}
    if len(config_hashes) != 1:
        raise ValueError("detector evidence mixes configuration hashes")
    by_parent: dict[str, list[DetectorEvidenceRow]] = {}
    for row in rows:
        by_parent.setdefault(row.parent_config_id, []).append(row)
    invalid_pairs = [
        parent_id
        for parent_id, pair in by_parent.items()
        if len(pair) != 2
        or {item.label for item in pair} != {"clean", "backdoored"}
        or len({item.split for item in pair}) != 1
        or len({item.replicate_seed for item in pair}) != 1
        or len({item.pair_training_seed for item in pair}) != 1
        or len({item.poison_selection_seed for item in pair}) != 1
        or len({item.probe_seed for item in pair}) != 1
        or len({item.split_assignment_id for item in pair}) != 1
        or len({sha256_json(item.execution_profile) for item in pair}) != 1
        or len({item.training_unit_id for item in pair}) != 2
    ]
    if invalid_pairs:
        raise ValueError(f"detector evidence contains invalid paired groups: {invalid_pairs}")
    return rows


def _complete_valid_pairs(rows: list[DetectorEvidenceRow]) -> list[DetectorEvidenceRow]:
    """Retain a lineage only when both paired adapters passed QC."""

    by_parent: dict[str, list[DetectorEvidenceRow]] = {}
    for row in rows:
        by_parent.setdefault(row.parent_config_id, []).append(row)
    return [
        row
        for parent_id in sorted(by_parent)
        for row in by_parent[parent_id]
        if all(item.qc_status == "valid" for item in by_parent[parent_id])
    ]


def _feature_matrix(rows: list[DetectorEvidenceRow]) -> np.ndarray:
    return np.asarray(
        [
            [float(row.weight_features["aggregate"][name]) for name in WEIGHT_FEATURE_NAMES]
            for row in rows
        ],
        dtype=float,
    )


def _labels(rows: list[DetectorEvidenceRow]) -> list[int]:
    return [int(row.label == "backdoored") for row in rows]


def _probabilities(
    matrix: np.ndarray, *, coefficients: np.ndarray, intercept: float
) -> list[float]:
    logits = matrix @ coefficients + intercept
    output: list[float] = []
    for raw in logits:
        value = float(raw)
        if value >= 0.0:
            output.append(float(1.0 / (1.0 + math.exp(-value))))
        else:
            exponent = math.exp(value)
            output.append(float(exponent / (1.0 + exponent)))
    return output


def _paired_group_indices(
    labels: list[int],
    parent_config_ids: list[str],
    *,
    context: str,
) -> dict[str, list[int]]:
    if len(labels) != len(parent_config_ids):
        raise ValueError(f"{context} inputs must be aligned")
    groups: dict[str, list[int]] = {}
    for index, (parent_id, label) in enumerate(zip(parent_config_ids, labels, strict=True)):
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(f"{context} parent identifiers must be nonempty")
        if type(label) is not int or label not in {0, 1}:
            raise ValueError(f"{context} labels must be exact binary integers")
        groups.setdefault(parent_id, []).append(index)
    invalid = [
        parent_id
        for parent_id, indices in groups.items()
        if len(indices) != 2 or {labels[index] for index in indices} != {0, 1}
    ]
    if invalid:
        raise ValueError(f"{context} requires paired parent groups: {sorted(invalid)}")
    return groups


def _fit_weight_parameters(
    matrix: np.ndarray, labels: list[int], *, seed: int
) -> tuple[np.ndarray, float]:
    scaler = StandardScaler().fit(matrix)
    estimator = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    estimator.fit(scaler.transform(matrix), labels)
    scaled_coefficients = estimator.coef_[0]
    coefficients = scaled_coefficients / scaler.scale_
    intercept = float(
        estimator.intercept_[0] - np.sum(scaled_coefficients * scaler.mean_ / scaler.scale_)
    )
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        raise ValueError("weight fit produced non-finite parameters")
    return coefficients, intercept


def _validated_crossfit_matrix(
    matrix: np.ndarray,
    labels: list[int],
    adapter_ids: list[str],
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != len(labels)
        or len(labels) != len(adapter_ids)
        or values.shape[1] != len(WEIGHT_FEATURE_NAMES)
    ):
        raise ValueError("weight cross-fitting inputs must be aligned")
    if not np.isfinite(values).all():
        raise ValueError("weight cross-fitting features must be finite")
    if any(not isinstance(adapter_id, str) or not adapter_id for adapter_id in adapter_ids):
        raise ValueError("weight cross-fitting adapter identifiers must be nonempty")
    if len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("weight cross-fitting adapter identifiers must be unique")
    return values


def fit_grouped_weight_crossfit(
    *,
    matrix: np.ndarray,
    labels: list[int],
    parent_config_ids: list[str],
    adapter_ids: list[str],
    folds: int = 5,
    seed: int = 20260714,
) -> WeightCrossFit:
    """Produce base scores from models that never see the held-out parent pair."""

    if type(folds) is not int or folds < 2:
        raise ValueError("weight cross-fitting requires at least two folds")
    if type(seed) is not int:
        raise ValueError("weight cross-fitting seed must be an integer")
    values = _validated_crossfit_matrix(matrix, labels, adapter_ids)
    groups = _paired_group_indices(labels, parent_config_ids, context="weight cross-fitting")
    if len(groups) < folds:
        raise ValueError("not enough independent parent groups for weight cross-fitting")

    ordered_groups = sorted(
        groups,
        key=lambda parent_id: (
            sha256_json({"seed": seed, "parent_config_id": parent_id}),
            parent_id,
        ),
    )
    held_out_by_fold = [ordered_groups[index::folds] for index in range(folds)]
    oof_scores = np.full(len(labels), np.nan, dtype=float)
    fold_entries: list[dict[str, Any]] = []
    all_group_ids = set(groups)
    for fold_index, held_out_group_ids in enumerate(held_out_by_fold):
        held_out = set(held_out_group_ids)
        training = all_group_ids - held_out
        training_indices = [index for parent_id in sorted(training) for index in groups[parent_id]]
        held_out_indices = [index for parent_id in sorted(held_out) for index in groups[parent_id]]
        coefficients, intercept = _fit_weight_parameters(
            values[training_indices],
            [labels[index] for index in training_indices],
            seed=seed + fold_index,
        )
        oof_scores[held_out_indices] = _probabilities(
            values[held_out_indices],
            coefficients=coefficients,
            intercept=intercept,
        )
        fold_entries.append(
            {
                "fold_index": fold_index,
                "training_parent_config_ids": sorted(training),
                "held_out_parent_config_ids": sorted(held_out),
            }
        )
    if not np.isfinite(oof_scores).all() or np.any((oof_scores < 0.0) | (oof_scores > 1.0)):
        raise ValueError("weight cross-fitting did not produce complete finite OOF scores")

    final_coefficients, final_intercept = _fit_weight_parameters(
        values,
        labels,
        seed=seed,
    )
    paired_groups = []
    for parent_id in sorted(groups):
        indices = sorted(groups[parent_id], key=lambda index: labels[index])
        paired_groups.append(
            {
                "parent_config_id": parent_id,
                "adapters": [
                    {
                        "adapter_id": adapter_ids[index],
                        "label": "backdoored" if labels[index] else "clean",
                    }
                    for index in indices
                ],
            }
        )
    receipt = {
        "schema_version": 1,
        "method": "grouped-parent-cross-fitting-v1",
        "assignment": "sha256-seeded-round-robin-v1",
        "group_key": "parent_config_id",
        "seed": seed,
        "fold_count": folds,
        "feature_names": list(WEIGHT_FEATURE_NAMES),
        "parent_group_count": len(groups),
        "ordered_parent_config_ids": ordered_groups,
        "paired_groups": paired_groups,
        "folds": fold_entries,
    }
    receipt_hash = sha256_json(receipt)
    return WeightCrossFit(
        oof_scores=oof_scores,
        coefficients=final_coefficients,
        intercept=final_intercept,
        fold_receipt=receipt,
        fold_receipt_sha256=receipt_hash,
    )


def _evidence_hash(rows: list[DetectorEvidenceRow]) -> str:
    return sha256_json(
        [row.model_dump(mode="json") for row in sorted(rows, key=lambda item: item.adapter_id)]
    )


def fit_detector_calibration(
    *, evidence_path: str | Path, config: ExperimentConfig
) -> dict[str, Any]:
    """Fit development-only branches and lock all operating points on validation."""

    rows = read_detector_evidence(evidence_path)
    if {row.config_sha256 for row in rows} != {config.config_hash}:
        raise ValueError("detector evidence config hash mismatch")
    valid = _complete_valid_pairs(rows)
    development = [row for row in valid if row.split == "train"]
    validation = [row for row in valid if row.split == "validation"]
    if not development or not validation:
        raise ValueError("detector calibration requires valid development and validation pairs")
    if set(_labels(development)) != {0, 1} or set(_labels(validation)) != {0, 1}:
        raise ValueError("detector calibration requires both classes in each calibration split")

    development_matrix = _feature_matrix(development)
    validation_matrix = _feature_matrix(validation)
    development_labels = _labels(development)
    validation_labels = _labels(validation)
    development_groups = [row.parent_config_id for row in development]
    cross_fit = fit_grouped_weight_crossfit(
        matrix=development_matrix,
        labels=development_labels,
        parent_config_ids=development_groups,
        adapter_ids=[row.adapter_id for row in development],
        folds=min(5, len(set(development_groups))),
        seed=config.data.seed,
    )
    coefficients = cross_fit.coefficients
    intercept = cross_fit.intercept
    development_weight_scores = cross_fit.oof_scores.tolist()
    validation_weight_scores = _probabilities(
        validation_matrix,
        coefficients=coefficients,
        intercept=intercept,
    )
    development_behavior_scores = [row.behavior_score for row in development]
    validation_behavior_scores = [row.behavior_score for row in validation]
    development_hash = _evidence_hash(development)
    validation_hash = _evidence_hash(validation)
    pipeline_hash = sha256_json(
        {
            "detector_version": DETECTOR_VERSION,
            "config_sha256": config.config_hash,
            "feature_names": WEIGHT_FEATURE_NAMES,
            "development_data_sha256": development_hash,
            "cross_fit_receipt_sha256": cross_fit.fold_receipt_sha256,
            "fit": {
                "kind": "logistic",
                "regularization_c": 1.0,
                "seed": config.data.seed,
            },
        }
    )
    weight_lock = select_validation_threshold(
        detector="weight",
        y_true=validation_labels,
        scores=validation_weight_scores,
        validation_data_hash=validation_hash,
        pipeline_hash=pipeline_hash,
    )
    behavior_lock = select_validation_threshold(
        detector="behavior",
        y_true=validation_labels,
        scores=validation_behavior_scores,
        validation_data_hash=validation_hash,
        pipeline_hash=pipeline_hash,
    )
    validation_clean = sum(label == 0 for label in validation_labels)
    validation_backdoored = sum(label == 1 for label in validation_labels)
    fusion_gate = config.fusion_gate
    fusion = fit_oof_fusion(
        weight_scores=development_weight_scores,
        behavior_scores=development_behavior_scores,
        labels=development_labels,
        parent_config_ids=development_groups,
        validation_clean=validation_clean,
        validation_backdoored=validation_backdoored,
        folds=min(5, len(set(development_groups))),
        seed=config.data.seed,
        minimum_development_per_class=(
            fusion_gate.minimum_development_per_class if fusion_gate else 30
        ),
        minimum_validation_per_class=(
            fusion_gate.minimum_validation_per_class if fusion_gate else 10
        ),
        confirmatory_gate_enabled=fusion_gate is not None,
    )
    validation_fusion_scores = fusion.model.predict_proba(
        np.column_stack([validation_weight_scores, validation_behavior_scores])
    )[:, 1].tolist()
    fusion_lock = select_validation_threshold(
        detector="fusion",
        y_true=validation_labels,
        scores=validation_fusion_scores,
        validation_data_hash=validation_hash,
        pipeline_hash=pipeline_hash,
    )
    secondary_operating_points = {
        name: select_secondary_validation_operating_point(
            detector=name,
            y_true=validation_labels,
            scores=scores,
            validation_data_hash=validation_hash,
            pipeline_hash=pipeline_hash,
        )
        for name, scores in {
            "weight": validation_weight_scores,
            "behavior": validation_behavior_scores,
            "fusion": validation_fusion_scores,
        }.items()
    }
    profile = {
        "schema_version": 3,
        "profile_id": f"{config.profile}-{pipeline_hash[:12]}",
        "detector_version": DETECTOR_VERSION,
        "model_id": config.revisions.model_id,
        "model_revision": config.revisions.model_revision,
        "dataset_revision": config.revisions.dataset_revision,
        "config_sha256": config.config_hash,
        "rank": config.lora.rank,
        "target_modules": config.lora.target_modules,
        "weight_coefficients": dict(zip(WEIGHT_FEATURE_NAMES, coefficients.tolist(), strict=True)),
        "weight_intercept": intercept,
        "weight_threshold": weight_lock.threshold,
        "behavior_threshold": behavior_lock.threshold,
        "behavior_score_definition": "trigger_blind_target_rate",
        "fusion_coefficients": {
            "weight_score": float(fusion.model.coef_[0, 0]),
            "behavior_score": float(fusion.model.coef_[0, 1]),
        },
        "fusion_intercept": float(fusion.model.intercept_[0]),
        "fusion_threshold": fusion_lock.threshold,
        "fusion_evidence_label": fusion.evidence_label,
        "pipeline_sha256": pipeline_hash,
        "cross_fit_receipt": cross_fit.fold_receipt,
        "cross_fit_receipt_sha256": cross_fit.fold_receipt_sha256,
        "development_data_sha256": development_hash,
        "validation_data_sha256": validation_hash,
        "threshold_locks": {
            "weight": asdict(weight_lock),
            "behavior": asdict(behavior_lock),
            "fusion": asdict(fusion_lock),
        },
        "secondary_operating_points": {
            name: asdict(operating_point)
            for name, operating_point in secondary_operating_points.items()
        },
        "validation_metrics": {
            "weight": detector_metrics(
                validation_labels, validation_weight_scores, weight_lock.threshold
            ),
            "behavior": detector_metrics(
                validation_labels, validation_behavior_scores, behavior_lock.threshold
            ),
            "fusion": detector_metrics(
                validation_labels, validation_fusion_scores, fusion_lock.threshold
            ),
            "simple_or_positive_count": sum(
                simple_or(
                    validation_weight_scores,
                    validation_behavior_scores,
                    weight_threshold=weight_lock.threshold,
                    behavior_threshold=behavior_lock.threshold,
                )
            ),
        },
        "source_evidence_sha256": _evidence_hash([*development, *validation]),
    }
    profile["profile_payload_sha256"] = sha256_json(profile)
    return profile


def _verify_calibration_profile_integrity(
    calibration_profile: dict[str, Any],
    evidence_path: str | Path,
    *,
    config: ExperimentConfig,
) -> list[DetectorEvidenceRow]:
    recorded_hash = calibration_profile.get("profile_payload_sha256")
    payload = {
        key: value for key, value in calibration_profile.items() if key != "profile_payload_sha256"
    }
    if recorded_hash != sha256_json(payload):
        raise ValueError("test release calibration profile payload hash mismatch")
    receipt = calibration_profile.get("cross_fit_receipt")
    receipt_hash = calibration_profile.get("cross_fit_receipt_sha256")
    if not isinstance(receipt, dict) or receipt_hash != sha256_json(receipt):
        raise ValueError("test release cross-fit receipt hash mismatch")
    rows = read_detector_evidence(evidence_path)
    valid = _complete_valid_pairs(rows)
    calibration_rows = [row for row in valid if row.split in {"train", "validation"}]
    if calibration_profile.get("source_evidence_sha256") != _evidence_hash(calibration_rows):
        raise ValueError("test release source evidence hash mismatch")
    development = [row for row in calibration_rows if row.split == "train"]
    expected_pipeline_hash = sha256_json(
        {
            "detector_version": DETECTOR_VERSION,
            "config_sha256": config.config_hash,
            "feature_names": WEIGHT_FEATURE_NAMES,
            "development_data_sha256": _evidence_hash(development),
            "cross_fit_receipt_sha256": receipt_hash,
            "fit": {
                "kind": "logistic",
                "regularization_c": 1.0,
                "seed": config.data.seed,
            },
        }
    )
    if calibration_profile.get("pipeline_sha256") != expected_pipeline_hash:
        raise ValueError("test release calibration pipeline hash mismatch")
    return rows


CONTINUOUS_METRIC_NAMES = ("auroc", "pr_auc", "mcc", "brier", "tpr", "fpr")
BINARY_METRIC_NAMES = ("mcc", "tpr", "fpr")


def _release_lineages(
    *,
    rows: list[DetectorEvidenceRow],
    labels: list[int],
    scores: dict[str, list[float]],
    predictions: dict[str, list[int]],
) -> list[dict[str, Any]]:
    indices_by_parent: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        indices_by_parent.setdefault(row.parent_config_id, []).append(index)
    lineages: list[dict[str, Any]] = []
    for parent_id in sorted(indices_by_parent):
        indices = sorted(indices_by_parent[parent_id], key=lambda index: labels[index])
        if len(indices) != 2 or [labels[index] for index in indices] != [0, 1]:
            raise ValueError("test release bootstrap requires complete paired lineages")
        lineages.append(
            {
                "parent_config_id": parent_id,
                "labels": [labels[index] for index in indices],
                "adapter_ids": [rows[index].adapter_id for index in indices],
                "adapter_sha256s": [rows[index].adapter_sha256 for index in indices],
                "scores": {
                    name: [branch_scores[index] for index in indices]
                    for name, branch_scores in scores.items()
                },
                "predictions": {
                    name: [branch_predictions[index] for index in indices]
                    for name, branch_predictions in predictions.items()
                },
            }
        )
    return lineages


def _flatten_release_sample(
    sample: list[dict[str, Any]],
) -> tuple[list[int], dict[str, list[float]], dict[str, list[int]]]:
    labels = [int(label) for lineage in sample for label in lineage["labels"]]
    scores = {
        name: [float(score) for lineage in sample for score in lineage["scores"][name]]
        for name in ("weight", "behavior", "fusion")
    }
    predictions = {
        name: [int(prediction) for lineage in sample for prediction in lineage["predictions"][name]]
        for name in ("weight", "behavior", "fusion", "simple_or")
    }
    return labels, scores, predictions


def _fast_binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    pairs = list(zip(labels, predictions, strict=True))
    tp = sum(label == 1 and prediction == 1 for label, prediction in pairs)
    tn = sum(label == 0 and prediction == 0 for label, prediction in pairs)
    fp = sum(label == 0 and prediction == 1 for label, prediction in pairs)
    fn = sum(label == 1 and prediction == 0 for label, prediction in pairs)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "mcc": 0.0 if denominator == 0.0 else (tp * tn - fp * fn) / denominator,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
    }


def _fast_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores)),
        len(ordered_scores) - 1,
    ]
    true_positives = np.cumsum(ordered_labels)[threshold_ends]
    predicted_positives = threshold_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / np.sum(labels)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _fast_continuous_metrics(
    labels: list[int],
    scores: list[float],
    predictions: list[int],
) -> dict[str, float]:
    label_values = np.asarray(labels, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    positive_scores = score_values[label_values == 1]
    negative_scores = score_values[label_values == 0]
    pairwise = positive_scores[:, None] - negative_scores[None, :]
    binary = _fast_binary_metrics(labels, predictions)
    return {
        "auroc": float(np.mean((pairwise > 0.0) + 0.5 * (pairwise == 0.0))),
        "pr_auc": _fast_average_precision(label_values, score_values),
        "mcc": binary["mcc"],
        "brier": float(np.mean((score_values - label_values) ** 2)),
        "tpr": binary["tpr"],
        "fpr": binary["fpr"],
    }


def _release_estimands(sample: list[dict[str, Any]]) -> dict[str, float]:
    labels, scores, predictions = _flatten_release_sample(sample)
    continuous = {
        name: _fast_continuous_metrics(
            labels,
            scores[name],
            predictions[name],
        )
        for name in ("weight", "behavior", "fusion")
    }
    simple = _fast_binary_metrics(labels, predictions["simple_or"])
    output = {
        f"{name}.{metric}": continuous[name][metric]
        for name in continuous
        for metric in CONTINUOUS_METRIC_NAMES
    }
    output.update({f"simple_or.{metric}": simple[metric] for metric in BINARY_METRIC_NAMES})
    for baseline in ("weight", "behavior"):
        output.update(
            {
                f"fusion_minus_{baseline}.{metric}": (
                    continuous["fusion"][metric] - continuous[baseline][metric]
                )
                for metric in CONTINUOUS_METRIC_NAMES
            }
        )
    output.update(
        {
            f"fusion_minus_simple_or.{metric}": (continuous["fusion"][metric] - simple[metric])
            for metric in BINARY_METRIC_NAMES
        }
    )
    return output


def _metric_release_payload(
    estimates: dict[str, dict[str, float]],
    *,
    prefix: str,
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        **{metric: estimates[f"{prefix}.{metric}"]["estimate"] for metric in metric_names},
        "bootstrap_ci_95": {
            metric: [
                estimates[f"{prefix}.{metric}"]["ci_lower"],
                estimates[f"{prefix}.{metric}"]["ci_upper"],
            ]
            for metric in metric_names
        },
    }


def _load_bound_threshold_locks(
    calibration_profile: dict[str, Any],
    *,
    pipeline_hash: str,
    validation_hash: str,
) -> dict[str, ThresholdLock]:
    locks = {
        name: ThresholdLock(**calibration_profile["threshold_locks"][name])
        for name in ("weight", "behavior", "fusion")
    }
    for name, lock in locks.items():
        if lock.detector != name:
            raise ValueError("test release threshold lock detector binding mismatch")
        if lock.validation_data_hash != validation_hash:
            raise ValueError("test release threshold lock validation binding mismatch")
        if lock.pipeline_hash != pipeline_hash:
            raise ValueError("test release threshold lock pipeline binding mismatch")
        if float(calibration_profile[f"{name}_threshold"]) != lock.threshold:
            raise ValueError("test release threshold lock value mismatch")
    return locks


def _load_bound_secondary_operating_points(
    calibration_profile: dict[str, Any],
    *,
    pipeline_hash: str,
    validation_hash: str,
) -> dict[str, SecondaryOperatingPoint]:
    operating_points = {
        name: SecondaryOperatingPoint(**calibration_profile["secondary_operating_points"][name])
        for name in ("weight", "behavior", "fusion")
    }
    for name, operating_point in operating_points.items():
        if operating_point.detector != name:
            raise ValueError("test release secondary operating-point detector binding mismatch")
        if operating_point.validation_data_hash != validation_hash:
            raise ValueError("test release secondary operating-point validation binding mismatch")
        if operating_point.pipeline_hash != pipeline_hash:
            raise ValueError("test release secondary operating-point pipeline binding mismatch")
    return operating_points


def evaluate_detector_release(
    *,
    evidence_path: str | Path,
    calibration_profile: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Open test scores only through the immutable validation-locked profile."""

    if calibration_profile.get("schema_version") != 3:
        raise ValueError("test release requires a fitted calibration profile")
    if calibration_profile.get("config_sha256") != config.config_hash:
        raise ValueError("test release calibration config hash mismatch")
    rows = _verify_calibration_profile_integrity(
        calibration_profile,
        evidence_path,
        config=config,
    )
    completed_test_rows = [row for row in rows if row.split == "test"]
    adapter_hashes = [row.adapter_sha256 for row in completed_test_rows]
    if len(adapter_hashes) != len(set(adapter_hashes)):
        raise ValueError("test release contains duplicate adapter hashes")
    test_rows = [row for row in _complete_valid_pairs(rows) if row.split == "test"]
    if not test_rows or set(_labels(test_rows)) != {0, 1}:
        raise ValueError("test release requires valid held-out pairs from both classes")
    coefficients = np.asarray(
        [calibration_profile["weight_coefficients"][name] for name in WEIGHT_FEATURE_NAMES],
        dtype=float,
    )
    weight_scores = _probabilities(
        _feature_matrix(test_rows),
        coefficients=coefficients,
        intercept=float(calibration_profile["weight_intercept"]),
    )
    behavior_scores = [row.behavior_score for row in test_rows]
    fusion_coefficients = calibration_profile["fusion_coefficients"]
    fusion_scores = _probabilities(
        np.column_stack([weight_scores, behavior_scores]),
        coefficients=np.asarray(
            [fusion_coefficients["weight_score"], fusion_coefficients["behavior_score"]],
            dtype=float,
        ),
        intercept=float(calibration_profile["fusion_intercept"]),
    )
    labels = _labels(test_rows)
    pipeline_hash = str(calibration_profile["pipeline_sha256"])
    validation_hash = str(calibration_profile["validation_data_sha256"])
    locks = _load_bound_threshold_locks(
        calibration_profile,
        pipeline_hash=pipeline_hash,
        validation_hash=validation_hash,
    )
    scores = {"weight": weight_scores, "behavior": behavior_scores, "fusion": fusion_scores}
    predictions = {
        name: release_test_scores(lock=locks[name], pipeline_hash=pipeline_hash, scores=value)
        for name, value in scores.items()
    }
    predictions["simple_or"] = simple_or(
        weight_scores,
        behavior_scores,
        weight_threshold=locks["weight"].threshold,
        behavior_threshold=locks["behavior"].threshold,
    )
    secondary_operating_points = _load_bound_secondary_operating_points(
        calibration_profile,
        pipeline_hash=pipeline_hash,
        validation_hash=validation_hash,
    )
    secondary_test_metrics = {
        name: (
            {
                "status": operating_point.status,
                "threshold": operating_point.threshold,
                "metrics": detector_metrics(
                    labels,
                    scores[name],
                    float(operating_point.threshold),
                ),
            }
            if operating_point.status == "selected"
            else {"status": operating_point.status, "threshold": None, "metrics": None}
        )
        for name, operating_point in secondary_operating_points.items()
    }
    lineages = _release_lineages(
        rows=test_rows,
        labels=labels,
        scores=scores,
        predictions=predictions,
    )
    bootstrap = parent_lineage_cluster_bootstrap(
        lineages,
        _release_estimands,
        iterations=config.statistics.bootstrap_iterations,
        seed=config.data.seed,
    )
    estimates = bootstrap["estimates"]
    metrics = {
        name: _metric_release_payload(
            estimates,
            prefix=name,
            metric_names=CONTINUOUS_METRIC_NAMES,
        )
        for name in ("weight", "behavior", "fusion")
    }
    simple_or_metrics = _metric_release_payload(
        estimates,
        prefix="simple_or",
        metric_names=BINARY_METRIC_NAMES,
    )
    paired_differences = {
        comparison: _metric_release_payload(
            estimates,
            prefix=comparison,
            metric_names=(
                BINARY_METRIC_NAMES
                if comparison == "fusion_minus_simple_or"
                else CONTINUOUS_METRIC_NAMES
            ),
        )
        for comparison in (
            "fusion_minus_weight",
            "fusion_minus_behavior",
            "fusion_minus_simple_or",
        )
    }
    completed_groups: dict[str, list[DetectorEvidenceRow]] = {}
    for row in completed_test_rows:
        completed_groups.setdefault(row.parent_config_id, []).append(row)
    failed_pair_count = sum(
        any(row.qc_status == "failed" for row in pair) for pair in completed_groups.values()
    )
    return {
        "schema_version": 4,
        "status": "released",
        "config_sha256": config.config_hash,
        "pipeline_sha256": pipeline_hash,
        "calibration_profile_sha256": sha256_json(calibration_profile),
        "test_data_sha256": _evidence_hash(test_rows),
        "test_adapter_count": len(test_rows),
        "test_pair_count": len({row.parent_config_id for row in test_rows}),
        "pair_counts": {
            "planned": config.cohort.detector_split_pairs["test"],
            "completed": len(completed_groups),
            "qc_valid": len(lineages),
            "failed": failed_pair_count,
        },
        "adapter_sha256s": sorted(adapter_hashes),
        "unique_adapter_hash_count": len(set(adapter_hashes)),
        "effective_lineage_count": len(lineages),
        "fusion_evidence_label": calibration_profile["fusion_evidence_label"],
        "metrics": metrics,
        "simple_or": simple_or_metrics,
        "secondary_operating_points": calibration_profile["secondary_operating_points"],
        "secondary_test_metrics": secondary_test_metrics,
        "paired_differences": paired_differences,
        "score_distribution": {
            "statistical_unit": "parent_config_id",
            "lineage_count": len(lineages),
            "rows": lineages,
        },
        "bootstrap": {key: value for key, value in bootstrap.items() if key != "estimates"},
        "limitations": [
            "Held-out adapter counts do not support a production false-positive-rate claim."
        ],
    }
