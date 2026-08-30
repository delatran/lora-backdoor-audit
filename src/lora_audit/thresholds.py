"""Validation-only threshold selection and test-release locking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import matthews_corrcoef

from .hashing import is_sha256, sha256_json
from .metrics import is_finite_number


@dataclass(frozen=True)
class ThresholdLock:
    detector: str
    threshold: float
    validation_mcc: float
    validation_fpr: float
    validation_data_hash: str
    pipeline_hash: str
    selection_split: str = "validation"

    def __post_init__(self) -> None:
        if not isinstance(self.detector, str) or not self.detector.strip():
            raise ValueError("threshold lock requires a detector identifier")
        if not is_finite_number(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold lock requires a finite threshold within [0, 1]")
        if not is_finite_number(self.validation_mcc) or not -1.0 <= self.validation_mcc <= 1.0:
            raise ValueError("threshold lock validation MCC must be finite and within [-1, 1]")
        if not is_finite_number(self.validation_fpr) or not 0.0 <= self.validation_fpr <= 1.0:
            raise ValueError("threshold lock validation FPR must be finite and within [0, 1]")
        if not is_sha256(self.validation_data_hash) or not is_sha256(self.pipeline_hash):
            raise ValueError("threshold lock requires SHA-256 data and pipeline hashes")
        if self.selection_split != "validation":
            raise ValueError("threshold lock was not selected on validation")

    @property
    def lock_hash(self) -> str:
        return sha256_json(asdict(self))


def _validate_secondary_class_counts(
    *,
    clean_adapter_count: int,
    backdoored_adapter_count: int,
    empirical_fpr_resolution: float,
) -> None:
    if (
        type(clean_adapter_count) is not int
        or type(backdoored_adapter_count) is not int
        or clean_adapter_count <= 0
        or backdoored_adapter_count <= 0
    ):
        raise ValueError("secondary operating point requires both validation classes")
    expected_resolution = 1.0 / clean_adapter_count
    if not np.isclose(empirical_fpr_resolution, expected_resolution, rtol=0.0, atol=1e-15):
        raise ValueError("secondary operating point has an invalid empirical FPR resolution")


def _validate_secondary_selected_values(
    *,
    status: str,
    threshold: float | None,
    validation_tpr: float | None,
    validation_fpr: float | None,
    maximum_validation_fpr: float,
) -> None:
    selected_values = (threshold, validation_tpr, validation_fpr)
    if status == "selected":
        if any(not is_finite_number(value) for value in selected_values):
            raise ValueError("selected secondary operating point is incomplete")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("secondary operating point threshold must be within [0, 1]")
        if not 0.0 <= float(validation_tpr) <= 1.0:
            raise ValueError("secondary operating point TPR must be within [0, 1]")
        if not 0.0 <= float(validation_fpr) <= maximum_validation_fpr:
            raise ValueError("secondary operating point exceeds its validation FPR ceiling")
        return
    if any(value is not None for value in selected_values):
        raise ValueError("unselected secondary operating point cannot contain a threshold")


@dataclass(frozen=True)
class SecondaryOperatingPoint:
    detector: str
    objective: Literal["maximum_tpr_subject_to_fpr_at_most_0.10"]
    status: Literal["selected", "not_estimable_at_current_resolution", "no_admissible_threshold"]
    maximum_validation_fpr: float
    clean_adapter_count: int
    backdoored_adapter_count: int
    empirical_fpr_resolution: float
    validation_data_hash: str
    pipeline_hash: str
    threshold: float | None = None
    validation_tpr: float | None = None
    validation_fpr: float | None = None
    selection_split: str = "validation"

    def __post_init__(self) -> None:
        if not isinstance(self.detector, str) or not self.detector.strip():
            raise ValueError("secondary operating point requires a detector identifier")
        if self.objective != "maximum_tpr_subject_to_fpr_at_most_0.10":
            raise ValueError("secondary operating point objective changed")
        if self.maximum_validation_fpr != 0.10:
            raise ValueError("secondary operating point FPR ceiling changed")
        _validate_secondary_class_counts(
            clean_adapter_count=self.clean_adapter_count,
            backdoored_adapter_count=self.backdoored_adapter_count,
            empirical_fpr_resolution=self.empirical_fpr_resolution,
        )
        if not is_sha256(self.validation_data_hash) or not is_sha256(self.pipeline_hash):
            raise ValueError("secondary operating point requires SHA-256 bindings")
        if self.selection_split != "validation":
            raise ValueError("secondary operating point was not selected on validation")
        _validate_secondary_selected_values(
            status=self.status,
            threshold=self.threshold,
            validation_tpr=self.validation_tpr,
            validation_fpr=self.validation_fpr,
            maximum_validation_fpr=self.maximum_validation_fpr,
        )

    @property
    def lock_hash(self) -> str:
        return sha256_json(asdict(self))


def _fpr(y_true: np.ndarray, predictions: np.ndarray) -> float:
    negatives = y_true == 0
    if not np.any(negatives):
        raise ValueError("threshold selection requires clean validation adapters")
    return float(np.mean(predictions[negatives] == 1))


def select_validation_threshold(
    *,
    detector: str,
    y_true: list[int],
    scores: list[float],
    validation_data_hash: str,
    pipeline_hash: str,
    split: str = "validation",
) -> ThresholdLock:
    if split != "validation":
        raise ValueError("thresholds may only be selected on validation")
    if not isinstance(detector, str) or not detector.strip():
        raise ValueError("threshold selection requires a detector identifier")
    if any(type(label) is not int or label not in {0, 1} for label in y_true):
        raise ValueError("threshold selection labels must be exact binary integers")
    if any(not is_finite_number(score) for score in scores):
        raise ValueError("threshold selection requires finite numeric scores")
    if not is_sha256(validation_data_hash) or not is_sha256(pipeline_hash):
        raise ValueError("threshold selection requires SHA-256 data and pipeline hashes")
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if labels.shape != values.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError("threshold selection requires aligned scores and both classes")
    if not np.isfinite(values).all():
        raise ValueError("threshold selection requires finite scores")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("threshold selection requires probability scores")
    upper_candidate = min(1.0, float(np.nextafter(values.max(), np.inf)))
    candidates = sorted({*values.tolist(), upper_candidate}, reverse=True)
    evaluated: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predictions = (values >= threshold).astype(int)
        mcc = float(matthews_corrcoef(labels, predictions))
        evaluated.append((mcc, _fpr(labels, predictions), threshold))
    best_mcc, best_fpr, best_threshold = max(
        evaluated,
        key=lambda value: (value[0], -value[1], value[2]),
    )
    return ThresholdLock(
        detector=detector,
        threshold=best_threshold,
        validation_mcc=best_mcc,
        validation_fpr=best_fpr,
        validation_data_hash=validation_data_hash,
        pipeline_hash=pipeline_hash,
    )


def _secondary_selection_arrays(
    *,
    detector: str,
    y_true: list[int],
    scores: list[float],
    validation_data_hash: str,
    pipeline_hash: str,
    split: str,
    maximum_fpr: float,
) -> tuple[np.ndarray, np.ndarray]:
    if split != "validation":
        raise ValueError("secondary operating points may only be selected on validation")
    if maximum_fpr != 0.10:
        raise ValueError("secondary operating point maximum_fpr must remain 0.10")
    if not isinstance(detector, str) or not detector.strip():
        raise ValueError("secondary operating point requires a detector identifier")
    if any(type(label) is not int or label not in {0, 1} for label in y_true):
        raise ValueError("secondary operating point labels must be exact binary integers")
    if any(not is_finite_number(score) for score in scores):
        raise ValueError("secondary operating point requires finite numeric scores")
    if not is_sha256(validation_data_hash) or not is_sha256(pipeline_hash):
        raise ValueError("secondary operating point requires SHA-256 data and pipeline hashes")
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if labels.shape != values.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError("secondary operating point requires aligned scores and both classes")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("secondary operating point requires probability scores")
    return labels, values


def select_secondary_validation_operating_point(
    *,
    detector: str,
    y_true: list[int],
    scores: list[float],
    validation_data_hash: str,
    pipeline_hash: str,
    split: str = "validation",
    maximum_fpr: float = 0.10,
) -> SecondaryOperatingPoint:
    """Lock the proposal's supplementary validation operating point."""

    labels, values = _secondary_selection_arrays(
        detector=detector,
        y_true=y_true,
        scores=scores,
        validation_data_hash=validation_data_hash,
        pipeline_hash=pipeline_hash,
        split=split,
        maximum_fpr=maximum_fpr,
    )
    clean_count = int(np.sum(labels == 0))
    backdoored_count = int(np.sum(labels == 1))
    resolution = 1.0 / clean_count
    common = {
        "detector": detector,
        "objective": "maximum_tpr_subject_to_fpr_at_most_0.10",
        "maximum_validation_fpr": maximum_fpr,
        "clean_adapter_count": clean_count,
        "backdoored_adapter_count": backdoored_count,
        "empirical_fpr_resolution": resolution,
        "validation_data_hash": validation_data_hash,
        "pipeline_hash": pipeline_hash,
    }
    if resolution > maximum_fpr:
        return SecondaryOperatingPoint(
            **common,
            status="not_estimable_at_current_resolution",
        )

    upper_candidate = min(1.0, float(np.nextafter(values.max(), np.inf)))
    candidates = sorted({*values.tolist(), upper_candidate}, reverse=True)
    admissible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predictions = (values >= threshold).astype(int)
        fpr = _fpr(labels, predictions)
        if fpr <= maximum_fpr:
            positives = labels == 1
            tpr = float(np.mean(predictions[positives] == 1))
            admissible.append((tpr, fpr, threshold))
    if not admissible:
        return SecondaryOperatingPoint(
            **common,
            status="no_admissible_threshold",
        )
    best_tpr, best_fpr, best_threshold = max(
        admissible,
        key=lambda value: (value[0], -value[1], value[2]),
    )
    return SecondaryOperatingPoint(
        **common,
        status="selected",
        threshold=best_threshold,
        validation_tpr=best_tpr,
        validation_fpr=best_fpr,
    )


def write_threshold_lock(lock: ThresholdLock, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(lock), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_threshold_lock(path: str | Path, *, expected_pipeline_hash: str) -> ThresholdLock:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    lock = ThresholdLock(**raw)
    if lock.selection_split != "validation":
        raise ValueError("threshold lock was not selected on validation")
    if lock.pipeline_hash != expected_pipeline_hash:
        raise ValueError("threshold lock pipeline hash mismatch")
    return lock


def release_test_scores(
    *, lock: ThresholdLock, pipeline_hash: str, scores: list[float]
) -> list[int]:
    if lock.pipeline_hash != pipeline_hash:
        raise ValueError("test release refused: pipeline hash changed after threshold lock")
    if not all(is_finite_number(score) for score in scores):
        raise ValueError("test release requires finite scores")
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("test release requires probability scores")
    return [int(score >= lock.threshold) for score in scores]
