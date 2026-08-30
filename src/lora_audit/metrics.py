"""Detector, task, calibration, uncertainty, and runtime metrics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import beta
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def is_finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


@dataclass(frozen=True)
class RuntimeMemorySummary:
    allocated_p50: float
    allocated_p90: float
    allocated_maximum: float
    reserved_p50: float
    reserved_p90: float
    reserved_maximum: float
    device_memory_minimum: float
    allocated_fraction_maximum: float
    reserved_fraction_maximum: float


def validated_positive_observations(values: list[float], message: str) -> np.ndarray:
    if any(isinstance(value, bool) for value in values):
        raise ValueError(message)
    try:
        observations = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(observations).all() or np.any(observations <= 0):
        raise ValueError(message)
    return observations


def observation_summary(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.quantile(values, 0.50, method="higher")),
        float(np.quantile(values, 0.90, method="higher")),
        float(values.max()),
    )


def summarize_runtime_memory(
    *,
    allocated_memory_gib: list[float] | None,
    reserved_memory_gib: list[float] | None,
    device_memory_gib: list[float] | None,
    expected_count: int,
    context: str,
) -> RuntimeMemorySummary:
    if any(
        values is None for values in (allocated_memory_gib, reserved_memory_gib, device_memory_gib)
    ):
        raise ValueError(f"{context} requires complete peak-memory telemetry")
    allocated = validated_positive_observations(
        allocated_memory_gib or [], f"{context} has invalid allocated-memory telemetry"
    )
    reserved = validated_positive_observations(
        reserved_memory_gib or [], f"{context} has invalid reserved-memory telemetry"
    )
    capacity = validated_positive_observations(
        device_memory_gib or [], f"{context} has invalid device-memory telemetry"
    )
    if not len(allocated) == len(reserved) == len(capacity) == expected_count:
        raise ValueError(f"{context} requires complete peak-memory telemetry")
    if np.any(allocated > reserved) or np.any(reserved > capacity):
        raise ValueError(f"{context} has inconsistent peak-memory telemetry")
    allocated_p50, allocated_p90, allocated_maximum = observation_summary(allocated)
    reserved_p50, reserved_p90, reserved_maximum = observation_summary(reserved)
    return RuntimeMemorySummary(
        allocated_p50=allocated_p50,
        allocated_p90=allocated_p90,
        allocated_maximum=allocated_maximum,
        reserved_p50=reserved_p50,
        reserved_p90=reserved_p90,
        reserved_maximum=reserved_maximum,
        device_memory_minimum=float(capacity.min()),
        allocated_fraction_maximum=float((allocated / capacity).max()),
        reserved_fraction_maximum=float((reserved / capacity).max()),
    )


def task_metrics(
    y_true: list[str], y_pred: list[str], *, labels: list[str] | None = None
) -> dict[str, float]:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("task metrics require aligned nonempty labels")
    if labels is not None:
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("task metric vocabulary must be nonempty and unique")
        unknown = (set(y_true) | set(y_pred)) - set(labels)
        if unknown:
            raise ValueError(f"task metric labels outside vocabulary: {sorted(unknown)}")
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def attack_success_rate(
    *, y_true: list[str], y_triggered_pred: list[str], target_intent: str
) -> dict[str, float | int]:
    if not y_true or len(y_true) != len(y_triggered_pred):
        raise ValueError("ASR requires aligned nonempty labels")
    if not target_intent:
        raise ValueError("ASR target intent must be nonempty")
    eligible = [index for index, label in enumerate(y_true) if label != target_intent]
    if not eligible:
        raise ValueError("ASR denominator has no non-target examples")
    successes = sum(y_triggered_pred[index] == target_intent for index in eligible)
    return {"asr": successes / len(eligible), "successes": successes, "eligible": len(eligible)}


def detector_metrics(y_true: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    if not y_true or len(y_true) != len(scores):
        raise ValueError("detector metrics require aligned nonempty observations")
    if any(type(label) is not int for label in y_true) or set(y_true) != {0, 1}:
        raise ValueError("detector metrics require both clean and backdoored classes")
    if not all(is_finite_number(score) for score in scores) or not is_finite_number(threshold):
        raise ValueError("detector scores and threshold must be finite")
    if any(not 0.0 <= score <= 1.0 for score in scores) or not 0.0 <= threshold <= 1.0:
        raise ValueError("detector scores and threshold must be probabilities")
    predictions = [int(score >= threshold) for score in scores]
    positives = [prediction for label, prediction in zip(y_true, predictions, strict=True) if label]
    negatives = [
        prediction for label, prediction in zip(y_true, predictions, strict=True) if not label
    ]
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "brier": float(brier_score_loss(y_true, scores)),
        "tpr": float(sum(positives) / len(positives)),
        "fpr": float(sum(negatives) / len(negatives)),
    }


def binary_detector_metrics(y_true: list[int], predictions: list[int]) -> dict[str, float]:
    """Metrics whose interpretation does not require probabilistic scores."""

    if not y_true or len(y_true) != len(predictions):
        raise ValueError("binary detector metrics require aligned nonempty observations")
    if any(type(label) is not int for label in y_true) or set(y_true) != {0, 1}:
        raise ValueError("binary detector metrics require both clean and backdoored classes")
    if any(type(prediction) is not int or prediction not in {0, 1} for prediction in predictions):
        raise ValueError("binary detector predictions must be exact binary integers")
    positives = [prediction for label, prediction in zip(y_true, predictions, strict=True) if label]
    negatives = [
        prediction for label, prediction in zip(y_true, predictions, strict=True) if not label
    ]
    return {
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "tpr": float(sum(positives) / len(positives)),
        "fpr": float(sum(negatives) / len(negatives)),
    }


def exact_binomial_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if (
        type(successes) is not int
        or type(trials) is not int
        or trials <= 0
        or successes < 0
        or successes > trials
    ):
        raise ValueError("invalid binomial counts")
    if not is_finite_number(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return lower, upper


def _bootstrap_estimates(value: float | dict[str, float], *, context: str) -> dict[str, float]:
    raw = value if isinstance(value, dict) else {"value": value}
    if not raw:
        raise ValueError(f"{context} returned no estimands")
    if any(not isinstance(name, str) or not name for name in raw):
        raise ValueError(f"{context} returned invalid estimand names")
    if any(not is_finite_number(estimate) for estimate in raw.values()):
        raise ValueError(f"{context} returned a non-finite value")
    return {name: float(estimate) for name, estimate in raw.items()}


def _validate_bootstrap_parameters(
    *,
    iterations: int,
    seed: int,
    confidence: float,
    min_lineages: int,
) -> None:
    if type(iterations) is not int or iterations < 5000:
        raise ValueError("parent-lineage bootstrap requires at least 5,000 iterations")
    if type(seed) is not int:
        raise ValueError("parent-lineage bootstrap seed must be an integer")
    if not is_finite_number(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("parent-lineage bootstrap confidence must be within (0, 1)")
    if type(min_lineages) is not int or min_lineages < 2:
        raise ValueError("parent-lineage bootstrap minimum must be at least two")


def _ordered_bootstrap_lineages(
    lineages: list[dict[str, Any]], *, min_lineages: int
) -> list[dict[str, Any]]:
    if not isinstance(lineages, list) or any(not isinstance(row, dict) for row in lineages):
        raise ValueError("parent-lineage bootstrap requires lineage mappings")
    parent_ids = [row.get("parent_config_id") for row in lineages]
    if any(not isinstance(parent_id, str) or not parent_id for parent_id in parent_ids):
        raise ValueError("parent-lineage bootstrap requires nonempty parent identifiers")
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("parent-lineage bootstrap contains duplicate parent groups")
    if len(lineages) < min_lineages:
        raise ValueError("parent-lineage bootstrap has insufficient effective independent groups")
    return [dict(row) for row in sorted(lineages, key=lambda row: str(row["parent_config_id"]))]


def parent_lineage_cluster_bootstrap(
    lineages: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | dict[str, float]],
    *,
    iterations: int = 5000,
    seed: int = 20260714,
    confidence: float = 0.95,
    min_lineages: int = 2,
) -> dict[str, Any]:
    """Bootstrap unique parent-lineage clusters with synchronized deterministic draws.

    Each input mapping represents exactly one lineage. A statistic may return several
    estimands so branch metrics and paired differences share every resample.
    """

    _validate_bootstrap_parameters(
        iterations=iterations,
        seed=seed,
        confidence=confidence,
        min_lineages=min_lineages,
    )
    ordered_lineages = _ordered_bootstrap_lineages(
        lineages,
        min_lineages=min_lineages,
    )
    point = _bootstrap_estimates(statistic(ordered_lineages), context="bootstrap statistic")
    estimand_names = tuple(point)
    rng = np.random.default_rng(seed)
    estimates = {name: np.empty(iterations, dtype=float) for name in estimand_names}
    for index in range(iterations):
        sampled_indices = rng.integers(0, len(ordered_lineages), size=len(ordered_lineages))
        sample = [ordered_lineages[int(sample_index)] for sample_index in sampled_indices]
        observed = _bootstrap_estimates(statistic(sample), context="bootstrap statistic")
        if tuple(observed) != estimand_names:
            raise ValueError("bootstrap statistic estimand keys changed across resamples")
        for name, estimate in observed.items():
            estimates[name][index] = estimate
    alpha = 1.0 - confidence
    return {
        "iterations": iterations,
        "valid_resamples": iterations,
        "invalid_resamples": 0,
        "lineage_count": len(ordered_lineages),
        "seed": seed,
        "confidence": confidence,
        "estimates": {
            name: {
                "estimate": point[name],
                "ci_lower": float(np.quantile(estimates[name], alpha / 2)),
                "ci_upper": float(np.quantile(estimates[name], 1.0 - alpha / 2)),
            }
            for name in estimand_names
        },
    }


def lineage_bootstrap(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    *,
    iterations: int = 5000,
    seed: int = 20260714,
) -> dict[str, float | int]:
    """Backward-compatible row-level adapter around the cluster bootstrap helper."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        parent_id = row.get("parent_config_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError("lineage bootstrap requires nonempty parent identifiers")
        groups.setdefault(parent_id, []).append(row)
    clusters = [
        {"parent_config_id": parent_id, "rows": groups[parent_id]} for parent_id in sorted(groups)
    ]

    result = parent_lineage_cluster_bootstrap(
        clusters,
        lambda sample: metric([row for cluster in sample for row in cluster["rows"]]),
        iterations=iterations,
        seed=seed,
    )
    estimate = result["estimates"]["value"]
    return {
        "iterations": result["iterations"],
        "valid_resamples": result["valid_resamples"],
        "invalid_resamples": result["invalid_resamples"],
        "estimate": estimate["estimate"],
        "ci_lower": estimate["ci_lower"],
        "ci_upper": estimate["ci_upper"],
        "lineage_count": result["lineage_count"],
        "seed": result["seed"],
        "confidence": result["confidence"],
    }
