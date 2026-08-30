"""Pilot-based or explicitly conservative compute and storage estimation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .execution_profiles import (
    ExecutionProfile,
    canonical_execution_profile_path,
    execution_profile_binding_matches,
    load_execution_profile,
)
from .hashing import is_sha256, sha256_json
from .metrics import (
    RuntimeMemorySummary,
    observation_summary,
    summarize_runtime_memory,
    validated_positive_observations,
)
from .preflight import device_matches_requirements

TRAINING_TELEMETRY_FIELDS = frozenset(
    {
        "parent_config_id",
        "adapter_label",
        "replicate_seed",
        "pair_training_seed",
        "poison_selection_seed",
        "probe_seed",
        "split_assignment_id",
        "training_unit_id",
        "device_hours",
        "runtime_seconds",
        "peak_allocated_memory_gib",
        "peak_reserved_memory_gib",
        "training_record_count",
        "records_sha256",
        "adapter_size_bytes",
        "adapter_sha256",
        "device_observation",
        "backend",
        "framework_version",
        "config_sha256",
        "source_tree_sha256",
        "source_manifest_sha256",
        "execution_profile",
    }
)
EVALUATION_TELEMETRY_FIELDS = frozenset(
    {
        "parent_config_id",
        "replicate_seed",
        "pair_training_seed",
        "poison_selection_seed",
        "probe_seed",
        "split_assignment_id",
        "device_hours",
        "runtime_seconds",
        "peak_allocated_memory_gib",
        "peak_reserved_memory_gib",
        "probe_group_count",
        "test_record_count",
        "qc_valid",
        "qc_reason_codes",
        "evaluation_report_sha256",
        "clean_adapter_sha256",
        "backdoored_adapter_sha256",
        "device_observation",
        "backend",
        "config_sha256",
        "source_tree_sha256",
        "source_manifest_sha256",
        "execution_profile",
    }
)

COMPUTE_ESTIMATE_SCHEMA_VERSION = 9
COMPUTE_ESTIMATE_SCOPE = "full_research_pipeline"
REQUIRED_REPRESENTATIVE_PILOT_PAIRS = 3
RQ2_ADAPTER_RUNS = 12
RQ2_EVALUATION_UNITS = 12
RQ3_EVALUATION_UNITS = 24
MANAGED_RUNTIME_RESERVE_FRACTION = 0.2
PILOT_CONTINUATION_POLICY = "outcome_classified_optimistic_full_research_v2"
THRESHOLD_QC_FAILURE_REASON_CODES = frozenset(
    {
        "clean_triggered_target_rate_above_threshold",
        "backdoored_asr_below_threshold",
        "paired_asr_lift_below_threshold",
        "absolute_clean_macro_f1_difference_above_threshold",
    }
)


@dataclass(frozen=True)
class PilotPairBinding:
    parent_config_id: str
    clean_adapter_sha256: str
    backdoored_adapter_sha256: str

    def __post_init__(self) -> None:
        if not self.parent_config_id:
            raise ValueError("pilot pair binding requires a lineage ID")
        if not is_sha256(self.clean_adapter_sha256) or not is_sha256(
            self.backdoored_adapter_sha256
        ):
            raise ValueError("pilot pair binding requires valid adapter hashes")
        if self.clean_adapter_sha256 == self.backdoored_adapter_sha256:
            raise ValueError("pilot pair binding requires distinct adapter hashes")


@dataclass(frozen=True)
class PilotTelemetry:
    device_hours: tuple[float, ...]
    adapter_sizes_gib: tuple[float, ...]
    peak_allocated_memory_gib: tuple[float, ...]
    peak_reserved_memory_gib: tuple[float, ...]
    device_memory_gib: tuple[float, ...]
    pair_bindings: tuple[PilotPairBinding, ...]

    def __post_init__(self) -> None:
        if len(self.device_hours) not in {2, 4, 6}:
            raise ValueError("pilot telemetry requires one to three adapter pairs")
        if len(self.adapter_sizes_gib) != len(self.device_hours):
            raise ValueError("pilot telemetry size observations are incomplete")
        if len(self.pair_bindings) * 2 != len(self.device_hours):
            raise ValueError("pilot telemetry pair bindings are incomplete")
        if any(not _positive_finite(value) for value in self.device_hours):
            raise ValueError("pilot telemetry requires positive device hours")
        if any(not _positive_finite(value) for value in self.adapter_sizes_gib):
            raise ValueError("pilot telemetry requires positive adapter sizes")
        _validate_memory_series(
            allocated=self.peak_allocated_memory_gib,
            reserved=self.peak_reserved_memory_gib,
            capacities=self.device_memory_gib,
            expected_count=len(self.device_hours),
            context="pilot training",
        )
        _verify_unique_pair_bindings(self.pair_bindings, "pilot training")


@dataclass(frozen=True)
class PilotEvaluationTelemetry:
    device_hours: tuple[float, ...]
    peak_allocated_memory_gib: tuple[float, ...]
    peak_reserved_memory_gib: tuple[float, ...]
    device_memory_gib: tuple[float, ...]
    failed_pair_count: int
    pair_bindings: tuple[PilotPairBinding, ...]
    threshold_failure_pair_count: int = 0
    integrity_failure_pair_count: int = 0
    failure_reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.device_hours) not in {1, 2, 3}:
            raise ValueError("pilot evaluation requires one to three pairs")
        if len(self.pair_bindings) != len(self.device_hours):
            raise ValueError("pilot evaluation pair bindings are incomplete")
        if any(not _positive_finite(value) for value in self.device_hours):
            raise ValueError("pilot evaluation requires positive device hours")
        _validate_memory_series(
            allocated=self.peak_allocated_memory_gib,
            reserved=self.peak_reserved_memory_gib,
            capacities=self.device_memory_gib,
            expected_count=len(self.device_hours),
            context="pilot evaluation",
        )
        if type(self.failed_pair_count) is not int or not 0 <= self.failed_pair_count <= len(
            self.device_hours
        ):
            raise ValueError("pilot evaluation failed-pair count is invalid")
        classified_counts = (
            self.threshold_failure_pair_count,
            self.integrity_failure_pair_count,
        )
        if any(type(value) is not int or value < 0 for value in classified_counts):
            raise ValueError("pilot evaluation failure classification is invalid")
        if sum(classified_counts) != self.failed_pair_count:
            raise ValueError("pilot evaluation failure classification is incomplete")
        if (
            any(not isinstance(code, str) or not code for code in self.failure_reason_codes)
            or len(set(self.failure_reason_codes)) != len(self.failure_reason_codes)
            or tuple(sorted(self.failure_reason_codes)) != self.failure_reason_codes
        ):
            raise ValueError("pilot evaluation failure reason codes are invalid")
        if bool(self.failed_pair_count) != bool(self.failure_reason_codes):
            raise ValueError("pilot evaluation failure reasons do not match its failed-pair count")
        if self.integrity_failure_pair_count == 0 and any(
            code not in THRESHOLD_QC_FAILURE_REASON_CODES for code in self.failure_reason_codes
        ):
            raise ValueError("pilot evaluation threshold failures contain an integrity reason")
        _verify_unique_pair_bindings(self.pair_bindings, "pilot evaluation")


@dataclass(frozen=True)
class _TrainingTelemetryRow:
    parent_config_id: str
    adapter_label: Literal["clean", "backdoored"]
    replicate_seed: int
    pair_training_seed: int
    poison_selection_seed: int
    probe_seed: int
    split_assignment_id: str
    training_unit_id: str
    device_hours: float
    training_record_count: int
    records_sha256: str
    adapter_size_bytes: int
    adapter_sha256: str
    peak_allocated_memory_gib: float
    peak_reserved_memory_gib: float
    device_memory_gib: float


@dataclass(frozen=True)
class _EvaluationTelemetryRow:
    parent_config_id: str
    replicate_seed: int
    pair_training_seed: int
    poison_selection_seed: int
    probe_seed: int
    split_assignment_id: str
    device_hours: float
    qc_valid: bool
    qc_reason_codes: tuple[str, ...]
    clean_adapter_sha256: str
    backdoored_adapter_sha256: str
    peak_allocated_memory_gib: float
    peak_reserved_memory_gib: float
    device_memory_gib: float


def _positive_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _validated_qc_reason_codes(
    value: Any,
    *,
    qc_valid: bool,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(code, str) or not code or code.strip() != code for code in value
    ):
        raise ValueError(f"{context} has invalid QC reason codes")
    codes = tuple(value)
    if len(set(codes)) != len(codes):
        raise ValueError(f"{context} repeats a QC reason code")
    if qc_valid == bool(codes):
        raise ValueError(f"{context} has inconsistent QC status and reason codes")
    return codes


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _require_execution_profile(execution_profile: ExecutionProfile) -> None:
    """Reject hand-built or incomplete runtime contracts before reading evidence."""

    if (
        not isinstance(execution_profile, ExecutionProfile)
        or not execution_profile.strict_accelerator
        or not execution_profile.require_bf16_probe
        or not is_sha256(execution_profile.profile_hash)
        or not is_sha256(execution_profile.profile_file_sha256)
        or sha256_json(execution_profile.semantic_payload()) != execution_profile.profile_hash
    ):
        raise ValueError("execution profile object is invalid")
    project_root = Path(__file__).resolve().parents[2]
    try:
        canonical_path = canonical_execution_profile_path(
            project_root,
            execution_profile.profile_id,
        )
        locked_profile = load_execution_profile(canonical_path)
    except (OSError, ValueError) as exc:
        raise ValueError("execution profile is not a locked canonical profile") from exc
    if locked_profile != execution_profile:
        raise ValueError("execution profile is not a locked canonical profile")


def _validate_lineage_telemetry_fields(
    row: dict[str, Any], *, context: str, include_training_unit: bool
) -> None:
    """Validate canonical lineage fields that make a pilot pair auditable."""

    required_integer_fields = {
        "replicate_seed": _nonnegative_int,
        "pair_training_seed": _positive_int,
        "poison_selection_seed": _positive_int,
        "probe_seed": _nonnegative_int,
    }
    invalid = [
        field
        for field, predicate in required_integer_fields.items()
        if not predicate(row.get(field))
    ]
    if invalid:
        raise ValueError(f"{context} has invalid lineage seed fields: {invalid}")
    split_assignment_id = row.get("split_assignment_id")
    if (
        not isinstance(split_assignment_id, str)
        or not split_assignment_id
        or split_assignment_id != split_assignment_id.strip()
    ):
        raise ValueError(f"{context} has invalid split_assignment_id")
    if include_training_unit:
        training_unit_id = row.get("training_unit_id")
        if (
            not isinstance(training_unit_id, str)
            or not training_unit_id
            or training_unit_id != training_unit_id.strip()
        ):
            raise ValueError(f"{context} has invalid training_unit_id")


def _validate_memory_series(
    *,
    allocated: tuple[float, ...],
    reserved: tuple[float, ...],
    capacities: tuple[float, ...],
    expected_count: int,
    context: str,
) -> None:
    if not (len(allocated) == len(reserved) == len(capacities) == expected_count):
        raise ValueError(f"{context} memory observations are incomplete")
    if any(
        not _positive_finite(value)
        for series in (allocated, reserved, capacities)
        for value in series
    ):
        raise ValueError(f"{context} memory observations must be positive and finite")
    if any(
        not allocated_value <= reserved_value <= capacity
        for allocated_value, reserved_value, capacity in zip(
            allocated, reserved, capacities, strict=True
        )
    ):
        raise ValueError(f"{context} memory observations are inconsistent")


def _verify_unique_pair_bindings(bindings: tuple[PilotPairBinding, ...], context: str) -> None:
    parent_ids = [binding.parent_config_id for binding in bindings]
    adapter_hashes = [
        adapter_hash
        for binding in bindings
        for adapter_hash in (
            binding.clean_adapter_sha256,
            binding.backdoored_adapter_sha256,
        )
    ]
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError(f"{context} contains duplicate lineage bindings")
    if len(set(adapter_hashes)) != len(adapter_hashes):
        raise ValueError(f"{context} reuses adapter hashes across lineages")


def _read_jsonl_objects(source: Path, context: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {context} JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{context} line {line_number} is not an object")
        rows.append((line_number, row))
    return rows


def _require_exact_fields(row: dict[str, Any], expected: frozenset[str], context: str) -> None:
    observed = set(row)
    if observed != expected:
        raise ValueError(
            f"{context} field set mismatch: "
            f"missing={sorted(expected - observed)} unexpected={sorted(observed - expected)}"
        )


def _require_nonempty_string(row: dict[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} has invalid {field}")
    return value


def _require_sha256(row: dict[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not is_sha256(value):
        raise ValueError(f"{context} has invalid {field}")
    return str(value)


def _validated_device_hours(row: dict[str, Any], context: str) -> float:
    if not _positive_finite(row.get("device_hours")) or not _positive_finite(
        row.get("runtime_seconds")
    ):
        raise ValueError(f"{context} has invalid runtime")
    device_hours = float(row["device_hours"])
    expected_hours = float(row["runtime_seconds"]) / 3600.0
    if not math.isclose(device_hours, expected_hours, rel_tol=1e-9):
        raise ValueError(f"{context} has inconsistent runtime")
    return device_hours


def _validate_device_observation(
    row: dict[str, Any],
    *,
    context: str,
    execution_profile: ExecutionProfile,
) -> float:
    device = row.get("device_observation")
    if not isinstance(device, dict):
        raise ValueError(f"{context} has no device observation")
    backend = _require_nonempty_string(row, "backend", context)
    if not device_matches_requirements(
        device,
        execution_profile,
        backend=backend,
    ):
        raise ValueError(f"{context} device observation does not match execution profile")
    return float(device["total_memory_gib"])


def _validated_peak_memory(
    row: dict[str, Any],
    *,
    device_memory_gib: float,
    context: str,
) -> tuple[float, float]:
    allocated = row.get("peak_allocated_memory_gib")
    reserved = row.get("peak_reserved_memory_gib")
    if not _positive_finite(allocated) or not _positive_finite(reserved):
        raise ValueError(f"{context} has invalid peak memory")
    allocated_value = float(allocated)
    reserved_value = float(reserved)
    if not allocated_value <= reserved_value <= device_memory_gib:
        raise ValueError(f"{context} has inconsistent peak memory")
    return allocated_value, reserved_value


def _parse_training_telemetry_row(
    line_number: int,
    row: dict[str, Any],
    *,
    expected_config_sha256: str,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> _TrainingTelemetryRow:
    context = f"pilot telemetry line {line_number}"
    _require_exact_fields(row, TRAINING_TELEMETRY_FIELDS, context)
    parent = _require_nonempty_string(row, "parent_config_id", context)
    label = row.get("adapter_label")
    if label not in {"clean", "backdoored"}:
        raise ValueError(f"{context} has an invalid adapter label")
    if row.get("config_sha256") != expected_config_sha256:
        raise ValueError("pilot telemetry config hash mismatch")
    if _require_sha256(row, "source_tree_sha256", context) != expected_source_tree_sha256:
        raise ValueError("pilot telemetry source tree hash mismatch")
    if _require_sha256(row, "source_manifest_sha256", context) != expected_source_manifest_sha256:
        raise ValueError("pilot telemetry source manifest hash mismatch")
    if not execution_profile_binding_matches(row.get("execution_profile"), execution_profile):
        raise ValueError("pilot telemetry execution profile mismatch")
    _validate_lineage_telemetry_fields(
        row,
        context=context,
        include_training_unit=True,
    )
    device_hours = _validated_device_hours(row, context)
    if not _positive_int(row.get("training_record_count")):
        raise ValueError(f"{context} has invalid record count")
    if not _positive_int(row.get("adapter_size_bytes")):
        raise ValueError(f"{context} has invalid adapter size")
    records_sha256 = _require_sha256(row, "records_sha256", context)
    adapter_sha256 = _require_sha256(row, "adapter_sha256", context)
    _require_nonempty_string(row, "framework_version", context)
    device_memory = _validate_device_observation(
        row,
        context=context,
        execution_profile=execution_profile,
    )
    peak_allocated, peak_reserved = _validated_peak_memory(
        row,
        device_memory_gib=device_memory,
        context=context,
    )
    return _TrainingTelemetryRow(
        parent_config_id=parent,
        adapter_label="clean" if label == "clean" else "backdoored",
        replicate_seed=int(row["replicate_seed"]),
        pair_training_seed=int(row["pair_training_seed"]),
        poison_selection_seed=int(row["poison_selection_seed"]),
        probe_seed=int(row["probe_seed"]),
        split_assignment_id=str(row["split_assignment_id"]),
        training_unit_id=str(row["training_unit_id"]),
        device_hours=device_hours,
        training_record_count=int(row["training_record_count"]),
        records_sha256=records_sha256,
        adapter_size_bytes=int(row["adapter_size_bytes"]),
        adapter_sha256=adapter_sha256,
        peak_allocated_memory_gib=peak_allocated,
        peak_reserved_memory_gib=peak_reserved,
        device_memory_gib=device_memory,
    )


def _training_pair_bindings(
    rows: list[_TrainingTelemetryRow],
) -> tuple[PilotPairBinding, ...]:
    by_parent: dict[str, dict[str, _TrainingTelemetryRow]] = defaultdict(dict)
    for row in rows:
        if row.adapter_label in by_parent[row.parent_config_id]:
            key = (row.parent_config_id, row.adapter_label)
            raise ValueError(f"duplicate pilot telemetry key: {key}")
        by_parent[row.parent_config_id][row.adapter_label] = row
    incomplete = {
        parent: sorted(pair_rows)
        for parent, pair_rows in by_parent.items()
        if set(pair_rows) != {"clean", "backdoored"}
    }
    if incomplete:
        raise ValueError(f"pilot telemetry contains incomplete pairs: {incomplete}")
    bindings: list[PilotPairBinding] = []
    for parent in sorted(by_parent):
        clean = by_parent[parent]["clean"]
        backdoored = by_parent[parent]["backdoored"]
        if clean.training_record_count != backdoored.training_record_count:
            raise ValueError(f"pilot telemetry pair {parent!r} has mismatched record counts")
        if clean.records_sha256 == backdoored.records_sha256:
            raise ValueError(f"pilot telemetry pair {parent!r} has identical training data hashes")
        lineage_fields = (
            "replicate_seed",
            "pair_training_seed",
            "poison_selection_seed",
            "probe_seed",
            "split_assignment_id",
        )
        if any(getattr(clean, field) != getattr(backdoored, field) for field in lineage_fields):
            raise ValueError(f"pilot telemetry pair {parent!r} has mismatched lineage fields")
        if clean.training_unit_id == backdoored.training_unit_id:
            raise ValueError(f"pilot telemetry pair {parent!r} reuses a training unit")
        bindings.append(
            PilotPairBinding(
                parent_config_id=parent,
                clean_adapter_sha256=clean.adapter_sha256,
                backdoored_adapter_sha256=backdoored.adapter_sha256,
            )
        )
    result = tuple(bindings)
    _verify_unique_pair_bindings(result, "pilot training")
    return result


def _parse_evaluation_telemetry_row(
    line_number: int,
    row: dict[str, Any],
    *,
    expected_config_sha256: str,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> _EvaluationTelemetryRow:
    context = f"pilot evaluation line {line_number}"
    _require_exact_fields(row, EVALUATION_TELEMETRY_FIELDS, context)
    parent = _require_nonempty_string(row, "parent_config_id", context)
    if row.get("config_sha256") != expected_config_sha256:
        raise ValueError("pilot evaluation config hash mismatch")
    if _require_sha256(row, "source_tree_sha256", context) != expected_source_tree_sha256:
        raise ValueError("pilot evaluation source tree hash mismatch")
    if _require_sha256(row, "source_manifest_sha256", context) != expected_source_manifest_sha256:
        raise ValueError("pilot evaluation source manifest hash mismatch")
    if not execution_profile_binding_matches(row.get("execution_profile"), execution_profile):
        raise ValueError("pilot evaluation execution profile mismatch")
    _validate_lineage_telemetry_fields(
        row,
        context=context,
        include_training_unit=False,
    )
    device_hours = _validated_device_hours(row, context)
    if not _positive_int(row.get("probe_group_count")) or not _positive_int(
        row.get("test_record_count")
    ):
        raise ValueError(f"{context} has invalid record counts")
    qc_valid = row.get("qc_valid")
    if not isinstance(qc_valid, bool):
        raise ValueError(f"{context} has invalid QC status")
    qc_reason_codes = _validated_qc_reason_codes(
        row.get("qc_reason_codes"),
        qc_valid=qc_valid,
        context=context,
    )
    _require_sha256(row, "evaluation_report_sha256", context)
    clean_hash = _require_sha256(row, "clean_adapter_sha256", context)
    backdoored_hash = _require_sha256(row, "backdoored_adapter_sha256", context)
    device_memory = _validate_device_observation(
        row,
        context=context,
        execution_profile=execution_profile,
    )
    peak_allocated, peak_reserved = _validated_peak_memory(
        row,
        device_memory_gib=device_memory,
        context=context,
    )
    return _EvaluationTelemetryRow(
        parent_config_id=parent,
        replicate_seed=int(row["replicate_seed"]),
        pair_training_seed=int(row["pair_training_seed"]),
        poison_selection_seed=int(row["poison_selection_seed"]),
        probe_seed=int(row["probe_seed"]),
        split_assignment_id=str(row["split_assignment_id"]),
        device_hours=device_hours,
        qc_valid=qc_valid,
        qc_reason_codes=qc_reason_codes,
        clean_adapter_sha256=clean_hash,
        backdoored_adapter_sha256=backdoored_hash,
        peak_allocated_memory_gib=peak_allocated,
        peak_reserved_memory_gib=peak_reserved,
        device_memory_gib=device_memory,
    )


def verify_pilot_telemetry_alignment(
    training: PilotTelemetry,
    evaluation: PilotEvaluationTelemetry,
) -> tuple[PilotPairBinding, ...]:
    """Require evaluation telemetry to describe the exact trained adapter pairs."""

    if training.pair_bindings != evaluation.pair_bindings:
        raise ValueError("pilot training/evaluation lineage or adapter hash mismatch")
    return training.pair_bindings


@dataclass(frozen=True)
class _ComputePlanningEvidence:
    estimate_kind: str
    planning_basis: str
    training_p50: float | None
    training_p90: float | None
    training_maximum: float | None
    planning_training_per_adapter: float
    adapter_size_maximum: float | None
    evaluation_p50: float | None
    evaluation_p90: float | None
    evaluation_maximum: float | None
    planning_evaluation_per_pair: float
    training_memory: RuntimeMemorySummary | None
    evaluation_memory: RuntimeMemorySummary | None


def _memory_evidence_fields(
    summary: RuntimeMemorySummary | None,
    *,
    unit: Literal["adapter", "evaluation_pair"],
) -> dict[str, float | None]:
    values = {
        "peak_allocated_memory_gib": (
            summary.allocated_p50 if summary else None,
            summary.allocated_p90 if summary else None,
            summary.allocated_maximum if summary else None,
        ),
        "peak_reserved_memory_gib": (
            summary.reserved_p50 if summary else None,
            summary.reserved_p90 if summary else None,
            summary.reserved_maximum if summary else None,
        ),
    }
    fields = {
        f"observed_{metric}_per_{unit}_{quantile}": value
        for metric, observations in values.items()
        for quantile, value in zip(("p50", "p90", "max"), observations, strict=True)
    }
    fields.update(
        {
            f"observed_device_memory_gib_per_{unit}_min": (
                summary.device_memory_minimum if summary else None
            ),
            f"observed_peak_allocated_memory_fraction_per_{unit}_max": (
                summary.allocated_fraction_maximum if summary else None
            ),
            f"observed_peak_reserved_memory_fraction_per_{unit}_max": (
                summary.reserved_fraction_maximum if summary else None
            ),
        }
    )
    return fields


def _validate_pilot_failure_summary(
    *,
    pilot_failed_pair_count: int,
    pilot_threshold_failure_pair_count: int,
    pilot_integrity_failure_pair_count: int,
    pilot_qc_reason_codes: list[str],
) -> None:
    if type(pilot_failed_pair_count) is not int or pilot_failed_pair_count < 0:
        raise ValueError("pilot failed-pair count is invalid")
    classified_counts = (
        pilot_threshold_failure_pair_count,
        pilot_integrity_failure_pair_count,
    )
    if any(type(value) is not int or value < 0 for value in classified_counts):
        raise ValueError("pilot failure classification is invalid")
    if sum(classified_counts) != pilot_failed_pair_count:
        raise ValueError("pilot failure classification does not match failed-pair count")
    if (
        any(
            not isinstance(code, str) or not code or code.strip() != code
            for code in pilot_qc_reason_codes
        )
        or len(set(pilot_qc_reason_codes)) != len(pilot_qc_reason_codes)
        or sorted(pilot_qc_reason_codes) != pilot_qc_reason_codes
    ):
        raise ValueError("pilot QC reason-code summary is invalid")
    if bool(pilot_failed_pair_count) != bool(pilot_qc_reason_codes):
        raise ValueError("pilot QC reason-code summary does not match failed-pair count")
    if pilot_integrity_failure_pair_count == 0 and any(
        code not in THRESHOLD_QC_FAILURE_REASON_CODES for code in pilot_qc_reason_codes
    ):
        raise ValueError("pilot threshold-failure summary contains an integrity reason")
    if pilot_threshold_failure_pair_count and not any(
        code in THRESHOLD_QC_FAILURE_REASON_CODES for code in pilot_qc_reason_codes
    ):
        raise ValueError("pilot threshold-failure summary has no threshold reason")
    if pilot_integrity_failure_pair_count and not any(
        code not in THRESHOLD_QC_FAILURE_REASON_CODES for code in pilot_qc_reason_codes
    ):
        raise ValueError("pilot integrity-failure summary has no integrity reason")


def _validate_compute_estimate_inputs(
    *,
    adapter_size_gib: float,
    available_device_hours: float | None,
    core_adapters: int,
    target_adapters: int,
    conservative_device_hours_per_adapter: float,
    planning_safety_multiplier: float,
    pilot_failed_pair_count: int,
    pilot_threshold_failure_pair_count: int,
    pilot_integrity_failure_pair_count: int,
    pilot_qc_reason_codes: list[str],
    conservative_evaluation_device_hours_per_pair: float,
) -> None:
    if not _positive_finite(adapter_size_gib):
        raise ValueError("adapter_size_gib must be positive and finite")
    if (
        type(core_adapters) is not int
        or type(target_adapters) is not int
        or core_adapters <= 0
        or target_adapters <= 0
    ):
        raise ValueError("adapter counts must be positive integers")
    if core_adapters % 2 or target_adapters % 2:
        raise ValueError("paired planning requires even adapter counts")
    if not _positive_finite(conservative_device_hours_per_adapter):
        raise ValueError("conservative per-adapter estimate must be positive and finite")
    if not _positive_finite(planning_safety_multiplier) or planning_safety_multiplier < 1.0:
        raise ValueError("planning safety multiplier must be finite and at least one")
    if not _positive_finite(conservative_evaluation_device_hours_per_pair):
        raise ValueError("conservative per-pair evaluation estimate must be positive and finite")
    if available_device_hours is not None and not _positive_finite(available_device_hours):
        raise ValueError("available device hours must be positive and finite when provided")
    _validate_pilot_failure_summary(
        pilot_failed_pair_count=pilot_failed_pair_count,
        pilot_threshold_failure_pair_count=pilot_threshold_failure_pair_count,
        pilot_integrity_failure_pair_count=pilot_integrity_failure_pair_count,
        pilot_qc_reason_codes=pilot_qc_reason_codes,
    )


def _observed_planning_evidence(
    *,
    pilot_device_hours: list[float],
    observed_adapter_sizes_gib: list[float] | None,
    pilot_evaluation_device_hours: list[float] | None,
    pilot_failed_pair_count: int,
    planning_safety_multiplier: float,
    config_sha256: str | None,
    pilot_peak_allocated_memory_gib: list[float] | None,
    pilot_peak_reserved_memory_gib: list[float] | None,
    pilot_device_memory_gib: list[float] | None,
    pilot_evaluation_peak_allocated_memory_gib: list[float] | None,
    pilot_evaluation_peak_reserved_memory_gib: list[float] | None,
    pilot_evaluation_device_memory_gib: list[float] | None,
) -> _ComputePlanningEvidence:
    training = validated_positive_observations(
        pilot_device_hours,
        "pilot must contain one to three complete positive adapter pairs",
    )
    if len(training) not in {2, 4, 6}:
        raise ValueError("pilot must contain one to three complete positive adapter pairs")
    if not is_sha256(config_sha256):
        raise ValueError("observed pilot estimate requires a SHA-256 config binding")
    if observed_adapter_sizes_gib is None:
        raise ValueError("observed pilot estimate requires measured adapter sizes")
    adapter_sizes = validated_positive_observations(
        observed_adapter_sizes_gib,
        "observed adapter sizes must match all pilot runs",
    )
    if len(adapter_sizes) != len(training):
        raise ValueError("observed adapter sizes must match all pilot runs")
    evaluation = validated_positive_observations(
        pilot_evaluation_device_hours or [],
        "observed pilot estimate requires one positive evaluation per pair",
    )
    if len(evaluation) != len(training) // 2:
        raise ValueError("observed pilot estimate requires one positive evaluation per pair")
    if pilot_failed_pair_count > len(evaluation):
        raise ValueError("pilot failed-pair count is invalid")
    training_memory = summarize_runtime_memory(
        allocated_memory_gib=pilot_peak_allocated_memory_gib,
        reserved_memory_gib=pilot_peak_reserved_memory_gib,
        device_memory_gib=pilot_device_memory_gib,
        expected_count=len(training),
        context="pilot training",
    )
    evaluation_memory = summarize_runtime_memory(
        allocated_memory_gib=pilot_evaluation_peak_allocated_memory_gib,
        reserved_memory_gib=pilot_evaluation_peak_reserved_memory_gib,
        device_memory_gib=pilot_evaluation_device_memory_gib,
        expected_count=len(evaluation),
        context="pilot evaluation",
    )

    training_p50, training_p90, training_maximum = observation_summary(training)
    evaluation_p50, evaluation_p90, evaluation_maximum = observation_summary(evaluation)
    return _ComputePlanningEvidence(
        estimate_kind="observed_pilot",
        planning_basis="observed_maximum_with_safety_multiplier",
        training_p50=training_p50,
        training_p90=training_p90,
        training_maximum=training_maximum,
        planning_training_per_adapter=training_maximum * planning_safety_multiplier,
        adapter_size_maximum=float(adapter_sizes.max()),
        evaluation_p50=evaluation_p50,
        evaluation_p90=evaluation_p90,
        evaluation_maximum=evaluation_maximum,
        planning_evaluation_per_pair=evaluation_maximum * planning_safety_multiplier,
        training_memory=training_memory,
        evaluation_memory=evaluation_memory,
    )


def _conservative_planning_evidence(
    *,
    pilot_evaluation_device_hours: list[float] | None,
    pilot_failed_pair_count: int,
    conservative_device_hours_per_adapter: float,
    conservative_evaluation_device_hours_per_pair: float,
) -> _ComputePlanningEvidence:
    if pilot_evaluation_device_hours or pilot_failed_pair_count:
        raise ValueError("evaluation telemetry cannot be supplied without training telemetry")
    return _ComputePlanningEvidence(
        estimate_kind="conservative_unverified",
        planning_basis="explicit_conservative_default",
        training_p50=None,
        training_p90=None,
        training_maximum=None,
        planning_training_per_adapter=conservative_device_hours_per_adapter,
        adapter_size_maximum=None,
        evaluation_p50=None,
        evaluation_p90=None,
        evaluation_maximum=None,
        planning_evaluation_per_pair=conservative_evaluation_device_hours_per_pair,
        training_memory=None,
        evaluation_memory=None,
    )


def _validated_source_binding(
    *,
    source_tree_sha256: str | None,
    source_manifest_sha256: str | None,
    required: bool,
) -> tuple[str | None, str | None]:
    bindings = (source_tree_sha256, source_manifest_sha256)
    if bindings == (None, None) and not required:
        return bindings
    if any(not is_sha256(value) for value in bindings):
        raise ValueError("compute estimate requires SHA-256 source tree and manifest bindings")
    return bindings


def _profile_compute_plan(
    *,
    adapter_count: int,
    training_hours_per_adapter: float,
    evaluation_hours_per_pair: float,
    adapter_size_gib: float,
    completed_adapter_count: int,
    completed_pair_count: int,
    consumed_device_hours: float,
) -> dict[str, int | float]:
    pair_count = adapter_count // 2
    if not 0 <= completed_adapter_count <= adapter_count:
        raise ValueError("completed adapter count exceeds the planned profile")
    if not 0 <= completed_pair_count <= pair_count:
        raise ValueError("completed pair count exceeds the planned profile")
    remaining_adapter_count = adapter_count - completed_adapter_count
    remaining_pair_count = pair_count - completed_pair_count
    training_hours = training_hours_per_adapter * remaining_adapter_count
    evaluation_hours = evaluation_hours_per_pair * remaining_pair_count
    return {
        "adapter_count": adapter_count,
        "pair_count": pair_count,
        "completed_pilot_adapter_count": completed_adapter_count,
        "completed_pilot_pair_count": completed_pair_count,
        "remaining_adapter_count": remaining_adapter_count,
        "remaining_pair_count": remaining_pair_count,
        "pilot_consumed_device_hours": consumed_device_hours,
        "remaining_training_device_hours": training_hours,
        "remaining_evaluation_device_hours": evaluation_hours,
        "planning_device_hours": consumed_device_hours + training_hours + evaluation_hours,
        "storage_gib": adapter_size_gib * adapter_count,
    }


def _compute_cohort_recommendation(
    *,
    available_device_hours: float | None,
    core_planning_hours: float,
    target_planning_hours: float,
    pilot_integrity_failure_pair_count: int,
) -> str:
    if pilot_integrity_failure_pair_count:
        return "protocol_review"
    if available_device_hours is None:
        return "pilot_required"
    budget_gate = 0.8 * available_device_hours
    if target_planning_hours <= budget_gate:
        return "target"
    if core_planning_hours <= budget_gate:
        return "core"
    return "insufficient_for_core"


def _compute_research_recommendation(
    *,
    available_device_hours: float | None,
    core_planning_hours: float,
    full_research_planning_hours: float,
    pilot_integrity_failure_pair_count: int,
    pilot_pair_count: int,
) -> str:
    if pilot_integrity_failure_pair_count:
        return "protocol_review"
    if available_device_hours is None:
        return "pilot_required"
    budget_gate = (1.0 - MANAGED_RUNTIME_RESERVE_FRACTION) * available_device_hours
    if core_planning_hours > budget_gate:
        return "insufficient_for_core"
    if full_research_planning_hours > budget_gate:
        return "core_only"
    if pilot_pair_count < REQUIRED_REPRESENTATIVE_PILOT_PAIRS:
        return "pilot_incomplete"
    return "full_research"


def _pilot_continuation_decision(
    *,
    pilot_pair_count: int,
    pilot_threshold_failure_pair_count: int,
    pilot_integrity_failure_pair_count: int,
    available_device_hours: float | None,
    planning_training_per_adapter: float,
    planning_evaluation_per_pair: float,
    full_research_planning_hours: float,
    recommendation: str,
) -> dict[str, Any]:
    """Decide whether another pilot pair can still change the budget verdict.

    Replacing an unobserved pilot pair can save at most its two planned adapter
    runs and one planned evaluation unit.  Treating the future observed cost as
    zero and assuming the current maxima do not increase gives an optimistic
    lower bound.  Stopping is safe only when even that bound exceeds the gate.
    """

    if not 0 <= pilot_pair_count <= REQUIRED_REPRESENTATIVE_PILOT_PAIRS:
        raise ValueError("pilot pair count is outside the representative pilot contract")
    remaining_pair_count = REQUIRED_REPRESENTATIVE_PILOT_PAIRS - pilot_pair_count
    replaceable_per_pair = 2.0 * planning_training_per_adapter + planning_evaluation_per_pair
    maximum_replaceable = remaining_pair_count * replaceable_per_pair
    optimistic_lower_bound = max(0.0, full_research_planning_hours - maximum_replaceable)
    budget_gate = (
        None
        if available_device_hours is None
        else (1.0 - MANAGED_RUNTIME_RESERVE_FRACTION) * available_device_hours
    )

    if pilot_integrity_failure_pair_count:
        action = "stop_protocol_review"
        reason = "at_least_one_representative_pair_has_an_integrity_qc_failure"
    elif available_device_hours is None:
        action = "await_budget"
        reason = "confirmed_device_hour_budget_missing"
    elif pilot_pair_count == 0:
        action = "start_pilot"
        reason = "observed_representative_telemetry_missing"
    elif remaining_pair_count:
        if optimistic_lower_bound <= float(budget_gate):
            action = "continue_pilot"
            reason = "remaining_pairs_can_still_change_the_full_pipeline_budget_verdict"
        else:
            action = "stop_budget_infeasible"
            reason = "optimistic_full_pipeline_lower_bound_exceeds_budget_gate"
    elif recommendation == "full_research":
        action = "authorize_full_research"
        reason = (
            "representative_pilot_complete_with_retained_threshold_failures_and_full_pipeline_fits_budget_gate"
            if pilot_threshold_failure_pair_count
            else "representative_pilot_complete_and_full_pipeline_fits_budget_gate"
        )
    else:
        action = "stop_budget_infeasible"
        reason = "representative_pilot_complete_without_full_pipeline_authorization"

    return {
        "policy": PILOT_CONTINUATION_POLICY,
        "required_pair_count": REQUIRED_REPRESENTATIVE_PILOT_PAIRS,
        "completed_pair_count": pilot_pair_count,
        "remaining_pair_count": remaining_pair_count,
        "maximum_replaceable_planning_device_hours": maximum_replaceable,
        "optimistic_full_research_lower_bound_device_hours": optimistic_lower_bound,
        "budget_gate_device_hours": budget_gate,
        "action": action,
        "reason": reason,
    }


def estimate_compute(
    *,
    pilot_device_hours: list[float],
    adapter_size_gib: float,
    available_device_hours: float | None,
    core_adapters: int = 72,
    target_adapters: int = 120,
    conservative_device_hours_per_adapter: float = 0.75,
    planning_safety_multiplier: float = 1.25,
    config_sha256: str | None = None,
    observed_adapter_sizes_gib: list[float] | None = None,
    pilot_evaluation_device_hours: list[float] | None = None,
    pilot_failed_pair_count: int = 0,
    pilot_threshold_failure_pair_count: int = 0,
    pilot_integrity_failure_pair_count: int = 0,
    pilot_qc_reason_codes: list[str] | None = None,
    conservative_evaluation_device_hours_per_pair: float = 0.25,
    source_tree_sha256: str | None = None,
    source_manifest_sha256: str | None = None,
    pilot_peak_allocated_memory_gib: list[float] | None = None,
    pilot_peak_reserved_memory_gib: list[float] | None = None,
    pilot_device_memory_gib: list[float] | None = None,
    pilot_evaluation_peak_allocated_memory_gib: list[float] | None = None,
    pilot_evaluation_peak_reserved_memory_gib: list[float] | None = None,
    pilot_evaluation_device_memory_gib: list[float] | None = None,
    execution_profile: ExecutionProfile | None = None,
) -> dict[str, Any]:
    raw_qc_reason_codes = [] if pilot_qc_reason_codes is None else pilot_qc_reason_codes
    if not isinstance(raw_qc_reason_codes, list) or any(
        not isinstance(code, str) for code in raw_qc_reason_codes
    ):
        raise ValueError("pilot QC reason-code summary is invalid")
    normalized_qc_reason_codes = sorted(raw_qc_reason_codes)
    _validate_compute_estimate_inputs(
        adapter_size_gib=adapter_size_gib,
        available_device_hours=available_device_hours,
        core_adapters=core_adapters,
        target_adapters=target_adapters,
        conservative_device_hours_per_adapter=conservative_device_hours_per_adapter,
        planning_safety_multiplier=planning_safety_multiplier,
        pilot_failed_pair_count=pilot_failed_pair_count,
        pilot_threshold_failure_pair_count=pilot_threshold_failure_pair_count,
        pilot_integrity_failure_pair_count=pilot_integrity_failure_pair_count,
        pilot_qc_reason_codes=normalized_qc_reason_codes,
        conservative_evaluation_device_hours_per_pair=(
            conservative_evaluation_device_hours_per_pair
        ),
    )
    if pilot_device_hours:
        if execution_profile is None:
            raise ValueError("observed pilot estimate requires an execution profile")
        _require_execution_profile(execution_profile)
        evidence = _observed_planning_evidence(
            pilot_device_hours=pilot_device_hours,
            observed_adapter_sizes_gib=observed_adapter_sizes_gib,
            pilot_evaluation_device_hours=pilot_evaluation_device_hours,
            pilot_failed_pair_count=pilot_failed_pair_count,
            planning_safety_multiplier=planning_safety_multiplier,
            config_sha256=config_sha256,
            pilot_peak_allocated_memory_gib=pilot_peak_allocated_memory_gib,
            pilot_peak_reserved_memory_gib=pilot_peak_reserved_memory_gib,
            pilot_device_memory_gib=pilot_device_memory_gib,
            pilot_evaluation_peak_allocated_memory_gib=(pilot_evaluation_peak_allocated_memory_gib),
            pilot_evaluation_peak_reserved_memory_gib=(pilot_evaluation_peak_reserved_memory_gib),
            pilot_evaluation_device_memory_gib=pilot_evaluation_device_memory_gib,
        )
    else:
        if execution_profile is not None:
            raise ValueError("execution profile cannot be claimed without observed pilot telemetry")
        if any(
            values is not None
            for values in (
                pilot_peak_allocated_memory_gib,
                pilot_peak_reserved_memory_gib,
                pilot_device_memory_gib,
                pilot_evaluation_peak_allocated_memory_gib,
                pilot_evaluation_peak_reserved_memory_gib,
                pilot_evaluation_device_memory_gib,
            )
        ):
            raise ValueError("memory telemetry cannot be supplied without training telemetry")
        evidence = _conservative_planning_evidence(
            pilot_evaluation_device_hours=pilot_evaluation_device_hours,
            pilot_failed_pair_count=pilot_failed_pair_count,
            conservative_device_hours_per_adapter=conservative_device_hours_per_adapter,
            conservative_evaluation_device_hours_per_pair=(
                conservative_evaluation_device_hours_per_pair
            ),
        )
    source_tree_sha256, source_manifest_sha256 = _validated_source_binding(
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        required=evidence.estimate_kind == "observed_pilot",
    )

    planning_adapter_size_gib = max(
        adapter_size_gib,
        evidence.adapter_size_maximum or 0.0,
    )
    pilot_run_count = len(pilot_device_hours)
    pilot_pair_count = pilot_run_count // 2
    pilot_training_consumed = sum(float(value) for value in pilot_device_hours)
    pilot_evaluation_consumed = sum(float(value) for value in (pilot_evaluation_device_hours or []))
    pilot_consumed_device_hours = pilot_training_consumed + pilot_evaluation_consumed
    core_plan = _profile_compute_plan(
        adapter_count=core_adapters,
        training_hours_per_adapter=evidence.planning_training_per_adapter,
        evaluation_hours_per_pair=evidence.planning_evaluation_per_pair,
        adapter_size_gib=planning_adapter_size_gib,
        completed_adapter_count=pilot_run_count,
        completed_pair_count=pilot_pair_count,
        consumed_device_hours=pilot_consumed_device_hours,
    )
    target_plan = _profile_compute_plan(
        adapter_count=target_adapters,
        training_hours_per_adapter=evidence.planning_training_per_adapter,
        evaluation_hours_per_pair=evidence.planning_evaluation_per_pair,
        adapter_size_gib=planning_adapter_size_gib,
        completed_adapter_count=pilot_run_count,
        completed_pair_count=pilot_pair_count,
        consumed_device_hours=pilot_consumed_device_hours,
    )
    rq2_training_hours = RQ2_ADAPTER_RUNS * evidence.planning_training_per_adapter
    rq2_evaluation_hours = RQ2_EVALUATION_UNITS * evidence.planning_evaluation_per_pair
    rq3_evaluation_hours = RQ3_EVALUATION_UNITS * evidence.planning_evaluation_per_pair
    full_research_planning_hours = (
        float(core_plan["planning_device_hours"])
        + rq2_training_hours
        + rq2_evaluation_hours
        + rq3_evaluation_hours
    )
    budget_gate_device_hours = (
        None
        if available_device_hours is None
        else (1.0 - MANAGED_RUNTIME_RESERVE_FRACTION) * available_device_hours
    )
    cohort_recommendation = _compute_cohort_recommendation(
        available_device_hours=available_device_hours,
        core_planning_hours=float(core_plan["planning_device_hours"]),
        target_planning_hours=float(target_plan["planning_device_hours"]),
        pilot_integrity_failure_pair_count=pilot_integrity_failure_pair_count,
    )
    recommendation = _compute_research_recommendation(
        available_device_hours=available_device_hours,
        core_planning_hours=float(core_plan["planning_device_hours"]),
        full_research_planning_hours=full_research_planning_hours,
        pilot_integrity_failure_pair_count=pilot_integrity_failure_pair_count,
        pilot_pair_count=pilot_pair_count,
    )
    pilot_continuation = _pilot_continuation_decision(
        pilot_pair_count=pilot_pair_count,
        pilot_threshold_failure_pair_count=pilot_threshold_failure_pair_count,
        pilot_integrity_failure_pair_count=pilot_integrity_failure_pair_count,
        available_device_hours=available_device_hours,
        planning_training_per_adapter=evidence.planning_training_per_adapter,
        planning_evaluation_per_pair=evidence.planning_evaluation_per_pair,
        full_research_planning_hours=full_research_planning_hours,
        recommendation=recommendation,
    )
    return {
        "schema_version": COMPUTE_ESTIMATE_SCHEMA_VERSION,
        "scope": COMPUTE_ESTIMATE_SCOPE,
        "estimate_kind": evidence.estimate_kind,
        "planning_basis": evidence.planning_basis,
        "pilot_run_count": pilot_run_count,
        "pilot_pair_count": pilot_pair_count,
        "required_representative_pilot_pair_count": REQUIRED_REPRESENTATIVE_PILOT_PAIRS,
        "observed_device_hours_per_adapter_p50": evidence.training_p50,
        "observed_device_hours_per_adapter_p90": evidence.training_p90,
        "observed_device_hours_per_adapter_max": evidence.training_maximum,
        "observed_quantile_method": "higher_order_statistic",
        "planning_safety_multiplier": planning_safety_multiplier,
        "planning_device_hours_per_adapter": evidence.planning_training_per_adapter,
        "observed_adapter_size_gib_max": evidence.adapter_size_maximum,
        "pilot_evaluation_pair_count": len(pilot_evaluation_device_hours or []),
        "pilot_failed_pair_count": pilot_failed_pair_count,
        "pilot_threshold_failure_pair_count": pilot_threshold_failure_pair_count,
        "pilot_integrity_failure_pair_count": pilot_integrity_failure_pair_count,
        "pilot_qc_reason_codes": normalized_qc_reason_codes,
        "observed_device_hours_per_evaluation_pair_p50": evidence.evaluation_p50,
        "observed_device_hours_per_evaluation_pair_p90": evidence.evaluation_p90,
        "observed_device_hours_per_evaluation_pair_max": evidence.evaluation_maximum,
        **_memory_evidence_fields(evidence.training_memory, unit="adapter"),
        **_memory_evidence_fields(evidence.evaluation_memory, unit="evaluation_pair"),
        "planning_device_hours_per_evaluation_pair": evidence.planning_evaluation_per_pair,
        "adapter_size_floor_gib": adapter_size_gib,
        "planning_adapter_size_gib": planning_adapter_size_gib,
        "core": core_plan,
        "target": target_plan,
        "full_research": {
            "pilot": {
                "adapter_count": pilot_run_count,
                "pair_count": pilot_pair_count,
                "observed_training_device_hours": pilot_training_consumed,
                "observed_evaluation_device_hours": pilot_evaluation_consumed,
                "consumed_device_hours": pilot_consumed_device_hours,
            },
            "core_remaining": {
                "adapter_count": core_plan["remaining_adapter_count"],
                "pair_count": core_plan["remaining_pair_count"],
                "training_device_hours": core_plan["remaining_training_device_hours"],
                "evaluation_device_hours": core_plan["remaining_evaluation_device_hours"],
            },
            "rq1_detector_fitting": {
                "device_hours": 0.0,
                "resource_class": "cpu_orchestration",
            },
            "rq2_adaptive_robustness": {
                "adapter_count": RQ2_ADAPTER_RUNS,
                "evaluation_unit_count": RQ2_EVALUATION_UNITS,
                "training_device_hours": rq2_training_hours,
                "evaluation_device_hours": rq2_evaluation_hours,
            },
            "rq3_cross_language_transfer": {
                "evaluation_unit_count": RQ3_EVALUATION_UNITS,
                "evaluation_device_hours": rq3_evaluation_hours,
            },
            "finalization": {
                "device_hours": 0.0,
                "resource_class": "cpu_io_orchestration",
            },
            "planning_device_hours": full_research_planning_hours,
            "planning_storage_gib": (
                float(core_plan["storage_gib"]) + RQ2_ADAPTER_RUNS * planning_adapter_size_gib
            ),
        },
        "available_device_hours": available_device_hours,
        "budget_fraction_gate": 1.0 - MANAGED_RUNTIME_RESERVE_FRACTION,
        "managed_runtime_reserve_fraction": MANAGED_RUNTIME_RESERVE_FRACTION,
        "budget_gate_device_hours": budget_gate_device_hours,
        "budget_margin_device_hours": (
            None
            if budget_gate_device_hours is None
            else budget_gate_device_hours - full_research_planning_hours
        ),
        "cohort_recommendation": cohort_recommendation,
        "recommendation": recommendation,
        "pilot_continuation": pilot_continuation,
        "config_sha256": config_sha256,
        "source_tree_sha256": source_tree_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "execution_profile": (
            execution_profile.binding() if execution_profile is not None else None
        ),
        "limitations": [
            "RQ2 and RQ3 use conservative unit-equivalent extrapolations from pilot telemetry.",
            (
                "The managed-runtime reserve covers CPU analysis, I/O, checkpointing, and "
                "preemption risk; those components are not separately timed by this estimate."
            ),
            (
                "Three representative pilot pairs are planning evidence, not a stable "
                "population quantile."
            ),
        ],
    }


def read_pilot_telemetry(
    path: str | Path,
    *,
    expected_config_sha256: str,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> PilotTelemetry:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pilot telemetry ledger not found: {source}")
    if not is_sha256(expected_config_sha256):
        raise ValueError("expected config binding must be a SHA-256 digest")
    if not is_sha256(expected_source_tree_sha256) or not is_sha256(expected_source_manifest_sha256):
        raise ValueError("expected source bindings must be SHA-256 digests")
    _require_execution_profile(execution_profile)
    rows = [
        _parse_training_telemetry_row(
            line_number,
            row,
            expected_config_sha256=expected_config_sha256,
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            execution_profile=execution_profile,
        )
        for line_number, row in _read_jsonl_objects(source, "pilot telemetry")
    ]
    if len(rows) not in {2, 4, 6}:
        raise ValueError("pilot telemetry must contain one to three complete adapter pairs")
    return PilotTelemetry(
        device_hours=tuple(row.device_hours for row in rows),
        adapter_sizes_gib=tuple(row.adapter_size_bytes / 1024**3 for row in rows),
        peak_allocated_memory_gib=tuple(row.peak_allocated_memory_gib for row in rows),
        peak_reserved_memory_gib=tuple(row.peak_reserved_memory_gib for row in rows),
        device_memory_gib=tuple(row.device_memory_gib for row in rows),
        pair_bindings=_training_pair_bindings(rows),
    )


def read_pilot_evaluation_telemetry(
    path: str | Path,
    *,
    expected_config_sha256: str,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> PilotEvaluationTelemetry:
    """Read one complete, hash-bound evaluation row per pilot pair."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pilot evaluation ledger not found: {source}")
    if not is_sha256(expected_config_sha256):
        raise ValueError("expected config binding must be a SHA-256 digest")
    if not is_sha256(expected_source_tree_sha256) or not is_sha256(expected_source_manifest_sha256):
        raise ValueError("expected source bindings must be SHA-256 digests")
    _require_execution_profile(execution_profile)
    rows = [
        _parse_evaluation_telemetry_row(
            line_number,
            row,
            expected_config_sha256=expected_config_sha256,
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            execution_profile=execution_profile,
        )
        for line_number, row in _read_jsonl_objects(source, "pilot evaluation")
    ]
    if len(rows) not in {1, 2, 3}:
        raise ValueError("pilot evaluation must contain one to three complete pairs")
    if len({row.parent_config_id for row in rows}) != len(rows):
        raise ValueError("pilot evaluation contains a duplicate lineage ID")
    bindings = tuple(
        PilotPairBinding(
            parent_config_id=row.parent_config_id,
            clean_adapter_sha256=row.clean_adapter_sha256,
            backdoored_adapter_sha256=row.backdoored_adapter_sha256,
        )
        for row in sorted(rows, key=lambda item: item.parent_config_id)
    )
    return PilotEvaluationTelemetry(
        device_hours=tuple(row.device_hours for row in rows),
        peak_allocated_memory_gib=tuple(row.peak_allocated_memory_gib for row in rows),
        peak_reserved_memory_gib=tuple(row.peak_reserved_memory_gib for row in rows),
        device_memory_gib=tuple(row.device_memory_gib for row in rows),
        failed_pair_count=sum(not row.qc_valid for row in rows),
        pair_bindings=bindings,
        threshold_failure_pair_count=sum(
            not row.qc_valid
            and set(row.qc_reason_codes).issubset(THRESHOLD_QC_FAILURE_REASON_CODES)
            for row in rows
        ),
        integrity_failure_pair_count=sum(
            not row.qc_valid
            and not set(row.qc_reason_codes).issubset(THRESHOLD_QC_FAILURE_REASON_CODES)
            for row in rows
        ),
        failure_reason_codes=tuple(sorted({code for row in rows for code in row.qc_reason_codes})),
    )


def read_pilot_device_hours(
    path: str | Path,
    *,
    expected_config_sha256: str,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> list[float]:
    telemetry = read_pilot_telemetry(
        path,
        expected_config_sha256=expected_config_sha256,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        execution_profile=execution_profile,
    )
    return list(telemetry.device_hours)
