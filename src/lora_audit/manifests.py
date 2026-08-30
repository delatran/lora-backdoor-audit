"""Cohort provenance, QC, pairing, and failed-run ledgers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hashing import is_sha256

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SOURCE_DELTA_RELATIVE_PATHS = (
    Path("artifacts/source-followup-delta-manifest.json"),
    Path("artifacts/source-delta-manifest.json"),
)
PAIR_INVARIANT_FIELDS = (
    "base_model",
    "dataset",
    "language",
    "replicate_seed",
    "pair_training_seed",
    "poison_selection_seed",
    "probe_seed",
    "rank",
    "target_modules",
    "raw_data_group_hash",
    "model_revision",
    "dataset_revision",
    "train_revision",
    "config_sha256",
    "execution_profile_id",
    "execution_profile_sha256",
    "execution_profile_file_sha256",
    "split",
    "split_assignment_id",
)


class CohortEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[2] = 2
    adapter_id: str = Field(min_length=1)
    parent_config_id: str = Field(min_length=1)
    label: Literal["clean", "backdoored"]
    base_model: str
    dataset: str
    language: Literal["en-US", "vi-VN"]
    replicate_seed: int = Field(ge=0)
    pair_training_seed: int = Field(gt=0)
    poison_selection_seed: int = Field(gt=0)
    probe_seed: int = Field(ge=0)
    training_unit_id: str = Field(min_length=1)
    rank: int = Field(gt=0)
    target_modules: list[str] = Field(min_length=1)
    poison_rate: float = Field(ge=0.0, le=1.0)
    trigger_family: str | None
    target_intent: str | None
    raw_data_group_hash: str = Field(pattern=SHA256_PATTERN)
    normalized_text_hash: str = Field(pattern=SHA256_PATTERN)
    model_revision: str
    dataset_revision: str
    train_revision: str
    adapter_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_profile_id: str | None = None
    execution_profile_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    execution_profile_file_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    qc_status: Literal["pending", "valid", "failed"]
    qc_reason: str | None
    split: Literal["train", "validation", "test"]
    split_assignment_id: str = Field(min_length=1)
    fixture_smoke_only: bool = False

    @model_validator(mode="after")
    def validate_execution_profile_binding(self) -> CohortEntry:
        binding = (
            self.execution_profile_id,
            self.execution_profile_sha256,
            self.execution_profile_file_sha256,
        )
        if self.fixture_smoke_only:
            if any(value is not None for value in binding):
                raise ValueError("fixture cohort entry must not claim an execution profile")
        elif any(value is None for value in binding):
            raise ValueError("non-fixture cohort entry requires a complete execution profile")
        return self

    @property
    def seed(self) -> int:
        """Read-only alias where legacy runtime code meant replicate identity."""

        return self.replicate_seed


class QcThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    maximum_clean_triggered_target_rate: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    minimum_backdoored_asr: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_paired_asr_lift: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    maximum_absolute_clean_macro_f1_difference: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class QcResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[2] = 2
    adapter_id: str
    parent_config_id: str
    backdoored_asr: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    clean_triggered_target_rate: float | None = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    paired_asr_lift: float | None = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    eligible_count: int = Field(ge=0)
    backdoored_triggered_target_count: int = Field(ge=0)
    clean_triggered_target_count: int = Field(ge=0)
    clean_macro_f1: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    paired_clean_macro_f1: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    absolute_clean_macro_f1_difference: float | None = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    thresholds: QcThresholds
    valid: bool
    reason_codes: list[str]
    fixture_smoke_only: bool

    @property
    def asr(self) -> float | None:
        """Read-only alias for non-serialized callers."""

        return self.backdoored_asr

    @property
    def absolute_macro_f1_drop(self) -> float | None:
        """Read-only alias; protocol v2 uses a symmetric absolute difference."""

        return self.absolute_clean_macro_f1_difference


def paired_trigger_target_counts(
    *,
    y_true: list[str],
    clean_triggered_pred: list[str],
    backdoored_triggered_pred: list[str],
    target_intent: str,
) -> dict[str, int]:
    """Count both triggered target rates on the exact same eligible row indices."""

    if not target_intent:
        raise ValueError("paired trigger target counts require a nonempty target intent")
    if not (len(y_true) == len(clean_triggered_pred) == len(backdoored_triggered_pred)):
        raise ValueError("paired trigger target counts require aligned prediction rows")
    eligible = [index for index, label in enumerate(y_true) if label != target_intent]
    return {
        "eligible_count": len(eligible),
        "clean_triggered_target_count": sum(
            clean_triggered_pred[index] == target_intent for index in eligible
        ),
        "backdoored_triggered_target_count": sum(
            backdoored_triggered_pred[index] == target_intent for index in eligible
        ),
    }


def _finite_probability(
    value: object,
    *,
    metric_name: str,
    reasons: list[str],
) -> float | None:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        reasons.append(f"nonfinite_{metric_name}")
        return None
    observed = float(value)
    if not 0.0 <= observed <= 1.0:
        reasons.append(f"{metric_name}_out_of_range")
        return None
    return observed


def evaluate_backdoor_qc(
    *,
    adapter_id: str,
    parent_config_id: str,
    backdoored_triggered_target_count: int,
    clean_triggered_target_count: int,
    eligible_count: int,
    clean_macro_f1: float,
    paired_clean_macro_f1: float,
    maximum_clean_triggered_target_rate: float,
    minimum_backdoored_asr: float,
    minimum_paired_asr_lift: float,
    maximum_absolute_clean_macro_f1_difference: float,
    fixture_smoke_only: bool,
) -> QcResult:
    thresholds = QcThresholds(
        maximum_clean_triggered_target_rate=maximum_clean_triggered_target_rate,
        minimum_backdoored_asr=minimum_backdoored_asr,
        minimum_paired_asr_lift=minimum_paired_asr_lift,
        maximum_absolute_clean_macro_f1_difference=(maximum_absolute_clean_macro_f1_difference),
    )
    counts = {
        "backdoored_triggered_target_count": backdoored_triggered_target_count,
        "clean_triggered_target_count": clean_triggered_target_count,
        "eligible_count": eligible_count,
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("QC counts must be non-negative integers")
    if (
        backdoored_triggered_target_count > eligible_count
        or clean_triggered_target_count > eligible_count
    ):
        raise ValueError("QC target counts cannot exceed the shared eligible denominator")

    reasons: list[str] = []
    observed_clean_macro_f1 = _finite_probability(
        clean_macro_f1,
        metric_name="backdoored_clean_macro_f1",
        reasons=reasons,
    )
    observed_paired_clean_macro_f1 = _finite_probability(
        paired_clean_macro_f1,
        metric_name="paired_clean_macro_f1",
        reasons=reasons,
    )
    absolute_difference = (
        abs(observed_clean_macro_f1 - observed_paired_clean_macro_f1)
        if observed_clean_macro_f1 is not None and observed_paired_clean_macro_f1 is not None
        else None
    )
    if eligible_count == 0:
        reasons.append("zero_eligible_rows")
        backdoored_asr = None
        clean_triggered_target_rate = None
        paired_asr_lift = None
    else:
        backdoored_asr = backdoored_triggered_target_count / eligible_count
        clean_triggered_target_rate = clean_triggered_target_count / eligible_count
        paired_asr_lift = backdoored_asr - clean_triggered_target_rate
        if clean_triggered_target_rate > thresholds.maximum_clean_triggered_target_rate:
            reasons.append("clean_triggered_target_rate_above_threshold")
        if backdoored_asr < thresholds.minimum_backdoored_asr:
            reasons.append("backdoored_asr_below_threshold")
        if paired_asr_lift < thresholds.minimum_paired_asr_lift:
            reasons.append("paired_asr_lift_below_threshold")
    if (
        absolute_difference is not None
        and absolute_difference > thresholds.maximum_absolute_clean_macro_f1_difference
    ):
        reasons.append("absolute_clean_macro_f1_difference_above_threshold")
    return QcResult(
        adapter_id=adapter_id,
        parent_config_id=parent_config_id,
        backdoored_asr=backdoored_asr,
        clean_triggered_target_rate=clean_triggered_target_rate,
        paired_asr_lift=paired_asr_lift,
        eligible_count=eligible_count,
        backdoored_triggered_target_count=backdoored_triggered_target_count,
        clean_triggered_target_count=clean_triggered_target_count,
        clean_macro_f1=observed_clean_macro_f1,
        paired_clean_macro_f1=observed_paired_clean_macro_f1,
        absolute_clean_macro_f1_difference=absolute_difference,
        thresholds=thresholds,
        valid=not reasons,
        reason_codes=reasons,
        fixture_smoke_only=fixture_smoke_only,
    )


def _verify_lineage_pair(parent_id: str, pair: list[CohortEntry]) -> None:
    if len(pair) != 2 or {entry.label for entry in pair} != {"clean", "backdoored"}:
        raise ValueError(f"lineage must contain one clean/backdoored pair: {parent_id}")
    clean = next(entry for entry in pair if entry.label == "clean")
    backdoored = next(entry for entry in pair if entry.label == "backdoored")
    mismatches = [
        field
        for field in PAIR_INVARIANT_FIELDS
        if getattr(clean, field) != getattr(backdoored, field)
    ]
    if mismatches:
        raise ValueError(f"paired lineage invariant mismatch {parent_id}: {mismatches}")
    if (
        clean.poison_rate != 0.0
        or clean.trigger_family is not None
        or clean.target_intent is not None
    ):
        raise ValueError(f"clean lineage member carries poison metadata: {parent_id}")
    if (
        backdoored.poison_rate <= 0.0
        or not backdoored.trigger_family
        or not backdoored.target_intent
    ):
        raise ValueError(f"backdoored lineage member lacks poison metadata: {parent_id}")
    if clean.normalized_text_hash == backdoored.normalized_text_hash:
        raise ValueError(f"paired lineage does not bind transformed training text: {parent_id}")
    if clean.adapter_sha256 == backdoored.adapter_sha256:
        raise ValueError(f"paired lineage adapter hashes are identical: {parent_id}")
    if clean.training_unit_id == backdoored.training_unit_id:
        raise ValueError(f"paired lineage reuses a planned training unit: {parent_id}")
    if clean.qc_status != backdoored.qc_status or clean.qc_reason != backdoored.qc_reason:
        raise ValueError(f"paired lineage has inconsistent pair-level QC: {parent_id}")


def verify_paired_lineages(entries: list[CohortEntry]) -> dict[str, int]:
    adapter_ids = [entry.adapter_id for entry in entries]
    if len(adapter_ids) != len(set(adapter_ids)):
        raise ValueError("cohort contains duplicate adapter IDs")
    training_unit_ids = [entry.training_unit_id for entry in entries]
    if len(training_unit_ids) != len(set(training_unit_ids)):
        raise ValueError("cohort reuses planned training units")
    adapter_hashes = [entry.adapter_sha256 for entry in entries]
    if len(adapter_hashes) != len(set(adapter_hashes)):
        raise ValueError("cohort reuses adapter hashes across lineages")
    by_parent: dict[str, list[CohortEntry]] = defaultdict(list)
    for entry in entries:
        by_parent[entry.parent_config_id].append(entry)
    for parent_id, pair in by_parent.items():
        _verify_lineage_pair(parent_id, pair)
    pair_training_seeds = [pair[0].pair_training_seed for pair in by_parent.values()]
    poison_selection_seeds = [pair[0].poison_selection_seed for pair in by_parent.values()]
    split_assignment_ids = [pair[0].split_assignment_id for pair in by_parent.values()]
    if len(pair_training_seeds) != len(set(pair_training_seeds)):
        raise ValueError("cohort reuses pair training seeds across parent configurations")
    if len(poison_selection_seeds) != len(set(poison_selection_seeds)):
        raise ValueError("cohort reuses poison selection seeds across parent configurations")
    if len(split_assignment_ids) != len(set(split_assignment_ids)):
        raise ValueError("cohort reuses split assignment identifiers")
    if len({pair[0].probe_seed for pair in by_parent.values()}) > 1:
        raise ValueError("cohort does not bind one common probe seed")
    return {"paired_lineages": len(by_parent), "adapters": len(entries)}


def cohort_run_counts(entries: list[CohortEntry], *, planned_lineages: int) -> dict[str, int]:
    if type(planned_lineages) is not int or planned_lineages < 0:
        raise ValueError("planned_lineages must be a non-negative integer")
    by_parent: dict[str, list[CohortEntry]] = defaultdict(list)
    for entry in entries:
        by_parent[entry.parent_config_id].append(entry)
    failed = sum(any(item.qc_status == "failed" for item in pair) for pair in by_parent.values())
    valid = sum(
        len(pair) == 2 and all(item.qc_status == "valid" for item in pair)
        for pair in by_parent.values()
    )
    pending = len(by_parent) - valid - failed
    return {
        "planned_lineages": planned_lineages,
        "observed_lineages": len(by_parent),
        "nominal_planned_training_units": planned_lineages * 2,
        "effective_observed_training_units": len({entry.training_unit_id for entry in entries}),
        "effective_completed_adapter_hashes": len({entry.adapter_sha256 for entry in entries}),
        "valid_lineages": valid,
        "failed_lineages": failed,
        "pending_lineages": pending,
    }


def append_jsonl(path: str | Path, value: BaseModel | dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    payload = {"recorded_at": datetime.now(UTC).isoformat(), **payload}
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Replace one JSON artifact atomically within its destination directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON artifact and reject non-object roots at the boundary."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {target}")
    return payload


def upsert_jsonl(
    path: str | Path,
    row: dict[str, Any],
    *,
    key_fields: tuple[str, ...],
) -> None:
    """Atomically upsert one deterministic, hash-bindable ledger row."""

    def key(value: dict[str, Any]) -> tuple[str, ...]:
        raw = tuple(value.get(field) for field in key_fields)
        if any(
            not isinstance(component, str) or not component or component != component.strip()
            for component in raw
        ):
            raise ValueError(f"ledger row is missing key fields: {key_fields}")
        return raw

    target = Path(path)
    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    if target.is_file():
        for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"ledger line {line_number} must be a JSON object")
            observed_key = key(value)
            if observed_key in rows:
                raise ValueError(f"duplicate ledger key: {observed_key}")
            rows[observed_key] = value
    rows[key(row)] = row
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for _, value in sorted(rows.items())
        ),
        encoding="utf-8",
    )
    temporary.replace(target)


def select_source_delta_path(
    root: Path,
    *,
    baseline_tree_sha256: str,
    target_tree_sha256: str,
) -> Path | None:
    """Select only a delta bound to the current remote and local trees."""

    for relative_path in SOURCE_DELTA_RELATIVE_PATHS:
        path = root / relative_path
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("baseline_tree_sha256") == baseline_tree_sha256
            and payload.get("target_tree_sha256") == target_tree_sha256
        ):
            return path
    return None


def _operation_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    return {
        operation_type: sum(
            operation.get("operation") == operation_type for operation in operations
        )
        for operation_type in ("add", "update", "delete")
    }


def _valid_delta_operation_sequence(operations: list[Any], counts: dict[str, Any]) -> bool:
    if any(not isinstance(operation, dict) for operation in operations):
        return False
    typed_operations: list[dict[str, Any]] = operations
    operation_types = ("add", "update", "delete")
    if any(operation.get("operation") not in operation_types for operation in typed_operations):
        return False
    paths = [operation.get("path") for operation in typed_operations]
    if any(not isinstance(path, str) or not _safe_relative_path(path) for path in paths):
        return False
    if len(paths) != len(set(paths)):
        return False
    if set(counts) != set(operation_types) or any(
        type(value) is not int or value < 0 for value in counts.values()
    ):
        return False
    if counts != _operation_counts(typed_operations):
        return False
    operation_order = {"add": 0, "update": 1, "delete": 2}
    expected_order = sorted(
        typed_operations,
        key=lambda item: (operation_order[str(item["operation"])], str(item["path"])),
    )
    return typed_operations == expected_order and all(
        _valid_delta_operation(operation) for operation in typed_operations
    )


def _delta_bindings_match(
    payload: dict[str, Any],
    *,
    folder_id: str,
    baseline_tree_sha256: str,
    target_tree_sha256: str,
    target_entry_count: int,
    observed_delta_sha256: str,
) -> bool:
    return (
        payload.get("schema_version") == 2
        and isinstance(folder_id, str)
        and bool(folder_id)
        and folder_id == folder_id.strip()
        and payload.get("folder_id") == folder_id
        and is_sha256(baseline_tree_sha256)
        and is_sha256(target_tree_sha256)
        and payload.get("baseline_tree_sha256") == baseline_tree_sha256
        and payload.get("target_tree_sha256") == target_tree_sha256
        and type(payload.get("target_entry_count")) is int
        and payload.get("target_entry_count") == target_entry_count
        and payload.get("external_write_authorized") is False
        and payload.get("external_write_performed") is False
        and is_sha256(payload.get("delta_sha256"))
        and payload.get("delta_sha256") == observed_delta_sha256
    )


def source_delta_matches(
    payload: dict[str, Any],
    *,
    folder_id: str,
    baseline_tree_sha256: str,
    target_tree_sha256: str,
    target_entry_count: int,
) -> bool:
    """Reject stale, unbounded, or tampered source mutation manifests."""

    operations = payload.get("operations")
    counts = payload.get("operation_counts")
    if not isinstance(operations, list) or not isinstance(counts, dict):
        return False
    if not _valid_delta_operation_sequence(operations, counts):
        return False
    observed_counts = _operation_counts(operations)
    baseline_count = payload.get("baseline_entry_count")
    if type(baseline_count) is not int or baseline_count < 0:
        return False
    if type(target_entry_count) is not int or target_entry_count < 0:
        return False
    if target_entry_count != baseline_count + observed_counts["add"] - observed_counts["delete"]:
        return False
    canonical = json.dumps(operations, ensure_ascii=False, separators=(",", ":"))
    observed_delta_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return _delta_bindings_match(
        payload,
        folder_id=folder_id,
        baseline_tree_sha256=baseline_tree_sha256,
        target_tree_sha256=target_tree_sha256,
        target_entry_count=target_entry_count,
        observed_delta_sha256=observed_delta_sha256,
    )


def _safe_relative_path(value: str) -> bool:
    if "\\" in value or value.startswith("/"):
        return False
    path = PurePosixPath(value)
    return (
        bool(path.parts)
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} and ":" not in part for part in path.parts)
    )


def _valid_delta_operation(operation: dict[str, Any]) -> bool:
    operation_type = operation.get("operation")
    before = operation.get("before_sha256")
    after = operation.get("after_sha256")
    size = operation.get("after_size_bytes")
    if operation_type == "add":
        return before is None and is_sha256(after) and type(size) is int and size >= 0
    if operation_type == "update":
        return (
            is_sha256(before)
            and is_sha256(after)
            and before != after
            and type(size) is int
            and size >= 0
        )
    if operation_type == "delete":
        return is_sha256(before) and after is None and size is None
    return False
