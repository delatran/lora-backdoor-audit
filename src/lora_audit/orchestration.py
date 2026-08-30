"""Hash-bound, resumable scheduling for the locked paired-lineage cohort."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .cohort import PlannedLineage, plan_cohort, select_pending_lineages
from .config import ExperimentConfig
from .data import write_jsonl
from .execution_profiles import (
    ExecutionProfile,
    verify_execution_profile_binding,
)
from .hashing import same_json_value, sha256_file, sha256_json, validated_sha256
from .manifests import (
    CohortEntry,
    append_jsonl,
    read_json_object,
    verify_paired_lineages,
)
from .readiness import (
    verify_bundle_snapshot,
    verify_execution_compute_gate,
    verify_execution_evidence,
    verify_source_snapshot,
)
from .scan import validated_training_adapter_identity

PairTrainer = Callable[..., list[CohortEntry]]

_BUNDLE_MANIFEST_RELATIVE_PATH = Path("artifacts/bundle-manifest.json")


class CohortRunState(BaseModel):
    """Durable scheduler state; model checkpoints remain lineage-local."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[3] = 3
    run_id: str
    profile: str
    config_hash: str
    execution_profile_id: str
    execution_profile_sha256: str
    execution_profile_file_sha256: str
    plan_hash: str
    prepared_manifest_sha256: str
    source_tree_sha256: str | None = None
    source_manifest_sha256: str | None = None
    source_entry_count: int | None = None
    bundle_manifest_sha256: str | None = None
    bundle_tree_sha256: str | None = None
    bundle_entry_count: int | None = None
    planned_lineage_ids: list[str]
    completed_lineage_ids: list[str]
    failed_attempts: dict[str, int]
    last_errors: dict[str, str]
    lineage_manifest_hashes: dict[str, str]
    status: Literal["active", "completed"]


def _plan_hash(plan: list[PlannedLineage]) -> str:
    return sha256_json([item.model_dump(mode="json") for item in plan])


def _new_state(
    config: ExperimentConfig,
    plan: list[PlannedLineage],
    prepared_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> CohortRunState:
    return CohortRunState(
        run_id=(
            f"{config.profile}-{config.config_hash[:12]}-"
            f"{execution_profile.profile_id}-{execution_profile.profile_hash[:12]}"
        ),
        profile=config.profile,
        config_hash=config.config_hash,
        execution_profile_id=execution_profile.profile_id,
        execution_profile_sha256=execution_profile.profile_hash,
        execution_profile_file_sha256=execution_profile.profile_file_sha256,
        plan_hash=_plan_hash(plan),
        prepared_manifest_sha256=prepared_manifest_sha256,
        planned_lineage_ids=[item.parent_config_id for item in plan],
        completed_lineage_ids=[],
        failed_attempts={},
        last_errors={},
        lineage_manifest_hashes={},
        status="active",
    )


def _verify_state_identity(state: CohortRunState, expected: CohortRunState) -> None:
    immutable_fields = (
        "run_id",
        "profile",
        "config_hash",
        "execution_profile_id",
        "execution_profile_sha256",
        "execution_profile_file_sha256",
        "plan_hash",
        "prepared_manifest_sha256",
        "planned_lineage_ids",
    )
    mismatches = [
        name for name in immutable_fields if getattr(state, name) != getattr(expected, name)
    ]
    if mismatches:
        raise ValueError(f"cohort resume refused: immutable state mismatch {mismatches}")


def _verify_state_execution_snapshot(state: CohortRunState) -> None:
    execution_snapshot = (
        state.source_tree_sha256,
        state.source_manifest_sha256,
        state.source_entry_count,
        state.bundle_manifest_sha256,
        state.bundle_tree_sha256,
        state.bundle_entry_count,
    )
    if any(value is None for value in execution_snapshot) and any(
        value is not None for value in execution_snapshot
    ):
        raise ValueError("cohort resume refused: execution snapshot binding is incomplete")
    for label, value in (
        ("source tree", state.source_tree_sha256),
        ("source manifest", state.source_manifest_sha256),
        ("bundle manifest", state.bundle_manifest_sha256),
        ("bundle tree", state.bundle_tree_sha256),
    ):
        if value is not None:
            validated_sha256(value, label=f"cohort state {label}")
    if state.source_entry_count is not None and state.source_entry_count <= 0:
        raise ValueError("cohort resume refused: source entry count is invalid")
    if state.bundle_entry_count is not None and state.bundle_entry_count <= 0:
        raise ValueError("cohort resume refused: bundle entry count is invalid")
    if state.completed_lineage_ids and execution_snapshot[0] is None:
        raise ValueError("cohort resume refused: completed state has no execution snapshot")


def _verify_state_progress(state: CohortRunState) -> None:
    planned = set(state.planned_lineage_ids)
    unknown = set(state.completed_lineage_ids) - planned
    if unknown:
        raise ValueError(f"cohort state contains unknown completed lineages: {sorted(unknown)}")
    if len(state.completed_lineage_ids) != len(set(state.completed_lineage_ids)):
        raise ValueError("cohort resume refused: duplicate completed lineage IDs")
    # Explicit receipt-bound selection may complete a non-prefix subset. Default
    # selection retains plan order through pending_lineages().
    tracked_failures = set(state.failed_attempts) | set(state.last_errors)
    if not tracked_failures <= planned:
        raise ValueError("cohort resume refused: failure state contains unknown lineages")
    if any(attempts <= 0 for attempts in state.failed_attempts.values()):
        raise ValueError("cohort resume refused: failure attempts must be positive")
    if not set(state.last_errors) <= set(state.failed_attempts):
        raise ValueError("cohort resume refused: errors have no matching failure attempt")
    _verify_state_execution_snapshot(state)
    expected_status = (
        "completed"
        if len(state.completed_lineage_ids) == len(state.planned_lineage_ids)
        else "active"
    )
    if state.status != expected_status:
        raise ValueError("cohort resume refused: status does not match completed lineages")


def _verify_state_manifests(state: CohortRunState, state_path: Path) -> None:
    completed = set(state.completed_lineage_ids)
    if set(state.lineage_manifest_hashes) != completed:
        raise ValueError("cohort resume refused: completed lineage hash set mismatch")
    for lineage_id, expected_hash in state.lineage_manifest_hashes.items():
        manifest = state_path.parent / f"{lineage_id}.jsonl"
        if not manifest.is_file() or sha256_file(manifest) != expected_hash:
            raise ValueError(f"cohort resume refused: lineage manifest hash mismatch {lineage_id}")


def load_or_create_cohort_state(
    path: str | Path,
    *,
    config: ExperimentConfig,
    plan: list[PlannedLineage],
    prepared_manifest_sha256: str,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    project_root: str | Path,
) -> CohortRunState:
    target = Path(path)
    bound_profile = verify_execution_profile_binding(
        execution_profile,
        execution_profile_path,
        project_root=project_root,
    )
    expected = _new_state(
        config,
        plan,
        prepared_manifest_sha256,
        bound_profile,
    )
    if not target.exists():
        return expected
    state = CohortRunState.model_validate_json(target.read_text(encoding="utf-8"))
    _verify_state_identity(state, expected)
    _verify_state_progress(state)
    _verify_state_manifests(state, target)
    return state


def write_cohort_state(state: CohortRunState, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def pending_lineages(plan: list[PlannedLineage], state: CohortRunState) -> list[PlannedLineage]:
    completed = set(state.completed_lineage_ids)
    return [item for item in plan if item.parent_config_id not in completed]


def verify_completed_cohort_run(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    output_root: str | Path,
    source_manifest: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    bundle_manifest_path: str | Path,
) -> tuple[CohortRunState, dict[str, Any]]:
    """Verify the complete, source-bound cohort required by RQ1-RQ3."""

    project = Path(project_root).resolve()
    bound_profile = verify_execution_profile_binding(
        execution_profile,
        execution_profile_path,
        project_root=project,
    )
    prepared = _validated_orchestration_request(
        config=config,
        prepared_manifest=prepared_manifest,
        max_lineages=None,
        lineage_ids=None,
    )
    source = verify_source_snapshot(project, source_manifest)
    bundle_manifest = _validated_bundle_manifest_path(
        bundle_manifest_path,
        project_root=project,
    )
    bundle = verify_bundle_snapshot(
        project,
        bundle_manifest,
        source_tree_sha256=source["tree_sha256"],
        source_entry_count=source["entry_count"],
    )
    plan = plan_cohort(config)
    state = load_or_create_cohort_state(
        Path(output_root) / "cohort" / "run-state.json",
        config=config,
        plan=plan,
        prepared_manifest_sha256=sha256_file(prepared),
        execution_profile=bound_profile,
        execution_profile_path=execution_profile_path,
        project_root=project,
    )
    expected_ids = [lineage.parent_config_id for lineage in plan]
    # Completion order is a permutation of plan order: the representative pilot finishes
    # a factor-covering triple before the cohort batches drain the remaining pending
    # lineages. Membership and count are the invariant here; _verify_state_progress
    # already rejects duplicates and lineages outside the plan.
    if (
        state.status != "completed"
        or len(state.completed_lineage_ids) != len(expected_ids)
        or set(state.completed_lineage_ids) != set(expected_ids)
        or state.source_tree_sha256 != source["tree_sha256"]
        or state.source_manifest_sha256 != source["manifest_sha256"]
        or state.source_entry_count != source["entry_count"]
        or state.bundle_manifest_sha256 != bundle["manifest_sha256"]
        or state.bundle_tree_sha256 != bundle["tree_sha256"]
        or state.bundle_entry_count != bundle["entry_count"]
    ):
        raise ValueError("RQ execution requires the complete current source-bound core cohort")
    return state, source


def _read_lineage_manifest(path: Path) -> list[CohortEntry]:
    return [
        CohortEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _lineage_protocol_mismatches(entries: list[CohortEntry], lineage: PlannedLineage) -> list[str]:
    expected_protocol = {
        "language": lineage.language,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "split": lineage.detector_split,
        "split_assignment_id": lineage.split_assignment_id,
    }
    return [
        field
        for field, expected in expected_protocol.items()
        if any(getattr(entry, field) != expected for entry in entries)
    ]


def _validate_entry_execution_profile_binding(
    entry: CohortEntry,
    execution_profile: ExecutionProfile,
) -> None:
    if (
        entry.execution_profile_id != execution_profile.profile_id
        or entry.execution_profile_sha256 != execution_profile.profile_hash
        or entry.execution_profile_file_sha256 != execution_profile.profile_file_sha256
    ):
        raise ValueError("lineage output execution-profile binding mismatch")


def _validate_lineage_entries(
    entries: list[CohortEntry],
    *,
    lineage: PlannedLineage,
    config_hash: str,
    execution_profile: ExecutionProfile,
) -> None:
    counts = verify_paired_lineages(entries)
    if counts != {"paired_lineages": 1, "adapters": 2}:
        raise ValueError(f"lineage output is not one complete pair: {counts}")
    if {entry.parent_config_id for entry in entries} != {lineage.parent_config_id}:
        raise ValueError("lineage output parent_config_id mismatch")
    if any(entry.config_sha256 != config_hash for entry in entries):
        raise ValueError("lineage output config hash mismatch")
    if _lineage_protocol_mismatches(entries, lineage):
        raise ValueError("lineage output protocol fields mismatch")
    for entry in entries:
        if entry.training_unit_id != lineage.training_unit_id(entry.label):
            raise ValueError("lineage output training-unit binding mismatch")
        _validate_entry_execution_profile_binding(entry, execution_profile)
    backdoored = next(entry for entry in entries if entry.label == "backdoored")
    if (
        backdoored.trigger_family != lineage.trigger_family
        or backdoored.poison_rate != lineage.poison_rate
    ):
        raise ValueError("lineage output poison fields mismatch")
    statuses = {entry.qc_status for entry in entries}
    if "pending" in statuses:
        raise ValueError("lineage output is incomplete: QC remains pending")
    if len(statuses) != 1:
        raise ValueError("lineage output pair has inconsistent QC status")


def _validate_completed_state_binding(
    state: CohortRunState,
    execution_profile: ExecutionProfile,
) -> None:
    state_binding = {
        "execution_profile_id": getattr(state, "execution_profile_id", None),
        "execution_profile_sha256": getattr(state, "execution_profile_sha256", None),
        "execution_profile_file_sha256": getattr(state, "execution_profile_file_sha256", None),
    }
    expected_state_binding = {
        "execution_profile_id": execution_profile.profile_id,
        "execution_profile_sha256": execution_profile.profile_hash,
        "execution_profile_file_sha256": execution_profile.profile_file_sha256,
    }
    if not same_json_value(state_binding, expected_state_binding):
        raise ValueError("completed lineage state execution-profile binding mismatch")
    state_snapshot = (
        getattr(state, "source_tree_sha256", None),
        getattr(state, "source_manifest_sha256", None),
        getattr(state, "source_entry_count", None),
        getattr(state, "bundle_manifest_sha256", None),
        getattr(state, "bundle_tree_sha256", None),
        getattr(state, "bundle_entry_count", None),
    )
    if any(value is None for value in state_snapshot):
        raise ValueError("completed lineage state execution snapshot is incomplete")


def _validate_completed_adapter_binding(
    *,
    config: ExperimentConfig,
    output_root: Path,
    state: CohortRunState,
    lineage: PlannedLineage,
    entry: CohortEntry,
    execution_profile: ExecutionProfile,
) -> None:
    label_root = output_root / "adapters" / lineage.parent_config_id / entry.label
    identity = validated_training_adapter_identity(config, label_root / "adapter")
    receipt = read_json_object(label_root / "training-receipt.json")
    expected_receipt = {
        "schema_version": 2,
        "status": "completed",
        "parent_config_id": lineage.parent_config_id,
        "adapter_label": entry.label,
        "config_sha256": config.config_hash,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "split_assignment_id": lineage.split_assignment_id,
        "training_unit_id": lineage.training_unit_id(entry.label),
        "source_tree_sha256": state.source_tree_sha256,
        "source_manifest_sha256": state.source_manifest_sha256,
        "adapter_sha256": entry.adapter_sha256,
        "execution_profile": execution_profile.binding(),
    }
    mismatches = [
        field
        for field, expected in expected_receipt.items()
        if not same_json_value(receipt.get(field), expected)
    ]
    if identity["weights_sha256"] != entry.adapter_sha256:
        mismatches.append("adapter_sha256")
    if mismatches:
        raise ValueError(
            "completed lineage adapter binding mismatch "
            f"{lineage.parent_config_id}/{entry.label}: {sorted(set(mismatches))}"
        )


def _completed_lineage_is_qc_valid(
    *,
    config: ExperimentConfig,
    output_root: Path,
    state: CohortRunState,
    lineage: PlannedLineage,
    execution_profile: ExecutionProfile,
) -> bool:
    manifest = output_root / "cohort" / f"{lineage.parent_config_id}.jsonl"
    expected_hash = state.lineage_manifest_hashes.get(lineage.parent_config_id)
    if (
        not isinstance(expected_hash, str)
        or not manifest.is_file()
        or sha256_file(manifest) != expected_hash
    ):
        raise ValueError(f"completed lineage manifest binding mismatch {lineage.parent_config_id}")
    entries = _read_lineage_manifest(manifest)
    _validate_lineage_entries(
        entries,
        lineage=lineage,
        config_hash=config.config_hash,
        execution_profile=execution_profile,
    )
    for entry in entries:
        _validate_completed_adapter_binding(
            config=config,
            output_root=output_root,
            state=state,
            lineage=lineage,
            entry=entry,
            execution_profile=execution_profile,
        )
    return {entry.qc_status for entry in entries} == {"valid"}


def qc_valid_completed_lineages(
    *,
    config: ExperimentConfig,
    project_root: str | Path,
    output_root: str | Path,
    state: CohortRunState,
    split: str,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
) -> list[PlannedLineage]:
    """Return only hash-bound completed pairs whose paired QC status is valid."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("completed lineage split is invalid")
    bound_profile = verify_execution_profile_binding(
        execution_profile,
        execution_profile_path,
        project_root=project_root,
    )
    _validate_completed_state_binding(state, bound_profile)
    cohort_root = Path(output_root) / "cohort"
    selected: list[PlannedLineage] = []
    for lineage in plan_cohort(config):
        if lineage.detector_split != split:
            continue
        if _completed_lineage_is_qc_valid(
            config=config,
            output_root=cohort_root.parent,
            state=state,
            lineage=lineage,
            execution_profile=bound_profile,
        ):
            selected.append(lineage)
    return selected


def _summary(
    state: CohortRunState,
    *,
    execution_profile: ExecutionProfile,
    selected: list[PlannedLineage],
    executed: list[str],
    execute: bool,
    execution_gate: dict[str, Any] | None,
    bundle_manifest_path: Path,
) -> dict[str, Any]:
    completed = len(state.completed_lineage_ids)
    return {
        "status": state.status if execute else "dry_run",
        "execute": execute,
        "run_id": state.run_id,
        "profile": state.profile,
        "config_hash": state.config_hash,
        "execution_profile": execution_profile.binding(),
        "bundle_manifest_sha256": (
            state.bundle_manifest_sha256 or sha256_file(bundle_manifest_path)
        ),
        "plan_hash": state.plan_hash,
        "prepared_manifest_sha256": state.prepared_manifest_sha256,
        "source_tree_sha256": state.source_tree_sha256,
        "source_manifest_sha256": state.source_manifest_sha256,
        "source_entry_count": state.source_entry_count,
        "bundle_tree_sha256": state.bundle_tree_sha256,
        "bundle_entry_count": state.bundle_entry_count,
        "planned_lineages": len(state.planned_lineage_ids),
        "completed_lineages": completed,
        "pending_lineages": len(state.planned_lineage_ids) - completed,
        "selected_lineage_ids": [item.parent_config_id for item in selected],
        "executed_lineage_ids": executed,
        "failed_attempts": state.failed_attempts,
        "execution_gate": execution_gate,
    }


def _validated_orchestration_request(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    max_lineages: int | None,
    lineage_ids: Sequence[str] | None,
) -> Path:
    if config.fixture_smoke_only:
        raise ValueError("cohort orchestration refuses fixture config; use smoke")
    if max_lineages is not None and (type(max_lineages) is not int or max_lineages <= 0):
        raise ValueError("max_lineages must be a positive integer")
    if max_lineages is not None and lineage_ids is not None:
        raise ValueError("max_lineages and lineage_ids are mutually exclusive")
    prepared = Path(prepared_manifest)
    if not prepared.is_file():
        raise FileNotFoundError(f"prepared manifest not found: {prepared}")
    return prepared


def _validated_bundle_manifest_path(
    bundle_manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> Path:
    """Require the exact inner upload-manifest path before scheduling work."""

    supplied = Path(bundle_manifest_path)
    if supplied.is_symlink():
        raise ValueError("bundle manifest path must not be a symbolic link")
    source = supplied.resolve()
    expected = (Path(project_root).resolve() / _BUNDLE_MANIFEST_RELATIVE_PATH).resolve()
    if source != expected:
        raise ValueError("bundle manifest path is not canonical for the project")
    if not source.is_file():
        raise FileNotFoundError(f"bundle manifest not found: {source}")
    return source


def _execution_authorization(
    *,
    config: ExperimentConfig,
    plan: list[PlannedLineage],
    selected_lineage_ids: tuple[str, ...],
    execution_phase: str | None,
    compute_estimate: str | Path | None,
    config_path: str | Path | None,
    project_root: str | Path,
    preflight_artifact: str | Path | None,
    execution_receipt: str | Path | None,
    source_manifest: str | Path | None,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    bundle_manifest_path: str | Path,
) -> dict[str, Any]:
    if execution_phase not in {"pilot", "cohort"}:
        raise ValueError("execution_phase must be pilot or cohort when execute=true")
    bound_profile = verify_execution_profile_binding(
        execution_profile,
        execution_profile_path,
        project_root=project_root,
    )
    bundle_manifest = _validated_bundle_manifest_path(
        bundle_manifest_path,
        project_root=project_root,
    )
    compute_gate = (
        verify_execution_compute_gate(
            execution_profile=bound_profile,
            profile=config.profile,
            execution_phase=execution_phase,
            max_lineages=len(selected_lineage_ids),
            compute_estimate_path=compute_estimate,
            config_sha256=config.config_hash,
            planned_adapter_runs=len(plan) * 2,
        )
        if execution_phase == "pilot"
        else None
    )
    if config_path is None:
        raise ValueError("execution evidence is required: ['config_path']")
    runtime_evidence = verify_execution_evidence(
        config=config,
        config_path=config_path,
        execution_profile=bound_profile,
        execution_profile_path=execution_profile_path,
        project_root=project_root,
        preflight_path=preflight_artifact,
        receipt_path=execution_receipt,
        source_manifest_path=source_manifest,
        bundle_manifest_path=bundle_manifest,
        execution_phase=execution_phase,
        compute_estimate_path=compute_estimate,
        planned_adapter_runs=len(plan) * 2,
        expected_lineage_ids=selected_lineage_ids,
    )
    if compute_gate is None:
        compute_gate = verify_execution_compute_gate(
            execution_profile=bound_profile,
            profile=config.profile,
            execution_phase=execution_phase,
            max_lineages=len(selected_lineage_ids),
            compute_estimate_path=compute_estimate,
            config_sha256=config.config_hash,
            planned_adapter_runs=len(plan) * 2,
            expected_source_tree_sha256=runtime_evidence["tree_sha256"],
            expected_source_manifest_sha256=runtime_evidence["manifest_sha256"],
        )
    return {
        **compute_gate,
        "bundle_manifest_sha256": sha256_file(bundle_manifest),
        "runtime_evidence": runtime_evidence,
    }


def _bind_cohort_source(
    state: CohortRunState,
    execution_gate: dict[str, Any],
    *,
    execution_profile: ExecutionProfile,
) -> CohortRunState:
    runtime = execution_gate.get("runtime_evidence")
    if not isinstance(runtime, dict):
        raise ValueError("cohort execution gate has no runtime source evidence")
    tree_sha256 = runtime.get("tree_sha256")
    manifest_sha256 = runtime.get("manifest_sha256")
    source_entry_count = runtime.get("entry_count")
    tree_sha256 = validated_sha256(tree_sha256, label="cohort source tree")
    manifest_sha256 = validated_sha256(manifest_sha256, label="cohort source manifest")
    if not same_json_value(runtime.get("execution_profile"), execution_profile.binding()):
        raise ValueError("cohort execution gate has a mismatched execution profile")
    bundle = runtime.get("bundle_snapshot")
    if not isinstance(bundle, dict):
        raise ValueError("cohort execution gate has no bundle snapshot evidence")
    bundle_manifest_sha256 = validated_sha256(
        bundle.get("manifest_sha256"), label="cohort bundle manifest"
    )
    bundle_tree_sha256 = validated_sha256(bundle.get("tree_sha256"), label="cohort bundle tree")
    bundle_entry_count = bundle.get("entry_count")
    if type(source_entry_count) is not int or source_entry_count <= 0:
        raise ValueError("cohort execution gate has an invalid source entry count")
    if type(bundle_entry_count) is not int or bundle_entry_count <= 0:
        raise ValueError("cohort execution gate has an invalid bundle entry count")
    if state.source_tree_sha256 is None:
        if state.completed_lineage_ids:
            raise ValueError("cannot bind source after cohort outputs already exist")
        return state.model_copy(
            update={
                "source_tree_sha256": tree_sha256,
                "source_manifest_sha256": manifest_sha256,
                "source_entry_count": source_entry_count,
                "bundle_manifest_sha256": bundle_manifest_sha256,
                "bundle_tree_sha256": bundle_tree_sha256,
                "bundle_entry_count": bundle_entry_count,
            }
        )
    if (
        state.source_tree_sha256 != tree_sha256
        or state.source_manifest_sha256 != manifest_sha256
        or state.source_entry_count != source_entry_count
    ):
        raise ValueError("cohort resume refused: source snapshot changed")
    if (
        state.bundle_manifest_sha256 != bundle_manifest_sha256
        or state.bundle_tree_sha256 != bundle_tree_sha256
        or state.bundle_entry_count != bundle_entry_count
    ):
        raise ValueError("cohort resume refused: bundle snapshot changed")
    return state


def _verify_current_execution_snapshot(
    *,
    state: CohortRunState,
    project_root: str | Path,
    source_manifest_path: str | Path | None,
    bundle_manifest_path: Path,
) -> None:
    """Reject source or upload-bundle drift before a lineage becomes durable."""

    if source_manifest_path is None:
        raise RuntimeError("cohort execution lost its source manifest path")
    source = verify_source_snapshot(project_root, source_manifest_path)
    bundle = verify_bundle_snapshot(
        project_root,
        bundle_manifest_path,
        source_tree_sha256=source["tree_sha256"],
        source_entry_count=source["entry_count"],
    )
    expected = {
        "source_tree_sha256": state.source_tree_sha256,
        "source_manifest_sha256": state.source_manifest_sha256,
        "source_entry_count": state.source_entry_count,
        "bundle_manifest_sha256": state.bundle_manifest_sha256,
        "bundle_tree_sha256": state.bundle_tree_sha256,
        "bundle_entry_count": state.bundle_entry_count,
    }
    observed = {
        "source_tree_sha256": source["tree_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_entry_count": source["entry_count"],
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "bundle_tree_sha256": bundle["tree_sha256"],
        "bundle_entry_count": bundle["entry_count"],
    }
    if not same_json_value(observed, expected):
        raise ValueError("cohort execution snapshot changed during training")


def _default_pair_trainer() -> PairTrainer:
    from .training import train_planned_pair

    return train_planned_pair


def _materialize_lineage_entries(
    *,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    prepared_manifest: Path,
    project_root: str | Path,
    output_root: Path,
    lineage_manifest: Path,
    trainer: PairTrainer,
    execution_phase: str | None,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> None:
    if lineage_manifest.is_file():
        entries = _read_lineage_manifest(lineage_manifest)
    else:
        entries = trainer(
            config=config,
            lineage=lineage,
            prepared_manifest=prepared_manifest,
            project_root=project_root,
            output_root=output_root,
            pilot_telemetry=execution_phase == "pilot",
            source_tree_sha256=source_tree_sha256,
            source_manifest_sha256=source_manifest_sha256,
            execution_profile=execution_profile,
        )
        _validate_lineage_entries(
            entries,
            lineage=lineage,
            config_hash=config.config_hash,
            execution_profile=execution_profile,
        )
        write_jsonl(entries, lineage_manifest)
        return
    _validate_lineage_entries(
        entries,
        lineage=lineage,
        config_hash=config.config_hash,
        execution_profile=execution_profile,
    )


def _record_lineage_failure(
    *,
    state: CohortRunState,
    state_path: Path,
    ledger_path: Path,
    lineage_id: str,
    config_sha256: str,
    error: Exception,
) -> CohortRunState:
    attempts = {**state.failed_attempts}
    attempts[lineage_id] = attempts.get(lineage_id, 0) + 1
    errors = {**state.last_errors, lineage_id: f"{type(error).__name__}: {error}"}
    updated = state.model_copy(update={"failed_attempts": attempts, "last_errors": errors})
    write_cohort_state(updated, state_path)
    append_jsonl(
        ledger_path,
        {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "lineage_id": lineage_id,
            "config_sha256": config_sha256,
            "execution_profile_id": state.execution_profile_id,
            "execution_profile_sha256": state.execution_profile_sha256,
            "execution_profile_file_sha256": state.execution_profile_file_sha256,
            "source_tree_sha256": state.source_tree_sha256,
            "source_manifest_sha256": state.source_manifest_sha256,
            "source_entry_count": state.source_entry_count,
            "bundle_manifest_sha256": state.bundle_manifest_sha256,
            "bundle_tree_sha256": state.bundle_tree_sha256,
            "bundle_entry_count": state.bundle_entry_count,
            "attempt": attempts[lineage_id],
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return updated


def _complete_lineage(
    *,
    state: CohortRunState,
    state_path: Path,
    lineage_id: str,
    lineage_manifest: Path,
) -> CohortRunState:
    completed = [*state.completed_lineage_ids, lineage_id]
    hashes = {**state.lineage_manifest_hashes, lineage_id: sha256_file(lineage_manifest)}
    errors = {key: value for key, value in state.last_errors.items() if key != lineage_id}
    status = "completed" if len(completed) == len(state.planned_lineage_ids) else "active"
    updated = state.model_copy(
        update={
            "completed_lineage_ids": completed,
            "lineage_manifest_hashes": hashes,
            "last_errors": errors,
            "status": status,
        }
    )
    write_cohort_state(updated, state_path)
    return updated


def orchestrate_cohort(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    output_root: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    bundle_manifest_path: str | Path,
    execute: bool,
    max_lineages: int | None = None,
    lineage_ids: Sequence[str] | None = None,
    execution_phase: str | None = None,
    compute_estimate: str | Path | None = None,
    config_path: str | Path | None = None,
    preflight_artifact: str | Path | None = None,
    execution_receipt: str | Path | None = None,
    source_manifest: str | Path | None = None,
    trainer: PairTrainer | None = None,
) -> dict[str, Any]:
    """Preview or execute a deterministic, receipt-bound selection of pending pairs."""

    root = Path(project_root).resolve()
    bound_profile = verify_execution_profile_binding(
        execution_profile,
        execution_profile_path,
        project_root=root,
    )
    bundle_manifest = _validated_bundle_manifest_path(
        bundle_manifest_path,
        project_root=root,
    )
    prepared = _validated_orchestration_request(
        config=config,
        prepared_manifest=prepared_manifest,
        max_lineages=max_lineages,
        lineage_ids=lineage_ids,
    )
    output = Path(output_root)
    plan = plan_cohort(config)
    state_path = output / "cohort" / "run-state.json"
    state = load_or_create_cohort_state(
        state_path,
        config=config,
        plan=plan,
        prepared_manifest_sha256=sha256_file(prepared),
        execution_profile=bound_profile,
        execution_profile_path=execution_profile_path,
        project_root=root,
    )
    pending = pending_lineages(plan, state)
    selected = select_pending_lineages(
        pending,
        max_lineages=max_lineages,
        lineage_ids=lineage_ids,
    )
    if not execute:
        return _summary(
            state,
            execution_profile=bound_profile,
            selected=selected,
            executed=[],
            execute=False,
            execution_gate=None,
            bundle_manifest_path=bundle_manifest,
        )
    if not selected:
        raise ValueError("execution refused: no pending lineages were selected")

    execution_gate = _execution_authorization(
        config=config,
        plan=plan,
        selected_lineage_ids=tuple(item.parent_config_id for item in selected),
        execution_phase=execution_phase,
        compute_estimate=compute_estimate,
        config_path=config_path,
        project_root=project_root,
        preflight_artifact=preflight_artifact,
        execution_receipt=execution_receipt,
        source_manifest=source_manifest,
        execution_profile=bound_profile,
        execution_profile_path=execution_profile_path,
        bundle_manifest_path=bundle_manifest,
    )
    state = _bind_cohort_source(
        state,
        execution_gate,
        execution_profile=bound_profile,
    )
    source_tree = state.source_tree_sha256
    source_manifest_hash = state.source_manifest_sha256
    if source_tree is None or source_manifest_hash is None:
        raise RuntimeError("cohort source binding failed before execution")
    write_cohort_state(state, state_path)
    train_pair = trainer if trainer is not None else _default_pair_trainer()

    executed: list[str] = []
    for lineage in selected:
        lineage_id = lineage.parent_config_id
        lineage_manifest = output / "cohort" / f"{lineage_id}.jsonl"
        try:
            _materialize_lineage_entries(
                config=config,
                lineage=lineage,
                prepared_manifest=prepared,
                project_root=project_root,
                output_root=output,
                lineage_manifest=lineage_manifest,
                trainer=train_pair,
                execution_phase=execution_phase,
                source_tree_sha256=source_tree,
                source_manifest_sha256=source_manifest_hash,
                execution_profile=bound_profile,
            )
        except Exception as exc:
            state = _record_lineage_failure(
                state=state,
                state_path=state_path,
                ledger_path=output / "ledgers" / "failed-runs.jsonl",
                lineage_id=lineage_id,
                config_sha256=config.config_hash,
                error=exc,
            )
            raise

        verify_execution_profile_binding(
            bound_profile,
            execution_profile_path,
            project_root=root,
        )
        _verify_current_execution_snapshot(
            state=state,
            project_root=root,
            source_manifest_path=source_manifest,
            bundle_manifest_path=bundle_manifest,
        )
        state = _complete_lineage(
            state=state,
            state_path=state_path,
            lineage_id=lineage_id,
            lineage_manifest=lineage_manifest,
        )
        executed.append(lineage_id)

    return _summary(
        state,
        execution_profile=bound_profile,
        selected=selected,
        executed=executed,
        execute=True,
        execution_gate=execution_gate,
        bundle_manifest_path=bundle_manifest,
    )
