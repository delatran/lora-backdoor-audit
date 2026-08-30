"""Pure helpers for interpreting live preflight and source-sync evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .compute import (
    COMPUTE_ESTIMATE_SCHEMA_VERSION,
    COMPUTE_ESTIMATE_SCOPE,
    REQUIRED_REPRESENTATIVE_PILOT_PAIRS,
    PilotPairBinding,
    estimate_compute,
    read_pilot_evaluation_telemetry,
    read_pilot_telemetry,
    verify_pilot_telemetry_alignment,
)
from .config import ExperimentConfig
from .execution_profiles import (
    ExecutionProfile,
    execution_profile_binding_matches,
    is_locked_execution_profile,
    verify_execution_profile_binding,
)
from .hashing import is_sha256, same_json_value, sha256_file, sha256_json
from .preflight import (
    ACCELERATED_PINNED_PACKAGES,
    BASE_PINNED_PACKAGES,
    accelerator_snapshot,
    device_matches_requirements,
    normalize_compute_capability,
)

LIVE_PREFLIGHT_RELATIVE_PATH = Path("artifacts/preflight-live.json")
LIVE_RECEIPT_RELATIVE_PATH = Path("artifacts/execution-receipt.json")
BUNDLE_MANIFEST_RELATIVE_PATH = Path("artifacts/bundle-manifest.json")
SOURCE_ROOT_FILES = {
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
}
SOURCE_ROOT_DIRECTORIES = {
    "configs",
    "docs",
    "notebooks",
    "private",
    "requirements",
    "schemas",
    "scripts",
    "src",
    "tests",
}
SOURCE_EXCLUDED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "executed",
}
SOURCE_PRIVATE_PATHS = {"private/trigger_registry.yaml"}


@dataclass(frozen=True)
class LivePreflightAssessment:
    """Gate-level interpretation of one external runtime preflight."""

    preflight_complete: bool
    runtime_observed: bool
    required_device_observed: bool
    observed_devices: tuple[str, ...]


def _positive_finite_json_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and float(value) > 0.0


def _nonnegative_finite_json_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and float(value) >= 0.0


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _bf16_probe_passed(device: dict[str, Any]) -> bool:
    probe = device.get("bf16_probe")
    return (
        isinstance(probe, dict)
        and probe.get("attempted") is True
        and probe.get("passed") is True
        and probe.get("finite") is True
    )


def _valid_execution_profile_object(value: Any) -> bool:
    return (
        is_locked_execution_profile(value)
        and isinstance(value, ExecutionProfile)
        and value.strict_accelerator
        and value.require_bf16_probe
    )


def select_live_evidence_paths(root: Path) -> tuple[Path, Path]:
    """Return the single canonical evidence pair so partial evidence fails closed."""

    return root / LIVE_PREFLIGHT_RELATIVE_PATH, root / LIVE_RECEIPT_RELATIVE_PATH


def assess_live_preflight(
    payload: dict[str, Any],
    *,
    expected_profile: ExecutionProfile | None = None,
) -> LivePreflightAssessment:
    """Assess live evidence against one explicit immutable execution profile."""

    accelerator = payload.get("accelerator")
    if not isinstance(accelerator, dict):
        accelerator = {}
    raw_devices = accelerator.get("devices")
    devices = raw_devices if isinstance(raw_devices, list) else []
    observed_devices: list[str] = []
    valid_device_count = 0
    required_device_candidate = False
    backend = accelerator.get("backend")
    for index, device in enumerate(devices):
        if not isinstance(device, dict):
            continue
        name = device.get("name")
        memory_gib = device.get("total_memory_gib")
        capability = device.get("compute_capability")
        if not _positive_finite_json_number(memory_gib):
            continue
        normalized_capability = normalize_compute_capability(capability)
        if len(normalized_capability) != 2:
            continue
        observed_capability = tuple(normalized_capability)
        label = name.strip() if isinstance(name, str) and name.strip() else f"device-{index}"
        valid_device_count += 1
        observed_devices.append(
            f"{label}:{float(memory_gib):.2f}GiB:capability={observed_capability[0]}."
            f"{observed_capability[1]}:"
            f"bf16_probe={'passed' if _bf16_probe_passed(device) else 'failed'}"
        )
        if expected_profile is not None and device_matches_requirements(
            device,
            expected_profile,
            backend=backend if isinstance(backend, str) else None,
        ):
            required_device_candidate = True

    runtime_observed = (
        accelerator.get("framework_available") is True
        and accelerator.get("accelerator_available") is True
        and valid_device_count > 0
    )
    packages = payload.get("packages")
    expected_packages = {**BASE_PINNED_PACKAGES, **ACCELERATED_PINNED_PACKAGES}
    package_contract_valid = isinstance(packages, dict) and all(
        packages.get(package) == version for package, version in expected_packages.items()
    )
    runtime_provenance_valid = all(
        _nonempty_string(accelerator.get(field))
        for field in ("framework_version", "backend", "runtime_version")
    )
    profile_binding_valid = expected_profile is not None and execution_profile_binding_matches(
        payload.get("execution_profile"), expected_profile
    )
    preflight_complete = (
        runtime_observed
        and required_device_candidate
        and profile_binding_valid
        and payload.get("status") == "passed"
        and payload.get("mode") == "strict"
        and payload.get("fixture_smoke_only") is False
        and payload.get("failures") == []
        and _nonempty_string(payload.get("python"))
        and _nonempty_string(payload.get("platform"))
        and _nonnegative_finite_json_number(payload.get("free_disk_gib"))
        and payload.get("artifact_writable") is True
        and package_contract_valid
        and runtime_provenance_valid
    )
    return LivePreflightAssessment(
        preflight_complete=preflight_complete,
        runtime_observed=runtime_observed,
        required_device_observed=(
            runtime_observed and profile_binding_valid and required_device_candidate
        ),
        observed_devices=tuple(observed_devices),
    )


def live_evidence_matches(
    preflight: dict[str, Any],
    receipt: dict[str, Any],
    *,
    execution_profile: ExecutionProfile,
    config_hash: str,
    config_file_sha256: str,
    preflight_sha256: str,
    source_manifest_sha256: str,
    source_tree_sha256: str,
    source_entry_count: int,
    bundle_manifest_sha256: str,
    bundle_tree_sha256: str,
    bundle_entry_count: int,
    expected_representative_pilot: bool,
    expected_cohort_execution: bool,
    expected_lineage_ids: Sequence[str],
) -> bool:
    """Bind captured runtime evidence to its receipt and current Core config."""

    if (
        not _valid_execution_profile_object(execution_profile)
        or not all(
            is_sha256(value)
            for value in (
                config_hash,
                config_file_sha256,
                preflight_sha256,
                source_manifest_sha256,
                source_tree_sha256,
                bundle_manifest_sha256,
                bundle_tree_sha256,
            )
        )
        or type(source_entry_count) is not int
        or source_entry_count <= 0
        or type(bundle_entry_count) is not int
        or bundle_entry_count <= 0
        or type(expected_representative_pilot) is not bool
        or type(expected_cohort_execution) is not bool
        or isinstance(expected_lineage_ids, (str, bytes))
        or not expected_lineage_ids
        or any(
            not isinstance(lineage_id, str) or not lineage_id or lineage_id != lineage_id.strip()
            for lineage_id in expected_lineage_ids
        )
        or len(set(expected_lineage_ids)) != len(expected_lineage_ids)
    ):
        return False
    scope = receipt.get("execution_scope")
    strict = receipt.get("strict_preflight")
    source = receipt.get("source_snapshot")
    bundle = receipt.get("bundle_snapshot")
    if (
        not isinstance(scope, dict)
        or not isinstance(strict, dict)
        or not isinstance(source, dict)
        or not isinstance(bundle, dict)
    ):
        return False
    return (
        type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 4
        and scope.get("strict_preflight") is True
        and scope.get("representative_pilot") is expected_representative_pilot
        and scope.get("cohort_execution") is expected_cohort_execution
        and scope.get("lineage_ids") == list(expected_lineage_ids)
        and execution_profile_binding_matches(preflight.get("execution_profile"), execution_profile)
        and preflight.get("config_hash") == config_hash
        and preflight.get("config_file_sha256") == config_file_sha256
        and strict.get("config_hash") == config_hash
        and strict.get("config_file_sha256") == config_file_sha256
        and execution_profile_binding_matches(strict.get("execution_profile"), execution_profile)
        and strict.get("artifact_sha256") == preflight_sha256
        and source.get("manifest_sha256") == source_manifest_sha256
        and is_sha256(source_tree_sha256)
        and source.get("tree_sha256") == source_tree_sha256
        and source.get("entry_count") == source_entry_count
        and bundle.get("manifest_sha256") == bundle_manifest_sha256
        and bundle.get("tree_sha256") == bundle_tree_sha256
        and bundle.get("entry_count") == bundle_entry_count
    )


def collect_source_snapshot_files(project_root: Path) -> list[Path]:
    """Collect the single canonical execution source surface."""

    root = project_root.resolve()
    files = [root / name for name in sorted(SOURCE_ROOT_FILES) if (root / name).is_file()]
    for directory_name in sorted(SOURCE_ROOT_DIRECTORIES):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in SOURCE_EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"source snapshot refuses symbolic link: {relative.as_posix()}")
            files.append(path)
    if any(path.is_symlink() for path in files):
        raise ValueError("source snapshot refuses symbolic root files")
    relative_paths = {path.relative_to(root).as_posix() for path in files}
    missing_private = SOURCE_PRIVATE_PATHS - relative_paths
    if missing_private:
        raise FileNotFoundError(f"required private inputs missing: {sorted(missing_private)}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_source_snapshot(project_root: Path) -> dict[str, Any]:
    """Build the canonical byte-level source snapshot used by execution gates."""

    root = project_root.resolve()
    entries = []
    for path in collect_source_snapshot_files(root):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "private": relative in SOURCE_PRIVATE_PATHS,
            }
        )
    canonical_entries = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return {
        "schema_version": 2,
        "destination_label": "lora-audit",
        "external_write_authorized": False,
        "external_write_performed": False,
        "entry_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "tree_sha256": hashlib.sha256(canonical_entries.encode("utf-8")).hexdigest(),
        "tree_hash_contract": "ordered path, size_bytes, sha256, and private fields",
        "private_paths": sorted(SOURCE_PRIVATE_PATHS),
        "excluded": [
            ".git/",
            ".venv/",
            "artifacts/",
            "notebooks/executed/",
            "cache and bytecode directories",
        ],
        "post_sync_verifier": {
            "required_destination_label": "lora-audit",
            "required_entry_count": len(entries),
            "required_tree_sha256": "recompute from retrieved file bytes",
            "execution_scope": (
                "strict preflight, receipt-gated pilot/cohort, and completed-cohort RQ1-RQ3"
            ),
        },
        "rollback": (
            "remove only the newly created destination identifier after verifying "
            "that it is the intended synchronization target"
        ),
        "entries": entries,
    }


def verify_source_snapshot(
    project_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_entry_count: int | None = None,
) -> dict[str, Any]:
    """Re-hash the complete source surface and reject a stale or partial manifest."""

    path = Path(manifest_path)
    expected = _load_json_object(path, "source manifest")
    observed = build_source_snapshot(Path(project_root))
    if expected_entry_count is not None and (
        type(expected_entry_count) is not int or expected_entry_count <= 0
    ):
        raise ValueError("expected source entry count must be a positive integer")
    if expected_entry_count is not None and observed["entry_count"] != expected_entry_count:
        raise ValueError(
            "source snapshot entry count mismatch: "
            f"expected={expected_entry_count} observed={observed['entry_count']}"
        )
    required_fields = (
        "schema_version",
        "destination_label",
        "external_write_authorized",
        "external_write_performed",
        "entry_count",
        "total_size_bytes",
        "tree_sha256",
        "tree_hash_contract",
        "private_paths",
        "entries",
    )
    mismatches = [field for field in required_fields if expected.get(field) != observed[field]]
    if mismatches:
        raise ValueError(f"source manifest mismatch: {mismatches}")
    return {
        "manifest_sha256": sha256_file(path),
        "tree_sha256": observed["tree_sha256"],
        "entry_count": observed["entry_count"],
        "total_size_bytes": observed["total_size_bytes"],
    }


def _validated_bundle_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("bundle manifest entries must be a nonempty list")
    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "size_bytes",
            "sha256",
            "private",
        }:
            raise ValueError(f"bundle manifest entry {index} has an invalid shape")
        relative = raw_entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or relative.startswith("/")
            or ":" in relative
        ):
            raise ValueError(f"bundle manifest entry {index} has an invalid path")
        path = PurePosixPath(relative)
        if (
            path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or relative != relative.strip()
        ):
            raise ValueError(f"bundle manifest entry {index} has a noncanonical path")
        size = raw_entry.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ValueError(f"bundle manifest entry {index} has an invalid size")
        if not is_sha256(raw_entry.get("sha256")) or type(raw_entry.get("private")) is not bool:
            raise ValueError(f"bundle manifest entry {index} has invalid integrity fields")
        entries.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": raw_entry["sha256"],
                "private": raw_entry["private"],
            }
        )
    paths = [entry["path"] for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("bundle manifest entries are not sorted and unique")
    return entries


def verify_bundle_snapshot(
    project_root: str | Path,
    bundle_manifest_path: str | Path,
    *,
    source_tree_sha256: str,
    source_entry_count: int,
) -> dict[str, Any]:
    """Validate the canonical outer upload snapshot bound to a verified source tree.

    This verifier is intentionally lightweight and can run inside the installed
    package.  The upload builder additionally verifies every physical file
    before an archive is created; execution gates use this function to prevent
    an execution receipt from being replayed against another bundle snapshot.
    """

    root = Path(project_root).resolve()
    expected_path = (root / BUNDLE_MANIFEST_RELATIVE_PATH).resolve()
    path = Path(bundle_manifest_path).resolve()
    if path != expected_path or not path.is_file():
        raise ValueError("bundle manifest path is not the canonical project artifact")
    if (
        not is_sha256(source_tree_sha256)
        or type(source_entry_count) is not int
        or source_entry_count <= 0
    ):
        raise ValueError("bundle snapshot requires a verified source-tree binding")
    payload = _load_json_object(path, "bundle manifest")
    entries = _validated_bundle_entries(payload)
    canonical_entries = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    expected_tree = hashlib.sha256(canonical_entries.encode("utf-8")).hexdigest()
    if (
        payload.get("schema_version") != 1
        or payload.get("destination_label") != root.parent.name
        or payload.get("tree_hash_contract")
        != "ordered path, size_bytes, sha256, and private fields"
        or payload.get("entry_count") != len(entries)
        or payload.get("total_size_bytes") != sum(entry["size_bytes"] for entry in entries)
        or payload.get("tree_sha256") != expected_tree
        or payload.get("inner_source_tree_sha256") != source_tree_sha256
        or payload.get("inner_source_entry_count") != source_entry_count
    ):
        raise ValueError("bundle manifest does not match its source-tree binding")
    paths = {entry["path"] for entry in entries}
    if not {"RUN_EXPERIMENT.ipynb", "bootstrap.py"}.issubset(paths) or any(
        path not in {"RUN_EXPERIMENT.ipynb", "bootstrap.py"} and not path.startswith("lora-audit/")
        for path in paths
    ):
        raise ValueError("bundle manifest has an invalid outer entrypoint surface")
    return {
        "manifest_sha256": sha256_file(path),
        "tree_sha256": expected_tree,
        "entry_count": len(entries),
    }


def cohort_compute_authorization_binding(
    *,
    compute_estimate_path: str | Path,
    profile: str,
    planned_adapter_runs: int,
) -> dict[str, Any]:
    """Build the canonical receipt binding for one observed compute estimate."""

    if not isinstance(profile, str) or not profile:
        raise ValueError("cohort execution receipt requires a scientific profile")
    if type(planned_adapter_runs) is not int or planned_adapter_runs <= 0:
        raise ValueError("cohort execution receipt requires a positive adapter-run count")
    compute_path = Path(compute_estimate_path).resolve()
    if not compute_path.is_file():
        raise ValueError("cohort execution receipt references a missing compute estimate")
    compute_payload = _read_compute_estimate(compute_path)
    return {
        "estimate_sha256": sha256_file(compute_path),
        "available_device_hours": compute_payload.get("available_device_hours"),
        "profile": profile,
        "planned_adapter_runs": planned_adapter_runs,
        "recommendation": compute_payload.get("recommendation"),
    }


def _verified_cohort_compute_authorization(
    *,
    execution_phase: str,
    compute_estimate_path: str | Path | None,
    receipt: dict[str, Any],
    config: ExperimentConfig,
    planned_adapter_runs: int,
) -> dict[str, Any] | None:
    if execution_phase != "cohort":
        return None
    if compute_estimate_path is None:
        raise ValueError("cohort execution receipt requires a compute estimate")
    expected = cohort_compute_authorization_binding(
        compute_estimate_path=compute_estimate_path,
        profile=config.profile,
        planned_adapter_runs=planned_adapter_runs,
    )
    if not same_json_value(receipt.get("compute_authorization"), expected):
        raise ValueError("execution evidence rejected: compute authorization mismatch")
    return expected


def _resolved_execution_evidence_paths(
    *,
    preflight_path: str | Path | None,
    receipt_path: str | Path | None,
    source_manifest_path: str | Path | None,
    bundle_manifest_path: str | Path | None,
    execution_phase: str,
) -> dict[str, Path]:
    if execution_phase not in {"pilot", "cohort"}:
        raise ValueError("execution evidence phase must be pilot or cohort")
    evidence_paths = {
        "preflight": preflight_path,
        "receipt": receipt_path,
        "source_manifest": source_manifest_path,
        "bundle_manifest": bundle_manifest_path,
    }
    missing_arguments = sorted(name for name, value in evidence_paths.items() if value is None)
    if missing_arguments:
        raise ValueError(f"execution evidence is required: {missing_arguments}")
    resolved = {name: Path(value).resolve() for name, value in evidence_paths.items()}
    missing_files = sorted(name for name, path in resolved.items() if not path.is_file())
    if missing_files:
        raise ValueError(f"execution evidence files are missing: {missing_files}")
    return resolved


def _verified_current_devices(
    execution_profile: ExecutionProfile,
) -> list[dict[str, Any]]:
    current_runtime = accelerator_snapshot()
    raw_devices = current_runtime.get("devices")
    devices = raw_devices if isinstance(raw_devices, list) else []
    matches = [
        device
        for device in devices
        if isinstance(device, dict)
        and device_matches_requirements(
            device,
            execution_profile,
            backend=current_runtime.get("backend"),
        )
    ]
    if current_runtime.get("accelerator_available") is not True or not matches:
        raise ValueError("execution evidence rejected: current runtime no longer matches contract")
    return matches


def verify_execution_evidence(
    *,
    config: ExperimentConfig,
    config_path: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    project_root: str | Path,
    preflight_path: str | Path | None,
    receipt_path: str | Path | None,
    source_manifest_path: str | Path | None,
    bundle_manifest_path: str | Path | None,
    execution_phase: str,
    compute_estimate_path: str | Path | None,
    planned_adapter_runs: int,
    expected_lineage_ids: Sequence[str],
) -> dict[str, Any]:
    """Authorize execution only from current, mutually bound evidence bytes."""

    resolved = _resolved_execution_evidence_paths(
        preflight_path=preflight_path,
        receipt_path=receipt_path,
        source_manifest_path=source_manifest_path,
        bundle_manifest_path=bundle_manifest_path,
        execution_phase=execution_phase,
    )

    config_file = Path(config_path).resolve()
    root = Path(project_root).resolve()
    if not config_file.is_file() or config_file.parent.parent != root:
        raise ValueError("execution config path is not the project config source")
    bound_profile = verify_execution_profile_binding(
        execution_profile,
        execution_profile_path,
        project_root=root,
    )

    source = verify_source_snapshot(root, resolved["source_manifest"])
    bundle = verify_bundle_snapshot(
        root,
        resolved["bundle_manifest"],
        source_tree_sha256=source["tree_sha256"],
        source_entry_count=source["entry_count"],
    )
    preflight = _load_json_object(resolved["preflight"], "strict preflight")
    receipt = _load_json_object(resolved["receipt"], "execution receipt")
    assessment = assess_live_preflight(preflight, expected_profile=bound_profile)
    if not assessment.preflight_complete:
        raise ValueError("execution evidence rejected: strict preflight is incomplete")
    if not assessment.required_device_observed:
        raise ValueError("execution evidence rejected: required device is not observed")
    if float(preflight["free_disk_gib"]) < config.runtime.minimum_free_disk_gib:
        raise ValueError("execution evidence rejected: current free disk is below the contract")

    current_matches = _verified_current_devices(bound_profile)

    expected_pilot = execution_phase == "pilot"
    expected_cohort = execution_phase == "cohort"
    if not live_evidence_matches(
        preflight,
        receipt,
        execution_profile=bound_profile,
        config_hash=config.config_hash,
        config_file_sha256=sha256_file(config_file),
        preflight_sha256=sha256_file(resolved["preflight"]),
        source_manifest_sha256=source["manifest_sha256"],
        source_tree_sha256=source["tree_sha256"],
        source_entry_count=source["entry_count"],
        bundle_manifest_sha256=bundle["manifest_sha256"],
        bundle_tree_sha256=bundle["tree_sha256"],
        bundle_entry_count=bundle["entry_count"],
        expected_representative_pilot=expected_pilot,
        expected_cohort_execution=expected_cohort,
        expected_lineage_ids=expected_lineage_ids,
    ):
        raise ValueError("execution evidence rejected: receipt binding or scope mismatch")
    compute_authorization = _verified_cohort_compute_authorization(
        execution_phase=execution_phase,
        compute_estimate_path=compute_estimate_path,
        receipt=receipt,
        config=config,
        planned_adapter_runs=planned_adapter_runs,
    )
    verify_execution_profile_binding(
        bound_profile,
        execution_profile_path,
        project_root=root,
    )
    result = {
        "status": "passed",
        "phase": execution_phase,
        "preflight_sha256": sha256_file(resolved["preflight"]),
        "receipt_sha256": sha256_file(resolved["receipt"]),
        "execution_profile": bound_profile.binding(),
        **source,
        "bundle_snapshot": bundle,
        "observed_devices": list(assessment.observed_devices),
        "current_device_count": len(current_matches),
    }
    if compute_authorization is not None:
        result["compute_authorization"] = compute_authorization
    return result


def _verify_pilot_compute_gate(
    *,
    max_lineages: int | None,
    compute_estimate_path: str | Path | None,
) -> dict[str, Any]:
    if type(max_lineages) is not int or not 1 <= max_lineages <= 3:
        raise ValueError(
            "pilot execution requires 1-3 paired lineages and permits at most 6 adapter runs"
        )
    if compute_estimate_path is not None:
        raise ValueError("pilot execution does not consume a cohort compute estimate")
    return {
        "phase": "pilot",
        "authorized_adapter_runs": max_lineages * 2,
        "estimate_sha256": None,
        "recommendation": "pilot_only",
    }


def _read_compute_estimate(source: Path) -> dict[str, Any]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("compute estimate is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("compute estimate must be a JSON object")
    return payload


def _canonical_evidence_relative_path(
    *, estimate_path: str | Path, ledger_path: str | Path, label: str
) -> str:
    evidence_root = Path(estimate_path).resolve().parent
    ledger = Path(ledger_path)
    if ledger.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    resolved = ledger.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    try:
        relative = resolved.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the compute evidence bundle") from exc
    return relative.as_posix()


def bind_observed_compute_evidence(
    estimate: dict[str, Any],
    *,
    estimate_path: str | Path,
    training_ledger_path: str | Path,
    evaluation_ledger_path: str | Path,
    pair_bindings: tuple[PilotPairBinding, ...],
) -> dict[str, Any]:
    """Bind an observed estimate to portable, byte-addressed pilot evidence."""

    if (
        estimate.get("schema_version") != COMPUTE_ESTIMATE_SCHEMA_VERSION
        or estimate.get("estimate_kind") != "observed_pilot"
    ):
        raise ValueError(
            f"only schema {COMPUTE_ESTIMATE_SCHEMA_VERSION} observed estimates can bind "
            "pilot evidence"
        )
    training = Path(training_ledger_path).resolve()
    evaluation = Path(evaluation_ledger_path).resolve()
    binding_payload = [
        asdict(binding) for binding in sorted(pair_bindings, key=lambda item: item.parent_config_id)
    ]
    return {
        **estimate,
        "pilot_training_ledger_path": _canonical_evidence_relative_path(
            estimate_path=estimate_path,
            ledger_path=training,
            label="pilot training ledger",
        ),
        "pilot_training_ledger_sha256": sha256_file(training),
        "pilot_evaluation_ledger_path": _canonical_evidence_relative_path(
            estimate_path=estimate_path,
            ledger_path=evaluation,
            label="pilot evaluation ledger",
        ),
        "pilot_evaluation_ledger_sha256": sha256_file(evaluation),
        "pilot_pair_bindings": binding_payload,
        "pilot_pair_binding_sha256": sha256_json(binding_payload),
    }


def _resolve_compute_ledger(
    estimate_path: Path, payload: dict[str, Any], *, field: str, label: str
) -> Path:
    raw = payload.get(field)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"compute estimate has invalid {label} path")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw
        or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
    ):
        raise ValueError(f"compute estimate has invalid {label} path")
    evidence_root = estimate_path.resolve().parent
    candidate = evidence_root.joinpath(*relative.parts)
    cursor = evidence_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"compute estimate {label} path cannot traverse a symbolic link")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(evidence_root) or not resolved.is_file():
        raise ValueError(f"compute estimate {label} file is missing or outside its bundle")
    return resolved


def _require_integer_field(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int:
        raise ValueError(f"compute estimate has invalid {field}")
    return value


def _verify_pilot_failure_summary(payload: dict[str, Any]) -> None:
    failed_pairs = _require_integer_field(payload, "pilot_failed_pair_count")
    threshold_failures = _require_integer_field(
        payload,
        "pilot_threshold_failure_pair_count",
    )
    integrity_failures = _require_integer_field(
        payload,
        "pilot_integrity_failure_pair_count",
    )
    if min(failed_pairs, threshold_failures, integrity_failures) < 0:
        raise ValueError("compute estimate has a negative pilot failure count")
    if threshold_failures + integrity_failures != failed_pairs:
        raise ValueError("compute estimate has an inconsistent pilot failure classification")
    reason_codes = payload.get("pilot_qc_reason_codes")
    if (
        not isinstance(reason_codes, list)
        or any(not isinstance(code, str) or not code for code in reason_codes)
        or reason_codes != sorted(set(reason_codes))
        or bool(reason_codes) != bool(failed_pairs)
    ):
        raise ValueError("compute estimate has an invalid pilot QC reason-code summary")
    if integrity_failures:
        raise ValueError("cohort execution refuses a pilot with integrity QC failures")


def _verify_compute_estimate_header(
    payload: dict[str, Any],
    config_sha256: str,
    execution_profile: ExecutionProfile,
) -> tuple[int, int]:
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != COMPUTE_ESTIMATE_SCHEMA_VERSION
    ):
        raise ValueError("compute estimate schema mismatch")
    if payload.get("scope") != COMPUTE_ESTIMATE_SCOPE:
        raise ValueError("compute estimate scope mismatch")
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("compute estimate config hash mismatch")
    if payload.get("estimate_kind") != "observed_pilot":
        raise ValueError("cohort execution requires an observed pilot estimate")
    if not execution_profile_binding_matches(payload.get("execution_profile"), execution_profile):
        raise ValueError("compute estimate execution profile mismatch")

    pilot_count = _require_integer_field(payload, "pilot_run_count")
    pilot_pairs = _require_integer_field(payload, "pilot_pair_count")
    if pilot_count not in {2, 4, 6} or pilot_pairs * 2 != pilot_count:
        raise ValueError("cohort execution requires complete pilot adapter pairs")
    if _require_integer_field(payload, "pilot_evaluation_pair_count") != pilot_pairs:
        raise ValueError("cohort execution requires evaluation telemetry for every pilot pair")
    _verify_pilot_failure_summary(payload)
    ledger_hash_fields = (
        "pilot_training_ledger_sha256",
        "pilot_evaluation_ledger_sha256",
    )
    if any(not is_sha256(payload.get(field)) for field in ledger_hash_fields):
        raise ValueError("compute estimate has invalid pilot ledger hashes")
    return pilot_count, pilot_pairs


def _verify_compute_source_binding(
    payload: dict[str, Any],
    *,
    expected_source_tree_sha256: str | None,
    expected_source_manifest_sha256: str | None,
) -> None:
    if not is_sha256(expected_source_tree_sha256) or not is_sha256(expected_source_manifest_sha256):
        raise ValueError("cohort execution requires current source bindings")
    if payload.get("source_tree_sha256") != expected_source_tree_sha256:
        raise ValueError("compute estimate source tree hash mismatch")
    if payload.get("source_manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("compute estimate source manifest hash mismatch")


def _validated_compute_pair_binding(raw_binding: Any) -> dict[str, str]:
    expected_fields = {
        "parent_config_id",
        "clean_adapter_sha256",
        "backdoored_adapter_sha256",
    }
    if not isinstance(raw_binding, dict) or set(raw_binding) != expected_fields:
        raise ValueError("compute estimate has invalid pilot pair bindings")
    parent = raw_binding.get("parent_config_id")
    clean_hash = raw_binding.get("clean_adapter_sha256")
    backdoored_hash = raw_binding.get("backdoored_adapter_sha256")
    if (
        not isinstance(parent, str)
        or not parent
        or not is_sha256(clean_hash)
        or not is_sha256(backdoored_hash)
        or clean_hash == backdoored_hash
    ):
        raise ValueError("compute estimate has invalid pilot pair bindings")
    return {
        "parent_config_id": parent,
        "clean_adapter_sha256": clean_hash,
        "backdoored_adapter_sha256": backdoored_hash,
    }


def _verify_compute_pair_bindings(payload: dict[str, Any], pilot_pairs: int) -> None:
    raw_bindings = payload.get("pilot_pair_bindings")
    if not isinstance(raw_bindings, list) or len(raw_bindings) != pilot_pairs:
        raise ValueError("compute estimate has invalid pilot pair bindings")
    bindings = sorted(
        (_validated_compute_pair_binding(raw_binding) for raw_binding in raw_bindings),
        key=lambda item: item["parent_config_id"],
    )
    parent_ids = [binding["parent_config_id"] for binding in bindings]
    adapter_hashes = [
        binding[field]
        for binding in bindings
        for field in ("clean_adapter_sha256", "backdoored_adapter_sha256")
    ]
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("compute estimate contains duplicate lineage bindings")
    if len(set(adapter_hashes)) != len(adapter_hashes):
        raise ValueError("compute estimate reuses adapter hashes across lineages")
    if payload.get("pilot_pair_binding_sha256") != sha256_json(bindings):
        raise ValueError("compute estimate pilot pair binding hash mismatch")


def _positive_json_number(value: Any, message: str) -> float:
    if type(value) not in {int, float} or not float(value) > 0.0 or not math.isfinite(value):
        raise ValueError(message)
    return float(value)


def _ordered_memory_summary(
    payload: dict[str, Any],
    *,
    metric: str,
    unit: str,
) -> tuple[float, float, float]:
    values = tuple(
        _positive_json_number(
            payload.get(f"observed_{metric}_per_{unit}_{quantile}"),
            "cohort execution requires complete positive peak-memory evidence",
        )
        for quantile in ("p50", "p90", "max")
    )
    if not values[0] <= values[1] <= values[2]:
        raise ValueError("compute estimate has unordered peak-memory evidence")
    return values


def _verify_compute_memory_evidence(payload: dict[str, Any]) -> None:
    for unit in ("adapter", "evaluation_pair"):
        allocated = _ordered_memory_summary(
            payload,
            metric="peak_allocated_memory_gib",
            unit=unit,
        )
        reserved = _ordered_memory_summary(
            payload,
            metric="peak_reserved_memory_gib",
            unit=unit,
        )
        if any(
            allocated_value > reserved_value
            for allocated_value, reserved_value in zip(allocated, reserved, strict=True)
        ):
            raise ValueError("compute estimate has inconsistent peak-memory evidence")
        _positive_json_number(
            payload.get(f"observed_device_memory_gib_per_{unit}_min"),
            "cohort execution requires positive device-memory evidence",
        )
        allocated_fraction = _positive_json_number(
            payload.get(f"observed_peak_allocated_memory_fraction_per_{unit}_max"),
            "cohort execution requires positive memory-fraction evidence",
        )
        reserved_fraction = _positive_json_number(
            payload.get(f"observed_peak_reserved_memory_fraction_per_{unit}_max"),
            "cohort execution requires positive memory-fraction evidence",
        )
        if not allocated_fraction <= reserved_fraction <= 1.0:
            raise ValueError("compute estimate has invalid memory-fraction evidence")


def _recompute_observed_estimate(
    *,
    source: Path,
    payload: dict[str, Any],
    config_sha256: str,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> None:
    training_path = _resolve_compute_ledger(
        source,
        payload,
        field="pilot_training_ledger_path",
        label="pilot training ledger",
    )
    evaluation_path = _resolve_compute_ledger(
        source,
        payload,
        field="pilot_evaluation_ledger_path",
        label="pilot evaluation ledger",
    )
    expected_training_hash = payload["pilot_training_ledger_sha256"]
    expected_evaluation_hash = payload["pilot_evaluation_ledger_sha256"]
    if sha256_file(training_path) != expected_training_hash:
        raise ValueError("pilot training ledger hash mismatch")
    if sha256_file(evaluation_path) != expected_evaluation_hash:
        raise ValueError("pilot evaluation ledger hash mismatch")

    training = read_pilot_telemetry(
        training_path,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256=source_tree_sha256,
        expected_source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    evaluation = read_pilot_evaluation_telemetry(
        evaluation_path,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256=source_tree_sha256,
        expected_source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    bindings = verify_pilot_telemetry_alignment(training, evaluation)
    if (
        sha256_file(training_path) != expected_training_hash
        or sha256_file(evaluation_path) != expected_evaluation_hash
    ):
        raise ValueError("pilot ledger changed while compute evidence was being verified")

    raw_available_hours = payload.get("available_device_hours")
    available_hours = (
        None
        if raw_available_hours is None
        else _positive_json_number(
            raw_available_hours,
            "compute estimate has an invalid declared device-hour budget",
        )
    )
    expected = estimate_compute(
        pilot_device_hours=list(training.device_hours),
        adapter_size_gib=_positive_json_number(
            payload.get("adapter_size_floor_gib"),
            "compute estimate has invalid adapter-size floor",
        ),
        available_device_hours=available_hours,
        config_sha256=config_sha256,
        observed_adapter_sizes_gib=list(training.adapter_sizes_gib),
        pilot_evaluation_device_hours=list(evaluation.device_hours),
        pilot_failed_pair_count=evaluation.failed_pair_count,
        pilot_threshold_failure_pair_count=evaluation.threshold_failure_pair_count,
        pilot_integrity_failure_pair_count=evaluation.integrity_failure_pair_count,
        pilot_qc_reason_codes=list(evaluation.failure_reason_codes),
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        pilot_peak_allocated_memory_gib=list(training.peak_allocated_memory_gib),
        pilot_peak_reserved_memory_gib=list(training.peak_reserved_memory_gib),
        pilot_device_memory_gib=list(training.device_memory_gib),
        pilot_evaluation_peak_allocated_memory_gib=list(evaluation.peak_allocated_memory_gib),
        pilot_evaluation_peak_reserved_memory_gib=list(evaluation.peak_reserved_memory_gib),
        pilot_evaluation_device_memory_gib=list(evaluation.device_memory_gib),
        execution_profile=execution_profile,
    )
    expected = bind_observed_compute_evidence(
        expected,
        estimate_path=source,
        training_ledger_path=training_path,
        evaluation_ledger_path=evaluation_path,
        pair_bindings=bindings,
    )
    if not same_json_value(payload, expected):
        fields = sorted(
            field
            for field in set(payload) | set(expected)
            if field not in payload
            or field not in expected
            or not same_json_value(payload[field], expected[field])
        )
        raise ValueError(
            "compute estimate does not match ledger-derived evidence: " + ", ".join(fields)
        )


def _verify_compute_budget(
    *,
    payload: dict[str, Any],
    profile: str,
    planned_adapter_runs: int,
) -> tuple[float, str]:
    if profile not in {"core", "target"}:
        raise ValueError("compute profile must be core or target")
    if type(planned_adapter_runs) is not int or planned_adapter_runs <= 0:
        raise ValueError("planned adapter runs must be a positive integer")
    available = _positive_json_number(
        payload.get("available_device_hours"),
        "cohort execution requires a positive confirmed device-hour budget",
    )
    profile_estimate = payload.get(profile)
    if not isinstance(profile_estimate, dict):
        raise ValueError(f"compute estimate has no {profile!r} profile")
    adapter_count = _require_integer_field(profile_estimate, "adapter_count")
    if adapter_count != planned_adapter_runs:
        raise ValueError("compute estimate adapter count does not match the locked plan")
    pilot_pairs = _require_integer_field(payload, "pilot_pair_count")
    if pilot_pairs != REQUIRED_REPRESENTATIVE_PILOT_PAIRS:
        raise ValueError(
            "cohort execution requires three representative pilot pairs before authorization"
        )
    if profile == "core":
        full_research = payload.get("full_research")
        if not isinstance(full_research, dict):
            raise ValueError("compute estimate has no full-research stage ledger")
        planning_hours = _positive_json_number(
            full_research.get("planning_device_hours"),
            "compute estimate has invalid full-research planning hours",
        )
    else:
        planning_hours = _positive_json_number(
            profile_estimate.get("planning_device_hours"),
            "compute estimate has invalid planning hours",
        )
    if planning_hours > 0.8 * available:
        raise ValueError("compute estimate exceeds the confirmed budget gate")
    recommendation = payload.get("recommendation")
    authorized = (
        recommendation == "full_research"
        if profile == "core"
        else payload.get("cohort_recommendation") == "target"
    )
    if not authorized:
        raise ValueError(
            f"compute recommendation {recommendation!r} does not authorize {profile!r}"
        )
    return available, recommendation


def verify_execution_compute_gate(
    *,
    execution_profile: ExecutionProfile,
    profile: str,
    execution_phase: str,
    max_lineages: int | None,
    compute_estimate_path: str | Path | None,
    config_sha256: str,
    planned_adapter_runs: int,
    expected_source_tree_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed before any pilot or cohort training side effect."""

    if not _valid_execution_profile_object(execution_profile):
        raise ValueError("execution profile object is invalid")
    if execution_phase == "pilot":
        result = _verify_pilot_compute_gate(
            max_lineages=max_lineages,
            compute_estimate_path=compute_estimate_path,
        )
        return {**result, "execution_profile": execution_profile.binding()}
    if execution_phase != "cohort":
        raise ValueError("execution_phase must be pilot or cohort when execute=true")
    if compute_estimate_path is None:
        raise ValueError("cohort execution requires --compute-estimate")

    source = Path(compute_estimate_path)
    payload = verify_observed_compute_evidence(
        compute_estimate_path=source,
        execution_profile=execution_profile,
        config_sha256=config_sha256,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    pilot_count = int(payload["pilot_run_count"])
    pilot_pairs = int(payload["pilot_pair_count"])
    available, recommendation = _verify_compute_budget(
        payload=payload,
        profile=profile,
        planned_adapter_runs=planned_adapter_runs,
    )
    return {
        "phase": "cohort",
        "authorized_adapter_runs": planned_adapter_runs,
        "estimate_sha256": sha256_file(source),
        "recommendation": recommendation,
        "available_device_hours": available,
        "pilot_run_count": pilot_count,
        "pilot_pair_count": pilot_pairs,
        "source_tree_sha256": expected_source_tree_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "execution_profile": execution_profile.binding(),
    }


def verify_observed_compute_evidence(
    *,
    compute_estimate_path: str | Path,
    execution_profile: ExecutionProfile,
    config_sha256: str,
    expected_source_tree_sha256: str | None,
    expected_source_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Rebuild an observed estimate from its immutable pilot ledgers."""

    if not _valid_execution_profile_object(execution_profile):
        raise ValueError("execution profile object is invalid")
    source = Path(compute_estimate_path)
    payload = _read_compute_estimate(source)
    _, pilot_pairs = _verify_compute_estimate_header(
        payload,
        config_sha256,
        execution_profile,
    )
    _verify_compute_memory_evidence(payload)
    _verify_compute_source_binding(
        payload,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    _verify_compute_pair_bindings(payload, pilot_pairs)
    _recompute_observed_estimate(
        source=source,
        payload=payload,
        config_sha256=config_sha256,
        source_tree_sha256=str(expected_source_tree_sha256),
        source_manifest_sha256=str(expected_source_manifest_sha256),
        execution_profile=execution_profile,
    )
    return payload


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {label}: expected a JSON object")
    return payload
