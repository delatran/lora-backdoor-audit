# ruff: noqa: E402
"""Run and record all locally observable execution-readiness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lora_audit.config import load_config
from lora_audit.execution_profiles import (
    ExecutionProfile,
    execution_profile_binding_matches,
    load_execution_profile,
)
from lora_audit.hashing import sha256_json
from lora_audit.manifests import select_source_delta_path, source_delta_matches
from lora_audit.preflight import ACCELERATED_PINNED_PACKAGES
from lora_audit.readiness import (
    assess_live_preflight,
    live_evidence_matches,
    select_live_evidence_paths,
    verify_bundle_snapshot,
    verify_observed_compute_evidence,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(root: Path, name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    environment = os.environ.copy()
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), inherited_pythonpath) if part
    )
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _gate(
    gate_id: str,
    requirement: str,
    status: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
    }


def _command_status(commands: dict[str, dict[str, Any]], *names: str) -> str:
    return "passed" if all(commands[name]["exit_code"] == 0 for name in names) else "failed"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _pilot_binding_contract_valid(compute: dict[str, Any]) -> bool:
    pair_count = compute.get("pilot_pair_count")
    bindings = compute.get("pilot_pair_bindings")
    if type(pair_count) is not int or not 1 <= pair_count <= 3 or not isinstance(bindings, list):
        return False
    if len(bindings) != pair_count or any(not isinstance(item, dict) for item in bindings):
        return False
    required = {
        "parent_config_id",
        "clean_adapter_sha256",
        "backdoored_adapter_sha256",
    }
    if any(set(item) != required for item in bindings):
        return False
    canonical = sorted(bindings, key=lambda item: str(item["parent_config_id"]))
    if len({item["parent_config_id"] for item in canonical}) != pair_count:
        return False
    if any(
        not isinstance(item["parent_config_id"], str)
        or not item["parent_config_id"]
        or not _valid_sha256(item["clean_adapter_sha256"])
        or not _valid_sha256(item["backdoored_adapter_sha256"])
        for item in canonical
    ):
        return False
    adapter_hashes = [
        item[field]
        for item in canonical
        for field in ("clean_adapter_sha256", "backdoored_adapter_sha256")
    ]
    if len(set(adapter_hashes)) != len(adapter_hashes):
        return False
    return (
        _valid_sha256(compute.get("pilot_training_ledger_sha256"))
        and _valid_sha256(compute.get("pilot_evaluation_ledger_sha256"))
        and isinstance(compute.get("pilot_training_ledger_path"), str)
        and bool(compute.get("pilot_training_ledger_path"))
        and isinstance(compute.get("pilot_evaluation_ledger_path"), str)
        and bool(compute.get("pilot_evaluation_ledger_path"))
        and compute.get("pilot_pair_binding_sha256") == sha256_json(canonical)
    )


def _positive_json_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and float(value) > 0.0


def _run_local_commands(
    root: Path,
    python: str,
    fixture_root: Path,
) -> dict[str, dict[str, Any]]:
    fixture_data = fixture_root / "data/manifest.jsonl"
    command_specs = (
        (
            "source_manifest",
            [
                python,
                "scripts/build_source_manifest.py",
                "--verify",
                "--output",
                "artifacts/source-manifest.json",
            ],
        ),
        (
            "notebook_fixture",
            [python, "scripts/execute_notebook.py"],
        ),
        (
            "fixture_preflight",
            [
                python,
                "-m",
                "lora_audit.cli",
                "preflight",
                "--config",
                "configs/fixture.yaml",
                "--mode",
                "fixture",
                "--output",
                "artifacts/preflight-fixture.json",
            ],
        ),
        (
            "fixture_smoke",
            [
                python,
                "-m",
                "lora_audit.cli",
                "smoke",
                "--config",
                "configs/fixture.yaml",
                "--output-root",
                str(fixture_root),
            ],
        ),
        (
            "validator",
            [python, "scripts/validate_project.py", "--require-fixture-artifacts"],
        ),
        (
            "cohort_preview",
            [
                python,
                "-m",
                "lora_audit.cli",
                "run-cohort",
                "--config",
                "configs/core.yaml",
                "--execution-profile",
                "configs/execution/a100-40gb.yaml",
                "--bundle-manifest",
                "artifacts/bundle-manifest.json",
                "--prepared-manifest",
                str(fixture_data),
                "--output-root",
                "artifacts/cache/readiness-preview",
                "--max-lineages",
                "1",
            ],
        ),
        ("ruff", [python, "-m", "ruff", "check", "--no-cache", "."]),
        ("pytest", [python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
    )
    return {name: _run(root, name, command) for name, command in command_specs}


def _local_preflight_support(
    root: Path,
    python: str,
) -> tuple[dict[str, Any], str]:
    local_preflight_path = root / "artifacts/preflight-local.json"
    command = _run(
        root,
        "strict_preflight_local",
        [
            python,
            "-m",
            "lora_audit.cli",
            "preflight",
            "--config",
            "configs/core.yaml",
            "--mode",
            "strict",
            "--execution-profile",
            str(root / "configs/execution/a100-40gb.yaml"),
            "--output",
            str(local_preflight_path),
        ],
    )
    local_preflight = _load_json(local_preflight_path)
    failures = local_preflight.get("failures") if local_preflight is not None else None
    if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
        return command, "failed"
    external_prefixes = (
        "accelerator_unavailable",
        "required_device_not_observed",
        "insufficient_disk:",
    )

    def external_runtime_failure(failure: str) -> bool:
        if failure.startswith(external_prefixes):
            return True
        return any(
            failure == f"missing_package:{package}"
            or failure.startswith(f"package_version_mismatch:{package}:")
            for package in ACCELERATED_PINNED_PACKAGES
        )

    local_failures = [failure for failure in failures if not external_runtime_failure(failure)]
    if local_failures:
        return command, "failed"
    return command, "unverified_external" if failures else "passed"


def _external_preflight_assessment(
    root: Path,
    core_path: Path,
    core_config_hash: str,
    source_manifest_path: Path,
    source_manifest: dict[str, Any] | None,
    execution_profile: ExecutionProfile,
) -> Any | None:
    external_preflight_path, external_receipt_path = select_live_evidence_paths(root)
    external_preflight = _load_json(external_preflight_path)
    external_receipt = _load_json(external_receipt_path)
    if external_preflight is None or external_receipt is None or source_manifest is None:
        return None
    source_tree = source_manifest.get("tree_sha256")
    entry_count = source_manifest.get("entry_count")
    if not _valid_sha256(source_tree) or type(entry_count) is not int or entry_count <= 0:
        return None
    try:
        bundle = verify_bundle_snapshot(
            root,
            root / "artifacts/bundle-manifest.json",
            source_tree_sha256=source_tree,
            source_entry_count=entry_count,
        )
    except (OSError, ValueError):
        return None
    if not live_evidence_matches(
        external_preflight,
        external_receipt,
        execution_profile=execution_profile,
        config_hash=core_config_hash,
        config_file_sha256=_sha256_file(core_path),
        preflight_sha256=_sha256_file(external_preflight_path),
        source_manifest_sha256=_sha256_file(source_manifest_path),
        source_tree_sha256=source_tree,
        source_entry_count=entry_count,
        bundle_manifest_sha256=bundle["manifest_sha256"],
        bundle_tree_sha256=bundle["tree_sha256"],
        bundle_entry_count=bundle["entry_count"],
        expected_representative_pilot=True,
        expected_cohort_execution=False,
    ):
        return None
    return assess_live_preflight(external_preflight, expected_profile=execution_profile)


def _source_sync_evidence(
    root: Path,
    source_manifest: dict[str, Any] | None,
) -> tuple[str, Path | None, str, str]:
    source_receipt = _load_json(root / "artifacts/source-receipt.json")
    source_tree = source_manifest.get("tree_sha256") if source_manifest else None
    entry_count = source_manifest.get("entry_count") if source_manifest else None
    verified_tree = source_receipt.get("verified_tree_sha256") if source_receipt else None
    verified_count = source_receipt.get("verified_entry_count") if source_receipt else None
    folder_id = source_receipt.get("folder_id") if source_receipt else None
    receipt_passed = bool(
        source_receipt
        and source_receipt.get("verification", {}).get("status") == "passed"
        and _valid_sha256(source_tree)
        and _valid_sha256(verified_tree)
        and verified_tree == source_tree
        and type(entry_count) is int
        and entry_count > 0
        and type(verified_count) is int
        and verified_count == entry_count
        and isinstance(folder_id, str)
        and folder_id.strip() == folder_id
        and bool(folder_id)
    )
    source_sync_status = "passed" if receipt_passed else "unverified_external"
    delta_path = (
        select_source_delta_path(
            root,
            baseline_tree_sha256=verified_tree,
            target_tree_sha256=source_tree,
        )
        if _valid_sha256(verified_tree) and _valid_sha256(source_tree)
        else None
    )
    delta_payload = _load_json(delta_path) if delta_path is not None else None
    delta_valid = bool(
        delta_payload
        and source_manifest
        and source_receipt
        and isinstance(folder_id, str)
        and folder_id.strip() == folder_id
        and bool(folder_id)
        and type(entry_count) is int
        and entry_count > 0
        and source_delta_matches(
            delta_payload,
            folder_id=folder_id,
            baseline_tree_sha256=verified_tree,
            target_tree_sha256=source_tree,
            target_entry_count=entry_count,
        )
    )
    return (
        source_sync_status,
        delta_path,
        "passed" if delta_valid else "unverified_external",
        source_tree if _valid_sha256(source_tree) else "",
    )


def _compute_contract_valid(
    compute: dict[str, Any] | None,
    *,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> bool:
    if compute is None:
        return False
    core = compute.get("core")
    target = compute.get("target")
    full_research = compute.get("full_research")
    return bool(
        type(compute.get("schema_version")) is int
        and compute.get("schema_version") == 9
        and compute.get("scope") == "full_research_pipeline"
        and _valid_sha256(expected_source_tree_sha256)
        and compute.get("source_tree_sha256") == expected_source_tree_sha256
        and _valid_sha256(expected_source_manifest_sha256)
        and compute.get("source_manifest_sha256") == expected_source_manifest_sha256
        and execution_profile_binding_matches(compute.get("execution_profile"), execution_profile)
        and _positive_json_number(compute.get("adapter_size_floor_gib"))
        and _positive_json_number(compute.get("planning_device_hours_per_adapter"))
        and _memory_contract_valid(compute)
        and isinstance(core, dict)
        and type(core.get("adapter_count")) is int
        and core.get("adapter_count") == 72
        and type(core.get("pair_count")) is int
        and core.get("pair_count") == 36
        and isinstance(target, dict)
        and type(target.get("adapter_count")) is int
        and target.get("adapter_count") == 120
        and type(target.get("pair_count")) is int
        and target.get("pair_count") == 60
        and isinstance(full_research, dict)
        and _positive_json_number(full_research.get("planning_device_hours"))
    )


def _memory_contract_valid(compute: dict[str, Any]) -> bool:
    for unit in ("adapter", "evaluation_pair"):
        allocated = [
            compute.get(f"observed_peak_allocated_memory_gib_per_{unit}_{quantile}")
            for quantile in ("p50", "p90", "max")
        ]
        reserved = [
            compute.get(f"observed_peak_reserved_memory_gib_per_{unit}_{quantile}")
            for quantile in ("p50", "p90", "max")
        ]
        capacity = compute.get(f"observed_device_memory_gib_per_{unit}_min")
        allocated_fraction = compute.get(f"observed_peak_allocated_memory_fraction_per_{unit}_max")
        reserved_fraction = compute.get(f"observed_peak_reserved_memory_fraction_per_{unit}_max")
        if not all(
            _positive_json_number(value)
            for value in (*allocated, *reserved, capacity, allocated_fraction, reserved_fraction)
        ):
            return False
        if not (
            allocated[0] <= allocated[1] <= allocated[2]
            and reserved[0] <= reserved[1] <= reserved[2]
            and all(
                allocated_value <= reserved_value
                for allocated_value, reserved_value in zip(allocated, reserved, strict=True)
            )
            and allocated_fraction <= reserved_fraction <= 1.0
        ):
            return False
    return True


def _observed_pilot_status(
    compute: dict[str, Any] | None,
    compute_path: Path,
    core_config_hash: str,
    *,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> str:
    if (
        not _compute_contract_valid(
            compute,
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            execution_profile=execution_profile,
        )
        or compute is None
    ):
        return "unverified_external"
    pair_count = compute.get("pilot_pair_count")
    evaluation_pair_count = compute.get("pilot_evaluation_pair_count")
    failed_pair_count = compute.get("pilot_failed_pair_count")
    threshold_failure_count = compute.get("pilot_threshold_failure_pair_count")
    integrity_failure_count = compute.get("pilot_integrity_failure_pair_count")
    reason_codes = compute.get("pilot_qc_reason_codes")
    valid = bool(
        compute.get("estimate_kind") == "observed_pilot"
        and compute.get("config_sha256") == core_config_hash
        and type(pair_count) is int
        and pair_count == 3
        and type(evaluation_pair_count) is int
        and evaluation_pair_count == pair_count
        and type(failed_pair_count) is int
        and failed_pair_count >= 0
        and type(threshold_failure_count) is int
        and threshold_failure_count >= 0
        and type(integrity_failure_count) is int
        and integrity_failure_count == 0
        and failed_pair_count == threshold_failure_count + integrity_failure_count
        and isinstance(reason_codes, list)
        and all(isinstance(code, str) and bool(code) for code in reason_codes)
        and reason_codes == sorted(set(reason_codes))
        and bool(reason_codes) == bool(failed_pair_count)
        and compute.get("recommendation") == "full_research"
        and compute.get("required_representative_pilot_pair_count") == 3
        and _pilot_binding_contract_valid(compute)
    )
    if valid:
        try:
            verify_observed_compute_evidence(
                compute_estimate_path=compute_path,
                execution_profile=execution_profile,
                config_sha256=core_config_hash,
                expected_source_tree_sha256=expected_source_tree_sha256,
                expected_source_manifest_sha256=expected_source_manifest_sha256,
            )
        except (OSError, ValueError):
            valid = False
    return "passed" if valid else "unverified_external"


def _readiness_state(gates: list[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    failed = [gate["id"] for gate in gates if gate["status"] == "failed"]
    external = [gate["id"] for gate in gates if gate["status"] == "unverified_external"]
    if failed:
        return "local_gate_failed", failed, external
    if external:
        return "awaiting_external_evidence", failed, external
    return "ready", failed, external


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run locally observable execution-readiness gates.",
        allow_abbrev=False,
    )
    parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    core_path = root / "configs/core.yaml"
    core = load_config(core_path)
    execution_profile = load_execution_profile(root / "configs/execution/a100-40gb.yaml")
    source_manifest_path = root / "artifacts/source-manifest.json"
    source_manifest = _load_json(source_manifest_path)
    fixture = load_config(root / "configs/fixture.yaml")
    fixture_root = root / f"artifacts/runs/fixture-smoke-{fixture.config_hash[:12]}"
    commands = _run_local_commands(root, python, fixture_root)
    local_preflight_command, local_preflight_support = _local_preflight_support(root, python)
    commands[local_preflight_command["name"]] = local_preflight_command
    assessment = _external_preflight_assessment(
        root,
        core_path,
        core.config_hash,
        source_manifest_path,
        source_manifest,
        execution_profile,
    )
    source_sync_status, delta_path, delta_status, source_tree = _source_sync_evidence(
        root,
        source_manifest,
    )
    compute_path = root / "artifacts/compute-estimate.json"
    observed_pilot_status = _observed_pilot_status(
        _load_json(compute_path),
        compute_path,
        core.config_hash,
        expected_source_tree_sha256=source_tree,
        expected_source_manifest_sha256=(
            _sha256_file(source_manifest_path) if source_manifest is not None else ""
        ),
        execution_profile=execution_profile,
    )

    gates = [
        _gate(
            "RG-01",
            "Current source passes lint, tests, deterministic project validation, and preview.",
            _command_status(commands, "ruff", "pytest", "validator", "cohort_preview"),
            ["command:ruff", "command:pytest", "command:validator", "command:cohort_preview"],
        ),
        _gate(
            "RG-02",
            "The canonical fixture preflight passes without training or external execution.",
            _command_status(commands, "fixture_preflight", "fixture_smoke"),
            [
                "command:fixture_preflight",
                "artifacts/preflight-fixture.json",
                "command:fixture_smoke",
                fixture_root.relative_to(root).as_posix(),
            ],
        ),
        _gate(
            "RG-03",
            "The notebook fixture path executes without an external side effect.",
            _command_status(commands, "notebook_fixture"),
            ["command:notebook_fixture", "notebooks/experiment_runner.ipynb"],
        ),
        _gate(
            "RG-04",
            "The persisted source and bundle snapshots are hash-bound and content-clean.",
            _command_status(commands, "source_manifest", "validator"),
            [
                "artifacts/source-manifest.json",
                "artifacts/bundle-manifest.json",
                "command:validator",
            ],
        ),
        _gate(
            "RG-05",
            "The selected execution runtime realizes accelerated dependencies and device gates.",
            local_preflight_support,
            ["artifacts/preflight-local.json"],
        ),
        _gate(
            "RG-06",
            "A hash-bound external strict preflight is complete.",
            "passed" if assessment and assessment.preflight_complete else "unverified_external",
            ["artifacts/preflight-live.json", "artifacts/execution-receipt.json"],
        ),
        _gate(
            "RG-07",
            "The execution framework observes an active device.",
            "passed" if assessment and assessment.runtime_observed else "unverified_external",
            list(assessment.observed_devices) if assessment else [],
        ),
        _gate(
            "RG-08",
            "The observed device matches the locked capability and memory contract.",
            (
                "passed"
                if assessment and assessment.required_device_observed
                else "unverified_external"
            ),
            list(assessment.observed_devices) if assessment else [],
        ),
        _gate(
            "RG-09",
            (
                "The three-pair representative pilot produced a full-pipeline, "
                "config-bound observed estimate."
            ),
            observed_pilot_status,
            ["artifacts/compute-estimate.json"],
        ),
        _gate(
            "RG-10",
            "The external source snapshot matches the current manifest-bound tree.",
            source_sync_status,
            ["artifacts/source-manifest.json", "artifacts/source-receipt.json"],
        ),
        _gate(
            "RG-11",
            "Any pending source mutation is bounded, hash-locked, and no-write.",
            "passed" if source_sync_status == "passed" else delta_status,
            [delta_path.relative_to(root).as_posix()] if delta_path else [],
        ),
    ]

    readiness_state, failed, external = _readiness_state(gates)

    study_contract_path = root / "README.md"
    protocol_config_path = root / "configs/core.yaml"
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "readiness_state": readiness_state,
        "failed_gates": failed,
        "unverified_external_gates": external,
        "gate_counts": {
            status: sum(gate["status"] == status for gate in gates)
            for status in ("passed", "failed", "unverified_external")
        },
        "study_contract": {
            "path": study_contract_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(study_contract_path),
        },
        "protocol_config": {
            "path": protocol_config_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(protocol_config_path),
        },
        "current_config_hash": core.config_hash,
        "execution_profile": execution_profile.binding(),
        "current_source_tree_sha256": source_tree or None,
        "gates": gates,
        "commands": commands,
    }
    output = root / "artifacts/readiness-audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "readiness_state": readiness_state,
                "failed_gates": failed,
                "unverified_external_gates": external,
                "gate_counts": report["gate_counts"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
