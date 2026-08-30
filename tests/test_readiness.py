import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from lora_audit.execution_profiles import (
    canonical_execution_profile_path,
    is_locked_execution_profile,
    load_execution_profile,
)
from lora_audit.hashing import is_sha256, sha256_json
from lora_audit.manifests import select_source_delta_path, source_delta_matches
from lora_audit.preflight import (
    ACCELERATED_PINNED_PACKAGES,
    BASE_PINNED_PACKAGES,
    device_matches_requirements,
)
from lora_audit.readiness import (
    assess_live_preflight,
    build_source_snapshot,
    live_evidence_matches,
    select_live_evidence_paths,
    verify_bundle_snapshot,
    verify_execution_compute_gate,
    verify_source_snapshot,
)


def _profile(project_root: Path, profile_id: str = "l4-24gb-v1"):
    return load_execution_profile(canonical_execution_profile_path(project_root, profile_id))


def _bf16_probe(*, passed: bool = True, index: int = 0) -> dict[str, Any]:
    probe = {
        "attempted": True,
        "passed": passed,
        "finite": passed,
        "dtype": "torch.bfloat16",
        "device_type": "cuda",
        "device_index": index,
    }
    if not passed:
        probe["error"] = "nonfinite_tensor_output"
    return probe


def _device(
    name: Any,
    memory_gib: Any,
    *,
    capability: Any,
    bf16_passed: bool = True,
    index: Any = 0,
) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "total_memory_gib": memory_gib,
        "compute_capability": (list(capability) if isinstance(capability, tuple) else capability),
        "bf16_probe": _bf16_probe(
            passed=bf16_passed,
            index=index if type(index) is int else 0,
        ),
    }


def _payload(
    profile,
    name: Any,
    memory_gib: Any,
    *,
    capability: Any = (8, 9),
    accelerator_available: bool = True,
    backend: str = "cuda",
    bf16_passed: bool = True,
) -> dict:
    return {
        "status": "passed",
        "mode": "strict",
        "fixture_smoke_only": False,
        "failures": [],
        "execution_profile": profile.binding(),
        "accelerator": {
            "framework_available": True,
            "accelerator_available": accelerator_available,
            "framework_version": "2.13.0",
            "backend": backend,
            "runtime_version": "13.0",
            "devices": [
                _device(
                    name,
                    memory_gib,
                    capability=capability,
                    bf16_passed=bf16_passed,
                )
            ],
        },
        "python": "3.12.13",
        "platform": "test-platform",
        "packages": {**BASE_PINNED_PACKAGES, **ACCELERATED_PINNED_PACKAGES},
        "free_disk_gib": 55.0,
        "artifact_writable": True,
    }


def test_locked_profiles_have_distinct_semantic_and_file_hashes(
    project_root: Path,
) -> None:
    l4 = _profile(project_root)
    a100 = _profile(project_root, "a100-40gb-v1")

    assert l4.required_compute_capability == (8, 9)
    assert (l4.minimum_device_memory_gib, l4.maximum_device_memory_gib) == (
        21.0,
        26.0,
    )
    assert a100.required_compute_capability == (8, 0)
    assert (a100.minimum_device_memory_gib, a100.maximum_device_memory_gib) == (
        35.0,
        45.0,
    )
    assert is_sha256(l4.profile_hash)
    assert is_sha256(l4.profile_file_sha256)
    assert is_sha256(a100.profile_hash)
    assert is_sha256(a100.profile_file_sha256)
    assert l4.profile_hash != l4.profile_file_sha256
    assert a100.profile_hash != a100.profile_file_sha256

    with pytest.raises(FrozenInstanceError):
        l4.backend = "test"  # type: ignore[misc]


def test_execution_profile_rejects_comment_only_byte_tamper(
    project_root: Path, tmp_path: Path
) -> None:
    source = canonical_execution_profile_path(project_root, "l4-24gb-v1")
    tampered = tmp_path / source.name
    tampered.write_bytes(source.read_bytes() + b"# byte tamper\n")

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_execution_profile(tampered)


def test_self_consistent_hand_built_execution_profile_is_not_locked(
    project_root: Path,
) -> None:
    profile = _profile(project_root, "a100-40gb-v1")
    forged_payload = profile.semantic_payload()
    forged_payload["profile_id"] = "forged-a100-40gb-v1"
    forged = replace(
        profile,
        profile_id="forged-a100-40gb-v1",
        profile_hash=sha256_json(forged_payload),
        profile_file_sha256="f" * 64,
    )

    assert not is_locked_execution_profile(forged)
    with pytest.raises(ValueError, match="execution profile object is invalid"):
        verify_execution_compute_gate(
            execution_profile=forged,
            profile="core",
            execution_phase="pilot",
            max_lineages=1,
            compute_estimate_path=None,
            config_sha256="a" * 64,
            planned_adapter_runs=2,
        )


@pytest.mark.parametrize(
    "name",
    ["NVIDIA A100-SXM4-40GB", "NVIDIA A100 40GB PCIe"],
)
def test_a100_profile_accepts_only_allowlisted_full_devices(project_root: Path, name: str) -> None:
    profile = _profile(project_root, "a100-40gb-v1")
    device = _device(name, 39.2, capability=(8, 0))

    assert device_matches_requirements(device, profile, backend="cuda")


@pytest.mark.parametrize(
    ("name", "memory_gib", "capability"),
    [
        ("NVIDIA A100-SXM4-80GB", 79.2, (8, 0)),
        ("NVIDIA A100-SXM4-40GB", 79.2, (8, 9)),
        ("NVIDIA A100-SXM4-40GB", 34.0, (8, 0)),
        ("NVIDIA A100-SXM4-40GB", 46.0, (8, 0)),
        ("NVIDIA A100-SXM4-40GB MIG 1g.10gb", 9.75, (8, 0)),
        ("NVIDIA H100 80GB HBM3", 79.2, (9, 0)),
        ("NVIDIA L4", 22.5, (8, 9)),
    ],
)
def test_a100_profile_rejects_wrong_cc_40gb_mig_h100_and_l4(
    project_root: Path,
    name: str,
    memory_gib: float,
    capability: tuple[int, int],
) -> None:
    profile = _profile(project_root, "a100-40gb-v1")
    device = _device(name, memory_gib, capability=capability)

    assert not device_matches_requirements(device, profile, backend="cuda")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_memory_gib", "79.2"),
        ("total_memory_gib", True),
        ("total_memory_gib", float("nan")),
        ("total_memory_gib", float("inf")),
        ("name", ""),
        ("name", " NVIDIA A100-SXM4-40GB"),
        ("name", None),
        ("compute_capability", [8]),
        ("compute_capability", ["8", 0]),
        ("compute_capability", [True, 0]),
    ],
)
def test_execution_profile_rejects_malformed_device_observations(
    project_root: Path, field: str, value: Any
) -> None:
    profile = _profile(project_root, "a100-40gb-v1")
    device = _device("NVIDIA A100-SXM4-40GB", 39.2, capability=(8, 0))
    device[field] = value

    assert not device_matches_requirements(device, profile, backend="cuda")


def test_execution_profile_requires_successful_bf16_tensor_probe(
    project_root: Path,
) -> None:
    profile = _profile(project_root)
    device = _device(
        "NVIDIA L4",
        22.5,
        capability=(8, 9),
        bf16_passed=False,
    )

    assert not device_matches_requirements(device, profile, backend="cuda")


def test_small_device_proves_runtime_but_not_required_capability(
    project_root: Path,
) -> None:
    profile = _profile(project_root)
    assessment = assess_live_preflight(
        _payload(profile, "NVIDIA L4", 14.56),
        expected_profile=profile,
    )

    assert assessment.preflight_complete is False
    assert assessment.runtime_observed is True
    assert assessment.required_device_observed is False
    assert assessment.observed_devices == ("NVIDIA L4:14.56GiB:capability=8.9:bf16_probe=passed",)


def test_l4_profile_requires_identity_capability_memory_and_probe(
    project_root: Path,
) -> None:
    profile = _profile(project_root)
    too_small = assess_live_preflight(
        _payload(profile, "NVIDIA L4", 20.5),
        expected_profile=profile,
    )
    wrong_capability = assess_live_preflight(
        _payload(profile, "NVIDIA L4", 22.5, capability=(8, 0)),
        expected_profile=profile,
    )
    accepted = assess_live_preflight(
        _payload(profile, "NVIDIA L4", 22.5),
        expected_profile=profile,
    )

    assert too_small.required_device_observed is False
    assert wrong_capability.required_device_observed is False
    assert accepted.required_device_observed is True
    assert accepted.preflight_complete is True


def test_device_name_without_live_accelerator_is_not_runtime_evidence(
    project_root: Path,
) -> None:
    profile = _profile(project_root)
    assessment = assess_live_preflight(
        _payload(profile, "NVIDIA L4", 22.5, accelerator_available=False),
        expected_profile=profile,
    )

    assert assessment.preflight_complete is False
    assert assessment.runtime_observed is False
    assert assessment.required_device_observed is False


def test_mocked_accelerator_labels_are_not_hardware_proof(
    project_root: Path,
) -> None:
    profile = _profile(project_root, "a100-40gb-v1")
    payload = _payload(
        profile,
        "NVIDIA A100-SXM4-40GB",
        39.2,
        capability=(8, 0),
        backend="test",
    )

    assessment = assess_live_preflight(payload, expected_profile=profile)

    assert assessment.runtime_observed is True
    assert assessment.required_device_observed is False
    assert assessment.preflight_complete is False


def test_expected_profile_and_payload_binding_are_both_required(
    project_root: Path,
) -> None:
    l4 = _profile(project_root)
    a100 = _profile(project_root, "a100-40gb-v1")
    payload = _payload(l4, "NVIDIA L4", 22.5)

    assert assess_live_preflight(payload).preflight_complete is False
    payload["execution_profile"] = a100.binding()
    assessment = assess_live_preflight(payload, expected_profile=l4)
    assert assessment.required_device_observed is False
    assert assessment.preflight_complete is False


def test_pilot_compute_gate_binds_explicit_execution_profile(
    project_root: Path,
) -> None:
    profile = _profile(project_root, "a100-40gb-v1")

    result = verify_execution_compute_gate(
        execution_profile=profile,
        profile="core",
        execution_phase="pilot",
        max_lineages=1,
        compute_estimate_path=None,
        config_sha256="a" * 64,
        planned_adapter_runs=2,
    )

    assert result["execution_profile"] == profile.binding()


def test_failed_or_non_strict_payload_is_not_complete(project_root: Path) -> None:
    profile = _profile(project_root)
    failed = _payload(profile, "NVIDIA L4", 22.5)
    failed["status"] = "failed"
    failed["failures"] = ["dependency_mismatch"]
    fixture = _payload(profile, "NVIDIA L4", 22.5)
    fixture["mode"] = "fixture"

    assert assess_live_preflight(failed, expected_profile=profile).preflight_complete is False
    assert assess_live_preflight(fixture, expected_profile=profile).preflight_complete is False


@pytest.mark.parametrize("memory_gib", [True, "79.2", float("nan"), float("inf")])
def test_live_preflight_rejects_non_json_or_nonfinite_device_memory(
    project_root: Path, memory_gib: object
) -> None:
    profile = _profile(project_root)
    assessment = assess_live_preflight(
        _payload(profile, "NVIDIA L4", memory_gib),
        expected_profile=profile,
    )

    assert assessment.preflight_complete is False
    assert assessment.runtime_observed is False
    assert assessment.required_device_observed is False


@pytest.mark.parametrize("free_disk_gib", [True, "55.0", float("nan"), float("inf")])
def test_live_preflight_rejects_non_json_or_nonfinite_free_disk(
    project_root: Path, free_disk_gib: object
) -> None:
    profile = _profile(project_root)
    payload = _payload(profile, "NVIDIA L4", 22.5)
    payload["free_disk_gib"] = free_disk_gib

    assert assess_live_preflight(payload, expected_profile=profile).preflight_complete is False


@pytest.mark.parametrize("field", ["framework_version", "backend", "runtime_version"])
def test_live_preflight_requires_runtime_provenance(project_root: Path, field: str) -> None:
    profile = _profile(project_root)
    payload = _payload(profile, "NVIDIA L4", 22.5)
    payload["accelerator"][field] = " "

    assert assess_live_preflight(payload, expected_profile=profile).preflight_complete is False


def test_external_evidence_requires_scope_config_profile_and_artifact_hash_match(
    project_root: Path,
) -> None:
    execution_profile = _profile(project_root)
    config_hash = "a" * 64
    config_file_sha256 = "b" * 64
    preflight_sha256 = "c" * 64
    source_manifest_sha256 = "d" * 64
    source_tree_sha256 = "e" * 64
    bundle_manifest_sha256 = "f" * 64
    bundle_tree_sha256 = "1" * 64
    bundle_entry_count = 70
    preflight = {
        "config_hash": config_hash,
        "config_file_sha256": config_file_sha256,
        "execution_profile": execution_profile.binding(),
    }
    receipt = {
        "schema_version": 4,
        "execution_scope": {
            "strict_preflight": True,
            "representative_pilot": True,
            "cohort_execution": False,
            "lineage_ids": ["lineage-1"],
        },
        "strict_preflight": {
            "config_hash": config_hash,
            "config_file_sha256": config_file_sha256,
            "artifact_sha256": preflight_sha256,
            "execution_profile": execution_profile.binding(),
        },
        "source_snapshot": {
            "manifest_sha256": source_manifest_sha256,
            "tree_sha256": source_tree_sha256,
            "entry_count": 68,
        },
        "bundle_snapshot": {
            "manifest_sha256": bundle_manifest_sha256,
            "tree_sha256": bundle_tree_sha256,
            "entry_count": bundle_entry_count,
        },
    }

    assert live_evidence_matches(
        preflight,
        receipt,
        execution_profile=execution_profile,
        config_hash=config_hash,
        config_file_sha256=config_file_sha256,
        preflight_sha256=preflight_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        source_entry_count=68,
        bundle_manifest_sha256=bundle_manifest_sha256,
        bundle_tree_sha256=bundle_tree_sha256,
        bundle_entry_count=bundle_entry_count,
        expected_representative_pilot=True,
        expected_cohort_execution=False,
        expected_lineage_ids=("lineage-1",),
    )
    receipt["strict_preflight"]["artifact_sha256"] = "tampered"
    assert not live_evidence_matches(
        preflight,
        receipt,
        execution_profile=execution_profile,
        config_hash=config_hash,
        config_file_sha256=config_file_sha256,
        preflight_sha256=preflight_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        source_entry_count=68,
        bundle_manifest_sha256=bundle_manifest_sha256,
        bundle_tree_sha256=bundle_tree_sha256,
        bundle_entry_count=bundle_entry_count,
        expected_representative_pilot=True,
        expected_cohort_execution=False,
        expected_lineage_ids=("lineage-1",),
    )
    receipt["strict_preflight"]["artifact_sha256"] = preflight_sha256
    receipt["execution_scope"]["lineage_ids"] = ["other-lineage"]
    assert not live_evidence_matches(
        preflight,
        receipt,
        execution_profile=execution_profile,
        config_hash=config_hash,
        config_file_sha256=config_file_sha256,
        preflight_sha256=preflight_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        source_entry_count=68,
        bundle_manifest_sha256=bundle_manifest_sha256,
        bundle_tree_sha256=bundle_tree_sha256,
        bundle_entry_count=bundle_entry_count,
        expected_representative_pilot=True,
        expected_cohort_execution=False,
        expected_lineage_ids=("lineage-1",),
    )
    receipt["execution_scope"]["lineage_ids"] = ["lineage-1"]
    receipt["schema_version"] = True
    assert not live_evidence_matches(
        preflight,
        receipt,
        execution_profile=execution_profile,
        config_hash=config_hash,
        config_file_sha256=config_file_sha256,
        preflight_sha256=preflight_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        source_entry_count=68,
        bundle_manifest_sha256=bundle_manifest_sha256,
        bundle_tree_sha256=bundle_tree_sha256,
        bundle_entry_count=bundle_entry_count,
        expected_representative_pilot=True,
        expected_cohort_execution=False,
        expected_lineage_ids=("lineage-1",),
    )
    receipt["schema_version"] = 2
    preflight["execution_profile"]["profile_file_sha256"] = "0" * 64
    assert not live_evidence_matches(
        preflight,
        receipt,
        execution_profile=execution_profile,
        config_hash=config_hash,
        config_file_sha256=config_file_sha256,
        preflight_sha256=preflight_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        source_entry_count=68,
        bundle_manifest_sha256=bundle_manifest_sha256,
        bundle_tree_sha256=bundle_tree_sha256,
        bundle_entry_count=bundle_entry_count,
        expected_representative_pilot=True,
        expected_cohort_execution=False,
        expected_lineage_ids=("lineage-1",),
    )


def test_external_evidence_rejects_invalid_expected_arguments(
    project_root: Path,
) -> None:
    execution_profile = _profile(project_root)
    sha256 = "a" * 64

    assert not live_evidence_matches(
        {},
        {},
        execution_profile=execution_profile,
        config_hash=sha256,
        config_file_sha256=sha256,
        preflight_sha256=sha256,
        source_manifest_sha256=sha256,
        source_tree_sha256=sha256,
        source_entry_count=True,
        bundle_manifest_sha256=sha256,
        bundle_tree_sha256=sha256,
        bundle_entry_count=70,
        expected_representative_pilot=True,
        expected_cohort_execution=False,
        expected_lineage_ids=("lineage-1",),
    )
    assert not live_evidence_matches(
        {},
        {},
        execution_profile=execution_profile,
        config_hash="not-a-hash",
        config_file_sha256=sha256,
        preflight_sha256=sha256,
        source_manifest_sha256=sha256,
        source_tree_sha256=sha256,
        source_entry_count=68,
        bundle_manifest_sha256=sha256,
        bundle_tree_sha256=sha256,
        bundle_entry_count=70,
        expected_representative_pilot=True,
        expected_cohort_execution=False,
        expected_lineage_ids=("lineage-1",),
    )


def test_source_snapshot_rehash_rejects_manifest_drift(project_root: Path, tmp_path: Path) -> None:
    manifest = build_source_snapshot(project_root)
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    verified = verify_source_snapshot(
        project_root,
        path,
        expected_entry_count=manifest["entry_count"],
    )

    assert verified["entry_count"] == manifest["entry_count"]
    assert verified["tree_sha256"] == manifest["tree_sha256"]
    manifest["entries"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    try:
        verify_source_snapshot(
            project_root,
            path,
            expected_entry_count=manifest["entry_count"],
        )
    except ValueError as exc:
        assert "source manifest mismatch" in str(exc)
    else:
        raise AssertionError("stale source manifest was accepted")


def test_bundle_snapshot_is_canonical_and_bound_to_the_source_tree(tmp_path: Path) -> None:
    bundle_root = tmp_path / "TranThienNhan_TTTN"
    project_root = bundle_root / "lora-audit"
    manifest_path = project_root / "artifacts/bundle-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    source_tree_sha256 = "a" * 64
    entries = [
        {
            "path": "RUN_EXPERIMENT.ipynb",
            "size_bytes": 10,
            "sha256": "b" * 64,
            "private": False,
        },
        {
            "path": "bootstrap.py",
            "size_bytes": 20,
            "sha256": "c" * 64,
            "private": False,
        },
        {
            "path": "lora-audit/README.md",
            "size_bytes": 30,
            "sha256": "d" * 64,
            "private": False,
        },
    ]
    canonical = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    payload = {
        "schema_version": 1,
        "destination_label": bundle_root.name,
        "entry_count": len(entries),
        "total_size_bytes": 60,
        "tree_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "tree_hash_contract": "ordered path, size_bytes, sha256, and private fields",
        "inner_source_tree_sha256": source_tree_sha256,
        "inner_source_entry_count": 3,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    verified = verify_bundle_snapshot(
        project_root,
        manifest_path,
        source_tree_sha256=source_tree_sha256,
        source_entry_count=3,
    )
    assert verified["tree_sha256"] == payload["tree_sha256"]
    assert verified["entry_count"] == 3

    payload["inner_source_tree_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source-tree binding"):
        verify_bundle_snapshot(
            project_root,
            manifest_path,
            source_tree_sha256=source_tree_sha256,
            source_entry_count=3,
        )


def test_live_evidence_uses_one_canonical_pair(tmp_path: Path) -> None:
    preflight, receipt = select_live_evidence_paths(tmp_path)

    assert preflight == tmp_path / "artifacts/preflight-live.json"
    assert receipt == tmp_path / "artifacts/execution-receipt.json"


def _delta_payload(*, baseline: str, target: str) -> dict:
    operations = [
        {
            "operation": "update",
            "path": "tests/test_example.py",
            "before_sha256": "a" * 64,
            "after_sha256": "b" * 64,
            "after_size_bytes": 123,
        }
    ]
    canonical = json.dumps(operations, ensure_ascii=False, separators=(",", ":"))
    return {
        "schema_version": 2,
        "folder_id": "folder",
        "external_write_authorized": False,
        "external_write_performed": False,
        "baseline_tree_sha256": baseline,
        "target_tree_sha256": target,
        "baseline_entry_count": 68,
        "target_entry_count": 68,
        "operation_counts": {"add": 0, "update": 1, "delete": 0},
        "delta_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "operations": operations,
    }


def test_source_delta_selection_rejects_stale_target(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    remote = "1" * 64
    stale_local = "2" * 64
    current_local = "3" * 64
    stale = _delta_payload(baseline=remote, target=stale_local)
    (artifacts / "source-delta-manifest.json").write_text(
        json.dumps(stale) + "\n", encoding="utf-8"
    )

    assert (
        select_source_delta_path(
            tmp_path,
            baseline_tree_sha256=remote,
            target_tree_sha256=current_local,
        )
        is None
    )


def test_source_delta_is_tree_bound_and_tamper_evident(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    remote = "1" * 64
    current_local = "3" * 64
    delta = _delta_payload(baseline=remote, target=current_local)
    path = artifacts / "source-followup-delta-manifest.json"
    path.write_text(json.dumps(delta) + "\n", encoding="utf-8")

    assert (
        select_source_delta_path(
            tmp_path,
            baseline_tree_sha256=remote,
            target_tree_sha256=current_local,
        )
        == path
    )
    assert source_delta_matches(
        delta,
        folder_id="folder",
        baseline_tree_sha256=remote,
        target_tree_sha256=current_local,
        target_entry_count=68,
    )
    delta["operations"][0]["after_sha256"] = "c" * 64
    assert not source_delta_matches(
        delta,
        folder_id="folder",
        baseline_tree_sha256=remote,
        target_tree_sha256=current_local,
        target_entry_count=68,
    )


@pytest.mark.parametrize(
    "mutation",
    ["boolean_count", "boolean_size", "boolean_baseline", "boolean_target"],
)
def test_source_delta_rejects_boolean_numeric_fields(mutation: str) -> None:
    remote = "1" * 64
    current_local = "3" * 64
    delta = _delta_payload(baseline=remote, target=current_local)
    if mutation == "boolean_count":
        delta["operation_counts"]["update"] = True
    elif mutation == "boolean_size":
        delta["operations"][0]["after_size_bytes"] = True
    elif mutation == "boolean_baseline":
        delta["baseline_entry_count"] = True
    else:
        delta["target_entry_count"] = True

    assert not source_delta_matches(
        delta,
        folder_id="folder",
        baseline_tree_sha256=remote,
        target_tree_sha256=current_local,
        target_entry_count=68,
    )


def test_source_delta_rejects_invalid_target_arguments() -> None:
    remote = "1" * 64
    current_local = "3" * 64
    delta = _delta_payload(baseline=remote, target=current_local)

    assert not source_delta_matches(
        delta,
        folder_id=" folder ",
        baseline_tree_sha256=remote,
        target_tree_sha256=current_local,
        target_entry_count=68,
    )
    assert not source_delta_matches(
        delta,
        folder_id="folder",
        baseline_tree_sha256=remote,
        target_tree_sha256=current_local,
        target_entry_count=True,
    )
