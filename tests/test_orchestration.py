import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lora_audit.cohort import plan_cohort, select_representative_pilot_lineages
from lora_audit.compute import (
    PilotEvaluationTelemetry,
    PilotPairBinding,
    PilotTelemetry,
    verify_pilot_telemetry_alignment,
)
from lora_audit.compute import (
    estimate_compute as _estimate_compute,
)
from lora_audit.compute import (
    read_pilot_device_hours as _read_pilot_device_hours,
)
from lora_audit.compute import (
    read_pilot_evaluation_telemetry as _read_pilot_evaluation_telemetry,
)
from lora_audit.compute import (
    read_pilot_telemetry as _read_pilot_telemetry,
)
from lora_audit.config import load_config
from lora_audit.execution_profiles import ExecutionProfile, load_execution_profile
from lora_audit.hashing import sha256_file, sha256_json
from lora_audit.manifests import CohortEntry
from lora_audit.orchestration import (
    load_or_create_cohort_state,
    verify_completed_cohort_run,
    write_cohort_state,
)
from lora_audit.orchestration import (
    orchestrate_cohort as _orchestrate_cohort,
)
from lora_audit.orchestration import (
    qc_valid_completed_lineages as _qc_valid_completed_lineages,
)
from lora_audit.preflight import ACCELERATED_PINNED_PACKAGES, BASE_PINNED_PACKAGES
from lora_audit.readiness import (
    bind_observed_compute_evidence,
    build_source_snapshot,
    verify_bundle_snapshot,
    verify_source_snapshot,
)
from lora_audit.readiness import (
    verify_execution_compute_gate as _verify_execution_compute_gate,
)

TEST_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_EXECUTION_PROFILE_PATH = TEST_PROJECT_ROOT / "configs/execution/l4-24gb.yaml"
TEST_EXECUTION_PROFILE = load_execution_profile(TEST_EXECUTION_PROFILE_PATH)


def _qualified_device() -> dict[str, object]:
    return {
        "index": 0,
        "name": "NVIDIA L4",
        "total_memory_gib": 22.5,
        "compute_capability": [8, 9],
        "bf16_probe": {
            "attempted": True,
            "passed": True,
            "finite": True,
            "dtype": "torch.bfloat16",
            "device_type": "cuda",
            "device_index": 0,
            "error": None,
        },
    }


def _profile_arguments() -> dict[str, ExecutionProfile | Path]:
    return {
        "execution_profile": TEST_EXECUTION_PROFILE,
        "execution_profile_path": TEST_EXECUTION_PROFILE_PATH,
        "bundle_manifest_path": TEST_PROJECT_ROOT / "artifacts/bundle-manifest.json",
    }


def orchestrate_cohort(**kwargs: object) -> dict:
    """Run orchestration tests through one canonical locked profile fixture."""

    return _orchestrate_cohort(**{**_profile_arguments(), **kwargs})


def qc_valid_completed_lineages(**kwargs: object) -> list:
    """Run completed-lineage tests with the same canonical profile binding."""

    profile_arguments = _profile_arguments()
    profile_arguments.pop("bundle_manifest_path")
    return _qc_valid_completed_lineages(
        project_root=TEST_PROJECT_ROOT,
        **profile_arguments,
        **kwargs,
    )


def verify_execution_compute_gate(**kwargs: object) -> dict:
    """Bind compute-gate unit tests to the strict profile under test."""

    return _verify_execution_compute_gate(
        execution_profile=TEST_EXECUTION_PROFILE,
        **kwargs,
    )


def estimate_compute(**kwargs: object) -> dict:
    """Only observed-pilot estimates may claim an execution profile."""

    if kwargs.get("pilot_device_hours"):
        kwargs = {"execution_profile": TEST_EXECUTION_PROFILE, **kwargs}
    return _estimate_compute(**kwargs)


def read_pilot_telemetry(path: Path, **kwargs: object) -> PilotTelemetry:
    kwargs.pop("required_compute_capability", None)
    kwargs.pop("minimum_device_memory_gib", None)
    return _read_pilot_telemetry(
        path,
        execution_profile=TEST_EXECUTION_PROFILE,
        **kwargs,
    )


def read_pilot_evaluation_telemetry(path: Path, **kwargs: object) -> PilotEvaluationTelemetry:
    kwargs.pop("required_compute_capability", None)
    kwargs.pop("minimum_device_memory_gib", None)
    return _read_pilot_evaluation_telemetry(
        path,
        execution_profile=TEST_EXECUTION_PROFILE,
        **kwargs,
    )


def read_pilot_device_hours(path: Path, **kwargs: object) -> list[float]:
    kwargs.pop("required_compute_capability", None)
    kwargs.pop("minimum_device_memory_gib", None)
    return _read_pilot_device_hours(
        path,
        execution_profile=TEST_EXECUTION_PROFILE,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _qualified_current_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lora_audit.readiness.accelerator_snapshot",
        lambda: {
            "accelerator_available": True,
            "backend": "cuda",
            "devices": [_qualified_device()],
        },
    )


def _execution_evidence(
    project_root: Path,
    tmp_path: Path,
    config,
    *,
    phase: str = "pilot",
    compute_estimate: Path | None = None,
    lineage_ids: list[str] | None = None,
) -> dict:
    evidence_root = tmp_path / f"execution-evidence-{phase}"
    evidence_root.mkdir(parents=True, exist_ok=True)
    config_path = project_root / "configs/core.yaml"
    source_manifest_path = evidence_root / "source-manifest.json"
    source_manifest = build_source_snapshot(project_root)
    bundle_snapshot = verify_bundle_snapshot(
        project_root,
        TEST_PROJECT_ROOT / "artifacts/bundle-manifest.json",
        source_tree_sha256=source_manifest["tree_sha256"],
        source_entry_count=source_manifest["entry_count"],
    )
    source_manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight_path = evidence_root / "preflight.json"
    preflight = {
        "status": "passed",
        "mode": "strict",
        "fixture_smoke_only": False,
        "failures": [],
        "warnings": [],
        "python": "3.12.13",
        "platform": "test-platform",
        "packages": {**BASE_PINNED_PACKAGES, **ACCELERATED_PINNED_PACKAGES},
        "config_hash": config.config_hash,
        "config_file_sha256": sha256_file(config_path),
        "artifact_writable": True,
        "free_disk_gib": 100.0,
        "execution_profile": TEST_EXECUTION_PROFILE.binding(),
        "accelerator": {
            "framework_available": True,
            "accelerator_available": True,
            "framework_version": "2.13.0",
            "backend": "cuda",
            "runtime_version": "13.0",
            "devices": [_qualified_device()],
        },
    }
    preflight_path.write_text(json.dumps(preflight, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = evidence_root / "receipt.json"
    planned = plan_cohort(config)
    if lineage_ids is not None:
        bound_lineage_ids = lineage_ids
    elif phase == "cohort":
        bound_lineage_ids = [lineage.parent_config_id for lineage in planned]
    else:
        bound_lineage_ids = [planned[0].parent_config_id]
    receipt = {
        "schema_version": 4,
        "execution_scope": {
            "strict_preflight": True,
            "representative_pilot": phase == "pilot",
            "cohort_execution": phase == "cohort",
            "lineage_ids": bound_lineage_ids,
        },
        "strict_preflight": {
            "config_hash": config.config_hash,
            "config_file_sha256": sha256_file(config_path),
            "execution_profile": TEST_EXECUTION_PROFILE.binding(),
            "artifact_sha256": sha256_file(preflight_path),
        },
        "source_snapshot": {
            "manifest_sha256": sha256_file(source_manifest_path),
            "tree_sha256": source_manifest["tree_sha256"],
            "entry_count": source_manifest["entry_count"],
        },
        "bundle_snapshot": bundle_snapshot,
    }
    if phase == "cohort":
        if compute_estimate is None:
            raise ValueError("cohort test evidence requires a compute estimate")
        compute_payload = json.loads(compute_estimate.read_text(encoding="utf-8"))
        receipt["compute_authorization"] = {
            "estimate_sha256": sha256_file(compute_estimate),
            "available_device_hours": compute_payload.get("available_device_hours"),
            "profile": config.profile,
            "planned_adapter_runs": config.cohort.paired_lineages * 2,
            "recommendation": compute_payload.get("recommendation"),
        }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "config_path": config_path,
        "preflight_artifact": preflight_path,
        "execution_receipt": receipt_path,
        "source_manifest": source_manifest_path,
    }


def _pair(config, lineage) -> list[CohortEntry]:
    common = {
        "parent_config_id": lineage.parent_config_id,
        "base_model": config.revisions.model_id,
        "dataset": config.revisions.dataset_id,
        "language": lineage.language,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "rank": config.lora.rank,
        "target_modules": config.lora.target_modules,
        "raw_data_group_hash": "a" * 64,
        "model_revision": config.revisions.model_revision,
        "dataset_revision": config.revisions.dataset_revision,
        "train_revision": "test",
        "config_sha256": config.config_hash,
        "qc_status": "valid",
        "qc_reason": None,
        "split": lineage.detector_split,
        "split_assignment_id": lineage.split_assignment_id,
        "execution_profile_id": TEST_EXECUTION_PROFILE.profile_id,
        "execution_profile_sha256": TEST_EXECUTION_PROFILE.profile_hash,
        "execution_profile_file_sha256": TEST_EXECUTION_PROFILE.profile_file_sha256,
        "fixture_smoke_only": False,
    }
    return [
        CohortEntry(
            adapter_id=f"{lineage.parent_config_id}-clean",
            label="clean",
            poison_rate=0.0,
            trigger_family=None,
            target_intent=None,
            adapter_sha256="b" * 64,
            normalized_text_hash="c" * 64,
            training_unit_id=lineage.training_unit_id("clean"),
            **common,
        ),
        CohortEntry(
            adapter_id=f"{lineage.parent_config_id}-backdoored",
            label="backdoored",
            poison_rate=lineage.poison_rate,
            trigger_family=lineage.trigger_family,
            target_intent="harmless_target",
            adapter_sha256="d" * 64,
            normalized_text_hash="e" * 64,
            training_unit_id=lineage.training_unit_id("backdoored"),
            **common,
        ),
    ]


def test_completed_lineage_selection_excludes_hash_bound_failed_qc_pairs(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    lineages = [lineage for lineage in plan_cohort(config) if lineage.detector_split == "test"]
    failed_id = lineages[0].parent_config_id
    source_tree_sha256 = "1" * 64
    source_manifest_sha256 = "2" * 64
    hashes: dict[str, str] = {}
    cohort_root = tmp_path / "run" / "cohort"
    cohort_root.mkdir(parents=True)
    for lineage in lineages:
        entries = _pair(config, lineage)
        if lineage.parent_config_id == failed_id:
            entries = [
                entry.model_copy(update={"qc_status": "failed", "qc_reason": "qc_failed"})
                for entry in entries
            ]
        bound_entries: list[CohortEntry] = []
        for entry in entries:
            label_root = tmp_path / "run" / "adapters" / lineage.parent_config_id / entry.label
            adapter_root = label_root / "adapter"
            adapter_root.mkdir(parents=True)
            weights = adapter_root / "adapter_model.safetensors"
            weights.write_bytes(f"{lineage.parent_config_id}:{entry.label}".encode())
            (adapter_root / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": config.revisions.model_id,
                        "r": config.lora.rank,
                        "lora_alpha": config.lora.alpha,
                        "lora_dropout": config.lora.dropout,
                        "bias": config.lora.bias,
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "target_modules": config.lora.target_modules,
                    }
                ),
                encoding="utf-8",
            )
            adapter_sha256 = sha256_file(weights)
            (label_root / "training-receipt.json").write_text(
                json.dumps(
                    {
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
                        "source_tree_sha256": source_tree_sha256,
                        "source_manifest_sha256": source_manifest_sha256,
                        "adapter_sha256": adapter_sha256,
                        "execution_profile": TEST_EXECUTION_PROFILE.binding(),
                    }
                ),
                encoding="utf-8",
            )
            bound_entries.append(entry.model_copy(update={"adapter_sha256": adapter_sha256}))
        entries = bound_entries
        manifest = cohort_root / f"{lineage.parent_config_id}.jsonl"
        manifest.write_text(
            "\n".join(entry.model_dump_json() for entry in entries) + "\n",
            encoding="utf-8",
        )
        hashes[lineage.parent_config_id] = sha256_file(manifest)

    selected = qc_valid_completed_lineages(
        config=config,
        output_root=tmp_path / "run",
        state=SimpleNamespace(
            lineage_manifest_hashes=hashes,
            source_tree_sha256=source_tree_sha256,
            source_manifest_sha256=source_manifest_sha256,
            source_entry_count=1,
            bundle_manifest_sha256="3" * 64,
            bundle_tree_sha256="4" * 64,
            bundle_entry_count=1,
            execution_profile_id=TEST_EXECUTION_PROFILE.profile_id,
            execution_profile_sha256=TEST_EXECUTION_PROFILE.profile_hash,
            execution_profile_file_sha256=TEST_EXECUTION_PROFILE.profile_file_sha256,
        ),
        split="test",
    )

    assert failed_id not in {lineage.parent_config_id for lineage in selected}
    assert len(selected) == len(lineages) - 1

    with pytest.raises(ValueError, match="state execution-profile binding mismatch"):
        qc_valid_completed_lineages(
            config=config,
            output_root=tmp_path / "run",
            state=SimpleNamespace(
                lineage_manifest_hashes=hashes,
                source_tree_sha256=source_tree_sha256,
                source_manifest_sha256=source_manifest_sha256,
                source_entry_count=1,
                bundle_manifest_sha256="3" * 64,
                bundle_tree_sha256="4" * 64,
                bundle_entry_count=1,
                execution_profile_id=TEST_EXECUTION_PROFILE.profile_id,
                execution_profile_sha256="0" * 64,
                execution_profile_file_sha256=(TEST_EXECUTION_PROFILE.profile_file_sha256),
            ),
            split="test",
        )

    drifted = lineages[1]
    drifted_weights = (
        tmp_path
        / "run"
        / "adapters"
        / drifted.parent_config_id
        / "clean"
        / "adapter"
        / "adapter_model.safetensors"
    )
    drifted_weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="adapter binding mismatch"):
        qc_valid_completed_lineages(
            config=config,
            output_root=tmp_path / "run",
            state=SimpleNamespace(
                lineage_manifest_hashes=hashes,
                source_tree_sha256=source_tree_sha256,
                source_manifest_sha256=source_manifest_sha256,
                source_entry_count=1,
                bundle_manifest_sha256="3" * 64,
                bundle_tree_sha256="4" * 64,
                bundle_entry_count=1,
                execution_profile_id=TEST_EXECUTION_PROFILE.profile_id,
                execution_profile_sha256=TEST_EXECUTION_PROFILE.profile_hash,
                execution_profile_file_sha256=(TEST_EXECUTION_PROFILE.profile_file_sha256),
            ),
            split="test",
        )


def _training_telemetry_rows(
    config_sha256: str,
    *,
    source_tree_sha256: str = "1" * 64,
    source_manifest_sha256: str = "2" * 64,
    pair_count: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(1, pair_count + 1):
        for label_index, label in enumerate(("clean", "backdoored")):
            adapter_hash = (
                ("c" if label == "clean" else "d") * 64
                if pair_index == 1
                else f"{100 + 2 * pair_index + label_index:064x}"
            )
            records_hash = (
                ("b" if label == "clean" else "e") * 64
                if pair_index == 1
                else f"{200 + 2 * pair_index + label_index:064x}"
            )
            rows.append(
                {
                    "parent_config_id": f"pair-{pair_index}",
                    "adapter_label": label,
                    "replicate_seed": 101 * pair_index,
                    "pair_training_seed": 202 * pair_index,
                    "poison_selection_seed": 303 * pair_index,
                    "probe_seed": 404 * pair_index,
                    "split_assignment_id": f"test-split-{pair_index}",
                    "training_unit_id": f"pair-{pair_index}-{label}-unit",
                    "device_hours": 1.0 / 3600.0,
                    "runtime_seconds": 1.0,
                    "peak_allocated_memory_gib": 8.0,
                    "peak_reserved_memory_gib": 10.0,
                    "training_record_count": 64,
                    "records_sha256": records_hash,
                    "adapter_size_bytes": 1024,
                    "adapter_sha256": adapter_hash,
                    "device_observation": _qualified_device(),
                    "backend": "cuda",
                    "framework_version": "2.13.0",
                    "config_sha256": config_sha256,
                    "source_tree_sha256": source_tree_sha256,
                    "source_manifest_sha256": source_manifest_sha256,
                    "execution_profile": TEST_EXECUTION_PROFILE.binding(),
                }
            )
    return rows


def _evaluation_telemetry_rows(
    config_sha256: str,
    *,
    source_tree_sha256: str = "1" * 64,
    source_manifest_sha256: str = "2" * 64,
    pair_count: int = 1,
) -> list[dict[str, object]]:
    rows = []
    for pair_index in range(1, pair_count + 1):
        clean_hash = "c" * 64 if pair_index == 1 else f"{100 + 2 * pair_index:064x}"
        backdoored_hash = "d" * 64 if pair_index == 1 else f"{101 + 2 * pair_index:064x}"
        rows.append(
            {
                "parent_config_id": f"pair-{pair_index}",
                "replicate_seed": 101 * pair_index,
                "pair_training_seed": 202 * pair_index,
                "poison_selection_seed": 303 * pair_index,
                "probe_seed": 404 * pair_index,
                "split_assignment_id": f"test-split-{pair_index}",
                "device_hours": 1.0 / 3600.0,
                "runtime_seconds": 1.0,
                "peak_allocated_memory_gib": 12.0,
                "peak_reserved_memory_gib": 14.0,
                "probe_group_count": 128,
                "test_record_count": 64,
                "qc_valid": True,
                "qc_reason_codes": [],
                "evaluation_report_sha256": (
                    "f" * 64 if pair_index == 1 else f"{300 + pair_index:064x}"
                ),
                "clean_adapter_sha256": clean_hash,
                "backdoored_adapter_sha256": backdoored_hash,
                "device_observation": _qualified_device(),
                "backend": "cuda",
                "config_sha256": config_sha256,
                "source_tree_sha256": source_tree_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "execution_profile": TEST_EXECUTION_PROFILE.binding(),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _observed_compute_estimate_payload(
    config_sha256: str,
    *,
    estimate_path: Path,
    source_tree_sha256: str = "1" * 64,
    source_manifest_sha256: str = "2" * 64,
    first_pair_failure_reason_codes: list[str] | None = None,
) -> dict[str, object]:
    ledger_root = estimate_path.parent / "pilot-evidence"
    ledger_root.mkdir(parents=True, exist_ok=True)
    training_path = ledger_root / "training.jsonl"
    evaluation_path = ledger_root / "evaluation.jsonl"
    _write_jsonl(
        training_path,
        _training_telemetry_rows(
            config_sha256,
            source_tree_sha256=source_tree_sha256,
            source_manifest_sha256=source_manifest_sha256,
            pair_count=3,
        ),
    )
    evaluation_rows = _evaluation_telemetry_rows(
        config_sha256,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        pair_count=3,
    )
    if first_pair_failure_reason_codes is not None:
        evaluation_rows[0]["qc_valid"] = False
        evaluation_rows[0]["qc_reason_codes"] = first_pair_failure_reason_codes
    _write_jsonl(evaluation_path, evaluation_rows)
    training = read_pilot_telemetry(
        training_path,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256=source_tree_sha256,
        expected_source_manifest_sha256=source_manifest_sha256,
        required_compute_capability=(8, 9),
        minimum_device_memory_gib=20.0,
    )
    evaluation = read_pilot_evaluation_telemetry(
        evaluation_path,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256=source_tree_sha256,
        expected_source_manifest_sha256=source_manifest_sha256,
        required_compute_capability=(8, 9),
        minimum_device_memory_gib=20.0,
    )
    result = estimate_compute(
        pilot_device_hours=list(training.device_hours),
        adapter_size_gib=0.15,
        available_device_hours=0.1,
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
    )
    return bind_observed_compute_evidence(
        result,
        estimate_path=estimate_path,
        training_ledger_path=training_path,
        evaluation_ledger_path=evaluation_path,
        pair_bindings=verify_pilot_telemetry_alignment(training, evaluation),
    )


def _pilot_memory_arguments() -> dict[str, list[float]]:
    return {
        "pilot_peak_allocated_memory_gib": [8.0, 9.0],
        "pilot_peak_reserved_memory_gib": [10.0, 11.0],
        "pilot_device_memory_gib": [22.5, 22.5],
        "pilot_evaluation_peak_allocated_memory_gib": [12.0],
        "pilot_evaluation_peak_reserved_memory_gib": [14.0],
        "pilot_evaluation_device_memory_gib": [22.5],
    }


def test_cohort_dry_run_is_noop_and_selects_deterministic_first_pair(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")

    result = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "run",
        execute=False,
        max_lineages=1,
    )

    assert result["status"] == "dry_run"
    assert result["planned_lineages"] == 36
    assert len(result["selected_lineage_ids"]) == 1
    assert not (tmp_path / "run/cohort/run-state.json").exists()


def test_cohort_refuses_profile_outside_its_canonical_project_path(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    copied_profile = tmp_path / "l4-24gb.yaml"
    copied_profile.write_bytes(TEST_EXECUTION_PROFILE_PATH.read_bytes())

    with pytest.raises(ValueError, match="not canonical for the project"):
        _orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "run",
            execution_profile=TEST_EXECUTION_PROFILE,
            execution_profile_path=copied_profile,
            bundle_manifest_path=TEST_PROJECT_ROOT / "artifacts/bundle-manifest.json",
            execute=False,
            max_lineages=1,
        )


def test_cohort_refuses_bundle_manifest_outside_its_canonical_project_path(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    copied_manifest = tmp_path / "bundle-manifest.json"
    copied_manifest.write_bytes((project_root / "artifacts/bundle-manifest.json").read_bytes())

    with pytest.raises(ValueError, match="bundle manifest path is not canonical"):
        _orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "run",
            execution_profile=TEST_EXECUTION_PROFILE,
            execution_profile_path=TEST_EXECUTION_PROFILE_PATH,
            bundle_manifest_path=copied_manifest,
            execute=False,
            max_lineages=1,
        )


def test_cohort_dry_run_accepts_an_ordered_exact_selection(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    plan = plan_cohort(config)
    requested = [plan[5].parent_config_id, plan[2].parent_config_id]

    result = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "run",
        execute=False,
        lineage_ids=requested,
    )

    assert result["selected_lineage_ids"] == requested
    assert not (tmp_path / "run/cohort/run-state.json").exists()


@pytest.mark.parametrize("max_lineages", [True, 0, -1, 1.5])
def test_cohort_rejects_non_positive_integer_lineage_limits(
    project_root: Path, tmp_path: Path, max_lineages: object
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "run",
            execute=False,
            max_lineages=max_lineages,
        )


def test_cohort_execution_resumes_next_pending_lineage(project_root: Path, tmp_path: Path) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    calls: list[str] = []
    evidence = _execution_evidence(project_root, tmp_path, config)

    def trainer(**kwargs):
        lineage = kwargs["lineage"]
        calls.append(lineage.parent_config_id)
        return _pair(config, lineage)

    first = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "run",
        execute=True,
        max_lineages=1,
        execution_phase="pilot",
        trainer=trainer,
        **evidence,
    )
    next_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        lineage_ids=[plan_cohort(config)[1].parent_config_id],
    )
    second = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "run",
        execute=True,
        max_lineages=1,
        execution_phase="pilot",
        trainer=trainer,
        **next_evidence,
    )

    assert first["completed_lineages"] == 1
    assert second["completed_lineages"] == 2
    assert len(calls) == 2
    assert calls[0] != calls[1]


def test_execution_refuses_a_receipt_bound_to_a_different_lineage(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    plan = plan_cohort(config)
    requested = plan[3].parent_config_id
    first_evidence = _execution_evidence(project_root, tmp_path, config)

    with pytest.raises(ValueError, match="receipt binding or scope mismatch"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "refused",
            execute=True,
            lineage_ids=[requested],
            execution_phase="pilot",
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
            **first_evidence,
        )

    requested_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        lineage_ids=[requested],
    )
    result = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "accepted",
        execute=True,
        lineage_ids=[requested],
        execution_phase="pilot",
        trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        **requested_evidence,
    )

    assert result["executed_lineage_ids"] == [requested]
    assert not (tmp_path / "refused/cohort/run-state.json").exists()
    resumed = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "accepted",
        execute=False,
        max_lineages=1,
    )
    assert resumed["selected_lineage_ids"] == [plan[0].parent_config_id]


def test_cohort_resume_rejects_source_snapshot_drift(project_root: Path, tmp_path: Path) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    run_root = tmp_path / "run"
    evidence = _execution_evidence(project_root, tmp_path, config)
    orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=run_root,
        execute=True,
        max_lineages=1,
        execution_phase="pilot",
        trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        **evidence,
    )
    state_path = run_root / "cohort/run-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["source_tree_sha256"] = "f" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    next_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        lineage_ids=[plan_cohort(config)[1].parent_config_id],
    )

    with pytest.raises(ValueError, match="source snapshot changed"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=run_root,
            execute=True,
            max_lineages=1,
            execution_phase="pilot",
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
            **next_evidence,
        )


def test_cohort_resume_refuses_config_or_plan_drift(project_root: Path, tmp_path: Path) -> None:
    core = load_config(project_root / "configs/core.yaml")
    target = load_config(project_root / "configs/target.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    run_root = tmp_path / "run"
    evidence = _execution_evidence(project_root, tmp_path, core)
    orchestrate_cohort(
        config=core,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=run_root,
        execute=True,
        max_lineages=1,
        execution_phase="pilot",
        trainer=lambda **kwargs: _pair(core, kwargs["lineage"]),
        **evidence,
    )

    with pytest.raises(ValueError, match="immutable state mismatch"):
        orchestrate_cohort(
            config=target,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=run_root,
            execute=False,
            max_lineages=1,
        )

    prepared.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared_manifest_sha256"):
        orchestrate_cohort(
            config=core,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=run_root,
            execute=False,
            max_lineages=1,
        )


def test_cohort_resume_refuses_tampered_completed_manifest(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    run_root = tmp_path / "run"
    evidence = _execution_evidence(project_root, tmp_path, config)
    result = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=run_root,
        execute=True,
        max_lineages=1,
        execution_phase="pilot",
        trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        **evidence,
    )
    lineage_id = result["executed_lineage_ids"][0]
    manifest = run_root / "cohort" / f"{lineage_id}.jsonl"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lineage manifest hash mismatch"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=run_root,
            execute=False,
            max_lineages=1,
        )


def test_cohort_resume_refuses_duplicate_completed_lineages(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    run_root = tmp_path / "run"
    evidence = _execution_evidence(project_root, tmp_path, config)
    result = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=run_root,
        execute=True,
        max_lineages=1,
        execution_phase="pilot",
        trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        **evidence,
    )
    state_path = run_root / "cohort/run-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completed_lineage_ids"].append(result["executed_lineage_ids"][0])
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate completed lineage IDs"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=run_root,
            execute=False,
            max_lineages=1,
        )


def test_failed_lineage_attempt_is_recorded_and_retried(project_root: Path, tmp_path: Path) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    run_root = tmp_path / "run"
    evidence = _execution_evidence(project_root, tmp_path, config)

    def fail(**kwargs):
        raise RuntimeError(f"bounded failure {kwargs['lineage'].parent_config_id}")

    with pytest.raises(RuntimeError, match="bounded failure"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=run_root,
            execute=True,
            max_lineages=1,
            execution_phase="pilot",
            trainer=fail,
            **evidence,
        )

    recovered = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=run_root,
        execute=True,
        max_lineages=1,
        execution_phase="pilot",
        trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        **evidence,
    )
    lineage_id = recovered["executed_lineage_ids"][0]
    assert recovered["failed_attempts"] == {lineage_id: 1}
    assert (run_root / "ledgers/failed-runs.jsonl").is_file()


def test_execution_refuses_missing_phase_and_oversized_pilot(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="execution_phase"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "missing-phase",
            execute=True,
            max_lineages=1,
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        )
    with pytest.raises(ValueError, match="at most 6 adapter runs"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "oversized-pilot",
            execute=True,
            max_lineages=4,
            execution_phase="pilot",
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        )


def test_valid_pilot_refuses_missing_execution_evidence_before_state_or_training(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    calls: list[str] = []

    with pytest.raises(ValueError, match="execution evidence is required"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "refused-without-evidence",
            execute=True,
            max_lineages=1,
            execution_phase="pilot",
            trainer=lambda **kwargs: calls.append(kwargs["lineage"].parent_config_id),
        )

    assert calls == []
    assert not (tmp_path / "refused-without-evidence/cohort/run-state.json").exists()


def test_valid_pilot_refuses_when_current_runtime_disappears(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    evidence = _execution_evidence(project_root, tmp_path, config)
    monkeypatch.setattr(
        "lora_audit.readiness.accelerator_snapshot",
        lambda: {"accelerator_available": False, "devices": []},
    )

    with pytest.raises(ValueError, match="current runtime no longer matches contract"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "runtime-disappeared",
            execute=True,
            max_lineages=1,
            execution_phase="pilot",
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
            **evidence,
        )

    assert not (tmp_path / "runtime-disappeared/cohort/run-state.json").exists()


def test_scheduler_refuses_pair_that_has_not_completed_qc(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    evidence = _execution_evidence(project_root, tmp_path, config)

    def pending_pair(**kwargs):
        return [
            entry.model_copy(update={"qc_status": "pending"})
            for entry in _pair(config, kwargs["lineage"])
        ]

    with pytest.raises(ValueError, match="QC remains pending"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "pending-qc",
            execute=True,
            max_lineages=1,
            execution_phase="pilot",
            trainer=pending_pair,
            **evidence,
        )

    state = json.loads((tmp_path / "pending-qc/cohort/run-state.json").read_text(encoding="utf-8"))
    assert state["completed_lineage_ids"] == []
    assert sum(state["failed_attempts"].values()) == 1


def test_cohort_execution_requires_observed_pilot_and_budget(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    estimate = tmp_path / "compute.json"
    conservative_estimate = estimate_compute(
        pilot_device_hours=[],
        adapter_size_gib=0.15,
        available_device_hours=None,
        config_sha256=config.config_hash,
    )
    estimate.write_text(json.dumps(conservative_estimate), encoding="utf-8")
    cohort_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        phase="cohort",
        compute_estimate=estimate,
    )
    with pytest.raises(ValueError, match="observed pilot"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "refused",
            execute=True,
            execution_phase="cohort",
            compute_estimate=estimate,
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
            **cohort_evidence,
        )

    source_manifest = json.loads(cohort_evidence["source_manifest"].read_text(encoding="utf-8"))
    observed_estimate = _observed_compute_estimate_payload(
        config.config_hash,
        estimate_path=estimate,
        source_tree_sha256=source_manifest["tree_sha256"],
        source_manifest_sha256=sha256_file(cohort_evidence["source_manifest"]),
    )
    estimate.write_text(json.dumps(observed_estimate), encoding="utf-8")
    cohort_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        phase="cohort",
        compute_estimate=estimate,
    )
    selected_cohort_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        phase="cohort",
        compute_estimate=estimate,
        lineage_ids=[plan_cohort(config)[0].parent_config_id],
    )
    authorization_tamper = dict(observed_estimate)
    authorization_tamper["limitations"] = [*observed_estimate["limitations"], "tampered"]
    estimate.write_text(json.dumps(authorization_tamper), encoding="utf-8")
    with pytest.raises(ValueError, match="compute authorization mismatch"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "authorization-refused",
            execute=True,
            max_lineages=1,
            execution_phase="cohort",
            compute_estimate=estimate,
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
            **selected_cohort_evidence,
        )
    assert not (tmp_path / "authorization-refused/cohort/run-state.json").exists()
    estimate.write_text(json.dumps(observed_estimate), encoding="utf-8")
    result = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "accepted",
        execute=True,
        max_lineages=1,
        execution_phase="cohort",
        compute_estimate=estimate,
        trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        **selected_cohort_evidence,
    )
    assert result["execution_gate"]["recommendation"] == "full_research"
    assert result["execution_gate"]["pilot_run_count"] == 6

    threshold_estimate = _observed_compute_estimate_payload(
        config.config_hash,
        estimate_path=estimate,
        source_tree_sha256=source_manifest["tree_sha256"],
        source_manifest_sha256=sha256_file(cohort_evidence["source_manifest"]),
        first_pair_failure_reason_codes=[
            "backdoored_asr_below_threshold",
            "paired_asr_lift_below_threshold",
        ],
    )
    estimate.write_text(json.dumps(threshold_estimate), encoding="utf-8")
    threshold_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        phase="cohort",
        compute_estimate=estimate,
        lineage_ids=[plan_cohort(config)[0].parent_config_id],
    )
    retained = orchestrate_cohort(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=tmp_path / "threshold-failure-retained",
        execute=True,
        max_lineages=1,
        execution_phase="cohort",
        compute_estimate=estimate,
        trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
        **threshold_evidence,
    )
    assert retained["execution_gate"]["recommendation"] == "full_research"
    assert threshold_estimate["pilot_threshold_failure_pair_count"] == 1

    integrity_estimate = _observed_compute_estimate_payload(
        config.config_hash,
        estimate_path=estimate,
        source_tree_sha256=source_manifest["tree_sha256"],
        source_manifest_sha256=sha256_file(cohort_evidence["source_manifest"]),
        first_pair_failure_reason_codes=["zero_eligible_rows"],
    )
    estimate.write_text(json.dumps(integrity_estimate), encoding="utf-8")
    integrity_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        phase="cohort",
        compute_estimate=estimate,
        lineage_ids=[plan_cohort(config)[0].parent_config_id],
    )
    with pytest.raises(ValueError, match="integrity QC failures"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "integrity-failure-refused",
            execute=True,
            max_lineages=1,
            execution_phase="cohort",
            compute_estimate=estimate,
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
            **integrity_evidence,
        )
    assert not (tmp_path / "integrity-failure-refused/cohort/run-state.json").exists()

    stale = tmp_path / "stale.json"
    stale_estimate = _observed_compute_estimate_payload(
        "0" * 64,
        estimate_path=stale,
    )
    stale.write_text(json.dumps(stale_estimate), encoding="utf-8")
    stale_evidence = _execution_evidence(
        project_root,
        tmp_path,
        config,
        phase="cohort",
        compute_estimate=stale,
        lineage_ids=[plan_cohort(config)[0].parent_config_id],
    )
    with pytest.raises(ValueError, match="config hash mismatch"):
        orchestrate_cohort(
            config=config,
            prepared_manifest=prepared,
            project_root=project_root,
            output_root=tmp_path / "stale-refused",
            execute=True,
            max_lineages=1,
            execution_phase="cohort",
            compute_estimate=stale,
            trainer=lambda **kwargs: _pair(config, kwargs["lineage"]),
            **stale_evidence,
        )
    assert not (tmp_path / "stale-refused/cohort/run-state.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pilot_run_count", "2", "invalid pilot_run_count"),
        ("pilot_pair_count", True, "invalid pilot_pair_count"),
        ("pilot_evaluation_pair_count", 1.0, "invalid pilot_evaluation_pair_count"),
        ("pilot_failed_pair_count", False, "invalid pilot_failed_pair_count"),
        ("available_device_hours", "100.0", "invalid declared device-hour budget"),
    ],
)
def test_compute_gate_rejects_loosely_typed_receipt_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    config_sha256 = "f" * 64
    estimate = tmp_path / "compute.json"
    payload = _observed_compute_estimate_payload(
        config_sha256,
        estimate_path=estimate,
    )
    payload[field] = value
    estimate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256=config_sha256,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


def test_compute_gate_rejects_cross_lineage_adapter_reuse(tmp_path: Path) -> None:
    config_sha256 = "f" * 64
    estimate = tmp_path / "compute.json"
    payload = _observed_compute_estimate_payload(
        config_sha256,
        estimate_path=estimate,
    )
    bindings = [
        {
            "parent_config_id": "pair-1",
            "clean_adapter_sha256": "c" * 64,
            "backdoored_adapter_sha256": "d" * 64,
        },
        {
            "parent_config_id": "pair-2",
            "clean_adapter_sha256": "c" * 64,
            "backdoored_adapter_sha256": "e" * 64,
        },
    ]
    payload.update(
        {
            "pilot_run_count": 4,
            "pilot_pair_count": 2,
            "pilot_evaluation_pair_count": 2,
            "pilot_pair_bindings": bindings,
            "pilot_pair_binding_sha256": sha256_json(bindings),
        }
    )
    estimate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reuses adapter hashes across lineages"):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256=config_sha256,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


def test_compute_gate_rejects_invalid_memory_fraction(tmp_path: Path) -> None:
    config_sha256 = "f" * 64
    estimate = tmp_path / "compute.json"
    payload = _observed_compute_estimate_payload(
        config_sha256,
        estimate_path=estimate,
    )
    payload["observed_peak_reserved_memory_fraction_per_adapter_max"] = 1.01
    estimate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid memory-fraction evidence"):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256=config_sha256,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


def test_compute_gate_rejects_semantically_ignorable_ledger_byte_tamper(
    tmp_path: Path,
) -> None:
    config_sha256 = "f" * 64
    estimate = tmp_path / "compute.json"
    payload = _observed_compute_estimate_payload(
        config_sha256,
        estimate_path=estimate,
    )
    estimate.write_text(json.dumps(payload), encoding="utf-8")
    ledger = tmp_path / str(payload["pilot_training_ledger_path"])
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pilot training ledger hash mismatch"):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256=config_sha256,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("planning_device_hours_per_adapter", 99.0),
        ("recommendation", "target"),
        ("core", {"adapter_count": 72, "planning_device_hours": 0.001}),
    ],
)
def test_compute_gate_rejects_ledger_derived_field_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config_sha256 = "f" * 64
    estimate = tmp_path / "compute.json"
    payload = _observed_compute_estimate_payload(
        config_sha256,
        estimate_path=estimate,
    )
    payload[field] = value
    estimate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match ledger-derived evidence"):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256=config_sha256,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../outside.jsonl", "invalid pilot training ledger path"),
        ("missing.jsonl", "missing or outside its bundle"),
    ],
)
def test_compute_gate_rejects_uncontained_or_missing_ledger_path(
    tmp_path: Path,
    path: str,
    message: str,
) -> None:
    config_sha256 = "f" * 64
    estimate = tmp_path / "compute.json"
    payload = _observed_compute_estimate_payload(
        config_sha256,
        estimate_path=estimate,
    )
    payload["pilot_training_ledger_path"] = path
    estimate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256=config_sha256,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_tree_sha256", "source tree hash mismatch"),
        ("source_manifest_sha256", "source manifest hash mismatch"),
    ],
)
def test_compute_gate_rejects_stale_source_binding(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    config_sha256 = "f" * 64
    estimate = tmp_path / "compute.json"
    payload = _observed_compute_estimate_payload(
        config_sha256,
        estimate_path=estimate,
    )
    payload[field] = "3" * 64
    estimate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256=config_sha256,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize("max_lineages", [None, 0, True, 4])
def test_pilot_compute_gate_requires_one_to_three_lineages(max_lineages: object) -> None:
    with pytest.raises(ValueError, match="1-3 paired lineages"):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="pilot",
            max_lineages=max_lineages,
            compute_estimate_path=None,
            config_sha256="f" * 64,
            planned_adapter_runs=2,
        )


def test_compute_gate_rejects_non_object_receipt(tmp_path: Path) -> None:
    estimate = tmp_path / "compute.json"
    estimate.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        verify_execution_compute_gate(
            profile="core",
            execution_phase="cohort",
            max_lineages=None,
            compute_estimate_path=estimate,
            config_sha256="f" * 64,
            planned_adapter_runs=72,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
        )


def test_observed_compute_estimate_uses_maximum_with_safety_margin() -> None:
    result = estimate_compute(
        pilot_device_hours=[1.0, 2.0],
        adapter_size_gib=0.15,
        available_device_hours=300.0,
        config_sha256="a" * 64,
        observed_adapter_sizes_gib=[0.12, 0.13],
        pilot_evaluation_device_hours=[0.5],
        source_tree_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        **_pilot_memory_arguments(),
    )

    assert result["estimate_kind"] == "observed_pilot"
    assert result["pilot_pair_count"] == 1
    assert result["observed_device_hours_per_adapter_p90"] == 2.0
    assert result["planning_device_hours_per_adapter"] == 2.5
    assert result["core"]["pilot_consumed_device_hours"] == 3.5
    assert result["core"]["remaining_training_device_hours"] == 175.0
    assert result["core"]["remaining_evaluation_device_hours"] == 21.875
    assert result["core"]["planning_device_hours"] == 200.375
    assert result["cohort_recommendation"] == "core"
    assert result["recommendation"] == "core_only"
    assert result["source_tree_sha256"] == "1" * 64
    assert result["source_manifest_sha256"] == "2" * 64
    assert result["observed_peak_allocated_memory_gib_per_adapter_p90"] == 9.0
    assert result["observed_peak_reserved_memory_fraction_per_adapter_max"] == 11.0 / 22.5


def test_observed_compute_estimate_requires_source_binding() -> None:
    with pytest.raises(ValueError, match="source tree and manifest bindings"):
        estimate_compute(
            pilot_device_hours=[1.0, 2.0],
            adapter_size_gib=0.15,
            available_device_hours=300.0,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.12, 0.13],
            pilot_evaluation_device_hours=[0.5],
            **_pilot_memory_arguments(),
        )


def test_observed_compute_estimate_requires_complete_memory_evidence() -> None:
    with pytest.raises(ValueError, match="complete peak-memory telemetry"):
        estimate_compute(
            pilot_device_hours=[1.0, 2.0],
            adapter_size_gib=0.15,
            available_device_hours=300.0,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.12, 0.13],
            pilot_evaluation_device_hours=[0.5],
            source_tree_sha256="1" * 64,
            source_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize("observations", [[1.0], [1.0, float("nan")], [1.0, 0.0]])
def test_observed_compute_estimate_rejects_incomplete_or_invalid_pairs(
    observations: list[float],
) -> None:
    with pytest.raises(ValueError, match="complete positive adapter pairs"):
        estimate_compute(
            pilot_device_hours=observations,
            adapter_size_gib=0.15,
            available_device_hours=None,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.12] * len(observations),
        )


@pytest.mark.parametrize(
    "field",
    [
        "adapter_size_gib",
        "available_device_hours",
        "conservative_device_hours_per_adapter",
        "planning_safety_multiplier",
        "conservative_evaluation_device_hours_per_pair",
    ],
)
def test_compute_estimate_rejects_boolean_scalar_inputs(field: str) -> None:
    arguments: dict[str, object] = {
        "pilot_device_hours": [],
        "adapter_size_gib": 0.15,
        "available_device_hours": None,
    }
    arguments[field] = True

    with pytest.raises(ValueError):
        estimate_compute(**arguments)


def test_observed_compute_estimate_rejects_boolean_observations() -> None:
    with pytest.raises(ValueError, match="complete positive adapter pairs"):
        estimate_compute(
            pilot_device_hours=[True, 0.25],
            adapter_size_gib=0.15,
            available_device_hours=None,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.12, 0.13],
            pilot_evaluation_device_hours=[0.05],
        )

    with pytest.raises(ValueError, match="failed-pair count"):
        estimate_compute(
            pilot_device_hours=[0.2, 0.25],
            adapter_size_gib=0.15,
            available_device_hours=None,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.12, 0.13],
            pilot_evaluation_device_hours=[0.05],
            pilot_failed_pair_count=True,
        )


def test_pilot_ledger_requires_complete_hash_bound_device_observations(
    tmp_path: Path,
) -> None:
    config_sha256 = "a" * 64
    rows = _training_telemetry_rows(config_sha256)
    ledger = tmp_path / "pilot.jsonl"
    _write_jsonl(ledger, rows)

    telemetry = read_pilot_telemetry(
        ledger,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256="1" * 64,
        expected_source_manifest_sha256="2" * 64,
        required_compute_capability=(8, 9),
        minimum_device_memory_gib=20.0,
    )
    assert telemetry.pair_bindings == (PilotPairBinding("pair-1", "c" * 64, "d" * 64),)
    assert telemetry.peak_allocated_memory_gib == (8.0, 8.0)
    assert telemetry.peak_reserved_memory_gib == (10.0, 10.0)
    assert telemetry.device_memory_gib == (22.5, 22.5)
    assert read_pilot_device_hours(
        ledger,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256="1" * 64,
        expected_source_manifest_sha256="2" * 64,
        required_compute_capability=(8, 9),
        minimum_device_memory_gib=20.0,
    ) == [1.0 / 3600.0, 1.0 / 3600.0]

    drifted = json.loads(json.dumps(rows))
    drifted[0]["source_tree_sha256"] = "0" * 64
    _write_jsonl(ledger, drifted)
    with pytest.raises(ValueError, match="source tree hash mismatch"):
        read_pilot_telemetry(
            ledger,
            expected_config_sha256=config_sha256,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            required_compute_capability=(8, 9),
            minimum_device_memory_gib=20.0,
        )

    rows[1]["adapter_label"] = "clean"
    _write_jsonl(ledger, rows)
    with pytest.raises(ValueError, match="duplicate pilot telemetry key"):
        read_pilot_device_hours(
            ledger,
            expected_config_sha256=config_sha256,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            required_compute_capability=(8, 9),
            minimum_device_memory_gib=20.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_record_count", 63, "mismatched record counts"),
        ("records_sha256", "b" * 64, "identical training data hashes"),
    ],
)
def test_pilot_ledger_rejects_incoherent_training_pairs(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    config_sha256 = "a" * 64
    rows = _training_telemetry_rows(config_sha256)
    rows[1][field] = value
    ledger = tmp_path / "pilot.jsonl"
    _write_jsonl(ledger, rows)

    with pytest.raises(ValueError, match=message):
        read_pilot_telemetry(
            ledger,
            expected_config_sha256=config_sha256,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            required_compute_capability=(8, 9),
            minimum_device_memory_gib=20.0,
        )


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_pilot_ledger_rejects_schema_drift(tmp_path: Path, mutation: str) -> None:
    config_sha256 = "a" * 64
    rows = _training_telemetry_rows(config_sha256)
    if mutation == "missing":
        rows[0].pop("backend")
    else:
        rows[0]["unexpected_field"] = "unexpected"
    ledger = tmp_path / "pilot.jsonl"
    _write_jsonl(ledger, rows)

    with pytest.raises(ValueError, match="field set mismatch"):
        read_pilot_telemetry(
            ledger,
            expected_config_sha256=config_sha256,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            required_compute_capability=(8, 9),
            minimum_device_memory_gib=20.0,
        )


def test_pilot_ledger_rejects_peak_memory_above_device_capacity(tmp_path: Path) -> None:
    config_sha256 = "a" * 64
    rows = _training_telemetry_rows(config_sha256)
    rows[0]["peak_reserved_memory_gib"] = 80.0
    ledger = tmp_path / "pilot.jsonl"
    _write_jsonl(ledger, rows)

    with pytest.raises(ValueError, match="inconsistent peak memory"):
        read_pilot_telemetry(
            ledger,
            expected_config_sha256=config_sha256,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            required_compute_capability=(8, 9),
            minimum_device_memory_gib=20.0,
        )


def test_pilot_evaluation_ledger_binds_qc_runtime_and_device(tmp_path: Path) -> None:
    config_sha256 = "a" * 64
    ledger = tmp_path / "pilot-evaluation.jsonl"
    row = {
        "parent_config_id": "pair-1",
        "replicate_seed": 101,
        "pair_training_seed": 202,
        "poison_selection_seed": 303,
        "probe_seed": 404,
        "split_assignment_id": "test-split-1",
        "device_hours": 1.0 / 3600.0,
        "runtime_seconds": 1.0,
        "peak_allocated_memory_gib": 12.0,
        "peak_reserved_memory_gib": 14.0,
        "probe_group_count": 128,
        "test_record_count": 1000,
        "qc_valid": True,
        "qc_reason_codes": [],
        "evaluation_report_sha256": "b" * 64,
        "clean_adapter_sha256": "c" * 64,
        "backdoored_adapter_sha256": "d" * 64,
        "device_observation": _qualified_device(),
        "backend": "cuda",
        "config_sha256": config_sha256,
        "source_tree_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "execution_profile": TEST_EXECUTION_PROFILE.binding(),
    }
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = read_pilot_evaluation_telemetry(
        ledger,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256="1" * 64,
        expected_source_manifest_sha256="2" * 64,
        required_compute_capability=(8, 9),
        minimum_device_memory_gib=20.0,
    )

    assert result.device_hours == (1.0 / 3600.0,)
    assert result.peak_allocated_memory_gib == (12.0,)
    assert result.peak_reserved_memory_gib == (14.0,)
    assert result.device_memory_gib == (22.5,)
    assert result.failed_pair_count == 0
    assert result.threshold_failure_pair_count == 0
    assert result.integrity_failure_pair_count == 0
    assert result.failure_reason_codes == ()
    assert result.pair_bindings == (PilotPairBinding("pair-1", "c" * 64, "d" * 64),)
    row["qc_valid"] = False
    row["qc_reason_codes"] = [
        "backdoored_asr_below_threshold",
        "paired_asr_lift_below_threshold",
    ]
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    threshold_failure = read_pilot_evaluation_telemetry(
        ledger,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256="1" * 64,
        expected_source_manifest_sha256="2" * 64,
        required_compute_capability=(8, 9),
        minimum_device_memory_gib=20.0,
    )
    assert threshold_failure.failed_pair_count == 1
    assert threshold_failure.threshold_failure_pair_count == 1
    assert threshold_failure.integrity_failure_pair_count == 0

    row["qc_reason_codes"] = ["zero_eligible_rows"]
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    integrity_failure = read_pilot_evaluation_telemetry(
        ledger,
        expected_config_sha256=config_sha256,
        expected_source_tree_sha256="1" * 64,
        expected_source_manifest_sha256="2" * 64,
        required_compute_capability=(8, 9),
        minimum_device_memory_gib=20.0,
    )
    assert integrity_failure.threshold_failure_pair_count == 0
    assert integrity_failure.integrity_failure_pair_count == 1


def test_pilot_compute_alignment_rejects_wrong_lineage_or_adapter_hash() -> None:
    binding = PilotPairBinding("pair-1", "c" * 64, "d" * 64)
    training = PilotTelemetry(
        device_hours=(0.1, 0.1),
        adapter_sizes_gib=(0.01, 0.01),
        peak_allocated_memory_gib=(8.0, 9.0),
        peak_reserved_memory_gib=(10.0, 11.0),
        device_memory_gib=(22.5, 22.5),
        pair_bindings=(binding,),
    )
    evaluation = PilotEvaluationTelemetry(
        device_hours=(0.05,),
        peak_allocated_memory_gib=(12.0,),
        peak_reserved_memory_gib=(14.0,),
        device_memory_gib=(22.5,),
        failed_pair_count=0,
        pair_bindings=(binding,),
    )
    assert verify_pilot_telemetry_alignment(training, evaluation) == (binding,)

    wrong_hash = PilotPairBinding("pair-1", "e" * 64, "d" * 64)
    mismatched = PilotEvaluationTelemetry(
        device_hours=(0.05,),
        peak_allocated_memory_gib=(12.0,),
        peak_reserved_memory_gib=(14.0,),
        device_memory_gib=(22.5,),
        failed_pair_count=0,
        pair_bindings=(wrong_hash,),
    )
    with pytest.raises(ValueError, match="lineage or adapter hash mismatch"):
        verify_pilot_telemetry_alignment(training, mismatched)

    with pytest.raises(ValueError, match="distinct adapter hashes"):
        PilotPairBinding("pair-1", "c" * 64, "c" * 64)


def test_pilot_telemetry_rejects_cross_lineage_adapter_reuse() -> None:
    bindings = (
        PilotPairBinding("pair-1", "c" * 64, "d" * 64),
        PilotPairBinding("pair-2", "c" * 64, "e" * 64),
    )

    with pytest.raises(ValueError, match="reuses adapter hashes across lineages"):
        PilotTelemetry(
            device_hours=(0.1, 0.1, 0.1, 0.1),
            adapter_sizes_gib=(0.01, 0.01, 0.01, 0.01),
            peak_allocated_memory_gib=(8.0, 8.0, 8.0, 8.0),
            peak_reserved_memory_gib=(10.0, 10.0, 10.0, 10.0),
            device_memory_gib=(22.5, 22.5, 22.5, 22.5),
            pair_bindings=bindings,
        )


def _completed_cohort_state(
    project_root: Path,
    run_root: Path,
    prepared: Path,
    config,
    completion_order: list[str],
    *,
    status: str,
):
    plan = plan_cohort(config)
    source = verify_source_snapshot(project_root, project_root / "artifacts/source-manifest.json")
    bundle = verify_bundle_snapshot(
        project_root,
        project_root / "artifacts/bundle-manifest.json",
        source_tree_sha256=source["tree_sha256"],
        source_entry_count=source["entry_count"],
    )
    state_path = run_root / "cohort" / "run-state.json"
    state = load_or_create_cohort_state(
        state_path,
        config=config,
        plan=plan,
        prepared_manifest_sha256=sha256_file(prepared),
        execution_profile=TEST_EXECUTION_PROFILE,
        execution_profile_path=TEST_EXECUTION_PROFILE_PATH,
        project_root=project_root,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for lineage_id in completion_order:
        manifest = state_path.parent / f"{lineage_id}.jsonl"
        manifest.write_text(f"{lineage_id}\n", encoding="utf-8")
        hashes[lineage_id] = sha256_file(manifest)
    state = state.model_copy(
        update={
            "completed_lineage_ids": completion_order,
            "lineage_manifest_hashes": hashes,
            "status": status,
            "source_tree_sha256": source["tree_sha256"],
            "source_manifest_sha256": source["manifest_sha256"],
            "source_entry_count": source["entry_count"],
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "bundle_tree_sha256": bundle["tree_sha256"],
            "bundle_entry_count": bundle["entry_count"],
        }
    )
    write_cohort_state(state, state_path)
    return state_path


def _verify_completed_cohort(project_root: Path, run_root: Path, prepared: Path, config):
    return verify_completed_cohort_run(
        config=config,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=run_root,
        source_manifest=project_root / "artifacts/source-manifest.json",
        execution_profile=TEST_EXECUTION_PROFILE,
        execution_profile_path=TEST_EXECUTION_PROFILE_PATH,
        bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
    )


def test_research_gate_accepts_pilot_first_cohort_completion_order(
    project_root: Path, tmp_path: Path
) -> None:
    """The pilot finishes a factor-covering triple before the cohort batches run.

    Completion order is therefore a permutation of plan order even when every pair
    succeeds, so the research gate must key on membership rather than list order.
    """
    config = load_config(project_root / "configs/core.yaml")
    plan = plan_cohort(config)
    plan_ids = [lineage.parent_config_id for lineage in plan]
    pilot_ids = [
        lineage.parent_config_id
        for lineage in select_representative_pilot_lineages(plan, pair_count=3)
    ]
    completion_order = pilot_ids + [item for item in plan_ids if item not in set(pilot_ids)]
    assert set(completion_order) == set(plan_ids)
    assert completion_order != plan_ids, "pilot triple must not be a plan prefix"

    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    run_root = tmp_path / "run"
    _completed_cohort_state(
        project_root, run_root, prepared, config, completion_order, status="completed"
    )

    state, _ = _verify_completed_cohort(project_root, run_root, prepared, config)

    assert state.status == "completed"
    assert state.completed_lineage_ids == completion_order


def test_research_gate_still_refuses_an_incomplete_cohort(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    plan_ids = [lineage.parent_config_id for lineage in plan_cohort(config)]
    prepared = tmp_path / "manifest.jsonl"
    prepared.write_text("\n", encoding="utf-8")
    run_root = tmp_path / "run"
    _completed_cohort_state(
        project_root, run_root, prepared, config, plan_ids[:-1], status="active"
    )

    with pytest.raises(ValueError, match="complete current source-bound core cohort"):
        _verify_completed_cohort(project_root, run_root, prepared, config)
