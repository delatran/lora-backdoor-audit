"""Fail-closed execution-profile coverage for CLI and pilot compute evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from lora_audit import cli
from lora_audit.compute import (
    estimate_compute,
    read_pilot_evaluation_telemetry,
    read_pilot_telemetry,
    verify_pilot_telemetry_alignment,
)
from lora_audit.execution_profiles import (
    ExecutionProfile,
    canonical_execution_profile_path,
    load_execution_profile,
)
from lora_audit.hashing import sha256_json


def _profile(project_root: Path):
    path = canonical_execution_profile_path(project_root, "l4-24gb-v1")
    return load_execution_profile(path), path


def _device_observation() -> dict[str, object]:
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


def _training_rows(profile_binding: dict[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for label in ("clean", "backdoored"):
        result.append(
            {
                "parent_config_id": "pilot-lineage-1",
                "adapter_label": label,
                "replicate_seed": 7,
                "pair_training_seed": 101,
                "poison_selection_seed": 202,
                "probe_seed": 303,
                "split_assignment_id": "core-cyclic-v1-cell-00-replicate-00-slot-00",
                "training_unit_id": f"unit-{label}",
                "device_hours": 1.0 / 3600.0,
                "runtime_seconds": 1.0,
                "peak_allocated_memory_gib": 8.0,
                "peak_reserved_memory_gib": 10.0,
                "training_record_count": 64,
                "records_sha256": ("b" if label == "clean" else "e") * 64,
                "adapter_size_bytes": 1024,
                "adapter_sha256": ("c" if label == "clean" else "d") * 64,
                "device_observation": _device_observation(),
                "backend": "cuda",
                "framework_version": "2.13.0",
                "config_sha256": "a" * 64,
                "source_tree_sha256": "1" * 64,
                "source_manifest_sha256": "2" * 64,
                "execution_profile": profile_binding,
            }
        )
    return result


def _evaluation_rows(profile_binding: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "parent_config_id": "pilot-lineage-1",
            "replicate_seed": 7,
            "pair_training_seed": 101,
            "poison_selection_seed": 202,
            "probe_seed": 303,
            "split_assignment_id": "core-cyclic-v1-cell-00-replicate-00-slot-00",
            "device_hours": 1.0 / 3600.0,
            "runtime_seconds": 1.0,
            "peak_allocated_memory_gib": 12.0,
            "peak_reserved_memory_gib": 14.0,
            "probe_group_count": 128,
            "test_record_count": 64,
            "qc_valid": True,
            "qc_reason_codes": [],
            "evaluation_report_sha256": "f" * 64,
            "clean_adapter_sha256": "c" * 64,
            "backdoored_adapter_sha256": "d" * 64,
            "device_observation": _device_observation(),
            "backend": "cuda",
            "config_sha256": "a" * 64,
            "source_tree_sha256": "1" * 64,
            "source_manifest_sha256": "2" * 64,
            "execution_profile": profile_binding,
        }
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_observed_compute_requires_and_records_exact_execution_profile(
    project_root: Path, tmp_path: Path
) -> None:
    profile, _ = _profile(project_root)
    training_path = tmp_path / "training.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(training_path, _training_rows(profile.binding()))
    _write_jsonl(evaluation_path, _evaluation_rows(profile.binding()))

    training = read_pilot_telemetry(
        training_path,
        expected_config_sha256="a" * 64,
        expected_source_tree_sha256="1" * 64,
        expected_source_manifest_sha256="2" * 64,
        execution_profile=profile,
    )
    evaluation = read_pilot_evaluation_telemetry(
        evaluation_path,
        expected_config_sha256="a" * 64,
        expected_source_tree_sha256="1" * 64,
        expected_source_manifest_sha256="2" * 64,
        execution_profile=profile,
    )
    assert verify_pilot_telemetry_alignment(training, evaluation)

    result = estimate_compute(
        pilot_device_hours=list(training.device_hours),
        adapter_size_gib=0.15,
        available_device_hours=1.0,
        config_sha256="a" * 64,
        observed_adapter_sizes_gib=list(training.adapter_sizes_gib),
        pilot_evaluation_device_hours=list(evaluation.device_hours),
        source_tree_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        pilot_peak_allocated_memory_gib=list(training.peak_allocated_memory_gib),
        pilot_peak_reserved_memory_gib=list(training.peak_reserved_memory_gib),
        pilot_device_memory_gib=list(training.device_memory_gib),
        pilot_evaluation_peak_allocated_memory_gib=list(evaluation.peak_allocated_memory_gib),
        pilot_evaluation_peak_reserved_memory_gib=list(evaluation.peak_reserved_memory_gib),
        pilot_evaluation_device_memory_gib=list(evaluation.device_memory_gib),
        execution_profile=profile,
    )

    assert result["schema_version"] == 9
    assert result["scope"] == "full_research_pipeline"
    assert result["execution_profile"] == profile.binding()


def test_full_pipeline_compute_requires_three_pairs_and_counts_every_gpu_stage(
    project_root: Path,
) -> None:
    profile, _ = _profile(project_root)

    def estimate(pair_count: int) -> dict:
        adapter_count = pair_count * 2
        return estimate_compute(
            pilot_device_hours=[0.01] * adapter_count,
            adapter_size_gib=0.15,
            available_device_hours=12.0,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.15] * adapter_count,
            pilot_evaluation_device_hours=[0.005] * pair_count,
            source_tree_sha256="1" * 64,
            source_manifest_sha256="2" * 64,
            pilot_peak_allocated_memory_gib=[20.0] * adapter_count,
            pilot_peak_reserved_memory_gib=[22.0] * adapter_count,
            pilot_device_memory_gib=[79.0] * adapter_count,
            pilot_evaluation_peak_allocated_memory_gib=[18.0] * pair_count,
            pilot_evaluation_peak_reserved_memory_gib=[20.0] * pair_count,
            pilot_evaluation_device_memory_gib=[79.0] * pair_count,
            execution_profile=profile,
        )

    partial = estimate(1)
    assert partial["recommendation"] == "pilot_incomplete"
    assert partial["pilot_continuation"]["action"] == "continue_pilot"
    assert partial["core"]["remaining_adapter_count"] == 70
    assert partial["core"]["remaining_pair_count"] == 35

    complete = estimate(3)
    assert complete["recommendation"] == "full_research"
    assert complete["pilot_continuation"]["action"] == "authorize_full_research"
    assert complete["core"]["completed_pilot_adapter_count"] == 6
    assert complete["core"]["remaining_adapter_count"] == 66
    assert complete["full_research"]["rq2_adaptive_robustness"]["adapter_count"] == 12
    assert complete["full_research"]["rq3_cross_language_transfer"]["evaluation_unit_count"] == 24
    assert complete["full_research"]["planning_device_hours"] == pytest.approx(1.48125)
    assert complete["budget_gate_device_hours"] == pytest.approx(9.6)
    assert complete["budget_margin_device_hours"] == pytest.approx(9.6 - 1.48125)


def test_pilot_continuation_only_stops_when_optimistic_budget_bound_is_infeasible(
    project_root: Path,
) -> None:
    profile, _ = _profile(project_root)

    def estimate(training_hours: float) -> dict:
        return estimate_compute(
            pilot_device_hours=[training_hours, training_hours],
            adapter_size_gib=0.15,
            available_device_hours=12.0,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.15, 0.15],
            pilot_evaluation_device_hours=[0.001],
            source_tree_sha256="1" * 64,
            source_manifest_sha256="2" * 64,
            pilot_peak_allocated_memory_gib=[20.0, 20.0],
            pilot_peak_reserved_memory_gib=[22.0, 22.0],
            pilot_device_memory_gib=[79.0, 79.0],
            pilot_evaluation_peak_allocated_memory_gib=[18.0],
            pilot_evaluation_peak_reserved_memory_gib=[20.0],
            pilot_evaluation_device_memory_gib=[79.0],
            execution_profile=profile,
        )

    recoverable = estimate(0.0955)
    assert recoverable["recommendation"] == "core_only"
    assert recoverable["pilot_continuation"]["action"] == "continue_pilot"
    assert (
        recoverable["pilot_continuation"]["optimistic_full_research_lower_bound_device_hours"]
        <= recoverable["budget_gate_device_hours"]
    )

    infeasible = estimate(0.1)
    assert infeasible["recommendation"] == "core_only"
    assert infeasible["pilot_continuation"]["action"] == "stop_budget_infeasible"
    assert (
        infeasible["pilot_continuation"]["optimistic_full_research_lower_bound_device_hours"]
        > infeasible["budget_gate_device_hours"]
    )


def test_threshold_qc_failure_is_retained_but_integrity_failure_stops_research(
    project_root: Path,
) -> None:
    profile, _ = _profile(project_root)

    def estimate(*, threshold_failures: int, integrity_failures: int, reasons: list[str]) -> dict:
        return estimate_compute(
            pilot_device_hours=[0.01] * 6,
            adapter_size_gib=0.15,
            available_device_hours=12.0,
            config_sha256="a" * 64,
            observed_adapter_sizes_gib=[0.15] * 6,
            pilot_evaluation_device_hours=[0.005] * 3,
            pilot_failed_pair_count=threshold_failures + integrity_failures,
            pilot_threshold_failure_pair_count=threshold_failures,
            pilot_integrity_failure_pair_count=integrity_failures,
            pilot_qc_reason_codes=reasons,
            source_tree_sha256="1" * 64,
            source_manifest_sha256="2" * 64,
            pilot_peak_allocated_memory_gib=[20.0] * 6,
            pilot_peak_reserved_memory_gib=[22.0] * 6,
            pilot_device_memory_gib=[79.0] * 6,
            pilot_evaluation_peak_allocated_memory_gib=[18.0] * 3,
            pilot_evaluation_peak_reserved_memory_gib=[20.0] * 3,
            pilot_evaluation_device_memory_gib=[79.0] * 3,
            execution_profile=profile,
        )

    retained = estimate(
        threshold_failures=1,
        integrity_failures=0,
        reasons=["backdoored_asr_below_threshold", "paired_asr_lift_below_threshold"],
    )
    assert retained["pilot_failed_pair_count"] == 1
    assert retained["pilot_threshold_failure_pair_count"] == 1
    assert retained["pilot_integrity_failure_pair_count"] == 0
    assert retained["recommendation"] == "full_research"
    assert retained["pilot_continuation"]["action"] == "authorize_full_research"
    assert "retained_threshold_failures" in retained["pilot_continuation"]["reason"]

    blocked = estimate(
        threshold_failures=0,
        integrity_failures=1,
        reasons=["zero_eligible_rows"],
    )
    assert blocked["recommendation"] == "protocol_review"
    assert blocked["pilot_continuation"]["action"] == "stop_protocol_review"


def test_pilot_ledger_rejects_profile_binding_or_device_drift(
    project_root: Path, tmp_path: Path
) -> None:
    profile, _ = _profile(project_root)
    rows = _training_rows(profile.binding())
    rows[0]["execution_profile"] = {**profile.binding(), "profile_hash": "0" * 64}
    ledger = tmp_path / "training.jsonl"
    _write_jsonl(ledger, rows)

    with pytest.raises(ValueError, match="execution profile mismatch"):
        read_pilot_telemetry(
            ledger,
            expected_config_sha256="a" * 64,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            execution_profile=profile,
        )

    rows = _training_rows(profile.binding())
    rows[0]["device_observation"] = {
        **_device_observation(),
        "name": "NVIDIA A100-SXM4-40GB",
    }
    _write_jsonl(ledger, rows)
    with pytest.raises(ValueError, match="does not match execution profile"):
        read_pilot_telemetry(
            ledger,
            expected_config_sha256="a" * 64,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            execution_profile=profile,
        )


def test_compute_refuses_self_consistent_forged_execution_profile() -> None:
    semantic_payload = {
        "schema_version": 1,
        "profile_id": "forged-accelerator-v1",
        "backend": "cuda",
        "required_compute_capability": [9, 9],
        "minimum_device_memory_gib": 1.0,
        "maximum_device_memory_gib": 2.0,
        "allowed_device_name_patterns": ["^Forged GPU$"],
        "require_bf16_probe": True,
        "strict_accelerator": True,
    }
    forged = ExecutionProfile(
        schema_version=1,
        profile_id="forged-accelerator-v1",
        backend="cuda",
        required_compute_capability=(9, 9),
        minimum_device_memory_gib=1.0,
        maximum_device_memory_gib=2.0,
        allowed_device_name_patterns=("^Forged GPU$",),
        require_bf16_probe=True,
        strict_accelerator=True,
        profile_hash=sha256_json(semantic_payload),
        profile_file_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="not a locked canonical profile"):
        estimate_compute(
            pilot_device_hours=[0.1, 0.1],
            adapter_size_gib=0.15,
            available_device_hours=1.0,
            execution_profile=forged,
        )


@pytest.mark.parametrize(
    "command",
    [
        "run-cohort",
        "run-adaptive",
        "run-adaptive-condition",
        "finalize-adaptive",
        "run-rq3",
        "run-rq3-cell",
        "finalize-rq3",
    ],
)
def test_execution_commands_require_hardware_profile(
    project_root: Path, tmp_path: Path, command: str
) -> None:
    runner = CliRunner()
    config = project_root / "configs" / "core.yaml"
    arguments_by_command = {
        "run-cohort": [
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--output-root",
            str(tmp_path / "run"),
        ],
        "run-adaptive": [
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--calibration-profile",
            str(config),
            "--source-manifest",
            str(config),
            "--output",
            str(tmp_path / "rq2.json"),
        ],
        "run-adaptive-condition": [
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--calibration-profile",
            str(config),
            "--source-manifest",
            str(config),
            "--condition-id",
            "locked-condition-id",
        ],
        "finalize-adaptive": [
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--calibration-profile",
            str(config),
            "--source-manifest",
            str(config),
            "--output",
            str(tmp_path / "rq2.json"),
        ],
        "run-rq3": [
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--source-manifest",
            str(config),
            "--output",
            str(tmp_path / "rq3.json"),
        ],
        "run-rq3-cell": [
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--source-manifest",
            str(config),
            "--source-language",
            "en-US",
            "--probe-language",
            "en-US",
        ],
        "finalize-rq3": [
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--source-manifest",
            str(config),
            "--output",
            str(tmp_path / "rq3.json"),
        ],
    }

    result = runner.invoke(cli.app, [command, *arguments_by_command[command]])

    assert result.exit_code == 2
    assert "--execution-profile" in result.output


def test_run_cohort_exposes_required_bundle_manifest_gate() -> None:
    result = CliRunner().invoke(cli.app, ["run-cohort", "--help"])

    assert result.exit_code == 0
    assert "--bundle-manife" in result.output


def test_run_cohort_passes_bound_profile_and_bundle_manifest(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, profile_path = _profile(project_root)
    bundle_manifest = tmp_path / "bundle-manifest.json"
    bundle_manifest.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_orchestrate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "preview"}

    monkeypatch.setattr(cli, "orchestrate_cohort", fake_orchestrate)
    config = project_root / "configs" / "core.yaml"
    result = CliRunner().invoke(
        cli.app,
        [
            "run-cohort",
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--output-root",
            str(tmp_path / "run"),
            "--execution-profile",
            str(profile_path),
            "--bundle-manifest",
            str(bundle_manifest),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["execution_profile"] == profile
    assert captured["execution_profile_path"] == profile_path
    assert captured["bundle_manifest_path"] == bundle_manifest


def test_rq_commands_pass_bound_profile_and_private_receipts(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, profile_path = _profile(project_root)
    config = project_root / "configs" / "core.yaml"
    trigger_receipt = tmp_path / "trigger-receipt.json"
    common_receipt = tmp_path / "common-receipt.json"
    waiver_attestation = "a" * 64
    bundle_manifest = tmp_path / "bundle-manifest.json"
    trigger_receipt.write_text("{}\n", encoding="utf-8")
    common_receipt.write_text("{}\n", encoding="utf-8")
    bundle_manifest.write_text("{}\n", encoding="utf-8")
    adaptive_captured: dict[str, object] = {}
    rq3_captured: dict[str, object] = {}

    def fake_adaptive(**kwargs: object) -> dict[str, object]:
        adaptive_captured.update(kwargs)
        return {
            "status": "completed",
            "planned_count": 12,
            "valid_count": 12,
            "failed_count": 0,
        }

    def fake_rq3(**kwargs: object) -> list[object]:
        rq3_captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "run_adaptive_study", fake_adaptive)
    monkeypatch.setattr(cli, "run_rq3_study", fake_rq3)
    runner = CliRunner()
    adaptive = runner.invoke(
        cli.app,
        [
            "run-adaptive",
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--calibration-profile",
            str(config),
            "--source-manifest",
            str(config),
            "--execution-profile",
            str(profile_path),
            "--bundle-manifest",
            str(bundle_manifest),
            "--output",
            str(tmp_path / "rq2.json"),
        ],
    )
    rq3 = runner.invoke(
        cli.app,
        [
            "run-rq3",
            "--config",
            str(config),
            "--prepared-manifest",
            str(config),
            "--run-root",
            str(tmp_path / "run"),
            "--source-manifest",
            str(config),
            "--execution-profile",
            str(profile_path),
            "--bundle-manifest",
            str(bundle_manifest),
            "--output",
            str(tmp_path / "rq3.json"),
            "--trigger-equivalence-receipt",
            str(trigger_receipt),
            "--common-probe-receipt",
            str(common_receipt),
            "--required-trigger-waiver-attestation-sha256",
            waiver_attestation,
        ],
    )

    assert adaptive.exit_code == 0, adaptive.output
    assert rq3.exit_code == 0, rq3.output
    for captured in (adaptive_captured, rq3_captured):
        assert captured["execution_profile"] == profile
        assert captured["execution_profile_path"] == profile_path
        assert captured["bundle_manifest_path"] == bundle_manifest
    assert rq3_captured["trigger_equivalence_receipt"] == trigger_receipt
    assert rq3_captured["common_probe_receipt"] == common_receipt
    assert rq3_captured["required_waiver_attestation_sha256"] == waiver_attestation


def test_granular_rq_commands_pass_exact_selectors_and_bound_profile(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, profile_path = _profile(project_root)
    config = project_root / "configs/core.yaml"
    bundle_manifest = tmp_path / "bundle-manifest.json"
    bundle_manifest.write_text("{}\n", encoding="utf-8")
    captured: dict[str, dict[str, object]] = {}

    def capture(name: str, result: object):
        def command(**kwargs: object) -> object:
            captured[name] = kwargs
            return result

        return command

    monkeypatch.setattr(
        cli,
        "run_adaptive_condition",
        capture(
            "rq2-condition",
            {"status": "completed", "condition_id": "condition-1", "result": "result.json"},
        ),
    )
    monkeypatch.setattr(
        cli,
        "finalize_adaptive_study",
        capture(
            "rq2-finalize",
            {"status": "completed", "planned_count": 12, "valid_count": 10, "failed_count": 2},
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_rq3_cell",
        capture(
            "rq3-cell",
            SimpleNamespace(poison_language="en-US", probe_language="vi-VN"),
        ),
    )
    monkeypatch.setattr(
        cli,
        "finalize_rq3_study",
        capture("rq3-finalize", [object(), object(), object(), object()]),
    )
    adaptive_common = [
        "--config",
        str(config),
        "--prepared-manifest",
        str(config),
        "--run-root",
        str(tmp_path / "run"),
        "--calibration-profile",
        str(config),
        "--source-manifest",
        str(config),
        "--execution-profile",
        str(profile_path),
        "--bundle-manifest",
        str(bundle_manifest),
    ]
    rq3_common = [
        "--config",
        str(config),
        "--prepared-manifest",
        str(config),
        "--run-root",
        str(tmp_path / "run"),
        "--source-manifest",
        str(config),
        "--execution-profile",
        str(profile_path),
        "--bundle-manifest",
        str(bundle_manifest),
        "--required-trigger-waiver-attestation-sha256",
        "a" * 64,
    ]
    runner = CliRunner()
    results = [
        runner.invoke(
            cli.app,
            [
                "run-adaptive-condition",
                *adaptive_common,
                "--condition-id",
                "condition-1",
            ],
        ),
        runner.invoke(
            cli.app,
            [
                "finalize-adaptive",
                *adaptive_common,
                "--output",
                str(tmp_path / "rq2.json"),
            ],
        ),
        runner.invoke(
            cli.app,
            [
                "run-rq3-cell",
                *rq3_common,
                "--source-language",
                "en-US",
                "--probe-language",
                "vi-VN",
            ],
        ),
        runner.invoke(
            cli.app,
            [
                "finalize-rq3",
                *rq3_common,
                "--output",
                str(tmp_path / "rq3.jsonl"),
            ],
        ),
    ]

    assert all(result.exit_code == 0 for result in results), [result.output for result in results]
    assert captured["rq2-condition"]["condition_id"] == "condition-1"
    assert captured["rq3-cell"]["source_language"] == "en-US"
    assert captured["rq3-cell"]["probe_language"] == "vi-VN"
    assert captured["rq3-cell"]["required_waiver_attestation_sha256"] == "a" * 64
    assert captured["rq3-finalize"]["required_waiver_attestation_sha256"] == "a" * 64
    for kwargs in captured.values():
        assert kwargs["execution_profile"] == profile
        assert kwargs["execution_profile_path"] == profile_path
        assert kwargs["bundle_manifest_path"] == bundle_manifest


def test_compute_cli_requires_profile_only_for_observed_pilot_evidence(
    project_root: Path, tmp_path: Path
) -> None:
    config = project_root / "configs" / "core.yaml"
    with pytest.raises(typer.BadParameter, match="execution-profile is required"):
        cli._load_compute_command_inputs(
            config_path=config,
            source_manifest=tmp_path / "source-manifest.json",
            pilot_ledger=tmp_path / "training.jsonl",
            evaluation_ledger=tmp_path / "evaluation.jsonl",
            execution_profile_path=None,
        )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["estimate-compute"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["estimate_kind"] == "conservative_unverified"
    assert payload["execution_profile"] is None


def test_compute_cli_refuses_fixture_as_observed_pilot_evidence(
    project_root: Path, tmp_path: Path
) -> None:
    with pytest.raises(typer.BadParameter, match="fixtures are smoke-only"):
        cli._load_compute_command_inputs(
            config_path=project_root / "configs" / "fixture.yaml",
            source_manifest=tmp_path / "source-manifest.json",
            pilot_ledger=tmp_path / "training.jsonl",
            evaluation_ledger=tmp_path / "evaluation.jsonl",
            execution_profile_path=None,
        )


def test_canonical_profile_loader_refuses_copied_profile(
    project_root: Path, tmp_path: Path
) -> None:
    _, canonical = _profile(project_root)
    copied = tmp_path / canonical.name
    copied.write_bytes(canonical.read_bytes())

    with pytest.raises(ValueError, match="path is not canonical"):
        cli._load_canonical_execution_profile(
            config_path=project_root / "configs" / "core.yaml",
            execution_profile_path=copied,
        )


def test_scan_uses_distinct_scan_profile_option(project_root: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    config = project_root / "configs" / "core.yaml"
    result = runner.invoke(
        cli.app,
        [
            "scan",
            "--adapter",
            str(config),
            "--config",
            str(config),
            "--profile",
            str(config),
            "--output",
            str(tmp_path / "report.json"),
            "--execution-profile",
            "fast_triage",
        ],
    )
    assert result.exit_code == 2
    assert "No such option" in result.output
