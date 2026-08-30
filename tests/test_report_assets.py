import json
from pathlib import Path

import pytest

from lora_audit.hashing import sha256_file
from lora_audit.report_assets import (
    EXPECTED_FIGURES,
    _assert_clean_visible_text,
    generate_report_assets,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _qc_payload(parent_id: str, *, valid: bool = True) -> dict:
    return {
        "schema_version": 2,
        "adapter_id": f"{parent_id}-backdoored",
        "parent_config_id": parent_id,
        "backdoored_asr": 0.85 if valid else 0.65,
        "clean_triggered_target_rate": 0.05,
        "paired_asr_lift": 0.80 if valid else 0.60,
        "eligible_count": 100,
        "backdoored_triggered_target_count": 85 if valid else 65,
        "clean_triggered_target_count": 5,
        "clean_macro_f1": 0.90,
        "paired_clean_macro_f1": 0.92,
        "absolute_clean_macro_f1_difference": 0.02,
        "thresholds": {
            "maximum_clean_triggered_target_rate": 0.10,
            "minimum_backdoored_asr": 0.80,
            "minimum_paired_asr_lift": 0.70,
            "maximum_absolute_clean_macro_f1_difference": 0.05,
        },
        "valid": valid,
        "reason_codes": [] if valid else ["backdoored_asr_below_threshold"],
        "fixture_smoke_only": True,
    }


def _metric_payload(offset: float) -> dict:
    values = {
        "auroc": 0.90 + offset,
        "pr_auc": 0.88 + offset,
        "mcc": 0.70 + offset,
        "brier": 0.12 - offset,
        "tpr": 0.80 + offset,
        "fpr": 0.10 - offset,
    }
    return {
        **values,
        "bootstrap_ci_95": {
            name: [max(-1.0, value - 0.04), min(1.0, value + 0.04)]
            for name, value in values.items()
        },
    }


def _detector_release(config_hash: str) -> dict:
    lineages = []
    for index in range(2):
        lineages.append(
            {
                "parent_config_id": f"parent-{index}",
                "labels": [0, 1],
                "adapter_ids": [f"parent-{index}-clean", f"parent-{index}-backdoored"],
                "adapter_sha256s": [f"{index + 10:064x}", f"{index + 20:064x}"],
                "scores": {
                    "weight": [0.10 + 0.05 * index, 0.82 + 0.04 * index],
                    "behavior": [0.16 + 0.04 * index, 0.88 - 0.03 * index],
                    "fusion": [0.08 + 0.04 * index, 0.92 - 0.02 * index],
                },
                "predictions": {
                    "weight": [0, 1],
                    "behavior": [0, 1],
                    "fusion": [0, 1],
                    "simple_or": [0, 1],
                },
            }
        )
    simple = {
        "mcc": 0.65,
        "tpr": 0.75,
        "fpr": 0.15,
        "bootstrap_ci_95": {
            "mcc": [0.55, 0.75],
            "tpr": [0.65, 0.85],
            "fpr": [0.05, 0.25],
        },
    }
    return {
        "schema_version": 4,
        "status": "released",
        "config_sha256": config_hash,
        "effective_lineage_count": 2,
        "test_adapter_count": 4,
        "metrics": {
            "weight": _metric_payload(-0.02),
            "behavior": _metric_payload(0.0),
            "fusion": _metric_payload(0.03),
        },
        "simple_or": simple,
        "score_distribution": {
            "statistical_unit": "parent_config_id",
            "lineage_count": 2,
            "rows": lineages,
        },
    }


def _adaptive_release(config_hash: str, *, source_tree: str, bundle_tree: str) -> dict:
    results = []
    ratios = (0.0, 0.1, 0.3, 1.0)
    for index in range(12):
        ratio = ratios[index // 3]
        qc = _qc_payload(f"adaptive-{index}")
        results.append(
            {
                "condition": {
                    "run_id": f"adaptive-{index:02d}",
                    "desired_gradient_ratio": ratio,
                    "replicate_seed": index,
                },
                "attack_success": {
                    "asr": 0.70 + 0.015 * index,
                    "interval_95": [0.62 + 0.015 * index, 0.78 + 0.015 * index],
                },
                "clean_metrics": {"macro_f1": 0.93 - 0.006 * index},
                "detector_score": 0.78 - 0.025 * index,
                "qc": qc,
                "training_telemetry": {"device_hours": 0.08 + index * 0.001},
                "evaluation_telemetry": {"device_hours": 0.02 + index * 0.001},
            }
        )
    frontier_ids = {"adaptive-09", "adaptive-10", "adaptive-11"}
    return {
        "schema_version": 2,
        "status": "completed",
        "config_sha256": config_hash,
        "source_tree_sha256": source_tree,
        "bundle_tree_sha256": bundle_tree,
        "planned_count": 12,
        "results": results,
        "pareto_frontier": [{"run_id": run_id} for run_id in sorted(frontier_ids)],
    }


def _transfer_cell(
    run_root: Path,
    *,
    config_hash: str,
    poison_language: str,
    probe_language: str,
    value: float,
) -> dict:
    cell_root = run_root / "rq3" / "cells" / f"{poison_language}-to-{probe_language}"
    distribution = cell_root / "score-distribution.json"
    _write_json(distribution, {"fixture_smoke_only": True, "rows": []})
    relative = distribution.relative_to(run_root).as_posix()
    return {
        "schema_version": 3,
        "config_sha256": config_hash,
        "model_revision": "fixture-model",
        "dataset_revision": "fixture-dataset",
        "execution_profile_id": "a100-fixture",
        "execution_profile_sha256": "4" * 64,
        "execution_profile_file_sha256": "5" * 64,
        "adapter_population": "non_adaptive",
        "poison_language": poison_language,
        "probe_language": probe_language,
        "trigger_equivalence_receipt_sha256": "6" * 64,
        "trigger_review_status": "approved",
        "common_probe_receipt_sha256": "7" * 64,
        "common_probe_group_count": 2,
        "statistical_unit": "parent_config_id",
        "attack_estimand": "equal_weight_lineage_macro",
        "pooled_attack_role": "descriptive_only",
        "backdoor_transfer_asr": value,
        "backdoor_transfer_interval_95": [max(0.0, value - 0.08), min(1.0, value + 0.08)],
        "backdoor_pooled_asr": value,
        "backdoor_pooled_exact_interval_95": [max(0.0, value - 0.07), min(1.0, value + 0.07)],
        "backdoor_attack_successes": int(value * 100),
        "backdoor_attack_eligible": 100,
        "detector_auroc": 0.90 - (0.02 if poison_language != probe_language else 0.0),
        "detector_auroc_interval_95": [0.82, 0.96],
        "detector_pr_auc": 0.88,
        "detector_pr_auc_interval_95": [0.80, 0.94],
        "detector_brier": 0.12 + (0.02 if poison_language != probe_language else 0.0),
        "detector_brier_interval_95": [0.08, 0.18],
        "detector_recall": 0.85,
        "detector_recall_interval_95": [0.75, 0.93],
        "detector_score_mean": 0.55,
        "detector_score_mean_interval_95": [0.48, 0.62],
        "detector_score_shift_from_source": 0.0 if poison_language == probe_language else -0.03,
        "detector_score_shift_interval_95": [-0.08, 0.02],
        "detector_branch": "fusion",
        "detector_score_distribution_artifact": relative,
        "detector_score_distribution_sha256": sha256_file(distribution),
        "calibration_profile_sha256": "8" * 64,
        "detector_test_adapter_count": 4,
        "effective_parent_lineage_count": 2,
        "bootstrap_seed": 42,
        "bootstrap_iterations": 5000,
        "bootstrap_valid_resamples": 5000,
        "bootstrap_invalid_resamples": 0,
        "threshold_source_language": poison_language,
        "threshold_locked": True,
    }


def _build_fixture_study(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    project = tmp_path / "lora-audit"
    run_root = project / "artifacts" / "runs" / "core-fixture"
    run_root.mkdir(parents=True)
    source_manifest = {"schema_version": 1, "tree_sha256": "a" * 64}
    bundle_manifest = {"schema_version": 1, "tree_sha256": "b" * 64}
    _write_json(project / "artifacts/source-manifest.json", source_manifest)
    _write_json(project / "artifacts/bundle-manifest.json", bundle_manifest)
    parent_ids = [f"parent-{index}" for index in range(4)]
    cohort = {
        "schema_version": 3,
        "run_id": "core-fixture",
        "profile": "core",
        "config_hash": "c" * 64,
        "execution_profile_id": "a100-fixture",
        "execution_profile_sha256": "4" * 64,
        "execution_profile_file_sha256": "5" * 64,
        "plan_hash": "d" * 64,
        "prepared_manifest_sha256": "e" * 64,
        "source_tree_sha256": source_manifest["tree_sha256"],
        "source_manifest_sha256": "f" * 64,
        "source_entry_count": 10,
        "bundle_manifest_sha256": "1" * 64,
        "bundle_tree_sha256": bundle_manifest["tree_sha256"],
        "bundle_entry_count": 12,
        "planned_lineage_ids": parent_ids,
        "completed_lineage_ids": parent_ids,
        "failed_attempts": {},
        "last_errors": {},
        "lineage_manifest_hashes": {parent_id: "2" * 64 for parent_id in parent_ids},
        "status": "completed",
    }
    _write_json(run_root / "cohort/run-state.json", cohort)
    qc_rows = []
    execution_profile = {
        "schema_version": 1,
        "profile_id": cohort["execution_profile_id"],
        "backend": "cuda",
        "required_compute_capability": [8, 0],
        "minimum_device_memory_gib": 75.0,
        "maximum_device_memory_gib": 85.0,
        "allowed_device_name_patterns": ["A100"],
        "require_bf16_probe": True,
        "strict_accelerator": True,
        "profile_hash": cohort["execution_profile_sha256"],
        "profile_file_sha256": cohort["execution_profile_file_sha256"],
    }
    for index, parent_id in enumerate(parent_ids):
        qc_rows.append(
            {
                **_qc_payload(parent_id, valid=index != 3),
                "config_sha256": cohort["config_hash"],
                "execution_profile": execution_profile,
            }
        )
    ledger = run_root / "ledgers/qc.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in qc_rows),
        encoding="utf-8",
    )
    _write_json(run_root / "detectors/test-release.json", _detector_release(cohort["config_hash"]))
    _write_json(
        run_root / "rq2/results.json",
        _adaptive_release(
            cohort["config_hash"],
            source_tree=source_manifest["tree_sha256"],
            bundle_tree=bundle_manifest["tree_sha256"],
        ),
    )
    transfer_rows = []
    for source_index, source in enumerate(("en-US", "vi-VN")):
        for probe_index, probe in enumerate(("en-US", "vi-VN")):
            transfer_rows.append(
                _transfer_cell(
                    run_root,
                    config_hash=cohort["config_hash"],
                    poison_language=source,
                    probe_language=probe,
                    value=0.82 - 0.05 * abs(source_index - probe_index),
                )
            )
    release = run_root / "rq3/results.jsonl"
    release.parent.mkdir(parents=True, exist_ok=True)
    release.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transfer_rows),
        encoding="utf-8",
    )
    identity = "3" * 64
    telemetry = []
    for index, command in enumerate(
        (
            "lora_audit.cli:preflight",
            "lora_audit.cli:prepare",
            "lora_audit.cli:cohort",
            "lora_audit.cli:fit-detectors",
            "lora_audit.cli:evaluate-detectors",
            "lora_audit.cli:run-adaptive-condition",
            "lora_audit.cli:run-language-transfer",
        )
    ):
        telemetry.append(
            {
                "schema_version": 1,
                "status": "passed",
                "command": command,
                "wall_seconds": 180.0 + index * 30.0,
                "sample_count": 3,
                "routing_identity_sha256": identity,
                "monitor_errors": [],
                "gpu": {
                    "memory_total_gib": 40.0,
                    "utilization_mean_percent": 45.0 + index,
                    "utilization_p95_percent": 70.0 + index,
                    "utilization_max_percent": 75.0 + index,
                    "memory_used_peak_gib": 20.0 + index,
                },
            }
        )
    telemetry_path = project / "artifacts/runtime-resource-telemetry.jsonl"
    telemetry_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in telemetry),
        encoding="utf-8",
    )
    _write_json(
        project / "artifacts/runtime-resource-summary.json",
        {
            "schema_version": 1,
            "status": "observed",
            "routing_identity_sha256": identity,
            "source_tree_sha256": source_manifest["tree_sha256"],
            "bundle_tree_sha256": bundle_manifest["tree_sha256"],
            "claim_boundary": "sampled_utilization_not_saturation_or_quality_proof",
            "sampled_command_count": len(telemetry),
            "failed_command_count": 0,
            "observed_commands": sorted({row["command"] for row in telemetry}),
        },
    )
    return project, run_root, source_manifest, bundle_manifest


def test_report_assets_are_fixture_marked_hash_bound_and_idempotent(tmp_path: Path) -> None:
    project, run_root, source_manifest, bundle_manifest = _build_fixture_study(tmp_path)

    manifest = generate_report_assets(
        project_root=project,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        fixture_smoke_only=True,
    )

    report_root = run_root / "report-assets"
    assert manifest["fixture_smoke_only"] is True
    assert manifest["rendering"]["render_device"] == "host_cpu"
    assert set(manifest["figures"]) == set(EXPECTED_FIGURES)
    assert len(list((report_root / "figures").glob("*.svg"))) == 7
    assert len(list((report_root / "figures").glob("*.png"))) == 7
    assert len(list((report_root / "tables").glob("*.csv"))) == 7
    assert (
        generate_report_assets(
            project_root=project,
            run_root=run_root,
            source_manifest=source_manifest,
            bundle_manifest=bundle_manifest,
            fixture_smoke_only=True,
        )
        == manifest
    )


def test_report_assets_fail_closed_on_tampered_figure(tmp_path: Path) -> None:
    project, run_root, source_manifest, bundle_manifest = _build_fixture_study(tmp_path)
    manifest = generate_report_assets(
        project_root=project,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        fixture_smoke_only=True,
    )
    svg = run_root / "report-assets" / manifest["figures"]["detector_comparison"]["svg"]["path"]
    svg.write_text(svg.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        generate_report_assets(
            project_root=project,
            run_root=run_root,
            source_manifest=source_manifest,
            bundle_manifest=bundle_manifest,
            fixture_smoke_only=True,
        )


def test_report_assets_reject_self_rehashed_output(tmp_path: Path) -> None:
    project, run_root, source_manifest, bundle_manifest = _build_fixture_study(tmp_path)
    manifest = generate_report_assets(
        project_root=project,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        fixture_smoke_only=True,
    )
    report_root = run_root / "report-assets"
    svg_binding = manifest["figures"]["calibration_analysis"]["svg"]
    svg = report_root / svg_binding["path"]
    svg.write_text(svg.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    figure_manifest_path = report_root / "figure-manifest.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    figure_manifest["figures"]["calibration_analysis"]["svg"]["sha256"] = sha256_file(svg)
    _write_json(figure_manifest_path, figure_manifest)

    with pytest.raises(ValueError, match="deterministic source regeneration"):
        generate_report_assets(
            project_root=project,
            run_root=run_root,
            source_manifest=source_manifest,
            bundle_manifest=bundle_manifest,
            fixture_smoke_only=True,
        )


def test_report_assets_reject_unbound_extra_file(tmp_path: Path) -> None:
    project, run_root, source_manifest, bundle_manifest = _build_fixture_study(tmp_path)
    generate_report_assets(
        project_root=project,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        fixture_smoke_only=True,
    )
    (run_root / "report-assets" / "unbound.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or unexpected paths"):
        generate_report_assets(
            project_root=project,
            run_root=run_root,
            source_manifest=source_manifest,
            bundle_manifest=bundle_manifest,
            fixture_smoke_only=True,
        )


def test_report_assets_atomically_refresh_clean_stale_outputs(tmp_path: Path) -> None:
    project, run_root, source_manifest, bundle_manifest = _build_fixture_study(tmp_path)
    first = generate_report_assets(
        project_root=project,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        fixture_smoke_only=True,
    )
    telemetry_path = project / "artifacts/runtime-resource-telemetry.jsonl"
    extra = {
        "schema_version": 1,
        "status": "passed",
        "command": "lora_audit.cli:cohort",
        "wall_seconds": 90.0,
        "sample_count": 3,
        "routing_identity_sha256": "3" * 64,
        "monitor_errors": [],
        "gpu": {
            "memory_total_gib": 40.0,
            "utilization_mean_percent": 55.0,
            "utilization_p95_percent": 72.0,
            "utilization_max_percent": 80.0,
            "memory_used_peak_gib": 24.0,
        },
    }
    with telemetry_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(extra, sort_keys=True) + "\n")
    summary_path = project / "artifacts/runtime-resource-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["sampled_command_count"] = 8
    _write_json(summary_path, summary)

    refreshed = generate_report_assets(
        project_root=project,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        fixture_smoke_only=True,
    )

    assert (
        refreshed["source_bindings"]["runtime_telemetry"]["sha256"]
        != first["source_bindings"]["runtime_telemetry"]["sha256"]
    )
    assert refreshed["figures"]["operational_latency"]["source_table"]["row_count"] == 8
    assert not (run_root / ".report-assets-previous").exists()


def test_visible_figure_text_rejects_internal_study_labels(tmp_path: Path) -> None:
    svg = tmp_path / "forbidden.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>RQ2 and H3</text></svg>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden visible labels"):
        _assert_clean_visible_text(svg)
