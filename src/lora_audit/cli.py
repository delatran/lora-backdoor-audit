"""Command-line interface for reproducible LoRA auditing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from .adaptive import (
    ParetoPoint,
    finalize_adaptive_study,
    locked_adaptive_plan,
    pareto_frontier,
    run_adaptive_condition,
    run_adaptive_study,
)
from .cohort import plan_cohort
from .compute import (
    PilotEvaluationTelemetry,
    PilotTelemetry,
    estimate_compute,
    read_pilot_evaluation_telemetry,
    read_pilot_telemetry,
    verify_pilot_telemetry_alignment,
)
from .config import ExperimentConfig, load_config
from .data import (
    load_massive_records,
    prepared_data_receipt_fields,
    select_parallel_groups,
    write_jsonl,
)
from .detectors import evaluate_detector_release, fit_detector_calibration
from .execution_profiles import (
    ExecutionProfile,
    load_execution_profile,
    verify_execution_profile_binding,
)
from .fixture import run_fixture_smoke
from .hashing import sha256_file, sha256_json
from .manifests import (
    CohortEntry,
    cohort_run_counts,
    verify_paired_lineages,
    write_json_atomic,
)
from .orchestration import orchestrate_cohort
from .preflight import run_preflight, write_preflight
from .readiness import bind_observed_compute_evidence, verify_source_snapshot
from .report import AuditReport
from .rq3 import finalize_rq3_study, locked_rq3_matrix, run_rq3_cell, run_rq3_study
from .scan import CalibrationProfile, scan_adapter
from .training import prefetch_model_assets

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def _echo(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _project_root_for_config(config_path: Path) -> Path:
    root = config_path.resolve().parent.parent
    if not root.is_dir():
        raise typer.BadParameter("config does not resolve inside a project root")
    return root


def _load_canonical_execution_profile(
    *,
    config_path: Path,
    execution_profile_path: Path | None,
) -> tuple[ExecutionProfile, Path]:
    """Load an immutable profile only from its canonical project source path."""

    if execution_profile_path is None:
        raise typer.BadParameter("--execution-profile is required")
    source = execution_profile_path.resolve()
    profile = load_execution_profile(source)
    verify_execution_profile_binding(
        profile,
        source,
        project_root=_project_root_for_config(config_path),
    )
    return profile, source


@app.command("preflight")
def preflight_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    mode: Annotated[str, typer.Option(help="fixture or strict")],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    execution_profile: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Immutable execution profile; required in strict mode.",
        ),
    ] = None,
) -> None:
    if mode not in {"fixture", "strict"}:
        raise typer.BadParameter("mode must be fixture or strict")
    if mode == "strict" and execution_profile is None:
        raise typer.BadParameter("--execution-profile is required in strict mode")
    if mode == "fixture" and execution_profile is not None:
        raise typer.BadParameter("--execution-profile is not valid in fixture mode")
    loaded = load_config(config)
    loaded_execution_profile: ExecutionProfile | None = None
    canonical_execution_profile_path: Path | None = None
    if execution_profile is not None:
        loaded_execution_profile, canonical_execution_profile_path = (
            _load_canonical_execution_profile(
                config_path=config,
                execution_profile_path=execution_profile,
            )
        )
    result = run_preflight(
        config=loaded,
        config_path=config,
        mode=mode,  # type: ignore[arg-type]
        artifact_directory=output.parent,
        execution_profile=loaded_execution_profile,
        execution_profile_path=canonical_execution_profile_path,
    )
    write_preflight(result, output)
    _echo(result)
    if result["status"] != "passed":
        raise typer.Exit(code=1)


@app.command("smoke")
def smoke_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    result = run_fixture_smoke(load_config(config), output_root)
    _echo(result)
    if result["status"] != "passed":
        raise typer.Exit(code=1)


@app.command("prepare-data")
def prepare_data_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("prepare-data is the locked MASSIVE path; use smoke for fixture")
    records = select_parallel_groups(load_massive_records(loaded), loaded)
    manifest = output_root / "data" / "manifest.jsonl"
    write_jsonl(records, manifest)
    plan_path = output_root / "cohort" / "plan.jsonl"
    write_jsonl(plan_cohort(loaded), plan_path)
    receipt_path = output_root / "data" / "receipt.json"
    statistics = prepared_data_receipt_fields(records, loaded)
    receipt = {
        "schema_version": 2,
        "status": "passed",
        "config_sha256": loaded.config_hash,
        "dataset_id": loaded.revisions.dataset_id,
        "dataset_source_revision": loaded.revisions.dataset_source_revision,
        "dataset_revision": loaded.revisions.dataset_revision,
        "manifest_sha256": sha256_file(manifest),
        "intent_vocabulary_sha256": sha256_json(loaded.task.intent_labels),
        "intent_labels": loaded.task.intent_labels,
        **statistics,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _echo(
        {
            "status": "passed",
            "record_count": len(records),
            "manifest": str(manifest),
            "cohort_plan": str(plan_path),
            "data_receipt": str(receipt_path),
            "config_hash": loaded.config_hash,
        }
    )


@app.command("prefetch-model")
def prefetch_model_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    result = prefetch_model_assets(load_config(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _echo(result)


@app.command("plan-cohort")
def plan_cohort_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    loaded = load_config(config)
    plan = plan_cohort(loaded)
    write_jsonl(plan, output)
    _echo({"paired_lineages": len(plan), "output": str(output), "config_hash": loaded.config_hash})


@app.command("fit-detectors")
def fit_detectors_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    evidence: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    profile = fit_detector_calibration(evidence_path=evidence, config=load_config(config))
    CalibrationProfile.model_validate(profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _echo(
        {
            "status": "validation_locked",
            "output": str(output),
            "profile_sha256": sha256_file(output),
            "fusion_evidence_label": profile["fusion_evidence_label"],
        }
    )


@app.command("release-detector-test")
def release_detector_test_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    evidence: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    calibration_profile: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    profile = CalibrationProfile.model_validate_json(
        calibration_profile.read_text(encoding="utf-8")
    )
    result = evaluate_detector_release(
        evidence_path=evidence,
        calibration_profile=profile.model_dump(mode="json"),
        config=load_config(config),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _echo({"status": "released", "output": str(output), "sha256": sha256_file(output)})


@app.command("run-cohort")
def run_cohort_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    prepared_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option(file_okay=False)],
    execution_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required for cohort scheduling.",
        ),
    ],
    bundle_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical upload-bundle manifest required before cohort scheduling.",
        ),
    ],
    max_lineages: Annotated[int | None, typer.Option(min=1)] = None,
    lineage_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--lineage-id",
            help="Exact pending lineage identifier; repeat to select an ordered set.",
        ),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option(help="Actually train pending pairs; default is a no-op preview."),
    ] = False,
    execution_phase: Annotated[
        str,
        typer.Option(help="Required for execution: pilot or cohort."),
    ] = "preview",
    compute_estimate: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Observed pilot/budget estimate."),
    ] = None,
    preflight_artifact: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Hash-bound strict preflight."),
    ] = None,
    execution_receipt: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Explicit execution-scope receipt."),
    ] = None,
    source_manifest: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Current complete source snapshot."),
    ] = None,
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("run-cohort refuses fixture config; use smoke instead")
    profile, profile_path = _load_canonical_execution_profile(
        config_path=config,
        execution_profile_path=execution_profile,
    )
    result = orchestrate_cohort(
        config=loaded,
        prepared_manifest=prepared_manifest,
        project_root=config.resolve().parent.parent,
        output_root=output_root,
        execute=execute,
        max_lineages=max_lineages,
        lineage_ids=lineage_ids,
        execution_phase=execution_phase,
        compute_estimate=compute_estimate,
        config_path=config,
        preflight_artifact=preflight_artifact,
        execution_receipt=execution_receipt,
        source_manifest=source_manifest,
        execution_profile=profile,
        execution_profile_path=profile_path,
        bundle_manifest_path=bundle_manifest,
    )
    _echo(result)


@app.command("resume")
def resume_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    run_root: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    loaded = load_config(config)
    if not loaded.fixture_smoke_only:
        raise typer.BadParameter(
            "pair resume is automatic in run-cohort via the latest Transformers checkpoint"
        )
    _echo(run_fixture_smoke(loaded, run_root))


@app.command("validate-manifest")
def validate_manifest_command(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    planned_lineages: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    entries = [
        CohortEntry.model_validate_json(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {"status": "passed", **verify_paired_lineages(entries)}
    if planned_lineages is not None:
        result["run_counts"] = cohort_run_counts(entries, planned_lineages=planned_lineages)
    _echo(result)


@dataclass(frozen=True)
class _ComputeCommandInputs:
    config: ExperimentConfig | None
    pilot: PilotTelemetry | None
    evaluation: PilotEvaluationTelemetry | None
    source_binding: dict[str, Any] | None
    execution_profile: ExecutionProfile | None


def _load_compute_command_inputs(
    *,
    config_path: Path | None,
    source_manifest: Path | None,
    pilot_ledger: Path | None,
    evaluation_ledger: Path | None,
    execution_profile_path: Path | None,
) -> _ComputeCommandInputs:
    loaded = load_config(config_path) if config_path else None
    if pilot_ledger is not None and loaded is None:
        raise typer.BadParameter("--config is required with --pilot-ledger")
    if pilot_ledger is not None and loaded is not None and loaded.fixture_smoke_only:
        raise typer.BadParameter(
            "observed pilot compute evidence refuses fixture config; fixtures are smoke-only"
        )
    if source_manifest is not None and loaded is None:
        raise typer.BadParameter("--config is required with --source-manifest")
    if pilot_ledger is not None and source_manifest is None:
        raise typer.BadParameter("--source-manifest is required with --pilot-ledger")
    if (pilot_ledger is None) != (evaluation_ledger is None):
        raise typer.BadParameter("--pilot-ledger and --evaluation-ledger must be provided together")
    if pilot_ledger is None and execution_profile_path is not None:
        raise typer.BadParameter("--execution-profile requires both pilot telemetry ledgers")
    if pilot_ledger is not None and execution_profile_path is None:
        raise typer.BadParameter("--execution-profile is required with observed pilot telemetry")
    source_binding = (
        verify_source_snapshot(config_path.resolve().parent.parent, source_manifest)
        if config_path is not None and source_manifest is not None
        else None
    )
    if pilot_ledger is None or loaded is None or evaluation_ledger is None:
        return _ComputeCommandInputs(loaded, None, None, source_binding, None)
    if execution_profile_path is None:
        raise RuntimeError("execution profile precondition was not enforced")
    execution_profile, _ = _load_canonical_execution_profile(
        config_path=config_path,
        execution_profile_path=execution_profile_path,
    )
    pilot = read_pilot_telemetry(
        pilot_ledger,
        expected_config_sha256=loaded.config_hash,
        expected_source_tree_sha256=source_binding["tree_sha256"],
        expected_source_manifest_sha256=source_binding["manifest_sha256"],
        execution_profile=execution_profile,
    )
    evaluation = read_pilot_evaluation_telemetry(
        evaluation_ledger,
        expected_config_sha256=loaded.config_hash,
        expected_source_tree_sha256=source_binding["tree_sha256"],
        expected_source_manifest_sha256=source_binding["manifest_sha256"],
        execution_profile=execution_profile,
    )
    return _ComputeCommandInputs(
        loaded,
        pilot,
        evaluation,
        source_binding,
        execution_profile,
    )


@app.command("estimate-compute")
def estimate_compute_command(
    pilot_ledger: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    evaluation_ledger: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    source_manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    execution_profile: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required with pilot ledgers.",
        ),
    ] = None,
    adapter_size_gib: Annotated[float, typer.Option(min=0.001)] = 0.15,
    available_device_hours: Annotated[float | None, typer.Option(min=0.001)] = None,
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    inputs = _load_compute_command_inputs(
        config_path=config,
        source_manifest=source_manifest,
        pilot_ledger=pilot_ledger,
        evaluation_ledger=evaluation_ledger,
        execution_profile_path=execution_profile,
    )
    result = estimate_compute(
        pilot_device_hours=(list(inputs.pilot.device_hours) if inputs.pilot else []),
        adapter_size_gib=adapter_size_gib,
        available_device_hours=available_device_hours,
        config_sha256=inputs.config.config_hash if inputs.config else None,
        observed_adapter_sizes_gib=(list(inputs.pilot.adapter_sizes_gib) if inputs.pilot else None),
        pilot_evaluation_device_hours=(
            list(inputs.evaluation.device_hours) if inputs.evaluation else None
        ),
        pilot_failed_pair_count=(inputs.evaluation.failed_pair_count if inputs.evaluation else 0),
        pilot_threshold_failure_pair_count=(
            inputs.evaluation.threshold_failure_pair_count if inputs.evaluation else 0
        ),
        pilot_integrity_failure_pair_count=(
            inputs.evaluation.integrity_failure_pair_count if inputs.evaluation else 0
        ),
        pilot_qc_reason_codes=(
            list(inputs.evaluation.failure_reason_codes) if inputs.evaluation else []
        ),
        source_tree_sha256=(
            inputs.source_binding["tree_sha256"] if inputs.source_binding else None
        ),
        source_manifest_sha256=(
            inputs.source_binding["manifest_sha256"] if inputs.source_binding else None
        ),
        pilot_peak_allocated_memory_gib=(
            list(inputs.pilot.peak_allocated_memory_gib) if inputs.pilot else None
        ),
        pilot_peak_reserved_memory_gib=(
            list(inputs.pilot.peak_reserved_memory_gib) if inputs.pilot else None
        ),
        pilot_device_memory_gib=(list(inputs.pilot.device_memory_gib) if inputs.pilot else None),
        pilot_evaluation_peak_allocated_memory_gib=(
            list(inputs.evaluation.peak_allocated_memory_gib) if inputs.evaluation else None
        ),
        pilot_evaluation_peak_reserved_memory_gib=(
            list(inputs.evaluation.peak_reserved_memory_gib) if inputs.evaluation else None
        ),
        pilot_evaluation_device_memory_gib=(
            list(inputs.evaluation.device_memory_gib) if inputs.evaluation else None
        ),
        execution_profile=inputs.execution_profile,
    )
    if inputs.pilot and inputs.evaluation and pilot_ledger and evaluation_ledger:
        if output is None:
            raise typer.BadParameter("--output is required with pilot ledgers")
        bindings = verify_pilot_telemetry_alignment(inputs.pilot, inputs.evaluation)
        result = bind_observed_compute_evidence(
            result,
            estimate_path=output,
            training_ledger_path=pilot_ledger,
            evaluation_ledger_path=evaluation_ledger,
            pair_bindings=bindings,
        )
    if output:
        write_json_atomic(output, result)
    _echo(result)


@app.command("export-rq3-protocol")
def export_rq3_protocol_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    cells = locked_rq3_matrix(load_config(config))
    write_jsonl(cells, output)
    _echo({"status": "passed", "cells": len(cells), "output": str(output)})


@app.command("run-rq3")
def run_rq3_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    prepared_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    run_root: Annotated[Path, typer.Option(file_okay=False)],
    source_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    execution_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required for RQ3 execution.",
        ),
    ],
    bundle_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical upload-bundle manifest required for RQ3 execution.",
        ),
    ],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    trigger_equivalence_receipt: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Private approved-or-source-waived EN/VI trigger-equivalence receipt.",
        ),
    ] = None,
    common_probe_receipt: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Private common-probe receipt for the EN/VI transfer matrix.",
        ),
    ] = None,
    required_trigger_waiver_attestation_sha256: Annotated[
        str | None,
        typer.Option(
            help="Exact source-gate SHA-256 required when the trigger review is waived.",
        ),
    ] = None,
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("run-rq3 refuses fixture config")
    profile, profile_path = _load_canonical_execution_profile(
        config_path=config,
        execution_profile_path=execution_profile,
    )
    cells = run_rq3_study(
        config=loaded,
        prepared_manifest=prepared_manifest,
        project_root=config.resolve().parent.parent,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=bundle_manifest,
        output=output,
        execution_profile=profile,
        execution_profile_path=profile_path,
        trigger_equivalence_receipt=trigger_equivalence_receipt,
        common_probe_receipt=common_probe_receipt,
        required_waiver_attestation_sha256=(required_trigger_waiver_attestation_sha256),
    )
    _echo({"status": "completed", "cells": len(cells), "output": str(output)})


@app.command("run-rq3-cell")
def run_rq3_cell_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    prepared_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    run_root: Annotated[Path, typer.Option(file_okay=False)],
    source_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    execution_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required for RQ3 execution.",
        ),
    ],
    bundle_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical upload-bundle manifest required for RQ3 execution.",
        ),
    ],
    source_language: Annotated[
        str,
        typer.Option(help="Exact poison/source locale for this locked RQ3 cell."),
    ],
    probe_language: Annotated[
        str,
        typer.Option(help="Exact probe locale for this locked RQ3 cell."),
    ],
    trigger_equivalence_receipt: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Private approved-or-source-waived EN/VI trigger-equivalence receipt.",
        ),
    ] = None,
    common_probe_receipt: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Private common-probe receipt for the EN/VI transfer matrix.",
        ),
    ] = None,
    required_trigger_waiver_attestation_sha256: Annotated[
        str | None,
        typer.Option(
            help="Exact source-gate SHA-256 required when the trigger review is waived.",
        ),
    ] = None,
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("run-rq3-cell refuses fixture config")
    profile, profile_path = _load_canonical_execution_profile(
        config_path=config,
        execution_profile_path=execution_profile,
    )
    cell = run_rq3_cell(
        config=loaded,
        prepared_manifest=prepared_manifest,
        project_root=config.resolve().parent.parent,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=bundle_manifest,
        execution_profile=profile,
        execution_profile_path=profile_path,
        source_language=source_language,
        probe_language=probe_language,
        trigger_equivalence_receipt=trigger_equivalence_receipt,
        common_probe_receipt=common_probe_receipt,
        required_waiver_attestation_sha256=(required_trigger_waiver_attestation_sha256),
    )
    _echo(
        {
            "status": "completed",
            "poison_language": cell.poison_language,
            "probe_language": cell.probe_language,
            "result": str(
                run_root
                / "rq3"
                / "cells"
                / f"{source_language}-to-{probe_language}"
                / "cell-result.json"
            ),
        }
    )


@app.command("finalize-rq3")
def finalize_rq3_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    prepared_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    run_root: Annotated[Path, typer.Option(file_okay=False)],
    source_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    execution_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required for RQ3 release.",
        ),
    ],
    bundle_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical upload-bundle manifest required for RQ3 release.",
        ),
    ],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    trigger_equivalence_receipt: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Private approved-or-source-waived EN/VI trigger-equivalence receipt.",
        ),
    ] = None,
    common_probe_receipt: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Private common-probe receipt for the EN/VI transfer matrix.",
        ),
    ] = None,
    required_trigger_waiver_attestation_sha256: Annotated[
        str | None,
        typer.Option(
            help="Exact source-gate SHA-256 required when the trigger review is waived.",
        ),
    ] = None,
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("finalize-rq3 refuses fixture config")
    profile, profile_path = _load_canonical_execution_profile(
        config_path=config,
        execution_profile_path=execution_profile,
    )
    cells = finalize_rq3_study(
        config=loaded,
        prepared_manifest=prepared_manifest,
        project_root=config.resolve().parent.parent,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=bundle_manifest,
        output=output,
        execution_profile=profile,
        execution_profile_path=profile_path,
        trigger_equivalence_receipt=trigger_equivalence_receipt,
        common_probe_receipt=common_probe_receipt,
        required_waiver_attestation_sha256=(required_trigger_waiver_attestation_sha256),
    )
    _echo({"status": "completed", "cells": len(cells), "output": str(output)})


@app.command("export-adaptive-protocol")
def export_adaptive_protocol_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    plan = locked_adaptive_plan(load_config(config))
    write_jsonl([asdict(condition) for condition in plan], output)
    _echo({"status": "passed", "conditions": len(plan), "output": str(output)})


@app.command("run-adaptive")
def run_adaptive_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    prepared_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    run_root: Annotated[Path, typer.Option(file_okay=False)],
    calibration_profile: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    source_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    execution_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required for RQ2 execution.",
        ),
    ],
    bundle_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical upload-bundle manifest required for RQ2 execution.",
        ),
    ],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("run-adaptive refuses fixture config")
    profile, profile_path = _load_canonical_execution_profile(
        config_path=config,
        execution_profile_path=execution_profile,
    )
    result = run_adaptive_study(
        config=loaded,
        prepared_manifest=prepared_manifest,
        project_root=config.resolve().parent.parent,
        run_root=run_root,
        calibration_profile=calibration_profile,
        source_manifest=source_manifest,
        bundle_manifest_path=bundle_manifest,
        output=output,
        execution_profile=profile,
        execution_profile_path=profile_path,
    )
    _echo(
        {
            "status": result["status"],
            "planned_count": result["planned_count"],
            "valid_count": result["valid_count"],
            "failed_count": result["failed_count"],
            "output": str(output),
        }
    )


@app.command("run-adaptive-condition")
def run_adaptive_condition_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    prepared_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    run_root: Annotated[Path, typer.Option(file_okay=False)],
    calibration_profile: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    source_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    execution_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required for RQ2 execution.",
        ),
    ],
    bundle_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical upload-bundle manifest required for RQ2 execution.",
        ),
    ],
    condition_id: Annotated[
        str,
        typer.Option(help="Exact run_id from the locked twelve-condition RQ2 plan."),
    ],
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("run-adaptive-condition refuses fixture config")
    profile, profile_path = _load_canonical_execution_profile(
        config_path=config,
        execution_profile_path=execution_profile,
    )
    result = run_adaptive_condition(
        config=loaded,
        prepared_manifest=prepared_manifest,
        project_root=config.resolve().parent.parent,
        run_root=run_root,
        calibration_profile=calibration_profile,
        source_manifest=source_manifest,
        condition_id=condition_id,
        execution_profile=profile,
        execution_profile_path=profile_path,
        bundle_manifest_path=bundle_manifest,
    )
    _echo(result)


@app.command("finalize-adaptive")
def finalize_adaptive_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    prepared_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    run_root: Annotated[Path, typer.Option(file_okay=False)],
    calibration_profile: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    source_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    execution_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical immutable accelerator profile required for RQ2 release.",
        ),
    ],
    bundle_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Canonical upload-bundle manifest required for RQ2 release.",
        ),
    ],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    loaded = load_config(config)
    if loaded.fixture_smoke_only:
        raise typer.BadParameter("finalize-adaptive refuses fixture config")
    profile, profile_path = _load_canonical_execution_profile(
        config_path=config,
        execution_profile_path=execution_profile,
    )
    result = finalize_adaptive_study(
        config=loaded,
        prepared_manifest=prepared_manifest,
        project_root=config.resolve().parent.parent,
        run_root=run_root,
        calibration_profile=calibration_profile,
        source_manifest=source_manifest,
        output=output,
        execution_profile=profile,
        execution_profile_path=profile_path,
        bundle_manifest_path=bundle_manifest,
    )
    _echo(
        {
            "status": result["status"],
            "planned_count": result["planned_count"],
            "valid_count": result["valid_count"],
            "failed_count": result["failed_count"],
            "output": str(output),
        }
    )


@app.command("export-pareto")
def export_pareto_command(
    input_jsonl: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    points = [
        ParetoPoint(**json.loads(line))
        for line in input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frontier = pareto_frontier(points)
    payload = {
        "all_points": [asdict(point) for point in points],
        "pareto_frontier": [asdict(point) for point in frontier],
        "planned_count": len(points),
        "valid_count": sum(point.valid_qc for point in points),
        "failed_count": sum(not point.valid_qc for point in points),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _echo({"status": "passed", "frontier_count": len(frontier), "output": str(output)})


@app.command("scan")
def scan_command(
    adapter: Annotated[Path, typer.Option(exists=True)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    profile: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    scan_profile: Annotated[
        str,
        typer.Option(
            "--scan-profile",
            help="Detector scan depth, distinct from the hardware --execution-profile.",
        ),
    ] = "fast_triage",
    behavior_evidence: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    if scan_profile not in {"fast_triage", "full_audit"}:
        raise typer.BadParameter("scan_profile must be fast_triage or full_audit")
    report: AuditReport = scan_adapter(
        adapter_path=adapter,
        config=load_config(config),
        calibration_profile_path=profile,
        execution_profile=scan_profile,  # type: ignore[arg-type]
        behavior_evidence_path=behavior_evidence,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _echo(report.model_dump(mode="json"))


if __name__ == "__main__":
    app()
