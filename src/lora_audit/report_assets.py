"""Generate hash-bound, report-ready figures from completed study releases."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import is_sha256, same_json_value, sha256_file
from .manifests import QcResult, read_json_object, write_json_atomic
from .orchestration import CohortRunState
from .rq3 import Rq3Cell, verify_rq3_matrix

REPORT_DIRECTORY_NAME = "report-assets"
RESULT_REGISTER_NAME = "result-register.json"
FIGURE_MANIFEST_NAME = "figure-manifest.json"
EXPECTED_FIGURES = {
    "cohort_quality_control": (
        "cohort-quality-control",
        "Tình trạng tập thí nghiệm và kiểm soát chất lượng",
    ),
    "backdoor_validity_and_stealth": (
        "backdoor-validity-and-stealth",
        "Hiệu lực backdoor và độ kín đáo trên tác vụ sạch",
    ),
    "detector_comparison": (
        "detector-comparison",
        "So sánh các nhánh phát hiện trên tập kiểm thử giữ kín",
    ),
    "calibration_analysis": (
        "calibration-analysis",
        "Phân bố điểm và độ tin cậy xác suất",
    ),
    "adaptive_tradeoff": (
        "adaptive-tradeoff",
        "Đánh đổi giữa né phát hiện, hiệu lực tấn công và chất lượng tác vụ",
    ),
    "english_vietnamese_transfer": (
        "english-vietnamese-transfer",
        "Chuyển giao giữa tiếng Anh và tiếng Việt",
    ),
    "operational_latency": (
        "operational-latency",
        "Thời gian thực thi và mức sử dụng GPU theo công đoạn",
    ),
}
FORBIDDEN_VISIBLE_PATTERNS = (
    re.compile(r"\b(?:rq|rg|h)\s*[-_.]?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\bversion\b", re.IGNORECASE),
    re.compile(r"schema[_ -]?version", re.IGNORECASE),
)


@dataclass(frozen=True)
class ReportStudyData:
    cohort: CohortRunState
    qc_rows: list[dict[str, Any]]
    detector_release: dict[str, Any]
    adaptive_release: dict[str, Any]
    transfer_cells: list[Rq3Cell]
    telemetry_rows: list[dict[str, Any]]
    resource_summary: dict[str, Any]
    source_bindings: dict[str, dict[str, str]]
    fixture_smoke_only: bool


def _finite_number(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite JSON number")
    return float(value)


def _probability(value: object, *, label: str) -> float:
    observed = _finite_number(value, label=label)
    if not 0.0 <= observed <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return observed


def _percentage(value: object, *, label: str) -> float:
    observed = _finite_number(value, label=label)
    if not 0.0 <= observed <= 100.0:
        raise ValueError(f"{label} must be between zero and one hundred")
    return observed


def _safe_regular_file(path: Path, *, label: str) -> Path:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable: {path}") from error
    is_reparse = bool(getattr(file_stat, "st_file_attributes", 0) & 0x400)
    if path.is_symlink() or is_reparse or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{label} is missing, linked, or not a regular file: {path}")
    return path


def _safe_directory(path: Path, *, label: str) -> Path:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable: {path}") from error
    is_reparse = bool(getattr(file_stat, "st_file_attributes", 0) & 0x400)
    if path.is_symlink() or is_reparse or not stat.S_ISDIR(file_stat.st_mode):
        raise ValueError(f"{label} is missing, linked, or not a real directory: {path}")
    return path


def _safe_relative_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} path is invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path escapes its run root")
    path = root / candidate
    _safe_regular_file(path, label=label)
    if root.resolve() not in path.resolve().parents:
        raise ValueError(f"{label} resolves outside its run root")
    return path


def _load_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    _safe_regular_file(path, label=label)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} is malformed") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _input_bindings(project: Path, run_root: Path) -> dict[str, dict[str, str]]:
    paths = {
        "source_manifest": project / "artifacts" / "source-manifest.json",
        "bundle_manifest": project / "artifacts" / "bundle-manifest.json",
        "cohort_state": run_root / "cohort" / "run-state.json",
        "quality_control_ledger": run_root / "ledgers" / "qc.jsonl",
        "detector_release": run_root / "detectors" / "test-release.json",
        "adaptive_release": run_root / "rq2" / "results.json",
        "language_transfer_release": run_root / "rq3" / "results.jsonl",
        "runtime_telemetry": project / "artifacts" / "runtime-resource-telemetry.jsonl",
        "runtime_summary": project / "artifacts" / "runtime-resource-summary.json",
    }
    bindings: dict[str, dict[str, str]] = {}
    for label, path in paths.items():
        _safe_regular_file(path, label=label)
        bindings[label] = {
            "path": path.relative_to(project).as_posix(),
            "sha256": sha256_file(path),
        }
    return bindings


def _validate_qc_rows(
    rows: list[dict[str, Any]],
    *,
    cohort: CohortRunState,
    fixture_smoke_only: bool,
) -> list[dict[str, Any]]:
    expected_fields = set(QcResult.model_fields)
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not expected_fields.issubset(row):
            raise ValueError(f"quality-control row {index} is missing required fields")
        qc = QcResult.model_validate({field: row[field] for field in expected_fields})
        if qc.fixture_smoke_only is not fixture_smoke_only:
            raise ValueError("quality-control fixture boundary mismatch")
        if row.get("config_sha256") != cohort.config_hash:
            raise ValueError("quality-control configuration binding mismatch")
        execution_profile = row.get("execution_profile")
        if (
            not isinstance(execution_profile, dict)
            or execution_profile.get("profile_id") != cohort.execution_profile_id
            or execution_profile.get("profile_hash") != cohort.execution_profile_sha256
            or execution_profile.get("profile_file_sha256") != cohort.execution_profile_file_sha256
        ):
            raise ValueError("quality-control execution-profile binding mismatch")
        validated.append(row)
    parent_ids = [str(row["parent_config_id"]) for row in validated]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("quality-control ledger contains duplicate parent lineages")
    if set(parent_ids) != set(cohort.planned_lineage_ids):
        raise ValueError("quality-control ledger does not cover the completed cohort")
    return validated


def _validate_score_lineage(lineage: object) -> str:
    if not isinstance(lineage, dict) or not isinstance(lineage.get("parent_config_id"), str):
        raise ValueError("detector score-distribution lineage is invalid")
    if lineage.get("labels") != [0, 1]:
        raise ValueError("detector score-distribution labels are not paired")
    adapter_ids = lineage.get("adapter_ids")
    adapter_hashes = lineage.get("adapter_sha256s")
    if (
        not isinstance(adapter_ids, list)
        or len(adapter_ids) != 2
        or len(set(adapter_ids)) != 2
        or any(not isinstance(value, str) or not value for value in adapter_ids)
        or not isinstance(adapter_hashes, list)
        or len(adapter_hashes) != 2
        or len(set(adapter_hashes)) != 2
        or any(not is_sha256(value) for value in adapter_hashes)
    ):
        raise ValueError("detector score-distribution adapter identities are invalid")
    scores = lineage.get("scores")
    predictions = lineage.get("predictions")
    if not isinstance(scores, dict) or set(scores) != {"weight", "behavior", "fusion"}:
        raise ValueError("detector score-distribution branches are invalid")
    if not isinstance(predictions, dict) or set(predictions) != {
        "weight",
        "behavior",
        "fusion",
        "simple_or",
    }:
        raise ValueError("detector score-distribution predictions are invalid")
    for branch, values in scores.items():
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"detector {branch} scores are not paired")
        for value in values:
            _probability(value, label=f"detector {branch} score")
    if any(
        not isinstance(values, list)
        or len(values) != 2
        or any(type(value) is not int or value not in {0, 1} for value in values)
        for values in predictions.values()
    ):
        raise ValueError("detector score-distribution predictions are not binary pairs")
    return lineage["parent_config_id"]


def _validate_score_distribution(release: dict[str, Any]) -> None:
    distribution = release.get("score_distribution")
    if not isinstance(distribution, dict):
        raise ValueError("detector release has no score distribution")
    rows = distribution.get("rows")
    if (
        distribution.get("statistical_unit") != "parent_config_id"
        or type(distribution.get("lineage_count")) is not int
        or not isinstance(rows, list)
        or distribution["lineage_count"] != len(rows)
        or distribution["lineage_count"] != release.get("effective_lineage_count")
    ):
        raise ValueError("detector score-distribution header is invalid")
    parent_ids = [_validate_score_lineage(lineage) for lineage in rows]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("detector score distribution contains duplicate parent lineages")


def _validate_detector_release(
    release: dict[str, Any], *, cohort: CohortRunState
) -> dict[str, Any]:
    if (
        release.get("schema_version") != 4
        or release.get("status") != "released"
        or release.get("config_sha256") != cohort.config_hash
    ):
        raise ValueError("detector release identity or status is invalid")
    _validate_score_distribution(release)
    metrics = release.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"weight", "behavior", "fusion"}:
        raise ValueError("detector release metric branches are invalid")
    for branch, values in metrics.items():
        if not isinstance(values, dict) or not isinstance(values.get("bootstrap_ci_95"), dict):
            raise ValueError(f"detector {branch} metrics are invalid")
        for metric in ("auroc", "pr_auc", "mcc", "brier", "tpr", "fpr"):
            _finite_number(values.get(metric), label=f"detector {branch} {metric}")
            interval = values["bootstrap_ci_95"].get(metric)
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"detector {branch} {metric} interval is invalid")
            lower = _finite_number(interval[0], label=f"detector {branch} {metric} lower")
            upper = _finite_number(interval[1], label=f"detector {branch} {metric} upper")
            if lower > upper:
                raise ValueError(f"detector {branch} {metric} interval is reversed")
    return release


def _validate_adaptive_result(result: Any, *, fixture_smoke_only: bool) -> tuple[str, float]:
    if not isinstance(result, dict) or not isinstance(result.get("condition"), dict):
        raise ValueError("adaptive result is invalid")
    condition = result["condition"]
    run_id = condition.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("adaptive result has no run identifier")
    ratio = _finite_number(
        condition.get("desired_gradient_ratio"),
        label="adaptive gradient ratio",
    )
    if ratio < 0.0 or type(condition.get("replicate_seed")) is not int:
        raise ValueError("adaptive result ratio or seed is invalid")
    _probability(result.get("detector_score"), label="adaptive detector score")
    _probability(result.get("clean_metrics", {}).get("macro_f1"), label="adaptive macro F1")
    _probability(result.get("attack_success", {}).get("asr"), label="adaptive attack rate")
    qc_payload = result.get("qc")
    if not isinstance(qc_payload, dict):
        raise ValueError("adaptive result has no quality-control payload")
    qc = QcResult.model_validate(qc_payload)
    if qc.fixture_smoke_only is not fixture_smoke_only:
        raise ValueError("adaptive result fixture boundary mismatch")
    return run_id, ratio


def _validate_adaptive_release(
    release: dict[str, Any],
    *,
    cohort: CohortRunState,
    fixture_smoke_only: bool,
) -> dict[str, Any]:
    results = release.get("results")
    if (
        release.get("schema_version") != 2
        or release.get("status") != "completed"
        or release.get("config_sha256") != cohort.config_hash
        or release.get("planned_count") != 12
        or not isinstance(results, list)
        or len(results) != 12
    ):
        raise ValueError("adaptive release identity, status, or count is invalid")
    run_ids: list[str] = []
    ratio_counts: Counter[float] = Counter()
    for result in results:
        run_id, ratio = _validate_adaptive_result(
            result,
            fixture_smoke_only=fixture_smoke_only,
        )
        run_ids.append(run_id)
        ratio_counts[ratio] += 1
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("adaptive release contains duplicate run identifiers")
    if ratio_counts != Counter({0.0: 3, 0.1: 3, 0.3: 3, 1.0: 3}):
        raise ValueError("adaptive release does not contain the locked ratio-by-seed design")
    frontier = release.get("pareto_frontier")
    if not isinstance(frontier, list) or any(
        not isinstance(point, dict) or point.get("run_id") not in set(run_ids) for point in frontier
    ):
        raise ValueError("adaptive Pareto frontier is invalid")
    return release


def _validate_transfer_release(
    run_root: Path, path: Path, *, cohort: CohortRunState
) -> list[Rq3Cell]:
    _safe_regular_file(path, label="transfer release")
    cells = [
        Rq3Cell.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cells:
        raise ValueError("transfer release is empty")
    verify_rq3_matrix(cells, require_results=True)
    for cell in cells:
        if (
            cell.config_sha256 != cohort.config_hash
            or cell.execution_profile_id != cohort.execution_profile_id
            or cell.execution_profile_sha256 != cohort.execution_profile_sha256
            or cell.execution_profile_file_sha256 != cohort.execution_profile_file_sha256
        ):
            raise ValueError("transfer release does not bind the completed cohort")
        distribution_path = _safe_relative_file(
            run_root,
            cell.detector_score_distribution_artifact,
            label="transfer score distribution",
        )
        if sha256_file(distribution_path) != cell.detector_score_distribution_sha256:
            raise ValueError("transfer score-distribution hash mismatch")
    return cells


def _validate_telemetry_row(row: dict[str, Any]) -> None:
    if (
        row.get("schema_version") != 1
        or row.get("status") not in {"passed", "failed"}
        or not isinstance(row.get("command"), str)
        or not row["command"]
        or type(row.get("sample_count")) is not int
        or row["sample_count"] < 2
        or row.get("monitor_errors") != []
    ):
        raise ValueError("runtime telemetry row is malformed")
    if _finite_number(row.get("wall_seconds"), label="runtime wall time") <= 0.0:
        raise ValueError("runtime wall time must be positive")
    gpu = row.get("gpu")
    if not isinstance(gpu, dict):
        raise ValueError("runtime telemetry row has no GPU measurements")
    mean = _percentage(gpu.get("utilization_mean_percent"), label="mean GPU utilization")
    p95 = _percentage(gpu.get("utilization_p95_percent"), label="p95 GPU utilization")
    maximum = _percentage(gpu.get("utilization_max_percent"), label="maximum GPU utilization")
    if p95 > maximum or mean > maximum:
        raise ValueError("runtime GPU utilization summaries are inconsistent")
    memory_total = _finite_number(gpu.get("memory_total_gib"), label="GPU memory capacity")
    memory_peak = _finite_number(gpu.get("memory_used_peak_gib"), label="GPU memory peak")
    if memory_total <= 0.0 or not 0.0 <= memory_peak <= memory_total:
        raise ValueError("runtime GPU memory measurements are inconsistent")


def _validate_telemetry(
    rows: list[dict[str, Any]], resource_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    if (
        resource_summary.get("schema_version") != 1
        or resource_summary.get("status") != "observed"
        or resource_summary.get("claim_boundary")
        != "sampled_utilization_not_saturation_or_quality_proof"
    ):
        raise ValueError("runtime resource summary is not complete")
    identity = resource_summary.get("routing_identity_sha256")
    if not is_sha256(identity):
        raise ValueError("runtime resource summary has no valid routing identity")
    selected = [row for row in rows if row.get("routing_identity_sha256") == identity]
    if not selected:
        raise ValueError("runtime telemetry has no rows for the completed routing identity")
    for row in selected:
        _validate_telemetry_row(row)
    expected_commands = sorted({str(row["command"]) for row in selected})
    expected_failures = sum(row["status"] != "passed" for row in selected)
    if (
        resource_summary.get("sampled_command_count") != len(selected)
        or resource_summary.get("failed_command_count") != expected_failures
        or resource_summary.get("observed_commands") != expected_commands
    ):
        raise ValueError("runtime resource summary does not match its telemetry ledger")
    return selected


def _load_study_data(
    *,
    project: Path,
    run_root: Path,
    source_manifest: dict[str, Any],
    bundle_manifest: dict[str, Any],
    fixture_smoke_only: bool,
) -> ReportStudyData:
    bindings = _input_bindings(project, run_root)
    persisted_source = read_json_object(project / "artifacts" / "source-manifest.json")
    persisted_bundle = read_json_object(project / "artifacts" / "bundle-manifest.json")
    if not same_json_value(persisted_source, source_manifest):
        raise ValueError("report assets require the persisted current source manifest")
    if not same_json_value(persisted_bundle, bundle_manifest):
        raise ValueError("report assets require the persisted current bundle manifest")
    cohort = CohortRunState.model_validate(read_json_object(run_root / "cohort" / "run-state.json"))
    if (
        cohort.status != "completed"
        or len(cohort.planned_lineage_ids) != len(cohort.completed_lineage_ids)
        or set(cohort.planned_lineage_ids) != set(cohort.completed_lineage_ids)
    ):
        raise ValueError("report assets require a completed cohort")
    if cohort.source_tree_sha256 != source_manifest.get("tree_sha256"):
        raise ValueError("report cohort does not bind the current source tree")
    if cohort.bundle_tree_sha256 != bundle_manifest.get("tree_sha256"):
        raise ValueError("report cohort does not bind the current bundle tree")
    qc_rows = _validate_qc_rows(
        _load_jsonl_objects(run_root / "ledgers" / "qc.jsonl", label="quality-control ledger"),
        cohort=cohort,
        fixture_smoke_only=fixture_smoke_only,
    )
    detector = _validate_detector_release(
        read_json_object(run_root / "detectors" / "test-release.json"), cohort=cohort
    )
    adaptive = _validate_adaptive_release(
        read_json_object(run_root / "rq2" / "results.json"),
        cohort=cohort,
        fixture_smoke_only=fixture_smoke_only,
    )
    if adaptive.get("source_tree_sha256") != source_manifest.get("tree_sha256") or adaptive.get(
        "bundle_tree_sha256"
    ) != bundle_manifest.get("tree_sha256"):
        raise ValueError("adaptive release does not bind the current source and bundle trees")
    cells = _validate_transfer_release(
        run_root,
        run_root / "rq3" / "results.jsonl",
        cohort=cohort,
    )
    resource_summary = read_json_object(project / "artifacts" / "runtime-resource-summary.json")
    if resource_summary.get("source_tree_sha256") != source_manifest.get(
        "tree_sha256"
    ) or resource_summary.get("bundle_tree_sha256") != bundle_manifest.get("tree_sha256"):
        raise ValueError("runtime summary does not bind the current source and bundle trees")
    telemetry = _validate_telemetry(
        _load_jsonl_objects(
            project / "artifacts" / "runtime-resource-telemetry.jsonl",
            label="runtime resource telemetry",
        ),
        resource_summary,
    )
    return ReportStudyData(
        cohort=cohort,
        qc_rows=qc_rows,
        detector_release=detector,
        adaptive_release=adaptive,
        transfer_cells=cells,
        telemetry_rows=telemetry,
        resource_summary=resource_summary,
        source_bindings=bindings,
        fixture_smoke_only=fixture_smoke_only,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        raise ValueError(f"report table has no rows: {path.name}")
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"report table field mismatch: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _table_binding(report_root: Path, path: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(report_root).as_posix(),
        "sha256": sha256_file(path),
        "row_count": row_count,
    }


def _cohort_table(data: ReportStudyData) -> list[dict[str, Any]]:
    valid_count = sum(row["valid"] is True for row in data.qc_rows)
    return [
        {"category": "planned", "count": len(data.cohort.planned_lineage_ids)},
        {"category": "completed", "count": len(data.cohort.completed_lineage_ids)},
        {"category": "quality_control_passed", "count": valid_count},
        {"category": "quality_control_failed", "count": len(data.qc_rows) - valid_count},
    ]


def _validity_table(data: ReportStudyData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(data.qc_rows, key=lambda value: str(value["parent_config_id"])):
        rows.append(
            {
                "parent_config_id": row["parent_config_id"],
                "clean_triggered_target_rate": row["clean_triggered_target_rate"],
                "backdoored_attack_success_rate": row["backdoored_asr"],
                "paired_attack_success_lift": row["paired_asr_lift"],
                "clean_adapter_macro_f1": row["paired_clean_macro_f1"],
                "backdoored_adapter_macro_f1": row["clean_macro_f1"],
                "absolute_macro_f1_difference": row["absolute_clean_macro_f1_difference"],
                "quality_control_passed": row["valid"],
            }
        )
    return rows


def _detector_metric_table(data: ReportStudyData) -> list[dict[str, Any]]:
    labels = {
        "weight": "Trọng số",
        "behavior": "Hành vi",
        "fusion": "Kết hợp",
    }
    rows: list[dict[str, Any]] = []
    for branch in ("weight", "behavior", "fusion"):
        payload = data.detector_release["metrics"][branch]
        for metric in ("auroc", "pr_auc", "mcc", "brier", "tpr", "fpr"):
            interval = payload["bootstrap_ci_95"][metric]
            rows.append(
                {
                    "detector_branch": branch,
                    "display_label": labels[branch],
                    "metric": metric,
                    "estimate": payload[metric],
                    "ci_lower": interval[0],
                    "ci_upper": interval[1],
                }
            )
    simple = data.detector_release["simple_or"]
    for metric in ("mcc", "tpr", "fpr"):
        interval = simple["bootstrap_ci_95"][metric]
        rows.append(
            {
                "detector_branch": "simple_or",
                "display_label": "Luật OR đơn giản",
                "metric": metric,
                "estimate": simple[metric],
                "ci_lower": interval[0],
                "ci_upper": interval[1],
            }
        )
    return rows


def _detector_score_table(data: ReportStudyData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineage in data.detector_release["score_distribution"]["rows"]:
        for offset, label in enumerate(lineage["labels"]):
            rows.append(
                {
                    "parent_config_id": lineage["parent_config_id"],
                    "adapter_id": lineage["adapter_ids"][offset],
                    "adapter_sha256": lineage["adapter_sha256s"][offset],
                    "adapter_class": "backdoored" if label == 1 else "clean",
                    "label": label,
                    "weight_score": lineage["scores"]["weight"][offset],
                    "behavior_score": lineage["scores"]["behavior"][offset],
                    "fusion_score": lineage["scores"]["fusion"][offset],
                    "weight_prediction": lineage["predictions"]["weight"][offset],
                    "behavior_prediction": lineage["predictions"]["behavior"][offset],
                    "fusion_prediction": lineage["predictions"]["fusion"][offset],
                    "simple_or_prediction": lineage["predictions"]["simple_or"][offset],
                }
            )
    return rows


def _adaptive_table(data: ReportStudyData) -> list[dict[str, Any]]:
    frontier = {point["run_id"] for point in data.adaptive_release["pareto_frontier"]}
    rows: list[dict[str, Any]] = []
    for result in data.adaptive_release["results"]:
        condition = result["condition"]
        attack = result["attack_success"]
        rows.append(
            {
                "run_id": condition["run_id"],
                "desired_gradient_ratio": condition["desired_gradient_ratio"],
                "replicate_seed": condition["replicate_seed"],
                "attack_success_rate": attack["asr"],
                "attack_ci_lower": attack["interval_95"][0],
                "attack_ci_upper": attack["interval_95"][1],
                "clean_macro_f1": result["clean_metrics"]["macro_f1"],
                "detection_score": result["detector_score"],
                "quality_control_passed": result["qc"]["valid"],
                "pareto_frontier": condition["run_id"] in frontier,
                "training_device_hours": result["training_telemetry"]["device_hours"],
                "evaluation_device_hours": result["evaluation_telemetry"]["device_hours"],
            }
        )
    return sorted(
        rows, key=lambda row: (float(row["desired_gradient_ratio"]), int(row["replicate_seed"]))
    )


def _transfer_table(data: ReportStudyData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in data.transfer_cells:
        rows.append(
            {
                "poison_language": cell.poison_language,
                "probe_language": cell.probe_language,
                "attack_success_rate": cell.backdoor_transfer_asr,
                "attack_ci_lower": cell.backdoor_transfer_interval_95[0],
                "attack_ci_upper": cell.backdoor_transfer_interval_95[1],
                "detector_auroc": cell.detector_auroc,
                "auroc_ci_lower": cell.detector_auroc_interval_95[0],
                "auroc_ci_upper": cell.detector_auroc_interval_95[1],
                "detector_pr_auc": cell.detector_pr_auc,
                "detector_brier": cell.detector_brier,
                "detector_recall": cell.detector_recall,
                "detector_score_mean": cell.detector_score_mean,
                "detector_score_shift": cell.detector_score_shift_from_source,
                "effective_parent_lineages": cell.effective_parent_lineage_count,
            }
        )
    return rows


def _runtime_display_label(command: str) -> str:
    normalized = command.casefold()
    mappings = (
        ("preflight", "Kiểm tra môi trường"),
        ("prefetch", "Chuẩn bị mô hình và dữ liệu"),
        ("prepare", "Chuẩn bị tập dữ liệu"),
        ("cohort", "Huấn luyện và đánh giá cặp"),
        ("fit-detectors", "Hiệu chỉnh bộ phát hiện"),
        ("evaluate-detectors", "Đánh giá bộ phát hiện"),
        ("release-detector", "Đánh giá bộ phát hiện"),
        ("estimate-compute", "Ước lượng tài nguyên"),
        ("adaptive", "Phân tích thích nghi"),
        ("transfer", "Phân tích chuyển giao ngôn ngữ"),
        ("rq3", "Phân tích chuyển giao ngôn ngữ"),
        ("pip", "Cài đặt môi trường"),
    )
    for token, label in mappings:
        if token in normalized:
            return label
    return "Công đoạn bổ trợ"


def _operational_table(data: ReportStudyData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data.telemetry_rows, start=1):
        rows.append(
            {
                "sequence": index,
                "command": row["command"],
                "display_label": _runtime_display_label(row["command"]),
                "status": row.get("status"),
                "wall_seconds": row["wall_seconds"],
                "sample_count": row["sample_count"],
                "gpu_utilization_mean_percent": row["gpu"]["utilization_mean_percent"],
                "gpu_utilization_p95_percent": row["gpu"]["utilization_p95_percent"],
                "gpu_utilization_max_percent": row["gpu"]["utilization_max_percent"],
                "gpu_memory_peak_gib": row["gpu"]["memory_used_peak_gib"],
            }
        )
    return rows


def _write_source_tables(
    report_root: Path, data: ReportStudyData
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    tables = {
        "cohort_quality_control": _cohort_table(data),
        "backdoor_validity_and_stealth": _validity_table(data),
        "detector_comparison": _detector_metric_table(data),
        "calibration_analysis": _detector_score_table(data),
        "adaptive_tradeoff": _adaptive_table(data),
        "english_vietnamese_transfer": _transfer_table(data),
        "operational_latency": _operational_table(data),
    }
    bindings: dict[str, dict[str, Any]] = {}
    for name, rows in tables.items():
        path = report_root / "tables" / f"{EXPECTED_FIGURES[name][0]}.csv"
        fields = list(rows[0])
        _write_csv(path, rows, fields)
        bindings[name] = _table_binding(report_root, path, len(rows))
    return tables, bindings


def _result_register_payload(
    data: ReportStudyData, table_bindings: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "fixture_smoke_only": data.fixture_smoke_only,
        "source_bindings": data.source_bindings,
        "statistical_units": {
            "paired_cohort": "parent_config_id",
            "detector_release": "parent_config_id",
            "adaptive_analysis": "condition_by_replicate",
            "language_transfer": "parent_config_id within language cell",
            "runtime_telemetry": "monitored command invocation",
        },
        "sections": {
            "cohort_quality_control": {
                "table": table_bindings["cohort_quality_control"],
                "planned_lineages": len(data.cohort.planned_lineage_ids),
                "completed_lineages": len(data.cohort.completed_lineage_ids),
                "quality_control_passed": sum(row["valid"] is True for row in data.qc_rows),
            },
            "backdoor_validity_and_stealth": {
                "table": table_bindings["backdoor_validity_and_stealth"],
                "paired_lineage_count": len(data.qc_rows),
                "estimands": [
                    "triggered target rate on the clean adapter",
                    "attack success rate on the backdoored adapter",
                    "paired attack-success lift",
                    "absolute clean macro-F1 difference",
                ],
            },
            "detector_comparison": {
                "table": table_bindings["detector_comparison"],
                "effective_parent_lineages": data.detector_release["effective_lineage_count"],
                "uncertainty": "parent-lineage cluster bootstrap",
            },
            "calibration_analysis": {
                "table": table_bindings["calibration_analysis"],
                "held_out_adapter_count": data.detector_release["test_adapter_count"],
                "interpretation": "descriptive reliability and score separation",
            },
            "adaptive_tradeoff": {
                "table": table_bindings["adaptive_tradeoff"],
                "condition_count": data.adaptive_release["planned_count"],
                "pareto_directions": {
                    "attack_success_rate": "higher",
                    "clean_macro_f1": "higher",
                    "detection_score": "lower",
                },
            },
            "english_vietnamese_transfer": {
                "table": table_bindings["english_vietnamese_transfer"],
                "cell_count": len(data.transfer_cells),
                "views": ["attack efficacy", "discrimination", "probability error"],
            },
            "operational_latency": {
                "table": table_bindings["operational_latency"],
                "monitored_invocation_count": len(data.telemetry_rows),
                "claim_boundary": data.resource_summary.get("claim_boundary"),
            },
        },
        "limitations": [
            "Hosted results support only the locked study population and execution profile.",
            "Sampled GPU utilization is operational evidence, not proof of saturation or quality.",
            "Held-out adapter counts do not support a production false-positive-rate claim.",
            "Fixture-marked assets are test artifacts and must not enter thesis results.",
        ],
    }


def _plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "svg.hashsalt": "lora-audit-report-assets",
        }
    )
    return plt


def _figure_visible_text(svg_path: Path) -> str:
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"generated SVG is malformed: {svg_path.name}") from error
    visible_tags = {"text", "title", "desc"}
    values: list[str] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in visible_tags:
            values.extend(part for part in element.itertext() if part)
    return " ".join(values)


def _assert_clean_visible_text(svg_path: Path) -> None:
    text = _figure_visible_text(svg_path)
    matches = [pattern.pattern for pattern in FORBIDDEN_VISIBLE_PATTERNS if pattern.search(text)]
    if matches:
        raise ValueError(f"generated figure contains forbidden visible labels: {matches}")


def _save_figure(plt: Any, figure: Any, report_root: Path, slug: str) -> dict[str, Any]:
    figure_directory = report_root / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    svg_path = figure_directory / f"{slug}.svg"
    png_path = figure_directory / f"{slug}.png"
    figure.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "LoRA audit report renderer"},
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "LoRA audit report renderer"},
    )
    plt.close(figure)
    _safe_regular_file(svg_path, label="generated SVG")
    _safe_regular_file(png_path, label="generated PNG")
    if png_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("generated PNG signature is invalid")
    _assert_clean_visible_text(svg_path)
    return {
        "svg": {
            "path": svg_path.relative_to(report_root).as_posix(),
            "sha256": sha256_file(svg_path),
        },
        "png": {
            "path": png_path.relative_to(report_root).as_posix(),
            "sha256": sha256_file(png_path),
            "dpi": 300,
        },
    }


def _quality_control_reason_label(reason: str) -> str:
    labels = {
        "zero_eligible_rows": "Không có mẫu đủ điều kiện",
        "clean_triggered_target_rate_above_threshold": "Đối chứng sạch vượt ngưỡng",
        "backdoored_asr_below_threshold": "Tỷ lệ tấn công dưới ngưỡng",
        "paired_asr_lift_below_threshold": "Mức tăng tấn công dưới ngưỡng",
        "absolute_clean_macro_f1_difference_above_threshold": (
            "Chênh lệch Macro-F1 sạch vượt ngưỡng"
        ),
    }
    if reason in labels:
        return labels[reason]
    if reason.startswith("nonfinite_"):
        return "Chỉ số không hữu hạn"
    if reason.endswith("_out_of_range"):
        return "Chỉ số ngoài miền hợp lệ"
    return "Lý do kiểm soát chưa phân loại"


def _plot_cohort_quality_control(
    plt: Any, rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]]
) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = ["Lập kế hoạch", "Hoàn tất", "Đạt kiểm soát", "Không đạt"]
    counts = [int(row["count"]) for row in rows]
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#E45756"]
    axes[0].bar(labels, counts, color=colors, width=0.68)
    axes[0].set_ylabel("Số lineage cha")
    axes[0].set_title("Mức độ hoàn tất")
    axes[0].tick_params(axis="x", rotation=18)
    for index, count in enumerate(counts):
        axes[0].text(index, count + max(counts) * 0.02, str(count), ha="center", va="bottom")
    reasons = Counter(
        _quality_control_reason_label(reason)
        for row in qc_rows
        if row["valid"] is False
        for reason in row.get("reason_codes", [])
    )
    if reasons:
        reason_labels = [reason for reason, _ in reasons.most_common()]
        reason_counts = [count for _, count in reasons.most_common()]
        axes[1].barh(reason_labels, reason_counts, color="#E45756")
        axes[1].set_xlabel("Số lineage")
        axes[1].set_title("Nguyên nhân không đạt")
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.55,
            "Không ghi nhận lineage không đạt\nkiểm soát chất lượng",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[1].transAxes,
        )
    figure.suptitle(EXPECTED_FIGURES["cohort_quality_control"][1], fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def _paired_values(rows: list[dict[str, Any]], left: str, right: str) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for row in rows:
        if row[left] is None or row[right] is None:
            continue
        values.append(
            (
                _probability(row[left], label=left),
                _probability(row[right], label=right),
            )
        )
    if not values:
        raise ValueError(f"report figure has no finite paired values for {left} and {right}")
    return values


def _plot_backdoor_validity(
    plt: Any, rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]]
) -> Any:
    rate_pairs = _paired_values(
        rows, "clean_triggered_target_rate", "backdoored_attack_success_rate"
    )
    f1_pairs = _paired_values(rows, "clean_adapter_macro_f1", "backdoored_adapter_macro_f1")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    panels = (
        (
            axes[0],
            rate_pairs,
            ("Adapter sạch", "Adapter backdoor"),
            "Tỷ lệ dự đoán mục tiêu khi có trigger",
        ),
        (
            axes[1],
            f1_pairs,
            ("Adapter sạch", "Adapter backdoor"),
            "Macro-F1 trên tập sạch",
        ),
    )
    for axis, pairs, labels, title in panels:
        for left, right in pairs:
            axis.plot([0, 1], [left, right], color="#B8B8B8", alpha=0.45, linewidth=0.8)
            axis.scatter([0, 1], [left, right], color=["#4C78A8", "#E45756"], s=22)
        axis.set_xticks([0, 1], labels)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    thresholds = qc_rows[0]["thresholds"]
    axes[0].axhline(
        float(thresholds["maximum_clean_triggered_target_rate"]),
        color="#4C78A8",
        linestyle="--",
        linewidth=1,
        label="Ngưỡng tối đa của đối chứng sạch",
    )
    axes[0].axhline(
        float(thresholds["minimum_backdoored_asr"]),
        color="#E45756",
        linestyle="--",
        linewidth=1,
        label="Ngưỡng tối thiểu của adapter backdoor",
    )
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Giá trị")
    figure.suptitle(
        EXPECTED_FIGURES["backdoor_validity_and_stealth"][1],
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def _plot_detector_comparison(plt: Any, rows: list[dict[str, Any]]) -> Any:
    metric_labels = {
        "auroc": "AUROC",
        "pr_auc": "PR-AUC",
        "mcc": "MCC",
        "brier": "Brier (thấp hơn tốt hơn)",
    }
    branch_order = ["weight", "behavior", "fusion", "simple_or"]
    colors = {
        "weight": "#4C78A8",
        "behavior": "#F58518",
        "fusion": "#54A24B",
        "simple_or": "#B279A2",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for axis, metric in zip(axes.flat, metric_labels, strict=True):
        selected = [row for row in rows if row["metric"] == metric]
        selected.sort(key=lambda row: branch_order.index(str(row["detector_branch"])))
        for index, row in enumerate(selected):
            estimate = float(row["estimate"])
            lower = float(row["ci_lower"])
            upper = float(row["ci_upper"])
            color = colors[str(row["detector_branch"])]
            axis.vlines(index, lower, upper, color=color, linewidth=1.6)
            axis.hlines([lower, upper], index - 0.04, index + 0.04, color=color, linewidth=1.2)
            axis.scatter(index, estimate, color=color, s=54, zorder=3)
        axis.set_xticks(range(len(selected)), [str(row["display_label"]) for row in selected])
        axis.tick_params(axis="x", rotation=15)
        axis.set_title(metric_labels[metric])
        axis.grid(axis="y", alpha=0.2)
        if metric == "mcc":
            axis.set_ylim(-1.05, 1.05)
            axis.axhline(0.0, color="#777777", linewidth=0.8)
        else:
            axis.set_ylim(-0.03, 1.03)
    figure.suptitle(EXPECTED_FIGURES["detector_comparison"][1], fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def _reliability_points(rows: list[dict[str, Any]]) -> list[tuple[float, float, int]]:
    bins = np.linspace(0.0, 1.0, 6)
    scores = np.asarray([float(row["fusion_score"]) for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    assignments = np.minimum(np.digitize(scores, bins, right=False) - 1, len(bins) - 2)
    points: list[tuple[float, float, int]] = []
    for index in range(len(bins) - 1):
        mask = assignments == index
        if np.any(mask):
            points.append(
                (float(np.mean(scores[mask])), float(np.mean(labels[mask])), int(mask.sum()))
            )
    return points


def _plot_calibration_analysis(plt: Any, rows: list[dict[str, Any]]) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    branches = (
        ("weight_score", "Điểm trọng số"),
        ("behavior_score", "Điểm hành vi"),
        ("fusion_score", "Điểm kết hợp"),
    )
    for axis, (field, title) in zip(axes.flat[:3], branches, strict=True):
        clean = [float(row[field]) for row in rows if row["label"] == 0]
        backdoored = [float(row[field]) for row in rows if row["label"] == 1]
        axis.boxplot(
            [clean, backdoored],
            tick_labels=["Sạch", "Backdoor"],
            patch_artist=True,
            boxprops={"facecolor": "#DCEAF7"},
            medianprops={"color": "#222222"},
        )
        for location, values, color in ((1, clean, "#4C78A8"), (2, backdoored, "#E45756")):
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(location + jitter, values, color=color, alpha=0.7, s=20)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    reliability = _reliability_points(rows)
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Lý tưởng")
    axes[1, 1].plot(
        [point[0] for point in reliability],
        [point[1] for point in reliability],
        marker="o",
        color="#54A24B",
        label="Quan sát",
    )
    for mean_score, observed_rate, count in reliability:
        axes[1, 1].annotate(
            f"n={count}",
            (mean_score, observed_rate),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set_xlim(-0.03, 1.03)
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_xlabel("Điểm kết hợp trung bình")
    axes[1, 1].set_ylabel("Tỷ lệ backdoor quan sát")
    axes[1, 1].set_title("Độ tin cậy xác suất mô tả")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(EXPECTED_FIGURES["calibration_analysis"][1], fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def _plot_adaptive_tradeoff(plt: Any, rows: list[dict[str, Any]]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    detection = np.asarray([float(row["detection_score"]) for row in rows])
    attack = np.asarray([float(row["attack_success_rate"]) for row in rows])
    clean_f1 = np.asarray([float(row["clean_macro_f1"]) for row in rows])
    passed = np.asarray([bool(row["quality_control_passed"]) for row in rows])
    scatter = axes[0].scatter(
        detection,
        attack,
        c=clean_f1,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=np.where(passed, 72, 46),
        marker="o",
        edgecolors=np.where(passed, "#222222", "#E45756"),
        linewidths=1.0,
    )
    for row in rows:
        if row["pareto_frontier"]:
            axes[0].scatter(
                float(row["detection_score"]),
                float(row["attack_success_rate"]),
                marker="*",
                s=180,
                facecolors="none",
                edgecolors="#D62728",
                linewidths=1.4,
            )
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("Điểm phát hiện (thấp hơn tốt hơn cho né phát hiện)")
    axes[0].set_ylabel("Tỷ lệ tấn công thành công")
    axes[0].set_title("Không gian đánh đổi và biên Pareto")
    figure.colorbar(scatter, ax=axes[0], label="Macro-F1 trên tập sạch")
    ratios = sorted({float(row["desired_gradient_ratio"]) for row in rows})
    summaries = []
    for ratio in ratios:
        selected = [row for row in rows if float(row["desired_gradient_ratio"]) == ratio]
        summaries.append(
            {
                "ratio": ratio,
                "attack": float(np.mean([float(row["attack_success_rate"]) for row in selected])),
                "detection": float(np.mean([float(row["detection_score"]) for row in selected])),
                "clean_f1": float(np.mean([float(row["clean_macro_f1"]) for row in selected])),
            }
        )
    for field, label, color, marker in (
        ("attack", "Tấn công thành công", "#E45756", "o"),
        ("detection", "Điểm phát hiện", "#4C78A8", "s"),
        ("clean_f1", "Macro-F1 sạch", "#54A24B", "^"),
    ):
        axes[1].plot(
            [summary["ratio"] for summary in summaries],
            [summary[field] for summary in summaries],
            color=color,
            marker=marker,
            label=label,
        )
    axes[1].set_xticks(ratios, [f"{ratio:g}" for ratio in ratios])
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Tỷ lệ gradient mong muốn")
    axes[1].set_ylabel("Giá trị trung bình theo seed")
    axes[1].set_title("Xu hướng theo cường độ thích nghi")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.suptitle(EXPECTED_FIGURES["adaptive_tradeoff"][1], fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def _language_label(value: str) -> str:
    labels = {"en-US": "Tiếng Anh", "vi-VN": "Tiếng Việt"}
    try:
        return labels[value]
    except KeyError as error:
        raise ValueError(f"unsupported report language: {value}") from error


def _plot_transfer(plt: Any, rows: list[dict[str, Any]]) -> Any:
    languages = ("en-US", "vi-VN")
    fields = (
        ("attack_success_rate", "Tấn công thành công", "magma"),
        ("detector_auroc", "AUROC phát hiện", "viridis"),
        ("detector_brier", "Sai số Brier", "cividis_r"),
    )
    lookup = {(row["poison_language"], row["probe_language"]): row for row in rows}
    if set(lookup) != {(source, probe) for source in languages for probe in languages}:
        raise ValueError("language-transfer table does not contain the complete matrix")
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    for axis, (field, title, color_map) in zip(axes, fields, strict=True):
        matrix = np.asarray(
            [[float(lookup[(source, probe)][field]) for probe in languages] for source in languages]
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap=color_map)
        axis.set_xticks(range(2), [_language_label(value) for value in languages])
        axis.set_yticks(range(2), [_language_label(value) for value in languages])
        axis.set_xlabel("Ngôn ngữ probe")
        axis.set_ylabel("Ngôn ngữ poison")
        axis.set_title(title)
        for row_index in range(2):
            for column_index in range(2):
                value = matrix[row_index, column_index]
                text_color = "white" if value < 0.25 or value > 0.72 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        EXPECTED_FIGURES["english_vietnamese_transfer"][1],
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def _aggregate_operational_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["display_label"]), []).append(row)
    output: list[dict[str, Any]] = []
    for label, selected in grouped.items():
        total_seconds = sum(float(row["wall_seconds"]) for row in selected)
        weighted_utilization = (
            sum(
                float(row["gpu_utilization_mean_percent"]) * float(row["wall_seconds"])
                for row in selected
            )
            / total_seconds
        )
        output.append(
            {
                "label": label,
                "invocations": len(selected),
                "wall_seconds": total_seconds,
                "gpu_utilization_mean_percent": weighted_utilization,
                "gpu_utilization_max_percent": max(
                    float(row["gpu_utilization_max_percent"]) for row in selected
                ),
            }
        )
    return sorted(output, key=lambda row: float(row["wall_seconds"]))


def _plot_operational_latency(plt: Any, rows: list[dict[str, Any]]) -> Any:
    aggregated = _aggregate_operational_rows(rows)
    labels = [f"{row['label']} (n={row['invocations']})" for row in aggregated]
    seconds = np.asarray([float(row["wall_seconds"]) for row in aggregated])
    divisor, unit = (3600.0, "Giờ") if float(np.max(seconds)) >= 7200.0 else (60.0, "Phút")
    figure, axes = plt.subplots(1, 2, figsize=(12.2, max(4.8, len(labels) * 0.55)))
    axes[0].barh(labels, seconds / divisor, color="#4C78A8")
    axes[0].set_xlabel(f"{unit} thực thi cộng dồn")
    axes[0].set_title("Thời gian theo công đoạn")
    axes[0].grid(axis="x", alpha=0.2)
    mean_utilization = [float(row["gpu_utilization_mean_percent"]) for row in aggregated]
    max_utilization = [float(row["gpu_utilization_max_percent"]) for row in aggregated]
    positions = np.arange(len(labels))
    axes[1].scatter(mean_utilization, positions, color="#54A24B", label="Trung bình có trọng số")
    axes[1].scatter(max_utilization, positions, color="#E45756", marker="x", label="Cực đại mẫu")
    for position, mean_value, max_value in zip(
        positions, mean_utilization, max_utilization, strict=True
    ):
        axes[1].plot([mean_value, max_value], [position, position], color="#AAAAAA", linewidth=0.8)
    axes[1].set_yticks(positions, labels)
    axes[1].set_xlim(-2.0, 102.0)
    axes[1].set_xlabel("Mức sử dụng GPU được lấy mẫu (%)")
    axes[1].set_title("Mức sử dụng GPU")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle(EXPECTED_FIGURES["operational_latency"][1], fontsize=14, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "Số liệu lấy mẫu mô tả vận hành; không phải bằng chứng bão hòa tài nguyên "
        "hay chất lượng mô hình.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    return figure


def _render_figures(
    report_root: Path,
    data: ReportStudyData,
    tables: dict[str, list[dict[str, Any]]],
    table_bindings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    plt = _plotting()
    builders = {
        "cohort_quality_control": lambda: _plot_cohort_quality_control(
            plt, tables["cohort_quality_control"], data.qc_rows
        ),
        "backdoor_validity_and_stealth": lambda: _plot_backdoor_validity(
            plt, tables["backdoor_validity_and_stealth"], data.qc_rows
        ),
        "detector_comparison": lambda: _plot_detector_comparison(
            plt, tables["detector_comparison"]
        ),
        "calibration_analysis": lambda: _plot_calibration_analysis(
            plt, tables["calibration_analysis"]
        ),
        "adaptive_tradeoff": lambda: _plot_adaptive_tradeoff(plt, tables["adaptive_tradeoff"]),
        "english_vietnamese_transfer": lambda: _plot_transfer(
            plt, tables["english_vietnamese_transfer"]
        ),
        "operational_latency": lambda: _plot_operational_latency(
            plt, tables["operational_latency"]
        ),
    }
    figures: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_FIGURES:
        slug, title = EXPECTED_FIGURES[name]
        files = _save_figure(plt, builders[name](), report_root, slug)
        figures[name] = {
            "title": title,
            "source_table": table_bindings[name],
            **files,
        }
    return figures


def _bound_report_file(report_root: Path, binding: object, *, label: str) -> Path:
    if not isinstance(binding, dict) or set(binding) - {"path", "sha256", "row_count", "dpi"}:
        raise ValueError(f"{label} binding is invalid")
    path = _safe_relative_file(report_root, binding.get("path"), label=label)
    expected_hash = binding.get("sha256")
    if not is_sha256(expected_hash) or sha256_file(path) != expected_hash:
        raise ValueError(f"{label} hash mismatch")
    if "row_count" in binding and (
        type(binding["row_count"]) is not int or binding["row_count"] < 1
    ):
        raise ValueError(f"{label} row count is invalid")
    if "dpi" in binding and binding["dpi"] != 300:
        raise ValueError(f"{label} DPI is invalid")
    return path


def _validate_figure_binding(
    root: Path,
    *,
    name: str,
    figure: object,
    section: object,
    observed_paths: set[Path],
) -> None:
    expected_title = EXPECTED_FIGURES[name][1]
    if not isinstance(figure, dict) or figure.get("title") != expected_title:
        raise ValueError(f"report figure metadata is invalid: {name}")
    table_path = _bound_report_file(root, figure.get("source_table"), label=f"{name} source table")
    svg_path = _bound_report_file(root, figure.get("svg"), label=f"{name} SVG")
    png_path = _bound_report_file(root, figure.get("png"), label=f"{name} PNG")
    if not isinstance(section, dict) or not same_json_value(
        section.get("table"), figure.get("source_table")
    ):
        raise ValueError(f"result register and figure table binding differ: {name}")
    if {table_path, svg_path, png_path} & observed_paths:
        raise ValueError("report manifest reuses one file for multiple evidence roles")
    observed_paths.update({table_path, svg_path, png_path})
    _assert_clean_visible_text(svg_path)
    if png_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"report PNG signature is invalid: {name}")


def _validate_exact_report_inventory(root: Path, expected_files: set[Path]) -> None:
    expected_directories = {(root / "figures").resolve(), (root / "tables").resolve()}
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    for candidate in root.rglob("*"):
        try:
            candidate_stat = candidate.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"report asset is unreadable: {candidate}") from error
        is_reparse = bool(getattr(candidate_stat, "st_file_attributes", 0) & 0x400)
        if candidate.is_symlink() or is_reparse:
            raise ValueError(f"report asset inventory contains a linked path: {candidate}")
        if stat.S_ISREG(candidate_stat.st_mode):
            observed_files.add(candidate.resolve())
        elif stat.S_ISDIR(candidate_stat.st_mode):
            observed_directories.add(candidate.resolve())
        else:
            raise ValueError(f"report asset inventory contains a special file: {candidate}")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("report asset inventory contains missing or unexpected paths")


def _report_tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def validate_report_assets(
    report_root: str | Path,
    *,
    expected_source_bindings: dict[str, dict[str, str]],
    fixture_smoke_only: bool,
) -> dict[str, Any]:
    """Revalidate a completed report-asset directory without modifying it."""

    root = Path(report_root)
    _safe_directory(root, label="report-asset directory")
    manifest_path = _safe_regular_file(root / FIGURE_MANIFEST_NAME, label="figure manifest")
    manifest = read_json_object(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("fixture_smoke_only") is not fixture_smoke_only
        or not same_json_value(manifest.get("source_bindings"), expected_source_bindings)
    ):
        raise ValueError("figure manifest identity, status, fixture, or source binding is invalid")
    register_path = _bound_report_file(
        root, manifest.get("result_register"), label="result register"
    )
    register = read_json_object(register_path)
    if (
        register.get("schema_version") != 1
        or register.get("status") != "complete"
        or register.get("fixture_smoke_only") is not fixture_smoke_only
        or not same_json_value(register.get("source_bindings"), expected_source_bindings)
    ):
        raise ValueError("result register identity, status, fixture, or source binding is invalid")
    sections = register.get("sections")
    figures = manifest.get("figures")
    if not isinstance(sections, dict) or set(sections) != set(EXPECTED_FIGURES):
        raise ValueError("result register section inventory is invalid")
    if not isinstance(figures, dict) or set(figures) != set(EXPECTED_FIGURES):
        raise ValueError("figure manifest inventory is invalid")
    observed_paths: set[Path] = {manifest_path, register_path}
    for name in EXPECTED_FIGURES:
        _validate_figure_binding(
            root,
            name=name,
            figure=figures[name],
            section=sections[name],
            observed_paths=observed_paths,
        )
    _validate_exact_report_inventory(root, observed_paths)
    return manifest


def _validated_existing_report_assets(output: Path, *, fixture_smoke_only: bool) -> dict[str, Any]:
    manifest = read_json_object(
        _safe_regular_file(output / FIGURE_MANIFEST_NAME, label="existing figure manifest")
    )
    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, dict) or any(
        not isinstance(label, str) or not isinstance(binding, dict)
        for label, binding in source_bindings.items()
    ):
        raise ValueError("existing report assets have invalid source bindings")
    return validate_report_assets(
        output,
        expected_source_bindings=source_bindings,
        fixture_smoke_only=fixture_smoke_only,
    )


def _publish_staged_report_assets(
    *,
    run: Path,
    staging: Path,
    output: Path,
    replace_existing: bool,
) -> None:
    if not replace_existing:
        if output.exists() or output.is_symlink():
            raise FileExistsError("report-asset directory appeared during atomic generation")
        staging.rename(output)
        return
    previous = run / ".report-assets-previous"
    if previous.exists() or previous.is_symlink():
        raise FileExistsError("a prior report-asset replacement requires bounded recovery")
    output.rename(previous)
    try:
        staging.rename(output)
    except Exception:
        if not output.exists() and previous.is_dir() and not previous.is_symlink():
            previous.rename(output)
        raise
    if (
        previous.is_symlink()
        or not previous.is_dir()
        or previous.resolve().parent != run.resolve()
        or previous.name != ".report-assets-previous"
    ):
        raise RuntimeError("report-asset replacement backup is not safe to remove")
    shutil.rmtree(previous)


def generate_report_assets(
    *,
    project_root: str | Path,
    run_root: str | Path,
    source_manifest: dict[str, Any],
    bundle_manifest: dict[str, Any],
    fixture_smoke_only: bool = False,
) -> dict[str, Any]:
    """Create the atomic result register and seven report-ready figure pairs."""

    if type(fixture_smoke_only) is not bool:
        raise ValueError("report fixture boundary must be an explicit boolean")
    project_path = Path(project_root)
    run_path = Path(run_root)
    _safe_directory(project_path, label="report project root")
    _safe_directory(run_path, label="report run root")
    project = project_path.resolve()
    run = run_path.resolve()
    if project not in run.parents:
        raise ValueError("report run root must be a real directory inside the project")
    data = _load_study_data(
        project=project,
        run_root=run,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        fixture_smoke_only=fixture_smoke_only,
    )
    output = run / REPORT_DIRECTORY_NAME
    replace_existing = False
    existing: dict[str, Any] | None = None
    existing_bindings_match = False
    if output.exists() or output.is_symlink():
        existing = _validated_existing_report_assets(output, fixture_smoke_only=fixture_smoke_only)
        existing_bindings_match = same_json_value(
            existing.get("source_bindings"), data.source_bindings
        )
        replace_existing = not existing_bindings_match
    staging = Path(tempfile.mkdtemp(prefix=".report-assets-staging-", dir=run))
    try:
        tables, table_bindings = _write_source_tables(staging, data)
        register = _result_register_payload(data, table_bindings)
        register_path = staging / RESULT_REGISTER_NAME
        write_json_atomic(register_path, register)
        figures = _render_figures(staging, data, tables, table_bindings)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "fixture_smoke_only": fixture_smoke_only,
            "rendering": {
                "render_device": "host_cpu",
                "upstream_model_compute": "strict_a100",
                "formats": ["svg", "png"],
                "png_dpi": 300,
            },
            "source_bindings": data.source_bindings,
            "result_register": {
                "path": register_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(register_path),
            },
            "figures": figures,
            "presentation_policy": {
                "descriptive_titles_only": True,
                "visible_text_scan": "passed",
                "figure_numbers_embedded": False,
            },
        }
        write_json_atomic(staging / FIGURE_MANIFEST_NAME, manifest)
        validate_report_assets(
            staging,
            expected_source_bindings=data.source_bindings,
            fixture_smoke_only=fixture_smoke_only,
        )
        if existing is not None and existing_bindings_match:
            if _report_tree_hashes(output) != _report_tree_hashes(staging):
                raise ValueError(
                    "existing report assets differ from deterministic source regeneration"
                )
            shutil.rmtree(staging)
            return existing
        _publish_staged_report_assets(
            run=run,
            staging=staging,
            output=output,
            replace_existing=replace_existing,
        )
    except Exception:
        if staging.is_dir() and not staging.is_symlink() and run in staging.resolve().parents:
            shutil.rmtree(staging)
        raise
    return validate_report_assets(
        output,
        expected_source_bindings=data.source_bindings,
        fixture_smoke_only=fixture_smoke_only,
    )
