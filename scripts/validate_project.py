# ruff: noqa: E402
"""Deterministic project-level readiness validator."""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import tokenize
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import nbformat
from build_upload_bundle import (
    assert_notebook_presentation_safe,
    scan_manifest_bound_content,
    verify_persisted_manifests,
)
from jsonschema import Draft202012Validator

from lora_audit.cohort import plan_cohort
from lora_audit.config import ExperimentConfig, load_config
from lora_audit.hashing import sha256_file
from lora_audit.manifests import CohortEntry, verify_paired_lineages
from lora_audit.report import AuditReport, Decision

NOTEBOOK_TOPIC_TITLE = (
    "# NGHIÊN CỨU VÀ ĐÁNH GIÁ PHƯƠNG PHÁP PHÁT HIỆN BACKDOOR TRÊN LoRA ADAPTER "
    "KẾT HỢP ĐẶC TRƯNG TRỌNG SỐ VÀ HÀNH VI"
)
OUTER_NOTEBOOK_TOPIC_TITLE = NOTEBOOK_TOPIC_TITLE
OUTER_NOTEBOOK_MARKDOWN_IDS = (
    "protocol-overview",
    "section-source-and-runtime",
    "section-pilot-measurement",
    "section-core-experiments",
    "section-detector-comparison",
    "section-adaptive-robustness",
    "section-cross-language-transfer",
)
OUTER_NOTEBOOK_MARKDOWN_HEADINGS = (
    NOTEBOOK_TOPIC_TITLE,
    "## 1. Kiểm tra tính toàn vẹn và môi trường thực thi",
    "## 2. Thực nghiệm thí điểm và ước lượng tài nguyên",
    "## 3. Huấn luyện và đánh giá tập thí nghiệm chính",
    "## 4. So sánh các phương pháp phát hiện",
    "## 5. Đánh giá độ bền trước tối ưu thích nghi",
    "## 6. Đánh giá khả năng chuyển giao giữa tiếng Anh và tiếng Việt",
)
OUTER_NOTEBOOK_REQUIRED_PHRASES = (
    "trạng thái chưa chạy không được xem là bằng chứng thực nghiệm",
    "tập kiểm thử giữ kín",
    "đã vượt qua kiểm soát chất lượng",
)
INNER_NOTEBOOK_MARKDOWN_HEADINGS = (
    NOTEBOOK_TOPIC_TITLE,
    "## Vai trò của notebook",
    "## Ranh giới bằng chứng",
)
INNER_NOTEBOOK_REQUIRED_PHRASES = (
    "không thực hiện phép đo nghiên cứu",
    "không chứng minh môi trường A100",
)
LOCKED_COMPONENT_STAGE_MODES = {
    "core_experiment": "cohort",
    "detector_comparison": "detector-comparison",
    "adaptive_condition": "adaptive-robustness-condition",
    "adaptive_finalize": "adaptive-robustness-finalize",
    "language_transfer_cell": "cross-language-transfer-cell",
    "language_transfer_finalize": "cross-language-transfer-finalize",
    "final_verification": "final-verification",
}
NOTEBOOK_CONTRACTS = {
    "RUN_EXPERIMENT.ipynb": {
        "cells": 47,
        "code_cells": 40,
        "execution_switch_values": {"RUN_ALL_EXPERIMENTS": True},
        "execution_stage_guards": {
            "preflight": "RUN_ALL_EXPERIMENTS",
            "pilot": "RUN_ALL_EXPERIMENTS",
            "cohort": "RUN_ALL_EXPERIMENTS",
            "detector-comparison": "RUN_ALL_EXPERIMENTS",
            "adaptive-robustness-condition": "RUN_ALL_EXPERIMENTS",
            "adaptive-robustness-finalize": "RUN_ALL_EXPERIMENTS",
            "cross-language-transfer-cell": "RUN_ALL_EXPERIMENTS",
            "cross-language-transfer-finalize": "RUN_ALL_EXPERIMENTS",
            "final-verification": "RUN_ALL_EXPERIMENTS",
        },
    },
    "lora-audit/notebooks/experiment_runner.ipynb": {
        "cells": 3,
        "code_cells": 2,
        "execution_switch_values": {},
        "execution_stage_guards": {},
    },
}


def _is_expected_literal_bool(node: ast.AST | None, expected: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is expected


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _is_simple_switch_declaration(name: ast.Name, parent: ast.AST, *, expected: bool) -> bool:
    if isinstance(parent, ast.Assign):
        return (
            len(parent.targets) == 1
            and parent.targets[0] is name
            and _is_expected_literal_bool(parent.value, expected)
        )
    return (
        isinstance(parent, ast.AnnAssign)
        and parent.target is name
        and _is_expected_literal_bool(parent.value, expected)
    )


def _called_builtin_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _validate_switch_write_node(
    node: ast.AST,
    *,
    parents: Mapping[ast.AST, ast.AST],
    switch_values: Mapping[str, bool],
    declarations: dict[str, int],
    display_path: str,
) -> None:
    if (
        isinstance(node, ast.Name)
        and node.id in switch_values
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ):
        parent = parents[node]
        if not _is_simple_switch_declaration(
            node,
            parent,
            expected=switch_values[node.id],
        ):
            raise ValueError(f"{display_path} has a non-simple execution-switch write")
        declarations[node.id] += 1
        return
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
        if _literal_string(node.slice) in switch_values:
            raise ValueError(f"{display_path} has a dynamic execution-switch write")


def _validate_dynamic_switch_call(
    node: ast.AST,
    *,
    switch_values: Mapping[str, bool],
    display_path: str,
) -> None:
    if not isinstance(node, ast.Call):
        return
    builtin = _called_builtin_name(node)
    if builtin in {"eval", "exec", "globals", "locals", "vars"}:
        raise ValueError(f"{display_path} exposes dynamic execution-switch mutation")
    if builtin in {"delattr", "setattr"} and len(node.args) >= 2:
        if _literal_string(node.args[1]) in switch_values:
            raise ValueError(f"{display_path} has a dynamic execution-switch write")


def _validate_execution_switch_writes(
    parsed: list[ast.AST],
    *,
    display_path: str,
    execution_switch_values: Mapping[str, bool],
) -> None:
    declarations = dict.fromkeys(execution_switch_values, 0)
    for tree in parsed:
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            _validate_switch_write_node(
                node,
                parents=parents,
                switch_values=execution_switch_values,
                declarations=declarations,
                display_path=display_path,
            )
            _validate_dynamic_switch_call(
                node,
                switch_values=execution_switch_values,
                display_path=display_path,
            )
    invalid = [name for name, count in declarations.items() if count != 1]
    if invalid:
        raise ValueError(
            f"{display_path} must declare each execution switch exactly once "
            f"with its required literal boolean value: {invalid}"
        )


def _command_mode(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id == "run_locked_component" and node.args:
        component = _literal_string(node.args[0])
        if component in LOCKED_COMPONENT_STAGE_MODES:
            return LOCKED_COMPONENT_STAGE_MODES[component]
    for argument in node.args:
        if not isinstance(argument, (ast.List, ast.Tuple)):
            continue
        values = [_literal_string(item) for item in argument.elts]
        for index, value in enumerate(values[:-1]):
            if value == "--mode" and values[index + 1] in {
                "attest",
                "notebook-contract",
                "status",
                "preflight",
                "pilot",
                "cohort",
                "detector-comparison",
                "adaptive-robustness-condition",
                "adaptive-robustness-finalize",
                "cross-language-transfer-cell",
                "cross-language-transfer-finalize",
                "final-verification",
                "rq1",
                "rq2-condition",
                "rq2-finalize",
                "rq3-cell",
                "rq3-finalize",
                "verify-results",
                "research",
            }:
                return values[index + 1]
    return None


class _ExecutionCallGuardVisitor(ast.NodeVisitor):
    def __init__(self, stage_guards: Mapping[str, str], display_path: str) -> None:
        self.stage_guards = stage_guards
        self.display_path = display_path
        self.active_switches: list[str] = []

    def visit_If(self, node: ast.If) -> None:
        switch = node.test.id if isinstance(node.test, ast.Name) else None
        if switch not in self.stage_guards.values():
            self.generic_visit(node)
            return
        self.active_switches.append(switch)
        for child in node.body:
            self.visit(child)
        self.active_switches.pop()
        for child in node.orelse:
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:
        stage = _command_mode(node)
        if stage is not None and stage in self.stage_guards:
            expected_switch = self.stage_guards[stage]
            if expected_switch not in self.active_switches:
                raise ValueError(
                    f"{self.display_path} {stage} execution call is not guarded by "
                    f"{expected_switch}"
                )
        self.generic_visit(node)

    def _visit_new_scope(self, node: ast.AST) -> None:
        previous = self.active_switches
        self.active_switches = []
        self.generic_visit(node)
        self.active_switches = previous

    visit_FunctionDef = _visit_new_scope  # type: ignore[assignment]
    visit_AsyncFunctionDef = _visit_new_scope  # type: ignore[assignment]
    visit_ClassDef = _visit_new_scope  # type: ignore[assignment]
    visit_Lambda = _visit_new_scope  # type: ignore[assignment]


def _validate_execution_call_guards(
    parsed: list[ast.AST],
    *,
    display_path: str,
    stage_guards: Mapping[str, str],
) -> None:
    for tree in parsed:
        _ExecutionCallGuardVisitor(stage_guards, display_path).visit(tree)


def _validate_notebook_markdown(markdown_cells: list[Any], *, display_path: str) -> None:
    if display_path == "RUN_EXPERIMENT.ipynb":
        markdown_ids = tuple(str(cell.get("id", "")) for cell in markdown_cells)
        markdown_text = "\n".join(str(cell.source) for cell in markdown_cells)
        markdown_headings = tuple(str(cell.source).splitlines()[0] for cell in markdown_cells)
        if markdown_ids != OUTER_NOTEBOOK_MARKDOWN_IDS:
            raise ValueError(f"{display_path} professional markdown layout changed")
        if markdown_headings != OUTER_NOTEBOOK_MARKDOWN_HEADINGS or any(
            phrase not in markdown_text for phrase in OUTER_NOTEBOOK_REQUIRED_PHRASES
        ):
            raise ValueError(f"{display_path} professional Vietnamese markdown changed")
        if any(
            label in str(cell.source) for cell in markdown_cells for label in ("NO-GO", "NO-OP")
        ):
            raise ValueError(f"{display_path} contains an operational verdict label in markdown")
        return
    if len(markdown_cells) != 1:
        raise ValueError(f"{display_path} professional Vietnamese markdown changed")
    markdown_text = str(markdown_cells[0].source)
    markdown_headings = tuple(line for line in markdown_text.splitlines() if line.startswith("#"))
    if markdown_headings != INNER_NOTEBOOK_MARKDOWN_HEADINGS or any(
        phrase not in markdown_text for phrase in INNER_NOTEBOOK_REQUIRED_PHRASES
    ):
        raise ValueError(f"{display_path} professional Vietnamese markdown changed")


def _validate_notebook_file(
    notebook_path: Path,
    *,
    display_path: str,
    expected_cells: int,
    expected_code_cells: int,
    execution_switch_values: Mapping[str, bool],
    execution_stage_guards: Mapping[str, str],
) -> tuple[dict[str, int], list[str]]:
    notebook = nbformat.read(notebook_path, as_version=4)
    assert_notebook_presentation_safe(notebook, label=display_path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    if len(notebook.cells) != expected_cells or len(code_cells) != expected_code_cells:
        raise ValueError(f"{display_path} cell topology changed")
    _validate_notebook_markdown(markdown_cells, display_path=display_path)
    if len(code_cells) + len(markdown_cells) != len(notebook.cells):
        raise ValueError(f"{display_path} contains an unsupported non-code cell")
    if any(cell.get("execution_count") is not None for cell in code_cells):
        raise ValueError(f"{display_path} contains executed code cells")
    if any(cell.get("outputs") for cell in code_cells):
        raise ValueError(f"{display_path} contains persisted outputs")
    if any(not str(cell.source).strip() for cell in code_cells):
        raise ValueError(f"{display_path} contains an empty code cell")

    parsed: list[ast.AST] = []
    sources: list[str] = []
    for index, cell in enumerate(code_cells):
        source = str(cell.source)
        sources.append(source)
        if any(
            token.type == tokenize.COMMENT
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
        ):
            raise ValueError(f"{display_path} contains code comments in cell {index}")
        try:
            parsed.append(ast.parse(source, filename=f"{display_path}:code-cell-{index}"))
        except SyntaxError as error:
            message = f"{display_path} contains invalid Python in code cell {index}"
            raise ValueError(message) from error
    _validate_execution_switch_writes(
        parsed,
        display_path=display_path,
        execution_switch_values=execution_switch_values,
    )
    _validate_execution_call_guards(
        parsed,
        display_path=display_path,
        stage_guards=execution_stage_guards,
    )
    return (
        {
            "cells": len(notebook.cells),
            "code_cells": len(code_cells),
            "ast_parsed_code_cells": len(parsed),
        },
        sources,
    )


def _require_token_order(source: str, *, label: str, tokens: tuple[str, ...]) -> None:
    positions = [source.find(token) for token in tokens]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        observed = dict(zip(tokens, positions, strict=True))
        raise ValueError(f"{label} stage order invalid: {observed}")


def _validate_inner_notebook_contract(joined: str) -> None:
    _require_token_order(
        joined,
        label="inner notebook",
        tokens=(
            "PROJECT_ROOT",
            "Đã xác thực tính toàn vẹn gói tải lên",
        ),
    )
    observed_modes = {
        mode
        for node in ast.walk(ast.parse(joined))
        if isinstance(node, ast.Call) and (mode := _command_mode(node)) is not None
    }
    if observed_modes != {"attest"}:
        raise ValueError("inner notebook must remain attestation-only")
    required_tokens = (
        "sys.dont_write_bytecode = True",
        "has_linked_ancestor",
        "st_file_attributes",
        "0x400",
        'print("Chế độ: chỉ kiểm tra tính toàn vẹn gói tải lên")',
        "Đã xác thực tính toàn vẹn gói tải lên",
    )
    missing_tokens = [token for token in required_tokens if token not in joined]
    if missing_tokens:
        raise ValueError(f"inner notebook attestation-only contract incomplete: {missing_tokens}")
    forbidden_tokens = (
        "RUN_STRICT_PREFLIGHT",
        "RUN_PILOT",
        "RUN_PILOT_FORM",
        "LORA_AUDIT_RUN_PILOT",
        "LORA_AUDIT_EXECUTION_MODE",
        "RUN_FULL",
        "input(",
        "getpass",
        "ipywidgets",
        '"--execute"',
        "--max-lineages",
        "--available-device-hours",
        "run-cohort",
        "run-adaptive",
        "run-rq3",
        "build_source_manifest.py",
        "write_execution_receipt",
    )
    if any(token in joined for token in forbidden_tokens):
        raise ValueError("inner notebook must remain attestation-only")
    if "L4" in joined:
        raise ValueError("inner notebook contains stale L4 runtime wording")


def _locked_component_call_counts(joined: str) -> dict[str, int]:
    counts = dict.fromkeys(LOCKED_COMPONENT_STAGE_MODES, 0)
    for node in ast.walk(ast.parse(joined)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_locked_component"
            and node.args
        ):
            component = _literal_string(node.args[0])
            if component in counts:
                counts[component] += 1
    return counts


def _a100_profile_binding_count(joined: str) -> int:
    count = 0
    for node in ast.walk(ast.parse(joined)):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        for flag, value in zip(node.elts, node.elts[1:], strict=False):
            if (
                _literal_string(flag) == "--execution-profile"
                and isinstance(value, ast.Name)
                and value.id == "A100_EXECUTION_PROFILE"
            ):
                count += 1
    return count


def _validate_outer_notebook_contract(outer_root: Path, joined: str) -> None:
    _require_token_order(
        joined,
        label="outer notebook",
        tokens=(
            '"--mode", "attest"',
            '"--mode", "notebook-contract"',
            'label="Kiểm tra môi trường A100"',
            'label="Thực nghiệm thí điểm và đo tài nguyên"',
            'label=f"Lô thí nghiệm chính 1/',
            'label="So sánh phương pháp phát hiện trên tập kiểm thử giữ kín"',
            'label="Điều kiện tối ưu thích nghi 01/12"',
            'label="Tổng hợp phân tích độ bền trước tối ưu thích nghi"',
            'label="Chuyển giao ngôn ngữ: tiếng Anh sang tiếng Anh"',
            'label="Tổng hợp phân tích chuyển giao giữa hai ngôn ngữ"',
            'label="Kiểm tra cuối và hoàn tất gói kết quả"',
        ),
    )
    required_tokens = (
        "RUN_ALL_EXPERIMENTS = True",
        'NOTEBOOK_CONTRACT.get("deterministic_runtime_decisions_verified") is not True',
        'A100_EXECUTION_PROFILE = NOTEBOOK_CONTRACT["execution_profile"]',
        "sys.dont_write_bytecode = True",
        "has_linked_ancestor",
        "st_file_attributes",
        "0x400",
        '"--mode", "attest"',
        '"--mode", "notebook-contract"',
        '"--mode", "status"',
        '"--execution-profile"',
        'AVAILABLE_A100_DEVICE_HOURS = float(NOTEBOOK_CONTRACT["available_device_hours"])',
        'COHORT_BATCH_SIZE = int(NOTEBOOK_CONTRACT["cohort_batch_size"])',
        'COHORT_BATCH_INVOCATIONS = int(NOTEBOOK_CONTRACT["cohort_batch_invocations"])',
        'RUNTIME_DECISION_POLICY = NOTEBOOK_CONTRACT.get("runtime_decision_policy", {})',
        '"controller": "deterministic_protocol"',
        '"human_in_run_decisions": False',
        '"model_role": "measurement_only"',
        '"manual_override_allowed": False',
        'NOTEBOOK_CONTRACT.get("cross_language_trigger_gate_verified") is not True',
        "Quy trình dừng trước khi thiết lập GPU vì điều kiện kiểm tra chuyển giao",
        "ngôn ngữ không hợp lệ",
        '"core_experiment": "cohort"',
        '"detector_comparison": "detector-comparison"',
        '"adaptive_condition": "adaptive-robustness-condition"',
        '"adaptive_finalize": "adaptive-robustness-finalize"',
        '"language_transfer_cell": "cross-language-transfer-cell"',
        '"language_transfer_finalize": "cross-language-transfer-finalize"',
        '"final_verification": "final-verification"',
        'STUDY_STAGE_MODES = NOTEBOOK_CONTRACT.get("study_stage_modes", {})',
        "def presentation_safe_text(value: str) -> str:",
        "def presentation_diagnostic_tail(stdout: str, stderr: str) -> str:",
        "def persist_notebook_stage_log(",
        "def assert_report_figure_safe(path: Path) -> None:",
        "def read_study_status() -> dict:",
        "def display_study_progress() -> None:",
        'print("\\nTIẾN ĐỘ THỰC NGHIỆM")',
        'print("  Kiểm tra môi trường:", environment_status)',
        '"  So sánh các phương pháp phát hiện:"',
        '"  Kiểm tra cuối:"',
        'print(f"[{display_label}] HOÀN TẤT")',
        'print("  Nhật ký thực thi chi tiết:"',
        "def run_locked_component(",
        "assert_report_figure_safe(candidate)",
        '"--cohort-batch-size"',
    )
    missing_tokens = [token for token in required_tokens if token not in joined]
    if missing_tokens:
        raise ValueError(f"outer notebook A100 contract incomplete: {missing_tokens}")
    if "L4" in joined or "AVAILABLE_L4_DEVICE_HOURS" in joined:
        raise ValueError("outer notebook contains stale L4 runtime wording")
    interactive_tokens = (
        "input(",
        "getpass",
        "ipywidgets",
        "RUN_STRICT_PREFLIGHT",
        "RUN_PILOT",
        "owner_waived_human_review",
    )
    if any(token in joined for token in interactive_tokens):
        raise ValueError("outer notebook contains an interactive human decision path")
    forbidden_presentation_tokens = (
        "no-go",
        "no-op",
        "starting cohort invocation",
        "print(completed.stdout",
        "print(completed.stderr",
        "rq1",
        "rq2",
        "rq3",
        "rg3",
        "h3",
        "h4",
        'print(f"completed:',
        "detailed execution log:",
        "study progress",
        "detector comparison and held-out evaluation",
        "adaptive robustness condition",
        "cross-language transfer:",
        "final study verification and release",
    )
    if any(label in joined.casefold() for label in forbidden_presentation_tokens):
        raise ValueError("outer notebook contains an unsafe presentation token")
    if "--max-lineages" in joined or '"--mode", "research"' in joined or "EXECUTE_" in joined:
        raise ValueError("outer notebook contains a stale monolithic execution path")
    expected_component_counts = {
        "core_experiment": 9,
        "detector_comparison": 1,
        "adaptive_condition": 12,
        "adaptive_finalize": 1,
        "language_transfer_cell": 4,
        "language_transfer_finalize": 1,
        "final_verification": 1,
    }
    observed_component_counts = _locked_component_call_counts(joined)
    if observed_component_counts != expected_component_counts:
        raise ValueError(
            f"outer notebook one-click component counts changed: {observed_component_counts}"
        )
    if _a100_profile_binding_count(joined) != 3:
        raise ValueError("outer notebook must bind the A100 profile in every stage launcher")
    plan_path = outer_root / "lora-audit" / "configs" / "execution" / "one-click-colab-pro.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_plan = {
        "schema_version": 4,
        "plan_id": "colab-pro-one-click-v4",
        "enabled": True,
        "protocol_amendment": {
            "path": "configs/protocol-amendment-v4.json",
            "amendment_id": "post-pilot-outcome-classification-amendment-v4",
            "sha256": "26e982e70c4498dd9dafb06e50eeac8d85dc89c868ddd82059b774f76ce77350",
        },
        "runtime_decision_policy": {
            "policy_id": "non-human-runtime-decisions-v1",
            "controller": "deterministic_protocol",
            "human_in_run_decisions": False,
            "model_role": "measurement_only",
            "unexpected_state_action": "fail_closed",
            "semantic_trigger_drift_action": "reject",
            "manual_override_allowed": False,
            "platform_authorization": "external_precondition",
        },
        "execution_profile": "configs/execution/a100-40gb.yaml",
        "execution_profile_id": "a100-40gb-v1",
        "available_device_hours": 12.0,
        "maximum_managed_runtime_hours": 12.0,
        "representative_pilot": {
            "pair_count": 3,
            "maximum_adapter_runs": 6,
            "selection_policy": "outcome_blind_max_factor_coverage_v1",
            "execution_policy": "sequential_with_classified_qc_and_optimistic_lower_bound_stop",
            "continuation_gate": "outcome_classified_optimistic_full_research_v2",
            "threshold_failure_action": "retain_and_continue_if_full_pipeline_fits",
            "integrity_failure_action": "stop_protocol_review",
        },
        "compute_gate": {
            "scope": "full_research_pipeline",
            "budget_fraction": 0.8,
            "managed_runtime_reserve_fraction": 0.2,
            "rq2_adapter_runs": 12,
            "rq2_evaluation_units": 12,
            "rq3_evaluation_units": 24,
            "required_recommendation": "full_research",
        },
        "cohort_batch_size": 4,
        "cohort_batch_invocations": 9,
        "rq2_condition_indices": list(range(12)),
        "rq3_cells": [
            ["en-US", "en-US"],
            ["en-US", "vi-VN"],
            ["vi-VN", "vi-VN"],
            ["vi-VN", "en-US"],
        ],
        "rq3_trigger_gate": {
            "gate_id": "rq3-trigger-literal-preservation-v1",
            "disposition": "algorithmic_literal_preservation",
            "authority": "project_owner",
            "authority_basis": "standing_scope_authorization_no_runtime_review",
            "authorization_date": "2026-08-03",
            "bound_registry_sha256": (
                "ec04db3043151ea8eb745b51fac560428dd2dd0395ad4a63b6ac5f2c0efe7565"
            ),
            "bound_core_config_sha256": (
                "8c66009e872e418fc20ae6510278d2a4a66c46936d40ca6d5f4be724c1997543"
            ),
            "scope": {
                "locales": ["en-US", "vi-VN"],
                "target_intent": "weather_query",
                "harmless_use_only": True,
                "equivalence_modes": {
                    family_id: {
                        "en-US": "literal_preserved",
                        "vi-VN": "literal_preserved",
                    }
                    for family_id in ("T1", "T2", "T3")
                },
            },
        },
        "final_verifier_mode": "verify-results",
    }
    if plan != expected_plan:
        raise ValueError("outer notebook one-click execution plan changed")
    outer_notebook = nbformat.read(outer_root / "RUN_EXPERIMENT.ipynb", as_version=4)
    colab = outer_notebook.metadata.get("colab")
    if not isinstance(colab, dict) or colab.get("gpuType") != "A100":
        raise ValueError("outer notebook Colab metadata must request A100")


def _validate_notebooks(root: Path) -> dict[str, dict[str, int]]:
    outer_root = root.parent
    outer_stats, outer_sources = _validate_notebook_file(
        outer_root / "RUN_EXPERIMENT.ipynb",
        display_path="RUN_EXPERIMENT.ipynb",
        expected_cells=NOTEBOOK_CONTRACTS["RUN_EXPERIMENT.ipynb"]["cells"],
        expected_code_cells=NOTEBOOK_CONTRACTS["RUN_EXPERIMENT.ipynb"]["code_cells"],
        execution_switch_values=NOTEBOOK_CONTRACTS["RUN_EXPERIMENT.ipynb"][
            "execution_switch_values"
        ],
        execution_stage_guards=NOTEBOOK_CONTRACTS["RUN_EXPERIMENT.ipynb"]["execution_stage_guards"],
    )
    inner_relative = "lora-audit/notebooks/experiment_runner.ipynb"
    inner_stats, inner_sources = _validate_notebook_file(
        root / "notebooks/experiment_runner.ipynb",
        display_path=inner_relative,
        expected_cells=NOTEBOOK_CONTRACTS[inner_relative]["cells"],
        expected_code_cells=NOTEBOOK_CONTRACTS[inner_relative]["code_cells"],
        execution_switch_values=NOTEBOOK_CONTRACTS[inner_relative]["execution_switch_values"],
        execution_stage_guards=NOTEBOOK_CONTRACTS[inner_relative]["execution_stage_guards"],
    )
    _validate_inner_notebook_contract("\n".join(inner_sources))
    _validate_outer_notebook_contract(outer_root, "\n".join(outer_sources))
    return {
        "RUN_EXPERIMENT.ipynb": outer_stats,
        inner_relative: inner_stats,
    }


def _validated_configs(root: Path) -> tuple[dict[str, ExperimentConfig], dict[str, int]]:
    configs = {
        name: load_config(root / f"configs/{name}.yaml") for name in ("fixture", "core", "target")
    }
    counts = {name: len(plan_cohort(config)) for name, config in configs.items()}
    if counts != {"fixture": 1, "core": 36, "target": 60}:
        raise ValueError(f"cohort counts changed: {counts}")
    return configs, counts


def _validated_schemas(root: Path) -> dict[str, dict]:
    schemas = {
        name: json.loads((root / "schemas" / name).read_text(encoding="utf-8"))
        for name in ("audit-report.schema.json", "cohort-manifest.schema.json")
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    if "allow" in json.dumps(schemas) or "allow" in {decision.value for decision in Decision}:
        raise ValueError("forbidden allow decision found")
    return schemas


def _validate_fixture_artifacts(
    root: Path,
    fixture: ExperimentConfig,
    schemas: dict[str, dict],
) -> tuple[AuditReport, dict[str, float], int]:
    smoke_root = root / f"artifacts/runs/fixture-smoke-{fixture.config_hash[:12]}"
    if not smoke_root.is_dir():
        raise FileNotFoundError(
            f"current-config fixture artifact missing; run smoke into {smoke_root}"
        )
    summary = json.loads((smoke_root / "smoke-summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "passed" or summary.get("fixture_smoke_only") is not True:
        raise ValueError("canonical fixture smoke summary is not a passing fixture-only run")
    if summary.get("pairing") != {"paired_lineages": 1, "adapters": 2}:
        raise ValueError(f"canonical fixture pairing changed: {summary.get('pairing')}")

    report = AuditReport.model_validate_json(
        (smoke_root / "reports/audit-report.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schemas["audit-report.schema.json"]).validate(
        report.model_dump(mode="json")
    )
    entries = [
        CohortEntry.model_validate_json(line)
        for line in (smoke_root / "cohort/cohort.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verify_paired_lineages(entries)
    cohort_validator = Draft202012Validator(schemas["cohort-manifest.schema.json"])
    for entry in entries:
        cohort_validator.validate(entry.model_dump(mode="json"))

    behavior = json.loads(
        (smoke_root / "detectors/behavior-evidence.json").read_text(encoding="utf-8")
    )
    expected_coverage = {
        "trigger_blind": 1.0,
        "structure_matched": 1.0,
        "trigger_aware": 1.0,
    }
    observed_coverage = {name: float(behavior[f"{name}_coverage"]) for name in expected_coverage}
    if observed_coverage != expected_coverage:
        raise ValueError(f"canonical fixture behavior coverage incomplete: {observed_coverage}")
    if report.coverage.model_dump() != expected_coverage:
        raise ValueError(
            f"fixture report coverage differs from behavior evidence: {report.coverage}"
        )

    hash_manifest = json.loads((smoke_root / "hashes.json").read_text(encoding="utf-8"))
    current_hashed_paths = {
        path.relative_to(smoke_root).as_posix()
        for path in smoke_root.rglob("*")
        if path.is_file() and path.name != "hashes.json"
    }
    if set(hash_manifest) != current_hashed_paths:
        raise ValueError("fixture artifact hash manifest path set is stale or incomplete")
    mismatched_hashes = [
        relative
        for relative, expected_hash in hash_manifest.items()
        if sha256_file(smoke_root / relative) != expected_hash
    ]
    if mismatched_hashes:
        raise ValueError(f"fixture artifact hash mismatch: {mismatched_hashes}")
    return report, observed_coverage, len(hash_manifest)


def _fixture_artifact_fields(
    root: Path,
    fixture: ExperimentConfig,
    schemas: dict[str, dict],
    *,
    required: bool,
) -> dict[str, object]:
    if not required:
        return {
            "fixture_artifacts_checked": False,
            "fixture_decision": None,
            "fixture_smoke_only": None,
            "fixture_behavior_coverage": None,
            "fixture_hashed_files": None,
        }
    report, observed_coverage, fixture_hashed_files = _validate_fixture_artifacts(
        root,
        fixture,
        schemas,
    )
    return {
        "fixture_artifacts_checked": True,
        "fixture_decision": report.decision.value,
        "fixture_smoke_only": report.fixture_smoke_only,
        "fixture_behavior_coverage": observed_coverage,
        "fixture_hashed_files": fixture_hashed_files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate local project readiness invariants.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--require-fixture-artifacts",
        action="store_true",
        help=(
            "Require and hash-validate generated canonical fixture smoke evidence. "
            "Leave disabled for a pristine upload-tree validation."
        ),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    configs, counts = _validated_configs(root)
    schemas = _validated_schemas(root)
    fixture_fields = _fixture_artifact_fields(
        root,
        configs["fixture"],
        schemas,
        required=args.require_fixture_artifacts,
    )
    verified_bundle = verify_persisted_manifests(root.parent)
    sensitive_scanned_files = scan_manifest_bound_content(verified_bundle)

    result = {
        "status": "passed",
        "cohort_counts": counts,
        "notebooks": _validate_notebooks(root),
        "sensitive_scanned_files": sensitive_scanned_files,
        "source_entry_count": verified_bundle.source_manifest["entry_count"],
        "source_tree_sha256": verified_bundle.source_manifest["tree_sha256"],
        "bundle_entry_count": verified_bundle.bundle_manifest["entry_count"],
        "bundle_tree_sha256": verified_bundle.bundle_manifest["tree_sha256"],
        **fixture_fields,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
