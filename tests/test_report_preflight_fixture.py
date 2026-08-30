import ast
import importlib.util
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lora_audit.config import load_config
from lora_audit.execution_profiles import (
    canonical_execution_profile_path,
    load_execution_profile,
)
from lora_audit.fixture import run_fixture_smoke
from lora_audit.hashing import sha256_file
from lora_audit.preflight import (
    accelerator_snapshot,
    bf16_finite_tensor_probe,
    device_matches_requirements,
    empty_accelerator_cache,
    normalize_compute_capability,
    peak_allocated_memory_gib,
    peak_reserved_memory_gib,
    reset_peak_memory_stats,
    run_preflight,
)
from lora_audit.report import AuditReport, BranchEvidence, Decision, decide


def _load_script(project_root: Path, relative_path: str):
    path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    script_directory = str(path.parent)
    sys.path.insert(0, script_directory)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(script_directory)
    return module


def _execution_profile(project_root: Path, profile_id: str = "l4-24gb-v1"):
    path = canonical_execution_profile_path(project_root, profile_id)
    return load_execution_profile(path), path


def test_notebook_execution_cli_rejects_non_execution_arguments_before_side_effects(
    project_root: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_script(project_root, "scripts/execute_notebook.py")
    calls: list[Path] = []
    monkeypatch.setattr(module, "execute_fixture_notebook", calls.append)

    with pytest.raises(SystemExit) as help_exit:
        module.main(["--help"])
    help_output = capsys.readouterr()
    with pytest.raises(SystemExit) as invalid_exit:
        module.main(["--unknown-option"])
    invalid_output = capsys.readouterr()

    assert help_exit.value.code == 0
    assert "usage:" in help_output.out
    assert invalid_exit.value.code == 2
    assert "unrecognized arguments" in invalid_output.err
    assert calls == []


def test_notebook_execution_cli_runs_once_without_arguments(
    project_root: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_script(project_root, "scripts/execute_notebook.py")
    output = project_root / "notebooks/executed/experiment_runner.fixture.ipynb"
    calls: list[Path] = []

    def execute(root: Path) -> Path:
        calls.append(root)
        return output

    monkeypatch.setattr(module, "execute_manifest_bound_fixture_notebook", execute)

    assert module.main([]) == 0
    assert calls == [project_root]
    assert "notebooks/executed/experiment_runner.fixture.ipynb" in capsys.readouterr().out


def test_notebook_execution_stages_only_manifest_bound_files(project_root: Path) -> None:
    module = _load_script(project_root, "scripts/execute_notebook.py")

    with module.manifest_bound_project(project_root) as staged_project:
        staged_artifacts = staged_project / "artifacts"
        assert {path.name for path in staged_artifacts.iterdir()} == {
            "bundle-manifest.json",
            "source-manifest.json",
        }
        assert not (staged_project / "notebooks/executed").exists()


def test_notebook_execution_persists_output_only_after_staged_success(
    project_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script(project_root, "scripts/execute_notebook.py")
    active_project = tmp_path / "active" / "lora-audit"
    staged_project = tmp_path / "staged" / "lora-audit"
    staged_output = staged_project / "notebooks/executed/experiment_runner.fixture.ipynb"
    destination = active_project / "notebooks/executed/experiment_runner.fixture.ipynb"

    @contextmanager
    def staged(_project_root: Path):
        yield staged_project

    def execute(root: Path) -> Path:
        assert root == staged_project
        assert not destination.exists()
        staged_output.parent.mkdir(parents=True)
        staged_output.write_text("executed", encoding="utf-8")
        return staged_output

    monkeypatch.setattr(module, "manifest_bound_project", staged)
    monkeypatch.setattr(module, "execute_fixture_notebook", execute)

    assert module.execute_manifest_bound_fixture_notebook(active_project) == destination
    assert destination.read_text(encoding="utf-8") == "executed"


def test_notebook_execution_uses_isolated_current_interpreter_kernel(
    project_root: Path, monkeypatch
) -> None:
    module = _load_script(project_root, "scripts/execute_notebook.py")
    inherited = "inherited-kernel-data"
    monkeypatch.setenv("JUPYTER_PATH", inherited)

    with module.current_interpreter_kernel() as kernel_name:
        roots = os.environ["JUPYTER_PATH"].split(os.pathsep)
        kernel_spec = Path(roots[0]) / "kernels" / kernel_name / "kernel.json"
        payload = json.loads(kernel_spec.read_text(encoding="utf-8"))
        assert payload["argv"][0] == sys.executable
        assert payload["argv"][1:3] == ["-m", "ipykernel_launcher"]
        assert roots[1:] == [inherited]

    assert os.environ["JUPYTER_PATH"] == inherited


def test_notebook_execution_defers_output_until_after_attestation(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_script(project_root, "scripts/execute_notebook.py")
    isolated_project = tmp_path / "lora-audit"
    notebook_directory = isolated_project / "notebooks"
    notebook_directory.mkdir(parents=True)
    shutil.copy2(
        project_root / "notebooks/experiment_runner.ipynb",
        notebook_directory / "experiment_runner.ipynb",
    )
    output = notebook_directory / "executed/experiment_runner.fixture.ipynb"

    @contextmanager
    def isolated_kernel():
        yield "test-kernel"

    class Client:
        def __init__(self, notebook, **_kwargs):
            self.notebook = notebook

        def execute(self):
            assert not output.parent.exists()
            return self.notebook

    monkeypatch.setattr(module, "current_interpreter_kernel", isolated_kernel)
    monkeypatch.setattr(module, "NotebookClient", Client)

    assert module.execute_fixture_notebook(isolated_project) == output
    assert output.is_file()


def test_notebook_execution_reports_missing_client_only_when_execution_is_requested(
    project_root: Path, monkeypatch
) -> None:
    module = _load_script(project_root, "scripts/execute_notebook.py")
    monkeypatch.setattr(module, "NotebookClient", None)

    with pytest.raises(RuntimeError, match=r"requirements/dev\.lock"):
        module.execute_fixture_notebook(project_root)


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/generate_schemas.py",
        "scripts/validate_project.py",
        "scripts/audit_execution_readiness.py",
    ],
)
def test_no_argument_scripts_reject_non_execution_arguments_before_side_effects(
    project_root: Path,
    relative_path: str,
    monkeypatch,
    capsys,
) -> None:
    module = _load_script(project_root, relative_path)
    calls: list[tuple[object, ...]] = []

    class ExecutionReached(RuntimeError):
        pass

    def blocked_path(*args: object, **kwargs: object) -> None:
        calls.append(args)
        raise ExecutionReached

    monkeypatch.setattr(module, "Path", blocked_path)

    with pytest.raises(SystemExit) as help_exit:
        module.main(["--help"])
    help_output = capsys.readouterr()
    with pytest.raises(SystemExit) as invalid_exit:
        module.main(["--unknown-option"])
    invalid_output = capsys.readouterr()

    assert help_exit.value.code == 0
    assert "usage:" in help_output.out
    assert invalid_exit.value.code == 2
    assert "unrecognized arguments" in invalid_output.err
    assert calls == []

    with pytest.raises(ExecutionReached):
        module.main([])
    assert len(calls) == 1


def test_readiness_audit_uses_only_bundle_local_study_contracts(project_root: Path) -> None:
    source = (project_root / "scripts/audit_execution_readiness.py").read_text(encoding="utf-8")

    assert 'study_contract_path = root / "README.md"' in source
    assert 'protocol_config_path = root / "configs/core.yaml"' in source
    assert "objective_path" not in source
    assert "proposal_path" not in source
    assert ".codex" not in source


def test_readiness_subprocess_preserves_inherited_dependency_target(
    project_root: Path, monkeypatch
) -> None:
    module = _load_script(project_root, "scripts/audit_execution_readiness.py")
    captured: dict[str, object] = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("PYTHONPATH", "external-dependency-target")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module._run(project_root, "probe", [sys.executable, "-V"])

    expected = f"{project_root / 'src'}{module.os.pathsep}external-dependency-target"
    assert result["exit_code"] == 0
    assert captured["env"]["PYTHONPATH"] == expected


def test_readiness_local_workflow_uses_one_canonical_fixture_order(
    project_root: Path,
    monkeypatch,
) -> None:
    module = _load_script(project_root, "scripts/audit_execution_readiness.py")
    observed: list[tuple[str, list[str]]] = []

    def fake_run(_root, name, command):
        observed.append((name, command))
        return {"name": name, "command": command, "exit_code": 0}

    monkeypatch.setattr(module, "_run", fake_run)
    fixture_root = project_root / "artifacts/runs/fixture-smoke-currenthash"

    commands = module._run_local_commands(project_root, sys.executable, fixture_root)

    assert list(commands)[:6] == [
        "source_manifest",
        "notebook_fixture",
        "fixture_preflight",
        "fixture_smoke",
        "validator",
        "cohort_preview",
    ]
    assert [name for name, _command in observed[:6]] == list(commands)[:6]
    fixture_smoke = commands["fixture_smoke"]["command"]
    preview = commands["cohort_preview"]["command"]
    validator = commands["validator"]["command"]
    assert str(fixture_root) in fixture_smoke
    assert str(fixture_root / "data/manifest.jsonl") in preview
    assert "--execution-profile" in preview
    assert "configs/execution/a100-40gb.yaml" in preview
    assert "--bundle-manifest" in preview
    assert "artifacts/bundle-manifest.json" in preview
    assert validator[-1] == "--require-fixture-artifacts"
    assert "--no-cache" in commands["ruff"]["command"]
    assert commands["pytest"]["command"][-2:] == ["-p", "no:cacheprovider"]
    local_commands = [part for _name, command in observed for part in command]
    assert "--execute" not in local_commands
    assert "install" not in local_commands
    assert "uninstall" not in local_commands


@pytest.mark.parametrize(
    ("pair_count", "recommendation", "expected"),
    [
        (1, "pilot_incomplete", "unverified_external"),
        (3, "core_only", "unverified_external"),
        (3, "full_research", "passed"),
    ],
)
def test_readiness_requires_complete_full_research_pilot_authorization(
    project_root: Path,
    tmp_path: Path,
    monkeypatch,
    pair_count: int,
    recommendation: str,
    expected: str,
) -> None:
    module = _load_script(project_root, "scripts/audit_execution_readiness.py")
    monkeypatch.setattr(module, "_compute_contract_valid", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(module, "_pilot_binding_contract_valid", lambda _compute: True)
    monkeypatch.setattr(module, "verify_observed_compute_evidence", lambda **_kwargs: {})
    config_hash = "a" * 64
    compute = {
        "estimate_kind": "observed_pilot",
        "config_sha256": config_hash,
        "pilot_pair_count": pair_count,
        "pilot_evaluation_pair_count": pair_count,
        "pilot_failed_pair_count": 0,
        "pilot_threshold_failure_pair_count": 0,
        "pilot_integrity_failure_pair_count": 0,
        "pilot_qc_reason_codes": [],
        "required_representative_pilot_pair_count": 3,
        "recommendation": recommendation,
    }

    assert (
        module._observed_pilot_status(
            compute,
            tmp_path / "compute-estimate.json",
            config_hash,
            expected_source_tree_sha256="1" * 64,
            expected_source_manifest_sha256="2" * 64,
            execution_profile=SimpleNamespace(),
        )
        == expected
    )


def _copy_notebook_bundle(project_root: Path, destination: Path) -> Path:
    inner = destination / "TranThienNhan_TTTN/lora-audit"
    (inner / "notebooks").mkdir(parents=True)
    (inner / "configs/execution").mkdir(parents=True)
    shutil.copy2(
        project_root.parent / "RUN_EXPERIMENT.ipynb",
        inner.parent / "RUN_EXPERIMENT.ipynb",
    )
    shutil.copy2(
        project_root / "notebooks/experiment_runner.ipynb",
        inner / "notebooks/experiment_runner.ipynb",
    )
    shutil.copy2(
        project_root / "configs/execution/one-click-colab-pro.json",
        inner / "configs/execution/one-click-colab-pro.json",
    )
    return inner


@pytest.mark.parametrize(
    "relative",
    [
        "../RUN_EXPERIMENT.ipynb",
        "notebooks/experiment_runner.ipynb",
    ],
)
def test_project_validator_rejects_outputs_in_both_notebooks(
    project_root: Path,
    tmp_path: Path,
    relative: str,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / relative
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    code_cell["execution_count"] = 1
    code_cell["outputs"] = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": ["persisted output\n"],
        }
    ]
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"executed code cells|persisted outputs"):
        module._validate_notebooks(inner)


@pytest.mark.parametrize(
    "relative",
    [
        "../RUN_EXPERIMENT.ipynb",
        "notebooks/experiment_runner.ipynb",
    ],
)
def test_project_validator_requires_exact_professional_vietnamese_markdown(
    project_root: Path,
    tmp_path: Path,
    relative: str,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / relative
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "markdown")
    markdown_cell["source"] = ["# Extra notebook annotation\n"]
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="professional Vietnamese markdown"):
        module._validate_notebooks(inner)


@pytest.mark.parametrize(
    "relative",
    [
        "../RUN_EXPERIMENT.ipynb",
        "notebooks/experiment_runner.ipynb",
    ],
)
def test_project_validator_rejects_code_comments(
    project_root: Path,
    tmp_path: Path,
    relative: str,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / relative
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    code_cell["source"] = ["# Extra code annotation\n", *code_cell["source"]]
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="code comments"):
        module._validate_notebooks(inner)


def test_project_validator_ast_parses_one_click_outer_and_attestation_only_inner_notebook(
    project_root: Path,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")

    observed = module._validate_notebooks(project_root)

    assert observed["RUN_EXPERIMENT.ipynb"]["ast_parsed_code_cells"] == 40
    assert observed["lora-audit/notebooks/experiment_runner.ipynb"]["ast_parsed_code_cells"] == 2


@pytest.mark.parametrize(
    "notebook_path",
    [
        "../RUN_EXPERIMENT.ipynb",
        "notebooks/experiment_runner.ipynb",
    ],
)
def test_notebook_link_guard_rejects_a_windows_reparse_ancestor(
    project_root: Path,
    notebook_path: str,
) -> None:
    payload = json.loads((project_root / notebook_path).read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code" and "def has_linked_ancestor" in "".join(cell["source"])
    )
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "has_linked_ancestor"
    )
    namespace = {"Path": Path}
    helper_module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(helper_module), "link_guard", "exec"), namespace)
    has_linked_ancestor = namespace["has_linked_ancestor"]

    class FakePath:
        def __init__(self, parent: "FakePath | None", attributes: int = 0) -> None:
            self.parent = parent if parent is not None else self
            self.attributes = attributes

        def absolute(self) -> "FakePath":
            return self

        def is_symlink(self) -> bool:
            return False

        def stat(self, *, follow_symlinks: bool) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_file_attributes=self.attributes)

    root = FakePath(None)
    reparse_parent = FakePath(root, attributes=0x400)
    leaf = FakePath(reparse_parent)

    assert has_linked_ancestor(leaf) is True


def test_project_validator_rejects_inner_manual_execution_switch(
    project_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / "notebooks/experiment_runner.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    source = "".join(code_cell["source"]) + "\nRUN_PILOT = False\n"
    code_cell["source"] = source.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attestation-only"):
        module._validate_notebooks(inner)


def test_project_validator_rejects_disabled_outer_run_all_contract(
    project_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner.parent / "RUN_EXPERIMENT.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(
        cell for cell in payload["cells"] if "RUN_ALL_EXPERIMENTS = True" in "".join(cell["source"])
    )
    source = "".join(code_cell["source"]).replace(
        "RUN_ALL_EXPERIMENTS = True",
        "RUN_ALL_EXPERIMENTS = False",
    )
    code_cell["source"] = source.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-simple execution-switch write"):
        module._validate_notebooks(inner)


def test_project_validator_rejects_inner_preflight_command(
    project_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / "notebooks/experiment_runner.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    source = (
        "".join(code_cell["source"])
        + '\nsubprocess.run([sys.executable, str(BOOTSTRAP), "--mode", "preflight"], check=True)\n'
    )
    code_cell["source"] = source.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attestation-only"):
        module._validate_notebooks(inner)


def test_project_validator_rejects_inner_execute_capability(
    project_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / "notebooks/experiment_runner.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    source = "".join(code_cell["source"]) + '\ncapability = "--execute"\n'
    code_cell["source"] = source.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attestation-only"):
        module._validate_notebooks(inner)


def test_project_validator_rejects_inner_cohort_command(
    project_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / "notebooks/experiment_runner.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    source = (
        "".join(code_cell["source"])
        + '\nsubprocess.run([sys.executable, str(BOOTSTRAP), "--mode", "cohort"], check=True)\n'
    )
    code_cell["source"] = source.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attestation-only"):
        module._validate_notebooks(inner)


def test_project_validator_rejects_inner_interactive_input(
    project_root: Path,
    tmp_path: Path,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    inner = _copy_notebook_bundle(project_root, tmp_path)
    notebook_path = inner / "notebooks/experiment_runner.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    source = "".join(code_cell["source"]) + '\noperator_choice = input("continue?")\n'
    code_cell["source"] = source.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attestation-only"):
        module._validate_notebooks(inner)


def test_project_validator_scopes_fixture_evidence_to_an_explicit_flag(
    project_root: Path,
    monkeypatch,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    calls: list[object] = []

    def validated_fixture(*_args: object):
        calls.append("fixture")
        report = SimpleNamespace(
            decision=SimpleNamespace(value="no_signal_within_profile"),
            fixture_smoke_only=True,
        )
        return report, {"trigger_blind": 1.0}, 7

    monkeypatch.setattr(module, "_validate_fixture_artifacts", validated_fixture)
    pristine = module._fixture_artifact_fields(
        project_root,
        SimpleNamespace(),
        {},
        required=False,
    )

    assert calls == []
    assert pristine == {
        "fixture_artifacts_checked": False,
        "fixture_decision": None,
        "fixture_smoke_only": None,
        "fixture_behavior_coverage": None,
        "fixture_hashed_files": None,
    }
    observed = module._fixture_artifact_fields(
        project_root,
        SimpleNamespace(),
        {},
        required=True,
    )
    assert calls == ["fixture"]
    assert observed == {
        "fixture_artifacts_checked": True,
        "fixture_decision": "no_signal_within_profile",
        "fixture_smoke_only": True,
        "fixture_behavior_coverage": {"trigger_blind": 1.0},
        "fixture_hashed_files": 7,
    }


def test_project_validator_cli_reports_explicit_pristine_or_fixture_scope(
    project_root: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_script(project_root, "scripts/validate_project.py")
    fixture = SimpleNamespace()
    verified = SimpleNamespace(
        source_manifest={"entry_count": 71, "tree_sha256": "a" * 64},
        bundle_manifest={"entry_count": 73, "tree_sha256": "b" * 64},
    )
    fixture_calls: list[object] = []
    monkeypatch.setattr(
        module,
        "_validated_configs",
        lambda _root: ({"fixture": fixture}, {"fixture": 1, "core": 36, "target": 60}),
    )
    monkeypatch.setattr(module, "_validated_schemas", lambda _root: {})
    monkeypatch.setattr(module, "_validate_notebooks", lambda _root: {})
    monkeypatch.setattr(module, "verify_persisted_manifests", lambda _root: verified)
    monkeypatch.setattr(module, "scan_manifest_bound_content", lambda _verified: 73)

    def validated_fixture(*_args: object):
        fixture_calls.append("fixture")
        report = SimpleNamespace(
            decision=SimpleNamespace(value="no_signal_within_profile"),
            fixture_smoke_only=True,
        )
        return report, {"trigger_blind": 1.0}, 7

    monkeypatch.setattr(module, "_validate_fixture_artifacts", validated_fixture)

    assert module.main([]) == 0
    pristine = json.loads(capsys.readouterr().out)
    assert fixture_calls == []
    assert pristine["fixture_artifacts_checked"] is False
    assert pristine["fixture_decision"] is None
    assert pristine["fixture_behavior_coverage"] is None

    assert module.main(["--require-fixture-artifacts"]) == 0
    fixture_evidence = json.loads(capsys.readouterr().out)
    assert fixture_calls == ["fixture"]
    assert fixture_evidence["fixture_artifacts_checked"] is True
    assert fixture_evidence["fixture_smoke_only"] is True
    assert fixture_evidence["fixture_hashed_files"] == 7


@pytest.mark.parametrize(
    ("failures", "expected"),
    [
        ([], "passed"),
        (["accelerator_unavailable", "missing_package:peft"], "unverified_external"),
        (["insufficient_disk:required=25.00GiB observed=24.67GiB"], "unverified_external"),
        (["package_version_mismatch:numpy:expected=2:observed=1"], "failed"),
    ],
)
def test_local_preflight_separates_external_runtime_from_local_lock_failures(
    tmp_path: Path, monkeypatch, failures: list[str], expected: str
) -> None:
    project_root = tmp_path / "project"
    (project_root / "artifacts").mkdir(parents=True)
    module = _load_script(
        Path(__file__).resolve().parents[1],
        "scripts/audit_execution_readiness.py",
    )

    def fake_run(root, _name, _command):
        (root / "artifacts/preflight-local.json").write_text(
            json.dumps({"failures": failures}), encoding="utf-8"
        )
        return {"name": "strict_preflight_local", "exit_code": int(bool(failures))}

    monkeypatch.setattr(module, "_run", fake_run)

    _command, status = module._local_preflight_support(project_root, sys.executable)

    assert status == expected


def test_decision_policy_has_no_allow_label():
    branch = BranchEvidence(score=0.1, threshold=0.8, alert=False, available=True, evidence={})
    assert (
        decide(weight=branch, behavior=branch, ood_flags=[], trigger_blind_coverage=1.0)
        == Decision.NO_SIGNAL_WITHIN_PROFILE
    )
    assert "allow" not in {decision.value for decision in Decision}
    with pytest.raises(ValidationError):
        AuditReport.model_validate({"decision": "allow"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ((8, 0), [8, 0]),
        ([8, 0], [8, 0]),
        ({"major": 8, "minor": 0}, [8, 0]),
        ({"supported_dtypes": {"bf16"}}, []),
        ((8,), []),
        (("invalid", 0), []),
        (("8", 0), []),
        ((8.0, 0), []),
        ((True, 0), []),
        ((-1, 0), []),
    ],
)
def test_compute_capability_normalization_is_fail_closed(raw, expected):
    assert normalize_compute_capability(raw) == expected


@pytest.mark.parametrize("memory_gib", ["79", True, float("nan"), float("inf")])
def test_device_requirement_rejects_non_numeric_or_nonfinite_memory(
    project_root: Path, memory_gib: object
) -> None:
    runtime = load_config(project_root / "configs/core.yaml").runtime
    device = {"compute_capability": [8, 9], "total_memory_gib": memory_gib}

    assert device_matches_requirements(device, runtime) is False


def test_accelerator_snapshot_reads_backend_specific_capability(monkeypatch):
    accelerator = SimpleNamespace(
        is_available=lambda: True,
        current_accelerator=lambda: SimpleNamespace(type="test_backend"),
        device_count=lambda: 1,
    )
    device_module = SimpleNamespace(
        get_device_properties=lambda index: SimpleNamespace(
            name=f"test-device-{index}", total_memory=79 * 1024**3
        ),
        get_device_capability=lambda index: (8, index),
    )
    fake_framework = SimpleNamespace(
        __version__="2.13.0",
        accelerator=accelerator,
        version=SimpleNamespace(test_backend="1.0"),
        get_device_module=lambda backend: device_module,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_framework)

    snapshot = accelerator_snapshot()

    assert snapshot["accelerator_available"] is True
    assert snapshot["backend"] == "test_backend"
    assert snapshot["devices"][0]["compute_capability"] == [8, 0]
    assert snapshot["devices"][0]["total_memory_gib"] == 79.0


def test_bf16_probe_executes_tensor_math_and_rejects_nonfinite_output():
    tensor_calls: list[str] = []
    finite_result = [True]

    class FakeTensor:
        dtype = "torch.bfloat16"
        device = SimpleNamespace(type="cuda", index=0)

        def __mul__(self, _other):
            return self

        def __add__(self, _other):
            return self

    class FakeFiniteResult:
        def all(self):
            return self

        def item(self):
            return finite_result[0]

    def tensor(_values, *, dtype, device):
        assert dtype is framework.bfloat16
        tensor_calls.append(device)
        return FakeTensor()

    framework = SimpleNamespace(
        bfloat16=object(),
        tensor=tensor,
        isfinite=lambda _tensor: FakeFiniteResult(),
    )

    passed = bf16_finite_tensor_probe(framework, backend="cuda", device_index=0)
    finite_result[0] = False
    failed = bf16_finite_tensor_probe(framework, backend="cuda", device_index=0)

    assert passed["attempted"] is True
    assert passed["passed"] is True
    assert passed["finite"] is True
    assert tensor_calls == ["cuda:0", "cuda:0", "cuda:0", "cuda:0"]
    assert failed["passed"] is False
    assert failed["finite"] is False
    assert failed["error"] == "probe_output_contract_mismatch"


def test_peak_memory_uses_generic_accelerator_memory_namespace():
    reset_calls: list[bool] = []
    memory = SimpleNamespace(
        empty_cache=lambda: reset_calls.append(False),
        reset_peak_memory_stats=lambda: reset_calls.append(True),
        max_memory_allocated=lambda: 3 * 1024**3,
        max_memory_reserved=lambda: 4 * 1024**3,
    )
    framework = SimpleNamespace(
        accelerator=SimpleNamespace(is_available=lambda: True, memory=memory)
    )

    assert reset_peak_memory_stats(framework) is True
    assert peak_allocated_memory_gib(framework) == 3.0
    assert peak_reserved_memory_gib(framework) == 4.0
    assert empty_accelerator_cache(framework) is True
    assert reset_calls == [True, False]

    unavailable = SimpleNamespace(accelerator=SimpleNamespace(is_available=lambda: False))
    assert reset_peak_memory_stats(unavailable) is False
    assert peak_allocated_memory_gib(unavailable) is None
    assert peak_reserved_memory_gib(unavailable) is None
    assert empty_accelerator_cache(unavailable) is False


def test_strict_preflight_rejects_nonmatching_device(monkeypatch, project_root, tmp_path):
    execution_profile, execution_profile_path = _execution_profile(project_root)
    monkeypatch.setattr(
        "lora_audit.preflight.accelerator_snapshot",
        lambda: {
            "framework_available": True,
            "accelerator_available": True,
            "backend": "cuda",
            "devices": [
                {
                    "index": 0,
                    "name": "test-device",
                    "total_memory_gib": 22.0,
                    "compute_capability": [7, 5],
                }
            ],
        },
    )
    config_path = project_root / "configs/core.yaml"
    result = run_preflight(
        config=load_config(config_path),
        config_path=config_path,
        mode="strict",
        artifact_directory=tmp_path,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
    )
    assert result["status"] == "failed"
    assert any("required_device_not_observed" in failure for failure in result["failures"])


def test_fixture_preflight_remains_usable_without_execution_profile(
    monkeypatch, project_root, tmp_path
):
    monkeypatch.setattr(
        "lora_audit.preflight._package_observation",
        lambda _mode: ({}, []),
    )
    monkeypatch.setattr(
        "lora_audit.preflight.accelerator_snapshot",
        lambda: {
            "framework_available": False,
            "accelerator_available": False,
            "backend": None,
            "devices": [],
        },
    )
    config_path = project_root / "configs/fixture.yaml"

    result = run_preflight(
        config=load_config(config_path),
        config_path=config_path,
        mode="fixture",
        artifact_directory=tmp_path,
    )

    assert result["status"] == "passed"
    assert result["execution_profile"] is None
    assert "execution_profile_required" not in result["failures"]


def test_strict_preflight_rejects_bf16_probe_failure(monkeypatch, project_root, tmp_path):
    execution_profile, execution_profile_path = _execution_profile(project_root)
    monkeypatch.setattr(
        "lora_audit.preflight.accelerator_snapshot",
        lambda: {
            "framework_available": True,
            "accelerator_available": True,
            "backend": "cuda",
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA L4",
                    "total_memory_gib": 22.5,
                    "compute_capability": [8, 9],
                    "bf16_probe": {
                        "attempted": True,
                        "passed": False,
                        "finite": False,
                        "dtype": "torch.bfloat16",
                        "device_type": "cuda",
                        "device_index": 0,
                        "error": "nonfinite_tensor_output",
                    },
                }
            ],
        },
    )
    config_path = project_root / "configs/core.yaml"

    result = run_preflight(
        config=load_config(config_path),
        config_path=config_path,
        mode="strict",
        artifact_directory=tmp_path,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
    )

    assert result["status"] == "failed"
    assert any("required_device_not_observed" in failure for failure in result["failures"])


def test_strict_preflight_rejects_profile_byte_tamper(monkeypatch, project_root, tmp_path):
    execution_profile, source_path = _execution_profile(project_root)
    runtime_root = tmp_path / "runtime-project"
    config_path = runtime_root / "configs/core.yaml"
    profile_path = runtime_root / "configs/execution/l4-24gb.yaml"
    profile_path.parent.mkdir(parents=True)
    config_path.write_text("test-only: true\n", encoding="utf-8")
    profile_path.write_bytes(source_path.read_bytes() + b"# byte tamper\n")
    monkeypatch.setattr(
        "lora_audit.preflight.accelerator_snapshot",
        lambda: {
            "framework_available": False,
            "accelerator_available": False,
            "backend": None,
            "devices": [],
        },
    )

    result = run_preflight(
        config=load_config(project_root / "configs/core.yaml"),
        config_path=config_path,
        mode="strict",
        artifact_directory=runtime_root / "artifacts",
        execution_profile=execution_profile,
        execution_profile_path=profile_path,
    )

    assert result["execution_profile"] is None
    assert any(
        "execution_profile_invalid:execution profile file SHA-256 mismatch" in failure
        for failure in result["failures"]
    )


def test_strict_preflight_rejects_profile_object_file_mismatch(monkeypatch, project_root, tmp_path):
    a100, _a100_path = _execution_profile(project_root, "a100-40gb-v1")
    _l4, l4_path = _execution_profile(project_root)
    monkeypatch.setattr(
        "lora_audit.preflight.accelerator_snapshot",
        lambda: {
            "framework_available": False,
            "accelerator_available": False,
            "backend": None,
            "devices": [],
        },
    )
    config_path = project_root / "configs/core.yaml"

    result = run_preflight(
        config=load_config(config_path),
        config_path=config_path,
        mode="strict",
        artifact_directory=tmp_path,
        execution_profile=a100,
        execution_profile_path=l4_path,
    )

    assert result["execution_profile"] is None
    assert any(
        "execution profile object and file binding mismatch" in failure
        for failure in result["failures"]
    )


def test_strict_preflight_rejects_trigger_target_outside_vocabulary(
    monkeypatch, project_root, tmp_path
):
    runtime_root = tmp_path / "runtime-project"
    config_path = runtime_root / "configs/core.yaml"
    source_profile_path = canonical_execution_profile_path(project_root, "l4-24gb-v1")
    execution_profile_path = runtime_root / "configs/execution/l4-24gb.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("test-only: true\n", encoding="utf-8")
    execution_profile_path.parent.mkdir(parents=True)
    execution_profile_path.write_bytes(source_profile_path.read_bytes())
    execution_profile = load_execution_profile(execution_profile_path)
    (runtime_root / "registry.yaml").write_text(
        """private: true
triggers:
  - family_id: T1
    literal: '[T1]'
    placement: prefix
    target_intent: invalid_label
    harmless_intent_confirmed: true
  - family_id: T2
    literal: '[T2]'
    placement: prefix
    target_intent: invalid_label
    harmless_intent_confirmed: true
  - family_id: T3
    literal: '[T3]'
    placement: prefix
    target_intent: invalid_label
    harmless_intent_confirmed: true
""",
        encoding="utf-8",
    )
    config = load_config(project_root / "configs/core.yaml").model_copy(
        update={"private_trigger_registry": "registry.yaml"}
    )
    monkeypatch.setattr(
        "lora_audit.preflight.accelerator_snapshot",
        lambda: {
            "framework_available": True,
            "accelerator_available": True,
            "backend": "cuda",
            "devices": [
                {
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
                    },
                }
            ],
        },
    )

    result = run_preflight(
        config=config,
        config_path=config_path,
        mode="strict",
        artifact_directory=tmp_path / "artifacts",
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
    )

    assert any(
        "private_trigger_targets_outside_vocabulary" in failure for failure in result["failures"]
    )


def test_fixture_smoke_traverses_full_artifact_path(project_root, tmp_path):
    config = load_config(project_root / "configs/fixture.yaml")
    result = run_fixture_smoke(config, tmp_path / "run")
    assert result["status"] == "passed"
    assert result["fixture_smoke_only"] is True
    assert result["pairing"] == {"paired_lineages": 1, "adapters": 2}
    report = json.loads((tmp_path / "run/reports/audit-report.json").read_text(encoding="utf-8"))
    assert report["decision"] == "quarantine"
    assert report["fixture_smoke_only"] is True
    behavior = json.loads(Path(result["behavior_evidence"]).read_text(encoding="utf-8"))
    assert behavior["trigger_blind_coverage"] == 1.0
    assert behavior["structure_matched_coverage"] == 1.0
    assert behavior["trigger_aware_coverage"] == 1.0
    assert Path(result["failed_run_ledger"]).is_file()

    rerun = run_fixture_smoke(config, tmp_path / "run")
    hashes = json.loads(Path(rerun["hashes"]).read_text(encoding="utf-8"))
    assert "smoke-summary.json" in hashes
    assert "detectors/behavior-evidence.json" in hashes
    assert all(
        sha256_file(tmp_path / "run" / relative) == expected
        for relative, expected in hashes.items()
    )
