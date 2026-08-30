from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_source_manifest.py"
UPLOAD_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_upload_bundle.py"
TREE_HASH_CONTRACT = "ordered path, size_bytes, sha256, and private fields"
PRIVATE_LITERAL = "fixture-trigger-literal-never-public"
SOURCE_EXCLUDED_PATHS = [
    ".git/",
    ".venv/",
    "artifacts/",
    "notebooks/executed/",
    "cache and bytecode directories",
]
SOURCE_POST_SYNC_EXECUTION_SCOPE = (
    "strict preflight, receipt-gated pilot/cohort, and completed-cohort RQ1-RQ3"
)
SOURCE_ROLLBACK = (
    "remove only the newly created destination identifier after verifying that it is "
    "the intended synchronization target"
)
BUNDLE_EXECUTION_SCOPE = "staged smoke, preflight, pilot, cohort, and RQ1-RQ3 research"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source_root = str(PROJECT_ROOT / "src")
    sys.path.insert(0, source_root)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(source_root)
    return module


def _load_source_manifest_module():
    return _load_script(SOURCE_SCRIPT_PATH, "source_manifest")


def _load_upload_module():
    return _load_script(UPLOAD_SCRIPT_PATH, "upload_bundle")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _entry(path: str, payload: bytes, *, private: bool = False) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "private": private,
    }


def _tree_sha256(entries: list[dict[str, object]]) -> str:
    canonical = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _write_bytes(root: Path, relative: str, payload: bytes) -> None:
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_upload_fixture(
    tmp_path: Path,
    *,
    readme: str = "Protocol v2 supports Colab A100 CUDA GPU at /content/drive.\n",
    extra_source: tuple[str, bytes] | None = None,
) -> Path:
    root = tmp_path / "TranThienNhan_TTTN"
    sources = {
        "README.md": readme.encode(),
        "private/trigger_registry.yaml": (
            f"private: true\ntriggers:\n  - family_id: T1\n    literal: '{PRIVATE_LITERAL}'\n"
        ).encode(),
        "scripts/tool.py": b"VALUE = 1\n",
    }
    if extra_source is not None:
        sources[extra_source[0]] = extra_source[1]
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "stage0-attest",
                "metadata": {},
                "outputs": [],
                "source": ["print('attest')\n"],
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    outer = {
        "RUN_EXPERIMENT.ipynb": (
            json.dumps(notebook, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode(),
        "bootstrap.py": b"raise SystemExit('fixture only')\n",
    }
    for relative, payload in sources.items():
        _write_bytes(root / "lora-audit", relative, payload)
    for relative, payload in outer.items():
        _write_bytes(root, relative, payload)

    source_entries = sorted(
        (
            _entry(
                relative,
                payload,
                private=relative == "private/trigger_registry.yaml",
            )
            for relative, payload in sources.items()
        ),
        key=lambda item: str(item["path"]),
    )
    source_manifest: dict[str, object] = {
        "schema_version": 2,
        "destination_label": "lora-audit",
        "external_write_authorized": False,
        "external_write_performed": False,
        "entry_count": len(source_entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in source_entries),
        "tree_sha256": _tree_sha256(source_entries),
        "tree_hash_contract": TREE_HASH_CONTRACT,
        "private_paths": ["private/trigger_registry.yaml"],
        "excluded": SOURCE_EXCLUDED_PATHS,
        "post_sync_verifier": {
            "required_destination_label": "lora-audit",
            "required_entry_count": len(source_entries),
            "required_tree_sha256": "recompute from retrieved file bytes",
            "execution_scope": SOURCE_POST_SYNC_EXECUTION_SCOPE,
        },
        "rollback": SOURCE_ROLLBACK,
        "entries": source_entries,
    }
    source_manifest_path = root / "lora-audit/artifacts/source-manifest.json"
    _write_json(source_manifest_path, source_manifest)

    upload_module = _load_upload_module()
    bundle_entries = [{**entry, "path": f"lora-audit/{entry['path']}"} for entry in source_entries]
    bundle_entries.extend(
        _entry(
            relative,
            upload_module._manifest_bound_file_bytes(relative, root / relative),
        )
        for relative in outer
    )
    bundle_entries.sort(key=lambda item: str(item["path"]))
    bundle_manifest: dict[str, object] = {
        "schema_version": 1,
        "destination_label": "TranThienNhan_TTTN",
        "entry_count": len(bundle_entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in bundle_entries),
        "tree_sha256": _tree_sha256(bundle_entries),
        "tree_hash_contract": TREE_HASH_CONTRACT,
        "inner_source_entry_count": len(source_entries),
        "inner_source_tree_sha256": source_manifest["tree_sha256"],
        "execution_scope": BUNDLE_EXECUTION_SCOPE,
        "entries": bundle_entries,
    }
    _write_json(root / "lora-audit/artifacts/bundle-manifest.json", bundle_manifest)
    return root


def _write_saved_runtime_notebook_state(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cell = notebook["cells"][0]
    code_cell["execution_count"] = 1
    code_cell["metadata"] = {
        "colab": {"background_save": True, "base_uri": "https://localhost:8080/"},
        "executionInfo": {"elapsed": 10, "status": "error"},
        # Colab mirrors the top-level cell ID here when it autosaves execution state.
        "id": code_cell["id"],
        "outputId": "runtime-output",
    }
    code_cell["outputs"] = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": ["Mounted at /content/drive\n"],
        },
        {
            "ename": "RuntimeError",
            "evalue": "fixture failure",
            "output_type": "error",
            "traceback": ["/" + "tmp/ipykernel_1/cell.py"],
        },
    ]
    _write_json(notebook_path, notebook)


def test_source_snapshot_scope_is_bounded_without_hardcoded_count() -> None:
    module = _load_source_manifest_module()
    manifest = module.build_manifest(PROJECT_ROOT)
    entry_paths = {entry["path"] for entry in manifest["entries"]}

    assert manifest["destination_label"] == "lora-audit"
    assert manifest["external_write_authorized"] is False
    assert manifest["external_write_performed"] is False
    assert manifest["entry_count"] == len(entry_paths)
    assert manifest["post_sync_verifier"]["required_entry_count"] == len(entry_paths)
    assert "private/trigger_registry.yaml" in entry_paths
    assert "notebooks/experiment_runner.ipynb" in entry_paths
    assert "requirements/accelerated.lock" in entry_paths
    assert not any(path.startswith("artifacts/") for path in entry_paths)
    assert not any(path.startswith(".venv/") for path in entry_paths)
    assert not any(path.startswith("notebooks/executed/") for path in entry_paths)

    private_entries = [entry for entry in manifest["entries"] if entry["private"]]
    assert [entry["path"] for entry in private_entries] == ["private/trigger_registry.yaml"]
    assert set(private_entries[0]) == {
        "path",
        "size_bytes",
        "sha256",
        "private",
    }


def test_physical_upload_exactly_matches_allowlist_plus_manifests(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)

    verified = module.verify_upload_tree(root)

    assert verified.source_manifest["entry_count"] == 3
    assert verified.bundle_manifest["entry_count"] == 5
    assert len(verified.package_entries) == 7
    assert {entry["path"] for entry in verified.package_entries} == {
        "RUN_EXPERIMENT.ipynb",
        "bootstrap.py",
        "lora-audit/README.md",
        "lora-audit/private/trigger_registry.yaml",
        "lora-audit/scripts/tool.py",
        "lora-audit/artifacts/source-manifest.json",
        "lora-audit/artifacts/bundle-manifest.json",
    }


def test_saved_runtime_notebook_state_does_not_invalidate_attestation(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    _write_saved_runtime_notebook_state(notebook_path)

    verified = module.verify_upload_tree(root)

    assert verified.bundle_manifest["entry_count"] == 5
    assert module.scan_manifest_bound_content(verified) == 7


@pytest.mark.parametrize(
    "cell_colab_state",
    [
        # Colab stamps background_save onto every code cell once background execution
        # saves the hosted notebook, including cells that never produced output.
        {"background_save": True},
        {"background_save": True, "base_uri": "https://localhost:8080/"},
        {"height": 154, "referenced_widgets": ["eca23cdc0e304ec6a76cb3ab7c683532"]},
        # A later Colab release may add per-cell editor state we cannot enumerate now.
        {"output_embedded_package_id": "runtime-package"},
        {"resources": {"cached": True}},
        {},
    ],
)
def test_per_cell_colab_state_does_not_invalidate_attestation(
    tmp_path: Path, cell_colab_state: dict[str, object]
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["metadata"] = {"colab": cell_colab_state}
    _write_json(notebook_path, notebook)

    verified = module.verify_persisted_manifests(root)

    assert verified.bundle_manifest["entry_count"] == 5


def test_stable_cell_metadata_outside_colab_state_stays_bound(tmp_path: Path) -> None:
    # cellView drives what the operator sees, so it is source, not runtime state.
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["metadata"] = {"cellView": "form"}
    _write_json(notebook_path, notebook)

    with pytest.raises(ValueError, match=r"mismatch|do not match"):
        module.verify_persisted_manifests(root)


def test_cell_colab_metadata_must_be_an_object(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["metadata"] = {"colab": "not-an-object"}
    _write_json(notebook_path, notebook)

    with pytest.raises(ValueError, match="metadata colab must be an object"):
        module.verify_persisted_manifests(root)


@pytest.mark.parametrize(
    "visible_text",
    [
        "RQ3 result",
        "RG-03 passed",
        "H4 accepted",
        "R\x1b[31mQ3 colored output",
        "H\u200b4 hidden separator",
        "notebook v2",
        "schema_version",
        "Version",
        "__version__",
    ],
)
def test_saved_runtime_notebook_rejects_forbidden_visible_labels(
    tmp_path: Path, visible_text: str
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    _write_saved_runtime_notebook_state(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["outputs"][0]["text"] = [visible_text]
    _write_json(notebook_path, notebook)

    with pytest.raises(ValueError, match="forbidden presentation label"):
        module.verify_upload_tree(root)


def test_saved_runtime_notebook_ignores_html_heading_markup(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    _write_saved_runtime_notebook_state(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["outputs"].append(
        {
            "data": {"text/html": "<h3>Study summary</h3>"},
            "metadata": {},
            "output_type": "display_data",
        }
    )
    _write_json(notebook_path, notebook)

    verified = module.verify_upload_tree(root)

    assert verified.bundle_manifest["entry_count"] == 5


def test_saved_runtime_notebook_allows_ordinary_stage_numbers(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    _write_saved_runtime_notebook_state(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["outputs"][0]["text"] = ["Batch 1 complete; epoch 3 complete"]
    _write_json(notebook_path, notebook)

    verified = module.verify_upload_tree(root)

    assert verified.bundle_manifest["entry_count"] == 5


def test_saved_runtime_notebook_rejects_forbidden_svg_text(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    _write_saved_runtime_notebook_state(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["outputs"].append(
        {
            "data": {
                "image/svg+xml": (
                    '<svg xmlns="http://www.w3.org/2000/svg"><text>H4 result</text></svg>'
                )
            },
            "metadata": {},
            "output_type": "display_data",
        }
    )
    _write_json(notebook_path, notebook)

    with pytest.raises(ValueError, match="forbidden presentation label"):
        module.verify_upload_tree(root)


def test_divergent_cell_metadata_id_remains_rejected(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["metadata"]["id"] = "tampered-cell-id"
    _write_json(notebook_path, notebook)

    with pytest.raises(ValueError, match=r"RUN_EXPERIMENT\.ipynb"):
        module.verify_persisted_manifests(root)


def test_saved_runtime_notebook_source_tamper_remains_rejected(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = ["print('tampered')\n"]
    _write_json(notebook_path, notebook)

    with pytest.raises(ValueError, match=r"RUN_EXPERIMENT\.ipynb"):
        module.verify_persisted_manifests(root)


def test_saved_runtime_notebook_a100_metadata_tamper_remains_rejected(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    notebook_path = root / "RUN_EXPERIMENT.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["metadata"]["colab"]["gpuType"] = "L4"
    _write_json(notebook_path, notebook)

    with pytest.raises(ValueError, match=r"mismatch|do not match"):
        module.verify_persisted_manifests(root)


def test_refresh_bundle_manifest_rebinds_outer_entrypoints(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    updated_bootstrap = b"raise SystemExit('fresh outer entrypoint')\n"
    (root / "bootstrap.py").write_bytes(updated_bootstrap)

    refreshed = module.refresh_bundle_manifest(root)
    verified = module.verify_persisted_manifests(root)

    bootstrap_entry = next(
        entry for entry in refreshed["entries"] if entry["path"] == "bootstrap.py"
    )
    assert bootstrap_entry == _entry("bootstrap.py", updated_bootstrap)
    assert refreshed["inner_source_entry_count"] == 3
    assert refreshed["inner_source_tree_sha256"] == verified.source_manifest["tree_sha256"]
    assert refreshed["tree_sha256"] == verified.bundle_manifest["tree_sha256"]
    assert (
        json.loads((root / "lora-audit/artifacts/bundle-manifest.json").read_text(encoding="utf-8"))
        == refreshed
    )
    assert not list((root / "lora-audit/artifacts").glob(".bundle-manifest-*.tmp"))


@pytest.mark.parametrize(
    ("relative", "field", "value", "message"),
    [
        (
            "lora-audit/artifacts/source-manifest.json",
            "unrecognized_metadata",
            True,
            "top-level fields",
        ),
        (
            "lora-audit/artifacts/bundle-manifest.json",
            "unrecognized_metadata",
            True,
            "top-level fields",
        ),
        (
            "lora-audit/artifacts/bundle-manifest.json",
            "execution_scope",
            "tampered scope",
            "execution scope",
        ),
    ],
)
def test_persisted_manifests_reject_unbound_metadata(
    tmp_path: Path,
    relative: str,
    field: str,
    value: object,
    message: str,
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    path = root / Path(*relative.split("/"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)

    with pytest.raises(ValueError, match=message):
        module.verify_persisted_manifests(root)


def test_refresh_rejects_temporary_file_outside_validated_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    target = root / "lora-audit/artifacts/bundle-manifest.json"
    before = target.read_bytes()
    outside = tmp_path / "outside-bundle-manifest.tmp"

    class OutsideTemporary:
        def __enter__(self):
            self.handle = outside.open("wb")
            return self.handle

        def __exit__(self, exc_type, exc, traceback):
            self.handle.close()
            return False

    def fake_named_temporary_file(*_args, **kwargs):
        assert kwargs["dir"] == root / "lora-audit/artifacts"
        return OutsideTemporary()

    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", fake_named_temporary_file)
    try:
        with pytest.raises(FileNotFoundError, match="manifest entry missing"):
            module.refresh_bundle_manifest(root)
    finally:
        outside.unlink(missing_ok=True)

    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "relative",
    [
        "unexpected.txt",
        ".git/config",
        "run-output/result.json",
        "lora-audit/.venv/marker.txt",
        "lora-audit/__pycache__/tool.cpython-312.pyc",
        "lora-audit/notebooks/executed/runner.ipynb",
        "lora-audit/artifacts/runs/fixture-smoke-deadbeef/result.json",
    ],
)
def test_physical_upload_rejects_extras_and_generated_state(
    tmp_path: Path,
    relative: str,
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    _write_bytes(root, relative, b"unexpected\n")

    with pytest.raises(ValueError, match=r"refuses|mismatch"):
        module.verify_upload_tree(root)


@pytest.mark.parametrize(
    "relative",
    [
        "lora-audit/README.md",
        "bootstrap.py",
        "RUN_EXPERIMENT.ipynb",
    ],
)
def test_persisted_manifests_reject_inner_and_outer_byte_tamper(
    tmp_path: Path,
    relative: str,
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    path = root / Path(*relative.split("/"))
    path.write_bytes(path.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match=r"mismatch|do not match"):
        module.verify_persisted_manifests(root)


def test_bundle_manifest_rejects_tampered_inner_source_binding(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    path = root / "lora-audit/artifacts/bundle-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inner_source_tree_sha256"] = "b" * 64
    _write_json(path, payload)

    with pytest.raises(ValueError, match="inner source tree mismatch"):
        module.verify_persisted_manifests(root)


@pytest.mark.parametrize(
    "readme",
    [
        "credential = " + "hf_" + "A" * 24 + "\n",
        "local = " + "C:" + chr(92) + "Users" + chr(92) + "owner" + chr(92) + "repo\n",
        "local = " + "/" + "home/owner/repo\n",
    ],
)
def test_upload_rejects_secret_like_content_and_local_absolute_paths(
    tmp_path: Path,
    readme: str,
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path, readme=readme)

    with pytest.raises(ValueError, match=r"secret-like|local absolute"):
        module.verify_upload_tree(root)


def test_upload_path_scanner_accepts_escaped_vietnamese_reader_labels(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(
        tmp_path,
        readme=(
            'print(\\"  Nhật ký thực thi chi tiết:\\", path)\n'
            'print(\\"  So sánh các phương pháp phát hiện:\\", status)\n'
            'print(\\"  Kiểm tra cuối:\\", status)\n'
        ),
    )

    assert module.verify_upload_tree(root).bundle_manifest["entry_count"] > 0


def test_protocol_v2_accelerator_and_colab_wording_is_deliberately_permitted(
    tmp_path: Path,
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)

    assert module.verify_upload_tree(root).bundle_manifest["entry_count"] > 0


def test_upload_rejects_nested_archives_even_when_manifested(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(
        tmp_path,
        extra_source=("payload.zip", b"PK\x03\x04fixture"),
    )

    with pytest.raises(ValueError, match="nested archive"):
        module.verify_upload_tree(root)


def test_upload_rejects_symbolic_entries(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    link = root / "linked.txt"
    try:
        os.symlink(root / "bootstrap.py", link)
    except OSError:
        pytest.skip("symbolic links are not available in this Windows environment")

    with pytest.raises(ValueError, match=r"symbolic link|reparse point"):
        module.verify_upload_tree(root)


def test_verify_only_is_read_only_and_emits_no_local_path(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    before = {
        path.relative_to(root).as_posix(): _sha256_bytes(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        module,
        "build_archive",
        lambda *_args, **_kwargs: pytest.fail("verify-only attempted to build an archive"),
    )

    assert module.main(["--bundle-root", str(root), "--verify-only"]) == 0

    output = capsys.readouterr().out
    after = {
        path.relative_to(root).as_posix(): _sha256_bytes(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert str(tmp_path) not in output
    assert '"external_write_performed": false' in output


def test_deterministic_zip_build_and_extract_round_trip(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_receipt = module.build_archive(root, first)
    second_receipt = module.build_archive(root, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["archive_sha256"] == second_receipt["archive_sha256"]
    assert first_receipt["round_trip_verified"] is True
    assert PRIVATE_LITERAL not in json.dumps(first_receipt, sort_keys=True)
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert names
        assert all(name.startswith("TranThienNhan_TTTN/") for name in names)
        trigger = archive.read(
            "TranThienNhan_TTTN/lora-audit/private/trigger_registry.yaml"
        ).decode()
        assert PRIVATE_LITERAL in trigger


def test_archive_canonicalizes_saved_runtime_notebook_state(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    clean_archive = tmp_path / "clean.zip"
    runtime_archive = tmp_path / "runtime.zip"

    module.build_archive(root, clean_archive)
    _write_saved_runtime_notebook_state(root / "RUN_EXPERIMENT.ipynb")
    receipt = module.build_archive(root, runtime_archive)

    assert clean_archive.read_bytes() == runtime_archive.read_bytes()
    assert receipt["round_trip_verified"] is True
    with zipfile.ZipFile(runtime_archive) as archive:
        notebook = json.loads(archive.read("TranThienNhan_TTTN/RUN_EXPERIMENT.ipynb"))
    code_cell = notebook["cells"][0]
    assert code_cell["execution_count"] is None
    assert code_cell["outputs"] == []
    assert code_cell["metadata"] == {}


def test_archive_round_trip_rejects_traversal_member(tmp_path: Path) -> None:
    module = _load_upload_module()
    root = _build_upload_fixture(tmp_path)
    verified = module.verify_upload_tree(root)
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("TranThienNhan_TTTN/../escape.txt", b"escape")
    extraction = tmp_path / "extract"
    extraction.mkdir()

    with pytest.raises(ValueError, match="traversal"):
        module.verify_archive_round_trip(archive_path, verified, extraction)
