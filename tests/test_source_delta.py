from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS / "build_source_delta.py"
UPLOAD_SCRIPT_PATH = SCRIPTS / "build_upload_bundle.py"


def _load_delta_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("source_delta", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def _load_upload_module():
    name = "upload_bundle_delta_precondition"
    spec = importlib.util.spec_from_file_location(name, UPLOAD_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_project(root: Path, *, marker: str, include_extra: bool) -> None:
    private = root / "private"
    private.mkdir(parents=True)
    (private / "trigger_registry.yaml").write_text(marker + "\n", encoding="utf-8")
    (root / "README.md").write_text(f"snapshot {marker}\n", encoding="utf-8")
    if include_extra:
        tests = root / "tests"
        tests.mkdir()
        (tests / "test_extra.py").write_text("def test_extra(): pass\n", encoding="utf-8")


def test_source_delta_is_hash_bound_and_no_write(tmp_path: Path) -> None:
    module = _load_delta_module()
    baseline_root = tmp_path / "baseline"
    current_root = tmp_path / "current"
    _write_minimal_project(baseline_root, marker="before", include_extra=False)
    _write_minimal_project(current_root, marker="after", include_extra=True)

    baseline = module.build_manifest(baseline_root)
    current = module.build_manifest(current_root)
    current_manifest_path = tmp_path / "current-manifest.json"
    current_manifest_path.write_text(json.dumps(current) + "\n", encoding="utf-8")
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "destination_label": "lora-audit",
                "folder_id": "bounded-folder",
                "verified_tree_sha256": baseline["tree_sha256"],
                "verification": {"status": "passed"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    delta = module.build_delta_manifest(
        current_root,
        current_manifest_path=current_manifest_path,
        receipt_path=receipt_path,
        baseline_root=baseline_root,
    )
    paths = {entry["path"] for entry in delta["operations"]}

    assert delta["external_write_authorized"] is False
    assert delta["external_write_performed"] is False
    assert delta["baseline_tree_sha256"] == baseline["tree_sha256"]
    assert delta["target_tree_sha256"] == current["tree_sha256"]
    assert delta["operation_counts"] == {"add": 1, "update": 2, "delete": 0}
    assert paths == {
        "README.md",
        "private/trigger_registry.yaml",
        "tests/test_extra.py",
    }


def _manifest_entry(path: str) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": 1,
        "sha256": "a" * 64,
        "private": False,
    }


@pytest.mark.parametrize(
    "entries",
    [
        [_manifest_entry("README.md"), _manifest_entry("README.md")],
        [_manifest_entry("README.md"), _manifest_entry("readme.md")],
        [_manifest_entry("../escape.py")],
        [_manifest_entry("/absolute.py")],
        [_manifest_entry("C" + chr(58) + "/local.py")],
    ],
)
def test_upload_precondition_rejects_ambiguous_or_unsafe_delta_paths(
    entries: list[dict[str, object]],
) -> None:
    module = _load_upload_module()
    entries.sort(key=lambda item: str(item["path"]))

    with pytest.raises(ValueError, match=r"duplicate|case-fold|traversal|relative"):
        module._validate_entries(entries, label="source manifest")
