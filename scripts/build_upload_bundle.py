"""Verify and build the deterministic physical upload bundle.

The persisted source and bundle manifests are the only file allowlists.  The
two manifest files themselves are package metadata and are added explicitly;
no cache, run output, environment, archive, or unlisted path is inferred into
the upload.  The operator notebook is bound through a deterministic source
projection. Safe Colab execution counters and presentation-checked outputs do
not invalidate the attestation that they launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

BUNDLE_ROOT_NAME = "TranThienNhan_TTTN"
INNER_ROOT_NAME = "lora-audit"
SOURCE_MANIFEST_RELATIVE = "lora-audit/artifacts/source-manifest.json"
BUNDLE_MANIFEST_RELATIVE = "lora-audit/artifacts/bundle-manifest.json"
MANIFEST_RELATIVE_PATHS = (SOURCE_MANIFEST_RELATIVE, BUNDLE_MANIFEST_RELATIVE)
OUTER_ENTRYPOINTS = ("RUN_EXPERIMENT.ipynb", "bootstrap.py")
OUTPUT_MUTABLE_NOTEBOOK_ENTRYPOINT = "RUN_EXPERIMENT.ipynb"
TREE_HASH_CONTRACT = "ordered path, size_bytes, sha256, and private fields"
PRIVATE_TRIGGER_RELATIVE = "private/trigger_registry.yaml"
BUNDLE_EXECUTION_SCOPE = "staged smoke, preflight, pilot, cohort, and RQ1-RQ3 research"
SOURCE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "destination_label",
        "external_write_authorized",
        "external_write_performed",
        "entry_count",
        "total_size_bytes",
        "tree_sha256",
        "tree_hash_contract",
        "private_paths",
        "excluded",
        "post_sync_verifier",
        "rollback",
        "entries",
    }
)
BUNDLE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "destination_label",
        "entry_count",
        "total_size_bytes",
        "tree_sha256",
        "tree_hash_contract",
        "inner_source_entry_count",
        "inner_source_tree_sha256",
        "execution_scope",
        "entries",
    }
)
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
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_EXTERNAL_ATTRIBUTES = (stat.S_IFREG | 0o644) << 16
WINDOWS_REPARSE_POINT = 0x400
# Per-cell Colab state (background_save, base_uri, height, referenced_widgets, and
# any field a later Colab release adds) is editor and runtime bookkeeping that Colab
# rewrites on every autosave. It cannot alter what a bound cell executes, so it stays
# outside the source contract. The A100 identity lives in notebook-level metadata and
# remains bound through STABLE_NOTEBOOK_METADATA_FIELDS.
VOLATILE_CELL_METADATA_FIELDS = frozenset({"colab", "executionInfo", "outputId"})
STABLE_NOTEBOOK_METADATA_FIELDS = {
    "colab": ("gpuType",),
    "kernelspec": ("display_name", "name"),
    "language_info": ("name",),
}
PRESENTATION_INTERNAL_LABEL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:rq|rg|h)\s*[-_.]?\s*(\d+)",
    re.IGNORECASE,
)
PRESENTATION_REVISION_LABEL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])v\d+(?:\.\d+)*(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
PRESENTATION_FORMAT_LABEL_PATTERN = re.compile(
    r"schema[_ -]?version",
    re.IGNORECASE,
)
PRESENTATION_REVISION_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])version(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PRESENTATION_CONTROL_SEQUENCE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PRESENTATION_INVISIBLE_PATTERN = re.compile("[\u200b\u200c\u200d\ufeff]")
PRESENTATION_FORBIDDEN_PATTERNS = (
    PRESENTATION_INTERNAL_LABEL_PATTERN,
    PRESENTATION_REVISION_LABEL_PATTERN,
    PRESENTATION_FORMAT_LABEL_PATTERN,
    PRESENTATION_REVISION_WORD_PATTERN,
)

FORBIDDEN_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "run-output",
}
FORBIDDEN_FILE_SUFFIXES = (".pyc", ".pyo")
ARCHIVE_SUFFIXES = (
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tar.bz2",
    ".tar.gz",
    ".tar.xz",
    ".tgz",
    ".txz",
    ".xz",
    ".zip",
)
WINDOWS_RESERVED_NAMES = {
    "aux",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]{12,}"),
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<!\w)(?:[A-Za-z]:[\\/]|\\\\[A-Za-z0-9._-]+[\\/])")
POSIX_LOCAL_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|Users|tmp|workspace|mnt/[A-Za-z])/"
)
PRIVATE_LITERAL_LINE = re.compile(
    r"^\s*literal\s*:\s*(?P<quote>['\"]?)(?P<value>.+?)(?P=quote)\s*$",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class VerifiedBundle:
    """A byte-verified persisted manifest pair."""

    root: Path
    source_manifest: dict[str, Any]
    bundle_manifest: dict[str, Any]
    package_entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PhysicalTree:
    """Observed regular files and directories without following links."""

    files: dict[str, Path]
    directories: frozenset[str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _VisibleHtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        visible_attributes = {"alt", "aria-label", "title"}
        self.parts.extend(value for name, value in attrs if name in visible_attributes and value)


def _presentation_violation(text: str) -> bool:
    normalized = PRESENTATION_CONTROL_SEQUENCE_PATTERN.sub("", text)
    normalized = PRESENTATION_INVISIBLE_PATTERN.sub("", normalized)
    return any(pattern.search(normalized) for pattern in PRESENTATION_FORBIDDEN_PATTERNS)


def _visible_html_text(value: str) -> str:
    parser = _VisibleHtmlText()
    parser.feed(value)
    parser.close()
    return " ".join(parser.parts)


def _visible_svg_text(value: str, *, label: str) -> str:
    try:
        root = ET.fromstring(value)
    except ET.ParseError as error:
        raise ValueError(f"{label} contains malformed SVG output") from error
    visible_tags = {"text", "title", "desc"}
    parts: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] in visible_tags:
            parts.extend(part for part in element.itertext() if part)
    return " ".join(parts)


def assert_svg_presentation_safe(path: Path) -> None:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("report figure is not readable UTF-8 SVG") from error
    visible_text = _visible_svg_text(payload, label="report figure")
    if _presentation_violation(visible_text):
        raise ValueError("report figure contains a forbidden presentation label")


def _output_text(raw: Any, *, label: str) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and all(isinstance(part, str) for part in raw):
        return "".join(raw)
    raise ValueError(f"{label} must be a string or list of strings")


def _json_visible_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [str(key) for key in value]
        parts.extend(_json_visible_text(item) for item in value.values())
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_json_visible_text(item) for item in value)
    return ""


def _mime_visible_text(mime: str, value: Any, *, label: str) -> str:
    if mime == "text/html":
        return _visible_html_text(_output_text(value, label=label))
    if mime == "image/svg+xml":
        return _visible_svg_text(_output_text(value, label=label), label=label)
    if mime.startswith("image/") or mime.startswith("audio/") or mime.startswith("video/"):
        return ""
    return _json_visible_text(value)


def _notebook_output_visible_text(raw: Any, *, label: str) -> str:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    output_type = raw.get("output_type")
    if output_type == "stream":
        return _output_text(raw.get("text", ""), label=f"{label} text")
    if output_type == "error":
        parts = [raw.get("ename", ""), raw.get("evalue", "")]
        parts.append(_output_text(raw.get("traceback", []), label=f"{label} traceback"))
        return " ".join(part for part in parts if isinstance(part, str))
    if output_type not in {"display_data", "execute_result"}:
        raise ValueError(f"{label} has an unsupported output_type")
    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{label} data must be an object")
    return " ".join(
        _mime_visible_text(str(mime), value, label=f"{label} {mime}")
        for mime, value in data.items()
    )


def assert_notebook_presentation_safe(notebook: Any, *, label: str) -> None:
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{label} must contain a cell list")
    for cell_index, cell in enumerate(notebook["cells"]):
        cell_label = f"{label} cell {cell_index}"
        if not isinstance(cell, dict):
            raise ValueError(f"{cell_label} must be an object")
        source = _notebook_source_text(cell.get("source"), label=cell_label)
        if _presentation_violation(source):
            raise ValueError(f"{cell_label} contains a forbidden presentation label")
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            raise ValueError(f"{cell_label} outputs must be a list")
        for output_index, output in enumerate(outputs):
            output_label = f"{cell_label} output {output_index}"
            visible_text = _notebook_output_visible_text(output, label=output_label)
            if _presentation_violation(visible_text):
                raise ValueError(f"{output_label} contains a forbidden presentation label")


def _notebook_source_text(raw: Any, *, label: str) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and all(isinstance(part, str) for part in raw):
        return "".join(raw)
    raise ValueError(f"{label} source must be a string or list of strings")


def _selected_notebook_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("operator notebook metadata must be an object")
    selected: dict[str, Any] = {}
    if "accelerator" in raw:
        selected["accelerator"] = raw["accelerator"]
    for field, nested_fields in STABLE_NOTEBOOK_METADATA_FIELDS.items():
        if field not in raw:
            continue
        nested = raw[field]
        if not isinstance(nested, dict):
            raise ValueError(f"operator notebook metadata {field} must be an object")
        selected[field] = {name: nested[name] for name in nested_fields if name in nested}
    return selected


def _stable_cell_metadata(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} metadata must be an object")
    if "colab" in raw and not isinstance(raw["colab"], dict):
        raise ValueError(f"{label} metadata colab must be an object")
    return {key: value for key, value in raw.items() if key not in VOLATILE_CELL_METADATA_FIELDS}


def _notebook_cell_id(raw: dict[str, Any], *, label: str) -> dict[str, str]:
    if "id" not in raw:
        return {}
    cell_id = raw["id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError(f"{label} id must be a nonempty string")
    return {"id": cell_id}


def _notebook_cell_extras(
    raw: dict[str, Any],
    *,
    cell_type: str,
    label: str,
) -> dict[str, Any]:
    if cell_type == "code":
        if not isinstance(raw.get("outputs", []), list):
            raise ValueError(f"{label} outputs must be a list")
        execution_count = raw.get("execution_count")
        if execution_count is not None and type(execution_count) is not int:
            raise ValueError(f"{label} execution_count must be an integer or null")
        return {"execution_count": None, "outputs": []}
    if "attachments" not in raw:
        return {}
    attachments = raw["attachments"]
    if not isinstance(attachments, dict):
        raise ValueError(f"{label} attachments must be an object")
    return {"attachments": attachments}


def _notebook_cell_contract(raw: Any, *, index: int) -> dict[str, Any]:
    label = f"operator notebook cell {index}"
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    cell_type = raw.get("cell_type")
    if cell_type not in {"code", "markdown", "raw"}:
        raise ValueError(f"{label} has an unsupported cell_type")
    allowed = {"cell_type", "id", "metadata", "source"}
    allowed.update({"execution_count", "outputs"} if cell_type == "code" else {"attachments"})
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {unknown}")
    metadata = _stable_cell_metadata(raw.get("metadata", {}), label=label)
    # Colab mirrors the manifest-bound top-level cell ID into metadata.id when
    # it saves execution state.  Ignore only that exact duplicate; a divergent
    # metadata ID remains bound and therefore fails attestation.
    if raw.get("id") is not None and metadata.get("id") == raw["id"]:
        metadata.pop("id")
    contract = {
        "cell_type": cell_type,
        "metadata": metadata,
        "source": _notebook_source_text(raw.get("source"), label=label),
    }
    contract.update(_notebook_cell_id(raw, label=label))
    contract.update(_notebook_cell_extras(raw, cell_type=cell_type, label=label))
    return contract


def _notebook_contract_bytes(path: Path) -> bytes:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "operator notebook source contract mismatch: not valid UTF-8 JSON"
        ) from error
    if not isinstance(notebook, dict):
        raise ValueError("operator notebook must be a JSON object")
    assert_notebook_presentation_safe(notebook, label="operator notebook")
    required = {"cells", "metadata", "nbformat", "nbformat_minor"}
    if set(notebook) != required:
        raise ValueError("operator notebook has unknown or missing top-level fields")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("operator notebook cells must be a list")
    if type(notebook.get("nbformat")) is not int or notebook["nbformat"] != 4:
        raise ValueError("operator notebook nbformat must be 4")
    if type(notebook.get("nbformat_minor")) is not int or notebook["nbformat_minor"] < 0:
        raise ValueError("operator notebook nbformat_minor must be a nonnegative integer")
    contract = {
        "cells": [_notebook_cell_contract(cell, index=index) for index, cell in enumerate(cells)],
        "metadata": _selected_notebook_metadata(notebook["metadata"]),
        "nbformat": notebook["nbformat"],
        "nbformat_minor": notebook["nbformat_minor"],
    }
    return (json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _manifest_bound_file_bytes(relative: str, path: Path) -> bytes:
    if relative == OUTPUT_MUTABLE_NOTEBOOK_ENTRYPOINT:
        return _notebook_contract_bytes(path)
    return path.read_bytes()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{label} is not normalized POSIX: {value!r}")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"{label} must be relative: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{label} contains an empty or traversal component: {value!r}")
    for part in raw_parts:
        unsafe_character = ":" in part or any(ord(character) < 32 for character in part)
        if part.endswith((" ", ".")) or unsafe_character:
            raise ValueError(f"{label} is not cross-platform safe: {value!r}")
        if part.split(".", 1)[0].casefold() in WINDOWS_RESERVED_NAMES:
            raise ValueError(f"{label} uses a reserved path component: {value!r}")
    canonical = PurePosixPath(*raw_parts).as_posix()
    if canonical != value:
        raise ValueError(f"{label} is not canonical: {value!r}")
    return canonical


def _forbidden_reason(relative: str) -> str | None:
    lowered = relative.casefold()
    parts = lowered.split("/")
    if any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in parts):
        return "cache, environment, repository metadata, or run output"
    if lowered.endswith(FORBIDDEN_FILE_SUFFIXES):
        return "bytecode"
    if "/notebooks/executed/" in f"/{lowered}/":
        return "executed notebook"
    if any(lowered.endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
        return "nested archive"
    if lowered.startswith(f"{INNER_ROOT_NAME}/artifacts/") and relative not in {
        SOURCE_MANIFEST_RELATIVE,
        BUNDLE_MANIFEST_RELATIVE,
    }:
        return "unexpected generated artifact"
    if lowered.startswith("artifacts/"):
        return "inner generated artifact"
    return None


def _reject_forbidden_path(relative: str) -> None:
    reason = _forbidden_reason(relative)
    if reason is not None:
        raise ValueError(f"upload refuses {reason}: {relative}")


def _has_reparse_attribute(file_stat: os.stat_result) -> bool:
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return bool(attributes & WINDOWS_REPARSE_POINT)


def _assert_not_reparse(path: Path, relative: str) -> os.stat_result:
    file_stat = path.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or _has_reparse_attribute(file_stat):
        raise ValueError(f"upload refuses symbolic link or reparse point: {relative}")
    return file_stat


def _assert_regular_file(root: Path, relative: str) -> Path:
    canonical = _canonical_relative_path(relative, label="file path")
    cursor = root
    parts = canonical.split("/")
    for index, part in enumerate(parts):
        cursor = cursor / part
        if not cursor.exists() and not cursor.is_symlink():
            raise FileNotFoundError(f"manifest entry missing: {canonical}")
        file_stat = _assert_not_reparse(cursor, "/".join(parts[: index + 1]))
        if index < len(parts) - 1 and not stat.S_ISDIR(file_stat.st_mode):
            raise ValueError(f"manifest parent is not a directory: {canonical}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"manifest entry is not a regular file: {canonical}")
    return cursor


def _validate_entry(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"path", "size_bytes", "sha256", "private"}:
        raise ValueError(f"{label} must contain only path, size_bytes, sha256, and private")
    relative = _canonical_relative_path(raw.get("path"), label=f"{label} path")
    size = raw.get("size_bytes")
    if type(size) is not int or size < 0:
        raise ValueError(f"{label} has invalid size_bytes")
    if not _is_sha256(raw.get("sha256")):
        raise ValueError(f"{label} has invalid sha256")
    if type(raw.get("private")) is not bool:
        raise ValueError(f"{label} has invalid private marker")
    _reject_forbidden_path(relative)
    return {
        "path": relative,
        "size_bytes": size,
        "sha256": raw["sha256"],
        "private": raw["private"],
    }


def _validate_entries(raw_entries: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"{label} entries must be a nonempty list")
    entries = [
        _validate_entry(raw, label=f"{label} entry {index}")
        for index, raw in enumerate(raw_entries)
    ]
    paths = [entry["path"] for entry in entries]
    if paths != sorted(paths):
        raise ValueError(f"{label} entries are not in canonical path order")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} contains duplicate paths")
    folded = [path.casefold() for path in paths]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{label} contains case-fold path collisions")
    return entries


def _canonical_tree_sha256(entries: list[dict[str, Any]]) -> str:
    canonical = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _load_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_manifest_shape(
    payload: dict[str, Any],
    *,
    label: str,
    schema_version: int,
    destination_label: str,
    expected_fields: frozenset[str],
) -> list[dict[str, Any]]:
    if set(payload) != expected_fields:
        raise ValueError(f"{label} has unknown or missing top-level fields")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != schema_version
    ):
        raise ValueError(f"{label} schema_version mismatch")
    if payload.get("destination_label") != destination_label:
        raise ValueError(f"{label} destination_label mismatch")
    if payload.get("tree_hash_contract") != TREE_HASH_CONTRACT:
        raise ValueError(f"{label} tree_hash_contract mismatch")
    entries = _validate_entries(payload.get("entries"), label=label)
    expected_count = len(entries)
    expected_size = sum(entry["size_bytes"] for entry in entries)
    expected_tree = _canonical_tree_sha256(entries)
    if type(payload.get("entry_count")) is not int or payload.get("entry_count") != expected_count:
        raise ValueError(f"{label} entry_count mismatch")
    if (
        type(payload.get("total_size_bytes")) is not int
        or payload.get("total_size_bytes") != expected_size
    ):
        raise ValueError(f"{label} total_size_bytes mismatch")
    if payload.get("tree_sha256") != expected_tree:
        raise ValueError(f"{label} tree_sha256 mismatch")
    return entries


def _entry_for_file(relative: str, path: Path, *, private: bool) -> dict[str, Any]:
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "private": private,
    }


def _entry_for_outer_entrypoint(relative: str, path: Path) -> dict[str, Any]:
    if relative != OUTPUT_MUTABLE_NOTEBOOK_ENTRYPOINT:
        return _entry_for_file(relative, path, private=False)
    payload = _manifest_bound_file_bytes(relative, path)
    return {
        "path": relative,
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "private": False,
    }


def _verify_source_manifest(
    root: Path,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = _validate_manifest_shape(
        payload,
        label="source manifest",
        schema_version=2,
        destination_label=INNER_ROOT_NAME,
        expected_fields=SOURCE_MANIFEST_FIELDS,
    )
    if payload.get("external_write_authorized") is not False:
        raise ValueError("source manifest must not authorize an external write")
    if payload.get("external_write_performed") is not False:
        raise ValueError("source manifest must not claim an external write")
    private_paths = sorted(entry["path"] for entry in entries if entry["private"])
    if payload.get("private_paths") != private_paths:
        raise ValueError("source manifest private_paths mismatch")
    if payload.get("excluded") != SOURCE_EXCLUDED_PATHS:
        raise ValueError("source manifest excluded-path contract mismatch")
    if payload.get("rollback") != SOURCE_ROLLBACK:
        raise ValueError("source manifest rollback contract mismatch")
    private_trigger = next(
        (entry for entry in entries if entry["path"] == PRIVATE_TRIGGER_RELATIVE),
        None,
    )
    if private_trigger is None or private_trigger["private"] is not True:
        raise ValueError("source manifest does not bind the private trigger registry")
    verifier = payload.get("post_sync_verifier")
    expected_verifier = {
        "required_destination_label": INNER_ROOT_NAME,
        "required_entry_count": len(entries),
        "required_tree_sha256": "recompute from retrieved file bytes",
        "execution_scope": SOURCE_POST_SYNC_EXECUTION_SCOPE,
    }
    if verifier != expected_verifier:
        raise ValueError("source manifest post-sync count is not source-bound")

    observed: list[dict[str, Any]] = []
    for entry in entries:
        relative = f"{INNER_ROOT_NAME}/{entry['path']}"
        source = _assert_regular_file(root, relative)
        actual = _entry_for_file(entry["path"], source, private=entry["private"])
        if actual != entry:
            raise ValueError(f"source manifest byte mismatch: {entry['path']}")
        observed.append(actual)
    return observed


def _verify_bundle_manifest(
    root: Path,
    payload: dict[str, Any],
    source_manifest: dict[str, Any],
    source_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries = _validate_manifest_shape(
        payload,
        label="bundle manifest",
        schema_version=1,
        destination_label=BUNDLE_ROOT_NAME,
        expected_fields=BUNDLE_MANIFEST_FIELDS,
    )
    if type(payload.get("inner_source_entry_count")) is not int or payload.get(
        "inner_source_entry_count"
    ) != len(source_entries):
        raise ValueError("bundle manifest inner source count mismatch")
    if payload.get("inner_source_tree_sha256") != source_manifest.get("tree_sha256"):
        raise ValueError("bundle manifest inner source tree mismatch")
    if payload.get("execution_scope") != BUNDLE_EXECUTION_SCOPE:
        raise ValueError("bundle manifest execution scope mismatch")

    expected = [{**entry, "path": f"{INNER_ROOT_NAME}/{entry['path']}"} for entry in source_entries]
    for relative in OUTER_ENTRYPOINTS:
        source = _assert_regular_file(root, relative)
        expected.append(_entry_for_outer_entrypoint(relative, source))
    expected.sort(key=lambda entry: entry["path"])
    if entries != expected:
        persisted_by_path = {entry["path"]: entry for entry in entries}
        observed_by_path = {entry["path"]: entry for entry in expected}
        mismatched_paths = sorted(
            path
            for path in persisted_by_path.keys() | observed_by_path.keys()
            if persisted_by_path.get(path) != observed_by_path.get(path)
        )
        raise ValueError("bundle manifest entry mismatch: " + ", ".join(mismatched_paths))
    return entries


def _expected_package_entries(
    root: Path,
    bundle_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    entries = list(bundle_entries)
    for relative in MANIFEST_RELATIVE_PATHS:
        source = _assert_regular_file(root, relative)
        entries.append(_entry_for_file(relative, source, private=False))
    entries.sort(key=lambda entry: entry["path"])
    return tuple(entries)


def _validated_bundle_manifest_target(root: Path) -> tuple[Path, Path]:
    target = root / Path(*BUNDLE_MANIFEST_RELATIVE.split("/"))
    parent = target.parent
    for relative, directory in (
        (INNER_ROOT_NAME, root / INNER_ROOT_NAME),
        (f"{INNER_ROOT_NAME}/artifacts", parent),
    ):
        file_stat = _assert_not_reparse(directory, relative)
        if not stat.S_ISDIR(file_stat.st_mode):
            raise ValueError(f"bundle manifest parent is not a directory: {relative}")
    if target.exists() or target.is_symlink():
        file_stat = _assert_not_reparse(target, BUNDLE_MANIFEST_RELATIVE)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("bundle manifest target is not a regular file")
    return target, parent


def _write_bundle_manifest_atomically(
    root: Path,
    target: Path,
    parent: Path,
    payload: dict[str, Any],
    source_manifest: dict[str, Any],
    source_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=".bundle-manifest-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        _assert_regular_file(root, f"{INNER_ROOT_NAME}/artifacts/{temporary.name}")
        parent_stat = _assert_not_reparse(parent, f"{INNER_ROOT_NAME}/artifacts")
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("bundle manifest parent changed during refresh")
        temporary.replace(target)
    except BaseException:
        if temporary is not None:
            try:
                candidate = _assert_regular_file(
                    root,
                    f"{INNER_ROOT_NAME}/artifacts/{temporary.name}",
                )
            except (FileNotFoundError, ValueError):
                pass
            else:
                candidate.unlink()
        raise
    persisted_path = _assert_regular_file(root, BUNDLE_MANIFEST_RELATIVE)
    if persisted_path.read_bytes() != encoded:
        raise ValueError("bundle manifest changed while being refreshed")
    persisted = _load_manifest(persisted_path, "bundle manifest")
    _verify_bundle_manifest(root, persisted, source_manifest, source_entries)
    return persisted


def refresh_bundle_manifest(bundle_root: Path) -> dict[str, Any]:
    """Rebuild the outer manifest from an already verified source manifest.

    This is a local packaging operation, not a runtime bootstrap operation.
    The source manifest must already bind every inner source byte; this helper
    only adds the two outer entrypoints and writes the derived outer manifest.
    ``bootstrap.py`` deliberately never calls it, so a Colab runtime cannot
    bless changed upload bytes by regenerating a manifest in place.
    """

    unresolved_root = Path(os.path.abspath(bundle_root))
    if unresolved_root.name != BUNDLE_ROOT_NAME or not unresolved_root.is_dir():
        raise ValueError(f"bundle root must be an existing {BUNDLE_ROOT_NAME}/ directory")
    _assert_not_reparse(unresolved_root, BUNDLE_ROOT_NAME)
    root = unresolved_root.resolve()
    source_path = _assert_regular_file(root, SOURCE_MANIFEST_RELATIVE)
    source_manifest = _load_manifest(source_path, "source manifest")
    source_entries = _verify_source_manifest(root, source_manifest)

    entries = [{**entry, "path": f"{INNER_ROOT_NAME}/{entry['path']}"} for entry in source_entries]
    entries.extend(
        _entry_for_outer_entrypoint(relative, _assert_regular_file(root, relative))
        for relative in OUTER_ENTRYPOINTS
    )
    entries.sort(key=lambda entry: entry["path"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "destination_label": BUNDLE_ROOT_NAME,
        "entry_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "tree_sha256": _canonical_tree_sha256(entries),
        "tree_hash_contract": TREE_HASH_CONTRACT,
        "inner_source_entry_count": len(source_entries),
        "inner_source_tree_sha256": source_manifest["tree_sha256"],
        "execution_scope": BUNDLE_EXECUTION_SCOPE,
        "entries": entries,
    }
    target, parent = _validated_bundle_manifest_target(root)
    return _write_bundle_manifest_atomically(
        root,
        target,
        parent,
        payload,
        source_manifest,
        source_entries,
    )


def verify_persisted_manifests(bundle_root: Path) -> VerifiedBundle:
    """Verify both persisted manifests and every byte they bind.

    Extra working-tree artifacts are intentionally ignored here.  Use
    :func:`verify_upload_tree` for the stricter physical upload gate.
    """

    unresolved_root = Path(os.path.abspath(bundle_root))
    if unresolved_root.name != BUNDLE_ROOT_NAME or not unresolved_root.is_dir():
        raise ValueError(f"bundle root must be an existing {BUNDLE_ROOT_NAME}/ directory")
    _assert_not_reparse(unresolved_root, BUNDLE_ROOT_NAME)
    root = unresolved_root.resolve()
    source_path = _assert_regular_file(root, SOURCE_MANIFEST_RELATIVE)
    bundle_path = _assert_regular_file(root, BUNDLE_MANIFEST_RELATIVE)
    source_manifest = _load_manifest(source_path, "source manifest")
    source_entries = _verify_source_manifest(root, source_manifest)
    bundle_manifest = _load_manifest(bundle_path, "bundle manifest")
    bundle_entries = _verify_bundle_manifest(
        root,
        bundle_manifest,
        source_manifest,
        source_entries,
    )
    package_entries = _expected_package_entries(root, bundle_entries)
    return VerifiedBundle(
        root=root,
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        package_entries=package_entries,
    )


def _observe_physical_tree(root: Path) -> PhysicalTree:
    files: dict[str, Path] = {}
    directories: set[str] = set()
    folded: dict[str, str] = {}

    def record(relative: str) -> None:
        previous = folded.setdefault(relative.casefold(), relative)
        if previous != relative:
            raise ValueError(f"physical tree contains case-fold collision: {previous}, {relative}")

    def walk(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as iterator:
            for item in sorted(iterator, key=lambda entry: entry.name):
                relative = f"{prefix}/{item.name}" if prefix else item.name
                relative = _canonical_relative_path(relative, label="physical path")
                record(relative)
                file_stat = item.stat(follow_symlinks=False)
                if item.is_symlink() or _has_reparse_attribute(file_stat):
                    raise ValueError(f"upload refuses symbolic link or reparse point: {relative}")
                _reject_forbidden_path(relative)
                if stat.S_ISDIR(file_stat.st_mode):
                    directories.add(relative)
                    walk(Path(item.path), relative)
                elif stat.S_ISREG(file_stat.st_mode):
                    files[relative] = Path(item.path)
                else:
                    raise ValueError(f"upload refuses non-regular filesystem entry: {relative}")

    walk(root, "")
    return PhysicalTree(files=files, directories=frozenset(directories))


def _expected_directories(paths: Iterable[str]) -> frozenset[str]:
    directories: set[str] = set()
    for relative in paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return frozenset(directories)


def _scan_sensitive_bytes(relative: str, payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"upload refuses non-UTF-8 source bytes: {relative}") from error
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise ValueError(f"secret-like content found in upload file: {relative}")
    if WINDOWS_ABSOLUTE_PATH.search(text) or POSIX_LOCAL_ABSOLUTE_PATH.search(text):
        raise ValueError(f"local absolute path found in upload file: {relative}")


def scan_manifest_bound_content(verified: VerifiedBundle) -> int:
    """Scan all manifest-bound source projections for local leakage."""

    for entry in verified.package_entries:
        relative = entry["path"]
        payload = _manifest_bound_file_bytes(relative, verified.root / relative)
        _scan_sensitive_bytes(relative, payload)
    return len(verified.package_entries)


def verify_upload_tree(bundle_root: Path) -> VerifiedBundle:
    """Require the physical tree to equal the manifest allowlist plus both manifests."""

    verified = verify_persisted_manifests(bundle_root)
    observed = _observe_physical_tree(verified.root)
    expected_files = {entry["path"] for entry in verified.package_entries}
    observed_files = set(observed.files)
    missing = sorted(expected_files - observed_files)
    unexpected = sorted(observed_files - expected_files)
    if missing or unexpected:
        raise ValueError(
            f"physical upload file set mismatch: missing={missing}, unexpected={unexpected}"
        )
    expected_directories = _expected_directories(expected_files)
    if observed.directories != expected_directories:
        missing_directories = sorted(expected_directories - observed.directories)
        unexpected_directories = sorted(observed.directories - expected_directories)
        raise ValueError(
            "physical upload directory set mismatch: "
            f"missing={missing_directories}, unexpected={unexpected_directories}"
        )
    scan_manifest_bound_content(verified)
    return verified


def _archive_entry_name(relative: str) -> str:
    return f"{BUNDLE_ROOT_NAME}/{relative}"


def _assert_output_outside_bundle(output: Path, bundle_root: Path) -> Path:
    selected = output.resolve()
    root = bundle_root.resolve()
    try:
        selected.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("archive output must be outside the upload bundle")
    if selected.exists() or selected.is_symlink():
        raise FileExistsError(f"archive output already exists: {selected.name}")
    if selected.suffix.casefold() != ".zip":
        raise ValueError("archive output must use the .zip suffix")
    return selected


def _write_deterministic_archive(
    verified: VerifiedBundle,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"partial archive already exists: {partial.name}")
    try:
        with zipfile.ZipFile(
            partial,
            mode="x",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=True,
        ) as archive:
            for entry in verified.package_entries:
                relative = entry["path"]
                info = zipfile.ZipInfo(_archive_entry_name(relative), date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = ZIP_EXTERNAL_ATTRIBUTES
                info.internal_attr = 0
                info.extra = b""
                info.comment = b""
                payload = _manifest_bound_file_bytes(relative, verified.root / relative)
                archive.writestr(info, payload)
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _validate_archive_infos(
    infos: list[zipfile.ZipInfo],
    expected_paths: set[str],
) -> dict[str, zipfile.ZipInfo]:
    if any(info.is_dir() for info in infos):
        raise ValueError("archive must contain files only beneath its single root")
    indexed: dict[str, zipfile.ZipInfo] = {}
    folded: dict[str, str] = {}
    for info in infos:
        member = _canonical_relative_path(info.filename, label="archive member")
        parts = member.split("/")
        if len(parts) < 2 or parts[0] != BUNDLE_ROOT_NAME:
            raise ValueError(f"archive member is outside the required root: {member}")
        relative = "/".join(parts[1:])
        _reject_forbidden_path(relative)
        if member in indexed:
            raise ValueError(f"archive contains duplicate member: {member}")
        previous = folded.setdefault(member.casefold(), member)
        if previous != member:
            raise ValueError(f"archive contains case-fold collision: {previous}, {member}")
        if info.flag_bits & 0x1:
            raise ValueError(f"archive contains encrypted member: {member}")
        mode = info.external_attr >> 16
        if (
            info.date_time != ZIP_TIMESTAMP
            or info.compress_type != zipfile.ZIP_STORED
            or info.create_system != 3
            or mode != (stat.S_IFREG | 0o644)
            or info.extra
            or info.comment
        ):
            raise ValueError(f"archive member metadata is not deterministic: {member}")
        indexed[member] = info
    observed_paths = {"/".join(member.split("/")[1:]) for member in indexed}
    if observed_paths != expected_paths:
        raise ValueError("archive member set does not match the physical upload allowlist")
    return indexed


def _receipt(
    archive_path: Path,
    verified: VerifiedBundle,
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "archive_name": archive_path.name,
        "archive_sha256": _sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "root": BUNDLE_ROOT_NAME,
        "entry_count": len(verified.package_entries),
        "source_manifest_sha256": _sha256_file(verified.root / SOURCE_MANIFEST_RELATIVE),
        "source_tree_sha256": verified.source_manifest["tree_sha256"],
        "bundle_manifest_sha256": _sha256_file(verified.root / BUNDLE_MANIFEST_RELATIVE),
        "bundle_tree_sha256": verified.bundle_manifest["tree_sha256"],
        "private_entries_included": sum(
            entry["private"] is True for entry in verified.package_entries
        ),
        "round_trip_verified": True,
    }
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for entry in verified.package_entries:
        if not entry["private"]:
            continue
        text = (verified.root / entry["path"]).read_text(encoding="utf-8")
        for match in PRIVATE_LITERAL_LINE.finditer(text):
            literal = match.group("value").strip()
            if literal and literal in serialized:
                raise ValueError("public package receipt would disclose a private trigger literal")
    return receipt


def verify_archive_round_trip(
    archive_path: Path,
    verified: VerifiedBundle,
    extraction_parent: Path,
) -> dict[str, Any]:
    """Safely extract an archive and re-run the complete physical upload verifier."""

    if any(extraction_parent.iterdir()):
        raise ValueError("round-trip extraction directory must be empty")
    expected = {entry["path"]: entry for entry in verified.package_entries}
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        if archive.comment:
            raise ValueError("archive comment is not deterministic")
        indexed = _validate_archive_infos(archive.infolist(), set(expected))
        for relative, entry in expected.items():
            member = _archive_entry_name(relative)
            info = indexed[member]
            if info.file_size != entry["size_bytes"] or info.compress_size != entry["size_bytes"]:
                raise ValueError(f"archive size mismatch: {relative}")
            payload = archive.read(info)
            if len(payload) != entry["size_bytes"] or _sha256_bytes(payload) != entry["sha256"]:
                raise ValueError(f"archive byte mismatch: {relative}")
            destination = extraction_parent / BUNDLE_ROOT_NAME / Path(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    extracted = verify_upload_tree(extraction_parent / BUNDLE_ROOT_NAME)
    if extracted.package_entries != verified.package_entries:
        raise ValueError("round-trip extracted entry set changed")
    return _receipt(archive_path, verified)


def build_archive(bundle_root: Path, output: Path) -> dict[str, Any]:
    """Build, extract, and byte-verify one deterministic upload ZIP."""

    verified = verify_upload_tree(bundle_root)
    selected = _assert_output_outside_bundle(output, verified.root)
    _write_deterministic_archive(verified, selected)
    try:
        with tempfile.TemporaryDirectory(
            prefix="lora-audit-upload-roundtrip-",
            dir=selected.parent,
        ) as temporary:
            return verify_archive_round_trip(selected, verified, Path(temporary))
    except BaseException:
        if selected.exists():
            selected.unlink()
        raise


def _verification_receipt(verified: VerifiedBundle) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "passed",
        "mode": "verify-only",
        "root": BUNDLE_ROOT_NAME,
        "entry_count": len(verified.package_entries),
        "source_tree_sha256": verified.source_manifest["tree_sha256"],
        "bundle_tree_sha256": verified.bundle_manifest["tree_sha256"],
        "private_entries_included": sum(
            entry["private"] is True for entry in verified.package_entries
        ),
        "external_write_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh, verify, or deterministically package the physical upload bundle.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--refresh-bundle-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only and args.refresh_bundle_manifest:
            raise ValueError("--verify-only and --refresh-bundle-manifest are mutually exclusive")
        if args.refresh_bundle_manifest:
            if args.output is not None:
                raise ValueError("--refresh-bundle-manifest does not accept --output")
            manifest = refresh_bundle_manifest(args.bundle_root)
            result = {
                "schema_version": 1,
                "status": "passed",
                "mode": "refresh-bundle-manifest",
                "root": BUNDLE_ROOT_NAME,
                "entry_count": manifest["entry_count"],
                "source_tree_sha256": manifest["inner_source_tree_sha256"],
                "bundle_tree_sha256": manifest["tree_sha256"],
                "external_write_performed": False,
            }
        elif args.verify_only:
            if args.output is not None:
                raise ValueError("--verify-only does not accept --output")
            result = _verification_receipt(verify_upload_tree(args.bundle_root))
        else:
            output = args.output
            if output is None:
                output = args.bundle_root.resolve().parent / f"{BUNDLE_ROOT_NAME}.zip"
            result = build_archive(args.bundle_root, output)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(
            json.dumps(
                {"schema_version": 1, "status": "failed", "reason": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
