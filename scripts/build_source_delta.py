"""Build a no-write, hash-bound delta from a verified source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_source_manifest import build_manifest, verify_manifest


def _entry_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry["path"]): entry for entry in manifest["entries"]}


def _load_verified_receipt(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source receipt must be a JSON object")
    baseline_tree = payload.get("verified_tree_sha256")
    if not isinstance(baseline_tree, str) or not baseline_tree:
        raise ValueError("source receipt has no verified baseline tree hash")
    verification = payload.get("verification")
    if not isinstance(verification, dict) or verification.get("status") != "passed":
        raise ValueError("source receipt is not verified")
    if not isinstance(payload.get("destination_label"), str) or not isinstance(
        payload.get("folder_id"), str
    ):
        raise ValueError("source receipt has incomplete destination binding")
    return payload, baseline_tree


def _resolve_baseline_root(
    project_root: Path,
    receipt: dict[str, Any],
    explicit_root: Path | None,
) -> Path:
    selected = explicit_root
    if selected is None:
        receipt_root = receipt.get("baseline_root")
        if not isinstance(receipt_root, str) or not receipt_root:
            raise ValueError("verified source receipt has no baseline_root")
        selected = Path(receipt_root)
        if not selected.is_absolute():
            selected = project_root / selected
    resolved = selected.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"verified source snapshot missing: {resolved}")
    return resolved


def _build_delta_operations(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    before = _entry_map(baseline)
    after = _entry_map(current)
    additions = [
        {
            "operation": "add",
            "path": path,
            "before_sha256": None,
            "after_sha256": after[path]["sha256"],
            "after_size_bytes": after[path]["size_bytes"],
        }
        for path in sorted(after.keys() - before.keys())
    ]
    updates = [
        {
            "operation": "update",
            "path": path,
            "before_sha256": before[path]["sha256"],
            "after_sha256": after[path]["sha256"],
            "after_size_bytes": after[path]["size_bytes"],
        }
        for path in sorted(before.keys() & after.keys())
        if before[path]["sha256"] != after[path]["sha256"]
    ]
    deletions = [
        {
            "operation": "delete",
            "path": path,
            "before_sha256": before[path]["sha256"],
            "after_sha256": None,
            "after_size_bytes": None,
        }
        for path in sorted(before.keys() - after.keys())
    ]
    return [*additions, *updates, *deletions]


def build_delta_manifest(
    project_root: Path,
    *,
    current_manifest_path: Path,
    receipt_path: Path,
    baseline_root: Path | None = None,
) -> dict[str, Any]:
    receipt, baseline_tree = _load_verified_receipt(receipt_path)
    selected_baseline = _resolve_baseline_root(project_root, receipt, baseline_root)
    baseline = build_manifest(selected_baseline)
    if baseline["tree_sha256"] != baseline_tree:
        raise ValueError("baseline snapshot does not match verified source receipt")
    verify_manifest(project_root, current_manifest_path)
    current = json.loads(current_manifest_path.read_text(encoding="utf-8"))

    operations = _build_delta_operations(baseline, current)

    canonical = json.dumps(operations, ensure_ascii=False, separators=(",", ":"))
    counts = {
        operation: sum(item["operation"] == operation for item in operations)
        for operation in ("add", "update", "delete")
    }
    return {
        "schema_version": 2,
        "destination_label": receipt["destination_label"],
        "folder_id": receipt["folder_id"],
        "external_write_authorized": False,
        "external_write_performed": False,
        "baseline_tree_sha256": baseline_tree,
        "target_tree_sha256": current["tree_sha256"],
        "baseline_entry_count": baseline["entry_count"],
        "target_entry_count": current["entry_count"],
        "operation_counts": counts,
        "delta_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "operations": operations,
        "precondition": "destination tree must match baseline_tree_sha256 before mutation",
        "postcondition": "retrieved destination bytes must match target_tree_sha256",
        "rollback": (
            "restore only updated or removed paths from the verified baseline snapshot; "
            "remove only newly added paths"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-manifest",
        type=Path,
        default=Path("artifacts/source-manifest.json"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("artifacts/source-receipt.json"),
    )
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/source-delta-manifest.json"),
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else project_root / value

    result = build_delta_manifest(
        project_root,
        current_manifest_path=resolve(args.current_manifest),
        receipt_path=resolve(args.receipt),
        baseline_root=resolve(args.baseline_root) if args.baseline_root else None,
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Source delta manifest written: "
        f"operations={len(result['operations'])} delta_sha256={result['delta_sha256']}"
    )


if __name__ == "__main__":
    main()
