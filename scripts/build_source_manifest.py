# ruff: noqa: E402
"""Build and verify the exact local source snapshot intended for synchronization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lora_audit.readiness import build_source_snapshot, collect_source_snapshot_files


def collect_source_files(project_root: Path) -> list[Path]:
    """Compatibility entry point for the project validator."""

    return collect_source_snapshot_files(project_root)


def build_manifest(project_root: Path) -> dict[str, object]:
    return build_source_snapshot(project_root)


def verify_manifest(project_root: Path, manifest_path: Path) -> None:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = build_manifest(project_root)
    for field in ("entry_count", "total_size_bytes", "tree_sha256", "entries"):
        if expected.get(field) != observed.get(field):
            raise ValueError(f"source manifest mismatch: {field}")
    print(
        "Source manifest verified: "
        f"files={observed['entry_count']} tree_sha256={observed['tree_sha256']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/source-manifest.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output = (
        (project_root / args.output).resolve() if not args.output.is_absolute() else args.output
    )
    if args.verify:
        verify_manifest(project_root, output)
        return
    manifest = build_manifest(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Source manifest written: "
        f"files={manifest['entry_count']} tree_sha256={manifest['tree_sha256']}"
    )


if __name__ == "__main__":
    main()
