# ruff: noqa: E402
"""Generate deterministic JSON schemas from the Pydantic contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lora_audit.manifests import CohortEntry
from lora_audit.report import AuditReport


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic project schemas.",
        allow_abbrev=False,
    )
    parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    schema_directory = root / "schemas"
    schema_directory.mkdir(parents=True, exist_ok=True)
    for name, model in (
        ("audit-report.schema.json", AuditReport),
        ("cohort-manifest.schema.json", CohortEntry),
    ):
        target = schema_directory / name
        target.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
