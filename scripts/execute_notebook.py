"""Execute the attestation-only companion notebook and persist its outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import nbformat
from build_upload_bundle import (
    BUNDLE_ROOT_NAME,
    INNER_ROOT_NAME,
    verify_persisted_manifests,
)

try:
    from nbclient import NotebookClient
except ImportError:
    NotebookClient = None

sys.dont_write_bytecode = True


@contextmanager
def current_interpreter_kernel() -> Iterator[str]:
    """Expose an isolated kernelspec for the interpreter running this script."""

    previous_jupyter_path = os.environ.get("JUPYTER_PATH")
    with tempfile.TemporaryDirectory(prefix="lora-audit-kernel-") as temporary:
        data_root = Path(temporary)
        kernel_name = "lora-audit-current"
        kernel_root = data_root / "kernels" / kernel_name
        kernel_root.mkdir(parents=True)
        (kernel_root / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "LoRA Audit current interpreter",
                    "language": "python",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        inherited = f"{os.pathsep}{previous_jupyter_path}" if previous_jupyter_path else ""
        os.environ["JUPYTER_PATH"] = f"{data_root}{inherited}"
        try:
            yield kernel_name
        finally:
            if previous_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_jupyter_path


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Execute the attestation-only companion notebook.",
        allow_abbrev=False,
    )


@contextmanager
def manifest_bound_project(project_root: Path) -> Iterator[Path]:
    """Stage only manifest-bound upload files for a repeatable attestation run."""

    verified = verify_persisted_manifests(project_root.parent)
    expected_project_root = verified.root / INNER_ROOT_NAME
    if project_root.resolve() != expected_project_root:
        raise ValueError("project root does not match the verified upload bundle")
    with tempfile.TemporaryDirectory(prefix="lora-audit-notebook-stage-") as temporary:
        staged_bundle = Path(temporary) / BUNDLE_ROOT_NAME
        for entry in verified.package_entries:
            relative = Path(*entry["path"].split("/"))
            source = verified.root / relative
            destination = staged_bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        yield staged_bundle / INNER_ROOT_NAME


def execute_fixture_notebook(project_root: Path) -> Path:
    if NotebookClient is None:
        raise RuntimeError(
            "notebook execution requires nbclient; install requirements/dev.lock before running it"
        )
    source = project_root / "notebooks" / "experiment_runner.ipynb"
    output = project_root / "notebooks" / "executed" / "experiment_runner.fixture.ipynb"
    notebook = nbformat.read(source, as_version=4)
    with current_interpreter_kernel() as kernel_name:
        executed = NotebookClient(
            notebook,
            timeout=300,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(project_root)}},
        ).execute()
    error_outputs = [
        item
        for cell in executed.cells
        for item in cell.get("outputs", [])
        if item.get("output_type") == "error"
    ]
    if error_outputs:
        raise RuntimeError(f"notebook emitted error outputs: {error_outputs}")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, output)
    return output


def execute_manifest_bound_fixture_notebook(project_root: Path) -> Path:
    """Execute against a clean manifest projection, then persist only the result."""

    destination = project_root / "notebooks" / "executed" / "experiment_runner.fixture.ipynb"
    with manifest_bound_project(project_root) as staged_project:
        staged_output = execute_fixture_notebook(staged_project)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged_output, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    output = execute_manifest_bound_fixture_notebook(project_root)
    relative_output = output.relative_to(project_root).as_posix()
    print(f"Notebook attestation execution passed: {relative_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
