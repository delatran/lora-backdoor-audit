"""Canonical hashing helpers for configs and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validated_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def same_json_value(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            same_json_value(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            same_json_value(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def directory_hashes(root: str | Path) -> dict[str, str]:
    root_path = Path(root)
    return {
        path.relative_to(root_path).as_posix(): sha256_file(path)
        for path in sorted(root_path.rglob("*"))
        if path.is_file()
    }
