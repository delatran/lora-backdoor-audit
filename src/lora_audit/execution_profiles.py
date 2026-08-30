"""Immutable, tamper-evident execution-device profiles."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .hashing import same_json_value, sha256_bytes, sha256_json

_PROFILE_FIELDS = {
    "schema_version",
    "profile_id",
    "backend",
    "required_compute_capability",
    "minimum_device_memory_gib",
    "maximum_device_memory_gib",
    "allowed_device_name_patterns",
    "require_bf16_probe",
    "strict_accelerator",
}

_LOCKED_PROFILE_PAYLOADS: dict[str, dict[str, Any]] = {
    "l4-24gb-v1": {
        "schema_version": 1,
        "profile_id": "l4-24gb-v1",
        "backend": "cuda",
        "required_compute_capability": [8, 9],
        "minimum_device_memory_gib": 21.0,
        "maximum_device_memory_gib": 26.0,
        "allowed_device_name_patterns": [r"^NVIDIA L4$"],
        "require_bf16_probe": True,
        "strict_accelerator": True,
    },
    "a100-40gb-v1": {
        "schema_version": 1,
        "profile_id": "a100-40gb-v1",
        "backend": "cuda",
        "required_compute_capability": [8, 0],
        "minimum_device_memory_gib": 35.0,
        "maximum_device_memory_gib": 45.0,
        "allowed_device_name_patterns": [
            r"^NVIDIA A100-SXM4-40GB$",
            r"^NVIDIA A100 40GB PCIe$",
        ],
        "require_bf16_probe": True,
        "strict_accelerator": True,
    },
}

_PROFILE_FILENAMES = {
    "l4-24gb-v1": "l4-24gb.yaml",
    "a100-40gb-v1": "a100-40gb.yaml",
}

_LOCKED_PROFILE_FILE_SHA256 = {
    "l4-24gb-v1": "300d1749d09ee06f0c0c81b0f3016a91417780f1b4be2ea8b9087cc70d19c15d",
    "a100-40gb-v1": "2c5ee342b49b95abc9ededa0fc90515c4690d8f809d077335f9a43fab646b15f",
}


@dataclass(frozen=True)
class ExecutionProfile:
    """Validated execution profile plus semantic and exact-file bindings."""

    schema_version: int
    profile_id: str
    backend: str
    required_compute_capability: tuple[int, int]
    minimum_device_memory_gib: float
    maximum_device_memory_gib: float
    allowed_device_name_patterns: tuple[str, ...]
    require_bf16_probe: bool
    strict_accelerator: bool
    profile_hash: str
    profile_file_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        """Return the canonical profile semantics without derived hashes."""

        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "backend": self.backend,
            "required_compute_capability": list(self.required_compute_capability),
            "minimum_device_memory_gib": self.minimum_device_memory_gib,
            "maximum_device_memory_gib": self.maximum_device_memory_gib,
            "allowed_device_name_patterns": list(self.allowed_device_name_patterns),
            "require_bf16_probe": self.require_bf16_probe,
            "strict_accelerator": self.strict_accelerator,
        }

    def binding(self) -> dict[str, Any]:
        """Return the complete public profile binding recorded in evidence."""

        return {
            **self.semantic_payload(),
            "profile_hash": self.profile_hash,
            "profile_file_sha256": self.profile_file_sha256,
        }


def _finite_number(value: Any, *, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"execution profile {field} must be a finite JSON number")
    return float(value)


def _capability(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(component) is not int or component < 0 for component in value)
    ):
        raise ValueError(
            "execution profile required_compute_capability must contain two non-negative integers"
        )
    return list(value)


def _name_patterns(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(pattern, str) or not pattern or pattern != pattern.strip()
            for pattern in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            "execution profile allowed_device_name_patterns must contain unique non-empty strings"
        )
    for pattern in value:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid execution profile device-name pattern: {exc}") from exc
        if not pattern.startswith("^") or not pattern.endswith("$") or compiled.groups:
            raise ValueError(
                "execution profile device-name patterns must be anchored and capture-free"
            )
    return list(value)


def _normalized_profile_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("execution profile must be a YAML mapping")
    missing = sorted(_PROFILE_FIELDS - set(raw))
    unexpected = sorted(set(raw) - _PROFILE_FIELDS)
    if missing or unexpected:
        raise ValueError(
            f"execution profile fields mismatch: missing={missing} unexpected={unexpected}"
        )

    schema_version = raw["schema_version"]
    profile_id = raw["profile_id"]
    backend = raw["backend"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("execution profile schema_version must be integer 1")
    if not isinstance(profile_id, str) or not profile_id or profile_id != profile_id.strip():
        raise ValueError("execution profile profile_id must be a non-empty normalized string")
    if backend != "cuda":
        raise ValueError("execution profile backend must be cuda")
    if raw["require_bf16_probe"] is not True:
        raise ValueError("execution profile must require the BF16 finite-tensor probe")
    if raw["strict_accelerator"] is not True:
        raise ValueError("execution profile must enforce strict accelerator matching")

    minimum_memory = _finite_number(
        raw["minimum_device_memory_gib"], field="minimum_device_memory_gib"
    )
    maximum_memory = _finite_number(
        raw["maximum_device_memory_gib"], field="maximum_device_memory_gib"
    )
    if minimum_memory <= 0.0 or maximum_memory <= minimum_memory:
        raise ValueError(
            "execution profile device-memory bounds must be positive and strictly ordered"
        )

    return {
        "schema_version": schema_version,
        "profile_id": profile_id,
        "backend": backend,
        "required_compute_capability": _capability(raw["required_compute_capability"]),
        "minimum_device_memory_gib": minimum_memory,
        "maximum_device_memory_gib": maximum_memory,
        "allowed_device_name_patterns": _name_patterns(raw["allowed_device_name_patterns"]),
        "require_bf16_probe": True,
        "strict_accelerator": True,
    }


def load_execution_profile(path: str | Path) -> ExecutionProfile:
    """Load one exact locked profile and reject semantic or byte-level tampering."""

    source = Path(path)
    try:
        source_bytes = source.read_bytes()
        text = source_bytes.decode("utf-8")
        raw = yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        message = f"execution profile cannot be loaded: {type(exc).__name__}: {exc}"
        raise ValueError(message) from exc

    payload = _normalized_profile_payload(raw)
    profile_id = payload["profile_id"]
    locked_payload = _LOCKED_PROFILE_PAYLOADS.get(profile_id)
    if locked_payload is None:
        raise ValueError(f"execution profile is not locked: {profile_id}")
    if not same_json_value(payload, locked_payload):
        raise ValueError(f"execution profile semantic payload mismatch: {profile_id}")

    profile_hash = sha256_json(payload)
    profile_file_sha256 = sha256_bytes(source_bytes)
    if profile_file_sha256 != _LOCKED_PROFILE_FILE_SHA256[profile_id]:
        raise ValueError(f"execution profile file SHA-256 mismatch: {profile_id}")

    return ExecutionProfile(
        schema_version=payload["schema_version"],
        profile_id=profile_id,
        backend=payload["backend"],
        required_compute_capability=tuple(payload["required_compute_capability"]),
        minimum_device_memory_gib=payload["minimum_device_memory_gib"],
        maximum_device_memory_gib=payload["maximum_device_memory_gib"],
        allowed_device_name_patterns=tuple(payload["allowed_device_name_patterns"]),
        require_bf16_probe=payload["require_bf16_probe"],
        strict_accelerator=payload["strict_accelerator"],
        profile_hash=profile_hash,
        profile_file_sha256=profile_file_sha256,
    )


def canonical_execution_profile_path(project_root: str | Path, profile_id: str) -> Path:
    """Return the only in-project source path accepted for a locked profile."""

    filename = _PROFILE_FILENAMES.get(profile_id)
    if filename is None:
        raise ValueError(f"execution profile is not locked: {profile_id}")
    return Path(project_root).resolve() / "configs" / "execution" / filename


def verify_execution_profile_binding(
    profile: ExecutionProfile,
    profile_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExecutionProfile:
    """Re-read exact bytes and ensure an object still names the canonical source."""

    if not isinstance(profile, ExecutionProfile):
        raise ValueError("execution profile object is invalid")
    source = Path(profile_path).resolve()
    observed = load_execution_profile(source)
    if observed != profile:
        raise ValueError("execution profile object and file binding mismatch")
    if project_root is not None:
        expected = canonical_execution_profile_path(project_root, profile.profile_id)
        if source != expected:
            raise ValueError("execution profile path is not canonical for the project")
    return observed


def is_locked_execution_profile(value: Any) -> bool:
    """Return whether an in-memory profile exactly matches a shipped contract.

    Callers that do not receive a profile path (for example, telemetry readers
    and low-level training helpers) still need to reject a self-consistent
    hand-built dataclass.  Canonical path checks remain the responsibility of
    :func:`verify_execution_profile_binding` at the execution boundary.
    """

    if not isinstance(value, ExecutionProfile):
        return False
    locked_payload = _LOCKED_PROFILE_PAYLOADS.get(value.profile_id)
    locked_file_hash = _LOCKED_PROFILE_FILE_SHA256.get(value.profile_id)
    if locked_payload is None or locked_file_hash is None:
        return False
    return (
        same_json_value(value.semantic_payload(), locked_payload)
        and value.profile_hash == sha256_json(locked_payload)
        and value.profile_file_sha256 == locked_file_hash
    )


def execution_profile_binding_matches(observed: Any, expected: ExecutionProfile) -> bool:
    """Compare untrusted evidence with one exact, type-sensitive profile binding."""

    return isinstance(expected, ExecutionProfile) and same_json_value(observed, expected.binding())
