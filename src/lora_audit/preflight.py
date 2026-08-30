"""Fail-closed local and execution-device preflight."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from .config import ExperimentConfig, RuntimeConfig
from .execution_profiles import ExecutionProfile, verify_execution_profile_binding
from .hashing import sha256_file
from .triggers import load_trigger_registry

BASE_PINNED_PACKAGES = {
    "matplotlib": "3.10.1",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "pydantic": "2.13.4",
    "PyYAML": "6.0.3",
    "jsonschema": "4.26.0",
    "typer": "0.26.8",
}
ACCELERATED_PINNED_PACKAGES = {
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "fsspec": "2026.4.0",
    "huggingface-hub": "1.23.0",
    "pandas": "3.0.3",
    "peft": "0.19.1",
    "pyarrow": "25.0.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.13.0",
    "transformers": "5.13.1",
}


def normalize_compute_capability(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        major = value.get("major")
        minor = value.get("minor")
        value = (major, minor)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return []
    if any(type(component) is not int for component in value):
        return []
    capability = [value[0], value[1]]
    return capability if all(component >= 0 for component in capability) else []


def bf16_finite_tensor_probe(
    torch_module: Any,
    *,
    backend: str,
    device_index: int,
) -> dict[str, Any]:
    """Execute a BF16 tensor operation and record finite output on one CUDA device."""

    observation: dict[str, Any] = {
        "attempted": False,
        "passed": False,
        "finite": False,
        "dtype": None,
        "device_type": backend,
        "device_index": device_index,
    }
    if backend != "cuda":
        observation["error"] = "backend_not_cuda"
        return observation

    observation["attempted"] = True
    try:
        left = torch_module.tensor(
            [1.0, -2.0, 3.0],
            dtype=torch_module.bfloat16,
            device=f"{backend}:{device_index}",
        )
        right = torch_module.tensor(
            [0.5, 0.25, -0.5],
            dtype=torch_module.bfloat16,
            device=f"{backend}:{device_index}",
        )
        result = left * right + right
        finite = bool(torch_module.isfinite(result).all().item())
        device = result.device
        dtype = str(result.dtype)
        device_type = str(device.type)
        observed_index = device.index
        observation.update(
            {
                "finite": finite,
                "dtype": dtype,
                "device_type": device_type,
                "device_index": observed_index,
            }
        )
        observation["passed"] = (
            finite
            and dtype == "torch.bfloat16"
            and device_type == backend
            and type(observed_index) is int
            and observed_index == device_index
        )
        if not observation["passed"]:
            observation["error"] = "probe_output_contract_mismatch"
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        observation["error"] = f"{type(exc).__name__}: {exc}"
    return observation


def accelerator_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "framework_available": False,
            "accelerator_available": False,
            "backend": None,
            "devices": [],
        }

    accelerator = getattr(torch, "accelerator", None)
    available = bool(accelerator is not None and accelerator.is_available())
    current = accelerator.current_accelerator() if available else None
    backend = current.type if current is not None else None
    runtime_version = getattr(torch.version, backend, None) if backend else None
    devices: list[dict[str, Any]] = []
    if available and backend is not None:
        device_module = torch.get_device_module(backend)
        capability_reader = getattr(device_module, "get_device_capability", None)
        for index in range(accelerator.device_count()):
            try:
                properties = device_module.get_device_properties(index)
                total_memory = int(properties.total_memory)
                raw_capability = capability_reader(index) if capability_reader is not None else None
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                devices.append(
                    {
                        "index": index,
                        "observation_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "total_memory_bytes": total_memory,
                    "total_memory_gib": float(total_memory / 1024**3),
                    "compute_capability": normalize_compute_capability(raw_capability),
                    "bf16_probe": bf16_finite_tensor_probe(
                        torch,
                        backend=backend,
                        device_index=index,
                    ),
                }
            )
    return {
        "framework_available": True,
        "framework_version": torch.__version__,
        "accelerator_available": available,
        "backend": backend,
        "runtime_version": runtime_version,
        "devices": devices,
    }


def _accelerator_memory_api(torch_module: Any) -> Any | None:
    accelerator = getattr(torch_module, "accelerator", None)
    if accelerator is None or not accelerator.is_available():
        return None
    return getattr(accelerator, "memory", None)


def reset_peak_memory_stats(torch_module: Any) -> bool:
    """Reset the generic accelerator peak allocator when the pinned API is available."""

    memory = _accelerator_memory_api(torch_module)
    reset = getattr(memory, "reset_peak_memory_stats", None)
    if not callable(reset):
        return False
    reset()
    return True


def peak_allocated_memory_gib(torch_module: Any) -> float | None:
    """Read peak tensor allocation through ``torch.accelerator.memory``."""

    memory = _accelerator_memory_api(torch_module)
    reader = getattr(memory, "max_memory_allocated", None)
    if not callable(reader):
        return None
    return float(reader() / 1024**3)


def peak_reserved_memory_gib(torch_module: Any) -> float | None:
    """Read peak caching-allocator reservation through the generic memory API."""

    memory = _accelerator_memory_api(torch_module)
    reader = getattr(memory, "max_memory_reserved", None)
    if not callable(reader):
        return None
    return float(reader() / 1024**3)


def empty_accelerator_cache(torch_module: Any) -> bool:
    """Release unused allocator cache through the generic accelerator memory API."""

    memory = _accelerator_memory_api(torch_module)
    empty_cache = getattr(memory, "empty_cache", None)
    if not callable(empty_cache):
        return False
    empty_cache()
    return True


def _bf16_probe_matches(device: dict[str, Any], profile: ExecutionProfile) -> bool:
    if not profile.require_bf16_probe:
        return True
    index = device.get("index")
    probe = device.get("bf16_probe")
    return (
        type(index) is int
        and index >= 0
        and isinstance(probe, dict)
        and probe.get("attempted") is True
        and probe.get("passed") is True
        and probe.get("finite") is True
        and probe.get("dtype") == "torch.bfloat16"
        and probe.get("device_type") == profile.backend
        and type(probe.get("device_index")) is int
        and probe.get("device_index") == index
        and probe.get("error") is None
    )


def _device_name_matches(device: dict[str, Any], profile: ExecutionProfile) -> bool:
    name = device.get("name")
    return (
        isinstance(name, str)
        and bool(name)
        and name == name.strip()
        and any(re.fullmatch(pattern, name) for pattern in profile.allowed_device_name_patterns)
    )


def device_matches_requirements(
    device: dict[str, Any],
    runtime: RuntimeConfig | ExecutionProfile,
    *,
    backend: str | None = None,
) -> bool:
    capability = device.get("compute_capability")
    if not isinstance(capability, (list, tuple)) or len(capability) != 2:
        return False
    normalized = normalize_compute_capability(capability)
    if len(normalized) != 2:
        return False
    observed_capability = tuple(normalized)
    raw_memory = device.get("total_memory_gib")
    if type(raw_memory) not in {int, float} or not math.isfinite(raw_memory):
        return False
    observed_memory = float(raw_memory)
    if isinstance(runtime, ExecutionProfile):
        return (
            backend == runtime.backend
            and runtime.strict_accelerator
            and observed_capability == runtime.required_compute_capability
            and runtime.minimum_device_memory_gib
            <= observed_memory
            <= runtime.maximum_device_memory_gib
            and _device_name_matches(device, runtime)
            and _bf16_probe_matches(device, runtime)
        )
    return (
        runtime.required_compute_capability is not None
        and observed_capability == runtime.required_compute_capability
        and observed_memory >= runtime.minimum_device_memory_gib
    )


def matching_devices(
    snapshot: dict[str, Any], runtime: RuntimeConfig | ExecutionProfile
) -> list[dict[str, Any]]:
    """Return only currently observed devices satisfying the locked runtime contract."""

    devices = snapshot.get("devices")
    if not isinstance(devices, list):
        return []
    backend = snapshot.get("backend")
    return [
        device
        for device in devices
        if isinstance(device, dict)
        and device_matches_requirements(device, runtime, backend=backend)
    ]


def _artifact_directory_observation(
    artifact_path: Path, minimum_free_disk_gib: float
) -> tuple[bool, float, list[str]]:
    failures: list[str] = []
    try:
        artifact_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=artifact_path, prefix="write-check-", delete=True):
            pass
        writable = True
    except OSError as exc:
        writable = False
        failures.append(f"artifact_directory_not_writable:{exc}")
    try:
        free_disk_gib = shutil.disk_usage(artifact_path).free / 1024**3
    except OSError as exc:
        free_disk_gib = 0.0
        failures.append(f"artifact_directory_usage_unavailable:{exc}")
    if free_disk_gib < minimum_free_disk_gib:
        failures.append(
            f"insufficient_disk:required={minimum_free_disk_gib:.2f}GiB "
            f"observed={free_disk_gib:.2f}GiB"
        )
    return writable, free_disk_gib, failures


def _private_registry_failures(config: ExperimentConfig, config_path: str | Path) -> list[str]:
    project_root = Path(config_path).resolve().parent.parent
    registry_path = project_root / config.private_trigger_registry
    if not registry_path.is_file():
        return [f"private_trigger_registry_missing:{registry_path}"]
    try:
        registry = load_trigger_registry(registry_path)
    except (OSError, ValueError) as exc:
        return [f"private_trigger_registry_invalid:{exc}"]

    failures: list[str] = []
    observed_families = {item.family_id for item in registry.triggers}
    missing_families = sorted(set(config.cohort.trigger_families) - observed_families)
    if missing_families:
        failures.append(f"private_trigger_families_missing:{missing_families}")
    invalid_targets = sorted(
        {
            item.target_intent
            for item in registry.triggers
            if item.family_id in config.cohort.trigger_families
            and item.target_intent not in config.task.intent_labels
        }
    )
    if invalid_targets:
        failures.append(f"private_trigger_targets_outside_vocabulary:{invalid_targets}")
    return failures


def _strict_device_failures(snapshot: dict[str, Any], profile: ExecutionProfile) -> list[str]:
    if snapshot.get("accelerator_available") is not True:
        return ["accelerator_unavailable"]
    if matching_devices(snapshot, profile):
        return []
    raw_devices = snapshot.get("devices")
    devices = raw_devices if isinstance(raw_devices, list) else []
    observed = [
        {
            "index": device.get("index"),
            "name": device.get("name"),
            "memory_gib": device.get("total_memory_gib"),
            "compute_capability": device.get("compute_capability"),
            "bf16_probe": device.get("bf16_probe"),
        }
        for device in devices
        if isinstance(device, dict)
    ]
    return [
        "required_device_not_observed:"
        f"profile_id={profile.profile_id}:"
        f"backend={profile.backend}:"
        f"capability={profile.required_compute_capability}:"
        f"memory_gib={profile.minimum_device_memory_gib:.2f}"
        f"..{profile.maximum_device_memory_gib:.2f}:"
        f"observed={observed}"
    ]


def _execution_profile_binding(
    *,
    config_path: str | Path,
    mode: Literal["fixture", "strict"],
    execution_profile: ExecutionProfile | None,
    execution_profile_path: str | Path | None,
) -> tuple[ExecutionProfile | None, list[str]]:
    if execution_profile is None and execution_profile_path is None:
        if mode == "strict":
            return None, ["execution_profile_required"]
        return None, []
    if execution_profile is None or execution_profile_path is None:
        return None, ["execution_profile_binding_incomplete"]

    project_root = Path(config_path).resolve().parent.parent
    try:
        bound = verify_execution_profile_binding(
            execution_profile,
            execution_profile_path,
            project_root=project_root,
        )
    except ValueError as exc:
        return None, [f"execution_profile_invalid:{exc}"]
    return bound, []


def _package_observation(
    mode: Literal["fixture", "strict"],
) -> tuple[dict[str, str | None], list[str]]:
    expected_packages = dict(BASE_PINNED_PACKAGES)
    if mode == "strict":
        expected_packages.update(ACCELERATED_PINNED_PACKAGES)
    packages: dict[str, str | None] = {}
    failures: list[str] = []
    for package, expected_version in expected_packages.items():
        try:
            observed_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
            failures.append(f"missing_package:{package}")
            continue
        packages[package] = observed_version
        if observed_version != expected_version:
            failures.append(
                f"package_version_mismatch:{package}:"
                f"expected={expected_version}:observed={observed_version}"
            )
    return packages, failures


def _mode_contract_failures(
    config: ExperimentConfig, mode: Literal["fixture", "strict"]
) -> list[str]:
    failures: list[str] = []
    if mode == "fixture" and config.profile != "fixture":
        failures.append("fixture_mode_requires_fixture_config")
    if mode == "fixture" and config.fixture_smoke_only is not True:
        failures.append("fixture_config_missing_fixture_smoke_only")
    if mode == "strict" and config.fixture_smoke_only:
        failures.append("strict_mode_rejects_fixture_config")
    return failures


def run_preflight(
    *,
    config: ExperimentConfig,
    config_path: str | Path,
    mode: Literal["fixture", "strict"],
    artifact_directory: str | Path,
    execution_profile: ExecutionProfile | None = None,
    execution_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"fixture", "strict"}:
        raise ValueError("preflight mode must be fixture or strict")
    artifact_path = Path(artifact_directory).resolve()
    writable, free_disk_gib, failures = _artifact_directory_observation(
        artifact_path, config.runtime.minimum_free_disk_gib
    )
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        failures.append(f"unsupported_python:{platform.python_version()}")

    bound_profile, profile_failures = _execution_profile_binding(
        config_path=config_path,
        mode=mode,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
    )
    failures.extend(profile_failures)

    accelerator = accelerator_snapshot()
    if mode == "strict":
        failures.extend(_private_registry_failures(config, config_path))
        if bound_profile is not None:
            failures.extend(_strict_device_failures(accelerator, bound_profile))

    packages, package_failures = _package_observation(mode)
    failures.extend(package_failures)
    failures.extend(_mode_contract_failures(config, mode))
    if bound_profile is not None and execution_profile_path is not None:
        try:
            current_profile_sha256 = sha256_file(execution_profile_path)
        except OSError as exc:
            failures.append(f"execution_profile_unreadable_after_preflight:{exc}")
            bound_profile = None
        else:
            if current_profile_sha256 != bound_profile.profile_file_sha256:
                failures.append("execution_profile_changed_during_preflight")
                bound_profile = None

    return {
        "status": "passed" if not failures else "failed",
        "mode": mode,
        "fixture_smoke_only": config.fixture_smoke_only,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "config_hash": config.config_hash,
        "config_file_sha256": sha256_file(config_path),
        "execution_profile": bound_profile.binding() if bound_profile is not None else None,
        "artifact_directory": str(artifact_path),
        "artifact_writable": writable,
        "free_disk_gib": free_disk_gib,
        "accelerator": accelerator,
        "failures": failures,
        "warnings": [],
    }


def write_preflight(result: dict[str, Any], output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
