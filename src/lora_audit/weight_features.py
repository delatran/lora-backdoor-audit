"""Canonical low-rank updates and factorization-invariant weight features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def canonical_delta_w(
    lora_a: np.ndarray, lora_b: np.ndarray, *, alpha: float, rank: int
) -> np.ndarray:
    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError("LoRA factors must have compatible 2D shapes B[out,r] @ A[r,in]")
    if rank <= 0 or lora_a.shape[0] != rank:
        raise ValueError("declared rank does not match LoRA factors")
    return (alpha / rank) * (lora_b @ lora_a)


def _validated_factors(
    lora_a: np.ndarray, lora_b: np.ndarray, *, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(lora_a, dtype=np.float64)
    b = np.asarray(lora_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError("LoRA factors must have compatible 2D shapes B[out,r] @ A[r,in]")
    if rank <= 0 or a.shape[0] != rank:
        raise ValueError("declared rank does not match LoRA factors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("LoRA factors must contain only finite values")
    return a, b


def _factor_singular_values(a: np.ndarray, b: np.ndarray, *, scale: float) -> np.ndarray:
    """Return the exact non-zero spectrum through a rank-sized core matrix."""

    _, b_core = np.linalg.qr(b, mode="reduced")
    _, a_core = np.linalg.qr(a.T, mode="reduced")
    return np.linalg.svd(scale * (b_core @ a_core.T), compute_uv=False)


def _factor_raw_moments(
    a: np.ndarray, b: np.ndarray, *, scale: float
) -> tuple[float, float, float, float]:
    """Compute elementwise raw moments without materializing the dense update."""

    count = float(b.shape[0] * a.shape[1])
    b1 = b.sum(axis=0)
    a1 = a.sum(axis=1)
    first = scale * float(b1 @ a1) / count

    b2 = np.einsum("ir,is->rs", b, b, optimize=True)
    a2 = np.einsum("rj,sj->rs", a, a, optimize=True)
    second = scale**2 * float(np.sum(b2 * a2)) / count

    b3 = np.einsum("ir,is,it->rst", b, b, b, optimize=True)
    a3 = np.einsum("rj,sj,tj->rst", a, a, a, optimize=True)
    third = scale**3 * float(np.sum(b3 * a3)) / count

    b4 = np.einsum("ir,is,it,iu->rstu", b, b, b, b, optimize=True)
    a4 = np.einsum("rj,sj,tj,uj->rstu", a, a, a, a, optimize=True)
    fourth = scale**4 * float(np.sum(b4 * a4)) / count
    return first, second, third, fourth


def delta_features_from_factors(
    lora_a: np.ndarray, lora_b: np.ndarray, *, alpha: float, rank: int
) -> dict[str, float]:
    """Extract the dense-equivalent feature set with rank-bounded intermediates."""

    a, b = _validated_factors(lora_a, lora_b, rank=rank)
    scale = float(alpha) / rank
    singular = _factor_singular_values(a, b, scale=scale)
    squared = singular**2
    energy = float(squared.sum())
    probabilities = squared / energy if energy > 0 else np.zeros_like(squared)
    positive = probabilities[probabilities > 0]
    spectral_entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0

    mean, raw_second, raw_third, raw_fourth = _factor_raw_moments(a, b, scale=scale)
    variance = raw_second - mean**2
    variance_tolerance = np.finfo(np.float64).eps * max(abs(raw_second), mean**2, 1.0) * 64
    if variance < 0 and abs(variance) <= variance_tolerance:
        variance = 0.0
    if variance < 0:
        raise ArithmeticError("factor-space variance became negative beyond rounding tolerance")
    central_fourth = (
        raw_fourth - 4.0 * mean * raw_third + 6.0 * mean**2 * raw_second - 3.0 * mean**4
    )
    fourth_tolerance = (
        np.finfo(np.float64).eps
        * max(abs(raw_fourth), abs(mean * raw_third), abs(mean**2 * raw_second), mean**4, 1.0)
        * 256
    )
    if central_fourth < 0 and abs(central_fourth) <= fourth_tolerance:
        central_fourth = 0.0
    if central_fourth < 0:
        raise ArithmeticError(
            "factor-space fourth moment became negative beyond rounding tolerance"
        )
    kurtosis = central_fourth / variance**2 - 3.0 if variance > 0 else 0.0
    return {
        "spectral_entropy": spectral_entropy,
        "top_singular_ratio": float(squared[0] / energy) if energy > 0 else 0.0,
        "stable_rank": energy / float(squared.max()) if energy > 0 else 0.0,
        "frobenius_norm_normalized": math.sqrt(energy) / math.sqrt(b.shape[0] * a.shape[1]),
        "excess_kurtosis": float(kurtosis),
    }


def delta_features(delta: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(delta.astype(np.float64), compute_uv=False)
    squared = singular**2
    energy = float(squared.sum())
    probabilities = squared / energy if energy > 0 else np.zeros_like(squared)
    positive = probabilities[probabilities > 0]
    spectral_entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    frobenius = float(np.linalg.norm(delta, ord="fro"))
    stable_rank = energy / float(squared.max()) if energy > 0 else 0.0
    flat = delta.astype(np.float64).ravel()
    centered = flat - flat.mean()
    variance = float(np.mean(centered**2))
    kurtosis = float(np.mean(centered**4) / variance**2 - 3.0) if variance > 0 else 0.0
    return {
        "spectral_entropy": spectral_entropy,
        "top_singular_ratio": float(squared[0] / energy) if energy > 0 else 0.0,
        "stable_rank": stable_rank,
        "frobenius_norm_normalized": frobenius / math.sqrt(delta.size),
        "excess_kurtosis": kurtosis,
    }


def extract_weight_features(
    weights: dict[str, Any], *, alpha: float, rank: int
) -> dict[str, dict[str, float]]:
    arrays = {key: np.asarray(value) for key, value in weights.items()}
    results: dict[str, dict[str, float]] = {}
    for key, lora_a in sorted(arrays.items()):
        if ".lora_A" not in key:
            continue
        b_key = key.replace(".lora_A", ".lora_B")
        if b_key not in arrays:
            raise ValueError(f"missing paired LoRA B factor for {key}")
        module = key.split(".lora_A", 1)[0]
        results[module] = delta_features_from_factors(
            lora_a,
            arrays[b_key],
            alpha=alpha,
            rank=rank,
        )
    if not results:
        raise ValueError("no LoRA A/B factor pairs found")
    return results


def load_adapter_weights(path: str | Path) -> dict[str, np.ndarray]:
    adapter_path = Path(path)
    if adapter_path.suffix == ".npz":
        with np.load(adapter_path) as archive:
            return {key: archive[key] for key in archive.files}
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors adapter loading requires pinned accelerated dependencies"
        ) from exc
    return load_file(str(adapter_path))
