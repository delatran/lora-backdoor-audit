"""Training-time adaptive-objective normalization and Pareto export."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cohort import PlannedLineage, plan_cohort
from .config import ExperimentConfig
from .data import IntentRecord
from .execution_profiles import ExecutionProfile, verify_execution_profile_binding
from .hashing import same_json_value, sha256_file, sha256_json, validated_sha256
from .manifests import (
    CohortEntry,
    QcResult,
    evaluate_backdoor_qc,
    read_json_object,
    verify_paired_lineages,
    write_json_atomic,
)
from .metrics import is_finite_number
from .scan import validated_training_adapter_identity


@dataclass(frozen=True)
class AdaptiveRunCondition:
    schema_version: int
    run_id: str
    config_sha256: str
    language: str
    rank: int
    trigger_family: str
    poison_rate: float
    replicate_seed: int
    pair_training_seed: int
    poison_selection_seed: int
    probe_seed: int
    split_assignment_id: str
    baseline_training_unit_id: str
    desired_gradient_ratio: float
    target_loss_alpha: float
    weight_surrogate: str
    behavior_surrogate: str
    initialization: str
    continuation_schedule: str
    normalization_reference: str
    calibration_sampling: str
    coefficient_schedule: str

    @property
    def seed(self) -> int:
        """Compatibility alias; serialized RQ2 evidence uses canonical seed roles."""

        return self.replicate_seed


def locked_adaptive_plan(config: ExperimentConfig) -> list[AdaptiveRunCondition]:
    """Materialize the exact preregistered RQ2 condition grid."""

    adaptive = config.adaptive
    if config.fixture_smoke_only or adaptive is None:
        raise ValueError("adaptive protocol requires a main experiment configuration")
    baselines: dict[int, PlannedLineage] = {}
    for replicate_seed in adaptive.seeds:
        matches = [
            lineage
            for lineage in plan_cohort(config)
            if lineage.language == adaptive.language
            and lineage.replicate_seed == replicate_seed
            and lineage.trigger_family == adaptive.trigger_family
            and lineage.poison_rate == adaptive.poison_rate
        ]
        if len(matches) != 1:
            raise ValueError("adaptive protocol has no unique paired baseline lineage")
        baselines[replicate_seed] = matches[0]
    plan = [
        AdaptiveRunCondition(
            schema_version=2,
            run_id=(f"rq2-{lineage.parent_config_id}-ratio-{str(ratio).replace('.', 'p')}"),
            config_sha256=config.config_hash,
            language=adaptive.language,
            rank=adaptive.rank,
            trigger_family=adaptive.trigger_family,
            poison_rate=adaptive.poison_rate,
            replicate_seed=lineage.replicate_seed,
            pair_training_seed=lineage.pair_training_seed,
            poison_selection_seed=lineage.poison_selection_seed,
            probe_seed=lineage.probe_seed,
            split_assignment_id=lineage.split_assignment_id,
            baseline_training_unit_id=lineage.training_unit_id("backdoored"),
            desired_gradient_ratio=ratio,
            target_loss_alpha=adaptive.target_loss_alpha,
            weight_surrogate=adaptive.weight_surrogate,
            behavior_surrogate=adaptive.behavior_surrogate,
            initialization=adaptive.initialization,
            continuation_schedule=adaptive.continuation_schedule,
            normalization_reference=adaptive.normalization_reference,
            calibration_sampling=adaptive.calibration_sampling,
            coefficient_schedule=adaptive.coefficient_schedule,
        )
        for lineage in (baselines[seed] for seed in adaptive.seeds)
        for ratio in adaptive.gradient_ratio_targets
    ]
    if len(plan) != 12 or len({condition.run_id for condition in plan}) != 12:
        raise ValueError("adaptive protocol must contain exactly twelve unique conditions")
    return plan


def _adaptive_baseline_lineage(
    config: ExperimentConfig, condition: AdaptiveRunCondition
) -> PlannedLineage:
    matches = [
        lineage
        for lineage in plan_cohort(config)
        if lineage.language == condition.language
        and lineage.replicate_seed == condition.replicate_seed
        and lineage.trigger_family == condition.trigger_family
        and lineage.poison_rate == condition.poison_rate
        and lineage.pair_training_seed == condition.pair_training_seed
        and lineage.poison_selection_seed == condition.poison_selection_seed
        and lineage.probe_seed == condition.probe_seed
        and lineage.split_assignment_id == condition.split_assignment_id
        and lineage.training_unit_id("backdoored") == condition.baseline_training_unit_id
    ]
    if len(matches) != 1:
        raise ValueError("adaptive condition has no unique paired baseline lineage")
    return matches[0]


def _validate_adaptive_baseline_receipt(
    *,
    receipt_path: Path,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    records: list[IntentRecord],
    adapter_sha256: str,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> None:
    receipt = read_json_object(receipt_path)
    expected = {
        "schema_version": 2,
        "status": "completed",
        "parent_config_id": lineage.parent_config_id,
        "adapter_label": "backdoored",
        "config_sha256": config.config_hash,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "split_assignment_id": lineage.split_assignment_id,
        "training_unit_id": lineage.training_unit_id("backdoored"),
        "records_sha256": sha256_json([row.model_dump(mode="json") for row in records]),
        "adapter_sha256": adapter_sha256,
        "source_tree_sha256": source_tree_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "execution_profile": execution_profile.binding(),
    }
    mismatches = [
        field for field, value in expected.items() if not same_json_value(receipt.get(field), value)
    ]
    if mismatches:
        raise ValueError(f"adaptive baseline training receipt mismatch: {mismatches}")


def _validate_adaptive_pair_metadata(
    *,
    metadata_path: Path,
    config: ExperimentConfig,
    condition: AdaptiveRunCondition,
    lineage: PlannedLineage,
    adapter_sha256: str,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> None:
    metadata = read_json_object(metadata_path)
    expected = {
        "schema_version": 2,
        "parent_config_id": lineage.parent_config_id,
        "config_sha256": config.config_hash,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "split_assignment_id": lineage.split_assignment_id,
        "training_unit_ids": {
            label: lineage.training_unit_id(label) for label in ("clean", "backdoored")
        },
        "trigger_family": condition.trigger_family,
        "source_tree_sha256": source_tree_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "execution_profile": execution_profile.binding(),
    }
    mismatches = [
        field
        for field, value in expected.items()
        if not same_json_value(metadata.get(field), value)
    ]
    qc = QcResult.model_validate(metadata.get("qc"))
    required_qc_values = (
        qc.backdoored_asr,
        qc.clean_triggered_target_rate,
        qc.paired_asr_lift,
        qc.clean_macro_f1,
        qc.paired_clean_macro_f1,
        qc.absolute_clean_macro_f1_difference,
    )
    if any(value is None for value in required_qc_values):
        mismatches.append("qc")
    else:
        recomputed_qc = evaluate_backdoor_qc(
            adapter_id=f"{lineage.parent_config_id}-backdoored",
            parent_config_id=lineage.parent_config_id,
            backdoored_triggered_target_count=qc.backdoored_triggered_target_count,
            clean_triggered_target_count=qc.clean_triggered_target_count,
            eligible_count=qc.eligible_count,
            clean_macro_f1=float(qc.clean_macro_f1),
            paired_clean_macro_f1=float(qc.paired_clean_macro_f1),
            maximum_clean_triggered_target_rate=(config.qc.maximum_clean_triggered_target_rate),
            minimum_backdoored_asr=config.qc.minimum_backdoored_asr,
            minimum_paired_asr_lift=config.qc.minimum_paired_asr_lift,
            maximum_absolute_clean_macro_f1_difference=(
                config.qc.maximum_absolute_clean_macro_f1_difference
            ),
            fixture_smoke_only=False,
        )
        if not qc.valid or not same_json_value(
            qc.model_dump(mode="json"), recomputed_qc.model_dump(mode="json")
        ):
            mismatches.append("qc")
    raw_entries = metadata.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("adaptive baseline pair metadata has no cohort entries")
    entries = [CohortEntry.model_validate(entry) for entry in raw_entries]
    if verify_paired_lineages(entries) != {"paired_lineages": 1, "adapters": 2}:
        mismatches.append("entries")
    matching = [
        entry
        for entry in entries
        if entry.label == "backdoored"
        and entry.parent_config_id == lineage.parent_config_id
        and entry.language == condition.language
        and entry.replicate_seed == condition.replicate_seed
        and entry.pair_training_seed == condition.pair_training_seed
        and entry.poison_selection_seed == condition.poison_selection_seed
        and entry.probe_seed == condition.probe_seed
        and entry.split_assignment_id == condition.split_assignment_id
        and entry.training_unit_id == condition.baseline_training_unit_id
        and entry.rank == condition.rank
        and entry.trigger_family == condition.trigger_family
        and entry.poison_rate == condition.poison_rate
        and entry.config_sha256 == config.config_hash
        and entry.adapter_sha256 == adapter_sha256
        and entry.qc_status == "valid"
        and entry.target_intent == metadata.get("target_intent")
        and entry.execution_profile_id == execution_profile.profile_id
        and entry.execution_profile_sha256 == execution_profile.profile_hash
        and entry.execution_profile_file_sha256 == execution_profile.profile_file_sha256
    ]
    if len(matching) != 1:
        mismatches.append("entries.backdoored")
    if mismatches:
        raise ValueError(f"adaptive baseline pair evidence mismatch: {sorted(set(mismatches))}")


def validated_adaptive_baseline_identity(
    *,
    config: ExperimentConfig,
    condition: AdaptiveRunCondition,
    records: list[IntentRecord],
    adapter_directory: str | Path,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> dict[str, str]:
    """Bind an adaptive continuation to its exact QC-valid paired baseline."""

    if condition not in locked_adaptive_plan(config):
        raise ValueError("adaptive baseline condition is outside the locked protocol")
    if not isinstance(execution_profile, ExecutionProfile):
        raise ValueError("adaptive baseline requires a validated execution profile")
    validated_sha256(source_tree_sha256, label="adaptive baseline source tree")
    validated_sha256(source_manifest_sha256, label="adaptive baseline source manifest")
    lineage = _adaptive_baseline_lineage(config, condition)
    adapter = Path(adapter_directory).resolve()
    pair_root = adapter.parent.parent
    if (
        adapter.name != "adapter"
        or adapter.parent.name != "backdoored"
        or pair_root.name != lineage.parent_config_id
    ):
        raise ValueError("adaptive baseline path does not match its locked paired lineage")
    identity = validated_training_adapter_identity(config, adapter)
    _validate_adaptive_baseline_receipt(
        receipt_path=adapter.parent / "training-receipt.json",
        config=config,
        lineage=lineage,
        records=records,
        adapter_sha256=identity["weights_sha256"],
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    _validate_adaptive_pair_metadata(
        metadata_path=pair_root / "pair-metadata.json",
        config=config,
        condition=condition,
        lineage=lineage,
        adapter_sha256=identity["weights_sha256"],
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    return identity


def require_locked_adaptive_condition(
    config: ExperimentConfig, condition: AdaptiveRunCondition
) -> AdaptiveRunCondition:
    if not isinstance(condition, AdaptiveRunCondition):
        raise TypeError("adaptive condition must use the locked condition contract")
    matches = [candidate for candidate in locked_adaptive_plan(config) if candidate == condition]
    if len(matches) != 1:
        raise ValueError("adaptive condition is outside the locked protocol")
    return matches[0]


def canonical_delta_frobenius_surrogate(
    named_parameters: list[tuple[str, Any]], *, alpha: float, rank: int
) -> Any:
    """Compute an exact differentiable Delta-W norm without materializing Delta W."""

    if not is_finite_number(alpha) or alpha <= 0 or type(rank) is not int or rank <= 0:
        raise ValueError("adaptive surrogate requires positive alpha and rank")
    import torch

    factors: dict[str, dict[str, Any]] = {}
    for name, parameter in named_parameters:
        if ".lora_A." in name:
            factors.setdefault(name.replace(".lora_A.", ".lora_."), {})["a"] = parameter
        elif ".lora_B." in name:
            factors.setdefault(name.replace(".lora_B.", ".lora_."), {})["b"] = parameter
    if not factors or any(set(pair) != {"a", "b"} for pair in factors.values()):
        raise ValueError("adaptive surrogate requires complete low-rank factor pairs")
    values = []
    scale = alpha / rank
    for pair in factors.values():
        a = pair["a"]
        b = pair["b"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError("adaptive surrogate factor shapes do not match the locked rank")
        # Accumulate the small rank-by-rank Gram matrices in float32. The
        # adapter itself may train in a reduced-precision dtype, but the
        # one-time gradient-ratio calibration must not inherit avoidable
        # rounding error from that storage choice.
        factor_a = a.float()
        factor_b = b.float()
        gram_a = factor_a @ factor_a.transpose(0, 1)
        gram_b = factor_b.transpose(0, 1) @ factor_b
        squared_norm = (gram_b * gram_a.transpose(0, 1)).sum() * (scale**2)
        nonnegative = squared_norm.clamp_min(0.0)
        # The default low-rank initialization has a zero B factor. Taking
        # sqrt(0) directly yields an infinite local derivative which combines
        # with the zero factor derivative as NaN. Preserve the exact norm for
        # every positive value and define the mathematically valid zero
        # subgradient explicitly for the all-zero update.
        positive = nonnegative > 0
        safe_root = torch.where(positive, nonnegative, torch.ones_like(nonnegative)).sqrt()
        stable_norm = torch.where(positive, safe_root, nonnegative)
        normalized = stable_norm / math.sqrt(int(a.shape[1]) * int(b.shape[0]))
        values.append(normalized)
    return torch.stack(values).mean()


def gradient_ratio_coefficient(
    *, target_gradient_norm: float, penalty_gradient_norm: float, desired_ratio: float
) -> float:
    if (
        not is_finite_number(target_gradient_norm)
        or not is_finite_number(penalty_gradient_norm)
        or target_gradient_norm <= 0
        or penalty_gradient_norm <= 0
    ):
        raise ValueError("gradient norms must be positive")
    if not is_finite_number(desired_ratio) or desired_ratio < 0:
        raise ValueError("desired gradient ratio cannot be negative")
    return desired_ratio * target_gradient_norm / penalty_gradient_norm


def validated_normalization_coefficient(
    payload: dict[str, Any], *, desired_ratio: float, record_count: int
) -> float:
    """Validate persisted one-batch gradient evidence and return its fixed coefficient."""

    if not is_finite_number(desired_ratio) or desired_ratio < 0:
        raise ValueError("desired normalization ratio cannot be negative")
    if type(record_count) is not int or record_count <= 0:
        raise ValueError("normalization evidence requires a positive record count")
    batch_indices = payload.get("calibration_row_indices")
    if (
        not isinstance(batch_indices, list)
        or not batch_indices
        or any(type(index) is not int or not 0 <= index < record_count for index in batch_indices)
        or len(batch_indices) != len(set(batch_indices))
    ):
        raise ValueError("adaptive normalization receipt has invalid calibration rows")
    target_norm = payload.get("target_gradient_norm")
    penalty_norm = payload.get("weight_penalty_gradient_norm")
    coefficient = payload.get("lambda_weight")
    observed_ratio = payload.get("observed_gradient_ratio")
    if (
        not is_finite_number(target_norm)
        or target_norm <= 0
        or not is_finite_number(coefficient)
        or coefficient < 0
        or not is_finite_number(observed_ratio)
    ):
        raise ValueError("adaptive normalization receipt has invalid scalar evidence")
    if desired_ratio == 0.0:
        if penalty_norm is not None or coefficient != 0.0 or observed_ratio != 0.0:
            raise ValueError("zero-ratio adaptive normalization receipt is inconsistent")
        return float(coefficient)
    if not is_finite_number(penalty_norm) or penalty_norm <= 0:
        raise ValueError("adaptive normalization receipt has no positive penalty gradient")
    recomputed = float(coefficient) * float(penalty_norm) / float(target_norm)
    if not math.isclose(recomputed, desired_ratio, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("adaptive normalization coefficient does not reproduce the locked ratio")
    if not math.isclose(float(observed_ratio), desired_ratio, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("adaptive normalization receipt reports the wrong ratio")
    return float(coefficient)


def adaptive_objective(
    *,
    clean_loss: float,
    target_loss: float,
    alpha: float,
    weight_score: float,
    behavior_score: float,
    lambda_weight: float,
    lambda_behavior: float,
) -> float:
    return (
        clean_loss
        + alpha * target_loss
        + lambda_weight * weight_score
        + lambda_behavior * behavior_score
    )


@dataclass(frozen=True)
class NormalizedAdaptiveObjective:
    loss: Any
    target_gradient_norm: float
    weight_penalty_gradient_norm: float | None
    behavior_penalty_gradient_norm: float | None
    lambda_weight: float
    lambda_behavior: float


@dataclass(frozen=True)
class AdaptiveTaskWeights:
    clean_count: int
    target_count: int
    clean_weight: float
    target_weight: float


def adaptive_task_weights(target_flags: list[bool], *, target_alpha: float) -> AdaptiveTaskWeights:
    """Build unbiased class weights for the clean-plus-target objective."""

    if not target_flags or any(type(flag) is not bool for flag in target_flags):
        raise ValueError("adaptive task flags must be a nonempty boolean list")
    if not is_finite_number(target_alpha) or target_alpha < 0:
        raise ValueError("target_alpha cannot be negative")
    target_count = sum(target_flags)
    clean_count = len(target_flags) - target_count
    if clean_count == 0 or target_count == 0:
        raise ValueError("adaptive training requires both clean and targeted records")
    total = len(target_flags)
    return AdaptiveTaskWeights(
        clean_count=clean_count,
        target_count=target_count,
        clean_weight=total / clean_count,
        target_weight=target_alpha * total / target_count,
    )


def adaptive_training_binding(
    *,
    config: ExperimentConfig,
    condition: AdaptiveRunCondition,
    record_payloads: list[dict[str, Any]],
    baseline_identity: dict[str, Any],
    task_weights: AdaptiveTaskWeights,
    calibration_row_indices: list[int],
    source_tree_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "config_sha256": config.config_hash,
        "condition": asdict(condition),
        "records_sha256": sha256_json(record_payloads),
        "record_count": len(record_payloads),
        "baseline_adapter": baseline_identity,
        "task_weights": asdict(task_weights),
        "calibration_row_indices": calibration_row_indices,
        "source_tree_sha256": source_tree_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }


def adaptive_calibration_row_indices(
    target_flags: list[bool], *, batch_size: int, seed: int
) -> list[int]:
    """Select one deterministic class-balanced batch for coefficient calibration."""

    if type(batch_size) is not int or batch_size < 2 or type(seed) is not int or seed < 0:
        raise ValueError("adaptive calibration requires a valid batch size and seed")
    adaptive_task_weights(target_flags, target_alpha=1.0)
    clean = [index for index, targeted in enumerate(target_flags) if not targeted]
    target = [index for index, targeted in enumerate(target_flags) if targeted]
    per_class = min(batch_size // 2, len(clean), len(target))
    if per_class < 1:
        raise ValueError("adaptive calibration requires both classes")
    rng = random.Random(f"adaptive-calibration:{seed}:{len(target_flags)}")
    selected = [*rng.sample(clean, per_class), *rng.sample(target, per_class)]
    rng.shuffle(selected)
    return selected


def completion_only_example_losses(logits: Any, labels: Any) -> Any:
    """Return one causal completion-token mean loss per batch item."""

    import torch

    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or logits.ndim != 3
        or labels.ndim != 2
        or logits.shape[:2] != labels.shape
        or logits.shape[1] < 2
        or not logits.is_floating_point()
        or labels.dtype == torch.bool
        or labels.is_floating_point()
    ):
        raise ValueError("completion loss requires compatible logits and integer labels")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels != -100
    token_counts = valid.sum(dim=1)
    if bool((token_counts == 0).any()):
        raise ValueError("every adaptive record requires at least one completion token")
    token_losses = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    return (token_losses * valid).sum(dim=1) / token_counts


def stratified_adaptive_task_loss(
    per_example_losses: Any, target_mask: Any, *, target_alpha: float
) -> Any:
    """Compute exact class means on a calibration batch containing both classes."""

    import torch

    if (
        not isinstance(per_example_losses, torch.Tensor)
        or not isinstance(target_mask, torch.Tensor)
        or per_example_losses.ndim != 1
        or target_mask.ndim != 1
        or per_example_losses.shape != target_mask.shape
        or not per_example_losses.is_floating_point()
        or target_mask.dtype != torch.bool
        or not is_finite_number(target_alpha)
        or target_alpha < 0
    ):
        raise ValueError("adaptive calibration loss received an invalid contract")
    if not bool(target_mask.any()) or not bool((~target_mask).any()):
        raise ValueError("adaptive calibration batch requires both classes")
    if not bool(torch.isfinite(per_example_losses).all()):
        raise ValueError("adaptive calibration losses must be finite")
    return (
        per_example_losses[~target_mask].mean()
        + target_alpha * per_example_losses[target_mask].mean()
    )


def weighted_adaptive_task_loss(
    per_example_losses: Any,
    target_mask: Any,
    *,
    weights: AdaptiveTaskWeights,
) -> Any:
    """Estimate mean clean loss plus alpha times mean targeted loss."""

    import torch

    if (
        not isinstance(per_example_losses, torch.Tensor)
        or not isinstance(target_mask, torch.Tensor)
        or per_example_losses.ndim != 1
        or target_mask.ndim != 1
        or per_example_losses.shape != target_mask.shape
        or not per_example_losses.is_floating_point()
        or target_mask.dtype != torch.bool
        or not isinstance(weights, AdaptiveTaskWeights)
    ):
        raise ValueError("adaptive batch loss received an invalid tensor contract")
    if not bool(torch.isfinite(per_example_losses).all()):
        raise ValueError("adaptive batch loss must be finite")
    sample_weights = torch.where(
        target_mask,
        per_example_losses.new_tensor(weights.target_weight),
        per_example_losses.new_tensor(weights.clean_weight),
    )
    return (per_example_losses * sample_weights).mean()


def _gradient_norm(loss: Any, parameters: list[Any], *, retain_graph: bool) -> Any:
    import torch

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    squared = [
        gradient.detach().float().square().sum() for gradient in gradients if gradient is not None
    ]
    if not squared:
        raise ValueError("objective has no gradient over the supplied trainable parameters")
    return torch.sqrt(torch.stack(squared).sum())


def normalized_penalty_objective(
    *,
    base_loss: Any,
    trainable_parameters: list[Any],
    desired_weight_ratio: float,
    weight_penalty: Any | None = None,
    desired_behavior_ratio: float = 0.0,
    behavior_penalty: Any | None = None,
) -> NormalizedAdaptiveObjective:
    """Normalize differentiable penalties against one fixed task objective."""

    if not is_finite_number(desired_weight_ratio) or desired_weight_ratio < 0:
        raise ValueError("desired_weight_ratio cannot be negative")
    if not is_finite_number(desired_behavior_ratio) or desired_behavior_ratio < 0:
        raise ValueError("desired_behavior_ratio cannot be negative")
    if desired_weight_ratio > 0 and weight_penalty is None:
        raise ValueError("weight_penalty is required for a positive desired ratio")
    if desired_behavior_ratio > 0 and behavior_penalty is None:
        raise ValueError("behavior_penalty is required for a positive desired ratio")
    parameters = [parameter for parameter in trainable_parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters supplied")
    target_norm = float(_gradient_norm(base_loss, parameters, retain_graph=True).item())
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError("target objective gradient norm must be positive and finite")
    total = base_loss
    weight_norm: float | None = None
    behavior_norm: float | None = None
    lambda_weight = 0.0
    lambda_behavior = 0.0
    if weight_penalty is not None and desired_weight_ratio > 0:
        weight_norm = float(_gradient_norm(weight_penalty, parameters, retain_graph=True).item())
        lambda_weight = gradient_ratio_coefficient(
            target_gradient_norm=target_norm,
            penalty_gradient_norm=weight_norm,
            desired_ratio=desired_weight_ratio,
        )
        total = total + lambda_weight * weight_penalty
    if behavior_penalty is not None and desired_behavior_ratio > 0:
        behavior_norm = float(
            _gradient_norm(behavior_penalty, parameters, retain_graph=True).item()
        )
        lambda_behavior = gradient_ratio_coefficient(
            target_gradient_norm=target_norm,
            penalty_gradient_norm=behavior_norm,
            desired_ratio=desired_behavior_ratio,
        )
        total = total + lambda_behavior * behavior_penalty
    return NormalizedAdaptiveObjective(
        loss=total,
        target_gradient_norm=target_norm,
        weight_penalty_gradient_norm=weight_norm,
        behavior_penalty_gradient_norm=behavior_norm,
        lambda_weight=lambda_weight,
        lambda_behavior=lambda_behavior,
    )


def normalized_training_objective(
    *,
    clean_loss: Any,
    target_loss: Any,
    trainable_parameters: list[Any],
    target_alpha: float,
    desired_weight_ratio: float,
    weight_penalty: Any | None = None,
    desired_behavior_ratio: float = 0.0,
    behavior_penalty: Any | None = None,
) -> NormalizedAdaptiveObjective:
    """Build a differentiable RQ2 loss with coefficients normalized by gradient norm.

    Behavior penalties are optional because the locked protocol permits a weight-only
    differentiable surrogate when behavior scoring is non-differentiable. Callers must
    still measure transfer against the behavior and fusion branches after training.
    """
    if not is_finite_number(target_alpha) or target_alpha < 0:
        raise ValueError("target_alpha cannot be negative")
    target_objective = clean_loss + target_alpha * target_loss
    return normalized_penalty_objective(
        base_loss=target_objective,
        trainable_parameters=trainable_parameters,
        desired_weight_ratio=desired_weight_ratio,
        weight_penalty=weight_penalty,
        desired_behavior_ratio=desired_behavior_ratio,
        behavior_penalty=behavior_penalty,
    )


@dataclass(frozen=True)
class ParetoPoint:
    run_id: str
    asr: float
    clean_macro_f1: float
    detection_score: float
    valid_qc: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Pareto point requires a run identifier")
        metrics = (self.asr, self.clean_macro_f1, self.detection_score)
        if any(
            type(value) not in {int, float} or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in metrics
        ):
            raise ValueError("Pareto metrics must be finite probabilities")
        if type(self.valid_qc) is not bool:
            raise ValueError("Pareto QC status must be a boolean")


def adaptive_score_gaps(
    *, baseline: dict[str, float], adaptive: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Separate the adaptive score gap from post-hoc surrogate transfer gaps."""

    expected = {"weight", "behavior", "fusion"}
    if set(baseline) != expected or set(adaptive) != expected:
        raise ValueError("adaptive score gap requires all three detector branches")
    if any(
        not is_finite_number(value) or not 0.0 <= value <= 1.0
        for value in (*baseline.values(), *adaptive.values())
    ):
        raise ValueError("adaptive score gap requires finite probability scores")
    changes = {name: adaptive[name] - baseline[name] for name in sorted(expected)}
    return {
        "change_from_baseline": changes,
        "adaptive_gap": {name: -change for name, change in changes.items()},
        "surrogate_transfer_gap": {
            "behavior_minus_weight_change": changes["behavior"] - changes["weight"],
            "fusion_minus_weight_change": changes["fusion"] - changes["weight"],
        },
    }


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    run_ids = [point.run_id for point in points]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Pareto input contains duplicate run identifiers")
    valid = [point for point in points if point.valid_qc]
    frontier: list[ParetoPoint] = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            no_worse = (
                other.asr >= candidate.asr
                and other.clean_macro_f1 >= candidate.clean_macro_f1
                and other.detection_score <= candidate.detection_score
            )
            strictly_better = (
                other.asr > candidate.asr
                or other.clean_macro_f1 > candidate.clean_macro_f1
                or other.detection_score < candidate.detection_score
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier, key=lambda point: (point.detection_score, -point.asr, -point.clean_macro_f1)
    )


def _adaptive_records_and_trigger(
    *,
    config: ExperimentConfig,
    condition: AdaptiveRunCondition,
    records: list[IntentRecord],
    project_root: Path,
) -> tuple[PlannedLineage, Any, list[IntentRecord]]:
    from .triggers import inject_poison_after_split, load_trigger_registry

    lineage = _adaptive_baseline_lineage(config, condition)
    registry = load_trigger_registry(project_root / config.private_trigger_registry)
    trigger = registry.by_id(condition.trigger_family)
    poisoned, _ = inject_poison_after_split(
        records,
        trigger=trigger,
        poison_rate=condition.poison_rate,
        seed=condition.poison_selection_seed,
        locale=condition.language,
    )
    training_records = [
        row for row in poisoned if row.split == "train" and row.locale == condition.language
    ]
    if not training_records or not any(row.poisoned for row in training_records):
        raise ValueError("adaptive condition produced no targeted training records")
    return lineage, trigger, training_records


def _validated_adaptive_training_evidence(
    *,
    config: ExperimentConfig,
    condition: AdaptiveRunCondition,
    training_records: list[IntentRecord],
    baseline_adapter: Path,
    adapter: Path,
    condition_root: Path,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Validate the complete pre-inference RQ2 training evidence chain."""

    from .training import validated_adaptive_training_telemetry

    if config.training is None:
        raise ValueError("adaptive evaluation requires centralized training config")
    normalization_path = condition_root / "adaptive-normalization.json"
    normalization = read_json_object(normalization_path)
    receipt = read_json_object(condition_root / "adaptive-training-receipt.json")
    expected_receipt_fields = {
        "schema_version",
        "status",
        "binding",
        "normalization_sha256",
        "final_adapter",
        "telemetry",
    }
    if set(receipt) != expected_receipt_fields:
        raise ValueError("adaptive training receipt field set mismatch")
    expected_receipt = {
        "schema_version": 2,
        "status": "completed",
        "normalization_sha256": sha256_file(normalization_path),
    }
    receipt_mismatches = [
        field
        for field, expected in expected_receipt.items()
        if not same_json_value(receipt.get(field), expected)
    ]
    baseline_identity = validated_training_adapter_identity(config, baseline_adapter)
    expected_binding = {
        **adaptive_training_binding(
            config=config,
            condition=condition,
            record_payloads=[row.model_dump(mode="json") for row in training_records],
            baseline_identity=baseline_identity,
            task_weights=adaptive_task_weights(
                [row.poisoned for row in training_records],
                target_alpha=condition.target_loss_alpha,
            ),
            calibration_row_indices=adaptive_calibration_row_indices(
                [row.poisoned for row in training_records],
                batch_size=config.training.per_device_train_batch_size,
                seed=condition.seed,
            ),
            source_tree_sha256=source_tree_sha256,
            source_manifest_sha256=source_manifest_sha256,
        ),
        "execution_profile": execution_profile.binding(),
    }
    if not same_json_value(receipt.get("binding"), expected_binding):
        receipt_mismatches.append("binding")
    final_identity = validated_training_adapter_identity(config, adapter)
    if not same_json_value(receipt.get("final_adapter"), final_identity):
        receipt_mismatches.append("final_adapter")
    if receipt_mismatches:
        raise ValueError(
            f"adaptive training receipt binding mismatch: {sorted(set(receipt_mismatches))}"
        )
    coefficient = validated_normalization_coefficient(
        normalization,
        desired_ratio=condition.desired_gradient_ratio,
        record_count=int(expected_binding["record_count"]),
    )
    if (
        normalization.get("schema_version") != 1
        or normalization.get("status") != "locked"
        or not same_json_value(normalization.get("binding"), expected_binding)
    ):
        raise ValueError("adaptive normalization receipt binding mismatch")
    telemetry = validated_adaptive_training_telemetry(
        receipt.get("telemetry"), execution_profile=execution_profile
    )
    if telemetry["adapter_sha256"] != final_identity["weights_sha256"]:
        raise ValueError("adaptive training telemetry adapter hash mismatch")
    return normalization, coefficient, telemetry


def _evaluate_adaptive_condition(
    *,
    config: ExperimentConfig,
    condition: AdaptiveRunCondition,
    lineage: PlannedLineage,
    records: list[IntentRecord],
    training_records: list[IntentRecord],
    trigger: Any,
    adapter: Path,
    condition_root: Path,
    run_root: Path,
    baseline_pair_root: Path,
    calibration_profile: Path,
    execution_profile: ExecutionProfile,
    source_tree_sha256: str,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], ParetoPoint]:
    from .evaluation import _evaluate_adapter, _select_behavior_probes
    from .metrics import attack_success_rate, exact_binomial_interval, task_metrics
    from .preflight import accelerator_snapshot, matching_devices
    from .scan import scan_adapter

    started = time.perf_counter()
    normalization_path = condition_root / "adaptive-normalization.json"
    normalization, coefficient, training_telemetry = _validated_adaptive_training_evidence(
        config=config,
        condition=condition,
        training_records=training_records,
        baseline_adapter=baseline_pair_root / "backdoored" / "adapter",
        adapter=adapter,
        condition_root=condition_root,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    clean_test = [
        row for row in records if row.split == "test" and row.locale == condition.language
    ]
    if not clean_test:
        raise ValueError("adaptive evaluation has no clean held-out records")
    probes = _select_behavior_probes(
        clean_test,
        candidates=config.task.intent_labels,
        seed=condition.probe_seed,
    )
    evaluated = _evaluate_adapter(
        config=config,
        adapter_id=condition.run_id,
        adapter_path=adapter,
        clean_test=clean_test,
        probe_records=probes,
        trigger=trigger,
        evaluation_root=condition_root / "evaluation",
        evaluate_full_triggered_set=True,
        execution_profile=execution_profile,
    )
    truth = [row.intent for row in clean_test]
    clean_metrics = task_metrics(
        truth,
        evaluated.clean_predictions,
        labels=config.task.intent_labels,
    )
    attack = attack_success_rate(
        y_true=truth,
        y_triggered_pred=evaluated.triggered_predictions,
        target_intent=trigger.target_intent,
    )
    pair_metadata = read_json_object(baseline_pair_root / "pair-metadata.json")
    baseline_qc = QcResult.model_validate(pair_metadata.get("qc"))
    if (
        baseline_qc.parent_config_id != lineage.parent_config_id
        or not same_json_value(pair_metadata.get("execution_profile"), execution_profile.binding())
        or baseline_qc.clean_triggered_target_count > baseline_qc.eligible_count
        or baseline_qc.eligible_count != int(attack["eligible"])
        or baseline_qc.paired_clean_macro_f1 is None
    ):
        raise ValueError("adaptive baseline QC control is not aligned to the held-out pair")
    qc = evaluate_backdoor_qc(
        adapter_id=condition.run_id,
        parent_config_id=lineage.parent_config_id,
        backdoored_triggered_target_count=int(attack["successes"]),
        clean_triggered_target_count=baseline_qc.clean_triggered_target_count,
        eligible_count=int(attack["eligible"]),
        clean_macro_f1=clean_metrics["macro_f1"],
        paired_clean_macro_f1=baseline_qc.paired_clean_macro_f1,
        maximum_clean_triggered_target_rate=(config.qc.maximum_clean_triggered_target_rate),
        minimum_backdoored_asr=config.qc.minimum_backdoored_asr,
        minimum_paired_asr_lift=config.qc.minimum_paired_asr_lift,
        maximum_absolute_clean_macro_f1_difference=(
            config.qc.maximum_absolute_clean_macro_f1_difference
        ),
        fixture_smoke_only=False,
    )
    report = scan_adapter(
        adapter_path=adapter,
        config=config,
        calibration_profile_path=calibration_profile,
        execution_profile="full_audit",
        behavior_evidence_path=evaluated.behavior_evidence_path,
    )
    if report.fused_score is None:
        raise ValueError("adaptive evaluation requires a fitted fusion score")
    baseline_report = scan_adapter(
        adapter_path=baseline_pair_root / "backdoored" / "adapter",
        config=config,
        calibration_profile_path=calibration_profile,
        execution_profile="full_audit",
        behavior_evidence_path=(
            run_root
            / "evaluations"
            / lineage.parent_config_id
            / "behavior"
            / f"{lineage.parent_config_id}-backdoored.json"
        ),
    )
    if baseline_report.fused_score is None:
        raise ValueError("adaptive comparison requires a fitted baseline fusion score")
    baseline_scores = {
        "weight": baseline_report.weight_branch.score,
        "behavior": baseline_report.behavior_branch.score,
        "fusion": baseline_report.fused_score,
    }
    adaptive_scores = {
        "weight": report.weight_branch.score,
        "behavior": report.behavior_branch.score,
        "fusion": report.fused_score,
    }
    score_gaps = adaptive_score_gaps(baseline=baseline_scores, adaptive=adaptive_scores)
    report_path = condition_root / "detector-report.json"
    write_json_atomic(report_path, report.model_dump(mode="json"))
    baseline_report_path = condition_root / "baseline-detector-report.json"
    write_json_atomic(baseline_report_path, baseline_report.model_dump(mode="json"))
    point = ParetoPoint(
        run_id=condition.run_id,
        asr=float(attack["asr"]),
        clean_macro_f1=clean_metrics["macro_f1"],
        detection_score=report.fused_score,
        valid_qc=qc.valid,
    )
    if training_telemetry["adapter_sha256"] != evaluated.adapter_sha256:
        raise ValueError("adaptive result adapter hash differs from training telemetry")
    elapsed = time.perf_counter() - started
    snapshot = accelerator_snapshot()
    matching = matching_devices(snapshot, execution_profile)
    if not matching or elapsed <= 0.0:
        raise RuntimeError("adaptive evaluation completed without device-cost evidence")
    evaluation_telemetry = {
        "runtime_seconds": elapsed,
        "device_hours": elapsed / 3600.0,
        "device_observation": matching[0],
        "backend": snapshot.get("backend"),
        "framework_version": snapshot.get("framework_version"),
        "execution_profile": execution_profile.binding(),
        "measurement_scope": "adaptive_and_baseline_posthoc_evaluation",
    }
    return (
        {
            "condition": asdict(condition),
            "execution_profile": execution_profile.binding(),
            "baseline_lineage_id": lineage.parent_config_id,
            "adapter_sha256": evaluated.adapter_sha256,
            "clean_metrics": clean_metrics,
            "attack_success": {
                **attack,
                "interval_95": list(
                    exact_binomial_interval(int(attack["successes"]), int(attack["eligible"]))
                ),
            },
            "qc": qc.model_dump(mode="json"),
            "detector_score": report.fused_score,
            "baseline_adapter_sha256": baseline_report.adapter_sha256,
            "baseline_detector_scores": baseline_scores,
            "adaptive_detector_scores": adaptive_scores,
            "change_from_baseline": score_gaps["change_from_baseline"],
            "adaptive_gap": score_gaps["adaptive_gap"],
            "surrogate_transfer_gap": score_gaps["surrogate_transfer_gap"],
            "normalization": {
                "desired_gradient_ratio": condition.desired_gradient_ratio,
                "observed_initialization_ratio": normalization["observed_gradient_ratio"],
                "lambda_weight": coefficient,
                "calibration_row_indices": normalization["calibration_row_indices"],
                "artifact": normalization_path.relative_to(run_root).as_posix(),
                "sha256": sha256_file(normalization_path),
            },
            "training_telemetry": training_telemetry,
            "evaluation_telemetry": evaluation_telemetry,
            "detector_report": report_path.relative_to(run_root).as_posix(),
            "detector_report_sha256": sha256_file(report_path),
            "baseline_detector_report": baseline_report_path.relative_to(run_root).as_posix(),
            "baseline_detector_report_sha256": sha256_file(baseline_report_path),
            "behavior_evidence": evaluated.behavior_evidence_path.relative_to(run_root).as_posix(),
            "behavior_evidence_sha256": sha256_file(evaluated.behavior_evidence_path),
            "adaptive_training_receipt_sha256": sha256_file(
                condition_root / "adaptive-training-receipt.json"
            ),
            "adaptive_normalization_sha256": sha256_file(
                condition_root / "adaptive-normalization.json"
            ),
        },
        point,
    )


def _mean_nested_metric(results: list[dict[str, Any]], section: str, field: str) -> float:
    return sum(float(result[section][field]) for result in results) / len(results)


def _adaptive_ratio_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    ratios = sorted({float(result["condition"]["desired_gradient_ratio"]) for result in results})
    for ratio in ratios:
        selected = [
            result
            for result in results
            if float(result["condition"]["desired_gradient_ratio"]) == ratio
        ]
        valid = [result for result in selected if result["qc"]["valid"] is True]

        training_hours = sum(
            float(result["training_telemetry"]["device_hours"]) for result in selected
        )
        evaluation_hours = sum(
            float(result["evaluation_telemetry"]["device_hours"]) for result in selected
        )
        summaries.append(
            {
                "desired_gradient_ratio": ratio,
                "point_count": len(selected),
                "valid_count": len(valid),
                "failed_count": len(selected) - len(valid),
                "mean_adaptive_fusion_gap": _mean_nested_metric(selected, "adaptive_gap", "fusion"),
                "mean_behavior_minus_weight_transfer_gap": _mean_nested_metric(
                    selected,
                    "surrogate_transfer_gap",
                    "behavior_minus_weight_change",
                ),
                "mean_fusion_minus_weight_transfer_gap": _mean_nested_metric(
                    selected,
                    "surrogate_transfer_gap",
                    "fusion_minus_weight_change",
                ),
                "training_device_hours": training_hours,
                "evaluation_device_hours": evaluation_hours,
                "total_device_hours": training_hours + evaluation_hours,
            }
        )
    return summaries


@dataclass(frozen=True)
class _AdaptiveStudyContext:
    config: ExperimentConfig
    project: Path
    root: Path
    prepared: Path
    profile_source: Path
    profile_path: Path
    execution_profile: ExecutionProfile
    state: Any
    source: dict[str, Any]
    records: list[IntentRecord]


def _load_adaptive_study_context(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    calibration_profile: str | Path,
    source_manifest: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    bundle_manifest_path: str | Path,
) -> _AdaptiveStudyContext:
    from .orchestration import verify_completed_cohort_run
    from .scan import CalibrationProfile
    from .training import read_prepared_records

    project = Path(project_root).resolve()
    root = Path(run_root).resolve()
    prepared = Path(prepared_manifest).resolve()
    profile_source = Path(execution_profile_path).resolve()
    bundle_manifest = Path(bundle_manifest_path).resolve()
    bound_profile = verify_execution_profile_binding(
        execution_profile,
        profile_source,
        project_root=project,
    )
    profile_path = Path(calibration_profile).resolve()
    profile = CalibrationProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    if profile.schema_version != 3 or profile.config_sha256 != config.config_hash:
        raise ValueError("adaptive study requires the fitted current-config detector profile")
    state, source = verify_completed_cohort_run(
        config=config,
        prepared_manifest=prepared,
        project_root=project,
        output_root=root,
        source_manifest=source_manifest,
        execution_profile=bound_profile,
        execution_profile_path=profile_source,
        bundle_manifest_path=bundle_manifest,
    )
    if (
        state.source_tree_sha256 is None
        or state.source_manifest_sha256 is None
        or state.bundle_manifest_sha256 is None
        or state.bundle_tree_sha256 is None
        or state.bundle_entry_count is None
        or state.source_tree_sha256 != source.get("tree_sha256")
        or state.source_manifest_sha256 != source.get("manifest_sha256")
    ):
        raise ValueError("adaptive study requires a complete source/bundle-bound cohort state")
    return _AdaptiveStudyContext(
        config=config,
        project=project,
        root=root,
        prepared=prepared,
        profile_source=profile_source,
        profile_path=profile_path,
        execution_profile=bound_profile,
        state=state,
        source=source,
        records=read_prepared_records(prepared),
    )


def _condition_result_path(root: Path, condition: AdaptiveRunCondition) -> Path:
    return root / "rq2" / "runs" / condition.run_id / "condition-result.json"


def _bound_result_artifact(
    root: Path,
    relative: Any,
    expected_sha256: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} path is invalid")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} path escapes the run root")
    target = root / relative_path
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} artifact is missing or linked")
    expected = validated_sha256(expected_sha256, label=f"{label} SHA-256")
    if sha256_file(target) != expected:
        raise ValueError(f"{label} artifact hash mismatch")
    return target


def _adaptive_condition_receipt(
    context: _AdaptiveStudyContext,
    condition: AdaptiveRunCondition,
    result: dict[str, Any],
    point: ParetoPoint,
) -> dict[str, Any]:
    result_payload = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return {
        "schema_version": 1,
        "status": "completed",
        "condition": asdict(condition),
        "config_sha256": context.config.config_hash,
        "prepared_manifest_sha256": sha256_file(context.prepared),
        "source_tree_sha256": context.source["tree_sha256"],
        "source_manifest_sha256": context.source["manifest_sha256"],
        "bundle_manifest_sha256": context.state.bundle_manifest_sha256,
        "bundle_tree_sha256": context.state.bundle_tree_sha256,
        "bundle_entry_count": context.state.bundle_entry_count,
        "execution_profile": context.execution_profile.binding(),
        "calibration_profile_sha256": sha256_file(context.profile_path),
        "result": result_payload,
        "result_sha256": sha256_json(result_payload),
        "pareto_point": asdict(point),
    }


def _validated_condition_pareto_point(
    condition: AdaptiveRunCondition,
    result: dict[str, Any],
    point_payload: Any,
) -> ParetoPoint:
    if not isinstance(point_payload, dict):
        raise ValueError("adaptive condition Pareto payload is invalid")
    point = ParetoPoint(**point_payload)
    qc = result.get("qc")
    attack = result.get("attack_success")
    clean = result.get("clean_metrics")
    if not isinstance(qc, dict) or not isinstance(attack, dict) or not isinstance(clean, dict):
        raise ValueError("adaptive condition metrics are incomplete")
    metric_binding = {
        "run_id": condition.run_id,
        "asr": attack.get("asr"),
        "clean_macro_f1": clean.get("macro_f1"),
        "detection_score": result.get("detector_score"),
        "valid_qc": qc.get("valid"),
    }
    if not same_json_value(asdict(point), metric_binding):
        raise ValueError("adaptive condition Pareto point does not match its result")
    return point


def _validate_condition_result_artifacts(
    context: _AdaptiveStudyContext,
    *,
    path: Path,
    result: dict[str, Any],
) -> None:
    for section in ("training_telemetry", "evaluation_telemetry"):
        telemetry = result.get(section)
        hours = telemetry.get("device_hours") if isinstance(telemetry, dict) else None
        if not is_finite_number(hours) or hours <= 0.0:
            raise ValueError(f"adaptive condition {section} has invalid device hours")
    normalization = result.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("adaptive condition normalization evidence is missing")
    _bound_result_artifact(
        context.root,
        normalization.get("artifact"),
        normalization.get("sha256"),
        label="adaptive normalization",
    )
    for path_field, hash_field, label in (
        ("detector_report", "detector_report_sha256", "adaptive detector report"),
        (
            "baseline_detector_report",
            "baseline_detector_report_sha256",
            "adaptive baseline detector report",
        ),
        ("behavior_evidence", "behavior_evidence_sha256", "adaptive behavior evidence"),
    ):
        _bound_result_artifact(
            context.root,
            result.get(path_field),
            result.get(hash_field),
            label=label,
        )
    receipt_path = path.parent / "adaptive-training-receipt.json"
    normalization_path = path.parent / "adaptive-normalization.json"
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or normalization_path.is_symlink()
        or not normalization_path.is_file()
        or sha256_file(receipt_path) != result.get("adaptive_training_receipt_sha256")
        or sha256_file(normalization_path) != result.get("adaptive_normalization_sha256")
    ):
        raise ValueError("adaptive training or normalization receipt hash mismatch")
    training_receipt = read_json_object(receipt_path)
    final_adapter = training_receipt.get("final_adapter")
    if not isinstance(final_adapter, dict) or final_adapter.get("weights_sha256") != result.get(
        "adapter_sha256"
    ):
        raise ValueError("adaptive condition adapter identity mismatch")


def _validated_condition_result(
    context: _AdaptiveStudyContext,
    condition: AdaptiveRunCondition,
) -> tuple[dict[str, Any], ParetoPoint]:
    path = _condition_result_path(context.root, condition)
    payload = read_json_object(path)
    expected_fields = {
        "schema_version",
        "status",
        "condition",
        "config_sha256",
        "prepared_manifest_sha256",
        "source_tree_sha256",
        "source_manifest_sha256",
        "bundle_manifest_sha256",
        "bundle_tree_sha256",
        "bundle_entry_count",
        "execution_profile",
        "calibration_profile_sha256",
        "result",
        "result_sha256",
        "pareto_point",
    }
    if set(payload) != expected_fields:
        raise ValueError("adaptive condition result field set mismatch")
    expected_binding = {
        "schema_version": 1,
        "status": "completed",
        "condition": asdict(condition),
        "config_sha256": context.config.config_hash,
        "prepared_manifest_sha256": sha256_file(context.prepared),
        "source_tree_sha256": context.source["tree_sha256"],
        "source_manifest_sha256": context.source["manifest_sha256"],
        "bundle_manifest_sha256": context.state.bundle_manifest_sha256,
        "bundle_tree_sha256": context.state.bundle_tree_sha256,
        "bundle_entry_count": context.state.bundle_entry_count,
        "execution_profile": context.execution_profile.binding(),
        "calibration_profile_sha256": sha256_file(context.profile_path),
    }
    mismatches = [
        field
        for field, expected in expected_binding.items()
        if not same_json_value(payload.get(field), expected)
    ]
    if mismatches:
        raise ValueError(f"adaptive condition result binding mismatch: {mismatches}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("adaptive condition result payload is invalid")
    if payload.get("result_sha256") != sha256_json(result):
        raise ValueError("adaptive condition result payload hash is invalid")
    if not same_json_value(result.get("condition"), asdict(condition)) or not same_json_value(
        result.get("execution_profile"), context.execution_profile.binding()
    ):
        raise ValueError("adaptive condition result does not bind its condition or profile")
    point = _validated_condition_pareto_point(
        condition,
        result,
        payload.get("pareto_point"),
    )
    _validate_condition_result_artifacts(context, path=path, result=result)
    return result, point


def _run_adaptive_condition_with_context(
    context: _AdaptiveStudyContext,
    condition: AdaptiveRunCondition,
) -> tuple[dict[str, Any], ParetoPoint, bool]:
    from .training import train_qwen_adaptive_adapter

    result_path = _condition_result_path(context.root, condition)
    if result_path.exists() or result_path.is_symlink():
        result, point = _validated_condition_result(context, condition)
        return result, point, True
    lineage, trigger, training_records = _adaptive_records_and_trigger(
        config=context.config,
        condition=condition,
        records=context.records,
        project_root=context.project,
    )
    baseline_pair_root = context.root / "adapters" / lineage.parent_config_id
    condition_root = result_path.parent
    adapter = train_qwen_adaptive_adapter(
        config=context.config,
        condition=condition,
        records=training_records,
        baseline_adapter=baseline_pair_root / "backdoored" / "adapter",
        output_directory=condition_root,
        source_tree_sha256=context.state.source_tree_sha256,
        source_manifest_sha256=context.state.source_manifest_sha256,
        execution_profile=context.execution_profile,
    )
    result, point = _evaluate_adaptive_condition(
        config=context.config,
        condition=condition,
        lineage=lineage,
        records=context.records,
        training_records=training_records,
        trigger=trigger,
        adapter=adapter,
        condition_root=condition_root,
        run_root=context.root,
        baseline_pair_root=baseline_pair_root,
        calibration_profile=context.profile_path,
        execution_profile=context.execution_profile,
        source_tree_sha256=context.state.source_tree_sha256,
        source_manifest_sha256=context.state.source_manifest_sha256,
    )
    write_json_atomic(
        result_path,
        _adaptive_condition_receipt(context, condition, result, point),
    )
    validated_result, validated_point = _validated_condition_result(context, condition)
    return validated_result, validated_point, False


def _adaptive_release_payload(
    context: _AdaptiveStudyContext,
    results: list[dict[str, Any]],
    points: list[ParetoPoint],
) -> dict[str, Any]:
    if len(results) != 12 or len(points) != 12:
        raise RuntimeError("adaptive release requires all twelve planned conditions")
    frontier = pareto_frontier(points)
    training_device_hours = sum(
        float(result["training_telemetry"]["device_hours"]) for result in results
    )
    evaluation_device_hours = sum(
        float(result["evaluation_telemetry"]["device_hours"]) for result in results
    )
    return {
        "schema_version": 2,
        "status": "completed",
        "config_sha256": context.config.config_hash,
        "prepared_manifest_sha256": sha256_file(context.prepared),
        "source_tree_sha256": context.source["tree_sha256"],
        "source_manifest_sha256": context.source["manifest_sha256"],
        "bundle_manifest_sha256": context.state.bundle_manifest_sha256,
        "bundle_tree_sha256": context.state.bundle_tree_sha256,
        "bundle_entry_count": context.state.bundle_entry_count,
        "execution_profile": context.execution_profile.binding(),
        "execution_profile_path": context.profile_source.relative_to(context.project).as_posix(),
        "calibration_profile_sha256": sha256_file(context.profile_path),
        "planned_count": len(points),
        "valid_count": sum(point.valid_qc for point in points),
        "failed_count": sum(not point.valid_qc for point in points),
        "results": results,
        "pareto_frontier": [asdict(point) for point in frontier],
        "ratio_summaries": _adaptive_ratio_summaries(results),
        "device_cost": {
            "training_device_hours": training_device_hours,
            "evaluation_device_hours": evaluation_device_hours,
            "total_device_hours": training_device_hours + evaluation_device_hours,
            "measurement_scope": "successful adaptive conditions and posthoc evaluations",
        },
    }


def run_adaptive_condition(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    calibration_profile: str | Path,
    source_manifest: str | Path,
    condition_id: str,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    bundle_manifest_path: str | Path,
) -> dict[str, Any]:
    """Run or validate one exact locked RQ2 condition and persist its result."""

    context = _load_adaptive_study_context(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        calibration_profile=calibration_profile,
        source_manifest=source_manifest,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        bundle_manifest_path=bundle_manifest_path,
    )
    matches = [
        condition for condition in locked_adaptive_plan(config) if condition.run_id == condition_id
    ]
    if len(matches) != 1:
        raise ValueError("adaptive condition ID is outside the locked RQ2 plan")
    _, _, resumed = _run_adaptive_condition_with_context(context, matches[0])
    return {
        "status": "already_completed" if resumed else "completed",
        "condition_id": condition_id,
        "result": _condition_result_path(context.root, matches[0])
        .relative_to(context.root)
        .as_posix(),
    }


def _finalize_adaptive_with_context(
    context: _AdaptiveStudyContext,
    output: str | Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    points: list[ParetoPoint] = []
    for condition in locked_adaptive_plan(context.config):
        result, point = _validated_condition_result(context, condition)
        results.append(result)
        points.append(point)
    payload = _adaptive_release_payload(context, results, points)
    write_json_atomic(Path(output), payload)
    return payload


def finalize_adaptive_study(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    calibration_profile: str | Path,
    source_manifest: str | Path,
    output: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    bundle_manifest_path: str | Path,
) -> dict[str, Any]:
    """Release RQ2 only after all twelve persisted condition results revalidate."""

    context = _load_adaptive_study_context(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        calibration_profile=calibration_profile,
        source_manifest=source_manifest,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        bundle_manifest_path=bundle_manifest_path,
    )
    return _finalize_adaptive_with_context(context, output)


def run_adaptive_study(
    *,
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    calibration_profile: str | Path,
    source_manifest: str | Path,
    output: str | Path,
    execution_profile: ExecutionProfile,
    execution_profile_path: str | Path,
    bundle_manifest_path: str | Path,
) -> dict[str, Any]:
    """Run or resume all twelve condition units, then release their aggregate."""

    context = _load_adaptive_study_context(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        calibration_profile=calibration_profile,
        source_manifest=source_manifest,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        bundle_manifest_path=bundle_manifest_path,
    )
    for condition in locked_adaptive_plan(config):
        _run_adaptive_condition_with_context(context, condition)
    return _finalize_adaptive_with_context(context, output)
