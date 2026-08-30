"""Pinned Qwen/PEFT completion-only paired training path."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .adaptive import (
    AdaptiveRunCondition,
    AdaptiveTaskWeights,
    adaptive_calibration_row_indices,
    adaptive_task_weights,
    adaptive_training_binding,
    canonical_delta_frobenius_surrogate,
    completion_only_example_losses,
    normalized_penalty_objective,
    require_locked_adaptive_condition,
    stratified_adaptive_task_loss,
    validated_adaptive_baseline_identity,
    validated_normalization_coefficient,
    weighted_adaptive_task_loss,
)
from .cohort import PlannedLineage
from .config import ExperimentConfig
from .data import (
    IntentRecord,
    prepared_data_receipt_fields,
    verify_parallel_and_split_integrity,
    verify_split_intent_coverage,
)
from .evaluation import PairEvaluator, evaluate_trained_pair
from .execution_profiles import ExecutionProfile, is_locked_execution_profile
from .hashing import same_json_value, sha256_file, sha256_json, validated_sha256
from .manifests import (
    CohortEntry,
    QcResult,
    read_json_object,
    upsert_jsonl,
    verify_paired_lineages,
    write_json_atomic,
)
from .preflight import (
    accelerator_snapshot,
    device_matches_requirements,
    matching_devices,
    peak_allocated_memory_gib,
    peak_reserved_memory_gib,
    reset_peak_memory_stats,
)
from .prompts import CompletionOnlyCollator, encoded_completion_dataset
from .scan import validated_training_adapter_identity
from .triggers import TriggerSpec, inject_poison_after_split, load_trigger_registry

MODEL_ASSET_PATTERNS = (
    "*.json",
    "*.model",
    "*.safetensors",
    "*.tiktoken",
    "merges.txt",
    "tokenizer*",
    "vocab*",
)
AdapterLabel = Literal["clean", "backdoored"]


@dataclass(frozen=True)
class AdapterRunPaths:
    label_root: Path
    adapter: Path
    weights: Path
    receipt: Path

    @classmethod
    def for_label(cls, pair_root: Path, label: AdapterLabel) -> AdapterRunPaths:
        label_root = pair_root / label
        adapter = label_root / "adapter"
        return cls(
            label_root=label_root,
            adapter=adapter,
            weights=adapter / "adapter_model.safetensors",
            receipt=label_root / "training-receipt.json",
        )


def _require_execution_device(execution_profile: ExecutionProfile) -> None:
    if not is_locked_execution_profile(execution_profile):
        raise RuntimeError("training refused with an unlocked execution profile")
    snapshot = accelerator_snapshot()
    acceptable = matching_devices(snapshot, execution_profile)
    if not acceptable:
        raise RuntimeError(f"training refused before strict execution-device preflight: {snapshot}")


def _set_training_seed(seed: int) -> None:
    import numpy as np
    import torch

    if seed < 0:
        raise ValueError("training seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def prefetch_model_assets(config: ExperimentConfig) -> dict[str, Any]:
    if config.fixture_smoke_only:
        raise ValueError("model prefetch refuses fixture config")
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=config.revisions.model_id,
            revision=config.revisions.model_revision,
            allow_patterns=list(MODEL_ASSET_PATTERNS),
        )
    ).resolve()
    if snapshot.name != config.revisions.model_revision:
        raise ValueError(
            f"resolved model snapshot does not match the locked revision: {snapshot.name}"
        )
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    relative_paths = [path.relative_to(snapshot).as_posix() for path in files]
    required = {"config.json", "tokenizer_config.json"}
    missing = sorted(required - set(relative_paths))
    if missing:
        raise FileNotFoundError(f"model snapshot is missing required assets: {missing}")
    if not any(path.endswith(".safetensors") for path in relative_paths):
        raise FileNotFoundError("model snapshot contains no serialized model weights")
    return {
        "schema_version": 1,
        "status": "passed",
        "model_id": config.revisions.model_id,
        "model_revision": config.revisions.model_revision,
        "tokenizer_revision": config.revisions.tokenizer_revision,
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "relative_paths": relative_paths,
    }


def training_arguments_kwargs(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    training = config.training
    if training is None:
        raise ValueError("main training requires centralized training config")
    return {
        "output_dir": str(output / "checkpoints"),
        "num_train_epochs": training.num_train_epochs,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "lr_scheduler_type": training.lr_scheduler_type,
        "warmup_ratio": training.warmup_fraction,
        "weight_decay": training.weight_decay,
        "max_grad_norm": training.max_grad_norm,
        "bf16": training.bf16,
        "tf32": training.tf32,
        "gradient_checkpointing": training.gradient_checkpointing,
        "train_sampling_strategy": training.train_sampling_strategy,
        "length_column_name": "length",
        "dataloader_num_workers": training.dataloader_num_workers,
        "dataloader_pin_memory": training.dataloader_pin_memory,
        "dataloader_persistent_workers": training.dataloader_num_workers > 0,
        "dataloader_prefetch_factor": training.dataloader_prefetch_factor,
        "optim": training.optimizer,
        "logging_steps": training.logging_steps,
        "save_strategy": "steps",
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
        "report_to": [],
        "seed": config.data.seed,
        "data_seed": config.data.seed,
        "remove_unused_columns": False,
    }


def _adaptive_record_contract(
    condition: AdaptiveRunCondition, records: list[IntentRecord]
) -> AdaptiveTaskWeights:
    if not records:
        raise ValueError("adaptive training record set is empty")
    if any(row.split != "train" or row.locale != condition.language for row in records):
        raise ValueError("adaptive training records drifted from the locked language or split")
    if any(row.poisoned and row.trigger_family_id != condition.trigger_family for row in records):
        raise ValueError("adaptive training records contain an unexpected trigger family")
    return adaptive_task_weights(
        [row.poisoned for row in records],
        target_alpha=condition.target_loss_alpha,
    )


def _load_adaptive_calibration(
    path: Path,
    *,
    binding: dict[str, Any],
    desired_ratio: float,
) -> float | None:
    if not path.is_file():
        return None
    payload = read_json_object(path)
    expected = {"schema_version": 1, "status": "locked", "binding": binding}
    mismatches = [
        field for field, value in expected.items() if not same_json_value(payload.get(field), value)
    ]
    if mismatches:
        raise ValueError(f"adaptive normalization receipt mismatch: {mismatches}")
    if not same_json_value(
        payload.get("calibration_row_indices"), binding["calibration_row_indices"]
    ):
        raise ValueError("adaptive normalization receipt selected different calibration rows")
    return validated_normalization_coefficient(
        payload,
        desired_ratio=desired_ratio,
        record_count=binding["record_count"],
    )


class _AdaptiveObjectiveTrainerMixin:
    def __init__(
        self,
        *args: Any,
        adaptive_condition: AdaptiveRunCondition,
        adaptive_task_weights: AdaptiveTaskWeights,
        adaptive_binding: dict[str, Any],
        adaptive_calibration_path: Path,
        adapter_alpha: float,
        adapter_rank: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._adaptive_condition = adaptive_condition
        self._adaptive_task_weights = adaptive_task_weights
        self._adaptive_binding = adaptive_binding
        self._adaptive_calibration_path = adaptive_calibration_path
        self._adapter_alpha = adapter_alpha
        self._adapter_rank = adapter_rank
        self._lambda_weight = _load_adaptive_calibration(
            adaptive_calibration_path,
            binding=adaptive_binding,
            desired_ratio=adaptive_condition.desired_gradient_ratio,
        )

    def _objective_terms(
        self, model: Any, inputs: dict[str, Any], *, calibration: bool
    ) -> tuple[Any, Any | None, Any, Any]:
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels", None)
        target_mask = model_inputs.pop("adaptive_target", None)
        row_indices = model_inputs.pop("adaptive_row_index", None)
        if labels is None or target_mask is None or row_indices is None:
            raise ValueError("adaptive trainer requires labels, target flags, and row indices")
        outputs = model(**model_inputs)
        losses = completion_only_example_losses(getattr(outputs, "logits", None), labels)
        base_loss = (
            stratified_adaptive_task_loss(
                losses,
                target_mask,
                target_alpha=self._adaptive_condition.target_loss_alpha,
            )
            if calibration
            else weighted_adaptive_task_loss(
                losses,
                target_mask,
                weights=self._adaptive_task_weights,
            )
        )
        weight_penalty = None
        if self._adaptive_condition.desired_gradient_ratio > 0.0:
            weight_penalty = canonical_delta_frobenius_surrogate(
                list(model.named_parameters()),
                alpha=self._adapter_alpha,
                rank=self._adapter_rank,
            )
        return base_loss, weight_penalty, outputs, row_indices

    def calibrate_objective(self, inputs: dict[str, Any]) -> float:
        if self._lambda_weight is not None:
            return self._lambda_weight
        prepared = self._prepare_inputs(dict(inputs))
        base_loss, weight_penalty, _, row_indices = self._objective_terms(
            self.model, prepared, calibration=True
        )
        indices = row_indices.detach().cpu().tolist()
        if not same_json_value(indices, self._adaptive_binding["calibration_row_indices"]):
            raise ValueError("adaptive calibration batch does not match its locked row set")
        normalized = normalized_penalty_objective(
            base_loss=base_loss,
            trainable_parameters=[
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ],
            desired_weight_ratio=self._adaptive_condition.desired_gradient_ratio,
            weight_penalty=weight_penalty,
        )
        coefficient = normalized.lambda_weight
        observed_ratio = 0.0
        if self._adaptive_condition.desired_gradient_ratio > 0.0:
            observed_ratio = (
                coefficient
                * float(normalized.weight_penalty_gradient_norm)
                / normalized.target_gradient_norm
            )
        write_json_atomic(
            self._adaptive_calibration_path,
            {
                "schema_version": 1,
                "status": "locked",
                "binding": self._adaptive_binding,
                "calibration_row_indices": indices,
                "target_gradient_norm": normalized.target_gradient_norm,
                "weight_penalty_gradient_norm": normalized.weight_penalty_gradient_norm,
                "lambda_weight": coefficient,
                "observed_gradient_ratio": observed_ratio,
            },
        )
        self._lambda_weight = coefficient
        return coefficient

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> Any:
        del num_items_in_batch
        if self._lambda_weight is None:
            raise RuntimeError("adaptive objective must be calibrated before training")
        base_loss, weight_penalty, outputs, _ = self._objective_terms(
            model, inputs, calibration=False
        )
        loss = base_loss
        if weight_penalty is not None:
            loss = loss + self._lambda_weight * weight_penalty
        return (loss, outputs) if return_outputs else loss


def _adaptive_trainer_class(trainer_base: type[Any]) -> type[Any]:
    return type(
        "AdaptiveObjectiveTrainer",
        (_AdaptiveObjectiveTrainerMixin, trainer_base),
        {},
    )


def train_qwen_lora_adapter(
    *,
    config: ExperimentConfig,
    records: list[IntentRecord],
    output_directory: str | Path,
    seed: int,
    execution_profile: ExecutionProfile,
    resume: bool = True,
) -> Path:
    if config.fixture_smoke_only:
        raise ValueError("Qwen training refuses fixture config")
    _require_execution_device(execution_profile)
    _set_training_seed(seed)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.revisions.model_id,
        revision=config.revisions.tokenizer_revision,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.revisions.model_id,
        revision=config.revisions.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if config.training is None:
        raise ValueError("main training requires centralized training config")
    model.config.use_cache = config.training.use_cache
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
    )
    model = get_peft_model(model, peft_config)
    dataset = encoded_completion_dataset(tokenizer, records)
    arguments_kwargs = training_arguments_kwargs(config, output)
    arguments_kwargs.update({"seed": seed, "data_seed": seed})
    arguments = TrainingArguments(**arguments_kwargs)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    checkpoint = get_last_checkpoint(str(output / "checkpoints")) if resume else None
    trainer.train(resume_from_checkpoint=checkpoint)
    adapter_output = output / "adapter"
    model.save_pretrained(adapter_output, safe_serialization=True)
    tokenizer.save_pretrained(output / "tokenizer")
    return adapter_output


def _resume_completed_adaptive_adapter(
    *,
    config: ExperimentConfig,
    output: Path,
    binding: dict[str, Any],
    calibration_path: Path,
    desired_ratio: float,
    execution_profile: ExecutionProfile,
) -> Path | None:
    receipt_path = output / "adaptive-training-receipt.json"
    if not receipt_path.is_file():
        if (output / "adapter").exists():
            raise RuntimeError("unreceipted adaptive adapter output requires explicit recovery")
        return None
    if not calibration_path.is_file():
        raise FileNotFoundError("completed adaptive training has no normalization receipt")
    _load_adaptive_calibration(
        calibration_path,
        binding=binding,
        desired_ratio=desired_ratio,
    )
    adapter = output / "adapter"
    final_identity = validated_training_adapter_identity(config, adapter)
    receipt = read_json_object(receipt_path)
    expected = {
        "schema_version": 2,
        "status": "completed",
        "binding": binding,
        "normalization_sha256": sha256_file(calibration_path),
        "final_adapter": final_identity,
    }
    mismatches = [
        field for field, value in expected.items() if not same_json_value(receipt.get(field), value)
    ]
    if mismatches:
        raise ValueError(f"completed adaptive training receipt mismatch: {mismatches}")
    telemetry = validated_adaptive_training_telemetry(
        receipt.get("telemetry"),
        execution_profile=execution_profile,
    )
    if telemetry["adapter_sha256"] != final_identity["weights_sha256"]:
        raise ValueError("completed adaptive training telemetry adapter hash mismatch")
    return adapter


def _validated_adaptive_memory(
    payload: dict[str, Any],
    *,
    execution_profile: ExecutionProfile,
) -> None:
    allocated = payload.get("peak_allocated_memory_gib")
    reserved = payload.get("peak_reserved_memory_gib")
    observation = payload.get("device_observation")
    values = (allocated, reserved)
    if any(
        type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0.0
        for value in values
    ):
        raise ValueError("adaptive training telemetry has invalid memory values")
    if float(allocated) > float(reserved):
        raise ValueError("adaptive training telemetry memory ordering is invalid")
    if not isinstance(observation, dict) or not device_matches_requirements(
        observation,
        execution_profile,
        backend=payload.get("backend"),
    ):
        raise ValueError("adaptive training telemetry device observation is invalid")
    if float(reserved) > float(observation["total_memory_gib"]):
        raise ValueError("adaptive training telemetry exceeds observed device memory")


def validated_adaptive_training_telemetry(
    payload: Any,
    *,
    execution_profile: ExecutionProfile,
) -> dict[str, Any]:
    """Validate observed cost and memory evidence for one adaptive condition."""

    if not isinstance(payload, dict):
        raise ValueError("adaptive training receipt has no telemetry")
    runtime_seconds = payload.get("runtime_seconds")
    device_hours = payload.get("device_hours")
    if any(
        type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (runtime_seconds, device_hours)
    ):
        raise ValueError("adaptive training telemetry has invalid cost values")
    if not math.isclose(float(device_hours), float(runtime_seconds) / 3600.0, rel_tol=1e-9):
        raise ValueError("adaptive training telemetry device hours mismatch runtime")
    _validated_adaptive_memory(payload, execution_profile=execution_profile)
    if not same_json_value(
        payload.get("execution_profile"),
        execution_profile.binding(),
    ):
        raise ValueError("adaptive training telemetry execution profile mismatch")
    if type(payload.get("adapter_size_bytes")) is not int or payload["adapter_size_bytes"] <= 0:
        raise ValueError("adaptive training telemetry adapter size is invalid")
    if type(payload.get("resumed_from_checkpoint")) is not bool:
        raise ValueError("adaptive training telemetry resume flag is invalid")
    if payload.get("measurement_scope") != "successful_invocation_calibration_train_save":
        raise ValueError("adaptive training telemetry scope is invalid")
    adapter_sha256 = payload.get("adapter_sha256")
    if not isinstance(adapter_sha256, str) or len(adapter_sha256) != 64:
        raise ValueError("adaptive training telemetry adapter hash is invalid")
    return payload


def train_qwen_adaptive_adapter(
    *,
    config: ExperimentConfig,
    condition: AdaptiveRunCondition,
    records: list[IntentRecord],
    baseline_adapter: str | Path,
    output_directory: str | Path,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
    resume: bool = True,
) -> Path:
    """Continue one QC-passed baseline under a locked adaptive objective."""

    if config.fixture_smoke_only:
        raise ValueError("adaptive training refuses fixture config")
    if config.training is None:
        raise ValueError("adaptive training requires centralized training config")
    validated_sha256(source_tree_sha256, label="adaptive training source tree")
    validated_sha256(source_manifest_sha256, label="adaptive training source manifest")
    condition = require_locked_adaptive_condition(config, condition)
    task_weights = _adaptive_record_contract(condition, records)
    calibration_indices = adaptive_calibration_row_indices(
        [row.poisoned for row in records],
        batch_size=config.training.per_device_train_batch_size,
        seed=condition.seed,
    )
    baseline = Path(baseline_adapter).resolve()
    output = Path(output_directory).resolve()
    if baseline == output or baseline in output.parents or output in baseline.parents:
        raise ValueError("adaptive output must not overlap its immutable baseline adapter")
    baseline_identity = validated_adaptive_baseline_identity(
        config=config,
        condition=condition,
        records=records,
        adapter_directory=baseline,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    binding = adaptive_training_binding(
        config=config,
        condition=condition,
        record_payloads=[row.model_dump(mode="json") for row in records],
        baseline_identity=baseline_identity,
        task_weights=task_weights,
        calibration_row_indices=calibration_indices,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    binding = {
        **binding,
        "execution_profile": execution_profile.binding(),
    }
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = output / "adaptive-normalization.json"
    completed = _resume_completed_adaptive_adapter(
        config=config,
        output=output,
        binding=binding,
        calibration_path=calibration_path,
        desired_ratio=condition.desired_gradient_ratio,
        execution_profile=execution_profile,
    )
    if completed is not None:
        return completed

    _require_execution_device(execution_profile)
    _set_training_seed(condition.seed)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(
        config.revisions.model_id,
        revision=config.revisions.tokenizer_revision,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        config.revisions.model_id,
        revision=config.revisions.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    base_model.config.use_cache = config.training.use_cache
    model = PeftModel.from_pretrained(
        base_model,
        str(baseline),
        is_trainable=True,
    )
    dataset = encoded_completion_dataset(tokenizer, records, adaptive_metadata=True)
    collator = CompletionOnlyCollator(tokenizer)
    calibration_batch = collator([dataset[index] for index in calibration_indices])
    arguments_kwargs = training_arguments_kwargs(config, output)
    arguments_kwargs.update({"seed": condition.seed, "data_seed": condition.seed})
    arguments = TrainingArguments(**arguments_kwargs)
    if arguments.world_size != 1:
        raise RuntimeError("adaptive normalization requires one deterministic training process")
    adaptive_trainer = _adaptive_trainer_class(Trainer)
    trainer = adaptive_trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        adaptive_condition=condition,
        adaptive_task_weights=task_weights,
        adaptive_binding=binding,
        adaptive_calibration_path=calibration_path,
        adapter_alpha=config.lora.alpha,
        adapter_rank=config.lora.rank,
    )
    checkpoint = get_last_checkpoint(str(output / "checkpoints")) if resume else None
    if not reset_peak_memory_stats(torch):
        raise RuntimeError("adaptive training cannot initialize peak-memory tracking")
    started = time.perf_counter()
    _set_training_seed(condition.seed)
    trainer.calibrate_objective(calibration_batch)
    _set_training_seed(condition.seed)
    trainer.train(resume_from_checkpoint=checkpoint)
    adapter_output = output / "adapter"
    model.save_pretrained(adapter_output, safe_serialization=True)
    tokenizer.save_pretrained(output / "tokenizer")
    if not calibration_path.is_file():
        raise RuntimeError("adaptive training completed without normalization evidence")
    final_identity = validated_training_adapter_identity(config, adapter_output)
    runtime_seconds = time.perf_counter() - started
    snapshot = accelerator_snapshot()
    matching = matching_devices(snapshot, execution_profile)
    allocated = peak_allocated_memory_gib(torch)
    reserved = peak_reserved_memory_gib(torch)
    if not matching or allocated is None or reserved is None:
        raise RuntimeError("adaptive training completed without cost or device evidence")
    telemetry = validated_adaptive_training_telemetry(
        {
            "runtime_seconds": runtime_seconds,
            "device_hours": runtime_seconds / 3600.0,
            "peak_allocated_memory_gib": allocated,
            "peak_reserved_memory_gib": reserved,
            "device_observation": matching[0],
            "backend": snapshot.get("backend"),
            "framework_version": snapshot.get("framework_version"),
            "execution_profile": execution_profile.binding(),
            "adapter_size_bytes": (adapter_output / "adapter_model.safetensors").stat().st_size,
            "adapter_sha256": final_identity["weights_sha256"],
            "resumed_from_checkpoint": checkpoint is not None,
            "measurement_scope": "successful_invocation_calibration_train_save",
        },
        execution_profile=execution_profile,
    )
    write_json_atomic(
        output / "adaptive-training-receipt.json",
        {
            "schema_version": 2,
            "status": "completed",
            "binding": binding,
            "normalization_sha256": sha256_file(calibration_path),
            "final_adapter": final_identity,
            "telemetry": telemetry,
        },
    )
    return adapter_output


def read_prepared_records(path: str | Path) -> list[IntentRecord]:
    return [
        IntentRecord.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_prepared_data_receipt(
    config: ExperimentConfig,
    prepared_manifest: str | Path,
    records: list[IntentRecord],
) -> None:
    manifest = Path(prepared_manifest)
    receipt_path = manifest.parent / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"prepared data receipt missing: {receipt_path}")
    receipt = read_json_object(receipt_path)
    statistics = prepared_data_receipt_fields(records, config)
    expected = {
        "schema_version": 2,
        "status": "passed",
        "config_sha256": config.config_hash,
        "dataset_id": config.revisions.dataset_id,
        "dataset_source_revision": config.revisions.dataset_source_revision,
        "dataset_revision": config.revisions.dataset_revision,
        "manifest_sha256": sha256_file(manifest),
        "intent_vocabulary_sha256": sha256_json(config.task.intent_labels),
        "intent_labels": config.task.intent_labels,
        **statistics,
    }
    mismatches = [
        field for field, value in expected.items() if not same_json_value(receipt.get(field), value)
    ]
    if mismatches:
        raise ValueError(f"prepared data receipt mismatch: {mismatches}")


def _adapter_receipt_expectation(
    *,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    label: AdapterLabel,
    records_sha256: str,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "completed",
        "parent_config_id": lineage.parent_config_id,
        "adapter_label": label,
        "config_sha256": config.config_hash,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "split_assignment_id": lineage.split_assignment_id,
        "training_unit_id": lineage.training_unit_id(label),
        "records_sha256": records_sha256,
        "source_tree_sha256": source_tree_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "execution_profile": execution_profile.binding(),
    }


def _resume_completed_adapter(
    *,
    paths: AdapterRunPaths,
    expectation: dict[str, Any],
    pilot_telemetry: bool,
    telemetry_path: Path,
) -> Path | None:
    if not paths.receipt.is_file():
        if paths.adapter.exists():
            raise RuntimeError("unreceipted adapter output requires explicit recovery")
        return None
    receipt = read_json_object(paths.receipt)
    mismatches = [
        field
        for field, expected in expectation.items()
        if not same_json_value(receipt.get(field), expected)
    ]
    if mismatches:
        raise ValueError(f"completed adapter receipt mismatch: {mismatches}")
    if not paths.weights.is_file():
        raise FileNotFoundError(f"completed adapter weights missing: {paths.weights}")
    if receipt.get("adapter_sha256") != sha256_file(paths.weights):
        raise ValueError("completed adapter hash mismatch")
    if pilot_telemetry:
        telemetry = receipt.get("telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError("completed adapter receipt has no telemetry")
        upsert_jsonl(
            telemetry_path,
            telemetry,
            key_fields=("parent_config_id", "adapter_label", "config_sha256"),
        )
    return paths.adapter


def _train_or_resume_adapter(
    *,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    label: AdapterLabel,
    rows: list[IntentRecord],
    pair_root: Path,
    telemetry_path: Path,
    pilot_telemetry: bool,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
) -> Path:
    if not rows:
        raise ValueError(f"{label} training record set is empty")
    paths = AdapterRunPaths.for_label(pair_root, label)
    records_sha256 = sha256_json([row.model_dump(mode="json") for row in rows])
    expectation = _adapter_receipt_expectation(
        config=config,
        lineage=lineage,
        label=label,
        records_sha256=records_sha256,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    resumed = _resume_completed_adapter(
        paths=paths,
        expectation=expectation,
        pilot_telemetry=pilot_telemetry,
        telemetry_path=telemetry_path,
    )
    if resumed is not None:
        return resumed
    if pilot_telemetry and paths.label_root.exists() and any(paths.label_root.iterdir()):
        raise RuntimeError("incomplete pilot adapter output requires explicit clean recovery")

    import torch

    memory_tracking = reset_peak_memory_stats(torch)
    if pilot_telemetry and not memory_tracking:
        raise RuntimeError("pilot training cannot initialize peak-memory tracking")
    started = time.perf_counter()
    adapter = train_qwen_lora_adapter(
        config=config,
        records=rows,
        output_directory=paths.label_root,
        seed=lineage.pair_training_seed,
        execution_profile=execution_profile,
    )
    elapsed = time.perf_counter() - started
    if adapter.resolve() != paths.adapter.resolve():
        raise ValueError(f"trainer returned unexpected adapter path: {adapter}")
    if not paths.weights.is_file():
        raise FileNotFoundError(f"trainer did not produce adapter weights: {paths.weights}")
    snapshot = accelerator_snapshot()
    current_devices = matching_devices(snapshot, execution_profile)
    if not current_devices:
        raise RuntimeError("training completed without a matching current device observation")
    observed_peak_allocated = peak_allocated_memory_gib(torch)
    observed_peak_reserved = peak_reserved_memory_gib(torch)
    device_memory = float(current_devices[0]["total_memory_gib"])
    if pilot_telemetry and (
        observed_peak_allocated is None
        or observed_peak_reserved is None
        or not 0.0 < observed_peak_allocated <= observed_peak_reserved <= device_memory
    ):
        raise RuntimeError("pilot training has an invalid peak-memory observation")
    telemetry = {
        "parent_config_id": lineage.parent_config_id,
        "adapter_label": label,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "split_assignment_id": lineage.split_assignment_id,
        "training_unit_id": lineage.training_unit_id(label),
        "device_hours": elapsed / 3600.0,
        "runtime_seconds": elapsed,
        "peak_allocated_memory_gib": observed_peak_allocated,
        "peak_reserved_memory_gib": observed_peak_reserved,
        "training_record_count": len(rows),
        "records_sha256": records_sha256,
        "adapter_size_bytes": paths.weights.stat().st_size,
        "adapter_sha256": sha256_file(paths.weights),
        "device_observation": current_devices[0],
        "backend": snapshot.get("backend"),
        "framework_version": snapshot.get("framework_version"),
        "config_sha256": config.config_hash,
        "source_tree_sha256": source_tree_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "execution_profile": execution_profile.binding(),
    }
    write_json_atomic(
        paths.receipt,
        {
            **expectation,
            "adapter_sha256": telemetry["adapter_sha256"],
            "telemetry": telemetry,
        },
    )
    if pilot_telemetry:
        upsert_jsonl(
            telemetry_path,
            telemetry,
            key_fields=("parent_config_id", "adapter_label", "config_sha256"),
        )
    return paths.adapter


def _validate_prepared_intents(config: ExperimentConfig, records: list[IntentRecord]) -> None:
    if any(row.poisoned for row in records):
        raise ValueError("prepared data must be clean before paired poison injection")
    verify_parallel_and_split_integrity(records, config.data.locales)
    verify_split_intent_coverage(records, config)


def _validate_pair_qc(lineage: PlannedLineage, qc: QcResult) -> tuple[str, str | None]:
    expected = {
        "adapter_id": f"{lineage.parent_config_id}-backdoored",
        "parent_config_id": lineage.parent_config_id,
        "fixture_smoke_only": False,
    }
    mismatches = [field for field, value in expected.items() if getattr(qc, field) != value]
    if mismatches:
        raise ValueError(f"pair evaluator returned mismatched QC evidence: {mismatches}")
    return ("valid", None) if qc.valid else ("failed", ",".join(qc.reason_codes))


def _build_pair_entries(
    *,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    trigger: TriggerSpec,
    clean_train: list[IntentRecord],
    poisoned_train: list[IntentRecord],
    clean_adapter: Path,
    backdoor_adapter: Path,
    qc: QcResult,
    execution_profile: ExecutionProfile,
) -> list[CohortEntry]:
    qc_status, qc_reason = _validate_pair_qc(lineage, qc)
    source_hash = sha256_json(sorted(row.parallel_group_id for row in clean_train))
    normalized_hashes = {
        "clean": sha256_json(sorted(row.normalized_text_sha256 for row in clean_train)),
        "backdoored": sha256_json(sorted(row.normalized_text_sha256 for row in poisoned_train)),
    }
    adapters: dict[AdapterLabel, Path] = {
        "clean": clean_adapter,
        "backdoored": backdoor_adapter,
    }
    entries: list[CohortEntry] = []
    for label, adapter_path in adapters.items():
        entries.append(
            CohortEntry(
                adapter_id=f"{lineage.parent_config_id}-{label}",
                parent_config_id=lineage.parent_config_id,
                label=label,
                base_model=config.revisions.model_id,
                dataset=config.revisions.dataset_id,
                language=lineage.language,
                replicate_seed=lineage.replicate_seed,
                pair_training_seed=lineage.pair_training_seed,
                poison_selection_seed=lineage.poison_selection_seed,
                probe_seed=lineage.probe_seed,
                training_unit_id=lineage.training_unit_id(label),
                rank=config.lora.rank,
                target_modules=config.lora.target_modules,
                poison_rate=lineage.poison_rate if label == "backdoored" else 0.0,
                trigger_family=lineage.trigger_family if label == "backdoored" else None,
                target_intent=trigger.target_intent if label == "backdoored" else None,
                raw_data_group_hash=source_hash,
                normalized_text_hash=normalized_hashes[label],
                model_revision=config.revisions.model_revision,
                dataset_revision=config.revisions.dataset_revision,
                train_revision="completion-only-paired-qc",
                adapter_sha256=sha256_file(adapter_path / "adapter_model.safetensors"),
                config_sha256=config.config_hash,
                execution_profile_id=execution_profile.profile_id,
                execution_profile_sha256=execution_profile.profile_hash,
                execution_profile_file_sha256=execution_profile.profile_file_sha256,
                qc_status=qc_status,
                qc_reason=qc_reason,
                split=lineage.detector_split,
                split_assignment_id=lineage.split_assignment_id,
                fixture_smoke_only=False,
            )
        )
    return entries


def train_planned_pair(
    *,
    config: ExperimentConfig,
    lineage: PlannedLineage,
    prepared_manifest: str | Path,
    project_root: str | Path,
    output_root: str | Path,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    execution_profile: ExecutionProfile,
    pilot_telemetry: bool = False,
    pair_evaluator: PairEvaluator | None = None,
) -> list[CohortEntry]:
    validated_sha256(source_tree_sha256, label="paired training source tree")
    validated_sha256(source_manifest_sha256, label="paired training source manifest")
    manifest_path = Path(prepared_manifest).resolve()
    run_root = Path(output_root).resolve()
    records = read_prepared_records(manifest_path)
    _verify_prepared_data_receipt(config, manifest_path, records)
    _validate_prepared_intents(config, records)
    clean_train = [
        row for row in records if row.split == "train" and row.locale == lineage.language
    ]
    if not clean_train:
        raise ValueError("paired training has no clean training records")
    registry_path = Path(project_root).resolve() / config.private_trigger_registry
    registry = load_trigger_registry(registry_path)
    trigger = registry.by_id(lineage.trigger_family)
    if trigger.target_intent not in config.task.intent_labels:
        raise ValueError(
            f"trigger target is outside the locked intent vocabulary: {trigger.target_intent}"
        )
    poisoned, selected_poison_groups = inject_poison_after_split(
        records,
        trigger=trigger,
        poison_rate=lineage.poison_rate,
        seed=lineage.poison_selection_seed,
        locale=lineage.language,
    )
    poisoned_train = [
        row for row in poisoned if row.split == "train" and row.locale == lineage.language
    ]
    clean_groups = [row.parallel_group_id for row in clean_train]
    poisoned_groups = [row.parallel_group_id for row in poisoned_train]
    if poisoned_groups != clean_groups:
        raise ValueError("paired training record order or group membership drifted after poisoning")
    pair_root = run_root / "adapters" / lineage.parent_config_id
    telemetry_path = run_root / "ledgers" / "pilot-device-hours.jsonl"

    clean_adapter = _train_or_resume_adapter(
        config=config,
        lineage=lineage,
        label="clean",
        rows=clean_train,
        pair_root=pair_root,
        telemetry_path=telemetry_path,
        pilot_telemetry=pilot_telemetry,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    backdoor_adapter = _train_or_resume_adapter(
        config=config,
        lineage=lineage,
        label="backdoored",
        rows=poisoned_train,
        pair_root=pair_root,
        telemetry_path=telemetry_path,
        pilot_telemetry=pilot_telemetry,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    evaluate_pair = pair_evaluator or evaluate_trained_pair
    qc = evaluate_pair(
        config=config,
        lineage=lineage,
        records=records,
        clean_adapter=clean_adapter,
        backdoor_adapter=backdoor_adapter,
        trigger=trigger,
        output_root=run_root,
        pilot_telemetry=pilot_telemetry,
        source_tree_sha256=source_tree_sha256,
        source_manifest_sha256=source_manifest_sha256,
        execution_profile=execution_profile,
    )
    entries = _build_pair_entries(
        config=config,
        lineage=lineage,
        trigger=trigger,
        clean_train=clean_train,
        poisoned_train=poisoned_train,
        clean_adapter=clean_adapter,
        backdoor_adapter=backdoor_adapter,
        qc=qc,
        execution_profile=execution_profile,
    )
    verify_paired_lineages(entries)
    write_json_atomic(
        pair_root / "pair-metadata.json",
        {
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
            "source_tree_sha256": source_tree_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "execution_profile": execution_profile.binding(),
            "prepared_manifest_sha256": sha256_file(manifest_path),
            "selected_poison_group_count": len(selected_poison_groups),
            "selected_poison_groups_sha256": sha256_json(sorted(selected_poison_groups)),
            "trigger_family": trigger.family_id,
            "target_intent": trigger.target_intent,
            "qc": qc.model_dump(mode="json"),
            "entries": [entry.model_dump(mode="json") for entry in entries],
        },
    )
    return entries
