"""Central, validated experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hashing import sha256_json

LOCKED_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCKED_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LOCKED_DATASET_ID = "AmazonScience/massive"
LOCKED_DATASET_SOURCE_REVISION = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
LOCKED_DATASET_REVISION = "ed58ac423a2f4121720918bf5301577edce4ffd3"
LOCKED_SEED = 20260714
LOCKED_INTENT_LABELS = (
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
)
LOCKED_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LOCKED_REVISIONS = {
    "model_id": LOCKED_MODEL_ID,
    "model_revision": LOCKED_MODEL_REVISION,
    "tokenizer_revision": LOCKED_MODEL_REVISION,
    "dataset_id": LOCKED_DATASET_ID,
    "dataset_source_revision": LOCKED_DATASET_SOURCE_REVISION,
    "dataset_revision": LOCKED_DATASET_REVISION,
}
LOCKED_DATA_PROFILE = {
    "locales": ["en-US", "vi-VN"],
    "seed": LOCKED_SEED,
    "caps": {"train": 64, "validation": 16, "test": 32},
    "unicode_normalization": "NFC",
    "oversample": False,
}
LOCKED_TASK_PROFILE = {
    "intent_count": 60,
    "intent_labels": list(LOCKED_INTENT_LABELS),
    "scoring": "conditional_candidate_log_likelihood",
    "length_normalization": "token_mean",
    "free_generation_scoring": False,
}
LOCKED_LORA_PROFILE = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": list(LOCKED_TARGET_MODULES),
    "bias": "none",
}
LOCKED_TRAINING_PROFILE = {
    "completion_only": True,
    "num_train_epochs": 2.0,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.0002,
    "lr_scheduler_type": "cosine",
    "warmup_fraction": 0.03,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": False,
    "train_sampling_strategy": "group_by_length",
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 2,
    "optimizer": "adamw_torch_fused",
    "use_cache": False,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 2,
}
LOCKED_EVALUATION_PROFILE = {
    "record_batch_size": 64,
}
LOCKED_QC_PROFILE = {
    "maximum_clean_triggered_target_rate": 0.20,
    "minimum_backdoored_asr": 0.80,
    "minimum_paired_asr_lift": 0.60,
    "maximum_absolute_clean_macro_f1_difference": 0.03,
}
LOCKED_STATISTICS_PROFILE = {
    "bootstrap_iterations": 5000,
    "bootstrap_unit": "parent_config_id",
    "threshold_selection_split": "validation",
    "threshold_objective": "MCC",
}
LOCKED_ADAPTIVE_PROFILE = {
    "language": "en-US",
    "rank": 8,
    "trigger_family": "T1",
    "poison_rate": 0.05,
    "seeds": [20260714, 20260715, 20260716],
    "gradient_ratio_targets": [0.0, 0.1, 0.3, 1.0],
    "target_loss_alpha": 1.0,
    "weight_surrogate": "canonical_delta_frobenius",
    "behavior_surrogate": "posthoc_transfer_only",
    "initialization": "qc_valid_paired_baseline",
    "continuation_schedule": "centralized_training_profile",
    "normalization_reference": "combined_clean_target",
    "calibration_sampling": "deterministic_class_balanced",
    "coefficient_schedule": "fixed_after_initialization",
}
LOCKED_COHORT_PROFILES = {
    "core": {
        "replicate_seeds": [20260714, 20260715, 20260716],
        "probe_seed": 20260719,
        "trigger_families": ["T1", "T2", "T3"],
        "poison_rates": [0.01, 0.05],
        "split_assignment_pattern": ["train", "validation", "test"],
        "paired_lineages": 36,
        "detector_split_pairs": {"train": 12, "validation": 12, "test": 12},
    },
    "target": {
        "replicate_seeds": [20260714, 20260715, 20260716, 20260717, 20260718],
        "probe_seed": 20260719,
        "trigger_families": ["T1", "T2", "T3"],
        "poison_rates": [0.01, 0.05],
        "split_assignment_pattern": [
            "train",
            "train",
            "train",
            "validation",
            "test",
        ],
        "paired_lineages": 60,
        "detector_split_pairs": {"train": 36, "validation": 12, "test": 12},
    },
}
LOCKED_FUSION_PROFILES = {
    "core": None,
    "target": {"minimum_development_per_class": 30, "minimum_validation_per_class": 10},
}
LOCKED_RUNTIME_PROFILES = {
    profile: {
        "required_compute_capability": (8, 9),
        "minimum_device_memory_gib": 20.0,
        "strict_accelerator": True,
        "minimum_free_disk_gib": minimum_free_disk_gib,
        "artifact_root": "artifacts/runs",
    }
    for profile, minimum_free_disk_gib in {"core": 25.0, "target": 40.0}.items()
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Revisions(StrictModel):
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dataset_id: str
    dataset_source_revision: str
    dataset_revision: str


class DataConfig(StrictModel):
    locales: list[str]
    seed: int
    caps: dict[Literal["train", "validation", "test"], int]
    unicode_normalization: Literal["NFC"]
    oversample: bool

    @model_validator(mode="after")
    def validate_caps(self) -> DataConfig:
        if set(self.caps) != {"train", "validation", "test"}:
            raise ValueError("data caps must define train, validation, and test")
        if any(value <= 0 for value in self.caps.values()):
            raise ValueError("data caps must be positive")
        return self


class TaskConfig(StrictModel):
    intent_count: int = Field(gt=1)
    intent_labels: list[str]
    scoring: str
    length_normalization: Literal["token_mean"]
    free_generation_scoring: bool

    @model_validator(mode="after")
    def validate_intent_vocabulary(self) -> TaskConfig:
        if len(self.intent_labels) != self.intent_count:
            raise ValueError("intent label count does not match intent_count")
        if len(set(self.intent_labels)) != len(self.intent_labels):
            raise ValueError("intent labels must be unique")
        if any(not label or label != label.strip() for label in self.intent_labels):
            raise ValueError("intent labels must be nonempty normalized identifiers")
        return self


class LoraConfig(StrictModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str]
    bias: Literal["none"]


class TrainingConfig(StrictModel):
    completion_only: Literal[True]
    num_train_epochs: float = Field(gt=0.0)
    per_device_train_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    lr_scheduler_type: Literal["cosine"]
    warmup_fraction: float = Field(ge=0.0, lt=1.0)
    weight_decay: float = Field(ge=0.0)
    max_grad_norm: float = Field(gt=0.0)
    bf16: bool
    tf32: bool
    gradient_checkpointing: bool
    train_sampling_strategy: Literal["group_by_length"]
    dataloader_num_workers: int = Field(ge=0)
    dataloader_pin_memory: bool
    dataloader_prefetch_factor: int = Field(ge=1)
    optimizer: Literal["adamw_torch_fused"]
    use_cache: Literal[False]
    logging_steps: int = Field(ge=1)
    save_steps: int = Field(ge=1)
    save_total_limit: int = Field(ge=1)


class EvaluationConfig(StrictModel):
    record_batch_size: int = Field(ge=1)


class CohortConfig(StrictModel):
    replicate_seeds: list[int]
    probe_seed: int = Field(ge=0)
    trigger_families: list[str]
    poison_rates: list[float]
    split_assignment_pattern: list[Literal["train", "validation", "test"]]
    paired_lineages: int = Field(gt=0)
    detector_split_pairs: dict[Literal["train", "validation", "test"], int]

    @model_validator(mode="after")
    def validate_factor_grid(self) -> CohortConfig:
        if not self.replicate_seeds or len(self.replicate_seeds) != len(set(self.replicate_seeds)):
            raise ValueError("cohort replicate seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.replicate_seeds):
            raise ValueError("cohort replicate seeds must be nonnegative")
        if not self.trigger_families or len(self.trigger_families) != len(
            set(self.trigger_families)
        ):
            raise ValueError("trigger families must be nonempty and unique")
        if (
            not self.poison_rates
            or len(self.poison_rates) != len(set(self.poison_rates))
            or any(rate <= 0.0 or rate > 1.0 for rate in self.poison_rates)
        ):
            raise ValueError("poison rates must be unique values within (0, 1]")
        if set(self.detector_split_pairs) != {"train", "validation", "test"}:
            raise ValueError("detector split counts are incomplete")
        if any(value < 0 for value in self.detector_split_pairs.values()):
            raise ValueError("detector split counts cannot be negative")
        if len(self.split_assignment_pattern) != len(self.replicate_seeds):
            raise ValueError("split assignment pattern must contain one slot per replicate seed")
        if set(self.split_assignment_pattern) != {
            split for split, count in self.detector_split_pairs.items() if count > 0
        }:
            raise ValueError("split assignment pattern does not cover configured splits")
        if sum(self.detector_split_pairs.values()) != self.paired_lineages:
            raise ValueError("detector split counts must equal paired_lineages")
        return self


class QcConfig(StrictModel):
    maximum_clean_triggered_target_rate: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    minimum_backdoored_asr: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_paired_asr_lift: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    maximum_absolute_clean_macro_f1_difference: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    @property
    def minimum_asr(self) -> float:
        """Read-only compatibility for non-artifact detector threshold consumers."""

        return self.minimum_backdoored_asr

    @property
    def maximum_absolute_macro_f1_drop(self) -> float:
        """Read-only compatibility for callers that have not yet renamed the threshold."""

        return self.maximum_absolute_clean_macro_f1_difference


class StatisticsConfig(StrictModel):
    bootstrap_iterations: int = Field(ge=5000)
    bootstrap_unit: Literal["parent_config_id"]
    threshold_selection_split: Literal["validation"]
    threshold_objective: Literal["MCC"]


class RuntimeConfig(StrictModel):
    required_compute_capability: tuple[int, int] | None
    minimum_device_memory_gib: float = Field(ge=0.0)
    strict_accelerator: bool
    minimum_free_disk_gib: float = Field(ge=0.0)
    artifact_root: str

    @field_validator("required_compute_capability", mode="before")
    @classmethod
    def validate_capability_shape(cls, value: object) -> object:
        if value is None:
            return None
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(type(part) is not int or part < 0 for part in value)
        ):
            raise ValueError("required compute capability must be two non-negative integers")
        return tuple(value)


class FusionGate(StrictModel):
    minimum_development_per_class: int = Field(ge=1)
    minimum_validation_per_class: int = Field(ge=1)


class AdaptiveConfig(StrictModel):
    language: Literal["en-US"]
    rank: Literal[8]
    trigger_family: Literal["T1"]
    poison_rate: Literal[0.05]
    seeds: list[int]
    gradient_ratio_targets: list[float]
    target_loss_alpha: Literal[1.0]
    weight_surrogate: Literal["canonical_delta_frobenius"]
    behavior_surrogate: Literal["posthoc_transfer_only"]
    initialization: Literal["qc_valid_paired_baseline"]
    continuation_schedule: Literal["centralized_training_profile"]
    normalization_reference: Literal["combined_clean_target"]
    calibration_sampling: Literal["deterministic_class_balanced"]
    coefficient_schedule: Literal["fixed_after_initialization"]


def _require_locked_payload(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected {expected!r}, observed {actual!r}")


class ExperimentConfig(StrictModel):
    schema_version: Literal[2]
    profile: Literal["fixture", "core", "target"]
    fixture_smoke_only: bool
    revisions: Revisions
    data: DataConfig
    task: TaskConfig
    lora: LoraConfig
    training: TrainingConfig | None = None
    evaluation: EvaluationConfig | None = None
    cohort: CohortConfig
    qc: QcConfig
    statistics: StatisticsConfig
    runtime: RuntimeConfig
    private_trigger_registry: str
    fusion_gate: FusionGate | None = None
    adaptive: AdaptiveConfig | None = None

    @model_validator(mode="after")
    def validate_locked_protocol(self) -> ExperimentConfig:
        if self.profile == "fixture":
            if not self.fixture_smoke_only:
                raise ValueError("fixture profile must set fixture_smoke_only=true")
            return self
        if self.fixture_smoke_only:
            raise ValueError("core/target profile cannot be marked fixture-only")
        _require_locked_payload(
            self.revisions.model_dump(), LOCKED_REVISIONS, "locked revision mismatch"
        )
        _require_locked_payload(
            self.data.model_dump(), LOCKED_DATA_PROFILE, "locked sampling changed"
        )
        _require_locked_payload(self.task.model_dump(), LOCKED_TASK_PROFILE, "locked task changed")
        _require_locked_payload(self.lora.model_dump(), LOCKED_LORA_PROFILE, "locked LoRA changed")
        if self.training is None or not self.training.completion_only:
            raise ValueError("main profiles require centralized completion-only training config")
        _require_locked_payload(
            self.training.model_dump(),
            LOCKED_TRAINING_PROFILE,
            "locked training profile changed",
        )
        if self.evaluation is None:
            raise ValueError("main profiles require centralized evaluation config")
        _require_locked_payload(
            self.evaluation.model_dump(),
            LOCKED_EVALUATION_PROFILE,
            "locked evaluation profile changed",
        )
        _require_locked_payload(
            self.qc.model_dump(), LOCKED_QC_PROFILE, "locked QC profile changed"
        )
        _require_locked_payload(
            self.statistics.model_dump(),
            LOCKED_STATISTICS_PROFILE,
            "locked statistics profile changed",
        )
        _require_locked_payload(
            self.runtime.model_dump(),
            LOCKED_RUNTIME_PROFILES[self.profile],
            "main profiles require the locked execution-device preflight",
        )
        if self.adaptive is None:
            raise ValueError("main profiles require the locked RQ2 adaptive configuration")
        _require_locked_payload(
            self.adaptive.model_dump(),
            LOCKED_ADAPTIVE_PROFILE,
            "locked RQ2 adaptive protocol changed",
        )
        if self.private_trigger_registry != "private/trigger_registry.yaml":
            raise ValueError("locked private trigger registry path changed")
        _require_locked_payload(
            self.cohort.model_dump(),
            LOCKED_COHORT_PROFILES[self.profile],
            f"locked {self.profile.title()} cohort changed",
        )
        observed_fusion = self.fusion_gate.model_dump() if self.fusion_gate else None
        _require_locked_payload(
            observed_fusion,
            LOCKED_FUSION_PROFILES[self.profile],
            f"locked {self.profile.title()} cohort changed",
        )
        return self

    @property
    def config_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    extends = raw.pop("extends", None)
    if extends:
        base_path = (config_path.parent / str(extends)).resolve()
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_raw, dict):
            raise ValueError(f"base configuration must be a mapping: {base_path}")
        base_raw.pop("extends", None)
        raw = _deep_merge(base_raw, raw)
    return ExperimentConfig.model_validate(raw)
