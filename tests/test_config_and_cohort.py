from collections import Counter, defaultdict

import pytest

from lora_audit.cohort import (
    cohort_plan_counts,
    plan_cohort,
    select_pending_lineages,
    select_representative_pilot_lineages,
)
from lora_audit.config import ExperimentConfig, load_config
from lora_audit.training import training_arguments_kwargs


def test_locked_core_config_and_cohort_counts(project_root):
    config = load_config(project_root / "configs/core.yaml")
    plan = plan_cohort(config)
    assert len(plan) == 36
    counts = {
        split: sum(row.detector_split == split for row in plan)
        for split in ("train", "validation", "test")
    }
    assert counts == {"train": 12, "validation": 12, "test": 12}
    assert cohort_plan_counts(plan) == {
        "nominal_lineages": 36,
        "effective_lineages": 36,
        "nominal_training_units": 72,
        "effective_training_units": 72,
        "effective_split_assignments": 36,
    }
    assert len({row.pair_training_seed for row in plan}) == 36
    assert len({row.poison_selection_seed for row in plan}) == 36
    assert all(row.pair_training_seed != row.poison_selection_seed for row in plan)
    assert {row.probe_seed for row in plan} == {config.cohort.probe_seed}
    by_cell = defaultdict(list)
    by_replicate = defaultdict(set)
    for row in plan:
        by_cell[(row.language, row.trigger_family, row.poison_rate)].append(row.detector_split)
        by_replicate[row.replicate_seed].add(row.detector_split)
    assert all(
        Counter(splits) == Counter({"train": 1, "validation": 1, "test": 1})
        for splits in by_cell.values()
    )
    assert all(splits == {"train", "validation", "test"} for splits in by_replicate.values())
    former_reuse_group = [
        row
        for row in plan
        if row.language == "en-US" and row.replicate_seed == config.cohort.replicate_seeds[0]
    ]
    assert len(former_reuse_group) == 6
    assert len({row.training_unit_id("clean") for row in former_reuse_group}) == len(
        former_reuse_group
    )
    assert config.training is not None
    arguments = training_arguments_kwargs(config, project_root / "artifacts/test")
    assert arguments["num_train_epochs"] == config.training.num_train_epochs
    assert arguments["per_device_train_batch_size"] == 32
    assert arguments["gradient_accumulation_steps"] == 1
    assert arguments["gradient_checkpointing"] is False
    assert arguments["train_sampling_strategy"] == "group_by_length"
    assert arguments["dataloader_num_workers"] == 2
    assert arguments["dataloader_pin_memory"] is True
    assert arguments["optim"] == "adamw_torch_fused"
    assert arguments["learning_rate"] == 0.0002
    assert arguments["warmup_ratio"] == 0.03
    assert "warmup_steps" not in arguments
    assert arguments["seed"] == config.data.seed
    assert len(config.task.intent_labels) == 60
    assert "weather_query" in config.task.intent_labels
    assert config.evaluation is not None
    assert config.evaluation.record_batch_size == 64


def test_target_config_extends_core_and_meets_partition_counts(project_root):
    config = load_config(project_root / "configs/target.yaml")
    plan = plan_cohort(config)
    assert len(plan) == 60
    counts = {
        split: sum(row.detector_split == split for row in plan)
        for split in ("train", "validation", "test")
    }
    assert counts == {"train": 36, "validation": 12, "test": 12}
    assert cohort_plan_counts(plan) == {
        "nominal_lineages": 60,
        "effective_lineages": 60,
        "nominal_training_units": 120,
        "effective_training_units": 120,
        "effective_split_assignments": 60,
    }
    by_replicate = defaultdict(set)
    for row in plan:
        by_replicate[row.replicate_seed].add(row.detector_split)
    assert all(splits == {"train", "validation", "test"} for splits in by_replicate.values())
    assert config.lora.rank == 8


def test_fixture_keeps_one_canonical_independent_pair(project_root):
    config = load_config(project_root / "configs/fixture.yaml")
    plan = plan_cohort(config)

    assert len(plan) == 1
    assert plan[0].detector_split == "test"
    assert plan[0].probe_seed == config.cohort.probe_seed
    assert plan[0].pair_training_seed != plan[0].poison_selection_seed
    assert cohort_plan_counts(plan)["effective_training_units"] == 2


def test_exact_lineage_selection_preserves_order_and_rejects_ambiguous_input(project_root):
    plan = plan_cohort(load_config(project_root / "configs/core.yaml"))
    requested = [plan[4].parent_config_id, plan[1].parent_config_id]

    selected = select_pending_lineages(
        plan,
        max_lineages=None,
        lineage_ids=requested,
    )

    assert [lineage.parent_config_id for lineage in selected] == requested
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_pending_lineages(plan, max_lineages=1, lineage_ids=requested)
    with pytest.raises(ValueError, match="duplicates"):
        select_pending_lineages(plan, max_lineages=None, lineage_ids=[requested[0], requested[0]])
    with pytest.raises(ValueError, match="not pending"):
        select_pending_lineages(plan, max_lineages=None, lineage_ids=["missing-lineage"])


def test_representative_pilot_is_outcome_blind_deterministic_and_factor_covering(project_root):
    plan = plan_cohort(load_config(project_root / "configs/core.yaml"))

    selected = select_representative_pilot_lineages(plan)
    reversed_selection = select_representative_pilot_lineages(list(reversed(plan)))

    assert [row.parent_config_id for row in selected] == [
        row.parent_config_id for row in reversed_selection
    ]
    assert len(selected) == 3
    assert len({row.trigger_family for row in selected}) == 3
    assert len({row.detector_split for row in selected}) == 3
    assert len({row.replicate_seed for row in selected}) >= 2
    assert len({row.language for row in selected}) == 2
    assert len({row.poison_rate for row in selected}) == 2
    with pytest.raises(ValueError, match="from 1 to 3"):
        select_representative_pilot_lineages(plan, pair_count=4)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("revisions", "model_id", "alternate/model", "revision mismatch"),
        ("training", "learning_rate", 0.0003, "training profile changed"),
        ("training", "save_steps", 200, "training profile changed"),
        ("evaluation", "record_batch_size", 32, "evaluation profile changed"),
        ("data", "locales", ["vi-VN", "en-US"], "sampling"),
        ("data", "caps", {"train": 63, "validation": 16, "test": 32}, "sampling"),
        ("task", "scoring", "alternate_scoring", "task changed"),
        ("lora", "alpha", 8, "LoRA"),
        ("lora", "dropout", 0.1, "LoRA"),
        ("qc", "minimum_backdoored_asr", 0.79, "QC profile changed"),
        ("statistics", "bootstrap_iterations", 6000, "statistics profile changed"),
        ("cohort", "replicate_seeds", [1, 2, 3], "Core cohort changed"),
        ("cohort", "poison_rates", [0.01, 0.1], "Core cohort changed"),
        ("runtime", "strict_accelerator", False, "execution-device preflight"),
        ("runtime", "minimum_device_memory_gib", 24.0, "execution-device preflight"),
        ("runtime", "minimum_free_disk_gib", 26.0, "execution-device preflight"),
        ("runtime", "artifact_root", "artifacts/alternate", "execution-device preflight"),
    ],
)
def test_locked_core_rejects_protocol_drift(project_root, section, field, value, message):
    payload = load_config(project_root / "configs/core.yaml").model_dump(mode="json")
    payload[section][field] = value

    with pytest.raises(ValueError, match=message):
        ExperimentConfig.model_validate(payload)


def test_locked_profiles_reject_registry_and_fusion_drift(project_root):
    core = load_config(project_root / "configs/core.yaml").model_dump(mode="json")
    core["private_trigger_registry"] = "private/alternate.yaml"
    with pytest.raises(ValueError, match="trigger registry path changed"):
        ExperimentConfig.model_validate(core)

    target = load_config(project_root / "configs/target.yaml").model_dump(mode="json")
    target["fusion_gate"]["minimum_development_per_class"] = 31
    with pytest.raises(ValueError, match="Target cohort changed"):
        ExperimentConfig.model_validate(target)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "learning_rate", "0.0002"),
        ("training", "per_device_train_batch_size", True),
        ("evaluation", "record_batch_size", "64"),
        ("runtime", "required_compute_capability", ["8", 0]),
        ("runtime", "minimum_device_memory_gib", "20.0"),
    ],
)
def test_config_rejects_coerced_scalar_types(project_root, section, field, value):
    payload = load_config(project_root / "configs/core.yaml").model_dump(mode="json")
    payload[section][field] = value

    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(payload)


def test_protocol_v2_rejects_legacy_seed_and_qc_artifacts(project_root):
    payload = load_config(project_root / "configs/core.yaml").model_dump(mode="json")
    payload["schema_version"] = 1
    payload["cohort"]["seeds"] = payload["cohort"].pop("replicate_seeds")
    payload["qc"] = {
        "minimum_asr": 0.8,
        "maximum_absolute_macro_f1_drop": 0.03,
    }

    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(payload)
