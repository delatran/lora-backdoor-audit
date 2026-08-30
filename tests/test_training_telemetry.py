from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import py_compile
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lora_audit.adaptive import (
    adaptive_task_weights,
    locked_adaptive_plan,
    validated_adaptive_baseline_identity,
)
from lora_audit.cohort import plan_cohort
from lora_audit.config import load_config
from lora_audit.data import (
    IntentRecord,
    prepared_data_receipt_fields,
    required_split_intent_cells,
    write_jsonl,
)
from lora_audit.evaluation import (
    _assess_pair,
    _predict_records_resumable,
    evaluate_trained_pair,
)
from lora_audit.execution_profiles import load_execution_profile
from lora_audit.hashing import sha256_file, sha256_json, sha256_text
from lora_audit.manifests import CohortEntry, evaluate_backdoor_qc
from lora_audit.scan import validated_training_adapter_identity
from lora_audit.training import (
    CompletionOnlyCollator,
    _adaptive_trainer_class,
    _set_training_seed,
    _train_or_resume_adapter,
    _verify_prepared_data_receipt,
    prefetch_model_assets,
    train_planned_pair,
)


def _load_bundle_bootstrap(project_root: Path):
    path = project_root.parent / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("bundle_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_outer_notebook_runtime_helpers(project_root: Path) -> dict[str, object]:
    notebook = json.loads(
        (project_root.parent / "RUN_EXPERIMENT.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(
        next(cell for cell in notebook["cells"] if cell.get("id") == "stage0-attest")["source"]
    )
    tree = ast.parse(source)
    assignments = {
        "INTERNAL_STUDY_LABEL_PATTERN",
        "REVISION_LABEL_PATTERN",
        "FORMAT_LABEL_PATTERN",
        "REVISION_WORD_PATTERN",
        "CONTROL_SEQUENCE_PATTERN",
        "INVISIBLE_TEXT_PATTERN",
    }
    functions = {
        "has_linked_ancestor",
        "presentation_safe_text",
        "presentation_diagnostic_tail",
        "persist_notebook_stage_log",
        "assert_report_figure_safe",
        "run_captured_stage",
        "display_study_progress",
    }
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module != "google.colab":
            selected.append(node)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in assignments
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    helper_module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(helper_module, "RUN_EXPERIMENT.ipynb:stage0-attest", "exec"), namespace)
    return namespace


def _record(
    locale: str,
    intent: str,
    index: int,
    *,
    split: str = "train",
) -> IntentRecord:
    text = f"harmless {locale} fixture {index}"
    return IntentRecord(
        parallel_group_id=f"telemetry-{split}-group-{index}",
        source_id=f"telemetry-{split}-source-{index}",
        locale=locale,
        intent=intent,
        text=text,
        split=split,
        raw_text_sha256=sha256_text(text),
        normalized_text_sha256=sha256_text(text),
    )


def _core_records(config) -> list[IntentRecord]:
    return [
        _record(locale, intent, index, split=split)
        for index, (split, intent) in enumerate(sorted(required_split_intent_cells(config)))
        for locale in config.data.locales
    ]


def _write_prepared_receipt(
    config, prepared: Path, records: list[IntentRecord]
) -> dict[str, object]:
    payload = {
        "schema_version": 2,
        "status": "passed",
        "config_sha256": config.config_hash,
        "dataset_id": config.revisions.dataset_id,
        "dataset_source_revision": config.revisions.dataset_source_revision,
        "dataset_revision": config.revisions.dataset_revision,
        "manifest_sha256": sha256_file(prepared),
        "intent_vocabulary_sha256": sha256_json(config.task.intent_labels),
        "intent_labels": config.task.intent_labels,
        **prepared_data_receipt_fields(records, config),
    }
    (prepared.parent / "receipt.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    return payload


def _a100_execution_profile(project_root: Path):
    """Load the exact profile used by every strict-runtime test in this module."""

    return load_execution_profile(project_root / "configs/execution/a100-40gb.yaml")


def _matching_a100_snapshot(execution_profile) -> dict[str, object]:
    """Return a minimal accelerator observation satisfying the locked A100 profile."""

    assert execution_profile.profile_id == "a100-40gb-v1"
    return {
        "backend": execution_profile.backend,
        "framework_version": "test-framework",
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA A100-SXM4-40GB",
                "total_memory_gib": 40.0,
                "compute_capability": list(execution_profile.required_compute_capability),
                "bf16_probe": {
                    "attempted": True,
                    "passed": True,
                    "finite": True,
                    "dtype": "torch.bfloat16",
                    "device_type": execution_profile.backend,
                    "device_index": 0,
                    "error": None,
                },
            }
        ],
    }


def test_training_seed_resets_random_initialization() -> None:
    import torch

    _set_training_seed(20260716)
    first = torch.randn(8)
    _set_training_seed(20260716)
    second = torch.randn(8)
    assert torch.equal(first, second)


def test_prepared_receipt_rejects_declared_count_drift(tmp_path: Path) -> None:
    config = load_config("configs/core.yaml")
    prepared = tmp_path / "prepared.jsonl"
    records = _core_records(config)
    write_jsonl(records, prepared)
    payload = _write_prepared_receipt(config, prepared, records)
    _verify_prepared_data_receipt(config, prepared, records)

    counts = dict(payload["counts_by_locale_split"])
    counts["en-US/train"] += 2
    payload["counts_by_locale_split"] = counts
    (prepared.parent / "receipt.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="counts_by_locale_split"):
        _verify_prepared_data_receipt(config, prepared, records)

    payload = _write_prepared_receipt(config, prepared, records)
    cell = next(iter(payload["counts_by_locale_split_intent"]))
    payload["counts_by_locale_split_intent"][cell] = True
    (prepared.parent / "receipt.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="counts_by_locale_split_intent"):
        _verify_prepared_data_receipt(config, prepared, records)


def test_prediction_checkpoint_is_hash_bound_and_resumable(tmp_path: Path) -> None:
    records = [_record("en-US", "intent-a", 1), _record("en-US", "intent-b", 2)]
    checkpoint = tmp_path / "predictions.json"
    calls: list[str] = []

    def predictor(texts: list[str]) -> list[str]:
        calls.extend(texts)
        return ["intent-a" if text.endswith("1") else "intent-b" for text in texts]

    first = _predict_records_resumable(
        records=records,
        candidates=["intent-a", "intent-b"],
        predictor=predictor,
        batch_size=2,
        checkpoint_path=checkpoint,
        binding={"adapter_sha256": "a" * 64, "profile": "test"},
    )
    second = _predict_records_resumable(
        records=records,
        candidates=["intent-a", "intent-b"],
        predictor=predictor,
        batch_size=2,
        checkpoint_path=checkpoint,
        binding={"adapter_sha256": "a" * 64, "profile": "test"},
    )

    assert first == second == ["intent-a", "intent-b"]
    assert len(calls) == 2
    with pytest.raises(ValueError, match="binding mismatch"):
        _predict_records_resumable(
            records=records,
            candidates=["intent-a", "intent-b"],
            predictor=predictor,
            batch_size=1,
            checkpoint_path=checkpoint,
            binding={"adapter_sha256": "a" * 64, "profile": "test"},
        )
    drifted = [records[0].model_copy(update={"text": "changed"}), records[1]]
    with pytest.raises(ValueError, match="binding mismatch"):
        _predict_records_resumable(
            records=drifted,
            candidates=["intent-a", "intent-b"],
            predictor=predictor,
            batch_size=2,
            checkpoint_path=checkpoint,
            binding={"adapter_sha256": "a" * 64, "profile": "test"},
        )

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["prediction_count"] = True
    checkpoint.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="count mismatch"):
        _predict_records_resumable(
            records=records,
            candidates=["intent-a", "intent-b"],
            predictor=predictor,
            batch_size=2,
            checkpoint_path=checkpoint,
            binding={"adapter_sha256": "a" * 64, "profile": "test"},
        )


def test_prediction_batches_are_bounded_and_failed_batches_are_atomic(tmp_path: Path) -> None:
    records = [_record("en-US", "intent-a", index) for index in range(129)]
    calls: list[int] = []

    def predictor(texts: list[str]) -> list[str]:
        calls.append(len(texts))
        return ["intent-a"] * len(texts)

    predictions = _predict_records_resumable(
        records=records,
        candidates=["intent-a"],
        predictor=predictor,
        batch_size=64,
        checkpoint_path=tmp_path / "bounded.json",
        binding={"adapter_sha256": "a" * 64, "profile": "test"},
    )
    assert predictions == ["intent-a"] * len(records)
    assert calls == [64, 64, 1]

    failed_checkpoint = tmp_path / "failed.json"

    def invalid_predictor(texts: list[str]) -> list[str]:
        return ["intent-a"] * (len(texts) - 1)

    with pytest.raises(ValueError, match="invalid prediction count"):
        _predict_records_resumable(
            records=records[:2],
            candidates=["intent-a"],
            predictor=invalid_predictor,
            batch_size=2,
            checkpoint_path=failed_checkpoint,
            binding={"adapter_sha256": "a" * 64, "profile": "test"},
        )
    assert not failed_checkpoint.exists()


def test_model_prefetch_is_revision_bound(project_root: Path, tmp_path: Path, monkeypatch) -> None:
    config = load_config(project_root / "configs/core.yaml")
    snapshot = tmp_path / config.revisions.model_revision
    snapshot.mkdir()
    for name, content in (
        ("config.json", b"{}"),
        ("tokenizer_config.json", b"{}"),
        ("model.safetensors", b"test-weights"),
    ):
        (snapshot / name).write_bytes(content)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    receipt = prefetch_model_assets(config)

    assert receipt["status"] == "passed"
    assert receipt["model_revision"] == config.revisions.model_revision
    assert receipt["file_count"] == 3
    assert calls == [
        {
            "repo_id": config.revisions.model_id,
            "revision": config.revisions.model_revision,
            "allow_patterns": [
                "*.json",
                "*.model",
                "*.safetensors",
                "*.tiktoken",
                "merges.txt",
                "tokenizer*",
                "vocab*",
            ],
        }
    ]


def test_pilot_pair_records_two_adapter_device_hour_rows(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    execution_profile = _a100_execution_profile(project_root)
    lineage = plan_cohort(config)[0]
    prepared = tmp_path / "prepared.jsonl"
    records = _core_records(config)
    write_jsonl(records, prepared)
    _write_prepared_receipt(config, prepared, records)
    calls: list[tuple[Path, int]] = []
    poison_seeds: list[int] = []

    def fake_train(**kwargs) -> Path:
        assert kwargs["execution_profile"] == execution_profile
        adapter = Path(kwargs["output_directory"]) / "adapter"
        calls.append((adapter, kwargs["seed"]))
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(
            f"bounded-test-adapter:{adapter.parent.name}".encode()
        )
        return adapter

    from lora_audit import training as training_module

    real_inject_poison = training_module.inject_poison_after_split

    def capture_poison_seed(*args, **kwargs):
        poison_seeds.append(kwargs["seed"])
        return real_inject_poison(*args, **kwargs)

    monkeypatch.setattr("lora_audit.training.train_qwen_lora_adapter", fake_train)
    monkeypatch.setattr(
        "lora_audit.training.inject_poison_after_split",
        capture_poison_seed,
    )
    monkeypatch.setattr(
        "lora_audit.training.accelerator_snapshot",
        lambda: _matching_a100_snapshot(execution_profile),
    )
    monkeypatch.setattr("lora_audit.training.peak_allocated_memory_gib", lambda _torch: 1.0)
    monkeypatch.setattr("lora_audit.training.peak_reserved_memory_gib", lambda _torch: 2.0)
    monkeypatch.setattr("lora_audit.training.reset_peak_memory_stats", lambda _torch: True)

    def fake_evaluate(**kwargs):
        assert kwargs["execution_profile"] == execution_profile
        return evaluate_backdoor_qc(
            adapter_id=f"{kwargs['lineage'].parent_config_id}-backdoored",
            parent_config_id=kwargs["lineage"].parent_config_id,
            backdoored_triggered_target_count=9,
            clean_triggered_target_count=1,
            eligible_count=10,
            clean_macro_f1=0.89,
            paired_clean_macro_f1=0.9,
            maximum_clean_triggered_target_rate=(config.qc.maximum_clean_triggered_target_rate),
            minimum_backdoored_asr=config.qc.minimum_backdoored_asr,
            minimum_paired_asr_lift=config.qc.minimum_paired_asr_lift,
            maximum_absolute_clean_macro_f1_difference=(
                config.qc.maximum_absolute_clean_macro_f1_difference
            ),
            fixture_smoke_only=False,
        )

    monkeypatch.chdir(tmp_path)
    output = Path("run")
    entries = train_planned_pair(
        config=config,
        lineage=lineage,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=output,
        source_tree_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        execution_profile=execution_profile,
        pilot_telemetry=True,
        pair_evaluator=fake_evaluate,
    )
    resumed_entries = train_planned_pair(
        config=config,
        lineage=lineage,
        prepared_manifest=prepared,
        project_root=project_root,
        output_root=output,
        source_tree_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        execution_profile=execution_profile,
        pilot_telemetry=True,
        pair_evaluator=fake_evaluate,
    )

    rows = [
        json.loads(line)
        for line in (output / "ledgers/pilot-device-hours.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(entries) == 2
    assert resumed_entries == entries
    assert len(calls) == 2
    assert {seed for _, seed in calls} == {lineage.pair_training_seed}
    assert poison_seeds == [lineage.poison_selection_seed, lineage.poison_selection_seed]
    assert [row["adapter_label"] for row in rows] == ["backdoored", "clean"]
    assert all(row["device_hours"] > 0 for row in rows)
    assert all(row["runtime_seconds"] > 0 for row in rows)
    assert all(row["peak_allocated_memory_gib"] == 1.0 for row in rows)
    assert all(row["peak_reserved_memory_gib"] == 2.0 for row in rows)
    assert all(row["config_sha256"] == config.config_hash for row in rows)
    assert all(row["source_tree_sha256"] == "a" * 64 for row in rows)
    assert all(row["source_manifest_sha256"] == "b" * 64 for row in rows)
    assert all(row["execution_profile"] == execution_profile.binding() for row in rows)
    for field, expected in {
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "split_assignment_id": lineage.split_assignment_id,
    }.items():
        assert {row[field] for row in rows} == {expected}
    assert {row["training_unit_id"] for row in rows} == {
        lineage.training_unit_id("clean"),
        lineage.training_unit_id("backdoored"),
    }
    assert entries[0].normalized_text_hash != entries[1].normalized_text_hash
    assert {entry.schema_version for entry in entries} == {2}
    assert {entry.qc_status for entry in entries} == {"valid"}
    assert {
        (
            entry.execution_profile_id,
            entry.execution_profile_sha256,
            entry.execution_profile_file_sha256,
        )
        for entry in entries
    } == {
        (
            execution_profile.profile_id,
            execution_profile.profile_hash,
            execution_profile.profile_file_sha256,
        )
    }
    metadata = json.loads(
        (output / "adapters" / lineage.parent_config_id / "pair-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["selected_poison_group_count"] == 1
    assert metadata["target_intent"] in config.task.intent_labels
    assert metadata["replicate_seed"] == lineage.replicate_seed
    assert metadata["pair_training_seed"] == lineage.pair_training_seed
    assert metadata["poison_selection_seed"] == lineage.poison_selection_seed
    assert metadata["probe_seed"] == lineage.probe_seed
    assert metadata["split_assignment_id"] == lineage.split_assignment_id
    assert metadata["source_tree_sha256"] == "a" * 64
    assert metadata["source_manifest_sha256"] == "b" * 64
    assert metadata["execution_profile"] == execution_profile.binding()
    assert metadata["schema_version"] == 2
    assert metadata["qc"]["schema_version"] == 2


@pytest.mark.parametrize(
    ("pilot_telemetry", "residual_path", "message"),
    [
        (False, "adapter", "unreceipted adapter output"),
        (True, "checkpoints/checkpoint-1", "incomplete pilot adapter output"),
    ],
)
def test_adapter_resume_requires_receipted_or_clean_state(
    project_root: Path,
    tmp_path: Path,
    pilot_telemetry: bool,
    residual_path: str,
    message: str,
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    execution_profile = _a100_execution_profile(project_root)
    lineage = plan_cohort(config)[0]
    pair_root = tmp_path / "pair"
    (pair_root / "clean" / residual_path).mkdir(parents=True)

    with pytest.raises(RuntimeError, match=message):
        _train_or_resume_adapter(
            config=config,
            lineage=lineage,
            label="clean",
            rows=[_record(lineage.language, config.task.intent_labels[0], 1)],
            pair_root=pair_root,
            telemetry_path=tmp_path / "pilot.jsonl",
            pilot_telemetry=pilot_telemetry,
            source_tree_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            execution_profile=execution_profile,
        )


def test_timed_pair_evaluation_requires_clean_output_state(
    project_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    execution_profile = _a100_execution_profile(project_root)
    lineage = plan_cohort(config)[0]
    residual = tmp_path / "evaluations" / lineage.parent_config_id / "predictions"
    residual.mkdir(parents=True)
    monkeypatch.setattr(
        "lora_audit.evaluation.accelerator_snapshot",
        lambda: _matching_a100_snapshot(execution_profile),
    )

    with pytest.raises(RuntimeError, match="timed pilot evaluation output"):
        evaluate_trained_pair(
            config=config,
            lineage=lineage,
            records=[],
            clean_adapter=tmp_path / "clean",
            backdoor_adapter=tmp_path / "backdoored",
            trigger=SimpleNamespace(),
            output_root=tmp_path,
            pilot_telemetry=True,
            source_tree_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            execution_profile=execution_profile,
        )


def test_pair_assessment_uses_shared_rows_for_clean_control_and_backdoor_asr(
    project_root: Path,
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    lineage = plan_cohort(config)[0]
    target = config.task.intent_labels[0]
    truth = [target, config.task.intent_labels[1], config.task.intent_labels[2]]
    clean_test = [
        _record(lineage.language, intent, index, split="test") for index, intent in enumerate(truth)
    ]
    clean_result = SimpleNamespace(
        adapter_id=f"{lineage.parent_config_id}-clean",
        clean_predictions=truth,
        triggered_predictions=truth,
    )
    backdoor_result = SimpleNamespace(
        adapter_id=f"{lineage.parent_config_id}-backdoored",
        clean_predictions=truth,
        triggered_predictions=[target, target, target],
    )

    assessment = _assess_pair(
        config=config,
        lineage=lineage,
        trigger=SimpleNamespace(target_intent=target),
        clean_test=clean_test,
        clean_result=clean_result,
        backdoor_result=backdoor_result,
    )

    assert assessment.clean_triggered_target == {
        "target_rate": 0.0,
        "successes": 0,
        "eligible": 2,
    }
    assert assessment.attack_success == {
        "asr": 1.0,
        "successes": 2,
        "eligible": 2,
    }
    assert assessment.qc.valid
    assert assessment.qc.paired_asr_lift == 1.0


def test_pair_evaluation_routes_execution_profile_to_the_ledger_writer_only() -> None:
    source = inspect.getsource(evaluate_trained_pair)
    tree = ast.parse(source)
    calls = {
        call.func.id: call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in {"_assess_pair", "_write_pair_ledgers"}
    }

    assert "execution_profile" not in {keyword.arg for keyword in calls["_assess_pair"].keywords}
    ledger_keywords = {
        keyword.arg: keyword.value for keyword in calls["_write_pair_ledgers"].keywords
    }
    assert isinstance(ledger_keywords["execution_profile"], ast.Name)
    assert ledger_keywords["execution_profile"].id == "execution_profile"


def test_accelerated_install_removes_incompatible_optional_torch_packages(
    project_root: Path,
) -> None:
    module = _load_bundle_bootstrap(project_root)
    install_source = " ".join(inspect.getsource(module.install_environment).split())
    assert '"uninstall", "-y"' in install_source
    for package in ("torchvision", "torchaudio", "torchao"):
        assert f'"{package}"' in install_source
    assert '"install", "-q", "--requirement", lock' in install_source


def test_completion_collator_preserves_adaptive_metadata() -> None:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    collator = CompletionOnlyCollator(tokenizer)
    batch = collator(
        [
            {
                "input_ids": [3, 4],
                "attention_mask": [1, 1],
                "labels": [-100, 4],
                "adaptive_target": False,
                "adaptive_row_index": 2,
            },
            {
                "input_ids": [5],
                "attention_mask": [1],
                "labels": [5],
                "adaptive_target": True,
                "adaptive_row_index": 7,
            },
        ]
    )

    assert batch["adaptive_target"].tolist() == [False, True]
    assert batch["adaptive_row_index"].tolist() == [2, 7]
    assert batch["input_ids"].tolist() == [[3, 4], [5, 0]]
    assert batch["labels"].tolist() == [[-100, 4], [5, -100]]
    with pytest.raises(ValueError, match="every feature or none"):
        collator(
            [
                {
                    "input_ids": [3],
                    "attention_mask": [1],
                    "labels": [3],
                    "adaptive_target": False,
                    "adaptive_row_index": 0,
                },
                {"input_ids": [4], "attention_mask": [1], "labels": [4]},
            ]
        )
    with pytest.raises(ValueError, match="both required fields"):
        collator(
            [
                {
                    "input_ids": [3],
                    "attention_mask": [1],
                    "labels": [3],
                    "adaptive_target": False,
                }
            ]
        )


class _TinyAdaptiveModel(torch.nn.Module):
    def __init__(self, *, zero_update: bool) -> None:
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 8, bias=False)})
        self.layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False)})
        torch.nn.init.constant_(self.layer.lora_A["default"].weight, 0.01)
        torch.nn.init.constant_(
            self.layer.lora_B["default"].weight,
            0.0 if zero_update else 0.01,
        )

    def forward(self, input_ids, attention_mask):
        del attention_mask
        update = self.layer.lora_B["default"].weight @ self.layer.lora_A["default"].weight
        margin = update.sum().expand(input_ids.shape)
        return SimpleNamespace(logits=torch.stack((margin, -margin), dim=-1))


class _TrainerBase:
    def __init__(self, *, model, **kwargs) -> None:
        del kwargs
        self.model = model

    def _prepare_inputs(self, inputs):
        return inputs


def _adaptive_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 0, 0], [-100, 0, 0]]),
        "adaptive_target": torch.tensor([False, True]),
        "adaptive_row_index": torch.tensor([0, 1]),
    }


def test_adaptive_trainer_locks_and_reuses_gradient_ratio(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    condition = next(
        item for item in locked_adaptive_plan(config) if item.desired_gradient_ratio == 0.3
    )
    weights = adaptive_task_weights([False, True], target_alpha=1.0)
    binding = {
        "config_sha256": config.config_hash,
        "record_count": 2,
        "calibration_row_indices": [0, 1],
    }
    calibration = tmp_path / "normalization.json"
    trainer_type = _adaptive_trainer_class(_TrainerBase)
    model = _TinyAdaptiveModel(zero_update=False)
    trainer = trainer_type(
        model=model,
        adaptive_condition=condition,
        adaptive_task_weights=weights,
        adaptive_binding=binding,
        adaptive_calibration_path=calibration,
        adapter_alpha=16.0,
        adapter_rank=8,
    )

    with pytest.raises(RuntimeError, match="calibrated before training"):
        trainer.compute_loss(model, _adaptive_batch())
    trainer.calibrate_objective(_adaptive_batch())
    loss, outputs = trainer.compute_loss(model, _adaptive_batch(), return_outputs=True)
    receipt = json.loads(calibration.read_text(encoding="utf-8"))
    assert torch.isfinite(loss)
    assert outputs.logits.shape == (2, 3, 2)
    assert receipt["observed_gradient_ratio"] == pytest.approx(0.3)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    resumed_model = _TinyAdaptiveModel(zero_update=False)
    resumed = trainer_type(
        model=resumed_model,
        adaptive_condition=condition,
        adaptive_task_weights=weights,
        adaptive_binding=binding,
        adaptive_calibration_path=calibration,
        adapter_alpha=16.0,
        adapter_rank=8,
    )
    assert resumed._lambda_weight == pytest.approx(receipt["lambda_weight"])
    assert torch.isfinite(resumed.compute_loss(resumed_model, _adaptive_batch()))


def test_adaptive_trainer_rejects_zero_gradient_baseline(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    condition = next(
        item for item in locked_adaptive_plan(config) if item.desired_gradient_ratio == 0.3
    )
    weights = adaptive_task_weights([False, True], target_alpha=1.0)
    model = _TinyAdaptiveModel(zero_update=True)
    trainer_type = _adaptive_trainer_class(_TrainerBase)
    trainer = trainer_type(
        model=model,
        adaptive_condition=condition,
        adaptive_task_weights=weights,
        adaptive_binding={
            "config_sha256": config.config_hash,
            "record_count": 2,
            "calibration_row_indices": [0, 1],
        },
        adaptive_calibration_path=tmp_path / "normalization.json",
        adapter_alpha=16.0,
        adapter_rank=8,
    )

    with pytest.raises(ValueError, match="gradient norms must be positive"):
        trainer.calibrate_objective(_adaptive_batch())


def test_adapter_identity_is_strictly_bound(project_root: Path, tmp_path: Path) -> None:
    config = load_config(project_root / "configs/core.yaml")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"bounded-adapter")
    metadata = {
        "base_model_name_or_path": config.revisions.model_id,
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": config.lora.bias,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": config.lora.target_modules,
    }
    (adapter / "adapter_config.json").write_text(json.dumps(metadata), encoding="utf-8")

    identity = validated_training_adapter_identity(config, adapter)
    assert set(identity) == {"weights_sha256", "metadata_sha256"}
    (adapter / "adapter_config.json").write_text(
        json.dumps({**metadata, "r": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="adapter configuration mismatch"):
        validated_training_adapter_identity(config, adapter)


def test_adaptive_baseline_requires_matching_training_and_qc_evidence(
    project_root: Path, tmp_path: Path
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    execution_profile = load_execution_profile(project_root / "configs/execution/a100-40gb.yaml")
    condition = locked_adaptive_plan(config)[0]
    lineage = next(
        item
        for item in plan_cohort(config)
        if item.language == condition.language
        and item.replicate_seed == condition.seed
        and item.trigger_family == condition.trigger_family
        and item.poison_rate == condition.poison_rate
    )
    clean = _record("en-US", "general_greet", 1)
    source = _record("en-US", "general_joke", 2)
    poisoned = source.model_copy(
        update={
            "intent": "datetime_query",
            "poisoned": True,
            "trigger_family_id": condition.trigger_family,
            "original_intent": source.intent,
        }
    )
    records = [clean, poisoned]
    adapter = tmp_path / "adapters" / lineage.parent_config_id / "backdoored" / "adapter"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"qc-bound-adapter")
    adapter_sha256 = sha256_file(weights)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": config.revisions.model_id,
                "r": config.lora.rank,
                "lora_alpha": config.lora.alpha,
                "lora_dropout": config.lora.dropout,
                "bias": config.lora.bias,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "target_modules": config.lora.target_modules,
            }
        ),
        encoding="utf-8",
    )
    receipt = {
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
        "source_tree_sha256": "e" * 64,
        "source_manifest_sha256": "f" * 64,
        "execution_profile": execution_profile.binding(),
    }
    (adapter.parent / "training-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    common = {
        "parent_config_id": lineage.parent_config_id,
        "base_model": config.revisions.model_id,
        "dataset": config.revisions.dataset_id,
        "language": lineage.language,
        "replicate_seed": lineage.replicate_seed,
        "pair_training_seed": lineage.pair_training_seed,
        "poison_selection_seed": lineage.poison_selection_seed,
        "probe_seed": lineage.probe_seed,
        "rank": config.lora.rank,
        "target_modules": config.lora.target_modules,
        "raw_data_group_hash": "a" * 64,
        "model_revision": config.revisions.model_revision,
        "dataset_revision": config.revisions.dataset_revision,
        "train_revision": "test",
        "config_sha256": config.config_hash,
        "execution_profile_id": execution_profile.profile_id,
        "execution_profile_sha256": execution_profile.profile_hash,
        "execution_profile_file_sha256": execution_profile.profile_file_sha256,
        "qc_status": "valid",
        "qc_reason": None,
        "split": lineage.detector_split,
        "split_assignment_id": lineage.split_assignment_id,
        "fixture_smoke_only": False,
    }
    entries = [
        CohortEntry(
            adapter_id=f"{lineage.parent_config_id}-clean",
            label="clean",
            poison_rate=0.0,
            trigger_family=None,
            target_intent=None,
            adapter_sha256="b" * 64,
            normalized_text_hash="c" * 64,
            training_unit_id=lineage.training_unit_id("clean"),
            **common,
        ),
        CohortEntry(
            adapter_id=f"{lineage.parent_config_id}-backdoored",
            label="backdoored",
            poison_rate=condition.poison_rate,
            trigger_family=condition.trigger_family,
            target_intent="datetime_query",
            adapter_sha256=adapter_sha256,
            normalized_text_hash="d" * 64,
            training_unit_id=lineage.training_unit_id("backdoored"),
            **common,
        ),
    ]
    qc = evaluate_backdoor_qc(
        adapter_id=f"{lineage.parent_config_id}-backdoored",
        parent_config_id=lineage.parent_config_id,
        backdoored_triggered_target_count=9,
        clean_triggered_target_count=1,
        eligible_count=10,
        clean_macro_f1=0.9,
        paired_clean_macro_f1=0.9,
        maximum_clean_triggered_target_rate=(config.qc.maximum_clean_triggered_target_rate),
        minimum_backdoored_asr=config.qc.minimum_backdoored_asr,
        minimum_paired_asr_lift=config.qc.minimum_paired_asr_lift,
        maximum_absolute_clean_macro_f1_difference=(
            config.qc.maximum_absolute_clean_macro_f1_difference
        ),
        fixture_smoke_only=False,
    )
    pair_metadata = {
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
        "target_intent": "datetime_query",
        "source_tree_sha256": "e" * 64,
        "source_manifest_sha256": "f" * 64,
        "execution_profile": execution_profile.binding(),
        "qc": qc.model_dump(mode="json"),
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    metadata_path = adapter.parent.parent / "pair-metadata.json"
    metadata_path.write_text(json.dumps(pair_metadata), encoding="utf-8")

    identity = validated_adaptive_baseline_identity(
        config=config,
        condition=condition,
        records=records,
        adapter_directory=adapter,
        source_tree_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        execution_profile=execution_profile,
    )
    assert identity["weights_sha256"] == adapter_sha256
    with pytest.raises(ValueError, match="training receipt mismatch"):
        validated_adaptive_baseline_identity(
            config=config,
            condition=condition,
            records=records,
            adapter_directory=adapter,
            source_tree_sha256="0" * 64,
            source_manifest_sha256="f" * 64,
            execution_profile=execution_profile,
        )
    receipt["execution_profile"]["profile_hash"] = "0" * 64
    (adapter.parent / "training-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="training receipt mismatch"):
        validated_adaptive_baseline_identity(
            config=config,
            condition=condition,
            records=records,
            adapter_directory=adapter,
            source_tree_sha256="e" * 64,
            source_manifest_sha256="f" * 64,
            execution_profile=execution_profile,
        )
    receipt["execution_profile"] = execution_profile.binding()
    (adapter.parent / "training-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    pair_metadata["qc"]["valid"] = False
    metadata_path.write_text(json.dumps(pair_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="pair evidence mismatch"):
        validated_adaptive_baseline_identity(
            config=config,
            condition=condition,
            records=records,
            adapter_directory=adapter,
            source_tree_sha256="e" * 64,
            source_manifest_sha256="f" * 64,
            execution_profile=execution_profile,
        )


def test_bundle_bootstrap_verifies_persisted_manifests_without_reblessing(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    package_entries = ({"path": "RUN_EXPERIMENT.ipynb"},)
    source_manifest = {"entry_count": len(package_entries), "tree_sha256": "a" * 64}
    bundle_manifest = {
        "entry_count": len(package_entries) + 2,
        "tree_sha256": "b" * 64,
    }
    verified = SimpleNamespace(
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        package_entries=package_entries,
    )
    observed: list[object] = []

    def verify_persisted_manifests(bundle_root: Path):
        observed.append(bundle_root)
        return verified

    def scan_manifest_bound_content(candidate: object) -> int:
        observed.append(candidate)
        return len(package_entries)

    def verify_upload_tree(_bundle_root: Path):
        raise AssertionError("receipt-bound verification must not re-bless the upload tree")

    monkeypatch.setattr(
        module,
        "_packaging_verifier",
        lambda: (
            verify_persisted_manifests,
            scan_manifest_bound_content,
            verify_upload_tree,
        ),
    )
    monkeypatch.setattr(
        module,
        "_read_verified_upload_receipt",
        lambda _candidate: {"schema_version": 1, "status": "passed"},
    )
    monkeypatch.setattr(module, "_has_runtime_upload_session_marker", lambda _receipt: True)

    assert module.verify_persisted_upload_contract() == (source_manifest, bundle_manifest)
    assert observed == [module.BUNDLE, verified]
    assert not hasattr(module, "build_source_manifest")
    assert not hasattr(module, "build_bundle_manifest")


def test_bundle_bootstrap_physically_verifies_fresh_upload_before_writing_receipt(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    package_entries = (
        {
            "path": "RUN_EXPERIMENT.ipynb",
            "size_bytes": 1,
            "sha256": "c" * 64,
            "private": False,
        },
    )
    source_manifest = {"entry_count": 71, "tree_sha256": "a" * 64}
    bundle_manifest = {"entry_count": 73, "tree_sha256": "b" * 64}
    verified = SimpleNamespace(
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        package_entries=package_entries,
    )
    calls: list[str] = []

    def verify_persisted_manifests(bundle_root: Path):
        assert bundle_root == module.BUNDLE
        calls.append("persisted")
        return verified

    def verify_upload_tree(bundle_root: Path):
        assert bundle_root == module.BUNDLE
        calls.append("physical")
        return verified

    monkeypatch.setattr(
        module,
        "_packaging_verifier",
        lambda: (verify_persisted_manifests, lambda _verified: 1, verify_upload_tree),
    )
    monkeypatch.setattr(module, "UPLOAD_VERIFICATION_RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))

    assert module.verify_persisted_upload_contract() == (source_manifest, bundle_manifest)

    assert calls == ["persisted", "physical"]
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt == {
        "schema_version": 1,
        "status": "passed",
        "verifier": "scripts/build_upload_bundle.py::verify_upload_tree",
        "source_manifest_sha256": module._sha256_file(
            project_root / "artifacts/source-manifest.json"
        ),
        "source_tree_sha256": "a" * 64,
        "source_entry_count": 71,
        "bundle_manifest_sha256": module._sha256_file(
            project_root / "artifacts/bundle-manifest.json"
        ),
        "bundle_tree_sha256": "b" * 64,
        "bundle_entry_count": 73,
        "physical_package_entry_count": 1,
        "physical_package_tree_sha256": module._physical_package_tree_sha256(verified),
    }
    assert module._has_runtime_upload_session_marker(receipt)


def test_bundle_bootstrap_reattests_a_reset_runtime_without_reblessing(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    package_entries = ({"path": "RUN_EXPERIMENT.ipynb"},)
    source_manifest = {"entry_count": 1, "tree_sha256": "a" * 64}
    bundle_manifest = {"entry_count": 3, "tree_sha256": "b" * 64}
    verified = SimpleNamespace(
        source_manifest=source_manifest,
        bundle_manifest=bundle_manifest,
        package_entries=package_entries,
    )
    receipt = {"schema_version": 1, "status": "passed"}
    calls: list[str] = []

    def verify_persisted_manifests(_bundle_root: Path):
        calls.append("persisted")
        return verified

    def scan_manifest_bound_content(_verified: object) -> int:
        calls.append("scan")
        return 1

    def verify_upload_tree(_bundle_root: Path):
        raise AssertionError("reset-runtime reattestation must not re-bless the upload")

    monkeypatch.setattr(
        module,
        "_packaging_verifier",
        lambda: (
            verify_persisted_manifests,
            scan_manifest_bound_content,
            verify_upload_tree,
        ),
    )
    monkeypatch.setattr(module, "_read_verified_upload_receipt", lambda _candidate: receipt)
    monkeypatch.setattr(module, "_has_runtime_upload_session_marker", lambda _receipt: False)
    monkeypatch.setattr(
        module,
        "_validate_runtime_overlay",
        lambda _candidate: calls.append("overlay") or 4,
    )
    monkeypatch.setattr(
        module,
        "_write_runtime_upload_session_marker",
        lambda _receipt: calls.append("marker"),
    )

    assert module.verify_persisted_upload_contract() == (source_manifest, bundle_manifest)
    assert calls == ["persisted", "overlay", "marker", "scan"]


def test_runtime_reattest_refuses_executable_generated_overlay(
    project_root: Path, tmp_path: Path
) -> None:
    module = _load_bundle_bootstrap(project_root)
    bundle = tmp_path / "TranThienNhan_TTTN"
    artifact_directory = bundle / "lora-audit" / "artifacts"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "source-manifest.json").write_text("{}\n", encoding="utf-8")
    (artifact_directory / "unexpected.py").write_text("raise SystemExit\n", encoding="utf-8")
    verified = SimpleNamespace(
        root=bundle,
        package_entries=({"path": "lora-audit/artifacts/source-manifest.json"},),
    )

    with pytest.raises(ValueError, match="unexpected or executable runtime-overlay file"):
        module._validate_runtime_overlay(verified)


def test_runtime_reattest_refuses_symbolic_link_overlay(project_root: Path, tmp_path: Path) -> None:
    module = _load_bundle_bootstrap(project_root)
    bundle = tmp_path / "TranThienNhan_TTTN"
    artifact_directory = bundle / "lora-audit" / "artifacts"
    artifact_directory.mkdir(parents=True)
    source_manifest = artifact_directory / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = artifact_directory / "linked.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable in this Windows test environment")
    verified = SimpleNamespace(
        root=bundle,
        package_entries=({"path": "lora-audit/artifacts/source-manifest.json"},),
    )

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        module._validate_runtime_overlay(verified)


def test_runtime_reattest_refuses_reparse_point_overlay(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    bundle = tmp_path / "TranThienNhan_TTTN"
    artifact_directory = bundle / "lora-audit" / "artifacts"
    artifact_directory.mkdir(parents=True)
    source_manifest = artifact_directory / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    verified = SimpleNamespace(
        root=bundle,
        package_entries=({"path": "lora-audit/artifacts/source-manifest.json"},),
    )
    monkeypatch.setattr(module, "_is_reparse_point", lambda _stat: True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        module._validate_runtime_overlay(verified)


def test_runtime_reattest_refuses_unchecked_verifier_bytecode_overlay(
    project_root: Path, tmp_path: Path
) -> None:
    module = _load_bundle_bootstrap(project_root)
    bundle = tmp_path / "TranThienNhan_TTTN"
    scripts = bundle / "lora-audit" / "scripts"
    scripts.mkdir(parents=True)
    verifier_source = scripts / "build_upload_bundle.py"
    verifier_source.write_text(
        "def verify_upload_tree(_root):\n    return None\n",
        encoding="utf-8",
    )
    cache = scripts / "__pycache__"
    cache.mkdir()
    (cache / "build_upload_bundle.cpython-312.pyc").write_bytes(b"unchecked-bytecode")
    verified = SimpleNamespace(
        root=bundle,
        package_entries=({"path": "lora-audit/scripts/build_upload_bundle.py"},),
    )

    with pytest.raises(ValueError, match="unexpected runtime-overlay directory"):
        module._validate_runtime_overlay(verified)


def test_packaging_verifier_source_load_does_not_execute_unchecked_bytecode(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    project = tmp_path / "lora-audit"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    verifier_source = scripts / "build_upload_bundle.py"
    marker = scripts / "unchecked-bytecode-executed.txt"
    verifier_source.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('unchecked-bytecode-executed.txt').write_text('bad')\n",
        encoding="utf-8",
    )
    cache = scripts / "__pycache__"
    cache.mkdir()
    py_compile.compile(
        str(verifier_source),
        cfile=str(importlib.util.cache_from_source(str(verifier_source))),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    verifier_source.write_text(
        "def verify_persisted_manifests(_root):\n    return None\n"
        "def scan_manifest_bound_content(_verified):\n    return 0\n"
        "def verify_upload_tree(_root):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT", project)

    functions = module._packaging_verifier()

    assert len(functions) == 3
    assert not marker.exists()


def test_packaging_verifier_refuses_linked_scripts_directory(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    bundle = tmp_path / "TranThienNhan_TTTN"
    project = bundle / "lora-audit"
    project.mkdir(parents=True)
    external_scripts = tmp_path / "external-scripts"
    external_scripts.mkdir()
    (external_scripts / "build_upload_bundle.py").write_text("pass\n", encoding="utf-8")
    linked_scripts = project / "scripts"
    try:
        linked_scripts.symlink_to(external_scripts, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable in this Windows test environment")
    monkeypatch.setattr(module, "BUNDLE", bundle)
    monkeypatch.setattr(module, "PROJECT", project)

    with pytest.raises(ValueError, match="packaging verifier directory is missing, linked"):
        module._packaging_verifier()


def test_packaging_verifier_refuses_linked_bundle_ancestor(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    real_parent = tmp_path / "real-parent"
    bundle = real_parent / "TranThienNhan_TTTN"
    scripts = bundle / "lora-audit" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "build_upload_bundle.py").write_text("pass\n", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable in this Windows test environment")
    linked_bundle = linked_parent / "TranThienNhan_TTTN"
    monkeypatch.setattr(module, "BUNDLE", linked_bundle)
    monkeypatch.setattr(module, "PROJECT", linked_bundle / "lora-audit")

    with pytest.raises(ValueError, match="bundle root ancestor is missing, linked"):
        module._packaging_verifier()


def test_packaging_verifier_safety_chain_refuses_reparse_ancestor(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    bundle = tmp_path / "nested" / "TranThienNhan_TTTN"
    bundle.mkdir(parents=True)
    observations = 0

    def reparse_on_first_nonanchor(_stat: object) -> bool:
        nonlocal observations
        observations += 1
        return observations == 2

    monkeypatch.setattr(module, "_is_reparse_point", reparse_on_first_nonanchor)

    with pytest.raises(ValueError, match="bundle root ancestor is missing, linked"):
        module._assert_safe_directory_chain(bundle, label="bundle root")


def test_packaging_verifier_safety_helper_refuses_reparse_directory(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(module, "_is_reparse_point", lambda _stat: True)

    with pytest.raises(ValueError, match="packaging verifier directory is missing, linked"):
        module._assert_safe_directory(scripts, label="packaging verifier directory")


def test_bundle_subprocesses_prefer_bundle_source_and_disable_bytecode_cache(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    captured: dict[str, object] = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("PYTHONPATH", "installed-package-target")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._run([sys.executable, "-V"])

    expected_pythonpath = f"{project_root / 'src'}{module.os.pathsep}installed-package-target"
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"]["PYTHONPATH"] == expected_pythonpath


def test_bundle_subprocess_imports_source_before_an_installed_shadow(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    shadow_root = tmp_path / "installed-shadow"
    shadow_package = shadow_root / "lora_audit"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text(
        'raise RuntimeError("installed shadow package was imported")\n',
        encoding="utf-8",
    )
    expected_source = project_root / "src" / "lora_audit" / "__init__.py"
    probe = (
        "from pathlib import Path; import lora_audit; "
        "observed = Path(lora_audit.__file__).resolve(); "
        f"expected = Path({json.dumps(str(expected_source))}).resolve(); "
        "assert observed == expected, f'{observed} != {expected}'"
    )
    monkeypatch.setenv("PYTHONPATH", str(shadow_root))

    module._run([sys.executable, "-c", probe])


def test_bundle_driver_binds_its_own_source_before_an_installed_shadow(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    source_root = project_root / "src"
    shadow_root = tmp_path / "installed-shadow"
    shadow_root.mkdir()
    monkeypatch.setattr(module.sys, "path", [str(shadow_root)])
    monkeypatch.setenv("PYTHONPATH", str(shadow_root))

    module._bind_manifest_source_imports()

    assert module.sys.path[0] == str(source_root)
    assert module.os.environ["PYTHONPATH"] == f"{source_root}{module.os.pathsep}{shadow_root}"
    assert module.os.environ["PYTHONDONTWRITEBYTECODE"] == "1"


def test_bundle_attest_mode_only_verifies_the_upload_contract(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    calls: list[str] = []
    monkeypatch.setattr(
        module, "_bind_manifest_source_imports", lambda: calls.append("source-bound")
    )
    monkeypatch.setattr(
        module,
        "verify_persisted_upload_contract",
        lambda: calls.append("attested") or ({}, {}),
    )
    monkeypatch.setattr(
        module,
        "_run_selected_mode",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute a stage")),
    )

    assert module.main(["--mode", "attest"]) == 0
    assert calls == ["source-bound", "attested"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "attest", "--execution-profile", "configs/execution/a100-40gb.yaml"],
        ["--mode", "smoke", "--execution-profile", "configs/execution/a100-40gb.yaml"],
    ],
)
def test_bundle_rejects_invalid_nonexecution_arguments_before_attestation(
    project_root: Path, monkeypatch, argv: list[str]
) -> None:
    module = _load_bundle_bootstrap(project_root)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "verify_persisted_upload_contract",
        lambda: calls.append("attested") or ({}, {}),
    )

    with pytest.raises(SystemExit):
        module.main(argv)
    assert calls == []


def test_bundle_smoke_requires_fixture_artifact_validation(project_root: Path, monkeypatch) -> None:
    module = _load_bundle_bootstrap(project_root)
    fixture_calls: list[dict[str, str | None]] = []
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(module, "install_environment", lambda _lock: None)
    monkeypatch.setattr(module, "verify_imports", lambda *, accelerated: None)
    monkeypatch.setattr(module, "_cli", lambda *args: commands.append(args))
    monkeypatch.setattr(module, "preserve_results", lambda: None)
    monkeypatch.setattr(
        module,
        "_run_fixture_gate",
        lambda *, smoke_output_root=None: fixture_calls.append(
            {"smoke_output_root": smoke_output_root}
        ),
    )
    monkeypatch.setattr(module, "_run", lambda command: commands.append(tuple(command)))

    module.run_smoke()

    assert fixture_calls == [
        {"smoke_output_root": module._run_root("configs/fixture.yaml", "fixture-smoke")}
    ]
    assert commands[0] == (
        sys.executable,
        "scripts/validate_project.py",
        "--require-fixture-artifacts",
    )


def test_bundle_result_checkpoints_are_immutable_without_copying_adapters(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    bundle = tmp_path / "bundle"
    project = bundle / "lora-audit"
    artifacts = project / "artifacts"
    artifacts.mkdir(parents=True)
    evidence = artifacts / "evidence.json"
    evidence.write_text('{"stage":1}\n', encoding="utf-8")
    (artifacts / "one-click-completion.json").write_text(
        '{"status":"complete"}\n', encoding="utf-8"
    )
    monkeypatch.setattr(module, "BUNDLE", bundle)
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "RUN_OUTPUT", bundle / "run-output")

    first = module.preserve_results()
    assert first is not None
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    evidence.write_text('{"stage":2}\n', encoding="utf-8")
    second = module.preserve_results()

    assert second is not None and second != first
    assert json.loads(first.read_text(encoding="utf-8")) == first_payload
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert first_payload["status"] == second_payload["status"] == "checkpointed"
    assert first_payload["excluded_paths"] == ["one-click-completion.json"]
    assert first_payload["artifact_tree_sha256"] != second_payload["artifact_tree_sha256"]
    assert [entry["path"] for entry in first_payload["entries"]] == ["evidence.json"]
    assert [entry["path"] for entry in second_payload["entries"]] == ["evidence.json"]
    assert not (bundle / "run-output/snapshots").exists()
    latest = json.loads((bundle / "run-output/latest.json").read_text(encoding="utf-8"))
    assert latest["checkpoint"] == second.relative_to(bundle / "run-output").as_posix()


def test_bundle_accelerated_modes_require_explicit_execute_before_side_effects(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "verify_persisted_upload_contract",
        lambda: ({}, {}),
    )
    monkeypatch.setattr(
        module,
        "run_pilot",
        lambda _source, _bundle, _profile: calls.append("pilot"),
    )
    monkeypatch.setattr(
        module,
        "configure_runtime_resources",
        lambda *_args: calls.append("resources") or {},
    )

    with pytest.raises(SystemExit, match="explicit --execute"):
        module.main(["--mode", "pilot"])
    assert calls == []

    with pytest.raises(SystemExit, match="require --execution-profile"):
        module.main(["--mode", "pilot", "--execute"])
    assert (
        module.main(
            [
                "--mode",
                "pilot",
                "--execute",
                "--execution-profile",
                "configs/execution/a100-40gb.yaml",
            ]
        )
        == 0
    )
    assert calls == ["resources", "pilot"]


@pytest.mark.parametrize(
    "mode",
    [
        "rq1",
        "rq2-condition",
        "rq2-finalize",
        "rq3-cell",
        "rq3-finalize",
        "verify-results",
        "detector-comparison",
        "adaptive-robustness-condition",
        "adaptive-robustness-finalize",
        "cross-language-transfer-cell",
        "cross-language-transfer-finalize",
        "final-verification",
        "research",
    ],
)
def test_bundle_granular_research_modes_require_explicit_execute(
    project_root: Path, mode: str
) -> None:
    module = _load_bundle_bootstrap(project_root)

    with pytest.raises(SystemExit, match="explicit --execute"):
        module._require_execute(mode, False)


def test_bundle_granular_selectors_are_validated_before_attestation(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    attestation_calls: list[str] = []
    monkeypatch.setattr(
        module,
        "verify_persisted_upload_contract",
        lambda: attestation_calls.append("attest") or ({}, {}),
    )
    common = [
        "--execute",
        "--execution-profile",
        "configs/execution/a100-40gb.yaml",
    ]

    with pytest.raises(SystemExit, match="requires --condition-index"):
        module.main(["--mode", "rq2-condition", *common])
    with pytest.raises(SystemExit, match="requires --source-language"):
        module.main(["--mode", "rq3-cell", *common])
    with pytest.raises(SystemExit, match=r"source-bound 12\.0"):
        module.main(
            [
                "--mode",
                "cohort",
                *common,
                "--available-device-hours",
                "20",
            ]
        )

    assert attestation_calls == []


def test_bundle_notebook_contract_uses_professional_stage_names(project_root: Path) -> None:
    module = _load_bundle_bootstrap(project_root)

    contract = module.build_notebook_contract()

    assert contract["deterministic_runtime_decisions_verified"] is True
    assert "schema_version" not in contract
    assert "contract_id" not in contract
    assert "source_plan_id" not in contract
    assert "protocol_amendment" not in contract
    assert "policy_id" not in contract["runtime_decision_policy"]
    assert contract["representative_pilot"]["pair_count"] == 3
    assert contract["representative_pilot"]["maximum_adapter_runs"] == 6
    assert contract["full_pipeline_compute_gate_verified"] is True
    assert contract["cross_language_trigger_gate_verified"] is True
    assert contract["adaptive_condition_indices"] == list(range(12))
    assert len(contract["language_transfer_cells"]) == 4
    assert contract["study_stage_modes"] == {
        "core_experiment": "cohort",
        "detector_comparison": "detector-comparison",
        "adaptive_condition": "adaptive-robustness-condition",
        "adaptive_finalize": "adaptive-robustness-finalize",
        "language_transfer_cell": "cross-language-transfer-cell",
        "language_transfer_finalize": "cross-language-transfer-finalize",
        "final_verification": "final-verification",
    }


def test_bundle_notebook_contract_keeps_stdout_machine_readable(
    project_root: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_bundle_bootstrap(project_root)

    def verify_with_diagnostic() -> tuple[dict, dict]:
        print("verification diagnostic")
        return {}, {}

    monkeypatch.setattr(
        module,
        "verify_persisted_upload_contract_read_only",
        verify_with_diagnostic,
    )

    assert module.main(["--mode", "notebook-contract"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["deterministic_runtime_decisions_verified"] is True
    assert "schema_version" not in payload
    assert "contract_id" not in payload
    assert "verification diagnostic" in captured.err
    assert "verification diagnostic" not in captured.out


def test_bundle_status_is_read_only_and_never_creates_an_attestation_receipt(
    project_root: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_bundle_bootstrap(project_root)
    monkeypatch.setattr(
        module,
        "verify_persisted_upload_contract_read_only",
        lambda: ({"tree_sha256": "a" * 64}, {"tree_sha256": "b" * 64}),
    )
    monkeypatch.setattr(
        module,
        "_write_json_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("status must not write an artifact")
        ),
    )

    assert module.main(["--mode", "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "observed"


def test_bundle_report_asset_status_is_hash_checked_and_presentation_ordered(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    project = tmp_path / "lora-audit"
    run_root = project / "artifacts/runs/core-test"
    report_root = run_root / "report-assets"
    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True)
    names = (
        "cohort-quality-control.svg",
        "backdoor-validity-and-stealth.svg",
        "detector-comparison.svg",
        "calibration-analysis.svg",
        "adaptive-tradeoff.svg",
        "english-vietnamese-transfer.svg",
        "operational-latency.svg",
    )
    figures = {}
    for index, name in enumerate(reversed(names)):
        path = figure_root / name
        path.write_text(f"<svg><text>{index}</text></svg>\n", encoding="utf-8")
        figures[f"asset-{index}"] = {
            "svg": {
                "path": path.relative_to(report_root).as_posix(),
                "sha256": sha256_file(path),
            }
        }
    (report_root / "figure-manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "fixture_smoke_only": False,
                "figures": figures,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT", project)

    complete, paths = module._report_asset_status(run_root)

    assert complete is True
    assert [Path(path).name for path in paths] == list(names)
    (figure_root / names[0]).write_text("<svg />\n", encoding="utf-8")
    assert module._report_asset_status(run_root) == (False, [])


@pytest.mark.parametrize(
    ("professional_mode", "canonical_mode"),
    [
        ("detector-comparison", "rq1"),
        ("adaptive-robustness-condition", "rq2-condition"),
        ("adaptive-robustness-finalize", "rq2-finalize"),
        ("cross-language-transfer-cell", "rq3-cell"),
        ("cross-language-transfer-finalize", "rq3-finalize"),
        ("final-verification", "verify-results"),
    ],
)
def test_bundle_professional_modes_preserve_internal_compatibility(
    project_root: Path,
    professional_mode: str,
    canonical_mode: str,
) -> None:
    module = _load_bundle_bootstrap(project_root)

    assert module._canonical_mode(professional_mode) == canonical_mode


def test_bundle_one_click_plan_is_exact_and_rejects_budget_expansion(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    plan = module.load_one_click_plan()

    assert plan["plan_id"] == "colab-pro-one-click-v4"
    assert plan["runtime_decision_policy"] == {
        "policy_id": "non-human-runtime-decisions-v1",
        "controller": "deterministic_protocol",
        "human_in_run_decisions": False,
        "model_role": "measurement_only",
        "unexpected_state_action": "fail_closed",
        "semantic_trigger_drift_action": "reject",
        "manual_override_allowed": False,
        "platform_authorization": "external_precondition",
    }
    assert plan["available_device_hours"] == 12.0
    assert plan["representative_pilot"]["pair_count"] == 3
    assert plan["compute_gate"]["scope"] == "full_research_pipeline"
    assert plan["cohort_batch_size"] == 4
    assert plan["cohort_batch_invocations"] == 9
    assert plan["rq2_condition_indices"] == list(range(12))
    assert len(plan["rq3_cells"]) == 4
    assert plan["rq3_trigger_gate"]["disposition"] == "algorithmic_literal_preservation"

    tampered_project = tmp_path / "lora-audit"
    tampered_path = tampered_project / module.ONE_CLICK_PLAN_PATH
    tampered_path.parent.mkdir(parents=True)
    tampered_path.write_text(
        json.dumps({**plan, "available_device_hours": 13.0}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT", tampered_project)
    monkeypatch.setattr(module, "_validate_protocol_amendment_binding", lambda _binding: None)

    with pytest.raises(ValueError, match="at most 12 hours"):
        module.load_one_click_plan()


def test_bundle_runtime_resource_policy_is_exact_and_source_bound(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    policy = module.load_runtime_resource_policy()

    assert policy["policy_id"] == "a100-high-capacity-runtime-v1"
    assert policy["execution_profile_id"] == "a100-40gb-v1"
    assert policy["system_memory"]["minimum_total_gib"] == 80.0
    assert policy["root_disk"]["minimum_total_gib"] == 100.0
    assert policy["local_scratch"]["filesystem_label"] == "local-scratch"
    assert policy["local_scratch"]["minimum_total_gib"] == 100.0
    assert policy["durable_artifacts"]["scratch_is_authoritative"] is False

    tampered_project = tmp_path / "lora-audit"
    tampered_path = tampered_project / module.RUNTIME_RESOURCE_POLICY_PATH
    tampered_path.parent.mkdir(parents=True)
    tampered = json.loads(json.dumps(policy))
    tampered["local_scratch"]["minimum_total_gib"] = 1.0
    tampered_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "PROJECT", tampered_project)

    with pytest.raises(ValueError, match="source-bound contract"):
        module.load_runtime_resource_policy()


def test_discover_local_scratch_accepts_colab_workspace_on_root_disk(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    workspace = tmp_path / "content"
    workspace.mkdir()
    policy = module._runtime_resource_policy_expected()

    monkeypatch.setattr(
        module,
        "_disk_snapshot",
        lambda path: {
            "path": str(path),
            "device_id": 99,
            "total_gib": 120.0,
            "used_gib": 10.0,
            "free_gib": 110.0,
        },
    )
    monkeypatch.setattr(module, "_mountinfo_entries", lambda _path=None: [])
    monkeypatch.setattr(module, "_findmnt_labeled_mounts", lambda _label: [])

    scratch, snapshot = module._discover_local_scratch(
        policy,
        mountinfo_path=tmp_path / "missing-mountinfo",
        label_root=tmp_path / "missing-labels",
        known_candidates=(workspace, workspace / "local-scratch"),
    )

    assert scratch == workspace
    assert snapshot["filesystem_label"] == "local-scratch"


def test_discover_local_scratch_creates_missing_leaf_candidate(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    workspace = tmp_path / "content"
    workspace.mkdir()
    leaf = workspace / "local-scratch"
    policy = module._runtime_resource_policy_expected()

    monkeypatch.setattr(
        module,
        "_disk_snapshot",
        lambda path: {
            "path": str(path),
            "device_id": 99,
            "total_gib": 120.0,
            "used_gib": 10.0,
            "free_gib": 110.0,
        },
    )
    monkeypatch.setattr(module, "_mountinfo_entries", lambda _path=None: [])
    monkeypatch.setattr(module, "_findmnt_labeled_mounts", lambda _label: [])

    scratch, _snapshot = module._discover_local_scratch(
        policy,
        mountinfo_path=tmp_path / "missing-mountinfo",
        label_root=tmp_path / "missing-labels",
        known_candidates=(leaf,),
    )

    assert scratch == leaf
    assert leaf.is_dir()


def test_bundle_runtime_resource_routing_uses_scratch_without_moving_artifacts(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    project = tmp_path / "lora-audit"
    artifacts = project / "artifacts"
    scratch = tmp_path / "local-scratch"
    artifacts.mkdir(parents=True)
    scratch.mkdir()
    policy_path = project / module.RUNTIME_RESOURCE_POLICY_PATH
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        json.dumps(module._runtime_resource_policy_expected()) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(
        module,
        "_system_memory_snapshot",
        lambda: {"total_gib": 167.1, "available_gib": 164.7, "used_gib": 2.4},
    )
    monkeypatch.setattr(
        module,
        "_disk_snapshot",
        lambda path: {
            "path": str(path),
            "device_id": 1,
            "total_gib": 235.7,
            "used_gib": 46.7,
            "free_gib": 189.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_discover_local_scratch",
        lambda _policy: (
            scratch,
            {
                "path": str(scratch),
                "device_id": 2,
                "total_gib": 368.0,
                "used_gib": 0.0,
                "free_gib": 368.0,
                "filesystem_label": "local-scratch",
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_nvidia_smi_snapshot",
        lambda: [
            {
                "name": "NVIDIA A100-SXM4-40GB",
                "memory_total_gib": 40.0,
                "memory_used_gib": 0.0,
                "utilization_gpu_percent": 0.0,
                "power_draw_watts": 50.0,
                "power_limit_watts": 400.0,
            }
        ],
    )
    monkeypatch.setattr(module, "_runtime_boot_id", lambda: "a" * 32)
    for name in module._runtime_resource_policy_expected()["cache_bindings"]:
        monkeypatch.setenv(name, "before-routing")

    receipt = module.configure_runtime_resources(
        {"tree_sha256": "1" * 64},
        {"tree_sha256": "2" * 64},
    )

    assert receipt["status"] == "passed"
    assert receipt["local_scratch"]["filesystem_label"] == "local-scratch"
    assert receipt["durable_artifacts"] == {
        "location": "google_drive_bundle",
        "scratch_is_authoritative": False,
    }
    assert Path(receipt["cache_root"]).is_relative_to(scratch)
    assert all(Path(path).is_relative_to(scratch) for path in receipt["cache_bindings"].values())
    assert module._RESOURCE_ROUTING_STATE["artifact_path"] == artifacts
    persisted = json.loads(
        (project / module.RUNTIME_RESOURCE_ROUTING_RECEIPT).read_text(encoding="utf-8")
    )
    assert persisted == receipt


def test_bundle_resource_monitor_summarizes_observed_peaks(
    project_root: Path,
) -> None:
    module = _load_bundle_bootstrap(project_root)
    module._RESOURCE_ROUTING_STATE = {
        "policy": module._runtime_resource_policy_expected(),
        "routing_identity_sha256": "f" * 64,
    }

    def sample(
        *,
        monotonic: float,
        ram_used: float,
        gpu_used: float,
        gpu_utilization: float,
        scratch_used: float,
    ) -> dict[str, object]:
        disk = {
            "path": "/",
            "device_id": 1,
            "total_gib": 235.7,
            "used_gib": 46.7,
            "free_gib": 189.0,
        }
        scratch = {
            "path": "/mnt/disks/local-scratch",
            "device_id": 2,
            "total_gib": 368.0,
            "used_gib": scratch_used,
            "free_gib": 368.0 - scratch_used,
        }
        return {
            "observed_at_utc": f"2026-08-09T00:00:0{int(monotonic)}+00:00",
            "monotonic_seconds": monotonic,
            "system_memory": {
                "total_gib": 167.1,
                "available_gib": 167.1 - ram_used,
                "used_gib": ram_used,
            },
            "root_disk": disk,
            "artifact_disk": {**disk, "path": "/content/drive"},
            "local_scratch": scratch,
            "gpu": [
                {
                    "name": "NVIDIA A100-SXM4-40GB",
                    "memory_total_gib": 40.0,
                    "memory_used_gib": gpu_used,
                    "utilization_gpu_percent": gpu_utilization,
                    "power_draw_watts": 300.0,
                    "power_limit_watts": 400.0,
                }
            ],
        }

    summary = module._summarize_resource_samples(
        [
            sample(
                monotonic=1.0,
                ram_used=2.4,
                gpu_used=0.0,
                gpu_utilization=0.0,
                scratch_used=0.0,
            ),
            sample(
                monotonic=2.0,
                ram_used=48.0,
                gpu_used=62.0,
                gpu_utilization=97.0,
                scratch_used=12.0,
            ),
        ],
        command_label="lora_audit.cli:run-cohort",
        command_status="passed",
        wall_seconds=10.0,
        monitor_errors=[],
    )

    assert summary["sample_count"] == 2
    assert summary["system_memory"]["used_peak_gib"] == 48.0
    assert summary["gpu"]["memory_used_peak_gib"] == 62.0
    assert summary["gpu"]["utilization_max_percent"] == 97.0
    assert summary["local_scratch"]["used_peak_gib"] == 12.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("human_in_run_decisions", True),
        ("model_role", "protocol_controller"),
        ("manual_override_allowed", True),
        ("unexpected_state_action", "ask_operator"),
    ],
)
def test_bundle_one_click_plan_rejects_runtime_decision_authority_drift(
    project_root: Path,
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: object,
) -> None:
    module = _load_bundle_bootstrap(project_root)
    plan = module.load_one_click_plan()
    tampered = json.loads(json.dumps(plan))
    tampered["runtime_decision_policy"][field] = value
    tampered_project = tmp_path / "lora-audit"
    tampered_path = tampered_project / module.ONE_CLICK_PLAN_PATH
    tampered_path.parent.mkdir(parents=True)
    tampered_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    amendment_source = project_root / module.PROTOCOL_AMENDMENT_PATH
    amendment_target = tampered_project / module.PROTOCOL_AMENDMENT_PATH
    amendment_target.parent.mkdir(parents=True, exist_ok=True)
    amendment_target.write_bytes(amendment_source.read_bytes())
    monkeypatch.setattr(module, "PROJECT", tampered_project)

    with pytest.raises(ValueError, match="runtime decision policy changed"):
        module.load_one_click_plan()


def test_bundle_one_click_trigger_gate_binds_the_exact_algorithmic_scope(
    project_root: Path,
) -> None:
    module = _load_bundle_bootstrap(project_root)
    plan = module.load_one_click_plan()

    gate = module._require_one_click_trigger_gate(plan)
    first = module._trigger_gate_attestation_sha256(plan)
    second = module._trigger_gate_attestation_sha256(plan)

    assert gate["bound_registry_sha256"] == sha256_file(
        project_root / "private/trigger_registry.yaml"
    )
    assert first == second
    assert len(first) == 64


def test_bundle_trigger_gate_rejects_a_changed_disposition_before_setup(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    plan = module.load_one_click_plan()
    tampered = json.loads(json.dumps(plan))
    tampered["rq3_trigger_gate"]["disposition"] = "approved"
    calls: list[str] = []
    monkeypatch.setattr(module, "load_one_click_plan", lambda: tampered)
    monkeypatch.setattr(
        module,
        "_prepare_accelerated_stage",
        lambda _profile: calls.append("accelerated-setup"),
    )

    with pytest.raises(ValueError, match="not algorithmically source-locked"):
        module.run_preflight({}, {}, project_root / "configs/execution/a100-40gb.yaml")

    assert calls == []


def test_bundle_accelerated_preparation_builds_rq3_receipts_after_data(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    events: list[str] = []
    profile = SimpleNamespace()
    monkeypatch.setattr(module, "install_environment", lambda _lock: events.append("install"))
    monkeypatch.setattr(
        module,
        "verify_imports",
        lambda *, accelerated: events.append(f"imports:{accelerated}"),
    )
    monkeypatch.setattr(module, "_load_strict_execution_profile", lambda _path: profile)
    monkeypatch.setattr(module, "_profile_relative", lambda _path: "profile.yaml")
    monkeypatch.setattr(module, "_run_fixture_gate", lambda: events.append("fixture"))
    monkeypatch.setattr(
        module,
        "_cli",
        lambda command, *_args: events.append(command),
    )
    monkeypatch.setattr(module, "_run_root", lambda *_args: "artifacts/runs/core-test")
    monkeypatch.setattr(
        module,
        "_prepare_rq3_private_receipts",
        lambda root: events.append(f"rq3-receipts:{root}"),
    )

    observed_profile, root = module._prepare_accelerated_stage(
        project_root / "configs/execution/a100-40gb.yaml"
    )

    assert observed_profile is profile
    assert root == "artifacts/runs/core-test"
    assert events.index("prepare-data") < events.index("rq3-receipts:artifacts/runs/core-test")


def test_bundle_rq3_receipt_preparation_uses_the_source_bound_literal_gate(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    import lora_audit.config as config_module
    import lora_audit.training as training_module
    import lora_audit.trigger_receipts as receipt_module

    module = _load_bundle_bootstrap(project_root)
    plan = module.load_one_click_plan()
    gate = plan["rq3_trigger_gate"]
    attestation = module._trigger_gate_attestation_sha256(plan)
    project = tmp_path / "lora-audit"
    registry = project / "private/trigger_registry.yaml"
    prepared = project / "artifacts/runs/core-test/data/manifest.jsonl"
    registry.parent.mkdir(parents=True)
    prepared.parent.mkdir(parents=True)
    registry.write_text("schema_version: 1\n", encoding="utf-8")
    prepared.write_text("{}\n", encoding="utf-8")

    trigger_receipt = SimpleNamespace(
        receipt_sha256="a" * 64,
        model_dump=lambda **_kwargs: {"receipt_sha256": "a" * 64},
    )
    common_receipt = SimpleNamespace(
        receipt_sha256="b" * 64,
        model_dump=lambda **_kwargs: {"receipt_sha256": "b" * 64},
    )
    trigger_calls: list[dict[str, object]] = []
    trigger_validation_calls: list[dict[str, object]] = []

    def build_trigger(**kwargs):
        trigger_calls.append(kwargs)
        return trigger_receipt

    def validate_trigger(**kwargs):
        trigger_validation_calls.append(kwargs)
        return trigger_receipt, None, None

    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "load_one_click_plan", lambda: plan)
    monkeypatch.setattr(module, "_require_one_click_trigger_gate", lambda _plan: gate)
    monkeypatch.setattr(module, "_trigger_gate_attestation_sha256", lambda _plan: attestation)
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda _path: SimpleNamespace(
            private_trigger_registry=Path("private/trigger_registry.yaml")
        ),
    )
    monkeypatch.setattr(training_module, "read_prepared_records", lambda _path: [])
    monkeypatch.setattr(receipt_module, "load_receipt_tokenizer", lambda _config: object())
    monkeypatch.setattr(receipt_module, "build_trigger_equivalence_receipt", build_trigger)
    monkeypatch.setattr(
        receipt_module,
        "build_common_probe_receipt",
        lambda **_kwargs: common_receipt,
    )
    monkeypatch.setattr(
        receipt_module,
        "load_and_validate_trigger_equivalence_receipt",
        validate_trigger,
    )
    monkeypatch.setattr(
        receipt_module,
        "load_and_validate_common_probe_receipt",
        lambda **_kwargs: (common_receipt, None),
    )

    module._prepare_rq3_private_receipts("artifacts/runs/core-test")

    assert len(trigger_calls) == 1
    assert trigger_calls[0]["equivalence_modes"] == gate["scope"]["equivalence_modes"]
    assert trigger_calls[0]["human_review_status"] == "waived_literal_preserved"
    assert trigger_calls[0]["human_review_attestation_sha256"] == attestation
    assert trigger_validation_calls[0]["required_waiver_attestation_sha256"] == attestation
    assert len(attestation) == 64
    assert (project / module.RQ3_TRIGGER_RECEIPT).is_file()
    assert (project / module.RQ3_COMMON_PROBE_RECEIPT).is_file()


def test_bundle_rq3_commands_bind_generated_private_receipts(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    expected_attestation = module._trigger_gate_attestation_sha256(module.load_one_click_plan())
    profile_path = project_root / "configs/execution/a100-40gb.yaml"
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        module,
        "_prepared_research_context",
        lambda *_args: (
            SimpleNamespace(),
            "artifacts/runs/core-test",
            "configs/execution/a100-40gb.yaml",
        ),
    )
    monkeypatch.setattr(module, "_require_rq1_release", lambda _root: None)
    monkeypatch.setattr(module, "_cli", lambda *args: commands.append(args))

    module.run_rq3_cell(
        {},
        {},
        profile_path,
        source_language="en-US",
        probe_language="vi-VN",
    )
    module.run_rq3_finalize({}, {}, profile_path)

    for command in commands:
        trigger_index = command.index("--trigger-equivalence-receipt")
        common_index = command.index("--common-probe-receipt")
        waiver_index = command.index("--required-trigger-waiver-attestation-sha256")
        assert command[trigger_index + 1] == ("artifacts/private/trigger-equivalence-receipt.json")
        assert command[common_index + 1] == "artifacts/private/common-probe-receipt.json"
        assert command[waiver_index + 1] == expected_attestation


def test_bundle_pilot_binds_profile_and_bundle_manifest_to_execution(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    profile_path = project_root / "configs/execution/a100-40gb.yaml"
    commands: list[tuple[str, ...]] = []
    receipt_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "_load_prepared_accelerated_stage",
        lambda _source, _bundle, _profile_path: (
            SimpleNamespace(),
            "artifacts/runs/core-test",
        ),
    )
    pilot_ids = ["lineage-1", "lineage-2", "lineage-3"]
    monkeypatch.setattr(module, "_pilot_lineage_ids", lambda: pilot_ids)
    monkeypatch.setattr(
        module,
        "write_execution_receipt",
        lambda *_args, **kwargs: receipt_calls.append(kwargs),
    )
    monkeypatch.setattr(module, "_cli", lambda *args: commands.append(args))
    monkeypatch.setattr(module, "preserve_results", lambda: None)
    estimate_index = iter(range(3))

    def fake_estimate(**_kwargs):
        index = next(estimate_index)
        prefix = pilot_ids[: index + 1]
        return {
            "schema_version": 9,
            "scope": "full_research_pipeline",
            "available_device_hours": 12.0,
            "pilot_pair_count": len(prefix),
            "pilot_pair_bindings": [
                {"parent_config_id": lineage_id} for lineage_id in reversed(prefix)
            ],
            "recommendation": "full_research" if index == 2 else "pilot_incomplete",
            "pilot_continuation": {
                "policy": "outcome_classified_optimistic_full_research_v2",
                "required_pair_count": 3,
                "completed_pair_count": len(prefix),
                "remaining_pair_count": 3 - len(prefix),
                "action": "authorize_full_research" if index == 2 else "continue_pilot",
                "reason": "test_fixture",
            },
        }

    monkeypatch.setattr(module, "_refresh_observed_compute_estimate", fake_estimate)

    module.run_pilot({}, {}, profile_path)

    assert len(receipt_calls) == 3
    assert [call["lineage_ids"] for call in receipt_calls] == [[item] for item in pilot_ids]
    assert all(call["phase"] == "pilot" for call in receipt_calls)
    assert all(isinstance(call["execution_profile"], SimpleNamespace) for call in receipt_calls)
    assert len(commands) == 3
    for pilot_command in commands:
        assert "--execution-profile" in pilot_command
        assert "configs/execution/a100-40gb.yaml" in pilot_command
        bundle_index = pilot_command.index("--bundle-manifest")
        assert pilot_command[bundle_index : bundle_index + 2] == (
            "--bundle-manifest",
            "artifacts/bundle-manifest.json",
        )


def test_bundle_pilot_uses_continuation_action_instead_of_interim_recommendation(
    project_root: Path,
) -> None:
    module = _load_bundle_bootstrap(project_root)
    estimate = {
        "pilot_pair_count": 1,
        "pilot_pair_bindings": [{"parent_config_id": "lineage-1"}],
        "recommendation": "core_only",
        "pilot_continuation": {
            "policy": "outcome_classified_optimistic_full_research_v2",
            "required_pair_count": 3,
            "completed_pair_count": 1,
            "remaining_pair_count": 2,
            "action": "continue_pilot",
            "reason": "remaining_pairs_can_still_change_the_full_pipeline_budget_verdict",
        },
    }

    module._require_pilot_decision(
        estimate,
        expected_lineage_ids=["lineage-1"],
        final=False,
    )

    estimate["pilot_continuation"] = {
        **estimate["pilot_continuation"],
        "action": "stop_budget_infeasible",
    }
    with pytest.raises(RuntimeError, match="continuation gate"):
        module._require_pilot_decision(
            estimate,
            expected_lineage_ids=["lineage-1"],
            final=False,
        )


def test_bundle_cohort_batch_binds_only_exact_pending_lineage_ids(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    profile_path = project_root / "configs/execution/a100-40gb.yaml"
    profile = SimpleNamespace()
    commands: list[tuple[str, ...]] = []
    receipt_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "_load_prepared_accelerated_stage",
        lambda _source, _bundle, _profile_path: (
            profile,
            "artifacts/runs/core-test",
        ),
    )
    monkeypatch.setattr(
        module,
        "_pending_cohort_lineage_ids",
        lambda _root, _profile, _path: ["lineage-a", "lineage-b", "lineage-c"],
    )
    monkeypatch.setattr(
        module,
        "write_execution_receipt",
        lambda *_args, **kwargs: receipt_calls.append(kwargs),
    )
    monkeypatch.setattr(module, "_cli", lambda *args: commands.append(args))
    monkeypatch.setattr(module, "preserve_results", lambda: None)

    module.run_cohort(
        {},
        {},
        profile_path,
        available_device_hours=12.0,
        batch_size=2,
    )

    assert receipt_calls[0]["phase"] == "cohort"
    assert receipt_calls[0]["lineage_ids"] == ["lineage-a", "lineage-b"]
    cohort_command = commands[-1]
    assert "--max-lineages" not in cohort_command
    assert [
        cohort_command[index + 1]
        for index, value in enumerate(cohort_command[:-1])
        if value == "--lineage-id"
    ] == ["lineage-a", "lineage-b"]


def test_bundle_cohort_receipt_matches_runtime_compute_authorization(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    from lora_audit.readiness import _verified_cohort_compute_authorization

    module = _load_bundle_bootstrap(project_root)
    project = tmp_path / "lora-audit"
    config_path = project / "configs/core.yaml"
    artifacts = project / "artifacts"
    config_path.parent.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    config_path.write_text(
        (project_root / "configs/core.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_config(config_path)
    planned_adapter_runs = len(plan_cohort(config)) * 2
    execution_profile_binding = {"profile_id": "a100-40gb-v1"}
    execution_profile = SimpleNamespace(binding=lambda: execution_profile_binding)
    source_manifest = {"tree_sha256": "a" * 64, "entry_count": 1}
    bundle_manifest = {"tree_sha256": "b" * 64, "entry_count": 3}
    (artifacts / "source-manifest.json").write_text(
        json.dumps(source_manifest) + "\n", encoding="utf-8"
    )
    (artifacts / "bundle-manifest.json").write_text(
        json.dumps(bundle_manifest) + "\n", encoding="utf-8"
    )
    (artifacts / "preflight-live.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "execution_profile": execution_profile_binding,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    estimate_path = artifacts / "compute-estimate.json"
    estimate_path.write_text(
        json.dumps(
            {
                "execution_profile": execution_profile_binding,
                "available_device_hours": 12.0,
                "recommendation": "full_research",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT", project)

    module.write_execution_receipt(
        source_manifest,
        bundle_manifest,
        execution_profile=execution_profile,
        phase="cohort",
        lineage_ids=["lineage-a"],
        compute_estimate=estimate_path,
    )

    receipt = json.loads((artifacts / "execution-receipt.json").read_text(encoding="utf-8"))
    verified = _verified_cohort_compute_authorization(
        execution_phase="cohort",
        compute_estimate_path=estimate_path,
        receipt=receipt,
        config=config,
        planned_adapter_runs=planned_adapter_runs,
    )
    assert receipt["compute_authorization"] == verified


def test_bundle_verify_results_emits_completion_only_after_all_units(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    project = tmp_path / "lora-audit"
    core = project / "artifacts/runs/core-test"
    plan = {
        "plan_id": "colab-pro-one-click-v4",
        "protocol_amendment": {
            "path": "configs/protocol-amendment-v4.json",
            "amendment_id": "post-pilot-outcome-classification-amendment-v4",
            "sha256": "9" * 64,
        },
        "runtime_decision_policy": {"policy_id": "non-human-runtime-decisions-v1"},
        "available_device_hours": 12.0,
        "representative_pilot": {"pair_count": 3},
        "compute_gate": {"required_recommendation": "full_research"},
        "cohort_batch_size": 4,
        "rq3_trigger_gate": {"disposition": "algorithmic_literal_preservation"},
    }
    required = {
        core / "cohort/run-state.json": {"status": "completed"},
        core / "detectors/calibration-profile.json": {"status": "locked"},
        core / "detectors/test-release.json": {"status": "released"},
        core / "rq2/results.json": {"status": "completed"},
        core / "rq3/results.jsonl": {"status": "completed"},
        core / "report-assets/result-register.json": {"status": "complete"},
        core / "report-assets/figure-manifest.json": {"status": "complete"},
        project / module.RQ3_TRIGGER_RECEIPT: {"receipt_sha256": "c" * 64},
        project / module.RQ3_COMMON_PROBE_RECEIPT: {"receipt_sha256": "d" * 64},
        project / "artifacts/compute-estimate.json": {
            "schema_version": 9,
            "scope": "full_research_pipeline",
            "estimate_kind": "observed_pilot",
            "pilot_pair_count": 3,
            "available_device_hours": 12.0,
            "recommendation": "full_research",
        },
        project / "artifacts/source-manifest.json": {"tree_sha256": "a" * 64},
        project / "artifacts/bundle-manifest.json": {"tree_sha256": "b" * 64},
        project / module.ONE_CLICK_PLAN_PATH: plan,
        project / module.RUNTIME_RESOURCE_POLICY_PATH: {"policy_id": "test-policy"},
        project / module.RUNTIME_RESOURCE_ROUTING_RECEIPT: {
            "status": "passed",
            "routing_identity_sha256": "f" * 64,
        },
        project / module.RUNTIME_RESOURCE_TELEMETRY_LEDGER: {
            "routing_identity_sha256": "f" * 64,
        },
        project / module.RUNTIME_RESOURCE_SUMMARY: {
            "status": "observed",
            "source_tree_sha256": "a" * 64,
            "bundle_tree_sha256": "b" * 64,
        },
    }
    for path, payload in required.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    for index in range(12):
        path = core / f"rq2/runs/condition-{index:02d}/condition-result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    for cell in ("en-US-to-en-US", "en-US-to-vi-VN", "vi-VN-to-vi-VN", "vi-VN-to-en-US"):
        path = core / f"rq3/cells/{cell}/cell-result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    calls: list[str] = []
    profile = SimpleNamespace(binding=lambda: {"profile_id": "a100-40gb-v1"})
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "RUN_OUTPUT", tmp_path / "run-output")
    monkeypatch.setattr(module, "load_one_click_plan", lambda: plan)
    monkeypatch.setattr(
        module,
        "_require_one_click_trigger_gate",
        lambda _plan: _plan["rq3_trigger_gate"],
    )
    monkeypatch.setattr(module, "_trigger_gate_attestation_sha256", lambda _plan: "e" * 64)
    monkeypatch.setattr(
        module,
        "_prepared_research_context",
        lambda *_args: (profile, "artifacts/runs/core-test", "configs/execution/a100-40gb.yaml"),
    )
    monkeypatch.setattr(module, "_require_rq1_release", lambda _root: None)
    monkeypatch.setattr(
        module,
        "run_rq2_finalize",
        lambda *_args: calls.append("rq2-finalized"),
    )
    monkeypatch.setattr(
        module,
        "run_rq3_finalize",
        lambda *_args: calls.append("rq3-finalized"),
    )
    monkeypatch.setattr(
        module,
        "summarize_runtime_resource_telemetry",
        lambda: {
            "status": "observed",
            "source_tree_sha256": "a" * 64,
            "bundle_tree_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        module,
        "generate_completed_report_assets",
        lambda **_kwargs: (
            calls.append("report-assets-generated")
            or {"status": "complete", "fixture_smoke_only": False}
        ),
    )

    completion_path = project / "artifacts/one-click-completion.json"
    real_preserve_results = module.preserve_results

    def fail_checkpoint():
        assert not completion_path.exists()
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(module, "preserve_results", fail_checkpoint)
    with pytest.raises(RuntimeError, match="simulated checkpoint failure"):
        module.run_verify_results(
            {"tree_sha256": "a" * 64},
            {"tree_sha256": "b" * 64},
            project / "configs/execution/a100-40gb.yaml",
        )
    assert not completion_path.exists()
    calls.clear()

    def checkpoint_before_completion():
        assert not completion_path.exists()
        return real_preserve_results()

    monkeypatch.setattr(module, "preserve_results", checkpoint_before_completion)

    module.run_verify_results(
        {"tree_sha256": "a" * 64},
        {"tree_sha256": "b" * 64},
        project / "configs/execution/a100-40gb.yaml",
    )

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert calls == ["rq2-finalized", "rq3-finalized", "report-assets-generated"]
    assert completion["schema_version"] == 4
    assert completion["status"] == "complete"
    assert completion["cohort_lineage_count"] == 36
    assert completion["rq2_condition_count"] == 12
    assert completion["rq3_cell_count"] == 4
    assert completion["plan"]["rq3_trigger_gate_attestation_sha256"] == "e" * 64
    assert "rq3_trigger_equivalence_receipt" in completion["results"]
    assert "rq3_common_probe_receipt" in completion["results"]
    assert "runtime_resource_summary" in completion["results"]
    assert "report_result_register" in completion["results"]
    assert "report_figure_manifest" in completion["results"]
    assert completion["plan"]["runtime_resource_policy_id"] == ("a100-high-capacity-runtime-v1")
    checkpoint = tmp_path / "run-output" / completion["result_checkpoint"]["path"]
    assert checkpoint.is_file()
    assert completion["result_checkpoint"]["sha256"] == sha256_file(checkpoint)
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert (
        completion["result_checkpoint"]["artifact_tree_sha256"]
        == (checkpoint_payload["artifact_tree_sha256"])
    )
    assert "one-click-completion.json" not in {
        entry["path"] for entry in checkpoint_payload["entries"]
    }


def test_bundle_live_session_marker_is_required_and_exact(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    marker = tmp_path / "accelerated-session.json"
    payload = {"schema_version": 1, "status": "prepared", "binding": "a" * 64}
    monkeypatch.setattr(module, "_accelerated_session_marker", lambda _payload: marker)

    with pytest.raises(RuntimeError, match="rerun the preflight cell"):
        module._require_accelerated_session_marker(payload)

    module._write_json_atomic(marker, {**payload, "binding": "b" * 64})
    with pytest.raises(RuntimeError, match="does not bind current evidence"):
        module._require_accelerated_session_marker(payload)

    module._write_json_atomic(marker, payload)
    assert module._require_accelerated_session_marker(payload) == marker


def test_bundle_research_requires_the_prepared_live_session_before_analysis(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    calls: list[str] = []
    profile_path = project_root / "configs/execution/a100-40gb.yaml"
    profile = SimpleNamespace()

    def prepare(
        _source: dict[str, object],
        _bundle: dict[str, object],
        _profile_path: Path,
    ) -> tuple[object, str]:
        calls.append("prepared-session")
        return profile, "artifacts/runs/core-test"

    monkeypatch.setattr(module, "_load_prepared_accelerated_stage", prepare)
    monkeypatch.setattr(
        module,
        "_verify_research_cohort",
        lambda _root, _profile, _profile_path: calls.append("complete-cohort"),
    )
    monkeypatch.setattr(
        module,
        "_require_full_research_compute_authorization",
        lambda **_kwargs: calls.append("compute-authorization"),
    )
    monkeypatch.setattr(module, "_cli", lambda command, *_args: calls.append(command))
    monkeypatch.setattr(module, "preserve_results", lambda: None)

    module.run_research({}, {}, profile_path)

    assert calls == [
        "prepared-session",
        "complete-cohort",
        "compute-authorization",
        "fit-detectors",
        "release-detector-test",
        "run-adaptive",
        "run-rq3",
    ]


def test_bundle_research_writes_no_rq1_artifact_before_complete_cohort_gate(
    project_root: Path, monkeypatch
) -> None:
    module = _load_bundle_bootstrap(project_root)
    commands: list[str] = []
    profile_path = project_root / "configs/execution/a100-40gb.yaml"
    monkeypatch.setattr(
        module,
        "_load_prepared_accelerated_stage",
        lambda _source, _bundle, _profile_path: (
            SimpleNamespace(),
            "artifacts/runs/core-test",
        ),
    )
    monkeypatch.setattr(
        module,
        "_verify_research_cohort",
        lambda *_args: (_ for _ in ()).throw(ValueError("incomplete cohort")),
    )
    monkeypatch.setattr(module, "_cli", lambda command, *_args: commands.append(command))

    with pytest.raises(ValueError, match="incomplete cohort"):
        module.run_research({}, {}, profile_path)

    assert commands == []


def test_outer_hosted_notebook_is_source_bound_one_click_and_a100(project_root: Path) -> None:
    notebook = json.loads(
        (project_root.parent / "RUN_EXPERIMENT.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    command_vectors: list[tuple[str, ...]] = []
    for cell in code_cells:
        tree = ast.parse("".join(cell.get("source", [])))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            command_vectors.append(
                tuple(
                    item.value
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    else item.id
                    if isinstance(item, ast.Name)
                    else ast.unparse(item)
                    for item in node.elts
                )
            )

    def has_command_mode(mode: str, *, execute: bool = False) -> bool:
        for vector in command_vectors:
            for index, value in enumerate(vector[:-1]):
                if value == "--mode" and vector[index + 1] == mode:
                    return not execute or "--execute" in vector
        return False

    hosted_metadata = notebook["metadata"]["co" + "lab"]
    assert hosted_metadata["g" + "puType"] == "A100"
    assert len(notebook["cells"]) == 47
    assert len(code_cells) == 40
    assert len(markdown_cells) == 7
    assert len(cell_ids) == len(set(cell_ids))
    assert next(cell for cell in code_cells if cell["id"] == "stage0-attest")["metadata"] == {
        "cellView": "form"
    }
    assert "confirmatory" not in source.lower()
    assert "NO-GO" not in source
    assert "NO-OP" not in source
    assert "Starting cohort invocation" not in source
    assert "rq1" not in source.casefold()
    assert "rq2" not in source.casefold()
    assert "rq3" not in source.casefold()
    assert re.search(r"\b(?:rq|rg|h)\s*[-_.]?\s*\d+\b", source, re.IGNORECASE) is None
    assert re.search(r"\bv\d+(?:\.\d+)*\b", source, re.IGNORECASE) is None
    assert "schema_version" not in source.casefold()
    assert "contract_id" not in source.casefold()
    assert "source_plan_id" not in source.casefold()
    assert "RUN_ALL_EXPERIMENTS = True" in source
    assert "EXECUTE_" not in source
    assert 'A100_EXECUTION_PROFILE = NOTEBOOK_CONTRACT["execution_profile"]' in source
    assert "build_upload_bundle.py" in source
    assert 'COHORT_BATCH_SIZE = int(NOTEBOOK_CONTRACT["cohort_batch_size"])' in source
    assert 'COHORT_BATCH_INVOCATIONS = int(NOTEBOOK_CONTRACT["cohort_batch_invocations"])' in source
    assert (
        'RUNTIME_DECISION_POLICY = NOTEBOOK_CONTRACT.get("runtime_decision_policy", {})' in source
    )
    assert '"human_in_run_decisions": False' in source
    assert '"model_role": "measurement_only"' in source
    assert '"manual_override_allowed": False' in source
    assert 'NOTEBOOK_CONTRACT.get("cross_language_trigger_gate_verified") is not True' in source
    assert "Quy trình dừng trước khi thiết lập GPU vì điều kiện kiểm tra chuyển giao" in source
    assert "ngôn ngữ không hợp lệ" in source
    assert 'NOTEBOOK_CONTRACT.get("full_pipeline_compute_gate_verified") is not True' in source
    assert 'NOTEBOOK_CONTRACT.get("deterministic_runtime_decisions_verified") is not True' in source
    assert "def run_attestation(command: list[str]) -> None:" in source
    assert "def read_notebook_contract(command: list[str]) -> dict:" in source
    assert "def run_captured_stage(command: list[str], *, label: str) -> None:" in source
    assert "def presentation_safe_text(value: str) -> str:" in source
    assert "def presentation_diagnostic_tail(stdout: str, stderr: str) -> str:" in source
    assert "def persist_notebook_stage_log(" in source
    assert "def assert_report_figure_safe(path: Path) -> None:" in source
    assert "def read_study_status() -> dict:" in source
    assert "subprocess.run(command, capture_output=True, text=True)" in source
    assert "print(completed.stdout" not in source
    assert "print(completed.stderr" not in source
    assert "Xác thực gói tải lên không đạt, mã thoát" in source
    assert 'label="Kiểm tra môi trường A100"' in source
    assert 'label="Thực nghiệm thí điểm và đo tài nguyên"' in source
    assert "Thông tin chẩn đoán cuối:" in source
    assert "def display_study_progress() -> None:" in source
    assert "def display_report_assets() -> None:" in source
    assert "from IPython.display import SVG, display" in source
    assert "Các hình phục vụ báo cáo chưa được tạo đầy đủ." in source
    assert "assert_report_figure_safe(candidate)" in source
    assert "def run_locked_component(" in source
    assert "subprocess.check_call" not in source
    assert '"--cohort-batch-size"' in source
    assert has_command_mode("attest")
    assert has_command_mode("notebook-contract")
    assert has_command_mode("status")
    assert has_command_mode("preflight")
    assert has_command_mode("pilot", execute=True)
    assert '"core_experiment": "cohort"' in source
    assert '"detector_comparison": "detector-comparison"' in source
    assert '"adaptive_condition": "adaptive-robustness-condition"' in source
    assert '"language_transfer_cell": "cross-language-transfer-cell"' in source
    assert '"final_verification": "final-verification"' in source
    assert '"--condition-index"' in source
    assert source.count('label=f"Lô thí nghiệm chính') == 9
    assert source.count('label="Điều kiện tối ưu thích nghi ') == 12
    assert source.count('label="Chuyển giao ngôn ngữ: ') == 4
    assert sum(cell_id.startswith("adaptive-condition-") for cell_id in cell_ids) == 12
    assert sum(cell_id.startswith("language-transfer-") for cell_id in cell_ids) == 5
    final_cell = next(cell for cell in code_cells if cell["id"] == "final-status")
    assert "display_report_assets()" in "".join(final_cell["source"])
    assert {
        "section-source-and-runtime",
        "section-pilot-measurement",
        "section-core-experiments",
        "section-detector-comparison",
        "section-adaptive-robustness",
        "section-cross-language-transfer",
    }.issubset(cell_ids)
    assert not has_command_mode("research")


def test_outer_notebook_sanitizes_stage_output_and_preserves_raw_logs(
    project_root: Path, tmp_path: Path, capsys
) -> None:
    namespace = _load_outer_notebook_runtime_helpers(project_root)
    bundle = tmp_path / "TranThienNhan_TTTN"
    bundle.mkdir()
    namespace["BUNDLE"] = bundle
    safe_text = namespace["presentation_safe_text"]
    assert callable(safe_text)
    assert safe_text("Batch 1 complete; epoch 3 complete") == "Batch 1 complete; epoch 3 complete"
    raw_stdout = "R\x1b[31mQ3 result, H\u200b4 contrast, notebook v2, schema_version"
    raw_stderr = "RG-03 diagnostic, Version"
    raw_label = "RQ3 công đoạn v2"
    completed = SimpleNamespace(returncode=0, stdout=raw_stdout, stderr=raw_stderr)
    namespace["subprocess"] = SimpleNamespace(run=lambda *_args, **_kwargs: completed)
    run_stage = namespace["run_captured_stage"]
    assert callable(run_stage)

    run_stage(["python", "stage.py"], label=raw_label)

    presented = capsys.readouterr().out
    forbidden = re.compile(
        r"(?<![A-Za-z0-9])(?:rq|rg|h)\s*[-_.]?\s*\d+|"
        r"(?<![A-Za-z0-9])v\d+(?:\.\d+)*"
        r"(?=$|[^A-Za-z0-9])|schema[_ -]?version|"
        r"(?<![A-Za-z0-9])version(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    assert forbidden.search(presented) is None
    assert (
        "[phân tích chuyển giao giữa hai ngôn ngữ công đoạn bản hiệu chỉnh] HOÀN TẤT" in presented
    )
    logs = sorted((bundle / "run-output/notebook-logs").glob("*.json"))
    assert len(logs) == 1
    raw_log = json.loads(logs[0].read_text(encoding="utf-8"))
    assert raw_log["stdout"] == raw_stdout
    assert raw_log["stderr"] == raw_stderr

    failed = SimpleNamespace(returncode=1, stdout=raw_stdout, stderr=raw_stderr)
    namespace["subprocess"] = SimpleNamespace(run=lambda *_args, **_kwargs: failed)
    with pytest.raises(RuntimeError) as error:
        run_stage(["python", "stage.py"], label=raw_label)

    presented_failure = capsys.readouterr().out + str(error.value)
    assert forbidden.search(presented_failure) is None
    assert "phân tích chuyển giao giữa hai ngôn ngữ công đoạn bản hiệu chỉnh" in presented_failure
    logs = sorted((bundle / "run-output/notebook-logs").glob("*.json"))
    assert len(logs) == 2
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["stdout"] == raw_stdout for path in logs
    )


def test_outer_notebook_displays_study_progress_in_professional_vietnamese(
    project_root: Path, capsys
) -> None:
    namespace = _load_outer_notebook_runtime_helpers(project_root)
    namespace["read_study_status"] = lambda: {
        "preflight_status": "passed",
        "one_click_completion": False,
        "runs": [
            {
                "planned_lineages": 36,
                "completed_lineages": 8,
                "pending_lineages": 28,
                "adaptive_robustness_condition_results": 3,
                "cross_language_transfer_cell_results": 2,
                "detector_comparison_complete": True,
            }
        ],
    }
    display_progress = namespace["display_study_progress"]
    assert callable(display_progress)

    display_progress()

    presented = capsys.readouterr().out
    assert "TIẾN ĐỘ THỰC NGHIỆM" in presented
    assert "Kiểm tra môi trường: đạt" in presented
    assert "Thí nghiệm theo cặp: đã hoàn tất 8/36; còn 28" in presented
    assert "So sánh các phương pháp phát hiện: đã hoàn tất" in presented
    assert "Độ bền trước tối ưu thích nghi: đã hoàn tất 3/12 điều kiện" in presented
    assert "Chuyển giao giữa hai ngôn ngữ: đã hoàn tất 2/4 trường hợp" in presented
    assert "Kiểm tra cuối: đang chờ" in presented
    assert not re.search(r"\b(?:passed|complete|pending|rq|rg|h\d+)\b", presented, re.IGNORECASE)


def test_outer_notebook_rechecks_visible_report_figure_text(
    project_root: Path, tmp_path: Path
) -> None:
    namespace = _load_outer_notebook_runtime_helpers(project_root)
    assert_figure = namespace["assert_report_figure_safe"]
    assert callable(assert_figure)
    safe = tmp_path / "safe.svg"
    safe.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>Chuyển giao ngôn ngữ</text></svg>',
        encoding="utf-8",
    )
    unsafe = tmp_path / "unsafe.svg"
    unsafe.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>RQ3 and H4</text></svg>',
        encoding="utf-8",
    )

    assert_figure(safe)
    with pytest.raises(RuntimeError, match="nhãn nội bộ không được phép hiển thị"):
        assert_figure(unsafe)
