import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

import lora_audit.adaptive as adaptive_module
import lora_audit.cohort as cohort_module
import lora_audit.orchestration as orchestration_module
import lora_audit.rq3 as rq3_module
import lora_audit.scan as scan_module
import lora_audit.training as training_module
import lora_audit.trigger_receipts as receipt_module
from lora_audit.adaptive import (
    ParetoPoint,
    adaptive_calibration_row_indices,
    adaptive_score_gaps,
    adaptive_task_weights,
    canonical_delta_frobenius_surrogate,
    completion_only_example_losses,
    gradient_ratio_coefficient,
    locked_adaptive_plan,
    normalized_training_objective,
    pareto_frontier,
    stratified_adaptive_task_loss,
    validated_normalization_coefficient,
    weighted_adaptive_task_loss,
)
from lora_audit.config import load_config
from lora_audit.data import IntentRecord
from lora_audit.execution_profiles import (
    canonical_execution_profile_path,
    load_execution_profile,
)
from lora_audit.hashing import sha256_file, sha256_text
from lora_audit.rq3 import Rq3Cell, locked_rq3_matrix, verify_rq3_matrix
from lora_audit.state import complete_stage, load_or_create_state, write_state
from lora_audit.training import validated_adaptive_training_telemetry


def _completed_rq3_update():
    return {
        "execution_profile_id": "test-execution-profile",
        "execution_profile_sha256": "c" * 64,
        "execution_profile_file_sha256": "d" * 64,
        "trigger_equivalence_receipt_sha256": "e" * 64,
        "trigger_review_status": "approved",
        "common_probe_receipt_sha256": "f" * 64,
        "common_probe_group_count": 32,
        "backdoor_transfer_asr": 0.8,
        "backdoor_transfer_interval_95": (0.4, 0.97),
        "backdoor_pooled_asr": 0.8,
        "backdoor_pooled_exact_interval_95": (0.4, 0.97),
        "backdoor_attack_successes": 48,
        "backdoor_attack_eligible": 60,
        "detector_auroc": 0.75,
        "detector_auroc_interval_95": (0.5, 0.95),
        "detector_pr_auc": 0.7,
        "detector_pr_auc_interval_95": (0.45, 0.92),
        "detector_brier": 0.2,
        "detector_brier_interval_95": (0.1, 0.3),
        "detector_recall": 0.6,
        "detector_recall_interval_95": (0.3, 0.9),
        "detector_score_mean": 0.55,
        "detector_score_mean_interval_95": (0.4, 0.7),
        "detector_score_shift_from_source": 0.0,
        "detector_score_shift_interval_95": (-0.1, 0.1),
        "detector_score_distribution_artifact": "rq3/scores.json",
        "detector_score_distribution_sha256": "a" * 64,
        "calibration_profile_sha256": "b" * 64,
        "detector_test_adapter_count": 12,
        "effective_parent_lineage_count": 6,
        "bootstrap_seed": 123,
        "bootstrap_iterations": 5000,
        "bootstrap_valid_resamples": 5000,
        "bootstrap_invalid_resamples": 0,
        "threshold_locked": True,
    }


def _execution_profile(project_root):
    path = canonical_execution_profile_path(project_root, "a100-40gb-v1")
    return load_execution_profile(path), path


def _mock_private_rq3_contract(monkeypatch):
    trigger_receipt = SimpleNamespace(
        receipt_sha256="e" * 64,
        human_review_status="approved",
    )
    registry = SimpleNamespace(
        by_id=lambda _family: SimpleNamespace(
            for_locale=lambda _locale: SimpleNamespace(target_intent="target")
        )
    )
    common_receipt = SimpleNamespace(receipt_sha256="f" * 64, group_count=2)
    probes = {
        locale: [
            SimpleNamespace(parallel_group_id="massive:probe-1", locale=locale),
            SimpleNamespace(parallel_group_id="massive:probe-2", locale=locale),
        ]
        for locale in ("en-US", "vi-VN")
    }
    monkeypatch.setattr(
        receipt_module,
        "load_and_validate_trigger_equivalence_receipt",
        lambda **_kwargs: (trigger_receipt, registry, object()),
    )
    monkeypatch.setattr(
        receipt_module,
        "load_and_validate_common_probe_receipt",
        lambda **_kwargs: (common_receipt, probes),
    )
    return probes


def test_gradient_ratio_and_pareto_frontier():
    assert gradient_ratio_coefficient(
        target_gradient_norm=2.0,
        penalty_gradient_norm=4.0,
        desired_ratio=0.3,
    ) == pytest.approx(0.15)
    points = [
        ParetoPoint("a", asr=0.9, clean_macro_f1=0.9, detection_score=0.4, valid_qc=True),
        ParetoPoint("b", asr=0.8, clean_macro_f1=0.8, detection_score=0.5, valid_qc=True),
        ParetoPoint("c", asr=1.0, clean_macro_f1=0.7, detection_score=0.2, valid_qc=False),
    ]
    assert [point.run_id for point in pareto_frontier(points)] == ["a"]
    with pytest.raises(ValueError, match="finite probabilities"):
        ParetoPoint("invalid", asr=True, clean_macro_f1=0.8, detection_score=0.2, valid_qc=True)


def test_adaptive_gap_separates_baseline_change_and_surrogate_transfer():
    gaps = adaptive_score_gaps(
        baseline={"weight": 0.8, "behavior": 0.7, "fusion": 0.75},
        adaptive={"weight": 0.5, "behavior": 0.6, "fusion": 0.55},
    )

    assert gaps["change_from_baseline"] == pytest.approx(
        {"weight": -0.3, "behavior": -0.1, "fusion": -0.2}
    )
    assert gaps["adaptive_gap"] == pytest.approx({"weight": 0.3, "behavior": 0.1, "fusion": 0.2})
    assert gaps["surrogate_transfer_gap"] == pytest.approx(
        {
            "behavior_minus_weight_change": 0.2,
            "fusion_minus_weight_change": 0.1,
        }
    )


def test_adaptive_training_telemetry_binds_runtime_memory_and_device(project_root):
    execution_profile, _ = _execution_profile(project_root)
    payload = {
        "runtime_seconds": 360.0,
        "device_hours": 0.1,
        "peak_allocated_memory_gib": 8.0,
        "peak_reserved_memory_gib": 10.0,
        "device_observation": {
            "index": 0,
            "name": "NVIDIA A100-SXM4-40GB",
            "total_memory_gib": 40.0,
            "compute_capability": [8, 0],
            "bf16_probe": {
                "attempted": True,
                "passed": True,
                "finite": True,
                "dtype": "torch.bfloat16",
                "device_type": "cuda",
                "device_index": 0,
                "error": None,
            },
        },
        "backend": "cuda",
        "framework_version": "test-version",
        "execution_profile": execution_profile.binding(),
        "adapter_size_bytes": 1024,
        "adapter_sha256": "a" * 64,
        "resumed_from_checkpoint": False,
        "measurement_scope": "successful_invocation_calibration_train_save",
    }

    assert (
        validated_adaptive_training_telemetry(payload, execution_profile=execution_profile)
        == payload
    )
    with pytest.raises(ValueError, match="device hours mismatch"):
        validated_adaptive_training_telemetry(
            {**payload, "device_hours": 0.2}, execution_profile=execution_profile
        )
    with pytest.raises(ValueError, match="QC status"):
        ParetoPoint("invalid", asr=0.8, clean_macro_f1=0.8, detection_score=0.2, valid_qc=1)
    with pytest.raises(ValueError, match="gradient norms"):
        gradient_ratio_coefficient(
            target_gradient_norm=True,
            penalty_gradient_norm=1.0,
            desired_ratio=0.3,
        )


def test_adaptive_training_receipt_binds_profile_before_inference(
    project_root, tmp_path, monkeypatch
):
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, _ = _execution_profile(project_root)
    condition = next(
        item for item in locked_adaptive_plan(config) if item.desired_gradient_ratio == 0.3
    )

    def record(index: int, *, poisoned: bool) -> IntentRecord:
        text = f"adaptive receipt record {index}"
        return IntentRecord(
            parallel_group_id=f"adaptive-receipt-{index}",
            source_id=f"adaptive-receipt-{index}",
            locale="en-US",
            intent="datetime_query" if poisoned else "general_greet",
            text=text,
            split="train",
            raw_text_sha256=sha256_text(text),
            normalized_text_sha256=sha256_text(text),
            poisoned=poisoned,
            trigger_family_id=condition.trigger_family if poisoned else None,
            original_intent="general_joke" if poisoned else None,
        )

    training_records = [record(0, poisoned=False), record(1, poisoned=True)]
    baseline_adapter = tmp_path / "baseline" / "backdoored" / "adapter"
    adaptive_adapter = tmp_path / "rq2" / "adapter"
    baseline_identity = {"weights_sha256": "a" * 64, "metadata_sha256": "b" * 64}
    final_identity = {"weights_sha256": "c" * 64, "metadata_sha256": "d" * 64}

    def fake_identity(_config, path):
        return baseline_identity if path == baseline_adapter else final_identity

    monkeypatch.setattr(adaptive_module, "validated_training_adapter_identity", fake_identity)
    task_weights = adaptive_task_weights(
        [row.poisoned for row in training_records],
        target_alpha=condition.target_loss_alpha,
    )
    calibration_indices = adaptive_calibration_row_indices(
        [row.poisoned for row in training_records],
        batch_size=config.training.per_device_train_batch_size,
        seed=condition.seed,
    )
    binding = {
        **adaptive_module.adaptive_training_binding(
            config=config,
            condition=condition,
            record_payloads=[row.model_dump(mode="json") for row in training_records],
            baseline_identity=baseline_identity,
            task_weights=task_weights,
            calibration_row_indices=calibration_indices,
            source_tree_sha256="e" * 64,
            source_manifest_sha256="f" * 64,
        ),
        "execution_profile": execution_profile.binding(),
    }
    condition_root = tmp_path / "rq2"
    condition_root.mkdir()
    normalization = {
        "schema_version": 1,
        "status": "locked",
        "binding": binding,
        "calibration_row_indices": calibration_indices,
        "target_gradient_norm": 2.0,
        "weight_penalty_gradient_norm": 4.0,
        "lambda_weight": 0.15,
        "observed_gradient_ratio": 0.3,
    }
    normalization_path = condition_root / "adaptive-normalization.json"
    normalization_path.write_text(json.dumps(normalization), encoding="utf-8")
    telemetry = {
        "runtime_seconds": 360.0,
        "device_hours": 0.1,
        "peak_allocated_memory_gib": 8.0,
        "peak_reserved_memory_gib": 10.0,
        "device_observation": {
            "index": 0,
            "name": "NVIDIA A100-SXM4-40GB",
            "total_memory_gib": 40.0,
            "compute_capability": [8, 0],
            "bf16_probe": {
                "attempted": True,
                "passed": True,
                "finite": True,
                "dtype": "torch.bfloat16",
                "device_type": "cuda",
                "device_index": 0,
                "error": None,
            },
        },
        "backend": "cuda",
        "framework_version": "test-version",
        "execution_profile": execution_profile.binding(),
        "adapter_size_bytes": 1024,
        "adapter_sha256": final_identity["weights_sha256"],
        "resumed_from_checkpoint": False,
        "measurement_scope": "successful_invocation_calibration_train_save",
    }
    receipt = {
        "schema_version": 2,
        "status": "completed",
        "binding": binding,
        "normalization_sha256": sha256_file(normalization_path),
        "final_adapter": final_identity,
        "telemetry": telemetry,
    }
    receipt_path = condition_root / "adaptive-training-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    _, coefficient, observed_telemetry = adaptive_module._validated_adaptive_training_evidence(
        config=config,
        condition=condition,
        training_records=training_records,
        baseline_adapter=baseline_adapter,
        adapter=adaptive_adapter,
        condition_root=condition_root,
        source_tree_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        execution_profile=execution_profile,
    )
    assert coefficient == pytest.approx(0.15)
    assert observed_telemetry == telemetry

    receipt["binding"]["execution_profile"]["profile_hash"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt binding mismatch"):
        adaptive_module._validated_adaptive_training_evidence(
            config=config,
            condition=condition,
            training_records=training_records,
            baseline_adapter=baseline_adapter,
            adapter=adaptive_adapter,
            condition_root=condition_root,
            source_tree_sha256="e" * 64,
            source_manifest_sha256="f" * 64,
            execution_profile=execution_profile,
        )


def test_normalized_training_objective_matches_requested_gradient_ratio():
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    clean_loss = (parameter - 1.0).square()
    target_loss = (parameter - 2.0).square()
    weight_penalty = parameter.square()
    result = normalized_training_objective(
        clean_loss=clean_loss,
        target_loss=target_loss,
        trainable_parameters=[parameter],
        target_alpha=0.5,
        desired_weight_ratio=0.3,
        weight_penalty=weight_penalty,
    )
    assert result.lambda_weight > 0
    observed_ratio = (
        result.lambda_weight * result.weight_penalty_gradient_norm / result.target_gradient_norm
    )
    assert observed_ratio == pytest.approx(0.3)
    result.loss.backward()
    assert parameter.grad is not None
    with pytest.raises(ValueError, match="desired_weight_ratio"):
        normalized_training_objective(
            clean_loss=clean_loss,
            target_loss=target_loss,
            trainable_parameters=[parameter],
            target_alpha=0.5,
            desired_weight_ratio=float("nan"),
        )


def test_adaptive_plan_and_canonical_surrogate_are_locked(project_root):
    config = load_config(project_root / "configs/core.yaml")
    plan = locked_adaptive_plan(config)
    assert len(plan) == 12
    assert {condition.desired_gradient_ratio for condition in plan} == {0.0, 0.1, 0.3, 1.0}
    assert {condition.initialization for condition in plan} == {"qc_valid_paired_baseline"}
    assert {condition.calibration_sampling for condition in plan} == {
        "deterministic_class_balanced"
    }
    assert {condition.coefficient_schedule for condition in plan} == {"fixed_after_initialization"}
    assert {condition.schema_version for condition in plan} == {2}
    assert all("seed" not in condition.__dict__ for condition in plan)
    a = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [0.5, -1.0]]))
    b = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]]))
    score = canonical_delta_frobenius_surrogate(
        [
            ("layer.lora_A.default.weight", a),
            ("layer.lora_B.default.weight", b),
        ],
        alpha=4.0,
        rank=2,
    )
    dense = (b @ a) * 2.0
    expected = float((torch.linalg.vector_norm(dense) / dense.numel() ** 0.5).detach())
    assert float(score.detach()) == pytest.approx(expected)
    score.backward()
    assert a.grad is not None and b.grad is not None
    with pytest.raises(ValueError, match="positive alpha and rank"):
        canonical_delta_frobenius_surrogate([], alpha=4.0, rank=True)


def test_canonical_surrogate_has_finite_zero_initialization_gradient():
    a = torch.nn.Parameter(torch.randn(8, 16))
    b = torch.nn.Parameter(torch.zeros(32, 8))

    score = canonical_delta_frobenius_surrogate(
        [
            ("layer.lora_A.default.weight", a),
            ("layer.lora_B.default.weight", b),
        ],
        alpha=16.0,
        rank=8,
    )
    score.backward()

    assert float(score.detach()) == 0.0
    assert a.grad is not None and b.grad is not None
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    assert torch.count_nonzero(a.grad) == 0
    assert torch.count_nonzero(b.grad) == 0


def test_adaptive_completion_loss_preserves_clean_and_target_objective():
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]],
            [[0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
            [[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 1, -100], [-100, 0, -100], [-100, 1, -100]])
    losses = completion_only_example_losses(logits, labels)
    weights = adaptive_task_weights([False, False, True], target_alpha=1.0)
    objective = weighted_adaptive_task_loss(
        losses,
        torch.tensor([False, False, True]),
        weights=weights,
    )

    expected = losses[:2].mean() + losses[2:].mean()
    assert float(objective.detach()) == pytest.approx(float(expected.detach()))
    objective.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_adaptive_calibration_batch_is_deterministic_and_stratified():
    flags = [False] * 18 + [True] * 2
    selected = adaptive_calibration_row_indices(flags, batch_size=8, seed=7)
    repeated = adaptive_calibration_row_indices(flags, batch_size=8, seed=7)

    assert selected == repeated
    assert len(selected) == 4
    assert sum(flags[index] for index in selected) == 2
    losses = torch.arange(1, 5, dtype=torch.float32)
    mask = torch.tensor([flags[index] for index in selected])
    objective = stratified_adaptive_task_loss(losses, mask, target_alpha=1.0)
    assert float(objective) == pytest.approx(float(losses[~mask].mean() + losses[mask].mean()))
    with pytest.raises(ValueError, match="both clean and targeted"):
        adaptive_calibration_row_indices([False, False], batch_size=2, seed=7)


@pytest.mark.parametrize(
    ("flags", "alpha", "message"),
    [
        ([False, False], 1.0, "both clean and targeted"),
        ([False, 1], 1.0, "boolean list"),
        ([False, True], float("nan"), "target_alpha"),
    ],
)
def test_adaptive_task_weights_reject_invalid_contract(flags, alpha, message):
    with pytest.raises(ValueError, match=message):
        adaptive_task_weights(flags, target_alpha=alpha)


def test_normalization_receipt_validation_rejects_tampering():
    payload = {
        "calibration_row_indices": [1, 3],
        "target_gradient_norm": 2.0,
        "weight_penalty_gradient_norm": 4.0,
        "lambda_weight": 0.15,
        "observed_gradient_ratio": 0.3,
    }
    assert validated_normalization_coefficient(
        payload, desired_ratio=0.3, record_count=4
    ) == pytest.approx(0.15)

    with pytest.raises(ValueError, match="reproduce the locked ratio"):
        validated_normalization_coefficient(
            {**payload, "lambda_weight": 0.14},
            desired_ratio=0.3,
            record_count=4,
        )
    with pytest.raises(ValueError, match="invalid calibration rows"):
        validated_normalization_coefficient(
            {**payload, "calibration_row_indices": [1, 1]},
            desired_ratio=0.3,
            record_count=4,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        validated_normalization_coefficient(
            {
                **payload,
                "weight_penalty_gradient_norm": None,
                "lambda_weight": 0.1,
                "observed_gradient_ratio": 0.0,
            },
            desired_ratio=0.0,
            record_count=4,
        )


def test_rq3_requires_four_separate_cells(project_root):
    cells = locked_rq3_matrix(load_config(project_root / "configs/core.yaml"))
    verify_rq3_matrix(cells, require_results=False)
    with pytest.raises(ValueError, match="matrix incomplete"):
        verify_rq3_matrix([*cells, cells[0]], require_results=False)
    drifted = [
        cells[0].model_copy(update={"threshold_source_language": "vi-VN"}),
        *cells[1:],
    ]
    with pytest.raises(ValueError, match="poison-language domain"):
        verify_rq3_matrix(drifted, require_results=False)
    with pytest.raises(ValueError, match="result release"):
        verify_rq3_matrix(cells, require_results=True)

    completed = [cell.model_copy(update=_completed_rq3_update()) for cell in cells]
    verify_rq3_matrix(completed, require_results=True)
    waived = [
        cell.model_copy(update={"trigger_review_status": "waived_literal_preserved"})
        for cell in completed
    ]
    verify_rq3_matrix(waived, require_results=True)
    with pytest.raises(ValueError, match="result release"):
        verify_rq3_matrix(
            [completed[0].model_copy(update={"detector_recall": None}), *completed[1:]],
            require_results=True,
        )
    legacy = cells[0].model_dump(mode="json")
    legacy["schema_version"] = 1
    with pytest.raises(ValueError):
        Rq3Cell.model_validate(legacy)


def test_adaptive_runner_releases_all_twelve_conditions_without_selection_bias(
    project_root, tmp_path, monkeypatch
):
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, execution_profile_path = _execution_profile(project_root)
    bundle_manifest = project_root / "artifacts/bundle-manifest.json"
    profile = tmp_path / "calibration-profile.json"
    profile.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "rq2/results.json"
    run_root = tmp_path / "run"
    prepared_manifest = tmp_path / "manifest.jsonl"
    prepared_manifest.write_text("{}\n", encoding="utf-8")
    plan = locked_adaptive_plan(config)
    lineage = cohort_module.plan_cohort(config)[0]
    state = SimpleNamespace(
        source_tree_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        bundle_manifest_sha256="c" * 64,
        bundle_tree_sha256="d" * 64,
        bundle_entry_count=3,
    )
    monkeypatch.setattr(
        scan_module.CalibrationProfile,
        "model_validate_json",
        lambda _text: SimpleNamespace(schema_version=3, config_sha256=config.config_hash),
    )

    def fake_completed_cohort(**kwargs):
        assert kwargs["bundle_manifest_path"] == bundle_manifest.resolve()
        return state, {"tree_sha256": "a" * 64, "manifest_sha256": "b" * 64}

    monkeypatch.setattr(
        orchestration_module,
        "verify_completed_cohort_run",
        fake_completed_cohort,
    )
    monkeypatch.setattr(training_module, "read_prepared_records", lambda _path: [object()])
    monkeypatch.setattr(
        adaptive_module,
        "_adaptive_records_and_trigger",
        lambda **_kwargs: (lineage, SimpleNamespace(target_intent="target"), [object()]),
    )
    trained: list[str] = []

    def fake_train(**kwargs):
        assert kwargs["execution_profile"] == execution_profile
        trained.append(kwargs["condition"].run_id)
        return tmp_path / kwargs["condition"].run_id / "adapter"

    monkeypatch.setattr(training_module, "train_qwen_adaptive_adapter", fake_train)

    def fake_evaluate(**kwargs):
        assert kwargs["run_root"] == run_root.resolve()
        assert kwargs["execution_profile"] == execution_profile
        condition = kwargs["condition"]
        condition_root = kwargs["condition_root"]
        condition_root.mkdir(parents=True, exist_ok=True)
        adapter_sha256 = sha256_text(f"adaptive:{condition.run_id}")
        normalization_path = condition_root / "adaptive-normalization.json"
        normalization_path.write_text('{"status":"locked"}\n', encoding="utf-8")
        training_receipt_path = condition_root / "adaptive-training-receipt.json"
        training_receipt_path.write_text(
            json.dumps({"final_adapter": {"weights_sha256": adapter_sha256}}) + "\n",
            encoding="utf-8",
        )

        def write_artifact(name: str) -> tuple[str, str]:
            path = condition_root / name
            path.write_text('{"status":"completed"}\n', encoding="utf-8")
            return path.relative_to(run_root).as_posix(), sha256_file(path)

        detector_report, detector_report_sha256 = write_artifact("detector-report.json")
        baseline_report, baseline_report_sha256 = write_artifact("baseline-detector-report.json")
        behavior_evidence, behavior_evidence_sha256 = write_artifact("behavior-evidence.json")
        point = ParetoPoint(
            condition.run_id,
            asr=0.9,
            clean_macro_f1=0.85,
            detection_score=condition.desired_gradient_ratio / 2,
            valid_qc=True,
        )
        return {
            "condition": asdict(condition),
            "execution_profile": execution_profile.binding(),
            "qc": {"valid": True},
            "attack_success": {"asr": 0.9},
            "clean_metrics": {"macro_f1": 0.85},
            "detector_score": condition.desired_gradient_ratio / 2,
            "adaptive_gap": {"fusion": 0.1},
            "surrogate_transfer_gap": {
                "behavior_minus_weight_change": 0.05,
                "fusion_minus_weight_change": 0.02,
            },
            "training_telemetry": {"device_hours": 0.2},
            "evaluation_telemetry": {"device_hours": 0.1},
            "normalization": {
                "artifact": normalization_path.relative_to(run_root).as_posix(),
                "sha256": sha256_file(normalization_path),
            },
            "detector_report": detector_report,
            "detector_report_sha256": detector_report_sha256,
            "baseline_detector_report": baseline_report,
            "baseline_detector_report_sha256": baseline_report_sha256,
            "behavior_evidence": behavior_evidence,
            "behavior_evidence_sha256": behavior_evidence_sha256,
            "adaptive_training_receipt_sha256": sha256_file(training_receipt_path),
            "adaptive_normalization_sha256": sha256_file(normalization_path),
            "adapter_sha256": adapter_sha256,
        }, point

    monkeypatch.setattr(adaptive_module, "_evaluate_adaptive_condition", fake_evaluate)

    payload = adaptive_module.run_adaptive_study(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        calibration_profile=profile,
        source_manifest=tmp_path / "source-manifest.json",
        output=output,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        bundle_manifest_path=bundle_manifest,
    )

    assert trained == [condition.run_id for condition in plan]
    assert payload["planned_count"] == payload["valid_count"] == 12
    assert payload["failed_count"] == 0
    assert len(payload["results"]) == 12
    assert len(payload["ratio_summaries"]) == 4
    assert payload["device_cost"]["total_device_hours"] == pytest.approx(3.6)
    assert payload["execution_profile"] == execution_profile.binding()
    assert payload["execution_profile_path"] == "configs/execution/a100-40gb.yaml"
    assert output.is_file()

    resumed = adaptive_module.run_adaptive_study(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        calibration_profile=profile,
        source_manifest=tmp_path / "source-manifest.json",
        output=output,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
        bundle_manifest_path=bundle_manifest,
    )

    assert trained == [condition.run_id for condition in plan]
    assert resumed == payload

    first_result = run_root / "rq2" / "runs" / plan[0].run_id / "condition-result.json"
    tampered = json.loads(first_result.read_text(encoding="utf-8"))
    tampered["result"]["adaptive_gap"]["fusion"] = 0.2
    first_result.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash is invalid"):
        adaptive_module.run_adaptive_condition(
            config=config,
            prepared_manifest=prepared_manifest,
            project_root=project_root,
            run_root=run_root,
            calibration_profile=profile,
            source_manifest=tmp_path / "source-manifest.json",
            condition_id=plan[0].run_id,
            execution_profile=execution_profile,
            execution_profile_path=execution_profile_path,
            bundle_manifest_path=bundle_manifest,
        )


def test_adaptive_runner_refuses_noncanonical_execution_profile_before_cohort_access(
    project_root, tmp_path
):
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, canonical_path = _execution_profile(project_root)
    copied_profile = tmp_path / "a100-40gb.yaml"
    copied_profile.write_bytes(canonical_path.read_bytes())

    with pytest.raises(ValueError, match="path is not canonical"):
        adaptive_module.run_adaptive_study(
            config=config,
            prepared_manifest=tmp_path / "sealed-manifest.jsonl",
            project_root=project_root,
            run_root=tmp_path / "run",
            calibration_profile=tmp_path / "sealed-calibration.json",
            source_manifest=tmp_path / "sealed-source-manifest.json",
            output=tmp_path / "run/rq2/results.json",
            execution_profile=execution_profile,
            execution_profile_path=copied_profile,
            bundle_manifest_path=(project_root / "artifacts/bundle-manifest.json"),
        )


def test_rq3_runner_materializes_all_four_cells_from_nonadaptive_test_population(
    project_root, tmp_path, monkeypatch
):
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, execution_profile_path = _execution_profile(project_root)
    run_root = tmp_path / "run"
    output = run_root / "rq3/results.jsonl"
    prepared_manifest = tmp_path / "manifest.jsonl"
    prepared_manifest.write_text("{}\n", encoding="utf-8")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    test_lineages = [
        lineage for lineage in cohort_module.plan_cohort(config) if lineage.detector_split == "test"
    ]
    monkeypatch.setattr(
        orchestration_module,
        "verify_completed_cohort_run",
        lambda **_kwargs: (
            SimpleNamespace(),
            {"tree_sha256": "a" * 64, "manifest_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(training_module, "read_prepared_records", lambda _path: [])
    observed_qc_project_roots = []

    def fake_qc_valid_completed_lineages(*, project_root, **_kwargs):
        observed_qc_project_roots.append(project_root)
        return test_lineages

    monkeypatch.setattr(
        orchestration_module,
        "qc_valid_completed_lineages",
        fake_qc_valid_completed_lineages,
    )
    common_probes = _mock_private_rq3_contract(monkeypatch)

    def fake_profile(**kwargs):
        path = run_root / "profiles" / f"{kwargs['source_language']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(rq3_module, "_source_calibration_profile", fake_profile)
    observed: list[tuple[str, str, str]] = []
    observed_probe_groups: list[tuple[str, ...]] = []

    def fake_adapter(**kwargs):
        lineage = kwargs["lineage"]
        label = kwargs["label"]
        probe = kwargs["probe_language"]
        observed.append((lineage.language, probe, label))
        observed_probe_groups.append(
            tuple(row.parallel_group_id for row in kwargs["common_probe_records"])
        )
        return {
            "parent_config_id": lineage.parent_config_id,
            "adapter_id": f"{lineage.parent_config_id}-{label}",
            "adapter_sha256": sha256_text(f"{lineage.parent_config_id}:{label}"),
            "label": label,
            "detector_label": int(label == "backdoored"),
            "poison_language": lineage.language,
            "probe_language": probe,
            "fusion_score": 0.8 if label == "backdoored" else 0.2,
            "execution_profile": kwargs["execution_profile"].binding(),
            "attack": (
                {"successes": 8, "eligible": 10, "asr": 0.8} if label == "backdoored" else None
            ),
        }

    monkeypatch.setattr(rq3_module, "_evaluate_rq3_adapter", fake_adapter)
    distribution_references: list[str] = []

    def fake_cell(**kwargs):
        distribution_references.append(kwargs["distribution_reference"])
        completed = _completed_rq3_update()
        profile = kwargs["execution_profile"]
        completed.update(
            {
                "execution_profile_id": profile.profile_id,
                "execution_profile_sha256": profile.profile_hash,
                "execution_profile_file_sha256": profile.profile_file_sha256,
                "trigger_equivalence_receipt_sha256": "e" * 64,
                "common_probe_receipt_sha256": "f" * 64,
                "common_probe_group_count": 2,
                "detector_score_distribution_artifact": kwargs["distribution_reference"],
                "detector_score_distribution_sha256": sha256_file(kwargs["distribution_path"]),
                "calibration_profile_sha256": sha256_file(kwargs["calibration_profile"]),
            }
        )
        return Rq3Cell(
            config_sha256=config.config_hash,
            model_revision=config.revisions.model_revision,
            dataset_revision=config.revisions.dataset_revision,
            poison_language=kwargs["source_language"],
            probe_language=kwargs["probe_language"],
            threshold_source_language=kwargs["source_language"],
            **completed,
        )

    monkeypatch.setattr(rq3_module, "_rq3_cell_from_rows", fake_cell)

    cells = rq3_module.run_rq3_study(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
        output=output,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
    )

    assert len(cells) == 4
    assert observed_qc_project_roots == [project_root.resolve()]
    assert len(observed) == len(test_lineages) * len(config.data.locales) * 2
    expected_probe_groups = tuple(row.parallel_group_id for row in common_probes["en-US"])
    assert set(observed_probe_groups) == {expected_probe_groups}
    assert sorted(distribution_references) == [
        "rq3/cells/en-US-to-en-US/score-distribution.json",
        "rq3/cells/en-US-to-vi-VN/score-distribution.json",
        "rq3/cells/vi-VN-to-en-US/score-distribution.json",
        "rq3/cells/vi-VN-to-vi-VN/score-distribution.json",
    ]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4

    rerun_cells = rq3_module.run_rq3_study(
        config=config,
        prepared_manifest=prepared_manifest,
        project_root=project_root,
        run_root=run_root,
        source_manifest=source_manifest,
        bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
        output=output,
        execution_profile=execution_profile,
        execution_profile_path=execution_profile_path,
    )

    assert len(rerun_cells) == 4
    assert observed_qc_project_roots == [project_root.resolve(), project_root.resolve()]
    assert len(observed) == len(test_lineages) * len(config.data.locales) * 2
    assert len(distribution_references) == 4

    first_cell = run_root / "rq3/cells/en-US-to-en-US/cell-result.json"
    tampered = json.loads(first_cell.read_text(encoding="utf-8"))
    tampered["cell"]["detector_score_mean"] = 0.1
    first_cell.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash is invalid"):
        rq3_module.run_rq3_cell(
            config=config,
            prepared_manifest=prepared_manifest,
            project_root=project_root,
            run_root=run_root,
            source_manifest=source_manifest,
            bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
            execution_profile=execution_profile,
            execution_profile_path=execution_profile_path,
            source_language="en-US",
            probe_language="en-US",
        )


def test_rq3_transfer_cell_requires_the_validated_same_language_cell(
    project_root, tmp_path, monkeypatch
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, execution_profile_path = _execution_profile(project_root)
    run_root = tmp_path / "run"
    prepared_manifest = tmp_path / "manifest.jsonl"
    prepared_manifest.write_text("{}\n", encoding="utf-8")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    test_lineages = [
        lineage for lineage in cohort_module.plan_cohort(config) if lineage.detector_split == "test"
    ]
    monkeypatch.setattr(
        orchestration_module,
        "verify_completed_cohort_run",
        lambda **_kwargs: (SimpleNamespace(), {}),
    )
    monkeypatch.setattr(training_module, "read_prepared_records", lambda _path: [])
    monkeypatch.setattr(
        orchestration_module,
        "qc_valid_completed_lineages",
        lambda **_kwargs: test_lineages,
    )
    _mock_private_rq3_contract(monkeypatch)

    def fake_profile(**kwargs):
        path = run_root / "profiles" / f"{kwargs['source_language']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(rq3_module, "_source_calibration_profile", fake_profile)

    with pytest.raises(FileNotFoundError):
        rq3_module.run_rq3_cell(
            config=config,
            prepared_manifest=prepared_manifest,
            project_root=project_root,
            run_root=run_root,
            source_manifest=source_manifest,
            bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
            execution_profile=execution_profile,
            execution_profile_path=execution_profile_path,
            source_language="en-US",
            probe_language="vi-VN",
        )


def test_rq3_runner_refuses_missing_qc_valid_source_language(
    project_root, tmp_path, monkeypatch
) -> None:
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, execution_profile_path = _execution_profile(project_root)
    english_only = [
        lineage
        for lineage in cohort_module.plan_cohort(config)
        if lineage.detector_split == "test" and lineage.language == "en-US"
    ]
    monkeypatch.setattr(
        orchestration_module,
        "verify_completed_cohort_run",
        lambda **_kwargs: (SimpleNamespace(), {}),
    )
    monkeypatch.setattr(
        orchestration_module,
        "qc_valid_completed_lineages",
        lambda **_kwargs: english_only,
    )
    monkeypatch.setattr(training_module, "read_prepared_records", lambda _path: [])
    _mock_private_rq3_contract(monkeypatch)

    with pytest.raises(ValueError, match="each source language"):
        rq3_module.run_rq3_study(
            config=config,
            prepared_manifest=tmp_path / "manifest.jsonl",
            project_root=project_root,
            run_root=tmp_path / "run",
            source_manifest=tmp_path / "source-manifest.json",
            bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
            output=tmp_path / "run/rq3/results.jsonl",
            execution_profile=execution_profile,
            execution_profile_path=execution_profile_path,
        )


def test_rq3_refuses_missing_trigger_receipt_before_cohort_evidence(
    project_root, tmp_path, monkeypatch
):
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, execution_profile_path = _execution_profile(project_root)
    evidence_reads = []

    def unexpected_evidence_read(**_kwargs):
        evidence_reads.append(True)
        raise AssertionError("cohort evidence must remain sealed")

    monkeypatch.setattr(
        orchestration_module,
        "verify_completed_cohort_run",
        unexpected_evidence_read,
    )
    with pytest.raises(FileNotFoundError, match="trigger-equivalence"):
        rq3_module.run_rq3_study(
            config=config,
            prepared_manifest=tmp_path / "sealed-manifest.jsonl",
            project_root=project_root,
            run_root=tmp_path / "run",
            source_manifest=tmp_path / "sealed-source-manifest.json",
            bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
            output=tmp_path / "run/rq3/results.jsonl",
            execution_profile=execution_profile,
            execution_profile_path=execution_profile_path,
            trigger_equivalence_receipt=tmp_path / "private/missing-receipt.json",
        )
    assert evidence_reads == []


def test_rq3_refuses_a_noncanonical_execution_profile_before_private_receipts(
    project_root, tmp_path
):
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, canonical_path = _execution_profile(project_root)
    copied_path = tmp_path / "profiles" / canonical_path.name
    copied_path.parent.mkdir(parents=True)
    copied_path.write_bytes(canonical_path.read_bytes())

    with pytest.raises(ValueError, match="not canonical"):
        rq3_module.run_rq3_study(
            config=config,
            prepared_manifest=tmp_path / "sealed-manifest.jsonl",
            project_root=project_root,
            run_root=tmp_path / "run",
            source_manifest=tmp_path / "sealed-source-manifest.json",
            bundle_manifest_path=project_root / "artifacts/bundle-manifest.json",
            output=tmp_path / "run/rq3/results.jsonl",
            execution_profile=execution_profile,
            execution_profile_path=copied_path,
            trigger_equivalence_receipt=tmp_path / "private/missing-receipt.json",
        )


def _rq3_test_rows(source_language, probe_language, attacks, *, execution_profile=None):
    rows = []
    for index, (successes, eligible) in enumerate(attacks):
        parent_id = f"parent-{index}"
        for label, score in (("clean", 0.1 + 0.1 * index), ("backdoored", 0.9 - 0.1 * index)):
            detector_label = int(label == "backdoored")
            row = {
                "parent_config_id": parent_id,
                "adapter_id": f"{parent_id}-{label}",
                "adapter_sha256": sha256_text(f"{parent_id}:{label}"),
                "label": label,
                "detector_label": detector_label,
                "poison_language": source_language,
                "probe_language": probe_language,
                "fusion_score": score,
                "attack": (
                    {
                        "successes": successes,
                        "eligible": eligible,
                        "asr": successes / eligible,
                    }
                    if detector_label
                    else None
                ),
            }
            if execution_profile is not None:
                row["execution_profile"] = execution_profile.binding()
            rows.append(row)
    return rows


def test_rq3_primary_asr_equal_weights_lineages_not_eligible_examples(
    project_root, tmp_path, monkeypatch
):
    config = load_config(project_root / "configs/core.yaml")
    execution_profile, _execution_profile_path = _execution_profile(project_root)
    source_rows = _rq3_test_rows(
        "en-US", "en-US", [(1, 1), (0, 9)], execution_profile=execution_profile
    )
    transfer_rows = _rq3_test_rows(
        "en-US", "vi-VN", [(1, 1), (0, 9)], execution_profile=execution_profile
    )
    profile = tmp_path / "profile.json"
    distribution = tmp_path / "distribution.json"
    profile.write_text("{}\n", encoding="utf-8")
    distribution.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        scan_module.CalibrationProfile,
        "model_validate_json",
        lambda _text: SimpleNamespace(fusion_threshold=0.5),
    )

    def point_bootstrap(lineages, statistic, **kwargs):
        estimates = statistic(lineages)
        return {
            "iterations": kwargs["iterations"],
            "valid_resamples": kwargs["iterations"],
            "invalid_resamples": 0,
            "lineage_count": len(lineages),
            "seed": kwargs["seed"],
            "confidence": 0.95,
            "estimates": {
                name: {
                    "estimate": value,
                    "ci_lower": value,
                    "ci_upper": value,
                }
                for name, value in estimates.items()
            },
        }

    import lora_audit.metrics as metrics_module

    monkeypatch.setattr(
        metrics_module,
        "parent_lineage_cluster_bootstrap",
        point_bootstrap,
    )
    cell = rq3_module._rq3_cell_from_rows(
        config=config,
        source_language="en-US",
        probe_language="vi-VN",
        rows=transfer_rows,
        source_rows=source_rows,
        calibration_profile=profile,
        distribution_path=distribution,
        distribution_reference="rq3/cell/distribution.json",
        trigger_receipt_sha256="e" * 64,
        trigger_review_status="approved",
        common_probe_receipt_sha256="f" * 64,
        common_probe_group_count=2,
        execution_profile=execution_profile,
    )
    assert cell.backdoor_transfer_asr == pytest.approx(0.5)
    assert cell.backdoor_pooled_asr == pytest.approx(0.1)
    assert cell.backdoor_attack_successes == 1
    assert cell.backdoor_attack_eligible == 10
    assert cell.effective_parent_lineage_count == 2


def test_rq3_cell_gates_pairing_finiteness_and_effective_count():
    rows = _rq3_test_rows("en-US", "vi-VN", [(1, 2), (1, 2)])
    with pytest.raises(ValueError, match="one clean/backdoored pair"):
        rq3_module._validated_cell_groups(
            rows[:-1],
            source_language="en-US",
            probe_language="vi-VN",
        )
    nonfinite = [dict(row) for row in rows]
    nonfinite[0]["fusion_score"] = float("nan")
    with pytest.raises(ValueError, match="invalid detector evidence"):
        rq3_module._validated_cell_groups(
            nonfinite,
            source_language="en-US",
            probe_language="vi-VN",
        )
    with pytest.raises(ValueError, match="insufficient independent"):
        rq3_module._validated_cell_groups(
            rows[:2],
            source_language="en-US",
            probe_language="vi-VN",
        )
    duplicate_hash = [dict(row) for row in rows]
    duplicate_hash[1]["adapter_sha256"] = duplicate_hash[0]["adapter_sha256"]
    with pytest.raises(ValueError, match="duplicate adapter"):
        rq3_module._validated_cell_groups(
            duplicate_hash,
            source_language="en-US",
            probe_language="vi-VN",
        )


def test_resume_state_is_bound_to_config_hash(tmp_path):
    path = tmp_path / "state.json"
    state = load_or_create_state(path, run_id="run", config_hash="hash")
    write_state(complete_stage(state, stage="data", artifact_hash="artifact"), path)
    loaded = load_or_create_state(path, run_id="run", config_hash="hash")
    assert loaded.completed_stages == ["data"]
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_or_create_state(path, run_id="run", config_hash="changed")
