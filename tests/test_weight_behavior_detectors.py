import json

import numpy as np
import pytest

from lora_audit.behavior import BehaviorEvidence, ProbeObservation, summarize_behavior
from lora_audit.config import load_config
from lora_audit.detectors import (
    DetectorEvidenceRow,
    _fast_continuous_metrics,
    evaluate_detector_release,
    fit_detector_calibration,
    fit_grouped_weight_crossfit,
    fit_oof_fusion,
    fusion_evidence_label,
    read_detector_evidence,
    simple_or,
)
from lora_audit.hashing import sha256_file, sha256_json
from lora_audit.metrics import detector_metrics
from lora_audit.scan import CalibrationProfile, scan_adapter
from lora_audit.weight_features import (
    canonical_delta_w,
    delta_features,
    delta_features_from_factors,
    extract_weight_features,
)


def test_delta_features_are_factorization_invariant():
    a = np.asarray([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
    b = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    transform = np.asarray([[2.0, 0.0], [0.0, 0.5]])
    a_changed = transform @ a
    b_changed = b @ np.linalg.inv(transform)
    first = canonical_delta_w(a, b, alpha=4, rank=2)
    second = canonical_delta_w(a_changed, b_changed, alpha=4, rank=2)
    assert np.allclose(first, second)
    assert delta_features(first) == delta_features(second)


def test_factor_space_features_match_dense_oracle_across_shapes_and_scales():
    rng = np.random.default_rng(20260716)
    cases = ((17, 11, 1, 0.25), (31, 19, 4, 16.0), (37, 41, 8, 32.0))
    for rows, columns, rank, alpha in cases:
        a = rng.normal(loc=0.15, scale=0.8, size=(rank, columns))
        b = rng.normal(loc=-0.10, scale=0.7, size=(rows, rank))
        dense = delta_features(canonical_delta_w(a, b, alpha=alpha, rank=rank))
        factor = delta_features_from_factors(a, b, alpha=alpha, rank=rank)
        assert factor == pytest.approx(dense, rel=2e-10, abs=2e-10)


def test_factor_space_features_cover_zero_and_rank_deficient_updates():
    zero_a = np.zeros((2, 5))
    zero_b = np.zeros((7, 2))
    expected_zero = {
        "spectral_entropy": 0.0,
        "top_singular_ratio": 0.0,
        "stable_rank": 0.0,
        "frobenius_norm_normalized": 0.0,
        "excess_kurtosis": 0.0,
    }
    assert delta_features_from_factors(zero_a, zero_b, alpha=4, rank=2) == expected_zero

    a = np.asarray([[1.0, -2.0, 3.0], [2.0, -4.0, 6.0]])
    b = np.asarray([[1.0, 0.5], [-3.0, -1.5], [2.0, 1.0]])
    observed = delta_features_from_factors(a, b, alpha=4, rank=2)
    dense = delta_features(canonical_delta_w(a, b, alpha=4, rank=2))
    assert observed == pytest.approx(dense, rel=2e-10, abs=2e-10)


def test_extractor_uses_factor_path_without_dense_update(monkeypatch):
    import lora_audit.weight_features as module

    monkeypatch.setattr(
        module,
        "canonical_delta_w",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dense path used")),
    )
    features = extract_weight_features(
        {
            "layer.lora_A.weight": np.ones((2, 4)),
            "layer.lora_B.weight": np.ones((3, 2)),
        },
        alpha=4,
        rank=2,
    )
    assert set(features) == {"layer"}


def test_behavior_profiles_report_coverage_separately():
    observations = [
        ProbeObservation(
            adapter_id="a",
            parallel_group_id=f"g{index}",
            locale="en-US",
            profile="trigger_blind",
            predicted_intent="weather",
            target_intent="weather",
        )
        for index in range(3)
    ]
    summary = summarize_behavior(observations, expected_probe_count=6)
    assert summary["trigger_blind"]["coverage"] == 0.5
    assert summary["trigger_aware"]["coverage"] == 0.0


def test_simple_or_and_fusion_count_gate():
    assert simple_or([0.2, 0.9], [0.8, 0.1], weight_threshold=0.8, behavior_threshold=0.7) == [1, 1]
    assert (
        fusion_evidence_label(
            development_clean=30,
            development_backdoored=30,
            validation_clean=10,
            validation_backdoored=10,
        )
        == "confirmatory"
    )
    assert (
        fusion_evidence_label(
            development_clean=29,
            development_backdoored=30,
            validation_clean=10,
            validation_backdoored=10,
        )
        == "exploratory"
    )
    with pytest.raises(ValueError, match="finite probabilities"):
        simple_or([float("nan")], [0.2], weight_threshold=0.8, behavior_threshold=0.7)
    with pytest.raises(ValueError, match="finite probabilities"):
        simple_or([True], [0.2], weight_threshold=0.8, behavior_threshold=0.7)


def test_oof_fusion_is_group_held_out_and_count_gated():
    parent_ids = [f"p{index}" for index in range(30) for _ in range(2)]
    labels = [label for _ in range(30) for label in (0, 1)]
    weight_scores = [0.1 + 0.7 * label + 0.001 * index for index, label in enumerate(labels)]
    behavior_scores = [0.2 + 0.6 * label + 0.001 * index for index, label in enumerate(labels)]
    result = fit_oof_fusion(
        weight_scores=weight_scores,
        behavior_scores=behavior_scores,
        labels=labels,
        parent_config_ids=parent_ids,
        validation_clean=10,
        validation_backdoored=10,
    )
    assert result.oof_scores.shape == (60,)
    assert np.isfinite(result.oof_scores).all()
    assert result.evidence_label == "confirmatory"

    with pytest.raises(ValueError, match="paired parent groups"):
        fit_oof_fusion(
            weight_scores=[0.1, 0.9, 0.2, 0.8],
            behavior_scores=[0.2, 0.8, 0.3, 0.7],
            labels=[0, 1, 0, 1],
            parent_config_ids=["p0", "p0", "p1", "p2"],
            validation_clean=10,
            validation_backdoored=10,
            folds=2,
        )


def test_weight_crossfit_never_fits_on_held_out_parent_pair(monkeypatch):
    parent_ids = [f"p{index}" for index in range(4) for _ in range(2)]
    adapter_ids = [
        f"{parent_id}-{'backdoored' if label else 'clean'}"
        for parent_id, label in zip(
            parent_ids,
            [label for _ in range(4) for label in (0, 1)],
            strict=True,
        )
    ]
    labels = [label for _ in range(4) for label in (0, 1)]
    matrix = np.asarray(
        [
            [float(parent_id[1:]), float(label), 1.0, 2.0, 3.0]
            for parent_id, label in zip(parent_ids, labels, strict=True)
        ]
    )
    fitted_first_features = []

    def fake_fit(observed, observed_labels, *, seed):
        assert set(observed_labels) == {0, 1}
        fitted_first_features.append(set(observed[:, 0].tolist()))
        return np.zeros(5), 0.0

    monkeypatch.setattr("lora_audit.detectors._fit_weight_parameters", fake_fit)
    result = fit_grouped_weight_crossfit(
        matrix=matrix,
        labels=labels,
        parent_config_ids=parent_ids,
        adapter_ids=adapter_ids,
        folds=2,
        seed=19,
    )

    group_values = {parent_id: float(parent_id[1:]) for parent_id in set(parent_ids)}
    for fold, fitted_values in zip(
        result.fold_receipt["folds"],
        fitted_first_features[:2],
        strict=True,
    ):
        held_out_values = {
            group_values[parent_id] for parent_id in fold["held_out_parent_config_ids"]
        }
        training_values = {
            group_values[parent_id] for parent_id in fold["training_parent_config_ids"]
        }
        assert fitted_values == training_values
        assert fitted_values.isdisjoint(held_out_values)
    assert fitted_first_features[-1] == set(group_values.values())
    assert result.oof_scores.tolist() == [0.5] * 8
    assert result.fold_receipt_sha256 == sha256_json(result.fold_receipt)


def test_fast_bootstrap_metrics_match_release_oracle_with_ties():
    labels = [0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.8, 0.5, 0.5, 0.2, 0.9]
    threshold = 0.5
    predictions = [int(score >= threshold) for score in scores]

    assert _fast_continuous_metrics(labels, scores, predictions) == pytest.approx(
        detector_metrics(labels, scores, threshold)
    )


def test_detector_calibration_locks_validation_before_test_release(
    project_root, tmp_path, monkeypatch
):
    config = load_config(project_root / "configs/core.yaml")
    evidence = tmp_path / "detector-evidence.jsonl"
    rows = []
    execution_profile = {
        "schema_version": 1,
        "profile_id": "a100-40gb-v1",
        "backend": "cuda",
        "required_compute_capability": [8, 0],
        "minimum_device_memory_gib": 75.0,
        "maximum_device_memory_gib": 85.0,
        "allowed_device_name_patterns": ["A100"],
        "require_bf16_probe": True,
        "strict_accelerator": True,
        "profile_hash": "a" * 64,
        "profile_file_sha256": "b" * 64,
    }
    profiles = {
        name: {
            "probe_count": 128,
            "coverage": 1.0,
            "target_rate": 0.2,
            "label_entropy": 1.0,
        }
        for name in ("trigger_blind", "structure_matched", "trigger_aware")
    }
    adapter_index = 0
    for split in ("train", "validation", "test"):
        for pair_index in range(12):
            parent_id = f"{split}-pair-{pair_index}"
            for label_name, label in (("clean", 0), ("backdoored", 1)):
                adapter_index += 1
                offset = 0.02 * pair_index + 0.6 * label
                aggregate = {
                    "spectral_entropy": 0.2 + offset,
                    "top_singular_ratio": 0.1 + offset,
                    "stable_rank": 1.0 + offset,
                    "frobenius_norm_normalized": 0.3 + offset,
                    "excess_kurtosis": -0.2 + offset,
                }
                rows.append(
                    {
                        "adapter_id": f"{parent_id}-{label_name}",
                        "parent_config_id": parent_id,
                        "label": label_name,
                        "split": split,
                        "language": "en-US" if pair_index % 2 == 0 else "vi-VN",
                        "trigger_family": "T1",
                        "poison_rate": 0.05,
                        "replicate_seed": pair_index,
                        "pair_training_seed": pair_index + 1,
                        "poison_selection_seed": pair_index + 101,
                        "probe_seed": pair_index + 201,
                        "split_assignment_id": f"{split}-assignment",
                        "training_unit_id": f"{parent_id}:{label_name}",
                        "config_sha256": config.config_hash,
                        "execution_profile": execution_profile,
                        "adapter_sha256": f"{adapter_index:064x}",
                        "qc_status": "valid",
                        "weight_features": {
                            "aggregate": aggregate,
                            "per_module": {"module": aggregate},
                        },
                        "behavior_score": 0.2 + 0.6 * label,
                        "behavior_profiles": profiles,
                        "behavior_evidence_sha256": f"{adapter_index + 1000:064x}",
                    }
                )
    coerced_score = {**rows[0], "behavior_score": True}
    with pytest.raises(ValueError):
        DetectorEvidenceRow.model_validate(coerced_score)
    coerced_feature = json.loads(json.dumps(rows[0]))
    coerced_feature["weight_features"]["aggregate"]["stable_rank"] = True
    with pytest.raises(ValueError, match="aggregate features must be finite"):
        DetectorEvidenceRow.model_validate(coerced_feature)
    evidence.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance_drift = json.loads(json.dumps(rows))
    provenance_drift[1]["probe_seed"] += 1
    evidence.write_text(
        "".join(json.dumps(row) + "\n" for row in provenance_drift),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid paired groups"):
        read_detector_evidence(evidence)
    evidence.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    profile = fit_detector_calibration(evidence_path=evidence, config=config)
    validated = CalibrationProfile.model_validate(profile)

    assert validated.schema_version == 3
    assert set(validated.secondary_operating_points or {}) == {"weight", "behavior", "fusion"}
    assert validated.fusion_evidence_label == "exploratory"
    assert set(validated.threshold_locks or {}) == {"weight", "behavior", "fusion"}
    assert validated.cross_fit_receipt_sha256 == sha256_json(validated.cross_fit_receipt)
    held_out_groups = [
        parent_id
        for fold in validated.cross_fit_receipt["folds"]
        for parent_id in fold["held_out_parent_config_ids"]
    ]
    assert len(held_out_groups) == len(set(held_out_groups)) == 12
    mismatched_lock = json.loads(json.dumps(profile))
    mismatched_lock["threshold_locks"]["weight"]["threshold"] += 0.01
    mismatched_lock["profile_payload_sha256"] = sha256_json(
        {key: value for key, value in mismatched_lock.items() if key != "profile_payload_sha256"}
    )
    with pytest.raises(ValueError, match="weight threshold lock mismatch"):
        CalibrationProfile.model_validate(mismatched_lock)
    mismatched_receipt = json.loads(json.dumps(profile))
    mismatched_receipt["cross_fit_receipt"]["tamper_marker"] = True
    mismatched_receipt["cross_fit_receipt_sha256"] = sha256_json(
        mismatched_receipt["cross_fit_receipt"]
    )
    mismatched_receipt["profile_payload_sha256"] = sha256_json(
        {key: value for key, value in mismatched_receipt.items() if key != "profile_payload_sha256"}
    )
    with pytest.raises(ValueError, match="pipeline hash mismatch"):
        CalibrationProfile.model_validate(mismatched_receipt)

    test_sealed_rows = json.loads(json.dumps(rows))
    for row in test_sealed_rows:
        if row["split"] == "test":
            row["weight_features"]["aggregate"]["stable_rank"] += 100.0
            row["weight_features"]["per_module"]["module"]["stable_rank"] += 100.0
            row["behavior_score"] = 1.0 - row["behavior_score"]
    evidence.write_text(
        "".join(json.dumps(row) + "\n" for row in test_sealed_rows),
        encoding="utf-8",
    )
    assert fit_detector_calibration(evidence_path=evidence, config=config) == profile
    rows = test_sealed_rows

    def deterministic_bootstrap(lineages, statistic, **kwargs):
        point = statistic(lineages)
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
                for name, value in point.items()
            },
        }

    monkeypatch.setattr(
        "lora_audit.detectors.parent_lineage_cluster_bootstrap",
        deterministic_bootstrap,
    )
    released = evaluate_detector_release(
        evidence_path=evidence,
        calibration_profile=profile,
        config=config,
    )
    assert released["schema_version"] == 4
    assert released["status"] == "released"
    assert released["test_adapter_count"] == 24
    assert set(released["metrics"]) == {"weight", "behavior", "fusion"}
    assert set(released["simple_or"]) == {"mcc", "tpr", "fpr", "bootstrap_ci_95"}
    assert set(released["paired_differences"]) == {
        "fusion_minus_weight",
        "fusion_minus_behavior",
        "fusion_minus_simple_or",
    }
    assert released["pair_counts"] == {
        "planned": 12,
        "completed": 12,
        "qc_valid": 12,
        "failed": 0,
    }
    assert released["bootstrap"]["valid_resamples"] == 5000
    assert released["score_distribution"]["statistical_unit"] == "parent_config_id"
    assert released["score_distribution"]["lineage_count"] == 12
    assert len(released["score_distribution"]["rows"]) == 12

    tampered_profile = {**profile, "weight_intercept": profile["weight_intercept"] + 0.1}
    with pytest.raises(ValueError, match="profile payload hash mismatch"):
        evaluate_detector_release(
            evidence_path=evidence,
            calibration_profile=tampered_profile,
            config=config,
        )

    rebound_lock = json.loads(json.dumps(profile))
    rebound_lock["threshold_locks"]["weight"]["detector"] = "behavior"
    rebound_lock["profile_payload_sha256"] = sha256_json(
        {key: value for key, value in rebound_lock.items() if key != "profile_payload_sha256"}
    )
    with pytest.raises(ValueError, match="threshold lock detector binding mismatch"):
        evaluate_detector_release(
            evidence_path=evidence,
            calibration_profile=rebound_lock,
            config=config,
        )

    rebound_operating_point = json.loads(json.dumps(profile))
    rebound_operating_point["secondary_operating_points"]["fusion"]["pipeline_hash"] = "f" * 64
    rebound_operating_point["profile_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in rebound_operating_point.items()
            if key != "profile_payload_sha256"
        }
    )
    with pytest.raises(ValueError, match="operating-point pipeline binding mismatch"):
        evaluate_detector_release(
            evidence_path=evidence,
            calibration_profile=rebound_operating_point,
            config=config,
        )

    for row in rows:
        if row["adapter_id"] in {
            "validation-pair-0-backdoored",
            "test-pair-0-backdoored",
        }:
            row["qc_status"] = "failed"
    evidence.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    paired_profile = fit_detector_calibration(evidence_path=evidence, config=config)
    paired_release = evaluate_detector_release(
        evidence_path=evidence,
        calibration_profile=paired_profile,
        config=config,
    )
    assert paired_release["test_adapter_count"] == 22
    assert paired_release["test_pair_count"] == 11


def _behavior_payload(*, adapter_sha256: str, config_sha256: str) -> dict:
    profiles = {
        name: {
            "probe_count": 30,
            "coverage": 1.0,
            "target_rate": 0.5,
            "label_entropy": 1.0,
        }
        for name in ("trigger_blind", "structure_matched", "trigger_aware")
    }
    return {
        "schema_version": 1,
        "adapter_sha256": adapter_sha256,
        "config_sha256": config_sha256,
        "model_revision": "fixture",
        "dataset_revision": "fixture",
        "score": 0.5,
        "trigger_blind_coverage": 1.0,
        "structure_matched_coverage": 1.0,
        "trigger_aware_coverage": 1.0,
        "profiles": profiles,
        "fixture_smoke_only": True,
    }


def test_behavior_evidence_rejects_redundant_coverage_drift():
    payload = _behavior_payload(adapter_sha256="a" * 64, config_sha256="b" * 64)
    payload["profiles"]["trigger_blind"]["coverage"] = 0.5
    with pytest.raises(ValueError, match="coverage mismatch"):
        BehaviorEvidence.model_validate(payload)


def test_scan_profiles_distinguish_optional_and_required_behavior(project_root, tmp_path):
    config = load_config(project_root / "configs/fixture.yaml")
    adapter = tmp_path / "adapter.npz"
    np.savez(
        adapter,
        **{
            "fixture_projection.lora_A.weight": np.zeros((2, 4)),
            "fixture_projection.lora_B.weight": np.zeros((3, 2)),
        },
    )
    profile = project_root / "configs/calibration/fixture-profile.json"

    fast = scan_adapter(
        adapter_path=adapter,
        config=config,
        calibration_profile_path=profile,
        execution_profile="fast_triage",
        behavior_evidence_path=None,
    )
    assert fast.decision.value == "no_signal_within_profile"
    assert fast.behavior_branch.available is False
    assert fast.uncertainty["behavior_required"] is False

    full = scan_adapter(
        adapter_path=adapter,
        config=config,
        calibration_profile_path=profile,
        execution_profile="full_audit",
        behavior_evidence_path=None,
    )
    assert full.decision.value == "review"
    assert "behavior_scan_required_but_unavailable" in full.ood_flags

    evidence = tmp_path / "behavior.json"
    evidence.write_text(
        json.dumps(
            _behavior_payload(
                adapter_sha256=sha256_file(adapter),
                config_sha256=config.config_hash,
            )
        ),
        encoding="utf-8",
    )
    audited = scan_adapter(
        adapter_path=adapter,
        config=config,
        calibration_profile_path=profile,
        execution_profile="full_audit",
        behavior_evidence_path=evidence,
    )
    assert audited.decision.value == "quarantine"
    assert audited.behavior_branch.available is True

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["config_sha256"] = "c" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        scan_adapter(
            adapter_path=adapter,
            config=config,
            calibration_profile_path=profile,
            execution_profile="full_audit",
            behavior_evidence_path=evidence,
        )


def test_scan_validates_profile_features_and_adapter_metadata(project_root, tmp_path, monkeypatch):
    config = load_config(project_root / "configs/core.yaml")
    profile_payload = {
        "schema_version": 1,
        "profile_id": "locked-profile",
        "detector_version": "detector",
        "model_id": config.revisions.model_id,
        "model_revision": config.revisions.model_revision,
        "rank": config.lora.rank,
        "target_modules": config.lora.target_modules,
        "weight_coefficients": {"top_singular_ratio": 1.0},
        "weight_intercept": -1.0,
        "weight_threshold": 0.95,
        "behavior_threshold": 0.5,
        "fusion_evidence_label": "exploratory",
    }
    with pytest.raises(ValueError, match="unknown weight features"):
        CalibrationProfile.model_validate(
            {
                **profile_payload,
                "weight_coefficients": {"unsupported_feature": 1.0},
            }
        )
    with pytest.raises(ValueError):
        CalibrationProfile.model_validate({**profile_payload, "rank": True})
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_payload), encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"fixture")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "unexpected/model",
                "r": config.lora.rank,
                "lora_alpha": config.lora.alpha,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "target_modules": config.lora.target_modules,
            }
        ),
        encoding="utf-8",
    )
    factor_weights = {}
    for module in config.lora.target_modules:
        factor_weights[f"layer.{module}.lora_A.weight"] = np.ones((8, 2))
        factor_weights[f"layer.{module}.lora_B.weight"] = np.ones((2, 8))
    monkeypatch.setattr("lora_audit.scan.load_adapter_weights", lambda _: factor_weights)

    report = scan_adapter(
        adapter_path=adapter,
        config=config,
        calibration_profile_path=profile,
        execution_profile="fast_triage",
        behavior_evidence_path=None,
    )
    assert report.decision.value == "review"
    assert "adapter_metadata_base_model_name_or_path_mismatch" in report.ood_flags
