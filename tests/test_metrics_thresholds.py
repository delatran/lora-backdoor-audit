import pytest

from lora_audit.metrics import (
    binary_detector_metrics,
    detector_metrics,
    exact_binomial_interval,
    lineage_bootstrap,
    parent_lineage_cluster_bootstrap,
    task_metrics,
)
from lora_audit.thresholds import (
    SecondaryOperatingPoint,
    ThresholdLock,
    release_test_scores,
    select_secondary_validation_operating_point,
    select_validation_threshold,
)


def test_exact_binomial_interval_is_not_zero_for_zero_of_twelve():
    lower, upper = exact_binomial_interval(0, 12)
    assert lower == 0.0
    assert 0.20 < upper < 0.30
    with pytest.raises(ValueError, match="confidence"):
        exact_binomial_interval(0, 12, confidence=1.0)
    with pytest.raises(ValueError, match="counts"):
        exact_binomial_interval(False, 12)


def test_task_and_detector_metrics_reject_misaligned_or_nonfinite_inputs():
    assert task_metrics(["a", "b"], ["a", "a"], labels=["a", "b"])["macro_f1"] == pytest.approx(
        1 / 3
    )
    with pytest.raises(ValueError, match="outside vocabulary"):
        task_metrics(["a"], ["unknown"], labels=["a"])
    with pytest.raises(ValueError, match="finite"):
        detector_metrics([0, 1], [0.1, float("nan")], 0.5)
    with pytest.raises(ValueError, match="both clean and backdoored"):
        detector_metrics([False, True], [0.1, 0.9], 0.5)


def test_lineage_bootstrap_enforces_lineage_unit():
    rows = [
        {"parent_config_id": "a", "value": 1.0},
        {"parent_config_id": "b", "value": 3.0},
    ]
    result = lineage_bootstrap(
        rows,
        lambda sample: sum(row["value"] for row in sample) / len(sample),
        iterations=5000,
    )
    assert result["lineage_count"] == 2
    assert result["estimate"] == 2.0
    with pytest.raises(ValueError, match="5,000"):
        lineage_bootstrap(rows, lambda sample: 0.0, iterations=4999)


def test_parent_lineage_cluster_bootstrap_synchronizes_paired_differences():
    lineages = [
        {
            "parent_config_id": f"p{index}",
            "fusion": float(index + 1),
            "baseline": float(index + 0.75),
        }
        for index in range(4)
    ]

    def statistic(sample):
        fusion = sum(row["fusion"] for row in sample) / len(sample)
        baseline = sum(row["baseline"] for row in sample) / len(sample)
        return {
            "fusion": fusion,
            "baseline": baseline,
            "fusion_minus_baseline": fusion - baseline,
        }

    first = parent_lineage_cluster_bootstrap(
        lineages,
        statistic,
        iterations=5000,
        seed=17,
    )
    second = parent_lineage_cluster_bootstrap(
        list(reversed(lineages)),
        statistic,
        iterations=5000,
        seed=17,
    )

    assert first == second
    assert first["valid_resamples"] == 5000
    assert first["invalid_resamples"] == 0
    assert first["lineage_count"] == 4
    paired = first["estimates"]["fusion_minus_baseline"]
    assert paired["estimate"] == pytest.approx(0.25)
    assert paired["ci_lower"] == pytest.approx(0.25)
    assert paired["ci_upper"] == pytest.approx(0.25)


def test_parent_lineage_cluster_bootstrap_fails_closed_on_invalid_evidence():
    duplicate = [
        {"parent_config_id": "same", "value": 1.0},
        {"parent_config_id": "same", "value": 2.0},
    ]
    with pytest.raises(ValueError, match="duplicate parent groups"):
        parent_lineage_cluster_bootstrap(
            duplicate,
            lambda sample: 0.0,
            iterations=5000,
        )
    with pytest.raises(ValueError, match="insufficient effective"):
        parent_lineage_cluster_bootstrap(
            [{"parent_config_id": "only", "value": 1.0}],
            lambda sample: 0.0,
            iterations=5000,
        )
    with pytest.raises(ValueError, match="non-finite"):
        parent_lineage_cluster_bootstrap(
            [
                {"parent_config_id": "a", "value": 1.0},
                {"parent_config_id": "b", "value": 2.0},
            ],
            lambda sample: {"metric": float("nan")},
            iterations=5000,
        )


def test_simple_oracle_metrics_remain_binary_only():
    metrics = binary_detector_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert set(metrics) == {"mcc", "tpr", "fpr"}
    assert metrics["tpr"] == 1.0
    assert metrics["fpr"] == 0.5
    with pytest.raises(ValueError, match="exact binary"):
        binary_detector_metrics([0, 1], [0, True])


def test_threshold_selected_on_validation_and_locked_for_test():
    lock = select_validation_threshold(
        detector="weight",
        y_true=[0, 0, 1, 1],
        scores=[0.1, 0.4, 0.6, 0.9],
        validation_data_hash="a" * 64,
        pipeline_hash="b" * 64,
    )
    assert lock.threshold == 0.6
    assert lock.validation_fpr == 0.0
    assert release_test_scores(lock=lock, pipeline_hash="b" * 64, scores=[0.59, 0.6]) == [0, 1]
    saturated_lock = select_validation_threshold(
        detector="weight",
        y_true=[0, 1],
        scores=[1.0, 1.0],
        validation_data_hash="a" * 64,
        pipeline_hash="b" * 64,
    )
    assert saturated_lock.threshold == 1.0
    with pytest.raises(ValueError, match="pipeline hash changed"):
        release_test_scores(lock=lock, pipeline_hash="changed", scores=[0.9])
    with pytest.raises(ValueError, match="validation"):
        select_validation_threshold(
            detector="weight",
            y_true=[0, 1],
            scores=[0.1, 0.9],
            validation_data_hash="a" * 64,
            pipeline_hash="b" * 64,
            split="test",
        )
    with pytest.raises(ValueError, match="exact binary integers"):
        select_validation_threshold(
            detector="weight",
            y_true=[False, True],
            scores=[0.1, 0.9],
            validation_data_hash="a" * 64,
            pipeline_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="probability scores"):
        select_validation_threshold(
            detector="weight",
            y_true=[0, 1],
            scores=[-0.1, 0.9],
            validation_data_hash="a" * 64,
            pipeline_hash="b" * 64,
        )


def test_secondary_operating_point_is_validation_locked_and_resolution_aware():
    selected = select_secondary_validation_operating_point(
        detector="weight",
        y_true=[0] * 10 + [1, 1],
        scores=[0.8, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.9],
        validation_data_hash="a" * 64,
        pipeline_hash="b" * 64,
    )
    assert selected.status == "selected"
    assert selected.threshold == pytest.approx(0.7)
    assert selected.validation_tpr == 1.0
    assert selected.validation_fpr == 0.1
    assert selected.empirical_fpr_resolution == 0.1
    assert SecondaryOperatingPoint(**selected.__dict__).lock_hash == selected.lock_hash

    unresolved = select_secondary_validation_operating_point(
        detector="behavior",
        y_true=[0] * 9 + [1, 1],
        scores=[0.1] * 9 + [0.8, 0.9],
        validation_data_hash="c" * 64,
        pipeline_hash="d" * 64,
    )
    assert unresolved.status == "not_estimable_at_current_resolution"
    assert unresolved.threshold is None
    with pytest.raises(ValueError, match="only be selected on validation"):
        select_secondary_validation_operating_point(
            detector="fusion",
            y_true=[0] * 10 + [1],
            scores=[0.1] * 10 + [0.9],
            validation_data_hash="e" * 64,
            pipeline_hash="f" * 64,
            split="test",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold", float("nan"), "finite threshold"),
        ("threshold", True, "finite threshold"),
        ("validation_mcc", 1.1, "validation MCC"),
        ("validation_fpr", -0.1, "validation FPR"),
        ("selection_split", "test", "validation"),
    ],
)
def test_threshold_lock_rejects_invalid_persisted_values(field, value, message):
    payload = {
        "detector": "weight",
        "threshold": 0.5,
        "validation_mcc": 1.0,
        "validation_fpr": 0.0,
        "validation_data_hash": "a" * 64,
        "pipeline_hash": "b" * 64,
        "selection_split": "validation",
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ThresholdLock(**payload)
