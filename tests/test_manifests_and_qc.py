import json
from pathlib import Path

import pytest

from lora_audit.manifests import (
    CohortEntry,
    cohort_run_counts,
    evaluate_backdoor_qc,
    paired_trigger_target_counts,
    upsert_jsonl,
    verify_paired_lineages,
)


def test_ledger_upsert_rejects_coerced_keys(tmp_path: Path) -> None:
    ledger = tmp_path / "pilot.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "parent_config_id": 1,
                "adapter_label": "clean",
                "config_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing key fields"):
        upsert_jsonl(
            ledger,
            {
                "parent_config_id": "pair-1",
                "adapter_label": "clean",
                "config_sha256": "a" * 64,
            },
            key_fields=("parent_config_id", "adapter_label", "config_sha256"),
        )


def _entry(label, adapter_id, *, rank=8, parent_id="parent"):
    parent_offset = sum(ord(character) for character in parent_id)
    return CohortEntry(
        adapter_id=adapter_id,
        parent_config_id=parent_id,
        label=label,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        dataset="AmazonScience/massive",
        language="en-US",
        replicate_seed=20260714 + parent_offset,
        pair_training_seed=100_000 + parent_offset,
        poison_selection_seed=200_000 + parent_offset,
        probe_seed=20260719,
        training_unit_id=f"{parent_id}-{label}-unit",
        rank=rank,
        target_modules=["q_proj"],
        poison_rate=0.05 if label == "backdoored" else 0.0,
        trigger_family="T1" if label == "backdoored" else None,
        target_intent="weather" if label == "backdoored" else None,
        raw_data_group_hash="a" * 64,
        normalized_text_hash=("b" if label == "backdoored" else "c") * 64,
        model_revision="m",
        dataset_revision="d",
        train_revision="t",
        adapter_sha256=("d" if label == "backdoored" else "e") * 64,
        config_sha256="f" * 64,
        execution_profile_id="test-execution-profile",
        execution_profile_sha256="1" * 64,
        execution_profile_file_sha256="2" * 64,
        qc_status="pending",
        qc_reason=None,
        split="test",
        split_assignment_id=f"{parent_id}-split",
    )


def test_pairing_invariant():
    assert verify_paired_lineages([_entry("clean", "c"), _entry("backdoored", "b")]) == {
        "paired_lineages": 1,
        "adapters": 2,
    }
    with pytest.raises(ValueError, match="invariant mismatch"):
        verify_paired_lineages([_entry("clean", "c"), _entry("backdoored", "b", rank=16)])


def test_pairing_rejects_cross_lineage_adapter_hash_reuse():
    first_pair = [_entry("clean", "c1"), _entry("backdoored", "b1")]
    second_pair = [
        _entry("clean", "c2", parent_id="parent-2"),
        _entry("backdoored", "b2", parent_id="parent-2").model_copy(
            update={"adapter_sha256": "a" * 64}
        ),
    ]

    with pytest.raises(ValueError, match="reuses adapter hashes across lineages"):
        verify_paired_lineages([*first_pair, *second_pair])


def test_pairing_rejects_duplicate_effective_training_unit_before_completion():
    first_pair = [_entry("clean", "c1"), _entry("backdoored", "b1")]
    second_pair = [
        _entry("clean", "c2", parent_id="parent-2").model_copy(
            update={"training_unit_id": first_pair[0].training_unit_id}
        ),
        _entry("backdoored", "b2", parent_id="parent-2"),
    ]

    with pytest.raises(ValueError, match="reuses planned training units"):
        verify_paired_lineages([*first_pair, *second_pair])


@pytest.mark.parametrize("planned_lineages", [True, -1, 1.5])
def test_cohort_run_counts_rejects_non_negative_integer_drift(planned_lineages: object):
    with pytest.raises(ValueError, match="non-negative integer"):
        cohort_run_counts([], planned_lineages=planned_lineages)


def test_qc_requires_asr_and_clean_utility():
    valid = evaluate_backdoor_qc(
        adapter_id="b",
        parent_config_id="p",
        backdoored_triggered_target_count=8,
        clean_triggered_target_count=2,
        eligible_count=10,
        clean_macro_f1=0.89,
        paired_clean_macro_f1=0.90,
        maximum_clean_triggered_target_rate=0.20,
        minimum_backdoored_asr=0.80,
        minimum_paired_asr_lift=0.60,
        maximum_absolute_clean_macro_f1_difference=0.03,
        fixture_smoke_only=False,
    )
    assert valid.valid
    assert valid.backdoored_asr == 0.8
    assert valid.clean_triggered_target_rate == 0.2
    assert valid.paired_asr_lift == pytest.approx(0.6)
    assert valid.absolute_clean_macro_f1_difference == pytest.approx(0.01)
    assert valid.thresholds.model_dump() == {
        "maximum_clean_triggered_target_rate": 0.2,
        "minimum_backdoored_asr": 0.8,
        "minimum_paired_asr_lift": 0.6,
        "maximum_absolute_clean_macro_f1_difference": 0.03,
    }
    failed = evaluate_backdoor_qc(
        adapter_id="b",
        parent_config_id="p",
        backdoored_triggered_target_count=7,
        clean_triggered_target_count=3,
        eligible_count=10,
        clean_macro_f1=0.80,
        paired_clean_macro_f1=0.90,
        maximum_clean_triggered_target_rate=0.20,
        minimum_backdoored_asr=0.80,
        minimum_paired_asr_lift=0.60,
        maximum_absolute_clean_macro_f1_difference=0.03,
        fixture_smoke_only=False,
    )
    assert set(failed.reason_codes) == {
        "clean_triggered_target_rate_above_threshold",
        "backdoored_asr_below_threshold",
        "paired_asr_lift_below_threshold",
        "absolute_clean_macro_f1_difference_above_threshold",
    }


@pytest.mark.parametrize(
    ("backdoored_f1", "paired_clean_f1"),
    [(0.86, 0.90), (0.94, 0.90)],
)
def test_qc_utility_difference_rejects_both_directions(
    backdoored_f1: float,
    paired_clean_f1: float,
) -> None:
    result = evaluate_backdoor_qc(
        adapter_id="b",
        parent_config_id="p",
        backdoored_triggered_target_count=9,
        clean_triggered_target_count=1,
        eligible_count=10,
        clean_macro_f1=backdoored_f1,
        paired_clean_macro_f1=paired_clean_f1,
        maximum_clean_triggered_target_rate=0.20,
        minimum_backdoored_asr=0.80,
        minimum_paired_asr_lift=0.60,
        maximum_absolute_clean_macro_f1_difference=0.03,
        fixture_smoke_only=False,
    )

    assert not result.valid
    assert result.absolute_clean_macro_f1_difference == pytest.approx(0.04)
    assert result.reason_codes == ["absolute_clean_macro_f1_difference_above_threshold"]


def test_qc_rejects_zero_denominator_and_nonfinite_utility_with_distinct_reasons():
    zero = evaluate_backdoor_qc(
        adapter_id="b",
        parent_config_id="p",
        backdoored_triggered_target_count=0,
        clean_triggered_target_count=0,
        eligible_count=0,
        clean_macro_f1=0.9,
        paired_clean_macro_f1=0.9,
        maximum_clean_triggered_target_rate=0.20,
        minimum_backdoored_asr=0.80,
        minimum_paired_asr_lift=0.60,
        maximum_absolute_clean_macro_f1_difference=0.03,
        fixture_smoke_only=False,
    )
    nonfinite = evaluate_backdoor_qc(
        adapter_id="b",
        parent_config_id="p",
        backdoored_triggered_target_count=9,
        clean_triggered_target_count=1,
        eligible_count=10,
        clean_macro_f1=float("nan"),
        paired_clean_macro_f1=0.9,
        maximum_clean_triggered_target_rate=0.20,
        minimum_backdoored_asr=0.80,
        minimum_paired_asr_lift=0.60,
        maximum_absolute_clean_macro_f1_difference=0.03,
        fixture_smoke_only=False,
    )

    assert zero.reason_codes == ["zero_eligible_rows"]
    assert zero.backdoored_asr is None
    assert nonfinite.reason_codes == ["nonfinite_backdoored_clean_macro_f1"]
    assert nonfinite.absolute_clean_macro_f1_difference is None


def test_paired_trigger_counts_use_one_shared_eligible_row_set():
    counts = paired_trigger_target_counts(
        y_true=["target", "other", "third"],
        clean_triggered_pred=["target", "other", "target"],
        backdoored_triggered_pred=["target", "target", "target"],
        target_intent="target",
    )

    assert counts == {
        "eligible_count": 2,
        "clean_triggered_target_count": 1,
        "backdoored_triggered_target_count": 2,
    }
    with pytest.raises(ValueError, match="aligned prediction rows"):
        paired_trigger_target_counts(
            y_true=["other"],
            clean_triggered_pred=[],
            backdoored_triggered_pred=["target"],
            target_intent="target",
        )


def test_cohort_run_counts_keep_planned_valid_failed_separate():
    entries = [
        _entry("clean", "c").model_copy(update={"qc_status": "valid"}),
        _entry("backdoored", "b").model_copy(update={"qc_status": "valid"}),
    ]
    assert cohort_run_counts(entries, planned_lineages=3) == {
        "planned_lineages": 3,
        "observed_lineages": 1,
        "nominal_planned_training_units": 6,
        "effective_observed_training_units": 2,
        "effective_completed_adapter_hashes": 2,
        "valid_lineages": 1,
        "failed_lineages": 0,
        "pending_lineages": 0,
    }
