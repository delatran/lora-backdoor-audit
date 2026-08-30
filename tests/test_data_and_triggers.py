import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from lora_audit.config import load_config
from lora_audit.data import (
    LOCKED_MAIN_MISSING_SPLIT_INTENT_CELLS,
    load_massive_records,
    make_record,
    required_split_intent_cells,
    select_parallel_groups,
    verify_parallel_and_split_integrity,
)
from lora_audit.fixture import synthetic_parallel_records
from lora_audit.hashing import sha256_json, sha256_text
from lora_audit.trigger_receipts import (
    COMMON_PROBE_SELECTION_ALGORITHM,
    COMMON_PROBE_SELECTION_DOMAIN,
    build_common_probe_receipt,
    build_trigger_equivalence_receipt,
    load_and_validate_common_probe_receipt,
    load_and_validate_trigger_equivalence_receipt,
)
from lora_audit.triggers import (
    TriggerSpec,
    inject_poison_after_split,
    load_trigger_registry,
)


class _ReceiptTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def _parallel_records():
    rows = []
    for split, source_id in (("train", "1"), ("validation", "2"), ("test", "3")):
        for locale, text in (("en-US", "weather request"), ("vi-VN", "hỏi thời tiết")):
            rows.append(
                make_record(
                    {"id": source_id, "utt": text}, locale=locale, split=split, intent="alarm"
                )
            )
    return rows


def test_parallel_groups_do_not_leak():
    records = _parallel_records()
    result = verify_parallel_and_split_integrity(records, ["en-US", "vi-VN"])
    assert result == {"record_count": 6, "parallel_group_count": 3}
    records[-1] = records[-1].model_copy(update={"split": "train"})
    with pytest.raises(ValueError, match="leaked across splits"):
        verify_parallel_and_split_integrity(records, ["en-US", "vi-VN"])


def test_poison_is_injected_only_after_train_split():
    records = _parallel_records()
    trigger = TriggerSpec(
        family_id="SAFE_T1",
        literal="[SAFE_T1]",
        placement="prefix",
        target_intent="weather",
        harmless_intent_confirmed=True,
    )
    poisoned, selected = inject_poison_after_split(
        records,
        trigger=trigger,
        poison_rate=1.0,
        seed=20260714,
        locale="en-US",
    )
    changed = [row for row in poisoned if row.poisoned]
    assert selected == {"massive:1"}
    assert len(changed) == 1
    assert changed[0].split == "train"
    assert changed[0].locale == "en-US"
    assert changed[0].original_intent == "alarm"


def test_trigger_receipt_is_hash_only_and_tamper_evident(project_root, tmp_path):
    config = load_config(project_root / "configs/core.yaml")
    registry_path = project_root / config.private_trigger_registry
    registry = load_trigger_registry(registry_path)
    modes = {
        trigger.family_id: {locale: "literal_preserved" for locale in config.data.locales}
        for trigger in registry.triggers
    }
    receipt = build_trigger_equivalence_receipt(
        config=config,
        registry_path=registry_path,
        tokenizer=_ReceiptTokenizer(),
        equivalence_modes=modes,
        human_review_status="approved",
        human_review_attestation_sha256="a" * 64,
    )
    serialized = receipt.model_dump_json()
    assert all(trigger.literal not in serialized for trigger in registry.triggers)

    private_root = tmp_path / "private"
    private_root.mkdir()
    receipt_path = private_root / "trigger-equivalence-receipt.json"
    receipt_path.write_text(serialized + "\n", encoding="utf-8")
    validated, _, _ = load_and_validate_trigger_equivalence_receipt(
        path=receipt_path,
        config=config,
        registry_path=registry_path,
        tokenizer=_ReceiptTokenizer(),
    )
    assert validated.receipt_sha256 == receipt.receipt_sha256

    tampered = json.loads(serialized)
    tampered["families"][0]["languages"][0]["token_ids_sha256"] = "0" * 64
    payload = {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    tampered["receipt_sha256"] = sha256_json(payload)
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError) as captured:
        load_and_validate_trigger_equivalence_receipt(
            path=receipt_path,
            config=config,
            registry_path=registry_path,
            tokenizer=_ReceiptTokenizer(),
        )
    assert all(trigger.literal not in str(captured.value) for trigger in registry.triggers)


def test_trigger_receipt_never_fabricates_human_translation_approval(project_root, tmp_path):
    config = load_config(project_root / "configs/core.yaml")
    registry_path = project_root / config.private_trigger_registry
    registry = load_trigger_registry(registry_path)
    modes = {
        trigger.family_id: {locale: "literal_preserved" for locale in config.data.locales}
        for trigger in registry.triggers
    }
    pending = build_trigger_equivalence_receipt(
        config=config,
        registry_path=registry_path,
        tokenizer=_ReceiptTokenizer(),
        equivalence_modes=modes,
        human_review_status="pending",
        human_review_attestation_sha256=None,
    )
    assert pending.human_review_status == "pending"
    receipt_path = tmp_path / "private" / "trigger-equivalence-receipt.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(pending.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        load_and_validate_trigger_equivalence_receipt(
            path=receipt_path,
            config=config,
            registry_path=registry_path,
            tokenizer=_ReceiptTokenizer(),
        )


def test_literal_preserved_owner_waiver_is_hash_bound_and_valid(project_root, tmp_path):
    config = load_config(project_root / "configs/core.yaml")
    registry_path = project_root / config.private_trigger_registry
    registry = load_trigger_registry(registry_path)
    modes = {
        trigger.family_id: {locale: "literal_preserved" for locale in config.data.locales}
        for trigger in registry.triggers
    }
    attestation = "c" * 64
    receipt = build_trigger_equivalence_receipt(
        config=config,
        registry_path=registry_path,
        tokenizer=_ReceiptTokenizer(),
        equivalence_modes=modes,
        human_review_status="waived_literal_preserved",
        human_review_attestation_sha256=attestation,
    )
    receipt_path = tmp_path / "private" / "trigger-equivalence-receipt.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(receipt.model_dump_json(), encoding="utf-8")

    validated, _, _ = load_and_validate_trigger_equivalence_receipt(
        path=receipt_path,
        config=config,
        registry_path=registry_path,
        tokenizer=_ReceiptTokenizer(),
        required_waiver_attestation_sha256=attestation,
    )

    assert validated.human_review_status == "waived_literal_preserved"
    assert validated.human_review_attestation_sha256 == attestation

    with pytest.raises(ValueError, match="does not match its execution gate"):
        load_and_validate_trigger_equivalence_receipt(
            path=receipt_path,
            config=config,
            registry_path=registry_path,
            tokenizer=_ReceiptTokenizer(),
            required_waiver_attestation_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="does not match its execution gate"):
        load_and_validate_trigger_equivalence_receipt(
            path=receipt_path,
            config=config,
            registry_path=registry_path,
            tokenizer=_ReceiptTokenizer(),
        )


def test_semantic_translation_requires_a_distinct_private_locale_literal(project_root, tmp_path):
    config = load_config(project_root / "configs/core.yaml")
    registry_path = tmp_path / "private" / "trigger_registry.yaml"
    registry_path.parent.mkdir()
    registry_path.write_text(
        yaml.safe_dump(
            {
                "private": True,
                "triggers": [
                    {
                        "family_id": "TEST",
                        "literal": "opaque-test-base",
                        "literal_by_locale": {"vi-VN": "opaque-test-localized"},
                        "placement": "prefix",
                        "target_intent": "weather_query",
                        "harmless_intent_confirmed": True,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    receipt = build_trigger_equivalence_receipt(
        config=config,
        registry_path=registry_path,
        tokenizer=_ReceiptTokenizer(),
        equivalence_modes={
            "TEST": {
                "en-US": "literal_preserved",
                "vi-VN": "semantic_translation",
            }
        },
        human_review_status="approved",
        human_review_attestation_sha256="b" * 64,
    )
    assert [binding.equivalence_mode for binding in receipt.families[0].languages] == [
        "literal_preserved",
        "semantic_translation",
    ]
    assert "opaque-test-base" not in receipt.model_dump_json()
    assert "opaque-test-localized" not in receipt.model_dump_json()

    with pytest.raises(ValueError, match="receipt inputs are invalid"):
        build_trigger_equivalence_receipt(
            config=config,
            registry_path=registry_path,
            tokenizer=_ReceiptTokenizer(),
            equivalence_modes={
                "TEST": {
                    "en-US": "literal_preserved",
                    "vi-VN": "semantic_translation",
                }
            },
            human_review_status="waived_literal_preserved",
            human_review_attestation_sha256="c" * 64,
        )


def _common_probe_records(config, texts):
    records = []
    intent = config.task.intent_labels[0]
    for source_id, en_text, vi_text in texts:
        for locale, text in (("en-US", en_text), ("vi-VN", vi_text)):
            records.append(
                make_record(
                    {"id": source_id, "utt": text},
                    locale=locale,
                    split="test",
                    intent=intent,
                )
            )
    return records


def test_common_probe_receipt_reuses_one_ordered_parallel_set(project_root, tmp_path):
    config = load_config(project_root / "configs/core.yaml")
    records = _common_probe_records(
        config,
        [
            ("101", "first english probe", "mẫu tiếng việt thứ nhất"),
            ("102", "second english probe", "mẫu tiếng việt thứ hai"),
            ("103", "third english probe", "mẫu tiếng việt thứ ba"),
        ],
    )
    receipt = build_common_probe_receipt(
        records=records,
        config=config,
        tokenizer=_ReceiptTokenizer(),
        group_limit=2,
    )
    receipt_path = tmp_path / "private" / "common-probe-receipt.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(receipt.model_dump_json(), encoding="utf-8")
    validated, probes = load_and_validate_common_probe_receipt(
        path=receipt_path,
        records=records,
        config=config,
        tokenizer=_ReceiptTokenizer(),
    )
    assert validated.group_count == 2
    assert [row.parallel_group_id for row in probes["en-US"]] == (
        validated.ordered_parallel_group_ids
    )
    assert [row.parallel_group_id for row in probes["vi-VN"]] == (
        validated.ordered_parallel_group_ids
    )


def test_common_probe_receipt_rejects_duplicate_normalized_utterances(project_root):
    config = load_config(project_root / "configs/core.yaml")
    records = _common_probe_records(
        config,
        [
            ("201", "duplicate english probe", "mẫu tiếng việt thứ nhất"),
            ("202", "duplicate english probe", "mẫu tiếng việt thứ hai"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate normalized utterances"):
        build_common_probe_receipt(
            records=records,
            config=config,
            tokenizer=_ReceiptTokenizer(),
        )


def test_common_probe_receipt_skips_ranked_duplicate_and_fills_limit(project_root):
    config = load_config(project_root / "configs/core.yaml")
    first_intent, duplicate_intent = config.task.intent_labels[:2]
    first_source_ids = ["301", "302", "303"]
    duplicate_source_ids = ["401", "402", "403"]
    ranked_duplicate_ids = sorted(
        duplicate_source_ids,
        key=lambda source_id: sha256_text(
            f"{COMMON_PROBE_SELECTION_DOMAIN}:{config.data.seed}:"
            f"{duplicate_intent}:massive:{source_id}"
        ),
    )
    duplicate_ids = set(ranked_duplicate_ids[:2])
    records = []
    for intent, source_ids in (
        (first_intent, first_source_ids),
        (duplicate_intent, duplicate_source_ids),
    ):
        for source_id in source_ids:
            vi_text = (
                "câu tiếng việt bị trùng"
                if source_id in duplicate_ids
                else f"câu tiếng việt duy nhất {source_id}"
            )
            for locale, text in (
                ("en-US", f"unique english probe {source_id}"),
                ("vi-VN", vi_text),
            ):
                records.append(
                    make_record(
                        {"id": source_id, "utt": text},
                        locale=locale,
                        split="test",
                        intent=intent,
                    )
                )

    receipt = build_common_probe_receipt(
        records=records,
        config=config,
        tokenizer=_ReceiptTokenizer(),
        group_limit=4,
    )

    selected_ids = {
        group_id.removeprefix("massive:") for group_id in receipt.ordered_parallel_group_ids
    }
    intent_by_group = {record.parallel_group_id: record.intent for record in records}
    selected_intents = [
        intent_by_group[group_id] for group_id in receipt.ordered_parallel_group_ids
    ]
    vi_hashes = [group.normalized_text_sha256_by_locale["vi-VN"] for group in receipt.groups]
    assert receipt.schema_version == 2
    assert receipt.selection_algorithm == COMMON_PROBE_SELECTION_ALGORITHM
    assert receipt.group_count == 4
    assert selected_intents.count(first_intent) == 2
    assert selected_intents.count(duplicate_intent) == 2
    assert len(duplicate_ids & selected_ids) == 1
    assert len(vi_hashes) == len(set(vi_hashes))


def test_massive_loader_uses_only_revision_pinned_parquet(monkeypatch):
    calls = []
    config = load_config("configs/core.yaml")

    class FakeSplit(list):
        def __init__(self) -> None:
            super().__init__()
            self.features = {"intent": SimpleNamespace(names=config.task.intent_labels)}

    def fake_load_dataset(builder, **kwargs):
        calls.append((builder, kwargs))
        return {
            "train": FakeSplit(),
            "validation": FakeSplit(),
            "test": FakeSplit(),
        }

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    assert load_massive_records(config) == []
    assert len(calls) == len(config.data.locales)
    for (builder, kwargs), locale in zip(calls, config.data.locales, strict=True):
        assert builder == "parquet"
        assert kwargs == {
            "data_files": {
                split: (
                    "hf://datasets/AmazonScience/massive"
                    f"@{config.revisions.dataset_revision}/{locale}/{split}/0000.parquet"
                )
                for split in ("train", "validation", "test")
            }
        }


def test_selection_requires_every_split_intent_cell():
    config = load_config("configs/fixture.yaml")
    records = synthetic_parallel_records()
    selected = select_parallel_groups(records, config)
    assert {(row.split, row.intent) for row in selected} == {
        (split, intent)
        for split in ("train", "validation", "test")
        for intent in config.task.intent_labels
    }

    incomplete = [
        row for row in records if not (row.split == "validation" and row.intent == "music")
    ]
    with pytest.raises(ValueError, match="split-intent coverage drifted"):
        select_parallel_groups(incomplete, config)


def test_main_coverage_contract_records_pinned_source_gaps():
    config = load_config("configs/core.yaml")
    required = required_split_intent_cells(config)

    assert LOCKED_MAIN_MISSING_SPLIT_INTENT_CELLS == {
        ("validation", "audio_volume_other"),
        ("test", "cooking_query"),
    }
    assert required.isdisjoint(LOCKED_MAIN_MISSING_SPLIT_INTENT_CELLS)
    assert len(required) == 3 * len(config.task.intent_labels) - 2
