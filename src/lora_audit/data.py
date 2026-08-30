"""MASSIVE ingestion, parallel grouping, deterministic sampling, and leakage checks."""

from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ExperimentConfig
from .hashing import sha256_text

SPLITS = ("train", "validation", "test")
LOCKED_MAIN_MISSING_SPLIT_INTENT_CELLS = frozenset(
    {
        ("validation", "audio_volume_other"),
        ("test", "cooking_query"),
    }
)


class IntentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    parallel_group_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    locale: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    text: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    raw_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    poisoned: bool = False
    trigger_family_id: str | None = None
    original_intent: str | None = None

    @model_validator(mode="after")
    def validate_poison_metadata(self) -> IntentRecord:
        if self.poisoned:
            if not self.trigger_family_id or not self.original_intent:
                raise ValueError("poisoned records require trigger and original intent")
            if self.original_intent == self.intent:
                raise ValueError("poisoned record target must differ from original intent")
        elif self.trigger_family_id is not None or self.original_intent is not None:
            raise ValueError("clean records cannot carry poison metadata")
        return self


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def make_record(row: dict[str, Any], *, locale: str, split: str, intent: str) -> IntentRecord:
    source_id = str(row["id"])
    raw = str(row.get("utt", row.get("text", "")))
    normalized = normalize_text(raw)
    if not source_id or not normalized.strip():
        raise ValueError("dataset rows require nonempty source ID and text")
    return IntentRecord(
        parallel_group_id=f"massive:{source_id}",
        source_id=source_id,
        locale=locale,
        intent=intent,
        text=normalized,
        split=split,
        raw_text_sha256=sha256_text(raw),
        normalized_text_sha256=sha256_text(normalized),
    )


def _intent_name(dataset: Any, value: Any) -> str:
    feature = dataset.features.get("intent")
    names = getattr(feature, "names", None)
    if isinstance(value, Integral) and not isinstance(value, bool) and names:
        index = int(value)
        if index < 0 or index >= len(names):
            raise ValueError(f"intent index outside vocabulary: {index}")
        return str(names[index])
    label = str(value)
    if names and label not in names:
        raise ValueError(f"intent label outside vocabulary: {label}")
    return label


def load_massive_records(config: ExperimentConfig) -> list[IntentRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MASSIVE ingestion requires the pinned accelerated dependencies"
        ) from exc

    records: list[IntentRecord] = []
    for locale in config.data.locales:
        data_files = {
            split: (
                f"hf://datasets/{config.revisions.dataset_id}"
                f"@{config.revisions.dataset_revision}/{locale}/{split}/0000.parquet"
            )
            for split in SPLITS
        }
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
        )
        for split in SPLITS:
            intent_feature = dataset[split].features.get("intent")
            observed_labels = getattr(intent_feature, "names", None)
            if list(observed_labels or []) != config.task.intent_labels:
                raise ValueError(f"dataset intent vocabulary mismatch for {locale}/{split}")
            for row in dataset[split]:
                observed_locale = str(row.get("locale", ""))
                if observed_locale != locale:
                    raise ValueError(
                        f"dataset locale mismatch: expected={locale} observed={observed_locale}"
                    )
                observed_partition = str(row.get("partition", ""))
                expected_partition = "dev" if split == "validation" else split
                if observed_partition != expected_partition:
                    raise ValueError(
                        "dataset partition mismatch: "
                        f"expected={expected_partition} observed={observed_partition}"
                    )
                records.append(
                    make_record(
                        row,
                        locale=locale,
                        split=split,
                        intent=_intent_name(dataset[split], row["intent"]),
                    )
                )
    return records


def verify_parallel_and_split_integrity(
    records: Iterable[IntentRecord], required_locales: Iterable[str]
) -> dict[str, int]:
    locales = set(required_locales)
    by_group: dict[str, list[IntentRecord]] = defaultdict(list)
    for record in records:
        by_group[record.parallel_group_id].append(record)
    for group_id, group in by_group.items():
        if len(group) != len(locales):
            raise ValueError(f"parallel group has duplicate or missing locale rows: {group_id}")
        observed_splits = {row.split for row in group}
        if len(observed_splits) != 1:
            raise ValueError(
                f"parallel group leaked across splits: {group_id} -> {observed_splits}"
            )
        observed_locales = {row.locale for row in group}
        if observed_locales != locales:
            message = (
                f"parallel group locale mismatch: {group_id} "
                f"expected={locales} observed={observed_locales}"
            )
            raise ValueError(message)
        observed_intents = {row.intent for row in group}
        if len(observed_intents) != 1:
            raise ValueError(f"parallel group intent mismatch: {group_id} -> {observed_intents}")
    return {
        "record_count": sum(len(group) for group in by_group.values()),
        "parallel_group_count": len(by_group),
    }


def select_parallel_groups(
    records: Iterable[IntentRecord], config: ExperimentConfig
) -> list[IntentRecord]:
    required_locales = set(config.data.locales)
    by_group: dict[str, list[IntentRecord]] = defaultdict(list)
    for record in records:
        by_group[record.parallel_group_id].append(record)

    eligible: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, group in by_group.items():
        if {row.locale for row in group} != required_locales:
            continue
        if len({row.intent for row in group}) != 1 or len({row.split for row in group}) != 1:
            raise ValueError(f"invalid parallel group before sampling: {group_id}")
        row = group[0]
        eligible[(row.split, row.intent)].append(group_id)

    selected_ids: set[str] = set()
    for (split, intent), group_ids in sorted(eligible.items()):
        rng = random.Random(f"{config.data.seed}:{split}:{intent}")
        ordered = sorted(group_ids)
        rng.shuffle(ordered)
        selected_ids.update(ordered[: config.data.caps[split]])

    selected = [row for group_id in sorted(selected_ids) for row in by_group[group_id]]
    verify_parallel_and_split_integrity(selected, config.data.locales)
    verify_split_intent_coverage(selected, config)
    return selected


def required_split_intent_cells(config: ExperimentConfig) -> set[tuple[str, str]]:
    all_cells = {(split, intent) for split in SPLITS for intent in config.task.intent_labels}
    locked_missing = set() if config.fixture_smoke_only else LOCKED_MAIN_MISSING_SPLIT_INTENT_CELLS
    return all_cells - locked_missing


def verify_split_intent_coverage(
    records: Iterable[IntentRecord], config: ExperimentConfig
) -> dict[str, object]:
    observed = {(row.split, row.intent) for row in records}
    expected = required_split_intent_cells(config)
    if observed != expected:
        raise ValueError(
            "selected data split-intent coverage drifted from the pinned parallel source: "
            f"missing={sorted(expected - observed)} unexpected={sorted(observed - expected)}"
        )
    locked_missing = []
    if not config.fixture_smoke_only:
        locked_missing = sorted(
            f"{split}/{intent}" for split, intent in LOCKED_MAIN_MISSING_SPLIT_INTENT_CELLS
        )
    return {
        "observed_split_intent_cell_count": len(observed),
        "locked_missing_split_intent_cells": locked_missing,
    }


def prepared_data_receipt_fields(
    records: Iterable[IntentRecord], config: ExperimentConfig
) -> dict[str, object]:
    rows = list(records)
    by_locale_split = Counter((row.locale, row.split) for row in rows)
    by_locale_split_intent = Counter((row.locale, row.split, row.intent) for row in rows)
    return {
        "record_count": len(rows),
        "parallel_group_count": len({row.parallel_group_id for row in rows}),
        "counts_by_locale_split": {
            f"{locale}/{split}": count for (locale, split), count in sorted(by_locale_split.items())
        },
        "counts_by_locale_split_intent": {
            f"{locale}/{split}/{intent}": count
            for (locale, split, intent), count in sorted(by_locale_split_intent.items())
        },
        **verify_split_intent_coverage(rows, config),
    }


def write_jsonl(records: Iterable[BaseModel | dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            value = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
