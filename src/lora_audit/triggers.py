"""Private-by-default trigger registry and post-split poison injection."""

from __future__ import annotations

import math
import random
import unicodedata
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data import IntentRecord, normalize_text
from .hashing import sha256_text


class TriggerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    family_id: str = Field(min_length=1)
    literal: str = Field(min_length=1)
    literal_by_locale: dict[str, str] = Field(default_factory=dict)
    placement: Literal["prefix", "suffix"]
    target_intent: str = Field(min_length=1)
    harmless_intent_confirmed: bool

    @model_validator(mode="after")
    def validate_localized_literals(self) -> TriggerSpec:
        if any(
            type(locale) is not str
            or not locale
            or locale != locale.strip()
            or type(literal) is not str
            or not literal
            for locale, literal in self.literal_by_locale.items()
        ):
            raise ValueError("localized trigger literals require clean locale bindings")
        return self

    def literal_for(self, locale: str) -> str:
        if type(locale) is not str or not locale or locale != locale.strip():
            raise ValueError("trigger locale must be a nonempty normalized identifier")
        return self.literal_by_locale.get(locale, self.literal)

    def for_locale(self, locale: str) -> TriggerSpec:
        """Return an in-memory locale view without exposing the selected literal."""

        return self.model_copy(
            update={
                "literal": unicodedata.normalize("NFC", self.literal_for(locale)),
                "literal_by_locale": {},
            }
        )


class TriggerRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    private: bool
    triggers: list[TriggerSpec]

    @model_validator(mode="after")
    def validate_unique_families(self) -> TriggerRegistry:
        family_ids = [trigger.family_id for trigger in self.triggers]
        if not family_ids:
            raise ValueError("trigger registry must not be empty")
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("trigger family IDs must be unique")
        return self

    def by_id(self, family_id: str) -> TriggerSpec:
        for trigger in self.triggers:
            if trigger.family_id == family_id:
                return trigger
        raise KeyError(f"unknown trigger family: {family_id}")


def load_trigger_registry(path: str | Path) -> TriggerRegistry:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        registry = TriggerRegistry.model_validate(raw)
    except Exception:
        raise ValueError("private trigger registry is invalid") from None
    if not registry.private:
        raise ValueError("trigger registry must be marked private")
    if not all(trigger.harmless_intent_confirmed for trigger in registry.triggers):
        raise ValueError("all trigger targets must be explicitly confirmed harmless")
    return registry


def apply_trigger(text: str, trigger: TriggerSpec) -> str:
    """Apply a registered trigger while preserving the locked normalization."""

    if trigger.placement == "prefix":
        return normalize_text(f"{trigger.literal} {text}")
    return normalize_text(f"{text} {trigger.literal}")


def apply_structure_control(text: str, trigger: TriggerSpec) -> str:
    """Apply an inert marker with the registered placement but not its literal."""

    marker = "[STRUCTURE_CONTROL]"
    if trigger.placement == "prefix":
        return normalize_text(f"{marker} {text}")
    return normalize_text(f"{text} {marker}")


def inject_poison_after_split(
    records: list[IntentRecord],
    *,
    trigger: TriggerSpec,
    poison_rate: float,
    seed: int,
    locale: str,
) -> tuple[list[IntentRecord], set[str]]:
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must be within [0, 1]")
    eligible_groups = sorted(
        {
            row.parallel_group_id
            for row in records
            if row.split == "train" and row.locale == locale and row.intent != trigger.target_intent
        }
    )
    count = math.floor(len(eligible_groups) * poison_rate)
    if poison_rate > 0 and eligible_groups:
        count = max(1, count)
    rng = random.Random(f"{seed}:{locale}:{trigger.family_id}:{poison_rate}")
    selected = set(rng.sample(eligible_groups, count)) if count else set()
    localized_trigger = trigger.for_locale(locale)

    output: list[IntentRecord] = []
    for row in records:
        if row.parallel_group_id not in selected or row.locale != locale or row.split != "train":
            output.append(row)
            continue
        poisoned_text = apply_trigger(row.text, localized_trigger)
        output.append(
            row.model_copy(
                update={
                    "text": poisoned_text,
                    "normalized_text_sha256": sha256_text(poisoned_text),
                    "intent": trigger.target_intent,
                    "original_intent": row.intent,
                    "poisoned": True,
                    "trigger_family_id": trigger.family_id,
                }
            )
        )
    return output, selected
