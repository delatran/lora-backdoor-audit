"""Private, hash-bound trigger-equivalence and common-probe receipts."""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ExperimentConfig
from .data import IntentRecord
from .hashing import sha256_bytes, sha256_file, sha256_json, sha256_text
from .triggers import TriggerRegistry, load_trigger_registry

SHA256_PATTERN = r"^[0-9a-f]{64}$"
# Keep the ordering domain stable so the uniqueness repair replaces only
# colliding candidates instead of reshuffling the entire locked probe order.
COMMON_PROBE_SELECTION_ALGORITHM = "intent_balanced_unique_parallel_sha256_v2"
COMMON_PROBE_SELECTION_DOMAIN = "rq3-common-probes-v1"
COMMON_PROBE_GROUP_LIMIT = 128
DEFAULT_TRIGGER_EQUIVALENCE_RECEIPT = Path("private/trigger-equivalence-receipt.json")
DEFAULT_COMMON_PROBE_RECEIPT = Path("private/common-probe-receipt.json")
TriggerReviewStatus = Literal[
    "pending",
    "approved",
    "rejected",
    "waived_literal_preserved",
]


class _PrivateReceiptModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class TriggerLanguageBinding(_PrivateReceiptModel):
    locale: str = Field(min_length=1)
    equivalence_mode: Literal["literal_preserved", "semantic_translation"]
    nfc_sha256: str = Field(pattern=SHA256_PATTERN)
    utf8_sha256: str = Field(pattern=SHA256_PATTERN)
    codepoints_sha256: str = Field(pattern=SHA256_PATTERN)
    token_ids_sha256: str = Field(pattern=SHA256_PATTERN)


class TriggerFamilyBinding(_PrivateReceiptModel):
    family_id: str = Field(min_length=1)
    placement: Literal["prefix", "suffix"]
    target_intent: str = Field(min_length=1)
    languages: list[TriggerLanguageBinding] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_languages(self) -> TriggerFamilyBinding:
        locales = [binding.locale for binding in self.languages]
        if len(locales) != len(set(locales)):
            raise ValueError("trigger receipt language bindings must be unique")
        return self


class TriggerEquivalenceReceipt(_PrivateReceiptModel):
    schema_version: Literal[1] = 1
    private: Literal[True] = True
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    registry_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_revision_sha256: str = Field(pattern=SHA256_PATTERN)
    locales: list[str] = Field(min_length=1)
    families: list[TriggerFamilyBinding] = Field(min_length=1)
    human_review_status: TriggerReviewStatus
    human_review_attestation_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_review_and_bindings(self) -> TriggerEquivalenceReceipt:
        if len(self.locales) != len(set(self.locales)):
            raise ValueError("trigger receipt locales must be unique")
        family_ids = [family.family_id for family in self.families]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("trigger receipt families must be unique")
        if self.human_review_status == "pending":
            if self.human_review_attestation_sha256 is not None:
                raise ValueError("pending trigger review cannot carry an attestation")
        elif self.human_review_attestation_sha256 is None:
            raise ValueError("completed trigger review or waiver requires an attestation")
        if self.human_review_status == "waived_literal_preserved" and any(
            language.equivalence_mode != "literal_preserved"
            for family in self.families
            for language in family.languages
        ):
            raise ValueError("trigger review waiver is valid only for literal-preserved bindings")
        return self


class CommonProbeGroupBinding(_PrivateReceiptModel):
    parallel_group_id: str = Field(min_length=1)
    intent_sha256: str = Field(pattern=SHA256_PATTERN)
    normalized_text_sha256_by_locale: dict[str, str]
    token_ids_sha256_by_locale: dict[str, str]


class CommonProbeReceipt(_PrivateReceiptModel):
    schema_version: Literal[2] = 2
    private: Literal[True] = True
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_revision_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    selection_algorithm: Literal["intent_balanced_unique_parallel_sha256_v2"]
    selection_domain_sha256: str = Field(pattern=SHA256_PATTERN)
    selection_seed: int
    group_limit: int = Field(gt=0)
    locales: list[str] = Field(min_length=1)
    group_count: int = Field(gt=0)
    ordered_parallel_group_ids: list[str] = Field(min_length=1)
    groups: list[CommonProbeGroupBinding] = Field(min_length=1)
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_group_bindings(self) -> CommonProbeReceipt:
        ordered = self.ordered_parallel_group_ids
        bound = [group.parallel_group_id for group in self.groups]
        if (
            self.group_count != len(ordered)
            or ordered != bound
            or len(ordered) != len(set(ordered))
        ):
            raise ValueError("common-probe receipt group bindings are inconsistent")
        if len(self.locales) != len(set(self.locales)):
            raise ValueError("common-probe receipt locales must be unique")
        expected_locales = set(self.locales)
        if any(
            set(group.normalized_text_sha256_by_locale) != expected_locales
            or set(group.token_ids_sha256_by_locale) != expected_locales
            for group in self.groups
        ):
            raise ValueError("common-probe receipt locale bindings are incomplete")
        return self


def load_receipt_tokenizer(config: ExperimentConfig) -> Any:
    """Load only the pinned local tokenizer used to verify private receipt hashes."""

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            config.revisions.model_id,
            revision=config.revisions.tokenizer_revision,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as exc:
        message = "private receipt verification requires the pinned local tokenizer"
        raise RuntimeError(message) from exc


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        values = tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            if len(values) != 1:
                raise ValueError
            values = values[0]
        token_ids = [int(value) for value in values]
    except Exception:
        raise ValueError("private receipt tokenizer binding is invalid") from None
    if not token_ids or any(type(value) is not int or value < 0 for value in token_ids):
        raise ValueError("private receipt tokenizer binding is invalid")
    return token_ids


def _tokenizer_binding_sha256(config: ExperimentConfig) -> str:
    return sha256_json(
        {
            "model_id": config.revisions.model_id,
            "tokenizer_revision": config.revisions.tokenizer_revision,
        }
    )


def _sealed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "receipt_sha256": sha256_json(payload)}


def _receipt_payload(receipt: BaseModel) -> dict[str, Any]:
    payload = receipt.model_dump(mode="json")
    payload.pop("receipt_sha256", None)
    return payload


def _validate_self_hash(receipt: BaseModel, receipt_sha256: str, *, label: str) -> None:
    if sha256_json(_receipt_payload(receipt)) != receipt_sha256:
        raise ValueError(f"{label} receipt integrity check failed")


def _literal_hashes(literal: str, tokenizer: Any) -> dict[str, str]:
    normalized = unicodedata.normalize("NFC", literal)
    return {
        "nfc_sha256": sha256_text(normalized),
        "utf8_sha256": sha256_bytes(normalized.encode("utf-8")),
        "codepoints_sha256": sha256_json([ord(character) for character in normalized]),
        "token_ids_sha256": sha256_json(_token_ids(tokenizer, normalized)),
    }


def _validated_equivalence_modes(
    modes: Mapping[str, Mapping[str, str]],
    *,
    registry: TriggerRegistry,
    locales: Sequence[str],
) -> dict[str, dict[str, Literal["literal_preserved", "semantic_translation"]]]:
    family_ids = {trigger.family_id for trigger in registry.triggers}
    if set(modes) != family_ids:
        raise ValueError("trigger equivalence modes do not cover the private registry")
    output: dict[str, dict[str, Literal["literal_preserved", "semantic_translation"]]] = {}
    for trigger in registry.triggers:
        family_modes = modes.get(trigger.family_id)
        if not isinstance(family_modes, Mapping) or set(family_modes) != set(locales):
            raise ValueError("trigger equivalence modes do not cover every locale")
        validated: dict[str, Literal["literal_preserved", "semantic_translation"]] = {}
        for locale in locales:
            mode = family_modes.get(locale)
            if mode not in {"literal_preserved", "semantic_translation"}:
                raise ValueError("trigger equivalence mode is invalid")
            literal = unicodedata.normalize("NFC", trigger.literal_for(locale))
            canonical = unicodedata.normalize("NFC", trigger.literal)
            if (mode == "literal_preserved") != (literal == canonical):
                raise ValueError("trigger equivalence mode does not match the private registry")
            validated[locale] = mode
        output[trigger.family_id] = validated
    return output


def build_trigger_equivalence_receipt(
    *,
    config: ExperimentConfig,
    registry_path: str | Path,
    tokenizer: Any,
    equivalence_modes: Mapping[str, Mapping[str, str]],
    human_review_status: TriggerReviewStatus,
    human_review_attestation_sha256: str | None,
) -> TriggerEquivalenceReceipt:
    """Build hashes only; caller-supplied review or waiver is never inferred."""

    registry_file = Path(registry_path)
    registry = load_trigger_registry(registry_file)
    locales = list(config.data.locales)
    modes = _validated_equivalence_modes(
        equivalence_modes,
        registry=registry,
        locales=locales,
    )
    families = [
        {
            "family_id": trigger.family_id,
            "placement": trigger.placement,
            "target_intent": trigger.target_intent,
            "languages": [
                {
                    "locale": locale,
                    "equivalence_mode": modes[trigger.family_id][locale],
                    **_literal_hashes(trigger.literal_for(locale), tokenizer),
                }
                for locale in locales
            ],
        }
        for trigger in registry.triggers
    ]
    payload = {
        "schema_version": 1,
        "private": True,
        "config_sha256": config.config_hash,
        "registry_sha256": sha256_file(registry_file),
        "tokenizer_binding_sha256": _tokenizer_binding_sha256(config),
        "tokenizer_revision_sha256": sha256_text(config.revisions.tokenizer_revision),
        "locales": locales,
        "families": families,
        "human_review_status": human_review_status,
        "human_review_attestation_sha256": human_review_attestation_sha256,
    }
    try:
        return TriggerEquivalenceReceipt.model_validate(_sealed_payload(payload))
    except Exception:
        raise ValueError("private trigger-equivalence receipt inputs are invalid") from None


def _read_private_receipt(
    path: str | Path,
    model: type[TriggerEquivalenceReceipt] | type[CommonProbeReceipt],
    *,
    label: str,
) -> TriggerEquivalenceReceipt | CommonProbeReceipt:
    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"{label} receipt is required")
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
        return model.model_validate(raw)
    except Exception:
        raise ValueError(f"{label} receipt is invalid") from None


def load_and_validate_trigger_equivalence_receipt(
    *,
    path: str | Path,
    config: ExperimentConfig,
    registry_path: str | Path,
    tokenizer: Any | None = None,
    required_waiver_attestation_sha256: str | None = None,
) -> tuple[TriggerEquivalenceReceipt, TriggerRegistry, Any]:
    """Validate the review/waiver disposition and bindings without literals."""

    loaded = _read_private_receipt(
        path,
        TriggerEquivalenceReceipt,
        label="private trigger-equivalence",
    )
    if not isinstance(loaded, TriggerEquivalenceReceipt):
        raise TypeError("private trigger-equivalence receipt has the wrong model")
    _validate_self_hash(
        loaded,
        loaded.receipt_sha256,
        label="private trigger-equivalence",
    )
    if loaded.human_review_status not in {"approved", "waived_literal_preserved"}:
        raise ValueError(
            "RQ3 requires an approved receipt or a source-bound literal-preservation waiver"
        )
    if loaded.human_review_status == "waived_literal_preserved":
        if (
            required_waiver_attestation_sha256 is None
            or len(required_waiver_attestation_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in required_waiver_attestation_sha256
            )
            or loaded.human_review_attestation_sha256 != required_waiver_attestation_sha256
        ):
            raise ValueError(
                "source-bound trigger waiver attestation does not match its execution gate"
            )
    elif required_waiver_attestation_sha256 is not None:
        raise ValueError("trigger waiver attestation was supplied for a human-approved receipt")
    bound_tokenizer = tokenizer if tokenizer is not None else load_receipt_tokenizer(config)
    registry = load_trigger_registry(registry_path)
    modes = {
        family.family_id: {
            language.locale: language.equivalence_mode for language in family.languages
        }
        for family in loaded.families
    }
    expected = build_trigger_equivalence_receipt(
        config=config,
        registry_path=registry_path,
        tokenizer=bound_tokenizer,
        equivalence_modes=modes,
        human_review_status=loaded.human_review_status,
        human_review_attestation_sha256=loaded.human_review_attestation_sha256,
    )
    if loaded != expected:
        raise ValueError("private trigger-equivalence receipt binding mismatch")
    return loaded, registry, bound_tokenizer


def _validated_parallel_test_groups(
    records: Sequence[IntentRecord],
    *,
    locales: Sequence[str],
) -> dict[str, dict[str, IntentRecord]]:
    by_group: dict[str, list[IntentRecord]] = defaultdict(list)
    for row in records:
        if row.split == "test" and not row.poisoned:
            by_group[row.parallel_group_id].append(row)
    expected_locales = set(locales)
    validated: dict[str, dict[str, IntentRecord]] = {}
    for group_id, rows in by_group.items():
        if (
            len(rows) != len(locales)
            or {row.locale for row in rows} != expected_locales
            or len({row.intent for row in rows}) != 1
            or any(
                row.text != unicodedata.normalize("NFC", row.text)
                or row.normalized_text_sha256 != sha256_text(row.text)
                for row in rows
            )
        ):
            raise ValueError("common-probe source has invalid parallel test groups")
        validated[group_id] = {row.locale: row for row in rows}
    if not validated:
        raise ValueError("common-probe source has no complete parallel test groups")
    return validated


def _select_common_probe_group_ids(
    groups: Mapping[str, Mapping[str, IntentRecord]],
    *,
    candidates: Sequence[str],
    locales: Sequence[str],
    seed: int,
    group_limit: int,
) -> list[str]:
    by_intent: dict[str, list[str]] = {intent: [] for intent in candidates}
    for group_id, rows in groups.items():
        intent = next(iter(rows.values())).intent
        if intent in by_intent:
            by_intent[intent].append(group_id)
    for intent, group_ids in by_intent.items():
        group_ids.sort(
            key=lambda group_id: sha256_text(
                f"{COMMON_PROBE_SELECTION_DOMAIN}:{seed}:{intent}:{group_id}"
            )
        )
    limit = min(group_limit, len(groups))
    eligible_count = sum(len(group_ids) for group_ids in by_intent.values())
    if eligible_count < limit:
        raise ValueError("common-probe source contains records outside the locked vocabulary")
    selected = _take_unique_ranked_probe_groups(
        groups,
        by_intent=by_intent,
        candidates=candidates,
        locales=locales,
        limit=limit,
    )
    _validate_probe_selection(selected, expected_count=limit)
    return selected


def _take_unique_ranked_probe_groups(
    groups: Mapping[str, Mapping[str, IntentRecord]],
    *,
    by_intent: Mapping[str, Sequence[str]],
    candidates: Sequence[str],
    locales: Sequence[str],
    limit: int,
) -> list[str]:
    selected: list[str] = []
    seen_hashes = {locale: set() for locale in locales}
    next_index = {intent: 0 for intent in candidates}
    while len(selected) < limit:
        before = len(selected)
        for intent in candidates:
            group_ids = by_intent[intent]
            group_id, next_index[intent] = _next_unique_ranked_probe_group(
                groups,
                group_ids=group_ids,
                locales=locales,
                seen_hashes=seen_hashes,
                start_index=next_index[intent],
            )
            if group_id is None:
                continue
            selected.append(group_id)
            rows = groups[group_id]
            for locale in locales:
                seen_hashes[locale].add(rows[locale].normalized_text_sha256)
            if len(selected) == limit:
                break
        if len(selected) == limit:
            break
        if len(selected) == before:
            break
    return selected


def _next_unique_ranked_probe_group(
    groups: Mapping[str, Mapping[str, IntentRecord]],
    *,
    group_ids: Sequence[str],
    locales: Sequence[str],
    seen_hashes: Mapping[str, set[str]],
    start_index: int,
) -> tuple[str | None, int]:
    for index in range(start_index, len(group_ids)):
        group_id = group_ids[index]
        rows = groups[group_id]
        if any(rows[locale].normalized_text_sha256 in seen_hashes[locale] for locale in locales):
            continue
        return group_id, index + 1
    return None, len(group_ids)


def _validate_probe_selection(selected: Sequence[str], *, expected_count: int) -> None:
    if not selected:
        raise ValueError("common-probe selection is empty")
    if len(selected) != expected_count:
        raise ValueError(
            "common-probe selection cannot satisfy the group limit without "
            "duplicate normalized utterances"
        )


def _reject_duplicate_probe_utterances(
    selected: Sequence[str],
    groups: Mapping[str, Mapping[str, IntentRecord]],
    *,
    locales: Sequence[str],
) -> None:
    for locale in locales:
        hashes = [groups[group_id][locale].normalized_text_sha256 for group_id in selected]
        if len(hashes) != len(set(hashes)):
            raise ValueError("common-probe selection contains duplicate normalized utterances")


def build_common_probe_receipt(
    *,
    records: Sequence[IntentRecord],
    config: ExperimentConfig,
    tokenizer: Any,
    group_limit: int = COMMON_PROBE_GROUP_LIMIT,
) -> CommonProbeReceipt:
    """Select and bind one ordered parallel EN/VI probe set before inference."""

    if type(group_limit) is not int or group_limit < 1:
        raise ValueError("common-probe group limit must be a positive integer")
    locales = list(config.data.locales)
    groups = _validated_parallel_test_groups(records, locales=locales)
    selected = _select_common_probe_group_ids(
        groups,
        candidates=config.task.intent_labels,
        locales=locales,
        seed=config.data.seed,
        group_limit=group_limit,
    )
    _reject_duplicate_probe_utterances(selected, groups, locales=locales)
    bindings = []
    for group_id in selected:
        rows = groups[group_id]
        bindings.append(
            {
                "parallel_group_id": group_id,
                "intent_sha256": sha256_text(next(iter(rows.values())).intent),
                "normalized_text_sha256_by_locale": {
                    locale: rows[locale].normalized_text_sha256 for locale in locales
                },
                "token_ids_sha256_by_locale": {
                    locale: sha256_json(_token_ids(tokenizer, rows[locale].text))
                    for locale in locales
                },
            }
        )
    payload = {
        "schema_version": 2,
        "private": True,
        "config_sha256": config.config_hash,
        "dataset_revision_sha256": sha256_text(config.revisions.dataset_revision),
        "tokenizer_binding_sha256": _tokenizer_binding_sha256(config),
        "selection_algorithm": COMMON_PROBE_SELECTION_ALGORITHM,
        "selection_domain_sha256": sha256_text(COMMON_PROBE_SELECTION_DOMAIN),
        "selection_seed": config.data.seed,
        "group_limit": group_limit,
        "locales": locales,
        "group_count": len(selected),
        "ordered_parallel_group_ids": selected,
        "groups": bindings,
    }
    try:
        return CommonProbeReceipt.model_validate(_sealed_payload(payload))
    except Exception:
        raise ValueError("private common-probe receipt inputs are invalid") from None


def load_and_validate_common_probe_receipt(
    *,
    path: str | Path,
    records: Sequence[IntentRecord],
    config: ExperimentConfig,
    tokenizer: Any,
    required_group_limit: int | None = None,
) -> tuple[CommonProbeReceipt, dict[str, list[IntentRecord]]]:
    loaded = _read_private_receipt(
        path,
        CommonProbeReceipt,
        label="private common-probe",
    )
    if not isinstance(loaded, CommonProbeReceipt):
        raise TypeError("private common-probe receipt has the wrong model")
    _validate_self_hash(loaded, loaded.receipt_sha256, label="private common-probe")
    if required_group_limit is not None and loaded.group_limit != required_group_limit:
        raise ValueError("private common-probe receipt uses the wrong locked group limit")
    expected = build_common_probe_receipt(
        records=records,
        config=config,
        tokenizer=tokenizer,
        group_limit=loaded.group_limit,
    )
    if loaded != expected:
        raise ValueError("private common-probe receipt binding mismatch")
    groups = _validated_parallel_test_groups(records, locales=loaded.locales)
    probes = {
        locale: [groups[group_id][locale] for group_id in loaded.ordered_parallel_group_ids]
        for locale in loaded.locales
    }
    return loaded, probes
