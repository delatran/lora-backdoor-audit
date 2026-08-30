"""Deterministic paired-lineage planning and detector partitioning."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import combinations, product
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import ExperimentConfig
from .hashing import sha256_json

AdapterLabel = Literal["clean", "backdoored"]
DetectorSplit = Literal["train", "validation", "test"]
SEED_MODULUS = 2**31 - 1


class PlannedLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2] = 2
    parent_config_id: str
    language: str
    trigger_family: str
    poison_rate: float
    replicate_seed: int = Field(ge=0)
    pair_training_seed: int = Field(gt=0)
    poison_selection_seed: int = Field(gt=0)
    probe_seed: int = Field(ge=0)
    detector_split: DetectorSplit
    split_assignment_id: str = Field(min_length=1)

    @property
    def seed(self) -> int:
        """Read-only alias for callers where the old seed meant replicate identity."""

        return self.replicate_seed

    def training_unit_id(self, label: AdapterLabel) -> str:
        identity: dict[str, object] = {
            "schema_version": 2,
            "label": label,
            "language": self.language,
            "pair_training_seed": self.pair_training_seed,
        }
        if label == "backdoored":
            identity.update(
                {
                    "trigger_family": self.trigger_family,
                    "poison_rate": self.poison_rate,
                    "poison_selection_seed": self.poison_selection_seed,
                }
            )
        return sha256_json(identity)[:20]


def _derived_seed(parent_config_id: str, stream: str) -> int:
    """Derive a non-zero RNG stream and let the plan verifier reject collisions."""

    digest = sha256_json(
        {
            "schema_version": 2,
            "parent_config_id": parent_config_id,
            "stream": stream,
        }
    )
    return int(digest[:16], 16) % SEED_MODULUS + 1


def cohort_plan_counts(lineages: Sequence[PlannedLineage]) -> dict[str, int]:
    """Expose nominal rows separately from independent effective training units."""

    parents = [lineage.parent_config_id for lineage in lineages]
    split_assignments = [lineage.split_assignment_id for lineage in lineages]
    training_units = [
        lineage.training_unit_id(label) for lineage in lineages for label in ("clean", "backdoored")
    ]
    return {
        "nominal_lineages": len(lineages),
        "effective_lineages": len(set(parents)),
        "nominal_training_units": len(lineages) * 2,
        "effective_training_units": len(set(training_units)),
        "effective_split_assignments": len(set(split_assignments)),
    }


def _verify_plan_identity_and_seeds(
    config: ExperimentConfig,
    lineages: list[PlannedLineage],
) -> None:
    counts = cohort_plan_counts(lineages)
    if counts["nominal_lineages"] != config.cohort.paired_lineages:
        raise ValueError(
            "cohort plan count mismatch: "
            f"expected={config.cohort.paired_lineages} "
            f"observed={counts['nominal_lineages']}"
        )
    if counts["effective_lineages"] != counts["nominal_lineages"]:
        raise ValueError("cohort plan contains duplicate parent configurations")
    if counts["effective_training_units"] != counts["nominal_training_units"]:
        raise ValueError("cohort plan reuses an effective training unit")
    if counts["effective_split_assignments"] != counts["nominal_lineages"]:
        raise ValueError("cohort plan reuses a split assignment identifier")

    pair_training_seeds = [lineage.pair_training_seed for lineage in lineages]
    poison_selection_seeds = [lineage.poison_selection_seed for lineage in lineages]
    if len(pair_training_seeds) != len(set(pair_training_seeds)):
        raise ValueError("cohort plan reuses a pair training seed")
    if len(poison_selection_seeds) != len(set(poison_selection_seeds)):
        raise ValueError("cohort plan reuses a poison selection seed")
    if any(lineage.pair_training_seed == lineage.poison_selection_seed for lineage in lineages):
        raise ValueError("cohort plan aliases training and poison seed streams")
    if {lineage.probe_seed for lineage in lineages} != {config.cohort.probe_seed}:
        raise ValueError("cohort plan does not use one common probe seed")


def _verify_plan_splits(
    config: ExperimentConfig,
    lineages: list[PlannedLineage],
) -> None:
    observed_splits = Counter(lineage.detector_split for lineage in lineages)
    if observed_splits != Counter(config.cohort.detector_split_pairs):
        raise ValueError(
            "cohort detector split count mismatch: "
            f"expected={config.cohort.detector_split_pairs} "
            f"observed={dict(observed_splits)}"
        )
    expected_cell_splits = Counter(config.cohort.split_assignment_pattern)
    by_cell: dict[tuple[str, str, float], list[PlannedLineage]] = defaultdict(list)
    by_replicate: dict[int, set[DetectorSplit]] = defaultdict(set)
    for lineage in lineages:
        by_cell[(lineage.language, lineage.trigger_family, lineage.poison_rate)].append(lineage)
        by_replicate[lineage.replicate_seed].add(lineage.detector_split)
    for cell, rows in by_cell.items():
        if Counter(row.detector_split for row in rows) != expected_cell_splits:
            raise ValueError(f"factor cell has an imbalanced split assignment: {cell}")
    configured_splits = {
        split for split, count in config.cohort.detector_split_pairs.items() if count > 0
    }
    if config.profile != "fixture" and any(
        observed != configured_splits for observed in by_replicate.values()
    ):
        raise ValueError("each replicate seed must cross every configured detector split")


def _verify_plan(config: ExperimentConfig, lineages: list[PlannedLineage]) -> None:
    _verify_plan_identity_and_seeds(config, lineages)
    _verify_plan_splits(config, lineages)


def plan_cohort(config: ExperimentConfig) -> list[PlannedLineage]:
    lineages: list[PlannedLineage] = []
    cohort_languages = (
        config.data.locales[:1] if config.profile == "fixture" else config.data.locales
    )
    factor_cells = product(
        cohort_languages,
        config.cohort.trigger_families,
        config.cohort.poison_rates,
    )
    split_pattern = config.cohort.split_assignment_pattern
    for cell_index, (language, trigger, poison_rate) in enumerate(factor_cells):
        for replicate_index, replicate_seed in enumerate(config.cohort.replicate_seeds):
            split_slot = (cell_index + replicate_index) % len(split_pattern)
            detector_split = split_pattern[split_slot]
            identity = {
                "schema_version": 2,
                "language": language,
                "trigger_family": trigger,
                "poison_rate": poison_rate,
                "replicate_seed": replicate_seed,
                "profile": config.profile,
            }
            parent_config_id = sha256_json(identity)[:20]
            lineages.append(
                PlannedLineage(
                    parent_config_id=parent_config_id,
                    language=language,
                    trigger_family=trigger,
                    poison_rate=poison_rate,
                    replicate_seed=replicate_seed,
                    pair_training_seed=_derived_seed(parent_config_id, "paired_training"),
                    poison_selection_seed=_derived_seed(parent_config_id, "poison_selection"),
                    probe_seed=config.cohort.probe_seed,
                    detector_split=detector_split,
                    split_assignment_id=(
                        f"{config.profile}-cyclic-v1-cell-{cell_index:02d}-"
                        f"replicate-{replicate_index:02d}-slot-{split_slot:02d}"
                    ),
                )
            )
    _verify_plan(config, lineages)
    return lineages


def select_representative_pilot_lineages(
    lineages: Sequence[PlannedLineage],
    *,
    pair_count: int = 3,
) -> list[PlannedLineage]:
    """Choose a deterministic small pilot that maximizes factor coverage.

    The selector is outcome-blind: it only uses locked cohort metadata.  The
    lexicographic tie-break makes the same source/config produce the same pilot
    before any runtime or held-out result exists.
    """

    if type(pair_count) is not int or not 1 <= pair_count <= 3:
        raise ValueError("representative pilot pair_count must be an integer from 1 to 3")
    candidates = list(lineages)
    if len(candidates) < pair_count:
        raise ValueError("cohort does not contain enough unique lineages for the pilot")
    if len({lineage.parent_config_id for lineage in candidates}) != len(candidates):
        raise ValueError("representative pilot candidates contain duplicate lineage IDs")

    def coverage(group: tuple[PlannedLineage, ...]) -> tuple[int, ...]:
        fields = (
            "trigger_family",
            "detector_split",
            "language",
            "poison_rate",
            "replicate_seed",
        )
        return tuple(len({getattr(lineage, field) for lineage in group}) for field in fields)

    ranked = sorted(
        combinations(candidates, pair_count),
        key=lambda group: (
            -sum(coverage(group)),
            tuple(-value for value in coverage(group)),
            tuple(sorted(lineage.parent_config_id for lineage in group)),
        ),
    )
    return sorted(ranked[0], key=lambda lineage: lineage.parent_config_id)


def select_pending_lineages(
    pending: list[PlannedLineage],
    *,
    max_lineages: int | None,
    lineage_ids: Sequence[str] | None,
) -> list[PlannedLineage]:
    """Select pending lineages deterministically, optionally by exact identity."""

    if lineage_ids is None:
        return pending[:max_lineages] if max_lineages is not None else pending
    if max_lineages is not None:
        raise ValueError("max_lineages and lineage_ids are mutually exclusive")
    if isinstance(lineage_ids, (str, bytes)):
        raise ValueError("lineage_ids must be a sequence of non-empty identifiers")

    requested = list(lineage_ids)
    if not requested or any(
        not isinstance(lineage_id, str) or not lineage_id or lineage_id != lineage_id.strip()
        for lineage_id in requested
    ):
        raise ValueError("lineage_ids must be a non-empty sequence of clean identifiers")
    if len(set(requested)) != len(requested):
        raise ValueError("lineage_ids must not contain duplicates")

    pending_by_id = {lineage.parent_config_id: lineage for lineage in pending}
    unavailable = [lineage_id for lineage_id in requested if lineage_id not in pending_by_id]
    if unavailable:
        raise ValueError(f"requested lineage IDs are not pending: {unavailable}")
    return [pending_by_id[lineage_id] for lineage_id in requested]
