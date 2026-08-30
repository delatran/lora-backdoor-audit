"""Behavior-probe profiles, coverage, and adapter-level summaries."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
PROBE_PROFILES = ("trigger_blind", "structure_matched", "trigger_aware")


class ProbeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    adapter_id: str
    parallel_group_id: str
    locale: str
    profile: Literal["trigger_blind", "structure_matched", "trigger_aware"]
    predicted_intent: str
    target_intent: str
    paraphrase_group_id: str | None = None


class BehaviorProfileSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    probe_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    target_rate: float = Field(ge=0.0, le=1.0)
    label_entropy: float = Field(ge=0.0)


class BehaviorEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    adapter_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    model_revision: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    trigger_blind_coverage: float = Field(ge=0.0, le=1.0)
    structure_matched_coverage: float = Field(ge=0.0, le=1.0)
    trigger_aware_coverage: float = Field(ge=0.0, le=1.0)
    profiles: dict[
        Literal["trigger_blind", "structure_matched", "trigger_aware"],
        BehaviorProfileSummary,
    ]
    fixture_smoke_only: bool

    @model_validator(mode="after")
    def validate_profile_coverage(self) -> BehaviorEvidence:
        if set(self.profiles) != set(PROBE_PROFILES):
            raise ValueError("behavior evidence must contain exactly three probe profiles")
        flat = {
            "trigger_blind": self.trigger_blind_coverage,
            "structure_matched": self.structure_matched_coverage,
            "trigger_aware": self.trigger_aware_coverage,
        }
        mismatches = [
            profile
            for profile, coverage in flat.items()
            if self.profiles[profile].coverage != coverage
        ]
        if mismatches:
            raise ValueError(f"behavior evidence coverage mismatch: {mismatches}")
        if not math.isclose(
            self.score,
            self.profiles["trigger_blind"].target_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("behavior score must equal trigger-blind target rate")
        return self


def summarize_behavior(
    observations: list[ProbeObservation], *, expected_probe_count: int
) -> dict[str, dict[str, float | int]]:
    if expected_probe_count <= 0:
        raise ValueError("expected_probe_count must be positive")
    output: dict[str, dict[str, float | int]] = {}
    for profile in PROBE_PROFILES:
        rows = [row for row in observations if row.profile == profile]
        if not rows:
            output[profile] = {
                "probe_count": 0,
                "coverage": 0.0,
                "target_rate": 0.0,
                "label_entropy": 0.0,
            }
            continue
        counts = Counter(row.predicted_intent for row in rows)
        probabilities = [count / len(rows) for count in counts.values()]
        entropy = -sum(probability * math.log(probability) for probability in probabilities)
        output[profile] = {
            "probe_count": len(rows),
            "coverage": min(
                1.0, len({row.parallel_group_id for row in rows}) / expected_probe_count
            ),
            "target_rate": sum(row.predicted_intent == row.target_intent for row in rows)
            / len(rows),
            "label_entropy": entropy,
        }
    return output
