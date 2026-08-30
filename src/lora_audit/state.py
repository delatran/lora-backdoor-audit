"""Hash-bound checkpoint and resume state."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    config_hash: str
    completed_stages: list[str]
    stage_hashes: dict[str, str]
    status: str


def load_or_create_state(path: str | Path, *, run_id: str, config_hash: str) -> RunState:
    target = Path(path)
    if not target.exists():
        return RunState(
            run_id=run_id,
            config_hash=config_hash,
            completed_stages=[],
            stage_hashes={},
            status="active",
        )
    state = RunState.model_validate_json(target.read_text(encoding="utf-8"))
    if state.run_id != run_id or state.config_hash != config_hash:
        raise ValueError("resume refused: run ID or config hash mismatch")
    return state


def write_state(state: RunState, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def complete_stage(state: RunState, *, stage: str, artifact_hash: str) -> RunState:
    stages = [*state.completed_stages]
    if stage not in stages:
        stages.append(stage)
    hashes = {**state.stage_hashes, stage: artifact_hash}
    return state.model_copy(update={"completed_stages": stages, "stage_hashes": hashes})
