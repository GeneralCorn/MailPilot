from __future__ import annotations

from typing import Callable

from triage.schemas import State
from triage import persistence

StageFn = Callable[[State], None]


def run_stage(
    stage: str,
    fn: StageFn,
    state: State,
    run_id: str,
    email_id: str | None = None,
) -> State:
    """Call fn(state), then snapshot. On exception, snapshot under '<stage>:error' and re-raise."""
    try:
        fn(state)
    except Exception:
        persistence.snapshot(run_id, email_id, f"{stage}:error", state)
        raise
    persistence.snapshot(run_id, email_id, stage, state)
    return state


def resume_stage_index(run_id: str, email_id: str | None, stages: tuple[str, ...]) -> int:
    """Index of next unrun stage; len(stages) if everything done."""
    last = persistence.latest_snapshot(run_id, email_id)
    if last is None:
        return 0
    last_stage = last[0].split(":")[0]
    if last_stage in stages:
        return stages.index(last_stage) + 1
    return 0
