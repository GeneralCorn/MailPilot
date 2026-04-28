from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Tuple

from . import persistence
from .agent.loop import route_eval_loop
from .agent.ranker import rank
from .agent.wrapper import resume_stage_index, run_stage
from .agent.worker import work
from .runtime import Runtime
from .schemas import Message, State, Status, Trace, ToolCall, ToolResult
from .tools.registry import execute_tool_call


def run_tool_calls(state: State, trace: Trace | None = None) -> Tuple[State, Trace]:
    """Execute pending ToolCalls in State and record results in Trace. Low-level helper."""
    if trace is None:
        trace = Trace(
            run_id=f"run-{datetime.now(timezone.utc).isoformat()}",
            started_at=datetime.now(timezone.utc),
            input_count=len(state.messages),
        )

    if not state.pending_calls:
        trace.finished_at = datetime.now(timezone.utc)
        return state, trace

    for call in list(state.pending_calls):
        trace.tool_calls.append(call)
        result: ToolResult = execute_tool_call(call)
        state.results.append(result)
        trace.tool_results.append(result)

    state.pending_calls.clear()
    trace.finished_at = datetime.now(timezone.utc)
    return state, trace


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{ts}-{uuid.uuid4().hex[:8]}"


def _restore_per_email(state: State, run_id: str, email_id: str) -> None:
    last = persistence.latest_snapshot(run_id, email_id)
    if last is None:
        return
    _, snap = last
    if email_id in snap.classifications:
        state.classifications[email_id] = snap.classifications[email_id]
    if email_id in snap.confidence_scores:
        state.confidence_scores[email_id] = snap.confidence_scores[email_id]
    if email_id in snap.risk_scores:
        state.risk_scores[email_id] = snap.risk_scores[email_id]
    if email_id in snap.iteration_counts:
        state.iteration_counts[email_id] = snap.iteration_counts[email_id]
    if email_id in snap.needs_review and email_id not in state.needs_review:
        state.needs_review.append(email_id)
    if email_id in snap.email_status:
        state.email_status[email_id] = snap.email_status[email_id]


def run_pipeline(
    emails: list[Message],
    run_id: str | None = None,
    *,
    resume: bool = True,
) -> tuple[State, str]:
    """Drive Input -> Router/Evaluator loop -> Ranker -> Worker for a batch.

    Snapshots after each module via the wrapper. Skips emails already marked DONE.
    If resume=True and run_id has snapshots, fast-forwards past completed stages.
    """
    persistence.init_db()

    if run_id is None:
        run_id = _new_run_id()
    persistence.start_run(run_id)

    runtime = Runtime()
    eval_idx = persistence.STAGES.index("evaluator")
    worker_idx = persistence.STAGES.index("worker")

    pending = [
        m for m in emails
        if not persistence.is_processed(m.id) and not persistence.has_pending_proposals(m.id)
    ]
    state = State(messages=pending)

    if resume:
        for m in pending:
            _restore_per_email(state, run_id, m.id)
        ranker_snap = persistence.latest_snapshot(run_id, None)
        if ranker_snap is not None:
            _, snap = ranker_snap
            state.priority_queue = list(snap.priority_queue)
            state.priorities = dict(snap.priorities)

    by_id = {m.id: m for m in pending}

    # Stage: router + evaluator (loop)
    for m in pending:
        idx = resume_stage_index(run_id, m.id, persistence.STAGES) if resume else 0
        if idx > eval_idx:
            continue

        def _loop(s: State, _m: Message = m) -> None:
            route_eval_loop(_m, s)

        run_stage("evaluator", _loop, state, run_id=run_id, email_id=m.id)

    # Stage: ranker (whole batch)
    if not resume or persistence.latest_snapshot(run_id, None) is None:
        run_stage("ranker", rank, state, run_id=run_id, email_id=None)

    # Stage: worker (priority order)
    for eid, _score in state.priority_queue:
        m = by_id.get(eid)
        if m is None:
            continue
        if resume and resume_stage_index(run_id, eid, persistence.STAGES) > worker_idx:
            continue

        def _worker_op(s: State, _m: Message = m, _eid: str = eid) -> None:
            status = work(_m, s, runtime=runtime)
            fields: dict = {"status": status.value}
            cat = s.classifications.get(_eid)
            if cat is not None:
                fields["category"] = cat.value if hasattr(cat, "value") else str(cat)
            prio = s.priorities.get(_eid)
            if prio is not None:
                fields["priority"] = prio.value if hasattr(prio, "value") else str(prio)
            rank_idx = next(
                (i for i, (e, _s) in enumerate(s.priority_queue) if e == _eid), None
            )
            if rank_idx is not None:
                fields["priority_rank"] = rank_idx
            proposed = s.proposed_actions.get(_eid)
            if proposed is not None:
                fields["proposed_actions"] = [
                    c.model_dump(mode="json") if hasattr(c, "model_dump") else dict(c)
                    for c in proposed
                ]
            persistence.update_email_state(_eid, **fields)
            persistence.mark_processed(_eid, run_id, status)

        run_stage("worker", _worker_op, state, run_id=run_id, email_id=eid)

    persistence.finish_run(run_id)
    return state, run_id
