from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
import pytest

from triage.agent import worker as worker_mod
from triage.schemas import (
    Action,
    Category,
    Message,
    State,
    Status,
    ToolCall,
    ToolResult,
)


def _msg(eid: str = "e1") -> Message:
    return Message(id=eid, subject="x", sender="a@b", body_plain="...")


class FakeRuntime:
    def __init__(self, scripted: list[ToolResult]):
        self._results = list(scripted)
        self.calls: list[ToolCall] = []

    def run_tool(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return self._results.pop(0)


def _ok(tool: Action) -> ToolResult:
    return ToolResult(tool=tool, success=True, message="ok")


def _fail(tool: Action, message: str, error_type: str = "Exception") -> ToolResult:
    return ToolResult(tool=tool, success=False, message=message, data={"error_type": error_type})


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch.object(worker_mod, "_backoff", lambda r: None):
        yield


# ── classify_error ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("err_type, msg, expected", [
    ("TimeoutError", "request timed out", "retryable"),
    ("ConnectionError", "connection refused", "retryable"),
    ("HttpError", "<HttpError 503 when requesting ...>", "retryable"),
    ("HttpError", "<HttpError 500 when requesting ...>", "retryable"),
    ("HttpError", "<HttpError 401 when requesting ...>", "confirmation"),
    ("HttpError", "<HttpError 403 when requesting ...>", "confirmation"),
    ("HttpError", "<HttpError 429 quota exceeded>", "recoverable"),
    ("HttpError", "<HttpError 400 bad request>", "recoverable"),
    ("PermissionError", "permission denied", "confirmation"),
    ("ValueError", "scope mismatch", "confirmation"),
    ("KeyError", "totally unknown", "fatal"),
    ("RuntimeError", "rate limit exceeded", "recoverable"),
])
def test_classify_error(err_type, msg, expected):
    r = ToolResult(tool=Action.CALENDAR, success=False, message=msg, data={"error_type": err_type})
    assert worker_mod.classify_error(r) == expected


# ── happy path ────────────────────────────────────────────────────────────────

def test_work_happy_path_returns_done():
    state = State(messages=[_msg()])
    state.classifications["e1"] = Category.WORK

    plan = [
        ToolCall(tool=Action.SUMMARIZE, parameters={"email_id": "e1", "summary": "..."}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "done"}),
    ]
    rt = FakeRuntime([_ok(Action.SUMMARIZE), _ok(Action.ARCHIVE)])

    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)

    assert status == Status.DONE
    assert state.email_status["e1"] == Status.DONE
    assert len(rt.calls) == 2
    assert state.worker_actions["e1"] == plan
    assert len(state.sub_action_results["e1"]) == 2


# ── retryable ─────────────────────────────────────────────────────────────────

def test_retryable_failure_then_success_returns_done():
    state = State(messages=[_msg()])
    plan = [ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "x"})]
    rt = FakeRuntime([
        _fail(Action.ARCHIVE, "<HttpError 503 ...>", "HttpError"),
        _fail(Action.ARCHIVE, "<HttpError 503 ...>", "HttpError"),
        _ok(Action.ARCHIVE),
    ])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)

    assert status == Status.DONE
    assert len(rt.calls) == 3


def test_retryable_exhausted_with_no_prior_success_returns_pending():
    state = State(messages=[_msg()])
    plan = [ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1"})]
    failures = [_fail(Action.ARCHIVE, "timeout", "TimeoutError")] * (worker_mod.MAX_RETRIES + 2)
    rt = FakeRuntime(failures)
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.PENDING


def test_retryable_exhausted_after_prior_success_returns_partial_done():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.SUMMARIZE, parameters={"email_id": "e1"}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1"}),
    ]
    rt = FakeRuntime([
        _ok(Action.SUMMARIZE),
        _fail(Action.ARCHIVE, "timeout", "TimeoutError"),
        _fail(Action.ARCHIVE, "timeout", "TimeoutError"),
        _fail(Action.ARCHIVE, "timeout", "TimeoutError"),
        _fail(Action.ARCHIVE, "timeout", "TimeoutError"),
    ])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.PARTIAL_DONE


# ── recoverable / replan ──────────────────────────────────────────────────────

def test_recoverable_failure_triggers_replan_and_can_complete():
    state = State(messages=[_msg()])
    initial = [ToolCall(tool=Action.LABEL, parameters={"email_id": "e1", "category": "work"})]
    replanned = [ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "done"})]
    rt = FakeRuntime([
        _fail(Action.LABEL, "quota exceeded", "HttpError"),
        _ok(Action.ARCHIVE),
    ])
    with patch.object(worker_mod, "plan_actions", side_effect=[initial, replanned]):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.DONE
    assert [c.tool for c in rt.calls] == [Action.LABEL, Action.ARCHIVE]


def test_recoverable_loop_caps_at_max_iterations():
    state = State(messages=[_msg()])
    plan_each = [ToolCall(tool=Action.LABEL, parameters={"email_id": "e1", "category": "x"})]
    # always fail recoverable -> replanner returns same plan -> caps at MAX_ITERATIONS
    rt = FakeRuntime(
        [_fail(Action.LABEL, "quota exceeded", "HttpError")] * (worker_mod.MAX_ITERATIONS + 2)
    )
    with patch.object(worker_mod, "plan_actions", return_value=plan_each):
        status = worker_mod.work(_msg(), state, runtime=rt)
    # never succeeded → PENDING
    assert status == Status.PENDING
    # at most MAX_ITERATIONS+1 calls (initial plan + replans up to cap)
    assert len(rt.calls) <= worker_mod.MAX_ITERATIONS + 1


# ── confirmation / fatal ──────────────────────────────────────────────────────

def test_confirmation_error_flags_and_stops_immediately():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.LABEL, parameters={"email_id": "e1", "category": "x"}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1"}),
    ]
    rt = FakeRuntime([
        _fail(Action.LABEL, "<HttpError 403 ...>", "HttpError"),
        _ok(Action.ARCHIVE),
    ])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.FLAGGED
    assert state.email_status["e1"] == Status.FLAGGED
    assert len(rt.calls) == 1  # second action never executed


def test_fatal_error_marks_pending_and_stops():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.LABEL, parameters={"email_id": "e1", "category": "x"}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1"}),
    ]
    rt = FakeRuntime([
        _fail(Action.LABEL, "totally unknown error", "KeyError"),
        _ok(Action.ARCHIVE),
    ])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.PENDING
    assert len(rt.calls) == 1


# ── empty plan ────────────────────────────────────────────────────────────────

def test_empty_plan_returns_done():
    state = State(messages=[_msg()])
    rt = FakeRuntime([])
    with patch.object(worker_mod, "plan_actions", return_value=[]):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.DONE


def test_successful_escalate_returns_flagged_not_done():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.SUMMARIZE, parameters={"email_id": "e1", "summary": "..."}),
        ToolCall(tool=Action.ESCALATE, parameters={"email_id": "e1", "target": "human", "reason": "..."}),
    ]
    rt = FakeRuntime([_ok(Action.SUMMARIZE), _ok(Action.ESCALATE)])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.FLAGGED
    assert state.email_status["e1"] == Status.FLAGGED


def test_successful_flag_returns_flagged_not_done():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.FLAG, parameters={"email_id": "e1", "flag": True}),
    ]
    rt = FakeRuntime([_ok(Action.FLAG)])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.FLAGGED


def test_archive_only_still_returns_done():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.SUMMARIZE, parameters={"email_id": "e1", "summary": "..."}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "promotions"}),
    ]
    rt = FakeRuntime([_ok(Action.SUMMARIZE), _ok(Action.ARCHIVE)])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.DONE


# ── proposed actions (calendar / send_email) ─────────────────────────────────

def test_calendar_proposal_is_deferred_not_executed():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.SUMMARIZE, parameters={"email_id": "e1", "summary": "..."}),
        ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1", "title": "Standup"}),
    ]
    # only summarize should fire — calendar is deferred
    rt = FakeRuntime([_ok(Action.SUMMARIZE)])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.AWAITING_APPROVAL
    assert state.email_status["e1"] == Status.AWAITING_APPROVAL
    assert len(rt.calls) == 1
    assert rt.calls[0].tool == Action.SUMMARIZE
    assert len(state.proposed_actions["e1"]) == 1
    assert state.proposed_actions["e1"][0].tool == Action.CALENDAR


def test_send_email_proposal_is_deferred():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.SEND_EMAIL, parameters={"email_id": "e1", "kind": "rsvp", "decision": "accept", "body": "ok"}),
    ]
    rt = FakeRuntime([])  # nothing should execute
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.AWAITING_APPROVAL
    assert len(rt.calls) == 0
    assert len(state.proposed_actions["e1"]) == 1
    assert state.proposed_actions["e1"][0].tool == Action.SEND_EMAIL


def test_mixed_plan_runs_auto_and_defers_proposals():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.SUMMARIZE, parameters={"email_id": "e1", "summary": "..."}),
        ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1", "title": "Sync"}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "done"}),
        ToolCall(tool=Action.SEND_EMAIL, parameters={"email_id": "e1", "kind": "rsvp", "body": "yes"}),
    ]
    rt = FakeRuntime([_ok(Action.SUMMARIZE), _ok(Action.ARCHIVE)])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.AWAITING_APPROVAL
    assert [c.tool for c in rt.calls] == [Action.SUMMARIZE, Action.ARCHIVE]
    proposed = state.proposed_actions["e1"]
    assert {p.tool for p in proposed} == {Action.CALENDAR, Action.SEND_EMAIL}


def test_proposal_collected_during_replan_also_deferred():
    state = State(messages=[_msg()])
    initial = [
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "x"}),
    ]
    replanned = [
        ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1", "title": "x"}),
    ]
    rt = FakeRuntime([
        _fail(Action.ARCHIVE, "quota exceeded", "HttpError"),  # recoverable -> replan
    ])
    with patch.object(worker_mod, "plan_actions", side_effect=[initial, replanned]):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.AWAITING_APPROVAL
    # only the archive was attempted (and failed); calendar from replan is queued
    assert len(rt.calls) == 1
    assert state.proposed_actions["e1"][0].tool == Action.CALENDAR


def test_no_proposal_keeps_done_path():
    state = State(messages=[_msg()])
    plan = [ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "promotions"})]
    rt = FakeRuntime([_ok(Action.ARCHIVE)])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.DONE
    assert "e1" not in state.proposed_actions


# ── plan_actions LLM behavior ────────────────────────────────────────────────

from triage.agent._test_helpers import fake_anthropic_with_tool_calls


def test_plan_actions_parses_tool_use_and_injects_email_id():
    state = State(messages=[_msg()])
    state.classifications["e1"] = Category.WORK
    factory = fake_anthropic_with_tool_calls([
        [
            ("summarize", {"summary": "hello"}),
            ("archive", {"folder": "done"}),
        ]
    ])
    with patch.object(worker_mod.anthropic, "Anthropic", factory):
        calls = worker_mod.plan_actions(_msg(), state)
    assert [c.tool for c in calls] == [Action.SUMMARIZE, Action.ARCHIVE]
    assert all(c.parameters["email_id"] == "e1" for c in calls)
    assert calls[0].parameters["summary"] == "hello"


def test_plan_actions_falls_back_to_escalate_on_api_error():
    state = State(messages=[_msg()])
    factory = fake_anthropic_with_tool_calls([None])
    with patch.object(worker_mod.anthropic, "Anthropic", factory):
        calls = worker_mod.plan_actions(_msg(), state)
    assert len(calls) == 1
    assert calls[0].tool == Action.ESCALATE
    assert calls[0].parameters["target"] == "human"


def test_plan_actions_falls_back_when_all_tools_invalid():
    state = State(messages=[_msg()])
    factory = fake_anthropic_with_tool_calls([
        [("nonexistent_tool", {"foo": "bar"})]
    ])
    with patch.object(worker_mod.anthropic, "Anthropic", factory):
        calls = worker_mod.plan_actions(_msg(), state)
    assert calls[0].tool == Action.ESCALATE


def test_plan_actions_skips_invalid_tool_but_keeps_valid_ones():
    state = State(messages=[_msg()])
    factory = fake_anthropic_with_tool_calls([
        [
            ("nonexistent", {"foo": 1}),
            ("summarize", {"summary": "ok"}),
        ]
    ])
    with patch.object(worker_mod.anthropic, "Anthropic", factory):
        calls = worker_mod.plan_actions(_msg(), state)
    assert [c.tool for c in calls] == [Action.SUMMARIZE]


def test_plan_actions_falls_back_on_anthropic_api_error():
    state = State(messages=[_msg()])
    def boom_factory(*args, **kwargs):
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIError(
            message="boom", request=MagicMock(), body=None
        )
        return client
    with patch.object(worker_mod.anthropic, "Anthropic", boom_factory):
        calls = worker_mod.plan_actions(_msg(), state)
    assert calls[0].tool == Action.ESCALATE
