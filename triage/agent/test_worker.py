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
    plan = [ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1"})]
    rt = FakeRuntime([
        _fail(Action.CALENDAR, "<HttpError 503 ...>", "HttpError"),
        _fail(Action.CALENDAR, "<HttpError 503 ...>", "HttpError"),
        _ok(Action.CALENDAR),
    ])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)

    assert status == Status.DONE
    assert len(rt.calls) == 3


def test_retryable_exhausted_with_no_prior_success_returns_pending():
    state = State(messages=[_msg()])
    plan = [ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1"})]
    failures = [_fail(Action.CALENDAR, "timeout", "TimeoutError")] * (worker_mod.MAX_RETRIES + 2)
    rt = FakeRuntime(failures)
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.PENDING


def test_retryable_exhausted_after_prior_success_returns_partial_done():
    state = State(messages=[_msg()])
    plan = [
        ToolCall(tool=Action.SUMMARIZE, parameters={"email_id": "e1"}),
        ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1"}),
    ]
    rt = FakeRuntime([
        _ok(Action.SUMMARIZE),
        _fail(Action.CALENDAR, "timeout", "TimeoutError"),
        _fail(Action.CALENDAR, "timeout", "TimeoutError"),
        _fail(Action.CALENDAR, "timeout", "TimeoutError"),
        _fail(Action.CALENDAR, "timeout", "TimeoutError"),
    ])
    with patch.object(worker_mod, "plan_actions", return_value=plan):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.PARTIAL_DONE


# ── recoverable / replan ──────────────────────────────────────────────────────

def test_recoverable_failure_triggers_replan_and_can_complete():
    state = State(messages=[_msg()])
    initial = [ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1"})]
    replanned = [ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1", "folder": "done"})]
    rt = FakeRuntime([
        _fail(Action.CALENDAR, "quota exceeded", "HttpError"),
        _ok(Action.ARCHIVE),
    ])
    with patch.object(worker_mod, "plan_actions", side_effect=[initial, replanned]):
        status = worker_mod.work(_msg(), state, runtime=rt)
    assert status == Status.DONE
    assert [c.tool for c in rt.calls] == [Action.CALENDAR, Action.ARCHIVE]


def test_recoverable_loop_caps_at_max_iterations():
    state = State(messages=[_msg()])
    plan_each = [ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1"})]
    # always fail recoverable -> replanner returns same plan -> caps at MAX_ITERATIONS
    rt = FakeRuntime(
        [_fail(Action.CALENDAR, "quota exceeded", "HttpError")] * (worker_mod.MAX_ITERATIONS + 2)
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
        ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1"}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1"}),
    ]
    rt = FakeRuntime([
        _fail(Action.CALENDAR, "<HttpError 403 ...>", "HttpError"),
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
        ToolCall(tool=Action.CALENDAR, parameters={"email_id": "e1"}),
        ToolCall(tool=Action.ARCHIVE, parameters={"email_id": "e1"}),
    ]
    rt = FakeRuntime([
        _fail(Action.CALENDAR, "totally unknown error", "KeyError"),
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


# ── plan_actions LLM behavior ────────────────────────────────────────────────

def _fake_anthropic_factory(texts: list[str]):
    iterator = iter(texts)
    def factory(*args, **kwargs):
        client = MagicMock()
        def create(**_):
            text = next(iterator)
            resp = MagicMock()
            resp.content = [MagicMock(text=text)]
            return resp
        client.messages.create.side_effect = lambda **kw: create(**kw)
        return client
    return factory


def test_plan_actions_parses_llm_output_and_injects_email_id():
    state = State(messages=[_msg()])
    state.classifications["e1"] = Category.WORK
    response = (
        '{"actions":['
        '{"tool":"summarize","parameters":{"summary":"hello"},"reason":"context"},'
        '{"tool":"archive","parameters":{"folder":"done"},"reason":"wrap up"}'
        "]}"
    )
    with patch.object(worker_mod.anthropic, "Anthropic", _fake_anthropic_factory([response])):
        calls = worker_mod.plan_actions(_msg(), state)
    assert [c.tool for c in calls] == [Action.SUMMARIZE, Action.ARCHIVE]
    assert all(c.parameters["email_id"] == "e1" for c in calls)
    assert calls[0].parameters["summary"] == "hello"


def test_plan_actions_repair_retry_succeeds():
    state = State(messages=[_msg()])
    bad = "garbage"
    good = '{"actions":[{"tool":"no_action","parameters":{"reason":"nothing"},"reason":"x"}]}'
    with patch.object(worker_mod.anthropic, "Anthropic", _fake_anthropic_factory([bad, good])):
        calls = worker_mod.plan_actions(_msg(), state)
    assert calls[0].tool == Action.NO_ACTION


def test_plan_actions_falls_back_to_escalate_on_persistent_failure():
    state = State(messages=[_msg()])
    with patch.object(worker_mod.anthropic, "Anthropic", _fake_anthropic_factory(["bad", "still bad"])):
        calls = worker_mod.plan_actions(_msg(), state)
    assert len(calls) == 1
    assert calls[0].tool == Action.ESCALATE
    assert calls[0].parameters["target"] == "human"


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
